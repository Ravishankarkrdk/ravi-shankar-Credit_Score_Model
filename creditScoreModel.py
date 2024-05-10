{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 15,
   "id": "7571811f",
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np \n",
    "import pandas as pd\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "from sklearn import svm"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "id": "c7b8102d",
   "metadata": {},
   "outputs": [],
   "source": [
    "df = pd.read_csv(\"loan.csv\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "id": "2032fb30",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Loan_ID</th>\n",
       "      <th>Gender</th>\n",
       "      <th>Married</th>\n",
       "      <th>Dependents</th>\n",
       "      <th>Education</th>\n",
       "      <th>Self_Employed</th>\n",
       "      <th>ApplicantIncome</th>\n",
       "      <th>CoapplicantIncome</th>\n",
       "      <th>LoanAmount</th>\n",
       "      <th>Loan_Amount_Term</th>\n",
       "      <th>Credit_History</th>\n",
       "      <th>Property_Area</th>\n",
       "      <th>Loan_Status</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>LP001002</td>\n",
       "      <td>Male</td>\n",
       "      <td>No</td>\n",
       "      <td>0</td>\n",
       "      <td>Graduate</td>\n",
       "      <td>No</td>\n",
       "      <td>5849</td>\n",
       "      <td>0.0</td>\n",
       "      <td>NaN</td>\n",
       "      <td>360.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>Urban</td>\n",
       "      <td>Y</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>LP001003</td>\n",
       "      <td>Male</td>\n",
       "      <td>Yes</td>\n",
       "      <td>1</td>\n",
       "      <td>Graduate</td>\n",
       "      <td>No</td>\n",
       "      <td>4583</td>\n",
       "      <td>1508.0</td>\n",
       "      <td>128.0</td>\n",
       "      <td>360.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>Rural</td>\n",
       "      <td>N</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>LP001005</td>\n",
       "      <td>Male</td>\n",
       "      <td>Yes</td>\n",
       "      <td>0</td>\n",
       "      <td>Graduate</td>\n",
       "      <td>Yes</td>\n",
       "      <td>3000</td>\n",
       "      <td>0.0</td>\n",
       "      <td>66.0</td>\n",
       "      <td>360.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>Urban</td>\n",
       "      <td>Y</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>LP001006</td>\n",
       "      <td>Male</td>\n",
       "      <td>Yes</td>\n",
       "      <td>0</td>\n",
       "      <td>Not Graduate</td>\n",
       "      <td>No</td>\n",
       "      <td>2583</td>\n",
       "      <td>2358.0</td>\n",
       "      <td>120.0</td>\n",
       "      <td>360.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>Urban</td>\n",
       "      <td>Y</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>LP001008</td>\n",
       "      <td>Male</td>\n",
       "      <td>No</td>\n",
       "      <td>0</td>\n",
       "      <td>Graduate</td>\n",
       "      <td>No</td>\n",
       "      <td>6000</td>\n",
       "      <td>0.0</td>\n",
       "      <td>141.0</td>\n",
       "      <td>360.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>Urban</td>\n",
       "      <td>Y</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "    Loan_ID Gender Married Dependents     Education Self_Employed  \\\n",
       "0  LP001002   Male      No          0      Graduate            No   \n",
       "1  LP001003   Male     Yes          1      Graduate            No   \n",
       "2  LP001005   Male     Yes          0      Graduate           Yes   \n",
       "3  LP001006   Male     Yes          0  Not Graduate            No   \n",
       "4  LP001008   Male      No          0      Graduate            No   \n",
       "\n",
       "   ApplicantIncome  CoapplicantIncome  LoanAmount  Loan_Amount_Term  \\\n",
       "0             5849                0.0         NaN             360.0   \n",
       "1             4583             1508.0       128.0             360.0   \n",
       "2             3000                0.0        66.0             360.0   \n",
       "3             2583             2358.0       120.0             360.0   \n",
       "4             6000                0.0       141.0             360.0   \n",
       "\n",
       "   Credit_History Property_Area Loan_Status  \n",
       "0             1.0         Urban           Y  \n",
       "1             1.0         Rural           N  \n",
       "2             1.0         Urban           Y  \n",
       "3             1.0         Urban           Y  \n",
       "4             1.0         Urban           Y  "
      ]
     },
     "execution_count": 17,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "id": "6fc5e633",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'pandas.core.frame.DataFrame'>\n",
      "RangeIndex: 614 entries, 0 to 613\n",
      "Data columns (total 13 columns):\n",
      " #   Column             Non-Null Count  Dtype  \n",
      "---  ------             --------------  -----  \n",
      " 0   Loan_ID            614 non-null    object \n",
      " 1   Gender             601 non-null    object \n",
      " 2   Married            611 non-null    object \n",
      " 3   Dependents         599 non-null    object \n",
      " 4   Education          614 non-null    object \n",
      " 5   Self_Employed      582 non-null    object \n",
      " 6   ApplicantIncome    614 non-null    int64  \n",
      " 7   CoapplicantIncome  614 non-null    float64\n",
      " 8   LoanAmount         592 non-null    float64\n",
      " 9   Loan_Amount_Term   600 non-null    float64\n",
      " 10  Credit_History     564 non-null    float64\n",
      " 11  Property_Area      614 non-null    object \n",
      " 12  Loan_Status        614 non-null    object \n",
      "dtypes: float64(4), int64(1), object(8)\n",
      "memory usage: 62.5+ KB\n"
     ]
    }
   ],
   "source": [
    "df.info()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "id": "ac9ad36a",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Loan_ID               0\n",
       "Gender               13\n",
       "Married               3\n",
       "Dependents           15\n",
       "Education             0\n",
       "Self_Employed        32\n",
       "ApplicantIncome       0\n",
       "CoapplicantIncome     0\n",
       "LoanAmount           22\n",
       "Loan_Amount_Term     14\n",
       "Credit_History       50\n",
       "Property_Area         0\n",
       "Loan_Status           0\n",
       "dtype: int64"
      ]
     },
     "execution_count": 19,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.isnull().sum()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "id": "f5fa406c",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<Axes: >"
      ]
     },
     "execution_count": 20,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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4mK99957yS5t1PvjH/+orVu3qqCgQPv379fdd9+te++9Vz//+c+TXdqIlfKfJLt27Vo9/fTTOnTokK6++upkl4O/6u3t1cmTJ/WXv/xFv/zlL/Wzn/1MTU1NhJQkO3XqlIqKinTgwAHNmjVLklRSUqLrrrtOdXV1yS0Og5w7d06f+cxntH79elVWVia7nFFtzJgxKioqUnNzc2zs3nvvVUtLi1566aUkVjZypfQelHXr1mnv3r06ePAg4cSYMWPG6LOf/ayKiopUU1OjWbNm6Uc/+lGyyxr1Wltb1d7ertmzZysjI0MZGRlqamrSf/7nfyojI0N9fX3JLhEfMn78eM2cOVNvvvlmsksZ9aZOnTroD6zPf/7zOnnyZJIqGvlS8iRZx3G0bt067d69W88//7zy8/OTXRIuwnEcRSKRZJcx6s2fP3/Qu0LuvPNOXXPNNXrggQeUnp6epMowlEgkoj/84Q/60pe+lOxSRr25c+cO+jiLN954Y1hfcIvhScmAsnbtWjU2NurXv/61srKyFA6HJUkTJ07UuHHjklwd/vVf/1WLFi1SXl6euru7tWvXLj3//POD3n2FT15WVtagc7XGjx+vnJwczuEy4P7779fSpUs1bdo0tbe36wc/+IG6urq0atWqZJc26n33u99VcXGxqqurtXz5cr3yyivatm2btm3bluzSRqyUDChbt26V9MGx8w/bvn27vvWtb33yBSHO//7v/2rlypU6c+aMJk6cqGuvvVbPPvusysrKkl0aYNo777yjb3zjG/rTn/6kv/3bv9U//MM/6OWXX+avdANuuOEG7d69W1VVVXrooYeUn5+vuro63X777ckubcRK+ZNkAQDAyJPSJ8kCAICRiYACAADMIaAAAABzCCgAAMAcAgoAADCHgAIAAMwhoAAAAHMIKAAAwBwCCgAAMIeAAgAAzCGgAAAAcwgoAADAnP8HuAhht/RYxrgAAAAASUVORK5CYII=",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "df['loanAmount_log']= np.log(df['LoanAmount'])\n",
    "df['loanAmount_log'].hist(bins=20)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "id": "cbaf8383",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Loan_ID               0\n",
       "Gender               13\n",
       "Married               3\n",
       "Dependents           15\n",
       "Education             0\n",
       "Self_Employed        32\n",
       "ApplicantIncome       0\n",
       "CoapplicantIncome     0\n",
       "LoanAmount           22\n",
       "Loan_Amount_Term     14\n",
       "Credit_History       50\n",
       "Property_Area         0\n",
       "Loan_Status           0\n",
       "loanAmount_log       22\n",
       "dtype: int64"
      ]
     },
     "execution_count": 21,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.isnull().sum()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "id": "2150374c",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<Axes: >"
      ]
     },
     "execution_count": 22,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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D3tebW2tzjnnHCUnJ2vs2LFatmzZAFXcN/3pz2172NraquLiYo0ePVopKSmaPHmy6uvrD3uOW/ZP6nt/g3n/XnvtNV1++eUKBALyeDx69tlno+43xigYDCoQCCglJUX5+fnasmXLEddds2aNTj/9dPl8Pp1++ulat25djDo4vFj0t3Llyh7385tvvolhJz07Un9r167V1KlTdfzxx8vj8Wjz5s29Wtct+9ef/pzaP8KNw+rr67Vz587IreuNCq+88srDnrd169ao83Jycgai3D5bsmSJli1bpsrKSn388ccqKyvTI488oqVLlx7ynMbGRl166aX60Y9+pIaGBt1777369a9/rTVr1gxg5b3Tn/66uGUPb731VtXU1Oipp57SBx98oMLCQl188cXasWNHj/PdtH9S3/vrMhj3b9++fTrrrLNUWVnZ4/1lZWUqLy9XZWWl6uvr5ff7VVBQoNbW1kOu+eabb+rqq6/WjBkz9P7772vGjBm66qqr9Pbbb8eqjUOKRX/St+/u+9293Llzp5KTk2PRwmEdqb99+/YpLy9Pixcv7vWabtq//vQnObR/Mf3kKpg777zTnHzyyaazs7PH+1955RUjyTQ3Nw9sYf00bdo0c/PNN0eNTZ8+3Vx//fWHPGfevHlm3LhxUWOzZs0y559/fkxqPBr96c9Ne9jW1mYSExPN888/HzV+1llnmQULFvR4jpv2rz/9uWX/JJl169ZFjjs7O43f7zeLFy+OjH3zzTdm+PDhZtmyZYdc56qrrjI/+clPosamTp1qrrnmGsdr7gun+luxYoUZPnx4DCvtn4P7+67GxkYjyTQ0NBxxHbfs33f1pT+n9o8rNzHU3t6u1atX6+abbz7iB3yOHz9emZmZuuiii/TKK68MUIV9d8EFF+hvf/ubPvnkE0nS+++/r40bN+rSSy895DlvvvmmCgsLo8amTp2qTZs2KRwOx7TevupPf13csIcHDhxQR0dHt38FpaSkaOPGjT2e46b9609/Xdywf9/V2NiopqamqL3x+XyaMmWK6urqDnneofbzcOfEQ3/7k6S9e/dq9OjROumkk3TZZZepoaEh1uUOGLfs39FwYv8INzH07LPP6uuvv9aNN954yDmZmZlavny51qxZo7Vr1+rUU0/VRRddpNdee23gCu2De+65R9dee63GjRsnr9er8ePHq7i4WNdee+0hz2lqaur2gagZGRk6cOCAvvzyy1iX3Cf96c9Ne5iWlqZJkybpoYce0hdffKGOjg6tXr1ab7/9tnbu3NnjOW7av/7056b9+66uDx7uaW8O/lDig8/r6znx0N/+xo0bp5UrV+q5557Tn/70JyUnJysvL0//+te/YlrvQHHL/vWXU/sXt49fGAqefPJJXXLJJQoEAoecc+qpp+rUU0+NHE+aNEnbt2/X7373O1144YUDUWafPPPMM1q9erWqqqp0xhlnaPPmzSouLlYgENDMmTMPed7BV67M/74x9pGuaA20/vTntj186qmndPPNN+vEE09UYmKizj77bBUVFem999475Dlu2T+p7/25bf8O1tPeHGlf+nNOvPS11vPPP1/nn39+5DgvL09nn322li5dqj/84Q8xq3MguWn/+sqp/ePKTYxs27ZNL7/8sm699dY+n3v++ecP2n9l3H333SopKdE111yjM888UzNmzNBdd92lRYsWHfIcv9/f7V8Vu3btUlJSkkaOHBnrkvukP/31ZDDv4cknn6za2lrt3btX27dv1zvvvKNwOKzs7Owe57tp/6S+99eTwbx/Xfx+vyT1uDcH/8v+4PP6ek489Le/gyUkJOjcc88d9PvZW27ZP6f0d/8INzGyYsUKnXDCCZo2bVqfz21oaFBmZmYMqjp6bW1tSkiIftokJiYe9qXSkyZNirxqrEt1dbUmTJggr9cbkzr7qz/99WQw72GXY445RpmZmWpubtaGDRt0xRVX9DjPTfv3Xb3trydu2L/s7Gz5/f6ovWlvb1dtba0mT558yPMOtZ+HOyce+tvfwYwx2rx586Dfz95yy/45pd/7d9S/koxuOjo6zKhRo8w999zT7b6SkhIzY8aMyPGjjz5q1q1bZz755BPz4YcfmpKSEiPJrFmzZiBL7rWZM2eaE0880Tz//POmsbHRrF271hx//PFm3rx5kTkH9/jZZ5+Z1NRUc9ddd5mPPvrIPPnkk8br9Zq//OUv8WjhsPrTn9v28KWXXjIvvvii+eyzz0x1dbU566yzzHnnnWfa29uNMe7eP2P63t9g3r/W1lbT0NBgGhoajCRTXl5uGhoazLZt24wxxixevNgMHz7crF271nzwwQfm2muvNZmZmaalpSWyxowZM0xJSUnk+I033jCJiYlm8eLF5uOPPzaLFy82SUlJ5q233rKiv2AwaF566SXz6aefmoaGBnPTTTeZpKQk8/bbbw+6/nbv3m0aGhrM+vXrjSTz9NNPm4aGBrNz585D9uem/etPf07tH+EmBjZs2GAkma1bt3a7b+bMmWbKlCmR4yVLlpiTTz7ZJCcnm+OOO85ccMEFZv369QNYbd+0tLSYO++804waNcokJyebsWPHmgULFphQKBSZc3CPxhjz6quvmvHjx5thw4aZMWPGmMcff3yAK++d/vTntj185plnzNixY82wYcOM3+83c+bMMV9//XXkfjfvnzF9728w71/Xy9QPvs2cOdMY8+3LpR944AHj9/uNz+czF154ofnggw+i1pgyZUpkfpc///nP5tRTTzVer9eMGzcubkEuFv0VFxebUaNGmWHDhpnvfe97prCw0NTV1Q1gV//nSP2tWLGix/sfeOCByBpu3r/+9OfU/nmM+d/fDAQAALAAv3MDAACsQrgBAABWIdwAAACrEG4AAIBVCDcAAMAqhBsAAGAVwg0AALAK4QYAAFiFcAMAAKxCuAEAAFYh3AAAAKsQbgAAgFX+P3D9fjmPc7OvAAAAAElFTkSuQmCC",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "df['TotalIncome']=df['ApplicantIncome']+df['CoapplicantIncome']\n",
    "df['TotalIncome_log']=np.log(df['TotalIncome'])\n",
    "df['TotalIncome_log'].hist(bins=20)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "id": "414be121",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Loan_ID              0\n",
       "Gender               0\n",
       "Married              0\n",
       "Dependents           0\n",
       "Education            0\n",
       "Self_Employed        0\n",
       "ApplicantIncome      0\n",
       "CoapplicantIncome    0\n",
       "LoanAmount           0\n",
       "Loan_Amount_Term     0\n",
       "Credit_History       0\n",
       "Property_Area        0\n",
       "Loan_Status          0\n",
       "loanAmount_log       0\n",
       "TotalIncome          0\n",
       "TotalIncome_log      0\n",
       "dtype: int64"
      ]
     },
     "execution_count": 23,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df['Gender'].fillna(df['Gender'].mode()[0], inplace=True)\n",
    "df['Married'].fillna(df['Married'].mode()[0], inplace=True)\n",
    "df['Self_Employed'].fillna(df['Self_Employed'].mode()[0], inplace=True)\n",
    "df['Dependents'].fillna(df['Dependents'].mode()[0], inplace=True)\n",
    "\n",
    "df.LoanAmount =df.LoanAmount.fillna(df.LoanAmount.mean())\n",
    "df.loanAmount_log =df.loanAmount_log.fillna(df.loanAmount_log.mean())\n",
    "\n",
    "df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mode()[0], inplace=True)\n",
    "df['Credit_History'].fillna(df['Credit_History'].mode()[0], inplace=True)\n",
    "\n",
    "df.isnull().sum() \n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "id": "e890c06f",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([['Male', 'No', '0', ..., 1.0, 4.857444178729352, 5849.0],\n",
       "       ['Male', 'Yes', '1', ..., 1.0, 4.852030263919617, 6091.0],\n",
       "       ['Male', 'Yes', '0', ..., 1.0, 4.189654742026425, 3000.0],\n",
       "       ...,\n",
       "       ['Male', 'Yes', '1', ..., 1.0, 5.53338948872752, 8312.0],\n",
       "       ['Male', 'Yes', '2', ..., 1.0, 5.231108616854587, 7583.0],\n",
       "       ['Female', 'No', '0', ..., 0.0, 4.890349128221754, 4583.0]],\n",
       "      dtype=object)"
      ]
     },
     "execution_count": 24,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "x = df.iloc[:,np.r_[1:5,9:11,13:15]].values\n",
    "y = df.iloc[:,12].values\n",
    "\n",
    "x"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "id": "d5f8ffb1",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array(['Y', 'N', 'Y', 'Y', 'Y', 'Y', 'Y', 'N', 'Y', 'N', 'Y', 'Y', 'Y',\n",
       "       'N', 'Y', 'Y', 'Y', 'N', 'N', 'Y', 'N', 'Y', 'N', 'N', 'N', 'Y',\n",
       "       'Y', 'Y', 'N', 'Y', 'N', 'N', 'N', 'Y', 'N', 'Y', 'N', 'Y', 'Y',\n",
       "       'Y', 'N', 'Y', 'Y', 'Y', 'Y', 'Y', 'Y', 'Y', 'N', 'Y', 'Y', 'Y',\n",
       "       'N', 'N', 'N', 'Y', 'Y', 'N', 'Y', 'Y', 'Y', 'Y', 'N', 'N', 'N',\n",
       "       'N', 'N', 'Y', 'Y', 'N', 'Y', 'Y', 'Y', 'N', 'Y', 'N', 'N', 'N',\n",
       "       'N', 'Y', 'Y', 'Y', 'N', 'N', 'Y', 'Y', 'Y', 'Y', 'Y', 'Y', 'Y',\n",
       "       'Y', 'Y', 'Y', 'Y', 'N', 'Y', 'Y', 'Y', 'Y', 'Y', 'Y', 'Y', 'Y',\n",
       "       'Y', 'Y', 'Y', 'N', 'N', 'Y', 'Y', 'Y', 'N', 'Y', 'Y', 'Y', 'Y',\n",
       "       'Y', 'N', 'Y', 'Y', 'Y', 'Y', 'Y', 'Y', 'Y', 'Y', 'Y', 'N', 'N',\n",
       "       'Y', 'Y', 'Y', 'Y', 'Y', 'N', 'N', 'Y', 'N', 'N', 'N', 'Y', 'Y',\n",
       "       'Y', 'Y', 'Y', 'Y', 'Y', 'N', 'Y', 'N', 'Y', 'N', 'N', 'Y', 'Y',\n",
       "       'Y', 'Y', 'Y', 'Y', 'Y', 'N', 'N', 'Y', 'Y', 'Y', 'N', 'Y', 'N',\n",
       "       'Y', 'Y', 'Y', 'N', 'Y', 'N', 'Y', 'Y', 'N', 'Y', 'N', 'N', 'N',\n",
       "       'Y', 'N', 'Y', 'Y', 'N', 'Y', 'Y', 'Y', 'Y', 'N', 'N', 'Y', 'Y',\n",
       "       'N', 'Y', 'Y', 'Y', 'N', 'Y', 'Y', 'N', 'Y', 'Y', 'Y', 'Y', 'Y',\n",
       "       'Y', 'N', 'N', 'N', 'Y', 'Y', 'Y', 'Y', 'N', 'Y', 'N', 'Y', 'N',\n",
       "       'Y', 'Y', 'Y', 'Y', 'N', 'N', 'Y', 'Y', 'Y', 'Y', 'Y', 'Y', 'Y',\n",
       "       'Y', 'Y', 'N', 'Y', 'Y', 'Y', 'Y', 'N', 'Y', 'Y', 'Y', 'N', 'Y',\n",
       "       'Y', 'Y', 'Y', 'N', 'N', 'Y', 'Y', 'N', 'Y', 'N', 'N', 'N', 'N',\n",
       "       'Y', 'Y', 'N', 'Y', 'Y', 'Y', 'Y', 'Y', 'N', 'Y', 'Y', 'Y', 'Y',\n",
       "       'Y', 'Y', 'Y', 'Y', 'Y', 'Y', 'Y', 'N', 'Y', 'Y', 'Y', 'N', 'Y',\n",
       "       'N', 'Y', 'Y', 'Y', 'Y', 'N', 'Y', 'N', 'Y', 'Y', 'Y', 'Y', 'N',\n",
       "       'N', 'N', 'Y', 'Y', 'Y', 'Y', 'N', 'Y', 'N', 'N', 'Y', 'Y', 'Y',\n",
       "       'Y', 'Y', 'N', 'Y', 'Y', 'Y', 'Y', 'N', 'Y', 'Y', 'Y', 'Y', 'Y',\n",
       "       'N', 'Y', 'Y', 'N', 'Y', 'Y', 'Y', 'Y', 'Y', 'Y', 'Y', 'Y', 'Y',\n",
       "       'N', 'Y', 'N', 'N', 'Y', 'Y', 'Y', 'Y', 'N', 'Y', 'Y', 'Y', 'Y',\n",
       "       'N', 'Y', 'N', 'Y', 'Y', 'Y', 'N', 'N', 'Y', 'N', 'Y', 'Y', 'Y',\n",
       "       'Y', 'N', 'N', 'N', 'Y', 'N', 'Y', 'Y', 'Y', 'N', 'Y', 'Y', 'Y',\n",
       "       'Y', 'N', 'Y', 'Y', 'Y', 'Y', 'Y', 'N', 'Y', 'Y', 'N', 'Y', 'Y',\n",
       "       'Y', 'Y', 'Y', 'Y', 'Y', 'Y', 'N', 'Y', 'Y', 'N', 'N', 'N', 'Y',\n",
       "       'Y', 'N', 'Y', 'Y', 'Y', 'N', 'N', 'N', 'Y', 'N', 'Y', 'N', 'Y',\n",
       "       'N', 'N', 'Y', 'Y', 'Y', 'N', 'Y', 'N', 'Y', 'Y', 'N', 'Y', 'Y',\n",
       "       'Y', 'Y', 'N', 'Y', 'Y', 'Y', 'Y', 'Y', 'Y', 'N', 'Y', 'Y', 'Y',\n",
       "       'Y', 'Y', 'Y', 'Y', 'Y', 'N', 'N', 'N', 'N', 'Y', 'N', 'Y', 'Y',\n",
       "       'Y', 'Y', 'N', 'Y', 'N', 'Y', 'Y', 'Y', 'Y', 'N', 'Y', 'N', 'Y',\n",
       "       'Y', 'N', 'Y', 'N', 'Y', 'Y', 'Y', 'Y', 'Y', 'N', 'Y', 'N', 'Y',\n",
       "       'Y', 'Y', 'Y', 'Y', 'Y', 'N', 'N', 'Y', 'N', 'Y', 'Y', 'Y', 'Y',\n",
       "       'N', 'Y', 'Y', 'Y', 'Y', 'N', 'Y', 'Y', 'Y', 'N', 'Y', 'Y', 'Y',\n",
       "       'N', 'Y', 'Y', 'N', 'Y', 'Y', 'N', 'N', 'Y', 'Y', 'N', 'N', 'N',\n",
       "       'Y', 'Y', 'Y', 'Y', 'N', 'Y', 'Y', 'Y', 'Y', 'Y', 'Y', 'Y', 'N',\n",
       "       'N', 'Y', 'Y', 'Y', 'Y', 'N', 'Y', 'Y', 'N', 'Y', 'Y', 'Y', 'Y',\n",
       "       'N', 'Y', 'N', 'Y', 'N', 'Y', 'Y', 'N', 'N', 'Y', 'Y', 'Y', 'Y',\n",
       "       'Y', 'Y', 'Y', 'Y', 'Y', 'N', 'Y', 'Y', 'N', 'N', 'N', 'Y', 'N',\n",
       "       'Y', 'N', 'N', 'Y', 'N', 'Y', 'Y', 'Y', 'Y', 'N', 'Y', 'N', 'N',\n",
       "       'N', 'Y', 'Y', 'Y', 'N', 'Y', 'N', 'Y', 'Y', 'Y', 'Y', 'N', 'N',\n",
       "       'Y', 'Y', 'N', 'Y', 'Y', 'Y', 'Y', 'N', 'Y', 'Y', 'Y', 'Y', 'Y',\n",
       "       'Y', 'Y', 'N'], dtype=object)"
      ]
     },
     "execution_count": 25,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "y"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "id": "cd637965",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "per of missing gender is 0.000000%\n"
     ]
    }
   ],
   "source": [
    "print(\"per of missing gender is %2f%%\" %((df['Gender'].isnull().sum()/df.shape[0])*100))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "id": "d12428b7",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "number of people who take loan as group by gender:\n",
      "Male      502\n",
      "Female    112\n",
      "Name: Gender, dtype: int64\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "<Axes: xlabel='Gender', ylabel='count'>"
      ]
     },
     "execution_count": 27,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAjsAAAGwCAYAAABPSaTdAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjcuMSwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy/bCgiHAAAACXBIWXMAAA9hAAAPYQGoP6dpAAAnd0lEQVR4nO3df3CU9YHH8c+SH0sI2QUS2GXrAkFCq01QCRbDiVACRBTQehUtnEKhV3pBagoIl0MrXG1S8QSupaXoqVAYG2dOaK+UgwQqOZEykBQUEFEwnFCyDULYJBA2kDz3R4dnbg2gzQ928+X9mtmZ7Pf57rPfJzMhb559duOwLMsSAACAoTpFegEAAADtidgBAABGI3YAAIDRiB0AAGA0YgcAABiN2AEAAEYjdgAAgNFiI72AaNDU1KSTJ08qKSlJDocj0ssBAABfgGVZqq2tlc/nU6dOVz9/Q+xIOnnypPx+f6SXAQAAWuD48eO66aabrrqd2JGUlJQk6a/fLJfLFeHVAACAL6KmpkZ+v9/+PX41xI5kv3TlcrmIHQAAOpjPuwSFC5QBAIDRiB0AAGA0YgcAABiN2AEAAEYjdgAAgNGIHQAAYDRiBwAAGI3YAQAARiN2AACA0YgdAABgNGIHAAAYjdgBAABGI3YAAIDRIho7ixYtksPhCLt5vV57u2VZWrRokXw+nxISEjRy5EgdPHgwbB+hUEizZ89WSkqKEhMTNXHiRJ04ceJ6HwoAAIhSET+z89WvflWVlZX2bf/+/fa2JUuWaOnSpVqxYoX27Nkjr9erMWPGqLa21p6Tl5enDRs2qKioSDt27FBdXZ3Gjx+vxsbGSBwOAACIMrERX0BsbNjZnMssy9Ly5cu1cOFCPfTQQ5KkNWvWyOPx6PXXX9fMmTMVDAb1yiuvaO3atRo9erQkad26dfL7/dq6datycnKu+JyhUEihUMi+X1NT0w5H1lzZkK9dl+cBOpIhZbsjvQQAhov4mZ2PPvpIPp9PqampevTRR/Xxxx9LkioqKhQIBDR27Fh7rtPp1IgRI7Rz505JUnl5uS5evBg2x+fzKT093Z5zJYWFhXK73fbN7/e309EBAIBIi2jsDB06VL/61a+0ZcsWvfzyywoEAho2bJhOnz6tQCAgSfJ4PGGP8Xg89rZAIKD4+Hh17979qnOuJD8/X8Fg0L4dP368jY8MAABEi4i+jDVu3Dj764yMDGVlZenmm2/WmjVrdNddd0mSHA5H2GMsy2o29lmfN8fpdMrpdLZi5QAAoKOI+MtY/19iYqIyMjL00Ucf2dfxfPYMTVVVlX22x+v1qqGhQdXV1VedAwAAbmxRFTuhUEiHDh1S7969lZqaKq/Xq5KSEnt7Q0ODSktLNWzYMElSZmam4uLiwuZUVlbqwIED9hwAAHBji+jLWPPmzdOECRPUp08fVVVV6bnnnlNNTY2mTp0qh8OhvLw8FRQUKC0tTWlpaSooKFCXLl00efJkSZLb7daMGTM0d+5cJScnq0ePHpo3b54yMjLsd2cBAIAbW0Rj58SJE/rWt76lTz/9VD179tRdd92lXbt2qW/fvpKk+fPnq76+Xrm5uaqurtbQoUNVXFyspKQkex/Lli1TbGysJk2apPr6emVnZ2v16tWKiYmJ1GEBAIAo4rAsy4r0IiKtpqZGbrdbwWBQLper3Z6Hz9kBmuNzdgC01Bf9/R1V1+wAAAC0NWIHAAAYjdgBAABGI3YAAIDRiB0AAGA0YgcAABiN2AEAAEYjdgAAgNGIHQAAYDRiBwAAGI3YAQAARiN2AACA0YgdAABgNGIHAAAYjdgBAABGI3YAAIDRiB0AAGA0YgcAABiN2AEAAEYjdgAAgNGIHQAAYDRiBwAAGI3YAQAARiN2AACA0YgdAABgNGIHAAAYjdgBAABGI3YAAIDRiB0AAGA0YgcAABiN2AEAAEYjdgAAgNGIHQAAYDRiBwAAGI3YAQAARiN2AACA0YgdAABgNGIHAAAYjdgBAABGI3YAAIDRiB0AAGA0YgcAABiN2AEAAEYjdgAAgNGIHQAAYDRiBwAAGI3YAQAARiN2AACA0YgdAABgNGIHAAAYjdgBAABGI3YAAIDRiB0AAGA0YgcAABiN2AEAAEYjdgAAgNGIHQAAYDRiBwAAGI3YAQAARoua2CksLJTD4VBeXp49ZlmWFi1aJJ/Pp4SEBI0cOVIHDx4Me1woFNLs2bOVkpKixMRETZw4USdOnLjOqwcAANEqKmJnz549eumllzRo0KCw8SVLlmjp0qVasWKF9uzZI6/XqzFjxqi2ttaek5eXpw0bNqioqEg7duxQXV2dxo8fr8bGxut9GAAAIApFPHbq6uo0ZcoUvfzyy+revbs9blmWli9froULF+qhhx5Senq61qxZo/Pnz+v111+XJAWDQb3yyit68cUXNXr0aN1xxx1at26d9u/fr61bt171OUOhkGpqasJuAADATBGPnVmzZun+++/X6NGjw8YrKioUCAQ0duxYe8zpdGrEiBHauXOnJKm8vFwXL14Mm+Pz+ZSenm7PuZLCwkK53W775vf72/ioAABAtIho7BQVFelPf/qTCgsLm20LBAKSJI/HEzbu8XjsbYFAQPHx8WFnhD4750ry8/MVDAbt2/Hjx1t7KAAAIErFRuqJjx8/rieffFLFxcXq3LnzVec5HI6w+5ZlNRv7rM+b43Q65XQ6/7YFAwCADiliZ3bKy8tVVVWlzMxMxcbGKjY2VqWlpfrpT3+q2NhY+4zOZ8/QVFVV2du8Xq8aGhpUXV191TkAAODGFrHYyc7O1v79+7Vv3z77NmTIEE2ZMkX79u1T//795fV6VVJSYj+moaFBpaWlGjZsmCQpMzNTcXFxYXMqKyt14MABew4AALixRexlrKSkJKWnp4eNJSYmKjk52R7Py8tTQUGB0tLSlJaWpoKCAnXp0kWTJ0+WJLndbs2YMUNz585VcnKyevTooXnz5ikjI6PZBc8AAODGFLHY+SLmz5+v+vp65ebmqrq6WkOHDlVxcbGSkpLsOcuWLVNsbKwmTZqk+vp6ZWdna/Xq1YqJiYngygEAQLRwWJZlRXoRkVZTUyO3261gMCiXy9Vuz1M25Gvttm+goxpStjvSSwDQQX3R398R/5wdAACA9kTsAAAAoxE7AADAaMQOAAAwGrEDAACMRuwAAACjETsAAMBoxA4AADAasQMAAIxG7AAAAKMROwAAwGjEDgAAMBqxAwAAjEbsAAAAoxE7AADAaMQOAAAwGrEDAACMRuwAAACjETsAAMBoxA4AADAasQMAAIxG7AAAAKMROwAAwGjEDgAAMBqxAwAAjEbsAAAAoxE7AADAaMQOAAAwGrEDAACMRuwAAACjETsAAMBoxA4AADAasQMAAIxG7AAAAKMROwAAwGjEDgAAMBqxAwAAjEbsAAAAoxE7AADAaMQOAAAwGrEDAACMRuwAAACjETsAAMBoxA4AADAasQMAAIxG7AAAAKMROwAAwGjEDgAAMBqxAwAAjEbsAAAAoxE7AADAaMQOAAAwGrEDAACMRuwAAACjETsAAMBoxA4AADAasQMAAIxG7AAAAKMROwAAwGgRjZ2VK1dq0KBBcrlccrlcysrK0n//93/b2y3L0qJFi+Tz+ZSQkKCRI0fq4MGDYfsIhUKaPXu2UlJSlJiYqIkTJ+rEiRPX+1AAAECUimjs3HTTTfrJT36isrIylZWVadSoUXrggQfsoFmyZImWLl2qFStWaM+ePfJ6vRozZoxqa2vtfeTl5WnDhg0qKirSjh07VFdXp/Hjx6uxsTFShwUAAKKIw7IsK9KL+P969OihF154QdOnT5fP51NeXp4WLFgg6a9ncTwej55//nnNnDlTwWBQPXv21Nq1a/XII49Ikk6ePCm/369NmzYpJyfnis8RCoUUCoXs+zU1NfL7/QoGg3K5XO12bGVDvtZu+wY6qiFluyO9BAAdVE1Njdxu9+f+/o6aa3YaGxtVVFSkc+fOKSsrSxUVFQoEAho7dqw9x+l0asSIEdq5c6ckqby8XBcvXgyb4/P5lJ6ebs+5ksLCQrndbvvm9/vb78AAAEBERTx29u/fr65du8rpdOp73/ueNmzYoFtvvVWBQECS5PF4wuZ7PB57WyAQUHx8vLp3737VOVeSn5+vYDBo344fP97GRwUAAKJFbKQX8OUvf1n79u3T2bNn9eabb2rq1KkqLS21tzscjrD5lmU1G/usz5vjdDrldDpbt3AAANAhRPzMTnx8vAYMGKAhQ4aosLBQt912m/793/9dXq9XkpqdoamqqrLP9ni9XjU0NKi6uvqqcwAAwI0t4rHzWZZlKRQKKTU1VV6vVyUlJfa2hoYGlZaWatiwYZKkzMxMxcXFhc2prKzUgQMH7DkAAODGFtGXsf7lX/5F48aNk9/vV21trYqKirR9+3Zt3rxZDodDeXl5KigoUFpamtLS0lRQUKAuXbpo8uTJkiS3260ZM2Zo7ty5Sk5OVo8ePTRv3jxlZGRo9OjRkTw0AAAQJSIaO3/5y1/02GOPqbKyUm63W4MGDdLmzZs1ZswYSdL8+fNVX1+v3NxcVVdXa+jQoSouLlZSUpK9j2XLlik2NlaTJk1SfX29srOztXr1asXExETqsAAAQBSJus/ZiYQv+j791uJzdoDm+JwdAC3V4T5nBwAAoD20KHZGjRqls2fPNhuvqanRqFGjWrsmAACANtOi2Nm+fbsaGhqajV+4cEFvv/12qxcFAADQVv6mC5Tfe+89++v3338/7DNwGhsbtXnzZn3pS19qu9UBAAC00t8UO7fffrscDoccDscVX65KSEjQz372szZbHAAAQGv9TbFTUVEhy7LUv39/7d69Wz179rS3xcfHq1evXrzlGwAARJW/KXb69u0rSWpqamqXxQAAALS1Fn+o4Icffqjt27erqqqqWfz88Ic/bPXCAAAA2kKLYufll1/WP/3TPyklJUVerzfsL4w7HA5iBwAARI0Wxc5zzz2nH//4x1qwYEFbrwcAAKBNtehzdqqrq/Xwww+39VoAAADaXIti5+GHH1ZxcXFbrwUAAKDNtehlrAEDBuiZZ57Rrl27lJGRobi4uLDt3//+99tkcQAAAK3Vor96npqaevUdOhz6+OOPW7Wo642/eg5EDn/1HEBLfdHf3y06s1NRUdHihQEAAFxPLbpmBwAAoKNo0Zmd6dOnX3P7q6++2qLFAAAAtLUWxU51dXXY/YsXL+rAgQM6e/bsFf9AKAAAQKS0KHY2bNjQbKypqUm5ubnq379/qxcFAADQVtrsmp1OnTrpBz/4gZYtW9ZWuwQAAGi1Nr1A+ejRo7p06VJb7hIAAKBVWvQy1pw5c8LuW5alyspK/f73v9fUqVPbZGEAAABtoUWxs3fv3rD7nTp1Us+ePfXiiy9+7ju1AAAArqcWxc5bb73V1usAAABoFy2KnctOnTqlw4cPy+FwaODAgerZs2dbrQsAAKBNtOgC5XPnzmn69Onq3bu37rnnHg0fPlw+n08zZszQ+fPn23qNAAAALdai2JkzZ45KS0v1u9/9TmfPntXZs2f129/+VqWlpZo7d25brxEAAKDFWvQy1ptvvqn//M//1MiRI+2x++67TwkJCZo0aZJWrlzZVusDAABolRad2Tl//rw8Hk+z8V69evEyFgAAiCotip2srCw9++yzunDhgj1WX1+vxYsXKysrq80WBwAA0Fotehlr+fLlGjdunG666Sbddtttcjgc2rdvn5xOp4qLi9t6jQAAAC3WotjJyMjQRx99pHXr1umDDz6QZVl69NFHNWXKFCUkJLT1GgEAAFqsRbFTWFgoj8ejf/zHfwwbf/XVV3Xq1CktWLCgTRYHAADQWi26ZmfVqlX6yle+0mz8q1/9qn75y1+2elEAAABtpUWxEwgE1Lt372bjPXv2VGVlZasXBQAA0FZaFDt+v1/vvPNOs/F33nlHPp+v1YsCAABoKy26Zuc73/mO8vLydPHiRY0aNUqStG3bNs2fP59PUAYAAFGlRbEzf/58nTlzRrm5uWpoaJAkde7cWQsWLFB+fn6bLhAAAKA1HJZlWS19cF1dnQ4dOqSEhASlpaXJ6XS25dqum5qaGrndbgWDQblcrnZ7nrIhX2u3fQMd1ZCy3ZFeAoAO6ov+/m7RmZ3LunbtqjvvvLM1uwAAAGhXLbpAGQAAoKMgdgAAgNGIHQAAYDRiBwAAGI3YAQAARiN2AACA0YgdAABgNGIHAAAYjdgBAABGI3YAAIDRiB0AAGA0YgcAABiN2AEAAEYjdgAAgNGIHQAAYDRiBwAAGI3YAQAARiN2AACA0YgdAABgtIjGTmFhoe68804lJSWpV69eevDBB3X48OGwOZZladGiRfL5fEpISNDIkSN18ODBsDmhUEizZ89WSkqKEhMTNXHiRJ04ceJ6HgoAAIhSEY2d0tJSzZo1S7t27VJJSYkuXbqksWPH6ty5c/acJUuWaOnSpVqxYoX27Nkjr9erMWPGqLa21p6Tl5enDRs2qKioSDt27FBdXZ3Gjx+vxsbGSBwWAACIIg7LsqxIL+KyU6dOqVevXiotLdU999wjy7Lk8/mUl5enBQsWSPrrWRyPx6Pnn39eM2fOVDAYVM+ePbV27Vo98sgjkqSTJ0/K7/dr06ZNysnJafY8oVBIoVDIvl9TUyO/369gMCiXy9Vux1c25Gvttm+goxpStjvSSwDQQdXU1Mjtdn/u7++oumYnGAxKknr06CFJqqioUCAQ0NixY+05TqdTI0aM0M6dOyVJ5eXlunjxYtgcn8+n9PR0e85nFRYWyu122ze/399ehwQAACIsamLHsizNmTNHd999t9LT0yVJgUBAkuTxeMLmejwee1sgEFB8fLy6d+9+1TmflZ+fr2AwaN+OHz/e1ocDAACiRGykF3DZE088offee087duxots3hcITdtyyr2dhnXWuO0+mU0+ls+WIBAECHERVndmbPnq3/+q//0ltvvaWbbrrJHvd6vZLU7AxNVVWVfbbH6/WqoaFB1dXVV50DAABuXBGNHcuy9MQTT2j9+vX6wx/+oNTU1LDtqamp8nq9KikpsccaGhpUWlqqYcOGSZIyMzMVFxcXNqeyslIHDhyw5wAAgBtXRF/GmjVrll5//XX99re/VVJSkn0Gx+12KyEhQQ6HQ3l5eSooKFBaWprS0tJUUFCgLl26aPLkyfbcGTNmaO7cuUpOTlaPHj00b948ZWRkaPTo0ZE8PAAAEAUiGjsrV66UJI0cOTJs/LXXXtO0adMkSfPnz1d9fb1yc3NVXV2toUOHqri4WElJSfb8ZcuWKTY2VpMmTVJ9fb2ys7O1evVqxcTEXK9DAQAAUSqqPmcnUr7o+/Rbi8/ZAZrjc3YAtFSH/JwdAACAtkbsAAAAoxE7AADAaMQOAAAwGrEDAACMRuwAAACjETsAAMBoxA4AADAasQMAAIxG7AAAAKMROwAAwGjEDgAAMBqxAwAAjEbsAAAAoxE7AADAaMQOAAAwGrEDAACMRuwAAACjETsAAMBoxA4AADAasQMAAIxG7AAAAKMROwAAwGjEDgAAMBqxAwAAjEbsAAAAoxE7AADAaMQOAAAwGrEDAACMRuwAAACjETsAAMBoxA4AADAasQMAAIxG7AAAAKMROwAAwGjEDgAAMBqxAwAAjEbsAAAAoxE7AADAaMQOAAAwGrEDAACMRuwAAACjETsAAMBoxA4AADAasQMAAIxG7AAAAKMROwAAwGjEDgAAMBqxAwAAjEbsAAAAoxE7AADAaMQOAAAwGrEDAACMRuwAAACjxUZ6AQBggnufeSPSSwCizuYfPRLpJUjizA4AADAcsQMAAIxG7AAAAKMROwAAwGgRjZ3/+Z//0YQJE+Tz+eRwOPSb3/wmbLtlWVq0aJF8Pp8SEhI0cuRIHTx4MGxOKBTS7NmzlZKSosTERE2cOFEnTpy4jkcBAACiWURj59y5c7rtttu0YsWKK25fsmSJli5dqhUrVmjPnj3yer0aM2aMamtr7Tl5eXnasGGDioqKtGPHDtXV1Wn8+PFqbGy8XocBAACiWETfej5u3DiNGzfuitssy9Ly5cu1cOFCPfTQQ5KkNWvWyOPx6PXXX9fMmTMVDAb1yiuvaO3atRo9erQkad26dfL7/dq6datycnKu27EAAIDoFLXX7FRUVCgQCGjs2LH2mNPp1IgRI7Rz505JUnl5uS5evBg2x+fzKT093Z5zJaFQSDU1NWE3AABgpqiNnUAgIEnyeDxh4x6Px94WCAQUHx+v7t27X3XOlRQWFsrtdts3v9/fxqsHAADRImpj5zKHwxF237KsZmOf9Xlz8vPzFQwG7dvx48fbZK0AACD6RG3seL1eSWp2hqaqqso+2+P1etXQ0KDq6uqrzrkSp9Mpl8sVdgMAAGaK2thJTU2V1+tVSUmJPdbQ0KDS0lINGzZMkpSZmam4uLiwOZWVlTpw4IA9BwAA3Ngi+m6suro6HTlyxL5fUVGhffv2qUePHurTp4/y8vJUUFCgtLQ0paWlqaCgQF26dNHkyZMlSW63WzNmzNDcuXOVnJysHj16aN68ecrIyLDfnQUAAG5sEY2dsrIyff3rX7fvz5kzR5I0depUrV69WvPnz1d9fb1yc3NVXV2toUOHqri4WElJSfZjli1bptjYWE2aNEn19fXKzs7W6tWrFRMTc92PBwAARB+HZVlWpBcRaTU1NXK73QoGg+16/U7ZkK+1276BjmpI2e5IL6FN3PvMG5FeAhB1Nv/okXbd/xf9/R211+wAAAC0BWIHAAAYjdgBAABGI3YAAIDRiB0AAGA0YgcAABiN2AEAAEYjdgAAgNGIHQAAYDRiBwAAGI3YAQAARiN2AACA0YgdAABgNGIHAAAYjdgBAABGI3YAAIDRiB0AAGA0YgcAABiN2AEAAEYjdgAAgNGIHQAAYDRiBwAAGI3YAQAARiN2AACA0YgdAABgNGIHAAAYjdgBAABGI3YAAIDRiB0AAGA0YgcAABiN2AEAAEYjdgAAgNGIHQAAYDRiBwAAGI3YAQAARiN2AACA0YgdAABgNGIHAAAYjdgBAABGI3YAAIDRiB0AAGA0YgcAABiN2AEAAEYjdgAAgNGIHQAAYDRiBwAAGI3YAQAARiN2AACA0YgdAABgNGIHAAAYjdgBAABGI3YAAIDRiB0AAGA0YgcAABiN2AEAAEYjdgAAgNGIHQAAYDRiBwAAGI3YAQAARjMmdn7xi18oNTVVnTt3VmZmpt5+++1ILwkAAEQBI2LnjTfeUF5enhYuXKi9e/dq+PDhGjdunD755JNILw0AAESYEbGzdOlSzZgxQ9/5znd0yy23aPny5fL7/Vq5cmWklwYAACIsNtILaK2GhgaVl5frn//5n8PGx44dq507d17xMaFQSKFQyL4fDAYlSTU1Ne23UEl1jY3tun+gI2rvn7vr5VLofKSXAESd9v75vrx/y7KuOa/Dx86nn36qxsZGeTyesHGPx6NAIHDFxxQWFmrx4sXNxv1+f7usEcA1uN2RXgGAduJ+Yfp1eZ7a2lq5r/FvSYePncscDkfYfcuymo1dlp+frzlz5tj3m5qadObMGSUnJ1/1MTBHTU2N/H6/jh8/LpfLFenlAGhD/HzfWCzLUm1trXw+3zXndfjYSUlJUUxMTLOzOFVVVc3O9lzmdDrldDrDxrp169ZeS0SUcrlc/GMIGIqf7xvHtc7oXNbhL1COj49XZmamSkpKwsZLSko0bNiwCK0KAABEiw5/ZkeS5syZo8cee0xDhgxRVlaWXnrpJX3yySf63ve+F+mlAQCACDMidh555BGdPn1a//qv/6rKykqlp6dr06ZN6tu3b6SXhijkdDr17LPPNnspE0DHx883rsRhfd77tQAAADqwDn/NDgAAwLUQOwAAwGjEDgAAMBqxA0g6duyYHA6H9u3bF+mlAIiAfv36afny5ZFeBtoJsYMOa9q0aXI4HFf8iIHc3Fw5HA5Nmzbt+i8MwDVd/tn97O3IkSORXhoMReygQ/P7/SoqKlJ9fb09duHCBf36179Wnz59IrgyANdy7733qrKyMuyWmpoa6WXBUMQOOrTBgwerT58+Wr9+vT22fv16+f1+3XHHHfbY5s2bdffdd6tbt25KTk7W+PHjdfTo0Wvu+/3339d9992nrl27yuPx6LHHHtOnn37abscC3EicTqe8Xm/YLSYmRr/73e+UmZmpzp07q3///lq8eLEuXbpkP87hcGjVqlUaP368unTpoltuuUV//OMfdeTIEY0cOVKJiYnKysoK+/k+evSoHnjgAXk8HnXt2lV33nmntm7des31BYNBffe731WvXr3kcrk0atQovfvuu+32/UD7InbQ4X3729/Wa6+9Zt9/9dVXNX16+F/aPXfunObMmaM9e/Zo27Zt6tSpk77xjW+oqanpivusrKzUiBEjdPvtt6usrEybN2/WX/7yF02aNKldjwW4kW3ZskX/8A//oO9///t6//33tWrVKq1evVo//vGPw+b96Ec/0uOPP659+/bpK1/5iiZPnqyZM2cqPz9fZWVlkqQnnnjCnl9XV6f77rtPW7du1d69e5WTk6MJEybok08+ueI6LMvS/fffr0AgoE2bNqm8vFyDBw9Wdna2zpw5037fALQfC+igpk6daj3wwAPWqVOnLKfTaVVUVFjHjh2zOnfubJ06dcp64IEHrKlTp17xsVVVVZYka//+/ZZlWVZFRYUlydq7d69lWZb1zDPPWGPHjg17zPHjxy1J1uHDh9vzsADjTZ061YqJibESExPt2ze/+U1r+PDhVkFBQdjctWvXWr1797bvS7Kefvpp+/4f//hHS5L1yiuv2GO//vWvrc6dO19zDbfeeqv1s5/9zL7ft29fa9myZZZlWda2bdssl8tlXbhwIewxN998s7Vq1aq/+XgReUb8uQjc2FJSUnT//fdrzZo19v/IUlJSwuYcPXpUzzzzjHbt2qVPP/3UPqPzySefKD09vdk+y8vL9dZbb6lr167Nth09elQDBw5sn4MBbhBf//rXtXLlSvt+YmKiBgwYoD179oSdyWlsbNSFCxd0/vx5denSRZI0aNAge7vH45EkZWRkhI1duHBBNTU1crlcOnfunBYvXqyNGzfq5MmTunTpkurr6696Zqe8vFx1dXVKTk4OG6+vr//cl78RnYgdGGH69On2aeuf//znzbZPmDBBfr9fL7/8snw+n5qampSenq6GhoYr7q+pqUkTJkzQ888/32xb796923bxwA3octz8f01NTVq8eLEeeuihZvM7d+5sfx0XF2d/7XA4rjp2+T81Tz31lLZs2aJ/+7d/04ABA5SQkKBvfvOb1/z57927t7Zv395sW7du3b7YASKqEDswwr333mv/w5WTkxO27fTp0zp06JBWrVql4cOHS5J27Nhxzf0NHjxYb775pvr166fYWH5MgOth8ODBOnz4cLMIaq23335b06ZN0ze+8Q1Jf72G59ixY9dcRyAQUGxsrPr169ema0FkcIEyjBATE6NDhw7p0KFDiomJCdvWvXt3JScn66WXXtKRI0f0hz/8QXPmzLnm/mbNmqUzZ87oW9/6lnbv3q2PP/5YxcXFmj59uhobG9vzUIAb1g9/+EP96le/0qJFi3Tw4EEdOnRIb7zxhp5++ulW7XfAgAFav3699u3bp3fffVeTJ0++6psTJGn06NHKysrSgw8+qC1btujYsWPauXOnnn76afsCaHQsxA6M4XK55HK5mo136tRJRUVFKi8vV3p6un7wgx/ohRdeuOa+fD6f3nnnHTU2NionJ0fp6el68skn5Xa71akTPzZAe8jJydHGjRtVUlKiO++8U3fddZeWLl2qvn37tmq/y5YtU/fu3TVs2DBNmDBBOTk5Gjx48FXnOxwObdq0Sffcc4+mT5+ugQMH6tFHH9WxY8fsa4TQsTgsy7IivQgAAID2wn9RAQCA0YgdAABgNGIHAAAYjdgBAABGI3YAAIDRiB0AAGA0YgcAABiN2AEAAEYjdgDc8EaOHKm8vLxILwNAOyF2AESFQCCgJ598UgMGDFDnzp3l8Xh0991365e//KXOnz8f6eUB6MD4c84AIu7jjz/W3/3d36lbt24qKChQRkaGLl26pA8//FCvvvqqfD6fJk6cGOllXlVjY6McDgd/Nw2IUvxkAoi43NxcxcbGqqysTJMmTdItt9yijIwM/f3f/71+//vfa8KECZKkYDCo7373u+rVq5dcLpdGjRqld999197PokWLdPvtt2vt2rXq16+f3G63Hn30UdXW1tpzzp07p8cff1xdu3ZV79699eKLLzZbT0NDg+bPn68vfelLSkxM1NChQ7V9+3Z7++rVq9WtWzdt3LhRt956q5xOp/73f/+3/b5BAFqF2AEQUadPn1ZxcbFmzZqlxMTEK85xOByyLEv333+/AoGANm3apPLycg0ePFjZ2dk6c+aMPffo0aP6zW9+o40bN2rjxo0qLS3VT37yE3v7U089pbfeeksbNmxQcXGxtm/frvLy8rDn+/a3v6133nlHRUVFeu+99/Twww/r3nvv1UcffWTPOX/+vAoLC/Uf//EfOnjwoHr16tXG3xkAbcYCgAjatWuXJclav3592HhycrKVmJhoJSYmWvPnz7e2bdtmuVwu68KFC2Hzbr75ZmvVqlWWZVnWs88+a3Xp0sWqqamxtz/11FPW0KFDLcuyrNraWis+Pt4qKiqyt58+fdpKSEiwnnzyScuyLOvIkSOWw+Gw/vznP4c9T3Z2tpWfn29ZlmW99tprliRr3759bfNNANCuuGYHQFRwOBxh93fv3q2mpiZNmTJFoVBI5eXlqqurU3Jycti8+vp6HT161L7fr18/JSUl2fd79+6tqqoqSX8969PQ0KCsrCx7e48ePfTlL3/Zvv+nP/1JlmVp4MCBYc8TCoXCnjs+Pl6DBg1qxREDuF6IHQARNWDAADkcDn3wwQdh4/3795ckJSQkSJKamprUu3fvsGtnLuvWrZv9dVxcXNg2h8OhpqYmSZJlWZ+7nqamJsXExKi8vFwxMTFh27p27Wp/nZCQ0CzQAEQnYgdARCUnJ2vMmDFasWKFZs+efdXrdgYPHqxAIKDY2Fj169evRc81YMAAxcXFadeuXerTp48kqbq6Wh9++KFGjBghSbrjjjvU2NioqqoqDR8+vEXPAyC6cIEygIj7xS9+oUuXLmnIkCF64403dOjQIR0+fFjr1q3TBx98oJiYGI0ePVpZWVl68MEHtWXLFh07dkw7d+7U008/rbKysi/0PF27dtWMGTP01FNPadu2bTpw4ICmTZsW9pbxgQMHasqUKXr88ce1fv16VVRUaM+ePXr++ee1adOm9voWAGhHnNkBEHE333yz9u7dq4KCAuXn5+vEiRNyOp269dZbNW/ePOXm5srhcGjTpk1auHChpk+frlOnTsnr9eqee+6Rx+P5ws/1wgsvqK6uThMnTlRSUpLmzp2rYDAYNue1117Tc889p7lz5+rPf/6zkpOTlZWVpfvuu6+tDx3AdeCwvsiL2AAAAB0UL2MBAACjETsAAMBoxA4AADAasQMAAIxG7AAAAKMROwAAwGjEDgAAMBqxAwAAjEbsAAAAoxE7AADAaMQOAAAw2v8BgMOyJE8NPqIAAAAASUVORK5CYII=",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "print(\"number of people who take loan as group by gender:\")\n",
    "print(df['Gender'].value_counts())\n",
    "sns.countplot(x='Gender',data=df,palette= 'Set1')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "id": "95652735",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "number of people who take loan as group by gender:\n",
      "Male      502\n",
      "Female    112\n",
      "Name: Gender, dtype: int64\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "<Axes: xlabel='Gender', ylabel='count'>"
      ]
     },
     "execution_count": 28,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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SU9PV0xMTLOzOFVVVc3O9lzmdDrldDpDxjp37txWS0SUSklJ4R9DwFD8fN84rnVG57J2f4FyfHy8Bg4cqC1btoSMb9myRYMHD47QqgAAQLRo92d2JGn27Nl6/PHHdeeddyo7O1svvPCCPv30U33/+1wLAADAjc6I2Hn00Ud1+vRp/du//ZsqKyuVmZmpjRs3qmfPnpFeGqKQ0+nUs88+2+ylTADtHz/fuBKH9UXv1wIAAGjH2v01OwAAANdC7AAAAKMROwAAwGjEDiDp2LFjcjgc2rdvX6SXAiACevXqpWXLlkV6GWgjxA7aralTp8rhcFzxIwZyc3PlcDg0derU678wANd0+Wf387cjR45EemkwFLGDds3n86mkpET19f//j6xeuHBBv/71r9WjR48IrgzAtdx3332qrKwMufXu3TvSy4KhiB20awMGDFCPHj20bt06e2zdunXy+Xy644477LFNmzbpnnvuUefOnZWWlqaxY8fq6NGj13zsDz74QPfff7+SkpLkdrv1+OOP67PPPmuzYwFuJE6nUx6PJ+QWExOj3/3udxo4cKA6duyoPn36aMGCBbp06ZJ9P4fDoZUrV2rs2LHq1KmTbrnlFv3xj3/UkSNHlJOTo8TERGVnZ4f8fB89elQPPvig3G63kpKSdNddd2nr1q3XXF8gEND3vvc9devWTSkpKRo2bJjee++9Nvt+oG0RO2j3vvOd7+iVV16xt19++WVNmzYtZM65c+c0e/Zs7d69W9u2bVOHDh30zW9+U01NTVd8zMrKSg0dOlS333679uzZo02bNukvf/mLJkyY0KbHAtzINm/erH/6p3/SD3/4Q33wwQdauXKlVq1apZ/85Cch8/793/9dkydP1r59+/S1r31NEydO1IwZM1RYWKg9e/ZIkp544gl7fl1dne6//35t3bpVe/fu1ejRozVu3Dh9+umnV1yHZVl64IEH5Pf7tXHjRpWXl2vAgAEaPny4zpw503bfALQdC2inpkyZYj344IPWqVOnLKfTaVVUVFjHjh2zOnbsaJ06dcp68MEHrSlTplzxvlVVVZYka//+/ZZlWVZFRYUlydq7d69lWZb1zDPPWKNGjQq5z/Hjxy1J1uHDh9vysADjTZkyxYqJibESExPt27e+9S1ryJAhVlFRUcjcNWvWWN27d7e3JVlPP/20vf3HP/7RkmS99NJL9tivf/1rq2PHjtdcw6233mr9/Oc/t7d79uxpLV261LIsy9q2bZuVkpJiXbhwIeQ+N998s7Vy5cq/+3gReUb8uQjc2NLT0/XAAw9o9erV9v/I0tPTQ+YcPXpUzzzzjHbt2qXPPvvMPqPz6aefKjMzs9ljlpeX66233lJSUlKzfUePHlW/fv3a5mCAG8Q3vvENrVixwt5OTExU3759tXv37pAzOY2Njbpw4YLOnz+vTp06SZL69+9v73e73ZKkrKyskLELFy6opqZGKSkpOnfunBYsWKANGzbo5MmTunTpkurr6696Zqe8vFx1dXVKS0sLGa+vr//Cl78RnYgdGGHatGn2aetf/OIXzfaPGzdOPp9PL774orxer5qampSZmamGhoYrPl5TU5PGjRunhQsXNtvXvXv38C4euAFdjpu/1dTUpAULFujhhx9uNr9jx47213FxcfbXDofjqmOX/1Pz1FNPafPmzfqP//gP9e3bVwkJCfrWt751zZ//7t27a/v27c32de7c+csdIKIKsQMj3HffffY/XKNHjw7Zd/r0aR06dEgrV67UkCFDJEk7duy45uMNGDBAb7zxhnr16qXYWH5MgOthwIABOnz4cLMIaq23335bU6dO1Te/+U1Jf72G59ixY9dch9/vV2xsrHr16hXWtSAyuEAZRoiJidGhQ4d06NAhxcTEhOzr0qWL0tLS9MILL+jIkSP6wx/+oNmzZ1/z8WbOnKkzZ87o29/+tt5991198sknKi0t1bRp09TY2NiWhwLcsH784x/r1Vdf1fz583Xw4EEdOnRIr7/+up5++ulWPW7fvn21bt067du3T++9954mTpx41TcnSNKIESOUnZ2thx56SJs3b9axY8e0c+dOPf300/YF0GhfiB0YIyUlRSkpKc3GO3TooJKSEpWXlyszM1M/+tGP9Pzzz1/zsbxer9555x01NjZq9OjRyszM1JNPPimXy6UOHfixAdrC6NGjtWHDBm3ZskV33XWX7r77bi1ZskQ9e/Zs1eMuXbpUXbp00eDBgzVu3DiNHj1aAwYMuOp8h8OhjRs36t5779W0adPUr18/PfbYYzp27Jh9jRDaF4dlWVakFwEAANBW+C8qAAAwGrEDAACMRuwAAACjETsAAMBoxA4AADAasQMAAIxG7AAAAKMROwAAwGjEDoAbXk5OjvLy8iK9DABthNgBEBX8fr+efPJJ9e3bVx07dpTb7dY999yjX/3qVzp//nyklwegHePPOQOIuE8++UT/8A//oM6dO6uoqEhZWVm6dOmSPvroI7388svyer0aP358pJd5VY2NjXI4HPzdNCBK8ZMJIOJyc3MVGxurPXv2aMKECbrllluUlZWlf/zHf9Tvf/97jRs3TpIUCAT0ve99T926dVNKSoqGDRum9957z36c+fPn6/bbb9eaNWvUq1cvuVwuPfbYY6qtrbXnnDt3TpMnT1ZSUpK6d++uxYsXN1tPQ0ODCgoK9JWvfEWJiYkaNGiQtm/fbu9ftWqVOnfurA0bNujWW2+V0+nU//3f/7XdNwhAqxA7ACLq9OnTKi0t1cyZM5WYmHjFOQ6HQ5Zl6YEHHpDf79fGjRtVXl6uAQMGaPjw4Tpz5ow99+jRo/rNb36jDRs2aMOGDSorK9NPf/pTe/9TTz2lt956S+vXr1dpaam2b9+u8vLykOf7zne+o3feeUclJSV6//339cgjj+i+++7Txx9/bM85f/68iouL9V//9V86ePCgunXrFubvDICwsQAggnbt2mVJstatWxcynpaWZiUmJlqJiYlWQUGBtW3bNislJcW6cOFCyLybb77ZWrlypWVZlvXss89anTp1smpqauz9Tz31lDVo0CDLsiyrtrbWio+Pt0pKSuz9p0+fthISEqwnn3zSsizLOnLkiOVwOKw///nPIc8zfPhwq7Cw0LIsy3rllVcsSda+ffvC800A0Ka4ZgdAVHA4HCHb7777rpqamjRp0iQFg0GVl5errq5OaWlpIfPq6+t19OhRe7tXr15KTk62t7t3766qqipJfz3r09DQoOzsbHt/amqqvvrVr9rbf/rTn2RZlvr16xfyPMFgMOS54+Pj1b9//1YcMYDrhdgBEFF9+/aVw+HQhx9+GDLep08fSVJCQoIkqampSd27dw+5duayzp0721/HxcWF7HM4HGpqapIkWZb1hetpampSTEyMysvLFRMTE7IvKSnJ/johIaFZoAGITsQOgIhKS0vTyJEjtXz5cs2aNeuq1+0MGDBAfr9fsbGx6tWrV4ueq2/fvoqLi9OuXbvUo0cPSVJ1dbU++ugjDR06VJJ0xx13qLGxUVVVVRoyZEiLngdAdOECZQAR98tf/lKXLl3SnXfeqddff12HDh3S4cOHtXbtWn344YeKiYnRiBEjlJ2drYceekibN2/WsWPHtHPnTj399NPas2fPl3qepKQkTZ8+XU899ZS2bdumAwcOaOrUqSFvGe/Xr58mTZqkyZMna926daqoqNDu3bu1cOFCbdy4sa2+BQDaEGd2AETczTffrL1796qoqEiFhYU6ceKEnE6nbr31Vs2ZM0e5ublyOBzauHGj5s2bp2nTpunUqVPyeDy699575Xa7v/RzPf/886qrq9P48eOVnJys/Px8BQKBkDmvvPKKnnvuOeXn5+vPf/6z0tLSlJ2drfvvvz/chw7gOnBYX+ZFbAAAgHaKl7EAAIDRiB0AAGA0YgcAABiN2AEAAEYjdgAAgNGIHQAAYDRiBwAAGI3YAQAARiN2AACA0YgdAABgNGIHAAAY7f8BB96RHqw6hCgAAAAASUVORK5CYII=",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "print(\"number of people who take loan as group by gender:\")\n",
    "print(df['Gender'].value_counts())\n",
    "sns.countplot(x='Gender',data=df,palette= 'Set2')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "id": "bd327d2b",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "number of people who take loan as group by gender:\n",
      "Male      502\n",
      "Female    112\n",
      "Name: Gender, dtype: int64\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "<Axes: xlabel='Gender', ylabel='count'>"
      ]
     },
     "execution_count": 29,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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v/71r61OnTpdcw233nqr9bOf/cy+37t3b2vZsmWWZVnW9u3braSkJOvChQshj7n55putVatW/d3Hi8gz4s9F4MaWkpKi++67T2vWrLH/R5aSkhIy5+jRo3rmmWe0e/duffbZZ/YZnU8//VTp6enN9lleXq4333xTXbp0abbt6NGjGjBgQNscDHCD+MY3vqGVK1fa9xMSEtS/f3+VlZWFnMlpbGzUhQsXdP78eXXu3FmSNHDgQHu72+2WJGVkZISMXbhwQTU1NUpKStK5c+e0aNEibdy4USdPntSlS5dUX19/1TM75eXlqqurU3Jycsh4fX39F779jehE7MAI06dPt09b//znP2+2fcKECfL5fHrppZfk9XrV1NSk9PR0NTQ0XHF/TU1NmjBhghYvXtxsW2pqangXD9yALsfN32pqatKiRYv04IMPNpvfqVMn++uOHTvaXzscjquOXf5PzVNPPaUtW7boP/7jP9S/f3/Fx8frW9/61jVf/6mpqdqxY0ezbV27dv1yB4ioQuzACPfcc4/9D1dOTk7IttOnT+vQoUNatWqVhg0bJknauXPnNfc3aNAgvfHGG+rTp49iY3mZANfDoEGDdPjw4WYR1Fpvv/22pk2bpm9+85uS/noNz7Fjx665Dr/fr9jYWPXp0yesa0FkcIEyjBATE6NDhw7p0KFDiomJCdnWrVs3JScn68UXX9SRI0f0hz/8QXPmzLnm/mbNmqUzZ87o29/+tt5991198sknKikp0fTp09XY2NiWhwLcsH70ox/pV7/6lRYuXKiDBw/q0KFDev311/X000+3ar/9+/fX+vXrtW/fPr333nuaPHnyVX85QZJGjx6trKwsPfDAA9qyZYuOHTumXbt26emnn7YvgEb7QuzAGElJSUpKSmo23qFDBxUXF6u8vFzp6en64Q9/qOeff/6a+/J6vXrnnXfU2NionJwcpaen68knn5TL5VKHDrxsgLaQk5OjjRs3auvWrRo8eLDuuusuLV26VL17927VfpctW6Zu3bpp6NChmjBhgnJycjRo0KCrznc4HNq0aZOGDx+u6dOna8CAAXrkkUd07Ngx+xohtC8Oy7KsSC8CAACgrfBfVAAAYDRiBwAAGI3YAQAARiN2AACA0YgdAABgNGIHAAAYjdgBAABGI3YAAIDRiB0AN7yRI0cqLy8v0ssA0EaIHQBRwe/368knn1T//v3VqVMnud1u3X333frlL3+p8+fPR3p5ANox/pwzgIj75JNP9A//8A/q2rWrCgsLlZGRoUuXLumjjz7SK6+8Iq/Xq4kTJ0Z6mVfV2Ngoh8PB300DohSvTAARl5ubq9jYWO3Zs0eTJk3SLbfcooyMDP3jP/6jfv/732vChAmSpEAgoO9973vq2bOnkpKSNGrUKL333nv2fhYuXKjbb79da9euVZ8+feRyufTII4+otrbWnnPu3Dk99thj6tKli1JTU/XCCy80W09DQ4Py8/P1la98RQkJCRoyZIh27Nhhb1+9erW6du2qjRs36tZbb5XT6dT//d//td03CECrEDsAIur06dMqKSnRrFmzlJCQcMU5DodDlmXpvvvuk9/v16ZNm1ReXq5BgwYpOztbZ86csecePXpUv/nNb7Rx40Zt3LhRpaWl+slPfmJvf+qpp/Tmm29qw4YNKikp0Y4dO1ReXh7yfN/5znf0zjvvqLi4WO+//74eeugh3XPPPfr444/tOefPn1dRUZH+67/+SwcPHlTPnj3D/J0BEDYWAETQ7t27LUnW+vXrQ8aTk5OthIQEKyEhwcrPz7e2b99uJSUlWRcuXAiZd/PNN1urVq2yLMuynn32Watz585WTU2Nvf2pp56yhgwZYlmWZdXW1lpxcXFWcXGxvf306dNWfHy89eSTT1qWZVlHjhyxHA6H9ec//znkebKzs62CggLLsizr1VdftSRZ+/btC883AUCb4podAFHB4XCE3H/33XfV1NSkKVOmKBgMqry8XHV1dUpOTg6ZV19fr6NHj9r3+/Tpo8TERPt+amqqqqqqJP31rE9DQ4OysrLs7d27d9dXv/pV+/6f/vQnWZalAQMGhDxPMBgMee64uDgNHDiwFUcM4HohdgBEVP/+/eVwOPThhx+GjPfr10+SFB8fL0lqampSampqyLUzl3Xt2tX+umPHjiHbHA6HmpqaJEmWZX3hepqamhQTE6Py8nLFxMSEbOvSpYv9dXx8fLNAAxCdiB0AEZWcnKwxY8ZoxYoVmj179lWv2xk0aJD8fr9iY2PVp0+fFj1X//791bFjR+3evVu9evWSJFVXV+ujjz7SiBEjJEl33HGHGhsbVVVVpWHDhrXoeQBEFy5QBhBxv/jFL3Tp0iXdeeedev3113Xo0CEdPnxY69at04cffqiYmBiNHj1aWVlZeuCBB7RlyxYdO3ZMu3bt0tNPP609e/Z8qefp0qWLZsyYoaeeekrbt2/XgQMHNG3atJBfGR8wYICmTJmixx57TOvXr1dFRYXKysq0ePFibdq0qa2+BQDaEGd2AETczTffrL1796qwsFAFBQU6ceKEnE6nbr31Vs2bN0+5ublyOBzatGmTFixYoOnTp+vUqVPyeDwaPny43G73l36u559/XnV1dZo4caISExM1d+5cBQKBkDmvvvqqnnvuOc2dO1d//vOflZycrKysLN17773hPnQA14HD+jJvYgMAALRTvI0FAACMRuwAAACjETsAAMBoxA4AADAasQMAAIxG7AAAAKMROwAAwGjEDgAAMBqxAwAAjEbsAAAAoxE7AADAaP8Pn0ueuR+1QNoAAAAASUVORK5CYII=",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "print(\"number of people who take loan as group by gender:\")\n",
    "print(df['Gender'].value_counts())\n",
    "sns.countplot(x='Gender',data=df,palette= 'Set3')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "id": "0ff7d2fd",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "number of people who take loan as group by marital status:\n",
      "Yes    401\n",
      "No     213\n",
      "Name: Married, dtype: int64\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "<Axes: xlabel='Married', ylabel='count'>"
      ]
     },
     "execution_count": 30,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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+Hn44YfPeGEAAACtoUWxs3z5cv3mN79RSkqKvF6vHA6Hvc/hcBA7AAAgZrQodh5//HE98cQTmjNnTmuvBwAAoFW16Pfs1NbW6tZbb23ttQAAALS6FsXOrbfeqi1btrT2WgAAAFpdiz7GuuSSS/TQQw9px44dSk9PV/v27SP2//a3v22VxQEAAJypFsXOc889py5duqikpEQlJSUR+xwOB7EDAABiRotip7KysrXXAQAA0CZa9MwOAADAuaJFd3YmT578o/uff/75Fi3GdOWZv4z2EoCYk1n+QbSXAMBwLYqd2traiNfHjh3T7t27dfjw4ZP+A6EAAADR0qLYWb9+fbOx48ePa+rUqbr44ovPeFEAAACtpdWe2WnXrp1+97vfadGiRa11SgAAgDPWqg8of/bZZ/r+++9b85QAAABnpEUfY82YMSPitWVZqq6u1saNGzVx4sRWWRgAAEBraFHsfPjhhxGv27Vrp27dumnBggU/+U0tAACAs6lFsfPOO++09joAAADaxBk9s/PVV19p+/bteu+99/TVV1+d9vGFhYW66qqrlJiYqO7du2v06NH65JNPIuZYlqX8/Hz5fD517NhRgwcP1p49eyLmhMNhTZ8+XSkpKercubNGjRqlgwcPnsmlAQAAQ7Qodo4eParJkycrNTVVN9xwg66//nr5fD5NmTJF33777c8+T0lJiaZNm6YdO3aouLhY33//vXJycnT06FF7zlNPPaWFCxdq6dKlKisrk9fr1dChQ3XkyBF7Tl5entavX6+ioiJt375d9fX1GjFihJqamlpyeQAAwCAtip0ZM2aopKREf/nLX3T48GEdPnxYr7/+ukpKSjRz5syffZ7Nmzdr0qRJ6tu3r/r376+VK1fqwIEDqqiokPSPuzqLFy/Wgw8+qLFjxyoQCGj16tX69ttvtXbtWklSKBTSihUrtGDBAmVnZ+uKK67QmjVrtGvXLm3duvWk7xsOh1VXVxexAQAAM7Uodl599VWtWLFCubm5crlccrlcuummm7R8+XK98sorLV5MKBSSJCUlJUn6xz84GgwGlZOTY89xOp0aNGiQSktLJUkVFRU6duxYxByfz6dAIGDPOVFhYaHcbre9+f3+Fq8ZAADEthbFzrfffiuPx9NsvHv37qf1MdY/syxLM2bM0HXXXadAICBJCgaDktTsvTwej70vGAwqISFB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lZWVyev1aujQoTpy5EgUVwwAAGJJzMdOfHy8vF6vvXXr1k3SP+7qLF68WA8++KDGjh2rQCCg1atX69tvv9XatWujvGoAABArYj529u3bJ5/Pp169eum2227T559/LkmqrKxUMBhUTk6OPdfpdGrQoEEqLS390XOGw2HV1dVFbAAAwEwxHTtXX321XnjhBb355ptavny5gsGgBg4cqEOHDikYDEqSPB5PxDEej8fedyqFhYVyu9325vf72+waAABAdMV07OTm5uqWW25Renq6srOztXHjRknS6tWr7TkOhyPiGMuymo2daO7cuQqFQvZWVVXV+osHAAAxIaZj50SdO3dWenq69u3bZ38r68S7ODU1Nc3u9pzI6XTK5XJFbAAAwEznVOyEw2Ht3btXqamp6tWrl7xer4qLi+39jY2NKikp0cCBA6O4SgAAEEvio72AHzNr1iyNHDlSF154oWpqavT444+rrq5OEydOlMPhUF5engoKCpSWlqa0tDQVFBSoU6dOmjBhQrSXDgAAYkRMx87Bgwd1++236+uvv1a3bt10zTXXaMeOHerZs6ckafbs2WpoaNDUqVNVW1urq6++Wlu2bFFiYmKUVw4AAGJFTMdOUVHRj+53OBzKz89Xfn7+2VkQAAA455xTz+wAAACcLmIHAAAYjdgBAABGI3YAAIDRiB0AAGA0YgcAABiN2AEAAEYjdgAAgNGIHQAAYDRiBwAAGI3YAQAARiN2AACA0YgdAABgNGIHAAAYjdgBAABGI3YAAIDRiB0AAGA0YgcAABiN2AEAAEYjdgAAgNGIHQAAYDRiBwAAGI3YAQAARiN2AACA0YgdAABgNGIHAAAYjdgBAABGI3YAAIDRiB0AAGA0YgcAABiN2AEAAEYjdgAAgNGIHQAAYDRjYueZZ55Rr1691KFDB2VkZOivf/1rtJcEAABigBGxs27dOuXl5enBBx/Uhx9+qOuvv165ubk6cOBAtJcGAACizIjYWbhwoaZMmaK77rpLl112mRYvXiy/369ly5ZFe2kAACDK4qO9gDPV2NioiooK/f73v48Yz8nJUWlp6UmPCYfDCofD9utQKCRJqqura7uFSqpvamrT8wPnorb+e3e2fB/+NtpLAGJOW//9/uH8lmX96LxzPna+/vprNTU1yePxRIx7PB4Fg8GTHlNYWKhHH3202bjf72+TNQL4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",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "print(\"number of people who take loan as group by marital status:\")\n",
    "print(df['Married'].value_counts())\n",
    "sns.countplot(x='Married',data=df,palette= 'Set1')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 31,
   "id": "02133cdb",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "number of people who take loan as group by dependents:\n",
      "0     360\n",
      "1     102\n",
      "2     101\n",
      "3+     51\n",
      "Name: Dependents, dtype: int64\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "<Axes: xlabel='Dependents', ylabel='count'>"
      ]
     },
     "execution_count": 31,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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vZ566inde++9+stf/qLBgwf/6FwAOobYAdAlNmzYoG3btulXv/qVUlNTVV9fr+joaJ177rknfM7evXu1f/9++Xw+SdKf//xn9erVS8OGDVNycrLOPvtsffbZZ7r55ps7PNdFF12kTZs2hez7/uNLL71UO3fu1JAhQzr8PjExMZK+uSL0XS6XS1deeaWuvPJK3XfffUpLS1NZWZkKCws7/F4AfhixA+C0BYNB1dfXq62tTZ9//rnWrVun4uJiTZw4Uf/8z/+sXr16KTs7W1OmTNGjjz6q888/X/v379eaNWs0ZcoUZWVlSfrmqkl+fr5++9vfKhAI6K677tL06dPl9XolffOJrbvuukvx8fHKy8tTMBhUdXW1GhsbTzoW7rrrLo0aNUqPPfaYpkyZovXr14f8CEuS7rvvPk2cOFF+v1833nijevXqpa1bt2rbtm166KGHTup90tLS5HK59Kc//Uk/+clPFBsbqx07dujtt99Wbm6ukpKS9Je//EVffPGFLrzwwlP41wZwysJ90xCA7i0/P99IMpJMdHS0GTRokBk3bpx5/vnnTVtbm7MuEAiYuXPnGp/PZ3r37m38fr+5+eabzd69e40x39ygPGLECPP0008bn89n+vTpY6ZNm2a+/PLLkPd7+eWXzSWXXGJiYmLMgAEDzDXXXGNeffVVY8z/v0H5ww8/dNY3NjYaSeadd95x9j333HMmNTXVxMbGmkmTJpnf/va3ITcoG/PNJ7JGjRplYmNjTXx8vLn88svNM8884xyXZMrKykKe4/F4zIoVK5zHDz74oPF6vcblcpn8/Hzz0UcfmQkTJphBgwYZt9tthg0bZp566qkO/KsDOBUuY07iB9gA0MWKior02muvacuWLeEeBYBl+KWCAADAasQOAACwGj/GAgAAVuPKDgAAsBqxAwAArEbsAAAAqxE7AADAasQOAACwGrEDAACsRuwAAACrETsAAMBq/w86wAyYTydD3QAAAABJRU5ErkJggg==",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "print(\"number of people who take loan as group by dependents:\")\n",
    "print(df['Dependents'].value_counts())\n",
    "sns.countplot(x='Dependents',data=df,palette= 'Set1')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 32,
   "id": "db52bc74",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "number of people who take loan as group by self employed:\n",
      "No     532\n",
      "Yes     82\n",
      "Name: Self_Employed, dtype: int64\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "<Axes: xlabel='Self_Employed', ylabel='count'>"
      ]
     },
     "execution_count": 32,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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GjRokD744AONGTNGjzzyiB599FE9/PDDev/993XzzTfr0Ucf1T/+7uNvvvlGzz33nNavX69du3bJ7/froYceuuB7rFy5UkuXLtW//du/6aOPPtLIkSM1btw4HT58WJI0bdo0bdy4Mejv6ne/+528Xq/uueceSdKPfvQj7dq1S5s2bdJHH32kBx54QD/4wQ/sfezZs0dTp05VRkaG9u3bp3vuuUeLFi1qtL8n4LpkAcBFvPLKK1a7du2s1q1bWwMHDrSysrKsDz/80LIsy3rzzTet6Oho6/Tp00Gvuemmm6w1a9ZYlmVZCxYssO644w572+bNm63L+a9nwYIFVosWLay2bdsGPdLT0+05Xbt2tR5++GH7eVlZmSXJeuqpp+yxd99915JklZWVWZZlWevWrbMkWbt377bnHDp0yJJk7dmz57xr93q91nPPPRe0vrvuusvKyMiwLMuyTp8+bcXExFi///3v7e29e/e2srOzLcuyrE8++cRyOBzW3/72t6B9pKSkWFlZWZZlWdbEiROtH/zgB0HbH3zwQcvlcl3aXxiAerjoDeCifvjDH2rMmDF655139O6772rbtm1asmSJfvvb3+rYsWOqqqpSbGxs0Guqq6vte3Mawy233KLXX389aCwqKiroea9evew/u91uSVJiYmK9sfLycnk8Hkl/vzyWlJRkz7n11lt1ww036NChQ+rXr1/Q/v1+v44ePapBgwYFjQ8aNMi+bOd0OvXwww9r7dq1SktL0759+/Thhx/anzh7//33ZVmWevToEbSPQCBg/x0eOnRIEyZMCNqenJwcdLkMwOUhdgB8p7P3swwfPlxPP/20pk2bpgULFigjI0OdOnWqd8+JpEb9qHSrVq108803X3ROy5Yt7T+fvRfofGNnzpwJet357hu62L1E526zLCtobNq0aerdu7dKS0u1du1apaSkqGvXrvZ7h4WFqbi4WGFhYUH7iYyMtPcHoHEROwAuW8+ePbVlyxb16dNHPp9P4eHh6tatW6iXddm+/fZbFRUV2WdxPv74Y504ceK8H2uPjo6W1+vVzp07NXjwYHu8sLAw6CxQYmKikpKS9OKLL2rjxo1atWqVve3OO+9UXV2dysvLdffdd593TT179tTu3buDxs59DuDyEDsALuj48eN64IEHNHXqVPXq1UtRUVEqKirSkiVLNH78eKWmpio5OVn33XefFi9erFtuuUVHjx7V1q1bdd999wVdIroS3377rXw+X9CYw+GwL001VMuWLTVz5kz98pe/VMuWLTVjxgwNGDCg3iWss/7lX/5FCxYs0E033aTevXtr3bp12rdvn373u98FzZs2bZpmzJhhf5rqrB49emjy5Ml69NFHtXTpUt1555366quv9NZbbykxMVGjR4/WE088oYEDB2rJkiW67777tH37di5hAVeI2AFwQZGRkerfv7+WL1+uTz/9VLW1tYqLi9OPf/xjPfnkk3I4HNq6davmz5+vqVOn6tixY/J4PBo8ePAVh8g/OnjwoDp16hQ05nQ6dfr06Svab5s2bTRv3jxNmjRJpaWl+v73v6+1a9decP4TTzwhv9+v2bNnq7y8XD179tTrr7+u+Pj4oHkTJ05UZmamJk2apNatWwdtW7dunRYtWqTZs2frb3/7m2JjY5WcnKzRo0dLkgYMGKDf/va3WrBggbKzs5Wamqqf//znevbZZ6/oWIHrmcPiAjGA61BeXp4yMzN14sSJRt/3kSNH1K1bN+3du1d9+vRp9P0DuDyc2QGARlJbW6uysjL967/+qwYMGEDoAE0EXyoIIKTOfnHh+R7vvPNOqJd3WXbt2qWuXbuquLhYv/71r0O9HAD/H5exAITUJ598csFt3/ve9xQREXENVwPARMQOAAAwGpexAACA0YgdAABgNGIHAAAYjdgBAABGI3YAAIDRiB0AAGA0YgcAABiN2AEAAEb7f7kTY2LG4/w7AAAAAElFTkSuQmCC",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "print(\"number of people who take loan as group by self employed:\")\n",
    "print(df['Self_Employed'].value_counts())\n",
    "sns.countplot(x='Self_Employed',data=df,palette= 'Set1')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 33,
   "id": "d394c4a2",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "number of people who take loan as group by Loanamount:\n",
      "146.412162    22\n",
      "120.000000    20\n",
      "110.000000    17\n",
      "100.000000    15\n",
      "160.000000    12\n",
      "              ..\n",
      "240.000000     1\n",
      "214.000000     1\n",
      "59.000000      1\n",
      "166.000000     1\n",
      "253.000000     1\n",
      "Name: LoanAmount, Length: 204, dtype: int64\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "<Axes: xlabel='LoanAmount', ylabel='count'>"
      ]
     },
     "execution_count": 33,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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l12WdvoDDzyQUTGSJEm5klEo2rRpU9Ljuro6/vSnP7F58+aUXxQrSZLU3mUUip588slG4+rr67nmmms4/PDD97soSZKkbGuze4ry8vL4/ve/z3333ddWTUqSJGVNm95o/dFHH7Fnz562bFKSJCkrMrp8Nn369KTHQRBQXV3NM888wyWXXNImhUmSJGVTRqHonXfeSXqcl5dH3759ueeee5r9yzRJkqT2KKNQ9NJLfm+OJEnqWDIKRaEvvviCDz/8kEgkwtChQ+nbt29b1SVJkpRVGd1ovW3bNi677DJKS0sZN24cY8eOpX///lx++eVs3769rWuUJEk64DIKRdOnT2fZsmX85je/YfPmzWzevJmnn36aZcuW8YMf/KCta5QkSTrgMrp89vjjj/Of//mfjB8/Pj7um9/8JgUFBZx33nnMnz+/reqTOoS5FQsAuOHRK3NcyT7LR54MwMjlb+a4EklqPzI6U7R9+3aKi4sbje/Xr5+XzyRJ0kEpo1A0atQoZs2axc6dO+PjduzYwT/8wz8watSoNitOkiQpWzK6fDZnzhzKy8s59NBDGTFiBJFIhBUrVhCLxXj++efbukZJkqQDLqNQNHz4cFatWsW//du/8d///d8EQcCUKVOYOnUqBQUFbV2jJEnSAZdRKJo9ezbFxcV897vfTRr/wAMP8MUXX3DLLbe0SXGSJEnZktE9RQsWLOCoo45qNP7YY4/lX//1X/e7KEmSpGzLKBStW7eO0tLSRuP79u1LdXX1fhclSZKUbRmForKyMl555ZVG41955RX69++/30VJkiRlW0b3FF1xxRXceOON1NXVMXHiRABeeOEFbr75Zv+jtSRJOihlFIpuvvlmNm7cyDXXXMPu3bsB6NKlC7fccgszZsxo0wIlSZKyIaNQFIlE+Kd/+idmzpzJypUrKSgoYMiQIcRisbauT5IkKSsyuqco1K1bN0466SSGDRuWUSB6+eWXmTx5Mv379ycSifDUU08lTQ+CgMrKSvr3709BQQHjx4/n/fff35+SJUmSUtqvULS/tm3bxogRI5g3b17K6XfffTf33nsv8+bN46233qKkpIQzzjiDLVu2ZLlSSZLU0WV0+aytlJeXU15ennJaEATMmTOH2267jXPOOQeAhx9+mOLiYh599FGuvLJ9fNu4JEnqGHJ6piid1atXs27dOs4888z4uFgs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/MshDxYPPfRQ2m+5l9T+GIoktVu///3v2blzJ+eeey4PPfRQrsuR1MEZiiS1yrJlyzj55JOJxWKUlpZy6623smfPnvj05557jjFjxtCzZ08OOeQQ/uZv/oaPPvooPv3jjz8mEonwxBNPMGHCBAoLCxkxYgSvvfZao3UtXLiQiooKLrroIh544IFGX3Q8aNAg7rjjDi6++GK6devGwIEDefrpp/niiy8466yz6NatG8OHD2f58uVJyz3++OMce+yxxGIxBg0axD333JM0PRKJ8NRTTyWN69mzZzyYNfccli5dyt/93d9RU1MT/wLqysrK1m5qSVlmKJLUYmvXruWb3/wmJ510En/84x+ZP38+Cxcu5I477ojPs23bNqZPn85bb73FCy+8QF5eHmeffTb19fVJbd12223cdNNNrFixgqFDh3LBBRckhastW7bwH//xH1x44YWcccYZbNu2jaVLlzaq6b777uO0007jnXfe4Vvf+hYXXXQRF198MRdeeCF/+MMfOOKII7j44ovjgertt9/mvPPOY8qUKbz33ntUVlYyc+bMjM5ENfUcRo8ezZw5c+jevTvV1dVUV1dz0003tbp9SVkWSFIDl1xySXDWWWc1Gv/DH/4wOPLII4P6+vr4uJ/97GdBt27dgr1796Zsa/369QEQvPfee0EQBMHq1asDIPjFL34Rn+f9998PgGDlypXxcVVVVcFxxx0Xf3zDDTcEU6dOTWp74MCBwYUXXhh/XF1dHQDBzJkz4+Nee+21AAiqq6uDIAiCioqK4Iwzzkhq5+///u+DY445Jv4YCJ588smkeXr06BE8+OCDLX4ODz74YNCjR4+U20RS++SZIkkttnLlSkaNGkUkEomPO+2009i6dSuffvopAB999BEVFRUcfvjhdO/encGDBwPwySefJLX113/91/HfS0tLAVi/fn183MKFC7nwwgvjjy+88EKeeOIJNm/e3GQ7xcXFAAwfPrzRuLDtlStXctpppyW1cdppp7Fq1Sr27t3bks3Q4ucg6eBiKJLUYkEQJAWicBwQHz958mQ2bNjA/fffzxtvvMEbb7wBwO7du5OW69SpU/z3cNnwEtsHH3zAG2+8wc0330x+fj75+fmceuqp7Nixg1/96lfNtpOu7XTPIXGZhuPq6uoabY9065F08MnPdQGSDh7HHHMMjz/+eFKwePXVVykqKmLAgAFs2LCBlStXsmDBAsaOHQvs+wuy1lq4cCHjxo3jZz/7WdL4RYsWsXDhQq6++ur9eg4Na3r11VcZOnQo0WgUgL59+1JdXR2fvmrVKrZv396q9XTu3LnVZ54k5ZahSFJKNTU1rFixImnctGnTmDNnDt/73ve47rrr+PDDD5k1axbTp08nLy+PXr16ccghh1BVVUVpaSmffPIJt956a6vWW1dXx6JFi7j99tsZNmxY0rQrrriCu+++mz/+8Y+MGDEio+f1gx/8gJNOOomf/OQnnH/++bz22mvMmzePn//85/F5Jk6cyLx58zj11FOpr6/nlltuSTor1BKDBg1i69atvPDCC4wYMYLCwkIKCwszqllSdnj5TFJKS5cu5fjjj08aZs2axbPPPsubb77JiBEjuOqqq7j88sv50Y9+BEBeXh6PPfYYb7/9NsOGDeP73/8+P/3pT1u13l//+tds2LCBs88+u9G0IUOGMHz4cBYuXJjx8zrhhBP493//dx577DGGDRvGj3/8Y26//XYuvfTS+Dz33HMPZWVljBs3joqKCm666aZWB5rRo0dz1VVXcf7559O3b1/uvvvujGuWlB2RoOGFc0mSpK8hzxRJkiRhKJIkSQIMRZIkSYChSJIkCTAUSZIkAYYiSZIkwFAkSZIEGIokSZIAQ5EkSRJgKJIkSQIMRZIkSQD8f1badhFzZkcGAAAAAElFTkSuQmCC",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "print(\"number of people who take loan as group by Loanamount:\")\n",
    "print(df['LoanAmount'].value_counts())\n",
    "sns.countplot(x='LoanAmount',data=df,palette= 'Set1')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 34,
   "id": "82c72bf5",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "number of people who take loan as group by Credit history:\n",
      "1.0    525\n",
      "0.0     89\n",
      "Name: Credit_History, dtype: int64\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "<Axes: xlabel='Credit_History', ylabel='count'>"
      ]
     },
     "execution_count": 34,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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HVVXV8vlcmnIkCHatm2bxo4dK0lavHixGhsblZ2drdraWg0bNkzFxcWKj4+3j7F69WpFRkZqypQpamxs1JgxY1RQUKCIiIjOnhoAADBIp75n5+mnn9b3vvc99enTRx6PJ+hmYIfDobfffrtLF9nd+J4dAJeK79kB2guX79np1JWdRx99VD/5yU/sm4gBAADCVae+Z6e2tlbf+MY3unotAAAAXa5TsfONb3xDxcXFXb0WAACALtepj7FuvPFGPfzww9q9e7cGDx6sqKiooP0PPvhglywOAADgUnUqdp566in16tVLpaWlQd92LH12gzKxAwAAwkWnYqeqqqqr1wEAANAtOnXPDgAAwJWiU1d2zvcnIdavX9+pxQAAAHS1TsVObW1t0OOWlhbt379fx48f7/APhAIAAIRKp2KnqKio3djp06eVnZ2tgQMHXvKiAAAAukqX3bPTo0cPPfTQQ1q9enVXHRIAAOCSdekNyh988IFOnTrVlYcEAAC4JJ36GCs3NzfosWVZqq6u1h/+8AfNmDGjSxYGAADQFToVO3v37g163KNHD/Xt21c///nPz/ubWgAAAJdTp2Ln9ddf7+p1AAAAdItOxU6bjz/+WO+9954cDoduuukm9e3bt6vWBQAA0CU6dYPyiRMnNGvWLPXr10933XWXRowYIa/Xq9mzZ+vkyZNdvUYAAIBO61Ts5ObmqrS0VL///e91/PhxHT9+XC+++KJKS0u1cOHCrl4jAABAp3XqY6wtW7bof/7nfzRq1Ch77J577lFsbKymTJmidevWddX6AAAALkmnruycPHlSbre73fi1117Lx1gAACCsdCp2hg8frkceeUT//Oc/7bHGxkatWLFCw4cP77LFAQAAXKpOfYy1Zs0aZWVlqX///rrtttvkcDhUUVEhp9Op4uLirl4jAABAp3UqdgYPHqyDBw9q06ZN+tvf/ibLsvTNb35T9913n2JjY7t6jQAAAJ3WqdjJz8+X2+3W/fffHzS+fv16ffzxx1qyZEmXLA4AAOBSdeqenSeffFL/8i//0m781ltv1RNPPHHJiwIAAOgqnYodv9+vfv36tRvv27evqqurL3lRAAAAXaVTsePz+fTGG2+0G3/jjTfk9XoveVEAAABdpVP37MyZM0c5OTlqaWnR3XffLUl67bXXtHjxYr5BGQAAhJVOxc7ixYv16aefKjs7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",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "print(\"number of people who take loan as group by Credit history:\")\n",
    "print(df['Credit_History'].value_counts())\n",
    "sns.countplot(x='Credit_History',data=df,palette= 'Set1')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 35,
   "id": "91970f70",
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.model_selection import train_test_split\n",
    "x_train, x_test, y_train, y_test = train_test_split(x, y, test_size =0.2, random_state=0)\n",
    "\n",
    "from sklearn.preprocessing import LabelEncoder\n",
    "Labelencoder_x =LabelEncoder()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 36,
   "id": "a53da6e7",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[1, 1, 0, ..., 1.0, 4.875197323201151, 267],\n",
       "       [1, 0, 1, ..., 1.0, 5.278114659230517, 407],\n",
       "       [1, 1, 0, ..., 0.0, 5.003946305945459, 249],\n",
       "       ...,\n",
       "       [1, 1, 3, ..., 1.0, 5.298317366548036, 363],\n",
       "       [1, 1, 0, ..., 1.0, 5.075173815233827, 273],\n",
       "       [0, 1, 0, ..., 1.0, 5.204006687076795, 301]], dtype=object)"
      ]
     },
     "execution_count": 36,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "for i in range(0, 5):\n",
    "    x_train[:,i]=Labelencoder_x.fit_transform(x_train[:,i])\n",
    "    x_train[:,7]=Labelencoder_x.fit_transform(x_train[:,7])\n",
    "    \n",
    "x_train"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 37,
   "id": "f49fd333",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1,\n",
       "       0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1,\n",
       "       1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0,\n",
       "       1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1,\n",
       "       1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0,\n",
       "       1, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1,\n",
       "       0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1,\n",
       "       1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0,\n",
       "       0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1,\n",
       "       0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1,\n",
       "       0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1,\n",
       "       1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1,\n",
       "       1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1,\n",
       "       1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1,\n",
       "       1, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1,\n",
       "       1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1,\n",
       "       1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0,\n",
       "       1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1,\n",
       "       1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1,\n",
       "       1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0,\n",
       "       1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1,\n",
       "       1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1,\n",
       "       1, 1, 1, 0, 1, 0, 1])"
      ]
     },
     "execution_count": 37,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "Labelencoder_y = LabelEncoder()\n",
    "y_train = Labelencoder_y.fit_transform(y_train)\n",
    "\n",
    "y_train"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 38,
   "id": "054974c7",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[1, 0, 0, 0, 5, 1.0, 4.430816798843313, 85],\n",
       "       [0, 0, 0, 0, 5, 1.0, 4.718498871295094, 28],\n",
       "       [1, 1, 0, 0, 5, 1.0, 5.780743515792329, 104],\n",
       "       [1, 1, 0, 0, 5, 1.0, 4.700480365792417, 80],\n",
       "       [1, 1, 2, 0, 5, 1.0, 4.574710978503383, 22],\n",
       "       [1, 1, 0, 1, 3, 0.0, 5.10594547390058, 70],\n",
       "       [1, 1, 3, 0, 3, 1.0, 5.056245805348308, 77],\n",
       "       [1, 0, 0, 0, 5, 1.0, 6.003887067106539, 114],\n",
       "       [1, 0, 0, 0, 5, 0.0, 4.820281565605037, 53],\n",
       "       [1, 1, 0, 0, 5, 1.0, 4.852030263919617, 55],\n",
       "       [0, 0, 0, 0, 5, 1.0, 4.430816798843313, 4],\n",
       "       [1, 1, 1, 0, 5, 1.0, 4.553876891600541, 2],\n",
       "       [0, 0, 0, 0, 5, 1.0, 5.634789603169249, 96],\n",
       "       [1, 1, 2, 0, 5, 1.0, 5.4638318050256105, 97],\n",
       "       [1, 1, 0, 0, 5, 1.0, 4.564348191467836, 117],\n",
       "       [1, 1, 1, 0, 5, 1.0, 4.204692619390966, 22],\n",
       "       [1, 0, 1, 1, 5, 1.0, 5.247024072160486, 32],\n",
       "       [1, 0, 0, 1, 5, 1.0, 4.882801922586371, 25],\n",
       "       [0, 0, 0, 0, 5, 1.0, 4.532599493153256, 1],\n",
       "       [1, 1, 0, 1, 5, 0.0, 5.198497031265826, 44],\n",
       "       [0, 1, 0, 0, 5, 0.0, 4.787491742782046, 71],\n",
       "       [1, 1, 0, 0, 5, 1.0, 4.962844630259907, 43],\n",
       "       [1, 1, 2, 0, 5, 1.0, 4.68213122712422, 91],\n",
       "       [1, 1, 2, 0, 5, 1.0, 5.10594547390058, 111],\n",
       "       [1, 1, 0, 0, 5, 1.0, 4.060443010546419, 35],\n",
       "       [1, 1, 1, 0, 5, 1.0, 5.521460917862246, 94],\n",
       "       [1, 0, 0, 0, 5, 1.0, 5.231108616854587, 98],\n",
       "       [1, 1, 0, 0, 5, 1.0, 5.231108616854587, 110],\n",
       "       [1, 1, 3, 0, 5, 0.0, 4.852030263919617, 41],\n",
       "       [0, 0, 0, 0, 5, 0.0, 4.634728988229636, 50],\n",
       "       [1, 1, 0, 0, 5, 1.0, 5.429345628954441, 99],\n",
       "       [1, 0, 0, 1, 5, 1.0, 3.871201010907891, 46],\n",
       "       [1, 1, 1, 1, 5, 1.0, 4.499809670330265, 52],\n",
       "       [1, 1, 0, 0, 5, 1.0, 5.19295685089021, 102],\n",
       "       [1, 1, 0, 0, 5, 1.0, 4.857444178729352, 95],\n",
       "       [0, 1, 0, 1, 5, 0.0, 5.181783550292085, 57],\n",
       "       [1, 1, 0, 0, 5, 1.0, 5.147494476813453, 65],\n",
       "       [1, 0, 0, 1, 5, 1.0, 4.836281906951478, 39],\n",
       "       [1, 1, 0, 0, 5, 1.0, 4.852030263919617, 75],\n",
       "       [1, 1, 2, 1, 5, 1.0, 4.68213122712422, 24],\n",
       "       [0, 0, 0, 0, 5, 1.0, 4.382026634673881, 9],\n",
       "       [1, 1, 3, 0, 5, 0.0, 4.812184355372417, 68],\n",
       "       [1, 1, 2, 0, 2, 1.0, 2.833213344056216, 0],\n",
       "       [1, 1, 1, 1, 5, 1.0, 5.062595033026967, 67],\n",
       "       [1, 0, 0, 0, 5, 1.0, 4.330733340286331, 21],\n",
       "       [1, 0, 0, 0, 5, 1.0, 5.231108616854587, 113],\n",
       "       [1, 1, 1, 0, 5, 1.0, 4.7535901911063645, 18],\n",
       "       [0, 0, 0, 0, 5, 1.0, 4.74493212836325, 37],\n",
       "       [1, 1, 1, 0, 5, 1.0, 4.852030263919617, 72],\n",
       "       [1, 0, 0, 0, 5, 1.0, 4.941642422609304, 78],\n",
       "       [1, 1, 3, 1, 5, 1.0, 4.30406509320417, 8],\n",
       "       [1, 1, 0, 0, 5, 1.0, 4.867534450455582, 84],\n",
       "       [1, 1, 0, 1, 5, 1.0, 4.672828834461906, 31],\n",
       "       [1, 0, 0, 0, 5, 1.0, 4.857444178729352, 61],\n",
       "       [1, 1, 0, 0, 5, 1.0, 4.718498871295094, 19],\n",
       "       [1, 1, 0, 0, 5, 1.0, 5.556828061699537, 107],\n",
       "       [1, 1, 0, 0, 5, 1.0, 4.553876891600541, 34],\n",
       "       [1, 0, 0, 1, 5, 1.0, 4.890349128221754, 74],\n",
       "       [1, 1, 2, 0, 5, 1.0, 5.123963979403259, 62],\n",
       "       [1, 0, 0, 0, 5, 1.0, 4.787491742782046, 27],\n",
       "       [0, 0, 0, 0, 5, 0.0, 4.919980925828125, 108],\n",
       "       [0, 0, 0, 0, 5, 1.0, 5.365976015021851, 103],\n",
       "       [1, 1, 0, 1, 5, 1.0, 4.74493212836325, 38],\n",
       "       [0, 0, 0, 0, 5, 0.0, 4.330733340286331, 13],\n",
       "       [1, 1, 2, 0, 5, 1.0, 4.890349128221754, 69],\n",
       "       [1, 1, 1, 0, 5, 1.0, 5.752572638825633, 112],\n",
       "       [1, 1, 0, 0, 5, 1.0, 5.075173815233827, 73],\n",
       "       [1, 0, 0, 0, 5, 1.0, 4.912654885736052, 47],\n",
       "       [1, 1, 0, 0, 5, 1.0, 5.204006687076795, 81],\n",
       "       [1, 0, 0, 1, 5, 1.0, 4.564348191467836, 60],\n",
       "       [1, 0, 0, 0, 5, 1.0, 4.204692619390966, 83],\n",
       "       [0, 1, 0, 0, 5, 1.0, 4.867534450455582, 5],\n",
       "       [1, 1, 2, 1, 5, 1.0, 5.056245805348308, 58],\n",
       "       [1, 1, 1, 1, 3, 1.0, 4.919980925828125, 79],\n",
       "       [0, 1, 0, 0, 5, 1.0, 4.969813299576001, 54],\n",
       "       [1, 1, 0, 1, 4, 1.0, 4.820281565605037, 56],\n",
       "       [1, 0, 0, 0, 5, 1.0, 4.499809670330265, 120],\n",
       "       [1, 0, 3, 0, 5, 1.0, 5.768320995793772, 118],\n",
       "       [1, 1, 2, 0, 5, 1.0, 4.718498871295094, 101],\n",
       "       [0, 0, 0, 0, 5, 0.0, 4.7535901911063645, 26],\n",
       "       [0, 0, 0, 0, 6, 1.0, 4.727387818712341, 33],\n",
       "       [1, 1, 1, 0, 5, 1.0, 6.214608098422191, 119],\n",
       "       [0, 0, 0, 0, 5, 1.0, 5.267858159063328, 89],\n",
       "       [1, 1, 2, 0, 5, 1.0, 5.231108616854587, 92],\n",
       "       [1, 0, 0, 0, 6, 1.0, 4.2626798770413155, 6],\n",
       "       [1, 1, 0, 0, 0, 1.0, 4.709530201312334, 90],\n",
       "       [1, 1, 0, 0, 5, 1.0, 4.700480365792417, 45],\n",
       "       [1, 1, 2, 0, 5, 1.0, 5.298317366548036, 109],\n",
       "       [1, 0, 1, 0, 3, 1.0, 4.727387818712341, 17],\n",
       "       [1, 1, 1, 0, 5, 1.0, 4.6443908991413725, 36],\n",
       "       [0, 1, 0, 1, 5, 1.0, 4.605170185988092, 16],\n",
       "       [1, 0, 0, 0, 5, 1.0, 4.30406509320417, 7],\n",
       "       [1, 1, 1, 0, 1, 1.0, 5.147494476813453, 88],\n",
       "       [1, 1, 3, 0, 4, 0.0, 5.19295685089021, 87],\n",
       "       [0, 0, 0, 0, 5, 1.0, 4.2626798770413155, 3],\n",
       "       [1, 0, 0, 1, 3, 0.0, 4.836281906951478, 59],\n",
       "       [1, 0, 0, 0, 3, 1.0, 5.1647859739235145, 82],\n",
       "       [1, 0, 0, 0, 5, 1.0, 4.969813299576001, 66],\n",
       "       [1, 1, 2, 1, 5, 1.0, 4.394449154672439, 51],\n",
       "       [1, 1, 1, 0, 5, 1.0, 5.231108616854587, 100],\n",
       "       [1, 1, 0, 0, 5, 1.0, 5.351858133476067, 93],\n",
       "       [1, 1, 0, 0, 5, 1.0, 4.605170185988092, 15],\n",
       "       [1, 1, 2, 0, 5, 1.0, 4.787491742782046, 106],\n",
       "       [1, 0, 0, 0, 3, 1.0, 4.787491742782046, 105],\n",
       "       [1, 1, 3, 0, 5, 1.0, 4.852030263919617, 64],\n",
       "       [1, 0, 0, 0, 5, 1.0, 4.8283137373023015, 49],\n",
       "       [1, 0, 0, 1, 5, 1.0, 4.6443908991413725, 42],\n",
       "       [0, 0, 0, 0, 5, 1.0, 4.477336814478207, 10],\n",
       "       [1, 1, 0, 1, 5, 1.0, 4.553876891600541, 20],\n",
       "       [1, 1, 3, 1, 3, 1.0, 4.394449154672439, 14],\n",
       "       [1, 0, 0, 0, 5, 1.0, 5.298317366548036, 76],\n",
       "       [0, 0, 0, 0, 5, 1.0, 4.90527477843843, 11],\n",
       "       [1, 0, 0, 0, 6, 1.0, 4.727387818712341, 18],\n",
       "       [1, 1, 2, 0, 5, 1.0, 4.248495242049359, 23],\n",
       "       [1, 1, 0, 1, 5, 0.0, 5.303304908059076, 63],\n",
       "       [1, 1, 0, 0, 3, 0.0, 4.499809670330265, 48],\n",
       "       [0, 0, 0, 0, 5, 1.0, 4.430816798843313, 30],\n",
       "       [1, 0, 0, 0, 5, 1.0, 4.897839799950911, 29],\n",
       "       [1, 1, 2, 0, 5, 1.0, 5.170483995038151, 86],\n",
       "       [1, 1, 3, 0, 5, 1.0, 4.867534450455582, 115],\n",
       "       [1, 1, 0, 0, 5, 1.0, 6.077642243349034, 116],\n",
       "       [1, 1, 3, 1, 3, 0.0, 4.248495242049359, 40],\n",
       "       [1, 1, 1, 0, 5, 1.0, 4.564348191467836, 12]], dtype=object)"
      ]
     },
     "execution_count": 38,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "for i in range(0, 5):\n",
    "    x_test[:,i] = Labelencoder_x.fit_transform(x_test[:, i])\n",
    "    x_test[:,7] = Labelencoder_x.fit_transform(x_test[:, 7])\n",
    "    \n",
    "x_test"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 39,
   "id": "cf93afae",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1,\n",
       "       1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1,\n",
       "       1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1,\n",
       "       1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 1,\n",
       "       1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0,\n",
       "       1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1])"
      ]
     },
     "execution_count": 39,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "labelencoder_y = LabelEncoder()\n",
    "\n",
    "y_test = Labelencoder_y.fit_transform(y_test)\n",
    "\n",
    "y_test"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 40,
   "id": "e387d0d4",
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.preprocessing import StandardScaler\n",
    "\n",
    "ss=StandardScaler()\n",
    "x_train =ss.fit_transform(x_train)\n",
    "x_test = ss.fit_transform(x_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 41,
   "id": "ab0f1c7d",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<style>#sk-container-id-1 {color: black;}#sk-container-id-1 pre{padding: 0;}#sk-container-id-1 div.sk-toggleable {background-color: white;}#sk-container-id-1 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-1 label.sk-toggleable__label-arrow:before {content: \"▸\";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-1 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-1 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-1 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-1 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-1 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-1 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: \"▾\";}#sk-container-id-1 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-1 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-1 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-1 div.sk-parallel-item::after {content: \"\";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-1 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-serial::before {content: \"\";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-1 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-1 div.sk-item {position: relative;z-index: 1;}#sk-container-id-1 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-1 div.sk-item::before, #sk-container-id-1 div.sk-parallel-item::before {content: \"\";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-1 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-1 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-1 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-1 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-1 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-1 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-1 div.sk-label-container {text-align: center;}#sk-container-id-1 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-1 div.sk-text-repr-fallback {display: none;}</style><div id=\"sk-container-id-1\" class=\"sk-top-container\"><div class=\"sk-text-repr-fallback\"><pre>RandomForestClassifier()</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class=\"sk-container\" hidden><div class=\"sk-item\"><div class=\"sk-estimator sk-toggleable\"><input class=\"sk-toggleable__control sk-hidden--visually\" id=\"sk-estimator-id-1\" type=\"checkbox\" checked><label for=\"sk-estimator-id-1\" class=\"sk-toggleable__label sk-toggleable__label-arrow\">RandomForestClassifier</label><div class=\"sk-toggleable__content\"><pre>RandomForestClassifier()</pre></div></div></div></div></div>"
      ],
      "text/plain": [
       "RandomForestClassifier()"
      ]
     },
     "execution_count": 41,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "from sklearn.ensemble import RandomForestClassifier\n",
    "\n",
    "rf_clf = RandomForestClassifier()\n",
    "rf_clf.fit(x_train, y_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 42,
   "id": "4d4fa152",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "acc of random forest clf is 0.7886178861788617\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "array([1, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1,\n",
       "       1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1,\n",
       "       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1,\n",
       "       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1,\n",
       "       1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,\n",
       "       1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1])"
      ]
     },
     "execution_count": 42,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "from sklearn import metrics\n",
    "y_pred =rf_clf.predict(x_test)\n",
    "\n",
    "print(\"acc of random forest clf is\", metrics.accuracy_score(y_pred, y_test))\n",
    "\n",
    "y_pred"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 43,
   "id": "91224ad5",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<style>#sk-container-id-2 {color: black;}#sk-container-id-2 pre{padding: 0;}#sk-container-id-2 div.sk-toggleable {background-color: white;}#sk-container-id-2 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-2 label.sk-toggleable__label-arrow:before {content: \"▸\";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-2 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-2 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-2 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-2 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-2 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-2 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: \"▾\";}#sk-container-id-2 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-2 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-2 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-2 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-2 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-2 div.sk-parallel-item::after {content: \"\";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-2 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-2 div.sk-serial::before {content: \"\";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-2 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-2 div.sk-item {position: relative;z-index: 1;}#sk-container-id-2 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-2 div.sk-item::before, #sk-container-id-2 div.sk-parallel-item::before {content: \"\";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-2 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-2 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-2 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-2 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-2 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-2 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-2 div.sk-label-container {text-align: center;}#sk-container-id-2 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-2 div.sk-text-repr-fallback {display: none;}</style><div id=\"sk-container-id-2\" class=\"sk-top-container\"><div class=\"sk-text-repr-fallback\"><pre>GaussianNB()</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class=\"sk-container\" hidden><div class=\"sk-item\"><div class=\"sk-estimator sk-toggleable\"><input class=\"sk-toggleable__control sk-hidden--visually\" id=\"sk-estimator-id-2\" type=\"checkbox\" checked><label for=\"sk-estimator-id-2\" class=\"sk-toggleable__label sk-toggleable__label-arrow\">GaussianNB</label><div class=\"sk-toggleable__content\"><pre>GaussianNB()</pre></div></div></div></div></div>"
      ],
      "text/plain": [
       "GaussianNB()"
      ]
     },
     "execution_count": 43,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "from sklearn.naive_bayes import GaussianNB\n",
    "nb_clf = GaussianNB()\n",
    "nb_clf.fit(x_train, y_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 44,
   "id": "9d810cf5",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "acc of gaussianNB is %. 0.8292682926829268\n"
     ]
    }
   ],
   "source": [
    "y_pred = nb_clf.predict(x_test)\n",
    "print(\"acc of gaussianNB is %.\", metrics.accuracy_score(y_pred, y_test))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 45,
   "id": "6011b5a1",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1,\n",
       "       1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1,\n",
       "       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1,\n",
       "       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1,\n",
       "       1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,\n",
       "       1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1])"
      ]
     },
     "execution_count": 45,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "y_pred"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 46,
   "id": "84acfe45",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<style>#sk-container-id-3 {color: black;}#sk-container-id-3 pre{padding: 0;}#sk-container-id-3 div.sk-toggleable {background-color: white;}#sk-container-id-3 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-3 label.sk-toggleable__label-arrow:before {content: \"▸\";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-3 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-3 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-3 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-3 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-3 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-3 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: \"▾\";}#sk-container-id-3 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-3 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-3 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-3 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-3 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-3 div.sk-parallel-item::after {content: \"\";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-3 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-3 div.sk-serial::before {content: \"\";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-3 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-3 div.sk-item {position: relative;z-index: 1;}#sk-container-id-3 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-3 div.sk-item::before, #sk-container-id-3 div.sk-parallel-item::before {content: \"\";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-3 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-3 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-3 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-3 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-3 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-3 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-3 div.sk-label-container {text-align: center;}#sk-container-id-3 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-3 div.sk-text-repr-fallback {display: none;}</style><div id=\"sk-container-id-3\" class=\"sk-top-container\"><div class=\"sk-text-repr-fallback\"><pre>DecisionTreeClassifier()</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class=\"sk-container\" hidden><div class=\"sk-item\"><div class=\"sk-estimator sk-toggleable\"><input class=\"sk-toggleable__control sk-hidden--visually\" id=\"sk-estimator-id-3\" type=\"checkbox\" checked><label for=\"sk-estimator-id-3\" class=\"sk-toggleable__label sk-toggleable__label-arrow\">DecisionTreeClassifier</label><div class=\"sk-toggleable__content\"><pre>DecisionTreeClassifier()</pre></div></div></div></div></div>"
      ],
      "text/plain": [
       "DecisionTreeClassifier()"
      ]
     },
     "execution_count": 46,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "from sklearn.tree import DecisionTreeClassifier\n",
    "dt_clf = DecisionTreeClassifier()\n",
    "dt_clf.fit(x_train, y_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 47,
   "id": "d37ed8bf",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "acc of DT is 0.7235772357723578\n"
     ]
    }
   ],
   "source": [
    "y_pred = dt_clf.predict(x_test)\n",
    "print(\"acc of DT is\", metrics.accuracy_score(y_pred, y_test))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 48,
   "id": "14e24146",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 1,\n",
       "       1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1,\n",
       "       1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1,\n",
       "       1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1,\n",
       "       1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0,\n",
       "       1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1])"
      ]
     },
     "execution_count": 48,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "y_pred"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 49,
   "id": "62eefc30",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<style>#sk-container-id-4 {color: black;}#sk-container-id-4 pre{padding: 0;}#sk-container-id-4 div.sk-toggleable {background-color: white;}#sk-container-id-4 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-4 label.sk-toggleable__label-arrow:before {content: \"▸\";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-4 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-4 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-4 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-4 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-4 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-4 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: \"▾\";}#sk-container-id-4 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-4 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-4 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-4 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-4 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-4 div.sk-parallel-item::after {content: \"\";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-4 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-4 div.sk-serial::before {content: \"\";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-4 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-4 div.sk-item {position: relative;z-index: 1;}#sk-container-id-4 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-4 div.sk-item::before, #sk-container-id-4 div.sk-parallel-item::before {content: \"\";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-4 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-4 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-4 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-4 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-4 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-4 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-4 div.sk-label-container {text-align: center;}#sk-container-id-4 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-4 div.sk-text-repr-fallback {display: none;}</style><div id=\"sk-container-id-4\" class=\"sk-top-container\"><div class=\"sk-text-repr-fallback\"><pre>KNeighborsClassifier()</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class=\"sk-container\" hidden><div class=\"sk-item\"><div class=\"sk-estimator sk-toggleable\"><input class=\"sk-toggleable__control sk-hidden--visually\" id=\"sk-estimator-id-4\" type=\"checkbox\" checked><label for=\"sk-estimator-id-4\" class=\"sk-toggleable__label sk-toggleable__label-arrow\">KNeighborsClassifier</label><div class=\"sk-toggleable__content\"><pre>KNeighborsClassifier()</pre></div></div></div></div></div>"
      ],
      "text/plain": [
       "KNeighborsClassifier()"
      ]
     },
     "execution_count": 49,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "from sklearn.neighbors import KNeighborsClassifier\n",
    "kn_clf = KNeighborsClassifier()\n",
    "kn_clf.fit(x_train, y_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 50,
   "id": "af076908",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "acc of KN is 0.7967479674796748\n"
     ]
    }
   ],
   "source": [
    "y_pred = kn_clf.predict(x_test)\n",
    "print(\"acc of KN is\",metrics.accuracy_score(y_pred,y_test))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "c50c2049",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "c91a82be",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "cf502baa",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "0d916b0b",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "1a3362a6",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.4"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
