//
// Created by tanmay on 6/12/22.
//

#ifndef LLVM_ATOMICCONDITION_H
#define LLVM_ATOMICCONDITION_H

#include <math.h>
#include <stdio.h>
#include <string>
#include <vector>

enum Operation {
  Add,
  Sub,
  Mul,
  Div,
  Sin,
  Cos,
  Tan,
  ArcSin,
  ArcCos,
  ArcTan,
  Sinh,
  Cosh,
  Tanh,
  Exp,
  Log,
  Sqrt,
};

using namespace std;



// Atomic Condition storage
template<typename FloatType>
class ACItem {
  string XName;
  FloatType X;
  string YName;
  FloatType Y;
  Operation OP;
  int WRTOperand;
  FloatType AC;
public:
  ACItem() {}
  ACItem(string XName, FloatType X,
         string YName, FloatType Y,
         Operation OP, int WRTOperand) : XName(XName), X(X), YName(YName), Y(Y), OP(OP), WRTOperand(WRTOperand) {}
  ACItem(string XName, FloatType X,
         string YName, FloatType Y,
         Operation OP, int WRTOperand,
         FloatType AC) : XName(XName), X(X), YName(YName), Y(Y), OP(OP), WRTOperand(WRTOperand), AC(AC) {}
  bool operator==(const ACItem &RHS) const {
    return XName == RHS.XName && X == RHS.X && YName == RHS.YName &&
           Y == RHS.Y && OP == RHS.OP && WRTOperand == RHS.WRTOperand &&
           AC == RHS.AC;
  }
  bool operator!=(const ACItem &RHS) const { return !(RHS == *this); }

  void setAC(FloatType ACValue) {AC = ACValue;}
  bool isUnary() { return false; }
  bool isBinary() { return OP==14 || OP==16 || OP== 18 || OP == 21; }
  const string &getXName() const { return XName; }
  const string &getYName() const { return YName; }
  FloatType getX() const { return X; }
  FloatType getY() const { return Y; }
  int getOp() const { return OP; }
  int getWrtOperand() const { return WRTOperand; }
  FloatType getAc() const { return AC; }
};

class ACTable {
public:
  uint64_t NumItems;
  vector<ACItem<float>> FP32ACItems;
  vector<ACItem<double>> FP64ACItems;

  ACTable() {};
  ACTable(uint64_t NumItems): NumItems(NumItems) {
//    ACItems = new ACItem[NumItems];
//    for(uint64_t i=0; i<NumItems; i++)
//      ACItems[i] = NULL;
  }

//  ~ACTable() {
//    delete[] ACItems;
//  }
};

ACTable *StorageTable;

float fACfp32BinaryAdd(string XName, float X, string YName, float Y, int WRTOperand);
float fACfp32BinarySub(string XName, float X, string YName, float Y, int WRTOperand);
float fACfp32BinaryMul(string XName, float X, string YName, float Y, int WRTOperand);
float fACfp32BinaryDiv(string XName, float X, string YName, float Y, int WRTOperand);

float fACfp32UnarySin(string XName, float X);
float fACfp32UnaryCos(string XName, float X);
float fACfp32UnaryTan(string XName, float X);
float fACfp32UnaryArcSin(string XName, float X);
float fACfp32UnaryArcCos(string XName, float X);
float fACfp32UnaryArcTan(string XName, float X);
float fACfp32UnarySinh(string XName, float X);
float fACfp32UnaryCosh(string XName, float X);
float fACfp32UnaryTanh(string XName, float X);
float fACfp32UnaryExp(string XName, float X);
float fACfp32UnaryLog(string XName, float X);
float fACfp32UnarySqrt(string XName, float X);



double fACfp64BinaryAdd(string XName, double X, string YName, double Y, int WRTOperand);
double fACfp64BinarySub(string XName, double X, string YName, double Y, int WRTOperand);
double fACfp64BinaryMul(string XName, double X, string YName, double Y, int WRTOperand);
double fACfp64BinaryDiv(string XName, double X, string YName, double Y, int WRTOperand);

double fACfp64UnarySin(string XName, double X);
double fACfp64UnaryCos(string XName, double X);
double fACfp64UnaryTan(string XName, double X);
double fACfp64UnaryArcSin(string XName, double X);
double fACfp64UnaryArcCos(string XName, double X);
double fACfp64UnaryArcTan(string XName, double X);
double fACfp64UnarySinh(string XName, double X);
double fACfp64UnaryCosh(string XName, double X);
double fACfp64UnaryTanh(string XName, double X);
double fACfp64UnaryExp(string XName, double X);
double fACfp64UnaryLog(string XName, double X);
double fACfp64UnarySqrt(string XName, double X);



void fACCreate(uint64_t NumItems) {
  StorageTable = new ACTable(NumItems);
}

// Driver function selecting atomic condition function for unary float operation
float fACfp32UnaryDriver(string XName, float X, Operation OP) {
  switch (OP) {
  case 4:
    return fACfp32UnarySin(XName, X);
  case 5:
    return fACfp32UnaryCos(XName, X);
  case 6:
    return fACfp32UnaryTan(XName, X);
  case 7:
    return fACfp32UnaryArcSin(XName, X);
  case 8:
    return fACfp32UnaryArcCos(XName, X);
  case 9:
    return fACfp32UnaryArcTan(XName, X);
  case 10:
    return fACfp32UnarySinh(XName, X);
  case 11:
    return fACfp32UnaryCosh(XName, X);
  case 12:
    return fACfp32UnaryTanh(XName, X);
  case 13:
    return fACfp32UnaryExp(XName, X);
  case 14:
    return fACfp32UnaryLog(XName, X);
  case 15:
    return fACfp32UnarySqrt(XName, X);
  default:
    printf("No such operation\n");
  }
  return 0.0;
}

// Driver function selecting atomic condition function for binary float operation
float fACfp32BinaryDriver(string XName, float X, string YName, float Y, Operation OP, int WRTOperand) {
  switch (OP) {
  case 0:
    return fACfp32BinaryAdd(XName, X, YName, Y, WRTOperand);
  case 1:
    return fACfp32BinarySub(XName, X, YName, Y, WRTOperand);
  case 2:
    return fACfp32BinaryMul(XName, X, YName, Y, WRTOperand);
  case 3:
    return fACfp32BinaryDiv(XName, X, YName, Y, WRTOperand);
  default:
    printf("No such operation\n");
  }
  return 0.0;
}

// Driver function selecting atomic condition function for unary double operation
double fACfp64UnaryDriver(string XName, double X, Operation OP) {
  switch (OP) {
  case 4:
    return fACfp64UnarySin(XName, X);
  case 5:
    return fACfp64UnaryCos(XName, X);
  case 6:
    return fACfp64UnaryTan(XName, X);
  case 7:
    return fACfp64UnaryArcSin(XName, X);
  case 8:
    return fACfp64UnaryArcCos(XName, X);
  case 9:
    return fACfp64UnaryArcTan(XName, X);
  case 10:
    return fACfp64UnarySinh(XName, X);
  case 11:
    return fACfp64UnaryCosh(XName, X);
  case 12:
    return fACfp64UnaryTanh(XName, X);
  case 13:
    return fACfp64UnaryExp(XName, X);
  case 14:
    return fACfp64UnaryLog(XName, X);
  case 15:
    return fACfp64UnarySqrt(XName, X);
  default:
    printf("No such operation\n");
  }
  return 0.0;
}

// Driver function selecting atomic condition function for binary double operation
double fACfp64BinaryDriver(string XName, double X, string YName, double Y, Operation OP, int WRTOperand) {
  switch (OP) {
  case 0:
    return fACfp64BinaryAdd(XName, X, YName, Y, WRTOperand);
  case 1:
    return fACfp64BinarySub(XName, X, YName, Y, WRTOperand);
  case 2:
    return fACfp64BinaryMul(XName, X, YName, Y, WRTOperand);
  case 3:
    return fACfp64BinaryDiv(XName, X, YName, Y, WRTOperand);
  default:
    printf("No such operation\n");
  }
  return 0.0;
}

// ---------------------------------------------------------------------------
// --------------- fp32 Atomic Condition Calculating Functions ---------------
// ---------------------------------------------------------------------------

// ---------------------------- Binary Operations ----------------------------

float fACfp32BinaryAdd(string XName, float X, string YName, float Y, int WRTOperand) {
  float AC;
  if(WRTOperand == 1) {
    AC = abs(X / (X+Y));
  } else if(WRTOperand == 2) {
    AC = abs(Y / (X+Y));
  } else {
    printf("There is no operand %d.\n", WRTOperand);
    return AC;
  }
  printf("AC of x+y | x=%f, y=%f WRT %c is %f.\n",
         X, Y, (WRTOperand==1?'x':'y'), AC);
  
  StorageTable->FP32ACItems.push_back(ACItem<float>(XName, X, YName, Y, Operation::Add, WRTOperand, AC));

  return AC;
}

float fACfp32BinarySub(string XName, float X, string YName, float Y, int WRTOperand) {
  float AC;
  if(WRTOperand == 1) {
    AC = abs(X / (X-Y));
  } else if(WRTOperand == 2) {
    AC = abs(Y / (Y-X));
  } else {
    printf("There is no operand %d.\n", WRTOperand);
    return AC;
  }
  printf("AC of x-y | x=%f, y=%f WRT %c is %f.\n",
         X, Y, (WRTOperand==1?'x':'y'), AC);
  
  StorageTable->FP32ACItems.push_back(ACItem<float>(XName, X, YName, Y, Operation::Sub, WRTOperand, AC));

  return AC;
}

float fACfp32BinaryMul(string XName, float X, string YName, float Y, int WRTOperand) {
  float AC=1.0;
  printf("AC of x*y | x=%f, y=%f WRT %c is %f.\n",
         X, Y, (WRTOperand==1?'x':'y'), AC);
  if (WRTOperand != 1 && WRTOperand != 2)
    return AC;
  
  StorageTable->FP32ACItems.push_back(ACItem<float>(XName, X, YName, Y, Operation::Mul, WRTOperand, AC));

  return AC;
}

float fACfp32BinaryDiv(string XName, float X, string YName, float Y, int WRTOperand) {
  float AC=1.0;
  printf("AC of x/y | x=%f, y=%f WRT %c is %f.\n",
         X, Y, (WRTOperand==1?'x':'y'), AC);
  if (WRTOperand != 1 && WRTOperand != 2)
    return AC;
  
  StorageTable->FP32ACItems.push_back(ACItem<float>(XName, X, YName, Y, Operation::Div, WRTOperand, AC));

  return AC;
}

// ---------------------------- Unary Operations ----------------------------

float fACfp32UnarySin(string XName, float X) {
  float AC = abs(X * (cos(X)/sin(X)));

  printf("AC of sin(x) | x=%f is %f.\n", X, AC);

  StorageTable->FP32ACItems.push_back(ACItem<float>(XName, X, "", 0, Operation::Sin, 1, AC));

  return AC;
}

float fACfp32UnaryCos(string XName, float X) {
  float AC = abs(X * tan(X));

  printf("AC of cos(x) | x=%f is %f.\n", X, AC);

  StorageTable->FP32ACItems.push_back(ACItem<float>(XName, X, "", 0, Operation::Cos, 1, AC));

  return AC;
}

float fACfp32UnaryTan(string XName, float X) {
  float AC = abs(X / (sin(X)*cos(X)));

  printf("AC of tan(x) | x=%f is %f.\n", X, AC);

  StorageTable->FP32ACItems.push_back(ACItem<float>(XName, X, "", 0, Operation::Tan, 1, AC));

  return AC;
}

float fACfp32UnaryArcSin(string XName, float X) {
  float AC = abs(X / (sqrt(1-pow(X,2)) * asin(X)));

  printf("AC of asin(x) | x=%f is %f.\n", X, AC);

  StorageTable->FP32ACItems.push_back(ACItem<float>(XName, X, "", 0, Operation::ArcSin, 1, AC));

  return AC;
}

float fACfp32UnaryArcCos(string XName, float X) {
  float AC = abs(-X / (sqrt(1-pow(X,2)) * acos(X)));

  printf("AC of acos(x) | x=%f is %f.\n", X, AC);

  StorageTable->FP32ACItems.push_back(ACItem<float>(XName, X, "", 0, Operation::ArcCos, 1, AC));

  return AC;
}

float fACfp32UnaryArcTan(string XName, float X) {
  float AC = abs(X / (pow(X,2)+1 * atan(X)));

  printf("AC of atan(x) | x=%f is %f.\n", X, AC);

  StorageTable->FP32ACItems.push_back(ACItem<float>(XName, X, "", 0, Operation::ArcTan, 1, AC));

  return AC;
}

float fACfp32UnarySinh(string XName, float X) {
  float AC = abs(X * (cosh(X)/sinh(X)));

  printf("AC of sinh(x) | x=%f is %f.\n", X, AC);

  StorageTable->FP32ACItems.push_back(ACItem<float>(XName, X, "", 0, Operation::Sinh, 1, AC));

  return AC;
}

float fACfp32UnaryCosh(string XName, float X) {
  float AC = abs(X * tanh(X));

  printf("AC of cosh(x) | x=%f is %f.\n", X, AC);

  StorageTable->FP32ACItems.push_back(ACItem<float>(XName, X, "", 0, Operation::Cosh, 1, AC));

  return AC;
}

float fACfp32UnaryTanh(string XName, float X) {
  float AC = abs(X / (sinh(X)*cosh(X)));

  printf("AC of tanh(x) | x=%f is %f.\n", X, AC);

  StorageTable->FP32ACItems.push_back(ACItem<float>(XName, X, "", 0, Operation::Tanh, 1, AC));

  return AC;
}

float fACfp32UnaryExp(string XName, float X) {
  float AC = abs(X );

  printf("AC of exp(x) | x=%f is %f.\n", X, AC);

  StorageTable->FP32ACItems.push_back(ACItem<float>(XName, X, "", 0, Operation::Exp, 1, AC));

  return AC;
}

float fACfp32UnaryLog(string XName, float X) {
  float AC = abs(1/log(X));

  printf("AC of log(x) | x=%f is %f.\n", X, AC);

  StorageTable->FP32ACItems.push_back(ACItem<float>(XName, X, "", 0, Operation::Log, 1, AC));

  return AC;
}

float fACfp32UnarySqrt(string XName, float X) {
  float AC = 0.5;

  printf("AC of sqrt(x) | x=%f is %f.\n", X, AC);

  StorageTable->FP32ACItems.push_back(ACItem<float>(XName, X, "", 0, Operation::Sqrt, 1, AC));

  return AC;
}


// ---------------------------------------------------------------------------
// --------------- fp64 Atomic Condition Calculating Functions ---------------
// ---------------------------------------------------------------------------

// ---------------------------- Binary Operations ----------------------------

double fACfp64BinaryAdd(string XName, double X, string YName, double Y, int WRTOperand) {
  double AC = 0;
  if(WRTOperand == 1) {
    AC = abs(X / (X+Y));
  } else if(WRTOperand == 2) {
    AC = abs(Y / (X+Y));
  } else {
    printf("There is no operand %d.\n", WRTOperand);
    return AC;
  }
  printf("AC of x+y | x=%lf, y=%lf WRT %c is %lf.\n",
         X, Y, (WRTOperand==1?'x':'y'), AC);

  StorageTable->FP64ACItems.push_back(ACItem<double>(XName, X, YName, Y, Operation::Add, WRTOperand, AC));

  return AC;
}

double fACfp64BinarySub(string XName, double X, string YName, double Y, int WRTOperand) {
  double AC;
  if(WRTOperand == 1) {
    AC = abs(X / (X-Y));
  } else if(WRTOperand == 2) {
    AC = abs(Y / (Y-X));
  } else {
    printf("There is no operand %d.\n", WRTOperand);
    return AC;
  }
  printf("AC of x-y | x=%lf, y=%lf WRT %c is %lf.\n",
         X, Y, (WRTOperand==1?'x':'y'), AC);

  StorageTable->FP64ACItems.push_back(ACItem<double>(XName, X, YName, Y, Operation::Sub, WRTOperand, AC));

  return AC;
}

double fACfp64BinaryMul(string XName, double X, string YName, double Y, int WRTOperand) {
  double AC=1.0;
  printf("AC of x*y | x=%lf, y=%lf WRT %c is %lf.\n",
         X, Y, (WRTOperand==1?'x':'y'), AC);
  if (WRTOperand != 1 && WRTOperand != 2)
    return AC;

  StorageTable->FP64ACItems.push_back(ACItem<double>(XName, X, YName, Y, Operation::Mul, WRTOperand, AC));

  return AC;
}

double fACfp64BinaryDiv(string XName, double X, string YName, double Y, int WRTOperand) {
  double AC=1.0;
  printf("AC of x/y | x=%lf, y=%lf WRT %c is %lf.\n",
         X, Y, (WRTOperand==1?'x':'y'), AC);
  if (WRTOperand != 1 && WRTOperand != 2)
    return AC;

  StorageTable->FP64ACItems.push_back(ACItem<double>(XName, X, YName, Y, Operation::Div, WRTOperand, AC));

  return AC;
}

// ---------------------------- Unary Operations ----------------------------

double fACfp64UnarySin(string XName, double X) {
  double AC = abs(X * (cos(X)/sin(X)));

  printf("AC of sin(x) | x=%f is %f.\n", X, AC);

  StorageTable->FP64ACItems.push_back(ACItem<double>(XName, X, "", 0, Operation::Sin, 1, AC));

  return AC;
}

double fACfp64UnaryCos(string XName, double X) {
  double AC = abs(X * tan(X));

  printf("AC of cos(x) | x=%f is %f.\n", X, AC);

  StorageTable->FP64ACItems.push_back(ACItem<double>(XName, X, "", 0, Operation::Cos, 1, AC));

  return AC;
}

double fACfp64UnaryTan(string XName, double X) {
  double AC = abs(X / (sin(X)*cos(X)));

  printf("AC of tan(x) | x=%f is %f.\n", X, AC);

  StorageTable->FP64ACItems.push_back(ACItem<double>(XName, X, "", 0, Operation::Tan, 1, AC));

  return AC;
}

double fACfp64UnaryArcSin(string XName, double X) {
  double AC = abs(X / (sqrt(1-pow(X,2)) * asin(X)));

  printf("AC of asin(x) | x=%f is %f.\n", X, AC);

  StorageTable->FP64ACItems.push_back(ACItem<double>(XName, X, "", 0, Operation::ArcSin, 1, AC));

  return AC;
}

double fACfp64UnaryArcCos(string XName, double X) {
  double AC = abs(-X / (sqrt(1-pow(X,2)) * acos(X)));

  printf("AC of acos(x) | x=%f is %f.\n", X, AC);

  StorageTable->FP64ACItems.push_back(ACItem<double>(XName, X, "", 0, Operation::ArcCos, 1, AC));

  return AC;
}

double fACfp64UnaryArcTan(string XName, double X) {
  double AC = abs(X / (pow(X,2)+1 * atan(X)));

  printf("AC of atan(x) | x=%f is %f.\n", X, AC);

  StorageTable->FP64ACItems.push_back(ACItem<double>(XName, X, "", 0, Operation::ArcTan, 1, AC));

  return AC;
}

double fACfp64UnarySinh(string XName, double X) {
  double AC = abs(X * (cosh(X)/sinh(X)));

  printf("AC of sinh(x) | x=%f is %f.\n", X, AC);

  StorageTable->FP64ACItems.push_back(ACItem<double>(XName, X, "", 0, Operation::Sinh, 1, AC));

  return AC;
}

double fACfp64UnaryCosh(string XName, double X) {
  double AC = abs(X * tanh(X));

  printf("AC of cosh(x) | x=%f is %f.\n", X, AC);

  StorageTable->FP64ACItems.push_back(ACItem<double>(XName, X, "", 0, Operation::Cosh, 1, AC));

  return AC;
}

double fACfp64UnaryTanh(string XName, double X) {
  double AC = abs(X / (sinh(X)*cosh(X)));

  printf("AC of tanh(x) | x=%f is %f.\n", X, AC);

  StorageTable->FP64ACItems.push_back(ACItem<double>(XName, X, "", 0, Operation::Tanh, 1, AC));

  return AC;
}

double fACfp64UnaryExp(string XName, double X) {
  double AC = abs(X );

  printf("AC of exp(x) | x=%f is %f.\n", X, AC);

  StorageTable->FP64ACItems.push_back(ACItem<double>(XName, X, "", 0, Operation::Exp, 1, AC));

  return AC;
}

double fACfp64UnaryLog(string XName, double X) {
  double AC = abs(1/log(X));

  printf("AC of log(x) | x=%f is %f.\n", X, AC);

  StorageTable->FP64ACItems.push_back(ACItem<double>(XName, X, "", 0, Operation::Log, 1, AC));

  return AC;
}

double fACfp64UnarySqrt(string XName, double X) {
  double AC = 0.5;

  printf("AC of sqrt(x) | x=%f is %f.\n", X, AC);

  StorageTable->FP64ACItems.push_back(ACItem<double>(XName, X, "", 0, Operation::Sqrt, 1, AC));

  return AC;
}



#endif // LLVM_ATOMICCONDITION_H
