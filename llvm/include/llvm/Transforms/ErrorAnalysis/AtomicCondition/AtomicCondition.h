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
  TruncToFloat
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

void fACCreate(uint64_t NumItems) {
  StorageTable = new ACTable(NumItems);
}

// ---------------------------------------------------------------------------
// --------------- fp32 Atomic Condition Calculating Functions ---------------
// ---------------------------------------------------------------------------

// ---------------------------- Binary Operations ----------------------------

void fACfp32BinaryAdd(const char *XName, float X, const char *YName, float Y, int WRTOperand) {
  float AC;
  if(WRTOperand == 1) {
    AC = abs(X / (X+Y));
  } else if(WRTOperand == 2) {
    AC = abs(Y / (X+Y));
  } else {
    printf("There is no operand %d.\n", WRTOperand);
    return ;
  }
  printf("AC of x+y | x=%f, y=%f WRT %c is %f.\n",
         X, Y, (WRTOperand==1?'x':'y'), AC);
  
  StorageTable->FP32ACItems.push_back(ACItem<float>(XName, X, YName, Y, Operation::Add, WRTOperand, AC));

  return ;
}

void fACfp32BinarySub(const char *XName, float X, const char *YName, float Y, int WRTOperand) {
  float AC;
  if(WRTOperand == 1) {
    AC = abs(X / (X-Y));
  } else if(WRTOperand == 2) {
    AC = abs(Y / (Y-X));
  } else {
    printf("There is no operand %d.\n", WRTOperand);
    return ;
  }
  printf("AC of x-y | x=%f, y=%f WRT %c is %f.\n",
         X, Y, (WRTOperand==1?'x':'y'), AC);
  
  StorageTable->FP32ACItems.push_back(ACItem<float>(XName, X, YName, Y, Operation::Sub, WRTOperand, AC));

  return ;
}

void fACfp32BinaryMul(const char *XName, float X, const char *YName, float Y, int WRTOperand) {
  float AC=1.0;
  printf("AC of x*y | x=%f, y=%f WRT %c is %f.\n",
         X, Y, (WRTOperand==1?'x':'y'), AC);
  if (WRTOperand != 1 && WRTOperand != 2)
    return ;
  
  StorageTable->FP32ACItems.push_back(ACItem<float>(XName, X, YName, Y, Operation::Mul, WRTOperand, AC));

  return ;
}

void fACfp32BinaryDiv(const char *XName, float X, const char *YName, float Y, int WRTOperand) {
  float AC=1.0;
  printf("AC of x/y | x=%f, y=%f WRT %c is %f.\n",
         X, Y, (WRTOperand==1?'x':'y'), AC);
  if (WRTOperand != 1 && WRTOperand != 2)
    return ;
  
  StorageTable->FP32ACItems.push_back(ACItem<float>(XName, X, YName, Y, Operation::Div, WRTOperand, AC));

  return ;
}

// ---------------------------- Unary Operations ----------------------------

void fACfp32UnarySin(const char *XName, float X) {
  float AC = abs(X * (cos(X)/sin(X)));

  printf("AC of sin(x) | x=%f is %f.\n", X, AC);

  StorageTable->FP32ACItems.push_back(ACItem<float>(XName, X, "", 0, Operation::Sin, 1, AC));

  return ;
}

void fACfp32UnaryCos(const char *XName, float X) {
  float AC = abs(X * tan(X));

  printf("AC of cos(x) | x=%f is %f.\n", X, AC);

  StorageTable->FP32ACItems.push_back(ACItem<float>(XName, X, "", 0, Operation::Cos, 1, AC));

  return ;
}

void fACfp32UnaryTan(const char *XName, float X) {
  float AC = abs(X / (sin(X)*cos(X)));

  printf("AC of tan(x) | x=%f is %f.\n", X, AC);

  StorageTable->FP32ACItems.push_back(ACItem<float>(XName, X, "", 0, Operation::Tan, 1, AC));

  return ;
}

void fACfp32UnaryArcSin(const char *XName, float X) {
  float AC = abs(X / (sqrt(1-pow(X,2)) * asin(X)));

  printf("AC of asin(x) | x=%f is %f.\n", X, AC);

  StorageTable->FP32ACItems.push_back(ACItem<float>(XName, X, "", 0, Operation::ArcSin, 1, AC));

  return ;
}

void fACfp32UnaryArcCos(const char *XName, float X) {
  float AC = abs(-X / (sqrt(1-pow(X,2)) * acos(X)));

  printf("AC of acos(x) | x=%f is %f.\n", X, AC);

  StorageTable->FP32ACItems.push_back(ACItem<float>(XName, X, "", 0, Operation::ArcCos, 1, AC));

  return ;
}

void fACfp32UnaryArcTan(const char *XName, float X) {
  float AC = abs(X / (pow(X,2)+1 * atan(X)));

  printf("AC of atan(x) | x=%f is %f.\n", X, AC);

  StorageTable->FP32ACItems.push_back(ACItem<float>(XName, X, "", 0, Operation::ArcTan, 1, AC));

  return ;
}

void fACfp32UnarySinh(const char *XName, float X) {
  float AC = abs(X * (cosh(X)/sinh(X)));

  printf("AC of sinh(x) | x=%f is %f.\n", X, AC);

  StorageTable->FP32ACItems.push_back(ACItem<float>(XName, X, "", 0, Operation::Sinh, 1, AC));

  return ;
}

void fACfp32UnaryCosh(const char *XName, float X) {
  float AC = abs(X * tanh(X));

  printf("AC of cosh(x) | x=%f is %f.\n", X, AC);

  StorageTable->FP32ACItems.push_back(ACItem<float>(XName, X, "", 0, Operation::Cosh, 1, AC));

  return ;
}

void fACfp32UnaryTanh(const char *XName, float X) {
  float AC = abs(X / (sinh(X)*cosh(X)));

  printf("AC of tanh(x) | x=%f is %f.\n", X, AC);

  StorageTable->FP32ACItems.push_back(ACItem<float>(XName, X, "", 0, Operation::Tanh, 1, AC));

  return ;
}

void fACfp32UnaryExp(const char *XName, float X) {
  float AC = abs(X );

  printf("AC of exp(x) | x=%f is %f.\n", X, AC);

  StorageTable->FP32ACItems.push_back(ACItem<float>(XName, X, "", 0, Operation::Exp, 1, AC));

  return ;
}

void fACfp32UnaryLog(const char *XName, float X) {
  float AC = abs(1/log(X));

  printf("AC of log(x) | x=%f is %f.\n", X, AC);

  StorageTable->FP32ACItems.push_back(ACItem<float>(XName, X, "", 0, Operation::Log, 1, AC));

  return ;
}

void fACfp32UnarySqrt(const char *XName, float X) {
  float AC = 0.5;

  printf("AC of sqrt(x) | x=%f is %f.\n", X, AC);

  StorageTable->FP32ACItems.push_back(ACItem<float>(XName, X, "", 0, Operation::Sqrt, 1, AC));

  return ;
}


// ---------------------------------------------------------------------------
// --------------- fp64 Atomic Condition Calculating Functions ---------------
// ---------------------------------------------------------------------------

// ---------------------------- Binary Operations ----------------------------

void fACfp64BinaryAdd(const char *XName, double X, const char *YName, double Y, int WRTOperand) {
  double AC;
  if(WRTOperand == 1) {
    AC = abs(X / (X+Y));
  } else if(WRTOperand == 2) {
    AC = abs(Y / (X+Y));
  } else {
    printf("There is no operand %d.\n", WRTOperand);
    return ;
  }
  printf("AC of x+y | x=%lf, y=%lf WRT %c is %lf.\n",
         X, Y, (WRTOperand==1?'x':'y'), AC);

  StorageTable->FP64ACItems.push_back(ACItem<double>(XName, X, YName, Y, Operation::Add, WRTOperand, AC));

  return ;
}

void fACfp64BinarySub(const char *XName, double X, const char *YName, double Y, int WRTOperand) {
  double AC;
  if(WRTOperand == 1) {
    AC = abs(X / (X-Y));
  } else if(WRTOperand == 2) {
    AC = abs(Y / (Y-X));
  } else {
    printf("There is no operand %d.\n", WRTOperand);
    return ;
  }
  printf("AC of x-y | x=%lf, y=%lf WRT %c is %lf.\n",
         X, Y, (WRTOperand==1?'x':'y'), AC);

  StorageTable->FP64ACItems.push_back(ACItem<double>(XName, X, YName, Y, Operation::Sub, WRTOperand, AC));

  return ;
}

void fACfp64BinaryMul(const char *XName, double X, const char *YName, double Y, int WRTOperand) {
  double AC=1.0;
  printf("AC of x*y | x=%lf, y=%lf WRT %c is %lf.\n",
         X, Y, (WRTOperand==1?'x':'y'), AC);
  if (WRTOperand != 1 && WRTOperand != 2)
    return ;

  StorageTable->FP64ACItems.push_back(ACItem<double>(XName, X, YName, Y, Operation::Mul, WRTOperand, AC));

  return ;
}

void fACfp64BinaryDiv(const char *XName, double X, const char *YName, double Y, int WRTOperand) {
  double AC=1.0;
  printf("AC of x/y | x=%lf, y=%lf WRT %c is %lf.\n",
         X, Y, (WRTOperand==1?'x':'y'), AC);
  if (WRTOperand != 1 && WRTOperand != 2)
    return ;

  StorageTable->FP64ACItems.push_back(ACItem<double>(XName, X, YName, Y, Operation::Div, WRTOperand, AC));

  return ;
}

// ---------------------------- Unary Operations ----------------------------

void fACfp64UnarySin(const char *XName, double X) {
  double AC = abs(X * (cos(X)/sin(X)));

  printf("AC of sin(x) | x=%f is %f.\n", X, AC);

  StorageTable->FP64ACItems.push_back(ACItem<double>(XName, X, "", 0, Operation::Sin, 1, AC));

  return ;
}

void fACfp64UnaryCos(const char *XName, double X) {
  double AC = abs(X * tan(X));

  printf("AC of cos(x) | x=%f is %f.\n", X, AC);

  StorageTable->FP64ACItems.push_back(ACItem<double>(XName, X, "", 0, Operation::Cos, 1, AC));

  return ;
}

void fACfp64UnaryTan(const char *XName, double X) {
  double AC = abs(X / (sin(X)*cos(X)));

  printf("AC of tan(x) | x=%f is %f.\n", X, AC);

  StorageTable->FP64ACItems.push_back(ACItem<double>(XName, X, "", 0, Operation::Tan, 1, AC));

  return ;
}

void fACfp64UnaryArcSin(const char *XName, double X) {
  double AC = abs(X / (sqrt(1-pow(X,2)) * asin(X)));

  printf("AC of asin(x) | x=%f is %f.\n", X, AC);

  StorageTable->FP64ACItems.push_back(ACItem<double>(XName, X, "", 0, Operation::ArcSin, 1, AC));

  return ;
}

void fACfp64UnaryArcCos(const char *XName, double X) {
  double AC = abs(-X / (sqrt(1-pow(X,2)) * acos(X)));

  printf("AC of acos(x) | x=%f is %f.\n", X, AC);

  StorageTable->FP64ACItems.push_back(ACItem<double>(XName, X, "", 0, Operation::ArcCos, 1, AC));

  return ;
}

void fACfp64UnaryArcTan(const char *XName, double X) {
  double AC = abs(X / (pow(X,2)+1 * atan(X)));

  printf("AC of atan(x) | x=%f is %f.\n", X, AC);

  StorageTable->FP64ACItems.push_back(ACItem<double>(XName, X, "", 0, Operation::ArcTan, 1, AC));

  return ;
}

void fACfp64UnarySinh(const char *XName, double X) {
  double AC = abs(X * (cosh(X)/sinh(X)));

  printf("AC of sinh(x) | x=%f is %f.\n", X, AC);

  StorageTable->FP64ACItems.push_back(ACItem<double>(XName, X, "", 0, Operation::Sinh, 1, AC));

  return ;
}

void fACfp64UnaryCosh(const char *XName, double X) {
  double AC = abs(X * tanh(X));

  printf("AC of cosh(x) | x=%f is %f.\n", X, AC);

  StorageTable->FP64ACItems.push_back(ACItem<double>(XName, X, "", 0, Operation::Cosh, 1, AC));

  return ;
}

void fACfp64UnaryTanh(const char *XName, double X) {
  double AC = abs(X / (sinh(X)*cosh(X)));

  printf("AC of tanh(x) | x=%f is %f.\n", X, AC);

  StorageTable->FP64ACItems.push_back(ACItem<double>(XName, X, "", 0, Operation::Tanh, 1, AC));

  return ;
}

void fACfp64UnaryExp(const char *XName, double X) {
  double AC = abs(X );

  printf("AC of exp(x) | x=%f is %f.\n", X, AC);

  StorageTable->FP64ACItems.push_back(ACItem<double>(XName, X, "", 0, Operation::Exp, 1, AC));

  return ;
}

void fACfp64UnaryLog(const char *XName, double X) {
  double AC = abs(1/log(X));

  printf("AC of log(x) | x=%f is %f.\n", X, AC);

  StorageTable->FP64ACItems.push_back(ACItem<double>(XName, X, "", 0, Operation::Log, 1, AC));

  return ;
}

void fACfp64UnarySqrt(const char *XName, double X) {
  double AC = 0.5;

  printf("AC of sqrt(x) | x=%f is %f.\n", X, AC);

  StorageTable->FP64ACItems.push_back(ACItem<double>(XName, X, "", 0, Operation::Sqrt, 1, AC));

  return ;
}

void fACfp64TruncToFloat(const char *XName, double X) {
  float AC = 1.0;
  
  printf("AC of trunc(x, fp32) | x=%f is %f.\n", X, AC);
  
  StorageTable->FP64ACItems.push_back(ACItem<double>(XName, X, "", 0, Operation::TruncToFloat, 1, AC));

  return ;
}

// Driver function selecting atomic condition function for unary float operation
void fACfp32UnaryDriver(const char *XName, float X, Operation OP) {
  switch (OP) {
  case 4:
    fACfp32UnarySin(XName, X);
    break;
  case 5:
    fACfp32UnaryCos(XName, X);
    break;
  case 6:
    fACfp32UnaryTan(XName, X);
    break;
  case 7:
    fACfp32UnaryArcSin(XName, X);
    break;
  case 8:
    fACfp32UnaryArcCos(XName, X);
    break;
  case 9:
    fACfp32UnaryArcTan(XName, X);
    break;
  case 10:
    fACfp32UnarySinh(XName, X);
    break;
  case 11:
    fACfp32UnaryCosh(XName, X);
    break;
  case 12:
    fACfp32UnaryTanh(XName, X);
    break;
  case 13:
    fACfp32UnaryExp(XName, X);
    break;
  case 14:
    fACfp32UnaryLog(XName, X);
    break;
  case 15:
    fACfp32UnarySqrt(XName, X);
    break;
  default:
    printf("No such operation\n");
    break;
  }

  return ;
}

// Driver function selecting atomic condition function for binary float operation
void fACfp32BinaryDriver(const char *XName, float X, const char *YName, float Y, Operation OP, int WRTOperand) {
  switch (OP) {
  case 0:
    fACfp32BinaryAdd(XName, X, YName, Y, WRTOperand);
    break;
  case 1:
    fACfp32BinarySub(XName, X, YName, Y, WRTOperand);
    break;
  case 2:
    fACfp32BinaryMul(XName, X, YName, Y, WRTOperand);
    break;
  case 3:
    fACfp32BinaryDiv(XName, X, YName, Y, WRTOperand);
    break;
  default:
    printf("No such operation\n");
    break;
  }

  return ;
}

// Driver function selecting atomic condition function for unary double operation
void fACfp64UnaryDriver(const char *XName, double X, Operation OP) {
  switch (OP) {
  case 4:
    fACfp64UnarySin(XName, X);
    break;
  case 5:
    fACfp64UnaryCos(XName, X);
    break;
  case 6:
    fACfp64UnaryTan(XName, X);
    break;
  case 7:
    fACfp64UnaryArcSin(XName, X);
    break;
  case 8:
    fACfp64UnaryArcCos(XName, X);
    break;
  case 9:
    fACfp64UnaryArcTan(XName, X);
    break;
  case 10:
    fACfp64UnarySinh(XName, X);
    break;
  case 11:
    fACfp64UnaryCosh(XName, X);
    break;
  case 12:
    fACfp64UnaryTanh(XName, X);
    break;
  case 13:
    fACfp64UnaryExp(XName, X);
    break;
  case 14:
    fACfp64UnaryLog(XName, X);
    break;
  case 15:
    fACfp64UnarySqrt(XName, X);
    break;
  case 16:
    fACfp64TruncToFloat(XName, X);
    break;
  default:
    printf("No such operation\n");
    break;
  }

  return ;
}

// Driver function selecting atomic condition function for binary double operation
void fACfp64BinaryDriver(const char *XName, double X, const char *YName, double Y, Operation OP, int WRTOperand) {
  switch (OP) {
  case 0:
    fACfp64BinaryAdd(XName, X, YName, Y, WRTOperand);
    break;
  case 1:
    fACfp64BinarySub(XName, X, YName, Y, WRTOperand);
    break;
  case 2:
    fACfp64BinaryMul(XName, X, YName, Y, WRTOperand);
    break;
  case 3:
    fACfp64BinaryDiv(XName, X, YName, Y, WRTOperand);
    break;
  default:
    printf("No such operation\n");
    break;
  }

  return ;
}

#endif // LLVM_ATOMICCONDITION_H

