//
// Created by tanmay on 6/12/22.
//

#ifndef LLVM_ATOMICCONDITION_H
#define LLVM_ATOMICCONDITION_H

#include <math.h>
#include <stdio.h>
#include <string>
#include <vector>

using namespace std;

// Atomic Condition storage
template<typename FloatType>
class ACItem {
  string XName;
  FloatType X;
  string YName;
  FloatType Y;
  int OP;
  int WRTOperand;
  FloatType AC;
public:
  ACItem() {}
  ACItem(string XName, FloatType X,
         string YName, FloatType Y,
         int OP, int WRTOperand) : XName(XName), X(X), YName(YName), Y(Y), OP(OP), WRTOperand(WRTOperand) {}
  ACItem(string XName, FloatType X,
         string YName, FloatType Y,
         int OP, int WRTOperand,
         FloatType AC) : XName(XName), X(X), YName(YName), Y(Y), OP(OP), WRTOperand(WRTOperand), AC(AC) {}
  bool operator==(const ACItem &rhs) const {
    return XName == rhs.XName && X == rhs.X && YName == rhs.YName &&
           Y == rhs.Y && OP == rhs.OP && WRTOperand == rhs.WRTOperand &&
           AC == rhs.AC;
  }
  bool operator!=(const ACItem &rhs) const { return !(rhs == *this); }

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

double fACfp64BinaryAdd(string XName, double X, string YName, double Y, int WRTOperand);
double fACfp64BinarySub(string XName, double X, string YName, double Y, int WRTOperand);
double fACfp64BinaryMul(string XName, double X, string YName, double Y, int WRTOperand);
double fACfp64BinaryDiv(string XName, double X, string YName, double Y, int WRTOperand);


void fACCreate(uint64_t NumItems) {
  StorageTable = new ACTable(NumItems);
}

// Driver function selecting atomic condition function for unary float operation
float fACfp32UnaryDriver(string XName, float X, int OP) {
  switch (OP) {
  default:
    printf("No such operation\n");
  }
  return 0.0;
}

// Driver function selecting atomic condition function for binary float operation
float fACfp32BinaryDriver(string XName, float X, string YName, float Y, int OP, int WRTOperand) {
  switch (OP) {
  case 14:
    return fACfp32BinaryAdd(XName, X, YName, Y, WRTOperand);
  case 16:
    return fACfp32BinarySub(XName, X, YName, Y, WRTOperand);
  case 18:
    return fACfp32BinaryMul(XName, X, YName, Y, WRTOperand);
  case 21:
    return fACfp32BinaryDiv(XName, X, YName, Y, WRTOperand);
  default:
    printf("No such operation\n");
  }
  return 0.0;
}

// Driver function selecting atomic condition function for unary double operation
double fACfp64UnaryDriver(string XName, double X, int OP) {
  switch (OP) {
  default:
    printf("No such operation\n");
  }
  return 0.0;
}

// Driver function selecting atomic condition function for binary double operation
double fACfp64BinaryDriver(string XName, double X, string YName, double Y, int OP, int WRTOperand) {
  switch (OP) {
  case 14:
    return fACfp64BinaryAdd(XName, X, YName, Y, WRTOperand);
  case 16:
    return fACfp64BinarySub(XName, X, YName, Y, WRTOperand);
  case 18:
    return fACfp64BinaryMul(XName, X, YName, Y, WRTOperand);
  case 21:
    return fACfp64BinaryDiv(XName, X, YName, Y, WRTOperand);
  default:
    printf("No such operation\n");
  }
  return 0.0;
}

// ---------------------------------------------------------------------------
// --------------- fp32 Atomic Condition Calculating Functions ---------------
// ---------------------------------------------------------------------------

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
  
  StorageTable->FP32ACItems.push_back(ACItem<float>(XName, X, YName, Y, 14, WRTOperand, AC));

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
  
  StorageTable->FP32ACItems.push_back(ACItem<float>(XName, X, YName, Y, 16, WRTOperand, AC));

  return AC;
}

float fACfp32BinaryMul(string XName, float X, string YName, float Y, int WRTOperand) {
  float AC=1.0;
  printf("AC of x*y | x=%f, y=%f WRT %c is %f.\n",
         X, Y, (WRTOperand==1?'x':'y'), AC);
  if (WRTOperand != 1 && WRTOperand != 2)
    return AC;
  
  StorageTable->FP32ACItems.push_back(ACItem<float>(XName, X, YName, Y, 18, WRTOperand, AC));

  return AC;
}

float fACfp32BinaryDiv(string XName, float X, string YName, float Y, int WRTOperand) {
  float AC=1.0;
  printf("AC of x/y | x=%f, y=%f WRT %c is %f.\n",
         X, Y, (WRTOperand==1?'x':'y'), AC);
  if (WRTOperand != 1 && WRTOperand != 2)
    return AC;
  
  StorageTable->FP32ACItems.push_back(ACItem<float>(XName, X, YName, Y, 21, WRTOperand, AC));

  return AC;
}


// ---------------------------------------------------------------------------
// --------------- fp64 Atomic Condition Calculating Functions ---------------
// ---------------------------------------------------------------------------

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

  StorageTable->FP64ACItems.push_back(ACItem<double>(XName, X, YName, Y, 14, WRTOperand, AC));

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

  StorageTable->FP64ACItems.push_back(ACItem<double>(XName, X, YName, Y, 16, WRTOperand, AC));

  return AC;
}

double fACfp64BinaryMul(string XName, double X, string YName, double Y, int WRTOperand) {
  double AC=1.0;
  printf("AC of x*y | x=%lf, y=%lf WRT %c is %lf.\n",
         X, Y, (WRTOperand==1?'x':'y'), AC);
  if (WRTOperand != 1 && WRTOperand != 2)
    return AC;

  StorageTable->FP64ACItems.push_back(ACItem<double>(XName, X, YName, Y, 18, WRTOperand, AC));

  return AC;
}

double fACfp64BinaryDiv(string XName, double X, string YName, double Y, int WRTOperand) {
  double AC=1.0;
  printf("AC of x/y | x=%lf, y=%lf WRT %c is %lf.\n",
         X, Y, (WRTOperand==1?'x':'y'), AC);
  if (WRTOperand != 1 && WRTOperand != 2)
    return AC;

  StorageTable->FP64ACItems.push_back(ACItem<double>(XName, X, YName, Y, 21, WRTOperand, AC));

  return AC;
}


#endif // LLVM_ATOMICCONDITION_H
