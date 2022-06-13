//
// Created by tanmay on 6/12/22.
//

#ifndef LLVM_ATOMICCONDITION_H
#define LLVM_ATOMICCONDITION_H

#include <math.h>
#include <iostream>

using namespace std;

// ----------------------------------------------------------------------
// --------------- Atomic Condition Calculating Functions ---------------
// ----------------------------------------------------------------------

template<typename FP>
FP acBinaryAdd(FP X, FP Y, int WRTOperand) {
  FP AC;
  if(WRTOperand == 1) {
    AC = abs(X / (X+Y));
  } else if(WRTOperand == 2) {
    AC = abs(Y / (X+Y));
  } else {
    cout << "There is no operand " << WRTOperand << ".\n";
  }
  cout <<"AC of x+y | x=" << X << ", y=" << Y <<
         " WRT " << (WRTOperand==1?'x':'y') << " is " << AC << '\n';

  return AC;
}

template<typename FP>
FP acBinarySub(FP X, FP Y, int WRTOperand) {
  FP AC;
  if(WRTOperand == 1) {
    AC = abs(X / (X-Y));
  } else if(WRTOperand == 2) {
    AC = abs(Y / (Y-X));
  } else {
    cout << "There is no operand " << WRTOperand << ".\n";
  }
  cout <<"AC of x-y | x=" << X << ", y=" << Y <<
      " WRT " << (WRTOperand==1?'x':'y') << " is " << AC << '\n';

  return AC;
}

template<typename FP>
FP acBinaryMul(FP X, FP Y, int WRTOperand) {
  FP AC=1;
  cout <<"AC of x*y | x=" << X << ", y=" << Y <<
      " WRT " << (WRTOperand==1?'x':'y') << " is " << AC << '\n';

  return AC;
}

template<typename FP>
FP acBinaryDiv(FP X, FP Y, int WRTOperand) {
  FP AC=1;
  cout <<"AC of x+y | x=" << X << ", y=" << Y <<
      " WRT " << (WRTOperand==1?'x':'y') << " is " << AC << '\n';

  return AC;
}



#endif // LLVM_ATOMICCONDITION_H
