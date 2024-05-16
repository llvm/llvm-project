#include "definitions.h"
forceinline int InlineFunction(int Param) {
  int Var_1 = Param;
  {
    int Var_2 = Param + Var_1;
    Var_1 = Var_2;
  }
  return Var_1;
}

int test(int Param_1, int Param_2) {
  int A = Param_1;
  A += InlineFunction(Param_2);
  return A;
}
