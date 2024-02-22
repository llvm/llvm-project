using INTPTR = const int *;
int foo(INTPTR ParamPtr, unsigned ParamUnsigned, bool ParamBool) {
  if (ParamBool) {
    typedef int INTEGER;
    const INTEGER CONSTANT = 7;
    return CONSTANT;
  }
  return ParamUnsigned;
}
