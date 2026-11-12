#include "a.h"

bool A::b(int x) {
  if (x)
    return true;

  return false;
}

bool B::member_func_a(A a) {
  return a.b(10); // We will try and add a breakpoint here which
                  // trigger an assert since we will attempt to
                  // to add ParamVarDecl a to CXXRecordDecl A
};
