#include "shared.h"

struct WrapperB {
  OuterY y;
  OuterX x;
};

WrapperB* foo() {
  return new WrapperB();
}
