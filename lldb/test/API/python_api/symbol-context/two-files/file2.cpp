#include "decls.h"

struct2::~struct2() {
  int x = g(); // Break2
}

void struct2::f() {}
