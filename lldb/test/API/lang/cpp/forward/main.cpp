#include "foo.h"

template <typename T> struct bar {
  T a;
};

int main() {
  bar<int> b{47};

  foo(&b);

  return 0;
}
