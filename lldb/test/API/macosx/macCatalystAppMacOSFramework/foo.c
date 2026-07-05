#include "foo.h"

void stop() {}

int foo() {
  stop();
  return 0;
}
