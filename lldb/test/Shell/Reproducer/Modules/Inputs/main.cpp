#include "Foo.h"

void stop() {}

int main(int argc, char **argv) {
  Foo foo;
  stop(); // break here.
  return 0;
}
