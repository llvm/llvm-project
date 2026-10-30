#include "inlined.h"

int main(int argc, char const *argv[]) {
  int result = inlined_add(argc, 42); // break here
  return result - result;
}
