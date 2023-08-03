#include "swift-types.h"


using namespace swift;
using namespace a;

int testArray() {
  auto array = createArray();
  return 0; // break here for array
}

int testArrayOfInts() {
  auto array = createArrayOfInts();
  return 0; // break here for array of ints
}


int testOptional() {
  auto optional = createOptional();
  return 0; // break here for optional
}

int testOptionalPrimitive() {
  auto optional = createOptionalPrimitive();
  return 0; // break here for optional primitive
}

int testString() {
  auto string = createString();
  return 0; // break here for string
}

int main() {
  testArray();
  testArrayOfInts();
  testOptional();
  testOptionalPrimitive();
  testString();
}
