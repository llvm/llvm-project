#include "swift-types.h"

int main() {
  using namespace a;
  auto swiftClass = returnSwiftClass();
  auto swiftSublass = returnSwiftSubclassAsClass();
  auto swiftStruct = returnSwiftStruct();
  auto wrapper = returnPair(swiftClass, swiftStruct);
  return 0; // Set breakpoint here.
}
