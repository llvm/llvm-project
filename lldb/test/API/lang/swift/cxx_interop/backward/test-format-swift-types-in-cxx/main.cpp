#include "swift-types.h"

int main() {
  using namespace a;
  auto swiftClass = returnSwiftClass();
  auto swiftSublass = returnSwiftSubclassAsClass();
  auto swiftStruct = returnSwiftStruct();
  return 0; // Set breakpoint here.
}
