#include "swift-types.h"

using namespace a;

int main() {
  swiftFunc(); 
  auto swiftClass = SwiftClass::init();
  swiftClass.swiftMethod();
  SwiftClass::swiftStaticMethod(); 
  swiftClass.getSwiftProperty(); 
  auto swiftSubclass = SwiftSubclass::init();
  SwiftClass swiftSubclassAsClass = swiftSubclass;
  swiftSubclassAsClass.overrideableMethod();
  return 0; // Break here 
}

