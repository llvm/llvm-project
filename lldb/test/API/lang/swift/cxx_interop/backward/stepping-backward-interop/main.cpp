#include "swift-types.h"

using namespace a;

int testFunc() {
  swiftFunc(); // Break here for func
  return 0;
}

int testMethodClass() {
  auto swiftClass = SwiftClass::init();
  swiftClass.swiftMethod(); // Break here for method - class
  return 0;
}

int testConstructorClass() {
  auto swiftClass = SwiftClass::init(); // Break here for constructor - class
  return 0;
}

int testStaticMethodClass() {
  SwiftClass::swiftStaticMethod(); // Break here for static method - class
  return 0;
}

int testGetterClass() {
  auto swiftClass = SwiftClass::init();
  swiftClass.getSwiftProperty(); // Break here for getter - class
  return 0;
}

int testSetterClass() {
  auto swiftClass = SwiftClass::init();
  auto str = getString();
  swiftClass.setSwiftProperty(str); // Break here for setter - class
  return 0;
}

int testOverridenMethodClass() {
  auto swiftClass = SwiftSubclass::init();
  swiftClass.overrideableMethod(); // Break here for overridden - class
  return 0;
}

void testClass() {
  testMethodClass();
  testConstructorClass();
  testStaticMethodClass();
  testGetterClass();
  testSetterClass();
  testOverridenMethodClass();
}


int testMethodStruct() {
  auto swiftStruct = SwiftStruct::init();
  swiftStruct.swiftMethod(); // Break here for method - struct
  return 0;
}

int testConstructorStruct() {
  auto swiftStruct = SwiftStruct::init(); // Break here for constructor - struct
  return 0;
}

int testStaticMethodStruct() {
  SwiftStruct::swiftStaticMethod(); // Break here for static method - struct
  return 0;
}

int testGetterStruct() {
  auto swiftStruct = SwiftStruct::init();
  swiftStruct.getSwiftProperty(); // Break here for getter - struct
  return 0;
}

int testSetterStruct() {
  auto swiftStruct = SwiftStruct::init();
  auto str = getString();
  swiftStruct.setSwiftProperty(str); // Break here for setter - struct
  return 0;
}

void testStruct() {
  testMethodStruct();
  testConstructorStruct();
  testStaticMethodStruct();
  testGetterStruct();
  testSetterStruct();
}

int main() {
  testFunc();
  testClass();
  testStruct();
  return 0;
}
