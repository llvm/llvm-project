#include "swift-types.h"

using namespace a;

int testSwiftClass() {
  auto swiftClass = returnSwiftClass();
  return 0; // Break here for class
}

int testSwiftSubclass() {
  auto swiftSublass = returnSwiftSubclass();
  return 0; // Break here for subclass
}

int testSwiftStruct() {
  auto swiftStruct = returnSwiftStruct();
  return 0; // Break here for struct
}

int testSwiftGenericStruct() {
  auto swiftClass = returnSwiftClass();
  auto swiftStruct = returnSwiftStruct();
  auto wrapper = returnStructPair(swiftClass, swiftStruct);
  return 0; // Break here for generic struct
}

int testSwiftGenericEnum() {
  auto swiftClass = returnSwiftClass();
  auto swiftEnum = returnGenericEnum(swiftClass);
  return 0; // Break here for generic enum
}

struct TypeWithSwiftIvars {
  SwiftClass swiftClass;
  SwiftSubclass swiftSubclass;
  SwiftStruct swiftStruct;

  TypeWithSwiftIvars()
      : swiftClass(returnSwiftClass()), swiftSubclass(returnSwiftSubclass()),
        swiftStruct(returnSwiftStruct()) {}
};

int testSwiftIvars() {
  TypeWithSwiftIvars type_with_ivars;
  return 0; // Break here for swift ivars
}

int testSwiftAlias() {
  auto aliased = returnAliasedClass();
  return 0; // Break here for swift alias
}

int main() {
  testSwiftClass();
  testSwiftSubclass();
  testSwiftStruct();
  testSwiftGenericStruct();
  testSwiftGenericEnum();
  testSwiftIvars();
  testSwiftAlias();
  return 0;
}
