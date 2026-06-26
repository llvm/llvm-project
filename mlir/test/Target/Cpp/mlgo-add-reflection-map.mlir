// RUN: mlir-opt -split-input-file --mlgo-add-reflection-map="field-attr-name=emitc.field_ref excluded-field-attrs=emitc.other_field" %s | mlir-translate -mlir-to-cpp | FileCheck %s

// Test that a reflection map and lookup function are generated in the class.

emitc.class @actionClass {
  emitc.field @fieldName0 : !emitc.array<1xf32>  {emitc.field_ref = ["another_feature"]}
  emitc.field @fieldName1 : !emitc.array<1xf32>  {emitc.field_ref = ["some_feature"]}
  emitc.func @"operator()"() {
    %0 = get_field @fieldName0 : !emitc.array<1xf32>
    return
  }
}

// CHECK:       #include <map>
// CHECK-NEXT:  #include <string>
// CHECK-NEXT:  class actionClass {
// CHECK-NEXT:   public:
// CHECK-NEXT:    float fieldName0[1];
// CHECK-NEXT:    float fieldName1[1];
// CHECK-NEXT:    const std::map<std::string, char*> reflectionMap = { { "another_feature", reinterpret_cast<char*>(&fieldName0) }, { "some_feature", reinterpret_cast<char*>(&fieldName1) } };
// CHECK-NEXT:    char* getBufferForName(std::string [[VAL_1:v[0-9]+]]) {
// CHECK-NEXT:      char* [[VAL_2:v[0-9]+]] = reflectionMap.at([[VAL_1]]);
// CHECK-NEXT:      return [[VAL_2]];
// CHECK-NEXT:    }
// CHECK-NEXT:    void operator()() {
// CHECK-NEXT:      return;
// CHECK-NEXT:    }
// CHECK-NEXT:  };

// -----

// Test that fields with excluded attributes are ignored.

emitc.class @actionClassExcluded {
  emitc.field @fieldName0 : !emitc.array<1xf32>  {emitc.field_ref = ["another_feature"]}
  emitc.field @fieldName1 : !emitc.array<1xf32>  {emitc.other_field = ["some_feature"]}
  emitc.func @"operator()"() {
    %0 = get_field @fieldName0 : !emitc.array<1xf32>
    return
  }
}

// CHECK:       #include <map>
// CHECK-NEXT:  #include <string>
// CHECK-NEXT:  class actionClassExcluded {
// CHECK-NEXT:   public:
// CHECK-NEXT:    float fieldName0[1];
// CHECK-NEXT:    float fieldName1[1];
// CHECK-NEXT:    const std::map<std::string, char*> reflectionMap = { { "another_feature", reinterpret_cast<char*>(&fieldName0) } };
// CHECK-NEXT:    char* getBufferForName(std::string [[VAL_1:v[0-9]+]]) {
// CHECK-NEXT:      char* [[VAL_2:v[0-9]+]] = reflectionMap.at([[VAL_1]]);
// CHECK-NEXT:      return [[VAL_2]];
// CHECK-NEXT:    }
// CHECK-NEXT:    void operator()() {
// CHECK-NEXT:      return;
// CHECK-NEXT:    }
// CHECK-NEXT:  };
