// RUN: mlir-opt -split-input-file --mlgo-add-reflection-map="included-field-attrs=emitc.field_ref" %s | mlir-translate -mlir-to-cpp | FileCheck %s

/// Test that a reflection map and lookup function are generated in the class.

emitc.class @foo {
  emitc.field @fieldName0 : !emitc.array<1xf32>  {emitc.field_ref = ["another_feature"]}
  emitc.field @fieldName1 : !emitc.array<1xf32>  {emitc.field_ref = ["some_feature"]}
  emitc.func @bar() {
    %0 = get_field @fieldName0 : !emitc.array<1xf32>
    return
  }
}

// CHECK:       #include <map>
// CHECK-NEXT:  #include <string>
// CHECK-NEXT:  class foo {
// CHECK-NEXT:   public:
// CHECK-NEXT:    float fieldName0[1];
// CHECK-NEXT:    float fieldName1[1];
// CHECK-NEXT:    const std::map<std::string, char*> reflectionMap = { { "another_feature", reinterpret_cast<char*>(&fieldName0) },
// CHECK-SAME:                                                         { "some_feature", reinterpret_cast<char*>(&fieldName1) } };
// CHECK-NEXT:    char* getBufferForName(std::string [[ARG:v[0-9]+]]) {
// CHECK-NEXT:      char* [[VAR:v[0-9]+]] = reflectionMap.at([[ARG]]);
// CHECK-NEXT:      return [[VAR]];
// CHECK-NEXT:    }
// CHECK-NEXT:    void bar() {
// CHECK-NEXT:      return;
// CHECK-NEXT:    }
// CHECK-NEXT:  };

// -----

/// Test that fields without included attributes are ignored.

emitc.class @foo_unsupported_attrs {
  emitc.field @fieldName0 : !emitc.array<1xf32>  {emitc.field_ref = ["another_feature"]}
  emitc.field @fieldName1 : !emitc.array<1xf32>  {emitc.other_field = ["some_feature"]}
  emitc.func @bar() {
    %0 = get_field @fieldName0 : !emitc.array<1xf32>
    return
  }
}

// CHECK:       #include <map>
// CHECK-NEXT:  #include <string>
// CHECK-NEXT:  class foo_unsupported_attrs {
// CHECK-NEXT:   public:
// CHECK-NEXT:    float fieldName0[1];
// CHECK-NEXT:    float fieldName1[1];
// CHECK-NEXT:    const std::map<std::string, char*> reflectionMap = { { "another_feature", reinterpret_cast<char*>(&fieldName0) } };
// CHECK-NEXT:    char* getBufferForName(std::string [[VAL_1:v[0-9]+]]) {
// CHECK-NEXT:      char* [[VAL_2:v[0-9]+]] = reflectionMap.at([[VAL_1]]);
// CHECK-NEXT:      return [[VAL_2]];
// CHECK-NEXT:    }
// CHECK-NEXT:    void bar() {
// CHECK-NEXT:      return;
// CHECK-NEXT:    }
// CHECK-NEXT:  };

// -----

/// Test that headers aren't added if the pass bails out.

emitc.class @negative_foo_no_attrs {
  emitc.field @fieldName0 : !emitc.array<1xf32>
  emitc.func @bar() {
    return
  }
}

// CHECK-NOT:   #include <map>
// CHECK-NOT:   #include <string>
// CHECK:       class negative_foo_no_attrs {
// CHECK-NEXT:   public:
// CHECK-NEXT:    float fieldName0[1];
// CHECK-NEXT:    void bar() {
// CHECK-NEXT:      return;
// CHECK-NEXT:    }
// CHECK-NEXT:  };
