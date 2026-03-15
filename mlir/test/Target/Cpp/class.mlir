// RUN: mlir-translate -mlir-to-cpp %s | FileCheck %s

emitc.class @modelClass {
  emitc.field @fieldName0 : !emitc.array<1xf32>
  emitc.field @fieldName1 : !emitc.array<1xf32>
  emitc.func @execute() {
    %0 = "emitc.constant"() <{value = 0 : index}> : () -> !emitc.size_t
    %1 = get_field @fieldName0 : !emitc.array<1xf32>
    %2 = get_field @fieldName1 : !emitc.array<1xf32>
    %3 = subscript %1[%0] : (!emitc.array<1xf32>, !emitc.size_t) -> !emitc.lvalue<f32>
    return
  }
}

// CHECK-LABEL: class modelClass {
// CHECK-NEXT:   public:
// CHECK-NEXT:    float fieldName0[1];
// CHECK-NEXT:    float fieldName1[1];
// CHECK-NEXT:    void execute() {
// CHECK-NEXT:     size_t v1 = 0;
// CHECK-NEXT:     return;
// CHECK-NEXT:    }
// CHECK-NEXT:  };

emitc.class final @finalClass {
  emitc.field @fieldName0 : !emitc.array<1xf32>
  emitc.field @fieldName1 : !emitc.array<1xf32>
  emitc.func @execute() {
    %0 = "emitc.constant"() <{value = 0 : index}> : () -> !emitc.size_t
    %1 = get_field @fieldName0 : !emitc.array<1xf32>
    %2 = get_field @fieldName1 : !emitc.array<1xf32>
    %3 = subscript %1[%0] : (!emitc.array<1xf32>, !emitc.size_t) -> !emitc.lvalue<f32>
    return
  }
}

// CHECK-LABEL: class finalClass final {
// CHECK-NEXT:   public:
// CHECK-NEXT:    float fieldName0[1];
// CHECK-NEXT:    float fieldName1[1];
// CHECK-NEXT:    void execute() {
// CHECK-NEXT:     size_t v1 = 0;
// CHECK-NEXT:     return;
// CHECK-NEXT:    }
// CHECK-NEXT:  };

emitc.class @mainClass {
  emitc.field @fieldName0 : !emitc.array<2xf32> = dense<0.0> {attrs = {emitc.name_hint = "another_feature"}}
  emitc.func @get_fieldName0() {
    %0 = emitc.get_field @fieldName0 : !emitc.array<2xf32>
    return
  }
}

// CHECK-LABEL: class mainClass {
// CHECK-NEXT:   public:
// CHECK-NEXT:    float fieldName0[2] = {0.0e+00f, 0.0e+00f};
// CHECK-NEXT:    void get_fieldName0() {
// CHECK-NEXT:     return;
// CHECK-NEXT:    }
// CHECK-NEXT:  };

emitc.class @reflectionClass {
  emitc.field @reflectionMap : !emitc.opaque<"const std::map<std::string, std::string>"> = #emitc.opaque<"{ { \22another_feature\22, \22fieldName0\22 } }">
  emitc.func @get_reflectionMap() {
    %0 = emitc.get_field @reflectionMap : !emitc.opaque<"const std::map<std::string, std::string>">
    return
  }
}

// CHECK-LABEL: class reflectionClass {
// CHECK-NEXT:   public:
// CHECK-NEXT:    const std::map<std::string, std::string> reflectionMap = { { "another_feature", "fieldName0" } };
// CHECK-NEXT:    void get_reflectionMap() {
// CHECK-NEXT:     return;
// CHECK-NEXT:    }
// CHECK-NEXT:  };

