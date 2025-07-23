// RUN: mlir-translate --mlir-to-cpp %s | FileCheck %s

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
// CHECK-NEXT: public:
// CHECK-NEXT:  float[1] fieldName0;
// CHECK-NEXT:  float[1] fieldName1;
// CHECK-NEXT:  void execute() {
// CHECK-NEXT:    size_t v1 = 0;
// CHECK-NEXT:    float[1] v2 = fieldName0;
// CHECK-NEXT:    float[1] v3 = fieldName1;
// CHECK-NEXT:    return;
// CHECK-NEXT:  }
// CHECK-EMPTY:
// CHECK-NEXT: };

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
// CHECK-NEXT: public:
// CHECK-NEXT:  float[1] fieldName0;
// CHECK-NEXT:  float[1] fieldName1;
// CHECK-NEXT:  void execute() {
// CHECK-NEXT:    size_t v1 = 0;
// CHECK-NEXT:    float[1] v2 = fieldName0;
// CHECK-NEXT:    float[1] v3 = fieldName1;
// CHECK-NEXT:    return;
// CHECK-NEXT:  }
// CHECK-EMPTY:
// CHECK-NEXT: };
