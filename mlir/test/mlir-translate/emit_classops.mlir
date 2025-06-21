// RUN: mlir-translate --mlir-to-cpp %s | FileCheck %s

emitc.class @modelClass {
    emitc.field @input_tensor : !emitc.array<1xf32> 
    emitc.func @execute() {
        %0 = "emitc.constant"() <{value = 0 : index}> : () -> !emitc.size_t
        %1 = get_field @input_tensor : !emitc.array<1xf32>
        %2 = subscript %1[%0] : (!emitc.array<1xf32>, !emitc.size_t) -> !emitc.lvalue<f32>
        return
    }
}

// CHECK: class modelClass final {
// CHECK-NEXT: public:
// CHECK-NEXT:const std::map<std::string, char*> buffer_map {
// CHECK-NEXT:    { "input_tensor", reinterpret_cast<char*>(&input_tensor) },
// CHECK-NEXT:  };
// CHECK-NEXT:  char* getBufferForName(const std::string& name) const {
// CHECK-NEXT:    auto it = buffer_map.find(name);
// CHECK-NEXT:    return (it == buffer_map.end()) ? nullptr : it->second;
// CHECK-NEXT:  }
// CHECK-NEXT:  float input_tensor[1];
// CHECK-NEXT:  void execute() {
// CHECK-NEXT:    size_t v1 = 0;
// CHECK-NEXT:    float v2[1] = input_tensor;
// CHECK-NEXT:    return;
// CHECK-NEXT:  }
// CHECK-EMPTY:
// CHECK-NEXT: };
