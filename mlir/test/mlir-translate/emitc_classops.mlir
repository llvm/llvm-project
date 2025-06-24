// RUN: mlir-translate --mlir-to-cpp %s | FileCheck %s

emitc.class @modelClass {
    emitc.field @input_tensor : !emitc.array<1xf32> 
    emitc.field @some_feature : !emitc.array<1xf32>  {emitc.opaque = ["some_feature"]}
    emitc.func @execute() {
        %0 = "emitc.constant"() <{value = 0 : index}> : () -> !emitc.size_t
        %1 = get_field @input_tensor : !emitc.array<1xf32>
        %2 = get_field @some_feature : !emitc.array<1xf32>
        %3 = subscript %1[%0] : (!emitc.array<1xf32>, !emitc.size_t) -> !emitc.lvalue<f32>
        return
    }
}

// CHECK: class modelClass final {
// CHECK-NEXT: public:
// CHECK-EMPTY:
// CHECK-NEXT:  const std::map<std::string, char*> _buffer_map {
// CHECK-NEXT:    { "input_tensor", reinterpret_cast<char*>(&None) },
// CHECK-NEXT:    { "some_feature", reinterpret_cast<char*>(&{emitc.opaque = ["some_feature"]}) }, 
// CHECK-NEXT:  };
// CHECK-NEXT:  char* getBufferForName(const std::string& name) const {
// CHECK-NEXT:     auto it = _buffer_map.find(name);
// CHECK-NEXT:     return (it == _buffer_map.end()) ? nullptr : it->second;
// CHECK-NEXT:  }
// CHECK-EMPTY:
// CHECK-NEXT:  float[1] input_tensor;
// CHECK-NEXT:  float[1] some_feature;
// CHECK-NEXT:  void execute() {
// CHECK-NEXT:    size_t v1 = 0;
// CHECK-NEXT:    float[1] v2 = input_tensor;
// CHECK-NEXT:    float[1] v3 = some_feature;
// CHECK-NEXT:    return;
// CHECK-NEXT:  }
// CHECK-EMPTY:
// CHECK-NEXT: };
