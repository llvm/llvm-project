// RUN: mlir-translate --mlir-to-cpp %s | FileCheck %s

emitc.class @modelClass {
    emitc.field @fieldName0 : !emitc.array<1xf32> 
    emitc.field @fieldName1 : !emitc.array<1xf32> 
    emitc.buffer_map ["another_feature", "some_feature", "output_0"]
    emitc.func @execute() {
        %0 = "emitc.constant"() <{value = 0 : index}> : () -> !emitc.size_t
        %1 = get_field @fieldName0 : !emitc.array<1xf32>
        %2 = get_field @fieldName1 : !emitc.array<1xf32>
        %3 = subscript %1[%0] : (!emitc.array<1xf32>, !emitc.size_t) -> !emitc.lvalue<f32>
        return
    }
}

// CHECK: class modelClass {
// CHECK-NEXT: public:
// CHECK-NEXT:  float[1] fieldName0;
// CHECK-NEXT:  float[1] fieldName1;
// CHECK-NEXT:   const std::map<std::string, char*> _buffer_map {
// CHECK-NEXT:     { ""another_feature"", reinterpret_cast<char*>(&"another_feature") },
// CHECK-NEXT:     { ""some_feature"", reinterpret_cast<char*>(&"some_feature") },
// CHECK-NEXT:     { ""output_0"", reinterpret_cast<char*>(&"output_0") },
// CHECK-NEXT:   };
// CHECK-NEXT:   char* getBufferForName(const std::string& name) const {
// CHECK-NEXT:     auto it = _buffer_map.find(name);
// CHECK-NEXT:     return (it == _buffer_map.end()) ? nullptr : it->second;
// CHECK-NEXT:   }
// CHECK-EMPTY: 
// CHECK-NEXT:  void execute() {
// CHECK-NEXT:    size_t v1 = 0;
// CHECK-NEXT:    float[1] v2 = fieldName0;
// CHECK-NEXT:    float[1] v3 = fieldName1;
// CHECK-NEXT:    return;
// CHECK-NEXT:  }
// CHECK-EMPTY:
// CHECK-NEXT: };
