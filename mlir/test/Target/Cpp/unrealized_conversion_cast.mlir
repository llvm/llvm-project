// RUN: mlir-translate -mlir-to-cpp %s | FileCheck %s

// CHECK-LABEL: void builtin_cast
func.func @builtin_cast(%arg0: !emitc.ptr<f32>){
    // CHECK : float (*v2)[1][3][4][4] = (float (*)[1][3][4][4])v1
  %1 = builtin.unrealized_conversion_cast %arg0 : !emitc.ptr<f32> to !emitc.array<1x3x4x4xf32>
return
}

// CHECK-LABEL: void builtin_cast_index
func.func @builtin_cast_index(%arg0:  !emitc.size_t){
    // CHECK : size_t v2 = (size_t)v1
  %1 = builtin.unrealized_conversion_cast %arg0 : !emitc.size_t to index
return
}
