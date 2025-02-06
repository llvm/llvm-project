// Most of the vector lowering is tested in vector-to-llvm.mlir, this file only for the interface smoke test
// RUN: mlir-opt --convert-to-llvm="filter-dialects=vector" --split-input-file %s | FileCheck %s

func.func @bitcast_f32_to_i32_vector_0d(%arg0: vector<f32>) -> vector<i32> {
  %0 = vector.bitcast %arg0 : vector<f32> to vector<i32>
  return %0 : vector<i32>
}

// CHECK-LABEL: @bitcast_f32_to_i32_vector_0d
// CHECK-SAME:  %[[ARG_0:.*]]: vector<f32>
// CHECK:       %[[VEC_F32_1D:.*]] = builtin.unrealized_conversion_cast %[[ARG_0]] : vector<f32> to vector<1xf32>
// CHECK:       %[[VEC_I32_1D:.*]] = llvm.bitcast %[[VEC_F32_1D]] : vector<1xf32> to vector<1xi32>
// CHECK:       %[[VEC_I32_0D:.*]] = builtin.unrealized_conversion_cast %[[VEC_I32_1D]] : vector<1xi32> to vector<i32>
// CHECK:       return %[[VEC_I32_0D]] : vector<i32>
