// RUN: mlir-translate -verify-diagnostics -split-input-file -mlir-to-llvmir %s

// -----

func.func @nvvm_invalid_shfl_pred(%arg0 : i32, %arg1 : f32, %arg2 : i32, %arg3 : i32) {
  // expected-error@+1 {{"return_value_and_is_valid" attribute must be specified when returning the predicate}}
  %0 = nvvm.shfl.sync bfly %arg0, %arg1, %arg2, %arg3 : f32 -> !llvm.struct<(f32, i1)>
}
