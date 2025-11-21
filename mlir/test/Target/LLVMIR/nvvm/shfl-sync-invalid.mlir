// RUN: mlir-translate -verify-diagnostics -split-input-file -mlir-to-llvmir %s

// -----

func.func @nvvm_invalid_shfl_pred(%arg0 : i32, %arg1 : f32, %arg2 : i32, %arg3 : i32) {
  // expected-error@+1 {{"return_value_and_is_valid" attribute must be specified when the return type is a struct type}}
  %0 = nvvm.shfl.sync bfly %arg0, %arg1, %arg2, %arg3 : f32 -> !llvm.struct<(f32, i1)>
}

// -----

func.func @nvvm_invalid_shfl_invalid_return_type_1(%arg0 : i32, %arg1 : f32, %arg2 : i32, %arg3 : i32) {
  // expected-error@+1 {{expected return type to be of type 'f32' but got 'i32' instead}}
  %0 = nvvm.shfl.sync bfly %arg0, %arg1, %arg2, %arg3 : f32 -> i32
}

// -----

func.func @nvvm_invalid_shfl_invalid_return_type_2(%arg0 : i32, %arg1 : f32, %arg2 : i32, %arg3 : i32) {
  // expected-error@+1 {{expected first element in the returned struct to be of type 'f32' but got 'i32' instead}}
  %0 = nvvm.shfl.sync bfly %arg0, %arg1, %arg2, %arg3 {return_value_and_is_valid} : f32 -> !llvm.struct<(i32, i1)>
}
