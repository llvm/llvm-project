// RUN: mlir-translate -mlir-to-llvmir -split-input-file %s | FileCheck %s

// CHECK-LABEL: define void @unsafe_fp_math_func()
// CHECK-SAME: #[[ATTRS:[0-9]+]]
llvm.func @unsafe_fp_math_func() attributes {unsafe_fp_math = true}  {
  llvm.return
}
// CHECK: attributes #[[ATTRS]] = { "unsafe-fp-math"="true" }

// -----

// CHECK-LABEL: define void @no_infs_fp_math_func()
// CHECK-SAME: #[[ATTRS:[0-9]+]]
llvm.func @no_infs_fp_math_func() attributes {no_infs_fp_math = true}  {
  llvm.return
}
// CHECK: attributes #[[ATTRS]] = { "no-infs-fp-math"="true" }

// -----

// CHECK-LABEL: define void @no_nans_fp_math_func()
// CHECK-SAME: #[[ATTRS:[0-9]+]]
llvm.func @no_nans_fp_math_func() attributes {no_nans_fp_math = true}  {
  llvm.return
}
// CHECK: attributes #[[ATTRS]] = { "no-nans-fp-math"="true" }

// -----

// CHECK-LABEL: define void @approx_func_fp_math_func()
// CHECK-SAME: #[[ATTRS:[0-9]+]]
llvm.func @approx_func_fp_math_func() attributes {approx_func_fp_math = true}  {
  llvm.return
}
// CHECK: attributes #[[ATTRS]] = { "approx-func-fp-math"="true" }

// -----

// CHECK-LABEL: define void @no_signed_zeros_fp_math_func()
// CHECK-SAME: #[[ATTRS:[0-9]+]]
llvm.func @no_signed_zeros_fp_math_func() attributes {no_signed_zeros_fp_math = true}  {
  llvm.return
}
// CHECK: attributes #[[ATTRS]] = { "no-signed-zeros-fp-math"="true" }
