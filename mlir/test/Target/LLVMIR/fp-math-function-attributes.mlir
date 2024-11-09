// RUN: mlir-translate -mlir-to-llvmir -split-input-file %s | FileCheck %s

// CHECK-LABEL: define void @unsafe_fp_math_func_true()
// CHECK-SAME: #[[ATTRS:[0-9]+]]
llvm.func @unsafe_fp_math_func_true() attributes {unsafe_fp_math = true}  {
  llvm.return
}
// CHECK: attributes #[[ATTRS]] = { "unsafe-fp-math"="true" }

// -----

// CHECK-LABEL: define void @unsafe_fp_math_func_false()
// CHECK-SAME: #[[ATTRS:[0-9]+]]
llvm.func @unsafe_fp_math_func_false() attributes {unsafe_fp_math = false}  {
  llvm.return
}
// CHECK: attributes #[[ATTRS]] = { "unsafe-fp-math"="false" }

// -----

// CHECK-LABEL: define void @no_infs_fp_math_func_true()
// CHECK-SAME: #[[ATTRS:[0-9]+]]
llvm.func @no_infs_fp_math_func_true() attributes {no_infs_fp_math = true}  {
  llvm.return
}
// CHECK: attributes #[[ATTRS]] = { "no-infs-fp-math"="true" }

// -----

// CHECK-LABEL: define void @no_infs_fp_math_func_false()
// CHECK-SAME: #[[ATTRS:[0-9]+]]
llvm.func @no_infs_fp_math_func_false() attributes {no_infs_fp_math = false}  {
  llvm.return
}
// CHECK: attributes #[[ATTRS]] = { "no-infs-fp-math"="false" }

// -----

// CHECK-LABEL: define void @no_nans_fp_math_func_true()
// CHECK-SAME: #[[ATTRS:[0-9]+]]
llvm.func @no_nans_fp_math_func_true() attributes {no_nans_fp_math = true}  {
  llvm.return
}
// CHECK: attributes #[[ATTRS]] = { "no-nans-fp-math"="true" }

// -----

// CHECK-LABEL: define void @no_nans_fp_math_func_false()
// CHECK-SAME: #[[ATTRS:[0-9]+]]
llvm.func @no_nans_fp_math_func_false() attributes {no_nans_fp_math = false}  {
  llvm.return
}
// CHECK: attributes #[[ATTRS]] = { "no-nans-fp-math"="false" }

// -----

// CHECK-LABEL: define void @approx_func_fp_math_func_true()
// CHECK-SAME: #[[ATTRS:[0-9]+]]
llvm.func @approx_func_fp_math_func_true() attributes {approx_func_fp_math = true}  {
  llvm.return
}
// CHECK: attributes #[[ATTRS]] = { "approx-func-fp-math"="true" }

// -----
//
// CHECK-LABEL: define void @approx_func_fp_math_func_false()
// CHECK-SAME: #[[ATTRS:[0-9]+]]
llvm.func @approx_func_fp_math_func_false() attributes {approx_func_fp_math = false}  {
  llvm.return
}
// CHECK: attributes #[[ATTRS]] = { "approx-func-fp-math"="false" }

// -----

// CHECK-LABEL: define void @no_signed_zeros_fp_math_func_true()
// CHECK-SAME: #[[ATTRS:[0-9]+]]
llvm.func @no_signed_zeros_fp_math_func_true() attributes {no_signed_zeros_fp_math = true}  {
  llvm.return
}
// CHECK: attributes #[[ATTRS]] = { "no-signed-zeros-fp-math"="true" }

// -----

// CHECK-LABEL: define void @no_signed_zeros_fp_math_func_false()
// CHECK-SAME: #[[ATTRS:[0-9]+]]
llvm.func @no_signed_zeros_fp_math_func_false() attributes {no_signed_zeros_fp_math = false}  {
  llvm.return
}
// CHECK: attributes #[[ATTRS]] = { "no-signed-zeros-fp-math"="false" }

// -----

// CHECK-LABEL: define void @denormal_fp_math_func_ieee()
// CHECK-SAME: #[[ATTRS:[0-9]+]]
llvm.func @denormal_fp_math_func_ieee() attributes {denormal_fp_math = "ieee"}  {
  llvm.return
}
// CHECK: attributes #[[ATTRS]] = { "denormal-fp-math"="ieee" }

// -----

// CHECK-LABEL: define void @denormal_fp_math_f32_func_preserve_sign()
// CHECK-SAME: #[[ATTRS:[0-9]+]]
llvm.func @denormal_fp_math_f32_func_preserve_sign() attributes {denormal_fp_math_f32 = "preserve-sign"}  {
  llvm.return
}
// CHECK: attributes #[[ATTRS]] = { "denormal-fp-math-f32"="preserve-sign" }

// -----

// CHECK-LABEL: define void @fp_contract_func_fast()
// CHECK-SAME: #[[ATTRS:[0-9]+]]
llvm.func @fp_contract_func_fast() attributes {fp_contract = "fast"}  {
  llvm.return
}
// CHECK: attributes #[[ATTRS]] = { "fp-contract"="fast" }
