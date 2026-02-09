// RUN: mlir-translate -mlir-to-llvmir -split-input-file %s | FileCheck %s

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
llvm.func @denormal_fp_math_func_ieee() attributes { denormal_fpenv = #llvm.denormal_fpenv<default_output_mode = ieee, default_input_mode = ieee, float_output_mode = ieee, float_input_mode = ieee>}  {
  llvm.return
}
// CHECK: attributes #[[ATTRS]] = { denormal_fpenv(ieee) }

// -----

// CHECK-LABEL: define void @denormal_fp_math_f32_func_preserve_sign()
// CHECK-SAME: #[[ATTRS:[0-9]+]]
llvm.func @denormal_fp_math_f32_func_preserve_sign() attributes {denormal_fpenv = #llvm.denormal_fpenv<default_output_mode = ieee, default_input_mode = ieee, float_output_mode = preservesign, float_input_mode = preservesign>}  {
  llvm.return
}
// CHECK: attributes #[[ATTRS]] = { denormal_fpenv(float: preservesign) }

// -----

// CHECK-LABEL: define void @denormal_fp_math_dynamic_f32_func_preserve_sign()
// CHECK-SAME: #[[ATTRS:[0-9]+]]
llvm.func @denormal_fp_math_dynamic_f32_func_preserve_sign() attributes {denormal_fpenv = #llvm.denormal_fpenv<default_output_mode = dynamic, default_input_mode = dynamic, float_output_mode = preservesign, float_input_mode = preservesign>}  {
  llvm.return
}
// CHECK: attributes #[[ATTRS]] = { denormal_fpenv(dynamic, float: preservesign) }

// -----

// CHECK-LABEL: define void @denormal_fp_math_mixed_modes()
// CHECK-SAME: #[[ATTRS:[0-9]+]]
llvm.func @denormal_fp_math_mixed_modes() attributes {denormal_fpenv = #llvm.denormal_fpenv<default_output_mode = dynamic, default_input_mode = positivezero, float_output_mode = dynamic, float_input_mode = ieee>}  {
  llvm.return
}
// CHECK: attributes #[[ATTRS]] = { denormal_fpenv(dynamic|positivezero, float: dynamic|ieee) }

// -----

// CHECK-LABEL: define void @fp_contract_func_fast()
// CHECK-SAME: #[[ATTRS:[0-9]+]]
llvm.func @fp_contract_func_fast() attributes {fp_contract = "fast"}  {
  llvm.return
}
// CHECK: attributes #[[ATTRS]] = { "fp-contract"="fast" }
