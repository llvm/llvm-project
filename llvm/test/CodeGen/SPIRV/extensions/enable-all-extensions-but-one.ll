; RUN: llc -O0 -mtriple=spirv32-unknown-unknown --spirv-extensions=all,-SPV_INTEL_arbitrary_precision_integers %s -o - | FileCheck %s

define i6 @foo() {
  %call = tail call i32 @llvm.bitreverse.i32(i32 42)
  ret i6 2
}

; CHECK-NOT: OpExtension "SPV_INTEL_arbitrary_precision_integers"
; CHECK-DAG: OpExtension "SPV_KHR_bit_instructions"
