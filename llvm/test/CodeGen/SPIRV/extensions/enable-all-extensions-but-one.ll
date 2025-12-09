; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv32-unknown-unknown --spirv-ext=all,-SPV_ALTERA_arbitrary_precision_integers %s -o - | FileCheck %s
; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv32-unknown-unknown --spirv-ext=-SPV_ALTERA_arbitrary_precision_integers,all %s -o - | FileCheck %s
; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv32-unknown-unknown --spirv-ext=KHR %s -o - | FileCheck %s
; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv32-unknown-unknown --spirv-ext=khr %s -o - | FileCheck %s

@G = global i32 0

define i6 @foo() {
  %call = tail call i32 @llvm.bitreverse.i32(i32 42)
  store i32 %call, ptr @G
  ret i6 2
}

; CHECK-NOT: OpExtension "SPV_ALTERA_arbitrary_precision_integers"
; CHECK-DAG: OpExtension "SPV_KHR_bit_instructions"

declare i32 @llvm.bitreverse.i32(i32)
