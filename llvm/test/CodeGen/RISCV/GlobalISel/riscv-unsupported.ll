; RUN: llc -mtriple=riscv32 -verify-machineinstrs -global-isel -global-isel-abort=2 -pass-remarks-missed='gisel*' %s -o - 2>&1 | FileCheck %s -check-prefixes=CHECK
; RUN: llc -mtriple=riscv64 -verify-machineinstrs -global-isel -global-isel-abort=2 -pass-remarks-missed='gisel*' %s -o - 2>&1 | FileCheck %s -check-prefixes=CHECK

; This file checks that we use the fallback path for things that are known to
; be unsupported on the RISC-V target. It should progressively shrink in size.

%byval.class = type { i32 }

declare void @test_byval_arg(ptr byval(%byval.class) %x)

define void @test_byval_param(ptr %x) {
; CHECK: remark: {{.*}} unable to translate instruction: call
; CHECK-LABEL: warning: Instruction selection used fallback path for test_byval_param
  call void @test_byval_arg(ptr byval(%byval.class) %x)
  ret void
}

define i128 @inline_asm_multi_reg_input(i128 %x) {
; CHECK: remark: {{.*}} unable to translate instruction: call
; CHECK-LABEL: warning: Instruction selection used fallback path for inline_asm_multi_reg_input
  %r = call i128 asm sideeffect "/* $0 $1 */", "=&r,r"(i128 %x)
  ret i128 %r
}
