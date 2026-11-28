; RUN: llc < %s -verify-machineinstrs -mtriple=x86_64-unknown-linux-gnu -relocation-model=pic | FileCheck %s --check-prefix=X64-PIC
; RUN: llc < %s -verify-machineinstrs -mtriple=x86_64-unknown-linux-gnu -relocation-model=static | FileCheck %s --check-prefix=X64-STATIC

; Verify that X86SpeculativeLoadHardening traces the indirect branch target
; address through JMP64r_NT (notrack jmpq *%reg), emitted for jump tables when
; cf-protection-branch (IBT) is enabled.

define i32 @switch_slh_ibt(i32 %x, ptr %p) speculative_load_hardening nounwind {
; X64-PIC-LABEL: switch_slh_ibt:
; X64-PIC:       notrack jmpq *[[REG:%r[a-z0-9]+]]
; X64-PIC:       [[L0:\.LBB0_[0-9]+]]:
; X64-PIC:       leaq [[L0]](%rip), [[TMP:%r[a-z0-9]+]]
; X64-PIC-NEXT:  cmpq [[TMP]], [[REG]]
; X64-PIC-NEXT:  cmovneq
; X64-PIC:       [[L1:\.LBB0_[0-9]+]]:
; X64-PIC:       leaq [[L1]](%rip), [[TMP2:%r[a-z0-9]+]]
; X64-PIC-NEXT:  cmpq [[TMP2]], [[REG]]
; X64-PIC-NEXT:  cmovneq
;
; X64-STATIC-LABEL: switch_slh_ibt:
; X64-STATIC:       notrack jmpq *[[REG:%r[a-z0-9]+]]
; X64-STATIC:       [[L0:\.LBB0_[0-9]+]]:
; X64-STATIC:       cmpq $[[L0]], [[REG]]
; X64-STATIC-NEXT:  cmovneq
; X64-STATIC:       [[L1:\.LBB0_[0-9]+]]:
; X64-STATIC:       cmpq $[[L1]], [[REG]]
; X64-STATIC-NEXT:  cmovneq
entry:
  switch i32 %x, label %def [
    i32 0, label %b0
    i32 1, label %b1
    i32 2, label %b2
    i32 3, label %b3
    i32 4, label %b4
  ]
b0:
  %v0 = load i32, ptr %p, align 4
  ret i32 %v0
b1:
  %g1 = getelementptr i32, ptr %p, i64 1
  %v1 = load i32, ptr %g1, align 4
  ret i32 %v1
b2:
  %g2 = getelementptr i32, ptr %p, i64 2
  %v2 = load i32, ptr %g2, align 4
  ret i32 %v2
b3:
  %g3 = getelementptr i32, ptr %p, i64 3
  %v3 = load i32, ptr %g3, align 4
  ret i32 %v3
b4:
  %g4 = getelementptr i32, ptr %p, i64 4
  %v4 = load i32, ptr %g4, align 4
  ret i32 %v4
def:
  ret i32 -1
}

!llvm.module.flags = !{!0}
!0 = !{i32 8, !"cf-protection-branch", i32 1}
