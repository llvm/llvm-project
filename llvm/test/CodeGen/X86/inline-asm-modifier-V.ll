; RUN: llc < %s -mtriple=i686-- -no-integrated-as | FileCheck -check-prefix=X86 %s
; RUN: llc < %s -mtriple=x86_64-- -no-integrated-as | FileCheck -check-prefix=X64 %s

; If the target does not have 64-bit integer registers, emit 32-bit register
; names.

; X86: call __x86_indirect_thunk_e{{[abcd]}}x
; X64: call __x86_indirect_thunk_r

define void @q_modifier(ptr %p) {
entry:
  tail call void asm sideeffect "call __x86_indirect_thunk_${0:V}", "r,~{dirflag},~{fpsr},~{flags}"(ptr %p)
  ret void
}
