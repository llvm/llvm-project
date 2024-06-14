; RUN: llc < %s -mtriple=x86_64-apple-darwin11 -no-integrated-as | FileCheck %s
; Any register is OK for %0, but it must be a register, not memory.

define i32 @foo() nounwind ssp {
entry:
; CHECK: GCROOT %eax
  %_r = alloca i32, align 4                       ; <ptr> [#uses=2]
  call void asm "/* GCROOT $0 */", "=*imr,0,~{dirflag},~{fpsr},~{flags}"(ptr elementtype(i32) %_r, i32 4) nounwind
  %0 = load i32, ptr %_r, align 4                     ; <i32> [#uses=1]
  ret i32 %0
}
