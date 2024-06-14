; RUN: llc < %s | FileCheck %s
; 7282062
; ModuleID = '<stdin>'
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128"
target triple = "x86_64-apple-darwin10.0"

define void @udiv8(ptr %quotient, i16 zeroext %a, i8 zeroext %b, i8 zeroext %c, ptr %remainder) nounwind ssp {
entry:
; CHECK-LABEL: udiv8:
; CHECK-NOT: movb %ah, (%r8)
  %a_addr = alloca i16, align 2                   ; <ptr> [#uses=2]
  %b_addr = alloca i8, align 1                    ; <ptr> [#uses=2]
  store i16 %a, ptr %a_addr
  store i8 %b, ptr %b_addr
  call void asm "\09\09movw\09$2, %ax\09\09\0A\09\09divb\09$3\09\09\09\0A\09\09movb\09%al, $0\09\0A\09\09movb %ah, ($4)", "=*m,=*m,*m,*m,R,~{dirflag},~{fpsr},~{flags},~{ax}"(ptr elementtype(i8) %quotient, ptr elementtype(i8) %remainder, ptr elementtype(i16) %a_addr, ptr elementtype(i8) %b_addr, ptr %remainder) nounwind
  ret void
; CHECK: ret
}
