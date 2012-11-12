; RUN: llc -march=x86-64 < %s | FileCheck %s
; 7282062
; ModuleID = '<stdin>'
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128"
target triple = "x86_64-apple-darwin10.0"

define void @udiv8(i8* %quotient, i16 zeroext %a, i8 zeroext %b, i8 zeroext %c, i8* %remainder) nounwind ssp {
entry:
; CHECK: udiv8:
; CHECK-NOT: movb %ah, (%r8)
  %a_addr = alloca i16, align 2                   ; <i16*> [#uses=2]
  %b_addr = alloca i8, align 1                    ; <i8*> [#uses=2]
  store i16 %a, i16* %a_addr
  store i8 %b, i8* %b_addr
  call void asm "\09\09movw\09$2, %ax\09\09\0A\09\09divb\09$3\09\09\09\0A\09\09movb\09%al, $0\09\0A\09\09movb %ah, ($4)", "=*m,=*m,*m,*m,R,~{dirflag},~{fpsr},~{flags},~{ax}"(i8* %quotient, i8* %remainder, i16* %a_addr, i8* %b_addr, i8* %remainder) nounwind
  ret void
; CHECK: ret
}
