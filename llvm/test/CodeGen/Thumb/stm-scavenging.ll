; RUN: llc < %s | FileCheck %s
target triple = "thumbv6---gnueabi"

; Use STM to save the three registers
; CHECK-LABEL: use_stm:
; CHECK: .save   {r7, lr}
; CHECK: .setfp  r7, sp
; CHECK: stm r3!, {r0, r1, r2}
; CHECK: bl throws_1
define void @use_stm(i32 %a, i32 %b, i32 %c, i32* %d) local_unnamed_addr noreturn "no-frame-pointer-elim"="true" {
entry:
  %arrayidx = getelementptr inbounds i32, i32* %d, i32 2
  store i32 %a, i32* %arrayidx, align 4
  %arrayidx1 = getelementptr inbounds i32, i32* %d, i32 3
  store i32 %b, i32* %arrayidx1, align 4
  %arrayidx2 = getelementptr inbounds i32, i32* %d, i32 4
  store i32 %c, i32* %arrayidx2, align 4
  tail call void @throws_1(i32 %a, i32 %b, i32 %c) noreturn
  unreachable
}

; Don't use STM: there is no available register to store
; the address. We could transform this with some extra math, but
; that currently isn't implemented.
; CHECK-LABEL: no_stm:
; CHECK: .save   {r7, lr}
; CHECK: .setfp  r7, sp
; CHECK: str r0,
; CHECK: str r1,
; CHECK: str r2,
; CHECK: bl throws_2
define void @no_stm(i32 %a, i32 %b, i32 %c, i32* %d) local_unnamed_addr noreturn "no-frame-pointer-elim"="true" {
entry:
  %arrayidx = getelementptr inbounds i32, i32* %d, i32 2
  store i32 %a, i32* %arrayidx, align 4
  %arrayidx1 = getelementptr inbounds i32, i32* %d, i32 3
  store i32 %b, i32* %arrayidx1, align 4
  %arrayidx2 = getelementptr inbounds i32, i32* %d, i32 4
  store i32 %c, i32* %arrayidx2, align 4
  tail call void @throws_2(i32 %a, i32 %b, i32 %c, i32* %d) noreturn
  unreachable
}


declare void @throws_1(i32, i32, i32) noreturn
declare void @throws_2(i32, i32, i32, i32*) noreturn
