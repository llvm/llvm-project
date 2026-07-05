; RUN: llc -asm-verbose=false < %s | FileCheck %s
; PR26063

target datalayout = "e-m:e-p:32:32-i64:64-v128:64:128-a:0:32-n32-S64"
target triple = "armv7--linux-gnueabihf"

; CHECK: .LBB0_1:
; CHECK-NEXT: bl      f{{$}}
; CHECK-NEXT: ldrb    r[[T0:[0-9]+]], [r{{[0-9]+}}, #1]!{{$}}
; CHECK-NEXT: cmp     r{{[0-9]+}}, #1{{$}}
; CHECK-NEXT: cmpne   r[[T0]], #0{{$}}
; CHECK-NEXT: bne     .LBB0_1{{$}}
define ptr @h(ptr readonly %a, i32 %b, i32 %c) {
entry:
  %0 = load i8, ptr %a, align 1
  %tobool4 = icmp ne i8 %0, 0
  %cmp5 = icmp ne i32 %b, 1
  %1 = and i1 %cmp5, %tobool4
  br i1 %1, label %while.body.preheader, label %while.end

while.body.preheader:                             ; preds = %entry
  br label %while.body

while.body:                                       ; preds = %while.body.preheader, %while.body
  %a.addr.06 = phi ptr [ %incdec.ptr, %while.body ], [ %a, %while.body.preheader ]
  %call = tail call i32 @f()
  %incdec.ptr = getelementptr inbounds i8, ptr %a.addr.06, i32 1
  %2 = load i8, ptr %incdec.ptr, align 1
  %tobool = icmp ne i8 %2, 0
  %cmp = icmp ne i32 %call, 1
  %3 = and i1 %cmp, %tobool
  br i1 %3, label %while.body, label %while.end.loopexit

while.end.loopexit:                               ; preds = %while.body
  %incdec.ptr.lcssa = phi ptr [ %incdec.ptr, %while.body ]
  br label %while.end

while.end:                                        ; preds = %while.end.loopexit, %entry
  %a.addr.0.lcssa = phi ptr [ %a, %entry ], [ %incdec.ptr.lcssa, %while.end.loopexit ]
  ret ptr %a.addr.0.lcssa
}

declare i32 @f(...)
