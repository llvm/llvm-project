; RUN: opt < %s -loop-reduce -S | FileCheck %s

target datalayout = "e-m:e-p:32:32-i64:64-v128:64:128-a:0:32-n32-S64"
target triple = "thumbv7m-arm-none-eabi"

; Check that the IV updates (incdec.ptr{,1,2}) are kept in the latch block
; and not moved to the header/exiting block. Inserting them in the header
; doubles register pressure and adds moves.

; CHECK-LABEL: @f
; CHECK: while.cond:
; CHECK: icmp sgt i32 %n.addr.0, 0
; CHECK: while.body:
; CHECK: incdec.ptr =
; CHECK: incdec.ptr1 =
; CHECK: incdec.ptr2 =
; CHECK: dec = 
define void @f(ptr nocapture readonly %a, ptr nocapture readonly %b, ptr nocapture %c, i32 %n) {
entry:
  br label %while.cond

while.cond:                                       ; preds = %while.body, %entry
  %a.addr.0 = phi ptr [ %a, %entry ], [ %incdec.ptr, %while.body ]
  %b.addr.0 = phi ptr [ %b, %entry ], [ %incdec.ptr1, %while.body ]
  %c.addr.0 = phi ptr [ %c, %entry ], [ %incdec.ptr2, %while.body ]
  %n.addr.0 = phi i32 [ %n, %entry ], [ %dec, %while.body ]
  %cmp = icmp sgt i32 %n.addr.0, 0
  br i1 %cmp, label %while.body, label %while.end

while.body:                                       ; preds = %while.cond
  %incdec.ptr = getelementptr inbounds float, ptr %a.addr.0, i32 1
  %tmp = load float, ptr %a.addr.0, align 4
  %incdec.ptr1 = getelementptr inbounds float, ptr %b.addr.0, i32 1
  %tmp1 = load float, ptr %b.addr.0, align 4
  %add = fadd float %tmp, %tmp1
  %incdec.ptr2 = getelementptr inbounds float, ptr %c.addr.0, i32 1
  store float %add, ptr %c.addr.0, align 4
  %dec = add nsw i32 %n.addr.0, -1
  br label %while.cond

while.end:                                        ; preds = %while.cond
  ret void
}
