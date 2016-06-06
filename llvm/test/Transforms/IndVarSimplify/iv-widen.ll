; RUN: opt < %s -indvars -S | FileCheck %s
; RUN: opt -lcssa -loop-simplify -S < %s | opt -S -passes='require<targetir>,require<scalar-evolution>,require<domtree>,loop(indvars)'

; Provide legal integer types.
target datalayout = "n8:16:32:64"


target triple = "x86_64-apple-darwin"

; CHECK-LABEL: @loop_0
; CHECK-LABEL: B18:
; Only one phi now.
; CHECK: phi
; CHECK-NOT: phi
; One trunc for the gep.
; CHECK: trunc i64 %indvars.iv to i32
; One trunc for the dummy() call.
; CHECK-LABEL: exit24:
; CHECK: trunc i64 {{.*}}lcssa.wide to i32
define void @loop_0(i32* %a) {
Prologue:
  br i1 undef, label %B18, label %B6

B18:                                        ; preds = %B24, %Prologue
  %.02 = phi i32 [ 0, %Prologue ], [ %tmp33, %B24 ]
  %tmp23 = zext i32 %.02 to i64
  %tmp33 = add i32 %.02, 1
  %o = getelementptr i32, i32* %a, i32 %.02
  %v = load i32, i32* %o
  %t = icmp eq i32 %v, 0
  br i1 %t, label %exit24, label %B24

B24:                                        ; preds = %B18
  %t2 = icmp eq i32 %tmp33, 20
  br i1 %t2, label %B6, label %B18

B6:                                       ; preds = %Prologue
  ret void

exit24:                      ; preds = %B18
  call void @dummy(i32 %.02)
  unreachable
}

define void @loop_1(i32 %lim) {
; CHECK-LABEL: @loop_1(
 entry:
  %entry.cond = icmp ne i32 %lim, 0
  br i1 %entry.cond, label %loop, label %leave

 loop:
; CHECK: loop:
; CHECK:  %indvars.iv = phi i64 [ 1, %loop.preheader ], [ %indvars.iv.next, %loop ]
; CHECK:  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
; CHECK:  [[IV_INC:%[^ ]+]] = add nsw i64 %indvars.iv, -1
; CHECK:  call void @dummy.i64(i64 [[IV_INC]])

  %iv = phi i32 [ 1, %entry ], [ %iv.inc, %loop ]
  %iv.inc = add i32 %iv, 1
  %iv.inc.sub = add i32 %iv, -1
  %iv.inc.sub.zext = zext i32 %iv.inc.sub to i64
  call void @dummy.i64(i64 %iv.inc.sub.zext)
  %be.cond = icmp ult i32 %iv.inc, %lim
  br i1 %be.cond, label %loop, label %leave

 leave:
  ret void
}

declare void @dummy(i32)
declare void @dummy.i64(i64)
