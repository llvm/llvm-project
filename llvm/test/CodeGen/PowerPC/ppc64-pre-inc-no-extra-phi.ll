; RUN: llc < %s -mtriple=powerpc64le-unknown-linux-gnu -mcpu=pwr9 -verify-machineinstrs | FileCheck %s
; RUN: llc < %s -mtriple=powerpc64-unknown-linux-gnu -mcpu=pwr9 -verify-machineinstrs | FileCheck %s
; RUN: llc < %s -mtriple=powerpc64le-unknown-linux-gnu -mcpu=pwr8 -verify-machineinstrs | FileCheck %s
; RUN: llc < %s -mtriple=powerpc64-unknown-linux-gnu -mcpu=pwr8 -verify-machineinstrs | FileCheck %s

@perm = local_unnamed_addr global [100 x i64] zeroinitializer, align 8

define void @sort_basket() local_unnamed_addr {
entry:
  br label %while.cond

while.cond:
  %l.0 = phi i64 [ 0, %entry ], [ %inc, %while.cond ]
  %arrayidx = getelementptr inbounds [100 x i64], [100 x i64]* @perm, i64 0, i64 %l.0
  %0 = load i64, i64* %arrayidx, align 8
  %cmp = icmp sgt i64 %0, 0
  %inc = add nuw nsw i64 %l.0, 1
  br i1 %cmp, label %while.cond, label %while.end

while.end:
  store i64 0, i64* %arrayidx, align 8
  ret void
; CHECK-LABEL: sort_basket
; CHECK: addi {{[0-9]+}}, {{[0-9]+}}, -8
; CHECK-NOT: addi {{[0-9]+}}, {{[0-9]+}}, 8
; CHECK: ldu {{[0-9]+}}, 8({{[0-9]+}})
; CHECK-NOT: addi {{[0-9]+}}, {{[0-9]+}}, 8
; CHECK: blr
}

