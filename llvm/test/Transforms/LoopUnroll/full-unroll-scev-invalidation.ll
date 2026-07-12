; RUN: opt -S -passes='loop(indvars,loop-unroll-full,indvars),simplifycfg' %s | FileCheck %s

@__chk = external global i64

define void @f4(i64 %0) {
; CHECK-LABEL: define void @f4(
; CHECK: store i64 5, ptr @__chk
entry:
  br label %lbl_b5

lbl_b5:
  br label %lbl_b10

lbl_b10:
  %1 = phi i8 [ 0, %lbl_b5 ], [ 1, %lbl_b10 ]
  %2 = phi i64 [ %0, %lbl_b5 ], [ 0, %lbl_b10 ]
  %bb13.1 = phi i32 [ 0, %lbl_b5 ], [ %3, %lbl_b10 ]
  %conv1 = trunc i64 %2 to i32
  %3 = call i32 @llvm.ctpop.i32(i32 %conv1)
  %loadedv = trunc i8 %1 to i1
  br i1 %loadedv, label %if.then3, label %lbl_b10

if.then3:
  %tobool4.not = icmp eq i32 %bb13.1, 0
  br i1 %tobool4.not, label %common.ret, label %if.then8

if.then8:
  br i1 true, label %if.then10, label %lbl_b5

common.ret:
  ret void

if.then10:
  %.lcssa.lcssa34 = phi i32 [ %3, %if.then8 ]
  store i64 5, ptr @__chk
  ret void
}

declare i32 @llvm.ctpop.i32(i32)
