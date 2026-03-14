; Test that this test case does not abort after the folding of load -> add ->
; store into an alsi. This folding is suppose to not happen as it would
; introduce a loop in the DAG.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z13 -disable-basic-aa -consthoist-gep | FileCheck %s

@g_295 = external dso_local unnamed_addr global i32, align 4
@g_672 = external dso_local unnamed_addr global i64, align 8
@g_1484 = external dso_local global <{ i8, i64, { i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i32, i8, i8, [2 x i8], i8, i8, i8, i8, i8, i8, i8, i8, i32, i8, i8, i8 }, i32 }>, align 2

define void @fun() {
; CHECK-LABEL: fun:

bb:
  br label %bb1

bb1:                                              ; preds = %bb1, %bb
  store i32 2, ptr getelementptr inbounds (<{ i8, i64, { i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i32, i8, i8, [2 x i8], i8, i8, i8, i8, i8, i8, i8, i8, i32, i8, i8, i8 }, i32 }>, ptr @g_1484, i64 0, i32 2, i32 16)
  %tmp = icmp slt i32 undef, 3
  br i1 %tmp, label %bb1, label %bb2

bb2:                                              ; preds = %bb1
  %tmp3 = load i32, ptr getelementptr inbounds (<{ i8, i64, { i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i32, i8, i8, [2 x i8], i8, i8, i8, i8, i8, i8, i8, i8, i32, i8, i8, i8 }, i32 }>, ptr @g_1484, i64 0, i32 2, i32 28)
  %tmp4 = load i64, ptr @g_672
  %tmp5 = add i64 %tmp4, 1
  store i64 %tmp5, ptr @g_672
  %tmp6 = icmp eq i64 %tmp5, 0
  %tmp7 = zext i1 %tmp6 to i32
  %tmp8 = icmp ult i32 %tmp3, %tmp7
  %tmp9 = zext i1 %tmp8 to i32
  store i32 %tmp9, ptr @g_295
  ret void
}

