; RUN: opt -passes=lower-expect  -S -o - < %s | FileCheck %s
; RUN: opt -S -passes='function(lower-expect)' < %s | FileCheck %s

; return __builtin_expect((a > b ? 1, goo(), 0);
;
; Function Attrs: noinline nounwind uwtable
define i32 @foo(i32 %arg, i32 %arg1)  {
; CHECK-LABEL: i32 @foo
bb:
  %tmp5 = icmp sgt i32 %arg, %arg1
  br i1 %tmp5, label %bb9, label %bb7
; CHECK: br i1 %tmp5{{.*}}!prof [[WEIGHT:![0-9]+]]

bb7:                                              ; preds = %bb
  %tmp8 = call i32 @goo()
  br label %bb9

bb9:                                              ; preds = %bb7, %bb9
  %tmp10 = phi i32 [ 1, %bb ], [ %tmp8, %bb7 ]
  %tmp11 = sext i32 %tmp10 to i64
  %expect = call i64 @llvm.expect.i64(i64 %tmp11, i64 0)
  %tmp12 = trunc i64 %expect to i32
  ret i32 %tmp12
}

define i32 @foo2(i32 %arg, i32 %arg1)  {
bb:
  %tmp5 = icmp sgt i32 %arg, %arg1
  br i1 %tmp5, label %bb6, label %bb7
; CHECK: br i1 %tmp5{{.*}}!prof [[WEIGHT:![0-9]+]]

bb6:                                              ; preds = %bb
  br label %bb9

bb7:                                              ; preds = %bb
  %tmp8 = call i32 @goo()
  br label %bb9

bb9:                                              ; preds = %bb7, %bb6
  %tmp10 = phi i32 [ 1, %bb6 ], [ %tmp8, %bb7 ]
  %tmp11 = sext i32 %tmp10 to i64
  %expect = call i64 @llvm.expect.i64(i64 %tmp11, i64 0)
  %tmp12 = trunc i64 %expect to i32
  ret i32 %tmp12
}

declare i32 @goo()
declare i64 @llvm.expect.i64(i64, i64)



!llvm.ident = !{!0}

!0 = !{!"clang version 5.0.0 (trunk 302965)"}

; CHECK: [[WEIGHT]] = !{!"branch_weights", i32 1, i32 2000}
