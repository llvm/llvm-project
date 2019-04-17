; RUN: opt -lower-expect  -S -o - < %s | FileCheck %s
; RUN: opt -S -passes='function(lower-expect)' < %s | FileCheck %s

define i32 @foo(i32 %arg) #0 {
; CHECK-LABEL: @foo(i32{{.*}})
bb:
  %tmp = sext i32 %arg to i64
  %tmp1 = call i64 @llvm.expect.i64(i64 %tmp, i64 4)
  %tmp2 = icmp ne i64 %tmp1, 0
  br i1 %tmp2, label %bb3, label %bb5
; CHECK: br i1 %tmp2{{.*}}!prof [[LIKELY:![0-9]+]]

bb3:                                              ; preds = %bb
  %tmp4 = call i32 (...) @bar()
  br label %bb5

bb5:                                              ; preds = %bb3, %bb
  ret i32 1
}

define i32 @foo2(i32 %arg) #0 {
; CHECK-LABEL: @foo2
bb:
  %tmp = sext i32 %arg to i64
  %tmp1 = call i64 @llvm.expect.i64(i64 %tmp, i64 4)
  %tmp2 = icmp eq i64 %tmp1, 2
  br i1 %tmp2, label %bb3, label %bb5
; CHECK: br i1 %tmp2{{.*}}!prof [[UNLIKELY:![0-9]+]]

bb3:                                              ; preds = %bb
  %tmp4 = call i32 (...) @bar()
  br label %bb5

bb5:                                              ; preds = %bb3, %bb
  ret i32 1
}

define i32 @foo3(i32 %arg) #0 {
; CHECK-LABEL: @foo3
bb:
  %tmp = sext i32 %arg to i64
  %tmp1 = call i64 @llvm.expect.i64(i64 %tmp, i64 4)
  %tmp2 = icmp eq i64 %tmp1, 4
  br i1 %tmp2, label %bb3, label %bb5
; CHECK: br i1 %tmp2{{.*}}!prof [[LIKELY]]

bb3:                                              ; preds = %bb
  %tmp4 = call i32 (...) @bar()
  br label %bb5

bb5:                                              ; preds = %bb3, %bb
  ret i32 1
}

define i32 @foo4(i32 %arg) #0 {
; CHECK-LABEL: @foo4
bb:
  %tmp = sext i32 %arg to i64
  %tmp1 = call i64 @llvm.expect.i64(i64 %tmp, i64 4)
  %tmp2 = icmp ne i64 %tmp1, 2
  br i1 %tmp2, label %bb3, label %bb5
; CHECK: br i1 %tmp2{{.*}}!prof [[LIKELY]]

bb3:                                              ; preds = %bb
  %tmp4 = call i32 (...) @bar()
  br label %bb5

bb5:                                              ; preds = %bb3, %bb
  ret i32 1
}

define i32 @foo5(i32 %arg, i32 %arg1) #0 {
; CHECK-LABEL: @foo5
bb:
  %tmp = sext i32 %arg1 to i64
  %tmp2 = call i64 @llvm.expect.i64(i64 %tmp, i64 4)
  %tmp3 = sext i32 %arg to i64
  %tmp4 = icmp ne i64 %tmp2, %tmp3
  br i1 %tmp4, label %bb5, label %bb7
; CHECK-NOT: !prof

bb5:                                              ; preds = %bb
  %tmp6 = call i32 (...) @bar()
  br label %bb7

bb7:                                              ; preds = %bb5, %bb
  ret i32 1
}

declare i64 @llvm.expect.i64(i64, i64) #1

declare i32 @bar(...) local_unnamed_addr #0

attributes #0 = { nounwind uwtable }
attributes #1 = { nounwind readnone }

!llvm.module.flags = !{!0}
!llvm.ident = !{!1}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{!"clang version 5.0.0 (trunk 304373)"}
; CHECK: [[LIKELY]] = !{!"branch_weights", i32 2000, i32 1}
; CHECK: [[UNLIKELY]] = !{!"branch_weights", i32 1, i32 2000}

