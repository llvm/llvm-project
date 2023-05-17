; RUN: opt -S -disable-output -unroll-partial-threshold=16 -debug-only=loop-unroll -passes='loop-unroll<runtime>' < %s 2>&1 | FileCheck %s

; REQUIRES: asserts

; This test is needed to make sure that the guard cost remains the same,
; independently on guard representation form (either intrinsic call or branch with
; widenable condition).

define void @test_guard_as_intrinsic(ptr %arr, i64 %n, i64 %bound) {
; CHECK-LABEL: Loop Unroll: F[test_guard_as_intrinsic] Loop %loop
; CHECK-NEXT:    Loop Size = 8
; CHECK-NEXT:    runtime unrolling with count: 2
entry:
  br label %loop

loop:
  %iv = phi i64 [0, %entry], [%iv.next, %loop]
  %gep = getelementptr i64, ptr %arr, i64 %iv
  %bound_check = icmp ult i64 %iv, %bound
  call void(i1, ...) @llvm.experimental.guard(i1 %bound_check) [ "deopt"() ]
  store i64 %iv, ptr %gep, align 8
  store i64 %iv, ptr %gep, align 8
  %iv.next = add i64 %iv, 1
  %loop_cond = icmp ult i64 %iv, %n
  br i1 %loop_cond, label %loop, label %exit

exit:
  ret void
}

define void @test_guard_as_branch(ptr %arr, i64 %n, i64 %bound) {
; CHECK-LABEL: Loop Unroll: F[test_guard_as_branch] Loop %loop
; CHECK-NEXT:    Loop Size = 8
; CHECK-NEXT:    runtime unrolling with count: 2
entry:
  br label %loop

loop:                                             ; preds = %guarded, %entry
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %guarded ]
  %gep = getelementptr i64, ptr %arr, i64 %iv
  %bound_check = icmp ult i64 %iv, %bound
  %widenable_cond = call i1 @llvm.experimental.widenable.condition()
  %exiplicit_guard_cond = and i1 %bound_check, %widenable_cond
  br i1 %exiplicit_guard_cond, label %guarded, label %deopt, !prof !0

deopt:                                            ; preds = %loop
  call void (...) @llvm.experimental.deoptimize.isVoid() [ "deopt"() ]
  ret void

guarded:                                          ; preds = %loop
  store i64 %iv, ptr %gep, align 8
  store i64 %iv, ptr %gep, align 8
  %iv.next = add i64 %iv, 1
  %loop_cond = icmp ult i64 %iv, %n
  br i1 %loop_cond, label %loop, label %exit

exit:                                             ; preds = %guarded
  ret void
}

; Function Attrs: nocallback nofree nosync willreturn
declare void @llvm.experimental.guard(i1, ...) #0

declare void @llvm.experimental.deoptimize.isVoid(...)

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(inaccessiblemem: readwrite)
declare i1 @llvm.experimental.widenable.condition() #1

!0 = !{!"branch_weights", i32 1048576, i32 1}
