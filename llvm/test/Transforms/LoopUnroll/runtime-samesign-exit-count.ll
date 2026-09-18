; RUN: opt -passes=loop-unroll -unroll-runtime -scev-cheap-expansion-budget=4 \
; RUN:   -pass-remarks=loop-unroll -S < %s 2>&1 | FileCheck %s

; The signed constant maximum is tighter, but expanding the signed trip count
; 1 + n - smin(n, 2) exceeds the expansion budget. Keep the unsigned exact count
; so that the runtime unroller only needs to expand n - 1.
; CHECK: remark: {{.*}}unrolled loop by a factor of 8 with run-time trip count
; CHECK-LABEL: define void @countdown(
; CHECK: loop.preheader:
; CHECK-NEXT: [[COUNT:%.*]] = add i32 %n, -1
; CHECK-NOT: @llvm.smin
; CHECK: ret void

define void @countdown(ptr %p, i32 %n) {
entry:
  %enter = icmp ugt i32 %n, 2
  br i1 %enter, label %loop, label %exit

loop:
  %i = phi i32 [ %n, %entry ], [ %next, %loop ]
  %ptr = getelementptr i32, ptr %p, i32 %i
  store i32 %i, ptr %ptr
  %next = add i32 %i, -1
  %continue = icmp samesign ugt i32 %i, 2
  br i1 %continue, label %loop, label %exit

exit:
  ret void
}
