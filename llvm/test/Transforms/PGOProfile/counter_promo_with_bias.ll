; RUN: opt < %s -passes=instrprof -runtime-counter-relocation -do-counter-promotion=true -S | FileCheck %s

target triple = "x86_64-unknown-linux-gnu"

@__profn_foo = private constant [3 x i8] c"foo"

define void @foo(i1 %c) {
entry:
; CHECK: %[[BIAS:[0-9]+]] = load i64, i64* @__llvm_profile_counter_bias
  br label %while.cond

while.cond:                                       ; preds = %land.rhs, %while.cond.preheader
; CHECK: %[[COUNT:[a-z0-9.]+]] = phi i64 [ %[[LAND_COUNT:[0-9]+]], %land.rhs ], [ 0, %entry ]
  br i1 %c, label %while.cond.cleanup_crit_edge, label %land.rhs

while.cond.cleanup_crit_edge:                     ; preds = %while.cond
; CHECK: %[[COUNTER_PTR:[0-9]+]] = add i64 ptrtoint ([1 x i64]* @__profc_foo to i64), %[[BIAS]]
; CHECK: %[[COUNTER_ADDR:[0-9]+]] = inttoptr i64 %[[COUNTER_PTR]] to i64*
; CHECK: %[[COUNTER_PROMO:[a-z0-9.]+]] = load i64, i64* %[[COUNTER_ADDR]]
; CHECK: %[[VALUE:[0-9]+]] = add i64 %[[COUNTER_PROMO]], %[[COUNT]]
; CHECK: store i64 %[[VALUE]], i64* %[[COUNTER_ADDR]]
  br label %cleanup

land.rhs:                                         ; preds = %while.cond
  call void @llvm.instrprof.increment(i8* getelementptr inbounds ([3 x i8], [3 x i8]* @__profn_foo, i32 0, i32 0), i64 0, i32 1, i32 0)
; CHECK: %[[LAND_COUNT]] = add i64 %[[COUNT]], 1
  br label %while.cond

cleanup:                                          ; preds = %while.cond.cleanup_crit_edge
  ret void
}

declare void @llvm.instrprof.increment(i8 *, i64, i32, i32)
