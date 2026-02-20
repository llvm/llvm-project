; RUN: opt -disable-output -passes=loop-unroll -debug-only=loop-unroll -pass-remarks=loop-unroll \
; RUN:     -pass-remarks-missed=loop-unroll -pass-remarks-analysis=loop-unroll < %s 2>&1 | FileCheck %s --match-full-lines --strict-whitespace
; RUN: opt -disable-output -passes=loop-unroll -debug-only=loop-unroll -pass-remarks=loop-unroll \
; RUN:     -pass-remarks-missed=loop-unroll -pass-remarks-analysis=loop-unroll -unroll-count=4 < %s 2>&1 \
; RUN:     | FileCheck %s --match-full-lines --strict-whitespace --check-prefix=PARTIAL-UNROLL
; RUN: opt -disable-output -passes=loop-unroll -debug-only=loop-unroll -pass-remarks=loop-unroll \
; RUN:     -pass-remarks-missed=loop-unroll -pass-remarks-analysis=loop-unroll -unroll-runtime < %s 2>&1 \
; RUN:     | FileCheck %s --match-full-lines --strict-whitespace --check-prefix=RUNTIME-UNROLL
; RUN: opt -disable-output -passes=loop-unroll -debug-only=loop-unroll -pass-remarks=loop-unroll \
; RUN:     -pass-remarks-missed=loop-unroll -pass-remarks-analysis=loop-unroll -unroll-allow-partial < %s 2>&1 \
; RUN:     | FileCheck %s --match-full-lines --strict-whitespace --check-prefix=PARTIAL-ALLOW
; RUN: opt -disable-output -passes=loop-unroll -debug-only=loop-unroll -pass-remarks=loop-unroll \
; RUN:     -pass-remarks-missed=loop-unroll -pass-remarks-analysis=loop-unroll -unroll-count=4 -unroll-allow-remainder=false < %s 2>&1 \
; RUN:     | FileCheck %s --match-full-lines --strict-whitespace --check-prefix=USER-COUNT-REJECT
; RUN: opt -disable-output -passes=loop-unroll -debug-only=loop-unroll -pass-remarks=loop-unroll \
; RUN:     -pass-remarks-missed=loop-unroll -pass-remarks-analysis=loop-unroll -unroll-allow-remainder=false < %s 2>&1 \
; RUN:     | FileCheck %s --match-full-lines --strict-whitespace --check-prefix=NO-REMAINDER
; RUN: opt -disable-output -passes=loop-unroll -debug-only=loop-unroll -pass-remarks=loop-unroll \
; RUN:     -pass-remarks-missed=loop-unroll -pass-remarks-analysis=loop-unroll -pragma-unroll-full-max-iterations=100 < %s 2>&1 \
; RUN:     | FileCheck %s --match-full-lines --strict-whitespace --check-prefix=PRAGMA-TC-TOO-LARGE
; RUN: opt -disable-output -passes=loop-unroll -debug-only=loop-unroll -pass-remarks=loop-unroll \
; RUN:     -pass-remarks-missed=loop-unroll -pass-remarks-analysis=loop-unroll -unroll-threshold=20 < %s 2>&1 \
; RUN:     | FileCheck %s --match-full-lines --strict-whitespace --check-prefix=COST-ANALYSIS
; RUN: opt -disable-output -passes=loop-unroll -debug-only=loop-unroll -pass-remarks=loop-unroll \
; RUN:     -pass-remarks-missed=loop-unroll -pass-remarks-analysis=loop-unroll -unroll-full-max-count=10 \
; RUN:     -pragma-unroll-full-max-iterations=10 < %s 2>&1 \
; RUN:     | FileCheck %s --match-full-lines --strict-whitespace --check-prefix=MAX-COUNT-10
; RUN: opt -disable-output -passes=loop-unroll -debug-only=loop-unroll -pass-remarks=loop-unroll \
; RUN:     -pass-remarks-missed=loop-unroll -pass-remarks-analysis=loop-unroll -unroll-peel-count=2 < %s 2>&1 \
; RUN:     | FileCheck %s --match-full-lines --strict-whitespace --check-prefix=PEEL
; RUN: opt -disable-output -passes=loop-unroll -debug-only=loop-unroll -pass-remarks=loop-unroll \
; RUN:     -pass-remarks-missed=loop-unroll -pass-remarks-analysis=loop-unroll -unroll-threshold=0 -unroll-partial-threshold=0 < %s 2>&1 \
; RUN:     | FileCheck %s --match-full-lines --strict-whitespace --check-prefix=THRESHOLDS-ZERO
; RUN: opt -disable-output -passes=loop-unroll -debug-only=loop-unroll -pass-remarks=loop-unroll \
; RUN:     -pass-remarks-missed=loop-unroll -pass-remarks-analysis=loop-unroll -unroll-threshold=30 < %s 2>&1 \
; RUN:     | FileCheck %s --match-full-lines --strict-whitespace --check-prefix=NESTED-COST
; RUN: opt -disable-output -passes=loop-unroll -debug-only=loop-unroll -pass-remarks=loop-unroll \
; RUN:     -pass-remarks-missed=loop-unroll -pass-remarks-analysis=loop-unroll -unroll-allow-partial -unroll-partial-threshold=8 < %s 2>&1 \
; RUN:     | FileCheck %s --match-full-lines --strict-whitespace --check-prefix=NO-PROFIT
; RUN: opt -disable-output -passes=loop-unroll -debug-only=loop-unroll -pass-remarks=loop-unroll \
; RUN:     -pass-remarks-missed=loop-unroll -pass-remarks-analysis=loop-unroll -unroll-allow-partial -pragma-unroll-threshold=10 \
; RUN:     -unroll-threshold=10 -unroll-partial-threshold=10 -pragma-unroll-full-max-iterations=10 < %s 2>&1 \
; RUN:     | FileCheck %s --match-full-lines --strict-whitespace --check-prefix=UNROLL-AS-DIRECTED-FAIL
; RUN: opt -disable-output -passes=loop-unroll -debug-only=loop-unroll -pass-remarks=loop-unroll \
; RUN:     -pass-remarks-missed=loop-unroll -pass-remarks-analysis=loop-unroll -unroll-count=8 -unroll-threshold=10 < %s 2>&1 \
; RUN:     | FileCheck %s --match-full-lines --strict-whitespace --check-prefix=USER-COUNT-EXCEED
; RUN: opt -disable-output -passes=loop-unroll -debug-only=loop-unroll -pass-remarks=loop-unroll \
; RUN:     -pass-remarks-missed=loop-unroll -pass-remarks-analysis=loop-unroll -unroll-threshold=1 -unroll-partial-threshold=1 < %s 2>&1 \
; RUN:     | FileCheck %s --match-full-lines --strict-whitespace --check-prefix=NO-STRATEGY
; RUN: opt -disable-output -passes=loop-unroll -debug-only=loop-unroll -pass-remarks=loop-unroll \
; RUN:     -pass-remarks-missed=loop-unroll -pass-remarks-analysis=loop-unroll -unroll-threshold=20 -pragma-unroll-threshold=20 < %s 2>&1 \
; RUN:     | FileCheck %s --match-full-lines --strict-whitespace --check-prefix=COST-NOT-PROFITABLE
; RUN: opt -disable-output -O2 --disable-loop-unrolling -debug-only=loop-unroll -pass-remarks=loop-unroll \
; RUN:     -pass-remarks-missed=loop-unroll -pass-remarks-analysis=loop-unroll < %s 2>&1 \
; RUN:     | FileCheck %s --match-full-lines --strict-whitespace --check-prefix=AUTO-DISABLED
; RUN: opt -disable-output -passes='loop-unroll<upperbound>' -debug-only=loop-unroll -pass-remarks=loop-unroll \
; RUN:     -pass-remarks-missed=loop-unroll -pass-remarks-analysis=loop-unroll < %s 2>&1 \
; RUN:     | FileCheck %s --match-full-lines --strict-whitespace --check-prefix=UPPER-BOUND-HEURISTIC
; RUN: opt -disable-output -passes=loop-unroll -debug-only=loop-unroll -pass-remarks=loop-unroll \
; RUN:     -pass-remarks-missed=loop-unroll -pass-remarks-analysis=loop-unroll -unroll-threshold=20 \
; RUN:     -pragma-unroll-threshold=20 -pragma-unroll-full-max-iterations=8 < %s 2>&1 \
; RUN:     | FileCheck %s --match-full-lines --strict-whitespace --check-prefix=FULL-COST-NOT-PROFITABLE

; REQUIRES: asserts

; CHECK-LABEL:Loop Unroll: F[full_unroll_simple] Loop %for.body (depth=1)
; CHECK-NEXT:Loop Size = 6
; CHECK-NEXT: Computing unroll count: TripCount=8, MaxTripCount=0, TripMultiple=8
; CHECK-NEXT: Trying pragma unroll...
; CHECK-NEXT: Trying full unroll...
; CHECK-NEXT:  Unrolling: size {{[0-9]+}} < threshold {{[0-9]+}}.
; CHECK-NEXT:  Exiting block %for.body: TripCount=8, TripMultiple=0, BreakoutTrip=0
; CHECK-NEXT:COMPLETELY UNROLLING loop %for.body with trip count 8!
; CHECK-NEXT:remark: <unknown>:0:0: completely unrolled loop with 8 iterations

define i32 @full_unroll_simple(ptr %A) {
entry:
  br label %for.body

for.body:
  %i = phi i32 [ 0, %entry ], [ %inc, %for.body ]
  %sum = phi i32 [ 0, %entry ], [ %add, %for.body ]
  %arrayidx = getelementptr inbounds i32, ptr %A, i32 %i
  %load = load i32, ptr %arrayidx
  %add = add i32 %sum, %load
  %inc = add i32 %i, 1
  %cmp = icmp ult i32 %inc, 8
  br i1 %cmp, label %for.body, label %exit

exit:
  ret i32 %add
}

; PARTIAL-UNROLL-LABEL:Loop Unroll: F[partial_unroll_user_count] Loop %for.body (depth=1)
; PARTIAL-UNROLL-NEXT:Loop Size = 6
; PARTIAL-UNROLL-NEXT: Computing unroll count: TripCount=16, MaxTripCount=0, TripMultiple=16
; PARTIAL-UNROLL-NEXT: Explicit unroll requested: user-count
; PARTIAL-UNROLL-NEXT: Trying pragma unroll...
; PARTIAL-UNROLL-NEXT:  Unrolling with user-specified count: 4
; PARTIAL-UNROLL-NEXT:  Exiting block %for.body: TripCount=16, TripMultiple=0, BreakoutTrip=0
; PARTIAL-UNROLL-NEXT:UNROLLING loop %for.body by 4!
; PARTIAL-UNROLL-NEXT:remark: <unknown>:0:0: unrolled loop by a factor of 4

define i32 @partial_unroll_user_count(ptr %A) {
entry:
  br label %for.body

for.body:
  %i = phi i32 [ 0, %entry ], [ %inc, %for.body ]
  %sum = phi i32 [ 0, %entry ], [ %add, %for.body ]
  %arrayidx = getelementptr inbounds i32, ptr %A, i32 %i
  %load = load i32, ptr %arrayidx
  %add = add i32 %sum, %load
  %inc = add i32 %i, 1
  %cmp = icmp ult i32 %inc, 16
  br i1 %cmp, label %for.body, label %exit

exit:
  ret i32 %add
}

; RUNTIME-UNROLL-LABEL:Loop Unroll: F[runtime_unroll_simple] Loop %for.body (depth=1)
; RUNTIME-UNROLL-NEXT:Loop Size = 6
; RUNTIME-UNROLL-NEXT: Computing unroll count: TripCount=0, MaxTripCount={{[0-9]+}}, TripMultiple=1
; RUNTIME-UNROLL-NEXT: Trying pragma unroll...
; RUNTIME-UNROLL-NEXT: Trying full unroll...
; RUNTIME-UNROLL-NEXT: Trying upper-bound unroll...
; RUNTIME-UNROLL-NEXT: Trying loop peeling...
; RUNTIME-UNROLL-NEXT: Trying partial unroll...
; RUNTIME-UNROLL-NEXT: Trying runtime unroll...
; RUNTIME-UNROLL-NEXT:  Runtime unrolling with count: 8
; RUNTIME-UNROLL-NEXT:  Exiting block %for.body: TripCount=0, TripMultiple=1, BreakoutTrip=1
; RUNTIME-UNROLL-NEXT:Trying runtime unrolling on Loop: 
; RUNTIME-UNROLL-NEXT:Loop at depth 1 containing: %for.body<header><latch><exiting>
; RUNTIME-UNROLL-NEXT:Using epilog remainder.
; RUNTIME-UNROLL-NEXT:UNROLLING loop %for.body by 8 with run-time trip count!
; RUNTIME-UNROLL-NEXT:remark: <unknown>:0:0: unrolled loop by a factor of 8 with run-time trip count

define i32 @runtime_unroll_simple(ptr %A, i32 %n) {
entry:
  %cmp.entry = icmp sgt i32 %n, 0
  br i1 %cmp.entry, label %for.body, label %exit

for.body:
  %i = phi i32 [ 0, %entry ], [ %inc, %for.body ]
  %sum = phi i32 [ 0, %entry ], [ %add, %for.body ]
  %arrayidx = getelementptr inbounds i32, ptr %A, i32 %i
  %load = load i32, ptr %arrayidx
  %add = add i32 %sum, %load
  %inc = add i32 %i, 1
  %cmp = icmp slt i32 %inc, %n
  br i1 %cmp, label %for.body, label %exit

exit:
  %result = phi i32 [ 0, %entry ], [ %add, %for.body ]
  ret i32 %result
}

; CHECK-LABEL:Loop Unroll: F[pragma_full_unroll] Loop %for.body (depth=1)
; CHECK-NEXT:Loop Size = 6
; CHECK-NEXT: Computing unroll count: TripCount=4, MaxTripCount=0, TripMultiple=4
; CHECK-NEXT: Explicit unroll requested: pragma-full
; CHECK-NEXT: Trying pragma unroll...
; CHECK-NEXT:  Fully unrolling with trip count: 4
; CHECK-NEXT:  Exiting block %for.body: TripCount=4, TripMultiple=0, BreakoutTrip=0
; CHECK-NEXT:COMPLETELY UNROLLING loop %for.body with trip count 4!
; CHECK-NEXT:remark: <unknown>:0:0: completely unrolled loop with 4 iterations

define i32 @pragma_full_unroll(ptr %A) {
entry:
  br label %for.body

for.body:
  %i = phi i32 [ 0, %entry ], [ %inc, %for.body ]
  %sum = phi i32 [ 0, %entry ], [ %add, %for.body ]
  %arrayidx = getelementptr inbounds i32, ptr %A, i32 %i
  %load = load i32, ptr %arrayidx
  %add = add i32 %sum, %load
  %inc = add i32 %i, 1
  %cmp = icmp ult i32 %inc, 4
  br i1 %cmp, label %for.body, label %exit, !llvm.loop !0

exit:
  ret i32 %add
}

; CHECK-LABEL:Loop Unroll: F[pragma_unroll_count] Loop %for.body (depth=1)
; CHECK-NEXT:Loop Size = 6
; CHECK-NEXT: Computing unroll count: TripCount=16, MaxTripCount=0, TripMultiple=16
; CHECK-NEXT: Explicit unroll requested: pragma-count(4)
; CHECK-NEXT: Trying pragma unroll...
; CHECK-NEXT:  Unrolling with pragma count: 4
; CHECK-NEXT:  Exiting block %for.body: TripCount=16, TripMultiple=0, BreakoutTrip=0
; CHECK-NEXT:UNROLLING loop %for.body by 4!
; CHECK-NEXT:remark: <unknown>:0:0: unrolled loop by a factor of 4

define i32 @pragma_unroll_count(ptr %A) {
entry:
  br label %for.body

for.body:
  %i = phi i32 [ 0, %entry ], [ %inc, %for.body ]
  %sum = phi i32 [ 0, %entry ], [ %add, %for.body ]
  %arrayidx = getelementptr inbounds i32, ptr %A, i32 %i
  %load = load i32, ptr %arrayidx
  %add = add i32 %sum, %load
  %inc = add i32 %i, 1
  %cmp = icmp ult i32 %inc, 16
  br i1 %cmp, label %for.body, label %exit, !llvm.loop !2

exit:
  ret i32 %add
}

; CHECK-LABEL:Loop Unroll: F[pragma_full_unroll_unknown_tc] Loop %for.body (depth=1)
; CHECK-NEXT:Loop Size = 6
; CHECK-NEXT: Computing unroll count: TripCount=0, MaxTripCount={{[0-9]+}}, TripMultiple=1
; CHECK-NEXT: Explicit unroll requested: pragma-full
; CHECK-NEXT: Trying pragma unroll...
; CHECK-NEXT:  Not fully unrolling: unknown trip count.
; CHECK-NEXT: Trying full unroll...
; CHECK-NEXT: Trying upper-bound unroll...
; CHECK-NEXT: Trying loop peeling...
; CHECK-NEXT: Trying partial unroll...
; CHECK-NEXT: Not fully unrolling as directed: loop has runtime trip count.
; CHECK-NEXT:remark: <unknown>:0:0: unable to fully unroll loop as directed by full unroll pragma because loop has a runtime trip count
; CHECK-NEXT: Trying runtime unroll...
; CHECK-NEXT:  Will not try to unroll loop with runtime trip count because -unroll-runtime not given.
; CHECK-NEXT:Not unrolling: no viable strategy found.

define i32 @pragma_full_unroll_unknown_tc(ptr %A, i32 %n) {
entry:
  %cmp.entry = icmp sgt i32 %n, 0
  br i1 %cmp.entry, label %for.body, label %exit

for.body:
  %i = phi i32 [ 0, %entry ], [ %inc, %for.body ]
  %sum = phi i32 [ 0, %entry ], [ %add, %for.body ]
  %arrayidx = getelementptr inbounds i32, ptr %A, i32 %i
  %load = load i32, ptr %arrayidx
  %add = add i32 %sum, %load
  %inc = add i32 %i, 1
  %cmp = icmp slt i32 %inc, %n
  br i1 %cmp, label %for.body, label %exit, !llvm.loop !0

exit:
  %result = phi i32 [ 0, %entry ], [ %add, %for.body ]
  ret i32 %result
}

; CHECK-LABEL:Loop Unroll: F[upper_bound_unroll] Loop %for.body (depth=1)
; CHECK-NEXT:Loop Size = 6
; CHECK-NEXT: Computing unroll count: TripCount=0, MaxTripCount=3, TripMultiple=1
; CHECK-NEXT: Explicit unroll requested: pragma-enable
; CHECK-NEXT: Trying pragma unroll...
; CHECK-NEXT:  Unrolling with max trip count: 3
; CHECK-NEXT:  Exiting block %for.body: TripCount=0, TripMultiple=1, BreakoutTrip=1
; CHECK-NEXT:COMPLETELY UNROLLING loop %for.body with trip count 3!
; CHECK-NEXT:remark: <unknown>:0:0: completely unrolled loop with 3 iterations

define i32 @upper_bound_unroll(ptr %A, i32 %n) {
entry:
  %n.clamped = and i32 %n, 3  ; max 3
  br label %for.body

for.body:
  %i = phi i32 [ 0, %entry ], [ %inc, %for.body ]
  %sum = phi i32 [ 0, %entry ], [ %add, %for.body ]
  %arrayidx = getelementptr inbounds i32, ptr %A, i32 %i
  %load = load i32, ptr %arrayidx
  %add = add i32 %sum, %load
  %inc = add i32 %i, 1
  %cmp = icmp ult i32 %inc, %n.clamped
  br i1 %cmp, label %for.body, label %exit, !llvm.loop !1

exit:
  ret i32 %add
}

; CHECK-LABEL:Loop Unroll: F[full_unroll_cost_exceeds] Loop %for.body (depth=1)
; CHECK-NEXT:Loop Size = 6
; CHECK-NEXT: Computing unroll count: TripCount=100, MaxTripCount=0, TripMultiple=100
; CHECK-NEXT: Trying pragma unroll...
; CHECK-NEXT: Trying full unroll...
; CHECK-NEXT:  Unrolled size {{[0-9]+}} exceeds threshold {{[0-9]+}}; checking for cost benefit.
; CHECK-NEXT:   Not analyzing loop cost: trip count too large.
; CHECK-NEXT:  Skipping: cost analysis unavailable.
; CHECK-NEXT: Trying upper-bound unroll...
; CHECK-NEXT: Trying loop peeling...
; CHECK-NEXT: Trying partial unroll...
; CHECK-NEXT:   Will not try to unroll partially because -unroll-allow-partial not given.
; CHECK-NEXT:Not unrolling: no viable strategy found.

define i32 @full_unroll_cost_exceeds(ptr %A) {
entry:
  br label %for.body

for.body:
  %i = phi i32 [ 0, %entry ], [ %inc, %for.body ]
  %sum = phi i32 [ 0, %entry ], [ %add, %for.body ]
  %arrayidx = getelementptr inbounds i32, ptr %A, i32 %i
  %load = load i32, ptr %arrayidx
  %add = add i32 %sum, %load
  %inc = add i32 %i, 1
  %cmp = icmp ult i32 %inc, 100
  br i1 %cmp, label %for.body, label %exit

exit:
  ret i32 %add
}

; CHECK-LABEL:Loop Unroll: F[unroll_disabled_metadata] Loop %for.body (depth=1)
; CHECK-NEXT:Not unrolling: transformation disabled by metadata.

define i32 @unroll_disabled_metadata(ptr %A) {
entry:
  br label %for.body

for.body:
  %i = phi i32 [ 0, %entry ], [ %inc, %for.body ]
  %sum = phi i32 [ 0, %entry ], [ %add, %for.body ]
  %arrayidx = getelementptr inbounds i32, ptr %A, i32 %i
  %load = load i32, ptr %arrayidx
  %add = add i32 %sum, %load
  %inc = add i32 %i, 1
  %cmp = icmp ult i32 %inc, 8
  br i1 %cmp, label %for.body, label %exit, !llvm.loop !3

exit:
  ret i32 %add
}

; CHECK-LABEL:Loop Unroll: F[runtime_small_max_tc] Loop %for.body (depth=1)
; CHECK-NEXT:Loop Size = 6
; CHECK-NEXT: Computing unroll count: TripCount=0, MaxTripCount=3, TripMultiple=1
; CHECK-NEXT: Trying pragma unroll...
; CHECK-NEXT: Trying full unroll...
; CHECK-NEXT: Trying upper-bound unroll...
; CHECK-NEXT: Trying loop peeling...
; CHECK-NEXT: Trying partial unroll...
; CHECK-NEXT: Trying runtime unroll...
; CHECK-NEXT: Not runtime unrolling: max trip count 3 is small (< 8) and not forced.
; CHECK-NEXT:Not unrolling: no viable strategy found.

define i32 @runtime_small_max_tc(ptr %A, i32 %n) {
entry:
  %n.clamped = and i32 %n, 3
  br label %for.body

for.body:
  %i = phi i32 [ 0, %entry ], [ %inc, %for.body ]
  %sum = phi i32 [ 0, %entry ], [ %add, %for.body ]
  %arrayidx = getelementptr inbounds i32, ptr %A, i32 %i
  %load = load i32, ptr %arrayidx
  %add = add i32 %sum, %load
  %inc = add i32 %i, 1
  %cmp = icmp ult i32 %inc, %n.clamped
  br i1 %cmp, label %for.body, label %exit

exit:
  ret i32 %add
}

; When using user-specified count on a trip count that isn't evenly divisible
; PARTIAL-UNROLL-LABEL:Loop Unroll: F[partial_unroll_with_remainder] Loop %for.body (depth=1)
; PARTIAL-UNROLL-NEXT:Loop Size = 6
; PARTIAL-UNROLL-NEXT: Computing unroll count: TripCount=10, MaxTripCount=0, TripMultiple=10
; PARTIAL-UNROLL-NEXT: Explicit unroll requested: user-count
; PARTIAL-UNROLL-NEXT: Trying pragma unroll...
; PARTIAL-UNROLL-NEXT:  Unrolling with user-specified count: 4
; PARTIAL-UNROLL-NEXT:Attempting unroll by factor 4 with remainder loop (trip count 10)
; PARTIAL-UNROLL-NEXT:remark: <unknown>:0:0: attempting unroll by factor 4 with remainder loop (trip count 10)
; PARTIAL-UNROLL-NEXT:  Exiting block %for.body: TripCount=10, TripMultiple=0, BreakoutTrip=2
; PARTIAL-UNROLL-NEXT:UNROLLING loop %for.body by 4!
; PARTIAL-UNROLL-NEXT:remark: <unknown>:0:0: unrolled loop by a factor of 4

define i32 @partial_unroll_with_remainder(ptr %A) {
entry:
  br label %for.body

for.body:
  %i = phi i32 [ 0, %entry ], [ %inc, %for.body ]
  %sum = phi i32 [ 0, %entry ], [ %add, %for.body ]
  %arrayidx = getelementptr inbounds i32, ptr %A, i32 %i
  %load = load i32, ptr %arrayidx
  %add = add i32 %sum, %load
  %inc = add i32 %i, 1
  %cmp = icmp ult i32 %inc, 10
  br i1 %cmp, label %for.body, label %exit

exit:
  ret i32 %add
}


; PARTIAL-ALLOW-LABEL:Loop Unroll: F[partial_unroll_cost_analysis] Loop %for.body (depth=1)
; PARTIAL-ALLOW-NEXT:Loop Size = 6
; PARTIAL-ALLOW-NEXT: Computing unroll count: TripCount=200, MaxTripCount=0, TripMultiple=200
; PARTIAL-ALLOW-NEXT: Trying pragma unroll...
; PARTIAL-ALLOW-NEXT: Trying full unroll...
; PARTIAL-ALLOW-NEXT:  Unrolled size {{[0-9]+}} exceeds threshold {{[0-9]+}}; checking for cost benefit.
; PARTIAL-ALLOW-NEXT:   Not analyzing loop cost: trip count too large.
; PARTIAL-ALLOW-NEXT:  Skipping: cost analysis unavailable.
; PARTIAL-ALLOW-NEXT: Trying upper-bound unroll...
; PARTIAL-ALLOW-NEXT: Trying loop peeling...
; PARTIAL-ALLOW-NEXT: Trying partial unroll...
; PARTIAL-ALLOW-NEXT:  Unrolled size exceeds threshold; reducing count from {{[0-9]+}} to {{[0-9]+}}.
; PARTIAL-ALLOW-NEXT:   Partially unrolling with count: 25
; PARTIAL-ALLOW-NEXT:  Exiting block %for.body: TripCount=200, TripMultiple=0, BreakoutTrip=0
; PARTIAL-ALLOW-NEXT:UNROLLING loop %for.body by 25!
; PARTIAL-ALLOW-NEXT:remark: <unknown>:0:0: unrolled loop by a factor of 25

define i32 @partial_unroll_cost_analysis(ptr %A) {
entry:
  br label %for.body

for.body:
  %i = phi i32 [ 0, %entry ], [ %inc, %for.body ]
  %sum = phi i32 [ 0, %entry ], [ %add, %for.body ]
  %arrayidx = getelementptr inbounds i32, ptr %A, i32 %i
  %load = load i32, ptr %arrayidx
  %add = add i32 %sum, %load
  %inc = add i32 %i, 1
  %cmp = icmp ult i32 %inc, 200
  br i1 %cmp, label %for.body, label %exit

exit:
  ret i32 %add
}

; CHECK-LABEL:Loop Unroll: F[runtime_unroll_disabled_pragma] Loop %for.body (depth=1)
; CHECK-NEXT:Loop Size = 6
; CHECK-NEXT: Computing unroll count: TripCount=0, MaxTripCount={{[0-9]+}}, TripMultiple=1
; CHECK-NEXT: Trying pragma unroll...
; CHECK-NEXT: Trying full unroll...
; CHECK-NEXT: Trying upper-bound unroll...
; CHECK-NEXT: Trying loop peeling...
; CHECK-NEXT: Trying partial unroll...
; CHECK-NEXT: Trying runtime unroll...
; CHECK-NEXT: Not runtime unrolling: disabled by pragma.
; CHECK-NEXT:Not unrolling: no viable strategy found.

define i32 @runtime_unroll_disabled_pragma(ptr %A, i32 %n) {
entry:
  %cmp.entry = icmp sgt i32 %n, 0
  br i1 %cmp.entry, label %for.body, label %exit

for.body:
  %i = phi i32 [ 0, %entry ], [ %inc, %for.body ]
  %sum = phi i32 [ 0, %entry ], [ %add, %for.body ]
  %arrayidx = getelementptr inbounds i32, ptr %A, i32 %i
  %load = load i32, ptr %arrayidx
  %add = add i32 %sum, %load
  %inc = add i32 %i, 1
  %cmp = icmp slt i32 %inc, %n
  br i1 %cmp, label %for.body, label %exit, !llvm.loop !4

exit:
  %result = phi i32 [ 0, %entry ], [ %add, %for.body ]
  ret i32 %result
}

; AUTO-DISABLED-LABEL:Loop Unroll: F[auto_unroll_not_enabled] Loop %for.body (depth=1)
; AUTO-DISABLED-NEXT:Not unrolling: automatic unrolling disabled and loop not explicitly enabled.

define i32 @auto_unroll_not_enabled(ptr %A) {
entry:
  br label %for.body

for.body:
  %i = phi i32 [ 0, %entry ], [ %inc, %for.body ]
  %sum = phi i32 [ 0, %entry ], [ %add, %for.body ]
  %arrayidx = getelementptr inbounds i32, ptr %A, i32 %i
  %load = load i32, ptr %arrayidx
  %add = add i32 %sum, %load
  %inc = add i32 %i, 1
  %cmp = icmp ult i32 %inc, 4
  br i1 %cmp, label %for.body, label %exit

exit:
  ret i32 %add
}

; USER-COUNT-REJECT-LABEL:Loop Unroll: F[user_count_rejected] Loop %for.body (depth=1)
; USER-COUNT-REJECT-NEXT:Loop Size = 6
; USER-COUNT-REJECT-NEXT: Computing unroll count: TripCount=10, MaxTripCount=0, TripMultiple=10
; USER-COUNT-REJECT-NEXT: Explicit unroll requested: user-count
; USER-COUNT-REJECT-NEXT: Trying pragma unroll...
; USER-COUNT-REJECT-NEXT:  Not unrolling with user count 4: remainder not allowed.
; USER-COUNT-REJECT-NEXT: Trying full unroll...
; USER-COUNT-REJECT-NEXT:  Unrolling: size 42 < threshold 16384.
; USER-COUNT-REJECT-NEXT:  Exiting block %for.body: TripCount=10, TripMultiple=0, BreakoutTrip=0
; USER-COUNT-REJECT-NEXT:COMPLETELY UNROLLING loop %for.body with trip count 10!
; USER-COUNT-REJECT-NEXT:remark: <unknown>:0:0: completely unrolled loop with 10 iterations

define i32 @user_count_rejected(ptr %A) {
entry:
  br label %for.body

for.body:
  %i = phi i32 [ 0, %entry ], [ %inc, %for.body ]
  %sum = phi i32 [ 0, %entry ], [ %add, %for.body ]
  %arrayidx = getelementptr inbounds i32, ptr %A, i32 %i
  %load = load i32, ptr %arrayidx
  %add = add i32 %sum, %load
  %inc = add i32 %i, 1
  %cmp = icmp ult i32 %inc, 10
  br i1 %cmp, label %for.body, label %exit

exit:
  ret i32 %add
}

; NO-REMAINDER-LABEL:Loop Unroll: F[pragma_count_rejected] Loop %for.body (depth=1)
; NO-REMAINDER-NEXT:Loop Size = 6
; NO-REMAINDER-NEXT: Computing unroll count: TripCount=10, MaxTripCount=0, TripMultiple=10
; NO-REMAINDER-NEXT: Explicit unroll requested: pragma-count(4)
; NO-REMAINDER-NEXT: Trying pragma unroll...
; NO-REMAINDER-NEXT:  Not unrolling with pragma count 4: remainder not allowed, count does not divide trip multiple 10.
; NO-REMAINDER-NEXT:remark: <unknown>:0:0: unable to unroll loop with count 4: remainder loop is restricted and count does not divide trip multiple 10
; NO-REMAINDER-NEXT: Trying full unroll...
; NO-REMAINDER-NEXT:  Unrolling: size 42 < threshold 16384.
; NO-REMAINDER-NEXT:  Exiting block %for.body: TripCount=10, TripMultiple=0, BreakoutTrip=0
; NO-REMAINDER-NEXT:COMPLETELY UNROLLING loop %for.body with trip count 10!
; NO-REMAINDER-NEXT:remark: <unknown>:0:0: completely unrolled loop with 10 iterations

define i32 @pragma_count_rejected(ptr %A) {
entry:
  br label %for.body

for.body:
  %i = phi i32 [ 0, %entry ], [ %inc, %for.body ]
  %sum = phi i32 [ 0, %entry ], [ %add, %for.body ]
  %arrayidx = getelementptr inbounds i32, ptr %A, i32 %i
  %load = load i32, ptr %arrayidx
  %add = add i32 %sum, %load
  %inc = add i32 %i, 1
  %cmp = icmp ult i32 %inc, 10
  br i1 %cmp, label %for.body, label %exit, !llvm.loop !10

exit:
  ret i32 %add
}

; The contradictory "unable to fully unroll" then "completely unrolled" remarks are expected:
; we're artificially limiting the pragma path with -pragma-unroll-full-max-iterations=100
; while the heuristic path remains unconstrained. This won't happen with default flags.
; PRAGMA-TC-TOO-LARGE-LABEL:Loop Unroll: F[pragma_full_tc_too_large] Loop %for.body (depth=1)
; PRAGMA-TC-TOO-LARGE-NEXT:Loop Size = 6
; PRAGMA-TC-TOO-LARGE-NEXT: Computing unroll count: TripCount=200, MaxTripCount=0, TripMultiple=200
; PRAGMA-TC-TOO-LARGE-NEXT: Explicit unroll requested: pragma-full
; PRAGMA-TC-TOO-LARGE-NEXT: Trying pragma unroll...
; PRAGMA-TC-TOO-LARGE-NEXT:   Won't unroll; trip count is too large.
; PRAGMA-TC-TOO-LARGE-NEXT:remark: <unknown>:0:0: unable to fully unroll loop: trip count 200 exceeds limit 100
; PRAGMA-TC-TOO-LARGE-NEXT: Trying full unroll...
; PRAGMA-TC-TOO-LARGE-NEXT:  Unrolling: size 802 < threshold 16384.
; PRAGMA-TC-TOO-LARGE-NEXT:  Exiting block %for.body: TripCount=200, TripMultiple=0, BreakoutTrip=0
; PRAGMA-TC-TOO-LARGE-NEXT:COMPLETELY UNROLLING loop %for.body with trip count 200!
; PRAGMA-TC-TOO-LARGE-NEXT:remark: <unknown>:0:0: completely unrolled loop with 200 iterations

define i32 @pragma_full_tc_too_large(ptr %A) {
entry:
  br label %for.body

for.body:
  %i = phi i32 [ 0, %entry ], [ %inc, %for.body ]
  %sum = phi i32 [ 0, %entry ], [ %add, %for.body ]
  %arrayidx = getelementptr inbounds i32, ptr %A, i32 %i
  %load = load i32, ptr %arrayidx
  %add = add i32 %sum, %load
  %inc = add i32 %i, 1
  %cmp = icmp ult i32 %inc, 200
  br i1 %cmp, label %for.body, label %exit, !llvm.loop !0

exit:
  ret i32 %add
}

; COST-ANALYSIS-LABEL:Loop Unroll: F[cost_analysis_detailed] Loop %for.body (depth=1)
; COST-ANALYSIS-NEXT:Loop Size = 9
; COST-ANALYSIS-NEXT: Computing unroll count: TripCount=10, MaxTripCount=0, TripMultiple=10
; COST-ANALYSIS-NEXT: Trying pragma unroll...
; COST-ANALYSIS-NEXT: Trying full unroll...
; COST-ANALYSIS-NEXT:  Unrolled size {{[0-9]+}} exceeds threshold 20; checking for cost benefit.
; COST-ANALYSIS-NEXT:    Starting LoopUnroll profitability analysis...
; COST-ANALYSIS:    Analysis finished:
; COST-ANALYSIS-NEXT:    UnrolledCost: {{[0-9]+}}, RolledDynamicCost: {{[0-9]+}}
; COST-ANALYSIS-NEXT:  Not unrolling: cost {{[0-9]+}} >= boosted threshold {{[0-9]+}}.
; COST-ANALYSIS-NEXT: Trying upper-bound unroll...
; COST-ANALYSIS-NEXT: Trying loop peeling...
; COST-ANALYSIS-NEXT: Trying partial unroll...
; COST-ANALYSIS-NEXT:   Will not try to unroll partially because -unroll-allow-partial not given.
; COST-ANALYSIS-NEXT:Not unrolling: no viable strategy found.

define i32 @cost_analysis_detailed(ptr %A, ptr %B) {
entry:
  br label %for.body

for.body:
  %i = phi i32 [ 0, %entry ], [ %inc, %for.body ]
  %sum = phi i32 [ 0, %entry ], [ %add, %for.body ]
  %arrayidx = getelementptr inbounds i32, ptr %A, i32 %i
  %load = load i32, ptr %arrayidx
  %arrayidx2 = getelementptr inbounds i32, ptr %B, i32 %i
  %load2 = load i32, ptr %arrayidx2
  %mul = mul i32 %load, %load2
  %add = add i32 %sum, %mul
  %inc = add i32 %i, 1
  %cmp = icmp ult i32 %inc, 10
  br i1 %cmp, label %for.body, label %exit

exit:
  ret i32 %add
}

; MAX-COUNT-10-LABEL:Loop Unroll: F[exceeds_max_count] Loop %for.body (depth=1)
; MAX-COUNT-10-NEXT:Loop Size = 6
; MAX-COUNT-10-NEXT: Computing unroll count: TripCount=20, MaxTripCount=0, TripMultiple=20
; MAX-COUNT-10-NEXT: Trying pragma unroll...
; MAX-COUNT-10-NEXT: Trying full unroll...
; MAX-COUNT-10-NEXT:  Not unrolling: trip count 20 exceeds max count 10.
; MAX-COUNT-10-NEXT: Trying upper-bound unroll...
; MAX-COUNT-10-NEXT: Trying loop peeling...
; MAX-COUNT-10-NEXT: Trying partial unroll...
; MAX-COUNT-10-NEXT:   Will not try to unroll partially because -unroll-allow-partial not given.
; MAX-COUNT-10-NEXT:Not unrolling: no viable strategy found.

define i32 @exceeds_max_count(ptr %A) {
entry:
  br label %for.body

for.body:
  %i = phi i32 [ 0, %entry ], [ %inc, %for.body ]
  %sum = phi i32 [ 0, %entry ], [ %add, %for.body ]
  %arrayidx = getelementptr inbounds i32, ptr %A, i32 %i
  %load = load i32, ptr %arrayidx
  %add = add i32 %sum, %load
  %inc = add i32 %i, 1
  %cmp = icmp ult i32 %inc, 20
  br i1 %cmp, label %for.body, label %exit

exit:
  ret i32 %add
}

; PEEL-LABEL:Loop Unroll: F[explicit_peel] Loop %for.body (depth=1)
; PEEL-NEXT:Loop Size = 6
; PEEL-NEXT: Computing unroll count: TripCount=100, MaxTripCount=0, TripMultiple=100
; PEEL-NEXT: Using explicit peel count: 2
; PEEL-NEXT:PEELING loop %for.body with iteration count 2!
; PEEL-NEXT:remark: <unknown>:0:0: peeled loop by 2 iterations

define i32 @explicit_peel(ptr %A) {
entry:
  br label %for.body

for.body:
  %i = phi i32 [ 0, %entry ], [ %inc, %for.body ]
  %sum = phi i32 [ 0, %entry ], [ %add, %for.body ]
  %arrayidx = getelementptr inbounds i32, ptr %A, i32 %i
  %load = load i32, ptr %arrayidx
  %add = add i32 %sum, %load
  %inc = add i32 %i, 1
  %cmp = icmp ult i32 %inc, 100
  br i1 %cmp, label %for.body, label %exit

exit:
  ret i32 %add
}

; CHECK-LABEL:Loop Unroll: F[heuristic_peel] Loop %for.header (depth=1)
; CHECK-NEXT:Loop Size = 11
; CHECK-NEXT: Computing unroll count: TripCount=0, MaxTripCount={{[0-9]+}}, TripMultiple=1
; CHECK-NEXT: Trying pragma unroll...
; CHECK-NEXT: Trying full unroll...
; CHECK-NEXT: Trying upper-bound unroll...
; CHECK-NEXT: Trying loop peeling...
; CHECK-NEXT: Peeling with count: 1
; CHECK-NEXT:PEELING loop %for.header with iteration count 1!
; CHECK-NEXT:remark: <unknown>:0:0: peeled loop by 1 iterations

define i32 @heuristic_peel(ptr %A, i32 %n) {
entry:
  %cmp.entry = icmp sgt i32 %n, 0
  br i1 %cmp.entry, label %for.header, label %exit.early

for.header:
  %i = phi i32 [ 0, %entry ], [ %inc, %for.latch ]
  %sum = phi i32 [ 0, %entry ], [ %add, %for.latch ]
  ; This comparison becomes false after first iteration - enables peeling
  %first = icmp eq i32 %i, 0
  br i1 %first, label %special, label %normal

special:
  %load1 = load i32, ptr %A
  br label %for.latch

normal:
  %arrayidx = getelementptr inbounds i32, ptr %A, i32 %i
  %load2 = load i32, ptr %arrayidx
  br label %for.latch

for.latch:
  %val = phi i32 [ %load1, %special ], [ %load2, %normal ]
  %add = add i32 %sum, %val
  %inc = add i32 %i, 1
  %cmp = icmp slt i32 %inc, %n
  br i1 %cmp, label %for.header, label %exit

exit:
  ret i32 %add

exit.early:
  ret i32 0
}

; THRESHOLDS-ZERO-LABEL:Loop Unroll: F[thresholds_zero] Loop %for.body (depth=1)
; THRESHOLDS-ZERO-NEXT:Not unrolling: all thresholds are zero.
; THRESHOLDS-ZERO-NEXT:remark: <unknown>:0:0: unable to unroll loop: unroll threshold is zero

define i32 @thresholds_zero(ptr %A) {
entry:
  br label %for.body

for.body:
  %i = phi i32 [ 0, %entry ], [ %inc, %for.body ]
  %sum = phi i32 [ 0, %entry ], [ %add, %for.body ]
  %arrayidx = getelementptr inbounds i32, ptr %A, i32 %i
  %load = load i32, ptr %arrayidx
  %add = add i32 %sum, %load
  %inc = add i32 %i, 1
  %cmp = icmp ult i32 %inc, 10
  br i1 %cmp, label %for.body, label %exit, !llvm.loop !1

exit:
  ret i32 %add
}

; NESTED-COST-LABEL:Loop Unroll: F[nested_cost_analysis] Loop %inner.header (depth=2)
; NESTED-COST-NEXT:Not unrolling: transformation disabled by metadata.
; NESTED-COST-LABEL:Loop Unroll: F[nested_cost_analysis] Loop %outer.header (depth=1)
; NESTED-COST-NEXT:Loop Size = 11
; NESTED-COST-NEXT: Computing unroll count: TripCount=4, MaxTripCount=0, TripMultiple=4
; NESTED-COST-NEXT: Trying pragma unroll...
; NESTED-COST-NEXT: Trying full unroll...
; NESTED-COST-NEXT:  Unrolled size {{[0-9]+}} exceeds threshold 30; checking for cost benefit.
; NESTED-COST-NEXT:   Not analyzing loop cost: not an innermost loop.
; NESTED-COST-NEXT:  Skipping: cost analysis unavailable.
; NESTED-COST-NEXT: Trying upper-bound unroll...
; NESTED-COST-NEXT: Trying loop peeling...
; NESTED-COST-NEXT: Trying partial unroll...
; NESTED-COST-NEXT:   Will not try to unroll partially because -unroll-allow-partial not given.
; NESTED-COST-NEXT:Not unrolling: no viable strategy found.

define i32 @nested_cost_analysis(ptr %A) {
entry:
  br label %outer.header

outer.header:
  %i = phi i32 [ 0, %entry ], [ %i.inc, %outer.latch ]
  br label %inner.header

inner.header:
  %j = phi i32 [ 0, %outer.header ], [ %j.inc, %inner.header ]
  %sum = phi i32 [ 0, %outer.header ], [ %add, %inner.header ]
  %idx = add i32 %i, %j
  %arrayidx = getelementptr inbounds i32, ptr %A, i32 %idx
  %load = load i32, ptr %arrayidx
  %add = add i32 %sum, %load
  %j.inc = add i32 %j, 1
  %inner.cmp = icmp ult i32 %j.inc, 100
  br i1 %inner.cmp, label %inner.header, label %outer.latch, !llvm.loop !3

outer.latch:
  %i.inc = add i32 %i, 1
  %outer.cmp = icmp ult i32 %i.inc, 4
  br i1 %outer.cmp, label %outer.header, label %exit

exit:
  ret i32 %add
}

; PRAGMA-TC-TOO-LARGE-LABEL:Loop Unroll: F[partial_instead_of_full] Loop %for.body (depth=1)
; PRAGMA-TC-TOO-LARGE-NEXT:Loop Size = 6
; PRAGMA-TC-TOO-LARGE-NEXT: Computing unroll count: TripCount=5000, MaxTripCount=0, TripMultiple=5000
; PRAGMA-TC-TOO-LARGE-NEXT: Explicit unroll requested: pragma-full
; PRAGMA-TC-TOO-LARGE-NEXT: Trying pragma unroll...
; PRAGMA-TC-TOO-LARGE-NEXT:   Won't unroll; trip count is too large.
; PRAGMA-TC-TOO-LARGE-NEXT:remark: <unknown>:0:0: unable to fully unroll loop: trip count 5000 exceeds limit 100
; PRAGMA-TC-TOO-LARGE-NEXT: Trying full unroll...
; PRAGMA-TC-TOO-LARGE-NEXT:  Unrolled size {{[0-9]+}} exceeds threshold {{[0-9]+}}; checking for cost benefit.
; PRAGMA-TC-TOO-LARGE-NEXT:   Not analyzing loop cost: trip count too large.
; PRAGMA-TC-TOO-LARGE-NEXT:  Skipping: cost analysis unavailable.
; PRAGMA-TC-TOO-LARGE-NEXT:remark: <unknown>:0:0: unable to fully unroll loop: estimated unrolled size {{[0-9]+}} exceeds threshold {{[0-9]+}}
; PRAGMA-TC-TOO-LARGE-NEXT: Trying upper-bound unroll...
; PRAGMA-TC-TOO-LARGE-NEXT: Trying loop peeling...
; PRAGMA-TC-TOO-LARGE-NEXT: Trying partial unroll...
; PRAGMA-TC-TOO-LARGE-NEXT:  Unrolled size exceeds threshold; reducing count from {{[0-9]+}} to {{[0-9]+}}.
; PRAGMA-TC-TOO-LARGE-NEXT:   Partially unrolling with count: 2500
; PRAGMA-TC-TOO-LARGE-NEXT: Partial unroll instead of full: unrolled size too large. Unrolling 2500 times instead of 5000.
; PRAGMA-TC-TOO-LARGE-NEXT:remark: <unknown>:0:0: unable to fully unroll loop as directed by full unroll pragma because unrolled size is too large
; PRAGMA-TC-TOO-LARGE-NEXT:  Exiting block %for.body: TripCount=5000, TripMultiple=0, BreakoutTrip=0
; PRAGMA-TC-TOO-LARGE-NEXT:UNROLLING loop %for.body by 2500!
; PRAGMA-TC-TOO-LARGE-NEXT:remark: <unknown>:0:0: unrolled loop by a factor of 2500

define i32 @partial_instead_of_full(ptr %A) {
entry:
  br label %for.body

for.body:
  %i = phi i32 [ 0, %entry ], [ %inc, %for.body ]
  %sum = phi i32 [ 0, %entry ], [ %add, %for.body ]
  %arrayidx = getelementptr inbounds i32, ptr %A, i32 %i
  %load = load i32, ptr %arrayidx
  %add = add i32 %sum, %load
  %inc = add i32 %i, 1
  %cmp = icmp ult i32 %inc, 5000
  br i1 %cmp, label %for.body, label %exit, !llvm.loop !0

exit:
  ret i32 %add
}

; NO-PROFIT-LABEL:Loop Unroll: F[no_profitable_count] Loop %for.body (depth=1)
; NO-PROFIT-NEXT:Loop Size = 6
; NO-PROFIT-NEXT: Computing unroll count: TripCount=100, MaxTripCount=0, TripMultiple=100
; NO-PROFIT-NEXT: Trying pragma unroll...
; NO-PROFIT-NEXT: Trying full unroll...
; NO-PROFIT-NEXT:  Unrolled size {{[0-9]+}} exceeds threshold {{[0-9]+}}; checking for cost benefit.
; NO-PROFIT-NEXT:   Not analyzing loop cost: trip count too large.
; NO-PROFIT-NEXT:  Skipping: cost analysis unavailable.
; NO-PROFIT-NEXT: Trying upper-bound unroll...
; NO-PROFIT-NEXT: Trying loop peeling...
; NO-PROFIT-NEXT: Trying partial unroll...
; NO-PROFIT-NEXT:  Unrolled size exceeds threshold; reducing count from {{[0-9]+}} to {{[0-9]+}}.
; NO-PROFIT-NEXT:  Will not partially unroll: no profitable count.
; NO-PROFIT-NEXT:   Partially unrolling with count: 0
; NO-PROFIT-NEXT:Not unrolling: no viable strategy found.

define i32 @no_profitable_count(ptr %A) {
entry:
  br label %for.body

for.body:
  %i = phi i32 [ 0, %entry ], [ %inc, %for.body ]
  %sum = phi i32 [ 0, %entry ], [ %add, %for.body ]
  %arrayidx = getelementptr inbounds i32, ptr %A, i32 %i
  %load = load i32, ptr %arrayidx
  %add = add i32 %sum, %load
  %inc = add i32 %i, 1
  %cmp = icmp ult i32 %inc, 100
  br i1 %cmp, label %for.body, label %exit

exit:
  ret i32 %add
}

; CHECK-LABEL:Loop Unroll: F[extended_convergence] Loop %for.body (depth=1)
; CHECK-NEXT: Not unrolling: contains convergent operations.
; CHECK-NEXT:remark: <unknown>:0:0: unable to unroll loop: contains convergent operations

declare void @convergent_func() convergent
declare token @llvm.experimental.convergence.anchor()

define i32 @extended_convergence(ptr %A, i32 %n) {
entry:
  br label %for.body, !llvm.loop !1

for.body:
  %i = phi i32 [ 0, %entry ], [ %inc, %for.body ]
  %sum = phi i32 [ 0, %entry ], [ %add, %for.body ]
  %tok = call token @llvm.experimental.convergence.anchor()
  %arrayidx = getelementptr inbounds i32, ptr %A, i32 %i
  %load = load i32, ptr %arrayidx
  %add = add i32 %sum, %load
  %inc = add i32 %i, 1
  %cmp = icmp slt i32 %inc, %n
  br i1 %cmp, label %for.body, label %exit, !llvm.loop !1

exit:
  ; Using convergence token outside the loop creates ExtendedLoop convergence
  call void @convergent_func() [ "convergencectrl"(token %tok) ]
  ret i32 %add
}

; CHECK-LABEL:Loop Unroll: F[noduplicate_prevents_unroll] Loop %for.body (depth=1)
; CHECK-NEXT: Not unrolling: contains non-duplicatable instructions.
; CHECK-NEXT:remark: <unknown>:0:0: unable to unroll loop: contains non-duplicatable instructions

declare void @noduplicate_func() noduplicate

define i32 @noduplicate_prevents_unroll(ptr %A) {
entry:
  br label %for.body

for.body:
  %i = phi i32 [ 0, %entry ], [ %inc, %for.body ]
  %sum = phi i32 [ 0, %entry ], [ %add, %for.body ]
  %arrayidx = getelementptr inbounds i32, ptr %A, i32 %i
  %load = load i32, ptr %arrayidx
  ; noduplicate attribute prevents loop unrolling
  call void @noduplicate_func()
  %add = add i32 %sum, %load
  %inc = add i32 %i, 1
  %cmp = icmp ult i32 %inc, 8
  br i1 %cmp, label %for.body, label %exit, !llvm.loop !1

exit:
  ret i32 %add
}

; UNROLL-AS-DIRECTED-FAIL-LABEL:Loop Unroll: F[unroll_as_directed_fail] Loop %for.body (depth=1)
; UNROLL-AS-DIRECTED-FAIL-NEXT:Loop Size = 19
; UNROLL-AS-DIRECTED-FAIL-NEXT: Computing unroll count: TripCount=100, MaxTripCount=0, TripMultiple=100
; UNROLL-AS-DIRECTED-FAIL-NEXT: Explicit unroll requested: pragma-enable
; UNROLL-AS-DIRECTED-FAIL-NEXT: Trying pragma unroll...
; UNROLL-AS-DIRECTED-FAIL-NEXT: Trying full unroll...
; UNROLL-AS-DIRECTED-FAIL-NEXT:  Unrolled size {{[0-9]+}} exceeds threshold {{[0-9]+}}; checking for cost benefit.
; UNROLL-AS-DIRECTED-FAIL-NEXT:   Not analyzing loop cost: trip count too large.
; UNROLL-AS-DIRECTED-FAIL-NEXT:  Skipping: cost analysis unavailable.
; UNROLL-AS-DIRECTED-FAIL-NEXT: Trying upper-bound unroll...
; UNROLL-AS-DIRECTED-FAIL-NEXT: Trying loop peeling...
; UNROLL-AS-DIRECTED-FAIL-NEXT: Trying partial unroll...
; UNROLL-AS-DIRECTED-FAIL-NEXT:  Unrolled size exceeds threshold; reducing count from {{[0-9]+}} to 0.
; UNROLL-AS-DIRECTED-FAIL-NEXT:  Will not partially unroll: no profitable count.
; UNROLL-AS-DIRECTED-FAIL-NEXT:   Partially unrolling with count: 0
; UNROLL-AS-DIRECTED-FAIL-NEXT: Not unrolling as directed: unrolled size too large.
; UNROLL-AS-DIRECTED-FAIL-NEXT:remark: <unknown>:0:0: unable to unroll loop as directed by unroll pragma because unrolled size is too large

define i32 @unroll_as_directed_fail(ptr %A, ptr %B, ptr %C, ptr %D) {
entry:
  br label %for.body

for.body:
  %i = phi i32 [ 0, %entry ], [ %inc, %for.body ]
  %sum = phi i32 [ 0, %entry ], [ %add8, %for.body ]
  %idx1 = add i32 %i, 0
  %arrayidx1 = getelementptr inbounds i32, ptr %A, i32 %idx1
  %load1 = load i32, ptr %arrayidx1
  %idx2 = add i32 %i, 1
  %arrayidx2 = getelementptr inbounds i32, ptr %B, i32 %idx2
  %load2 = load i32, ptr %arrayidx2
  %idx3 = add i32 %i, 2
  %arrayidx3 = getelementptr inbounds i32, ptr %C, i32 %idx3
  %load3 = load i32, ptr %arrayidx3
  %idx4 = add i32 %i, 3
  %arrayidx4 = getelementptr inbounds i32, ptr %D, i32 %idx4
  %load4 = load i32, ptr %arrayidx4
  %add1 = add i32 %sum, %load1
  %add2 = add i32 %add1, %load2
  %add3 = add i32 %add2, %load3
  %add8 = add i32 %add3, %load4
  %inc = add i32 %i, 1
  %cmp = icmp ult i32 %inc, 100
  br i1 %cmp, label %for.body, label %exit, !llvm.loop !1

exit:
  ret i32 %add8
}

; UNROLL-AS-DIRECTED-FAIL-LABEL:Loop Unroll: F[full_unroll_as_directed_fail] Loop %for.body (depth=1)
; UNROLL-AS-DIRECTED-FAIL-NEXT:Loop Size = 19
; UNROLL-AS-DIRECTED-FAIL-NEXT: Computing unroll count: TripCount=100, MaxTripCount=0, TripMultiple=100
; UNROLL-AS-DIRECTED-FAIL-NEXT: Explicit unroll requested: pragma-full
; UNROLL-AS-DIRECTED-FAIL-NEXT: Trying pragma unroll...
; UNROLL-AS-DIRECTED-FAIL-NEXT:   Won't unroll; trip count is too large.
; UNROLL-AS-DIRECTED-FAIL-NEXT:remark: <unknown>:0:0: unable to fully unroll loop: trip count 100 exceeds limit 10
; UNROLL-AS-DIRECTED-FAIL-NEXT: Trying full unroll...
; UNROLL-AS-DIRECTED-FAIL-NEXT:  Unrolled size {{[0-9]+}} exceeds threshold {{[0-9]+}}; checking for cost benefit.
; UNROLL-AS-DIRECTED-FAIL-NEXT:   Not analyzing loop cost: trip count too large.
; UNROLL-AS-DIRECTED-FAIL-NEXT:  Skipping: cost analysis unavailable.
; UNROLL-AS-DIRECTED-FAIL-NEXT:remark: <unknown>:0:0: unable to fully unroll loop: estimated unrolled size {{[0-9]+}} exceeds threshold {{[0-9]+}}
; UNROLL-AS-DIRECTED-FAIL-NEXT: Trying upper-bound unroll...
; UNROLL-AS-DIRECTED-FAIL-NEXT: Trying loop peeling...
; UNROLL-AS-DIRECTED-FAIL-NEXT: Trying partial unroll...
; UNROLL-AS-DIRECTED-FAIL-NEXT:  Unrolled size exceeds threshold; reducing count from {{[0-9]+}} to 0.
; UNROLL-AS-DIRECTED-FAIL-NEXT:  Will not partially unroll: no profitable count.
; UNROLL-AS-DIRECTED-FAIL-NEXT:   Partially unrolling with count: 0
; UNROLL-AS-DIRECTED-FAIL-NEXT: Not unrolling as directed: unrolled size too large.
; UNROLL-AS-DIRECTED-FAIL-NEXT:remark: <unknown>:0:0: unable to unroll loop as directed by unroll pragma because unrolled size is too large

define i32 @full_unroll_as_directed_fail(ptr %A, ptr %B, ptr %C, ptr %D) {
entry:
  br label %for.body

for.body:
  %i = phi i32 [ 0, %entry ], [ %inc, %for.body ]
  %sum = phi i32 [ 0, %entry ], [ %add8, %for.body ]
  %idx1 = add i32 %i, 0
  %arrayidx1 = getelementptr inbounds i32, ptr %A, i32 %idx1
  %load1 = load i32, ptr %arrayidx1
  %idx2 = add i32 %i, 1
  %arrayidx2 = getelementptr inbounds i32, ptr %B, i32 %idx2
  %load2 = load i32, ptr %arrayidx2
  %idx3 = add i32 %i, 2
  %arrayidx3 = getelementptr inbounds i32, ptr %C, i32 %idx3
  %load3 = load i32, ptr %arrayidx3
  %idx4 = add i32 %i, 3
  %arrayidx4 = getelementptr inbounds i32, ptr %D, i32 %idx4
  %load4 = load i32, ptr %arrayidx4
  %add1 = add i32 %sum, %load1
  %add2 = add i32 %add1, %load2
  %add3 = add i32 %add2, %load3
  %add8 = add i32 %add3, %load4
  %inc = add i32 %i, 1
  %cmp = icmp ult i32 %inc, 100
  br i1 %cmp, label %for.body, label %exit, !llvm.loop !0

exit:
  ret i32 %add8
}

; CHECK-LABEL:Loop Unroll: F[indirectbr_loop] Loop %for.body (depth=1)
; CHECK-NEXT: Not unrolling loop which is not in loop-simplify form.
; CHECK-NEXT:remark: <unknown>:0:0: unable to unroll loop: not in loop-simplify form

define i32 @indirectbr_loop(ptr %A, ptr %target) {
entry:
  indirectbr ptr %target, [label %for.body, label %exit]

for.body:
  %i = phi i32 [ 0, %entry ], [ %inc, %for.body ]
  %sum = phi i32 [ 0, %entry ], [ %add, %for.body ]
  %arrayidx = getelementptr inbounds i32, ptr %A, i32 %i
  %load = load i32, ptr %arrayidx
  %add = add i32 %sum, %load
  %inc = add i32 %i, 1
  %cmp = icmp ult i32 %inc, 10
  br i1 %cmp, label %for.body, label %exit, !llvm.loop !1

exit:
  %result = phi i32 [ 0, %entry ], [ %add, %for.body ]
  ret i32 %result
}

; USER-COUNT-EXCEED-LABEL:Loop Unroll: F[user_count_exceed] Loop %for.body (depth=1)
; USER-COUNT-EXCEED-NEXT:Loop Size = 5
; USER-COUNT-EXCEED-NEXT: Computing unroll count: TripCount=16, MaxTripCount=0, TripMultiple=16
; USER-COUNT-EXCEED-NEXT: Explicit unroll requested: user-count
; USER-COUNT-EXCEED-NEXT: Trying pragma unroll...
; USER-COUNT-EXCEED-NEXT:  Not unrolling with user count 8: exceeds threshold.
; USER-COUNT-EXCEED-NEXT: Trying full unroll...
; USER-COUNT-EXCEED-NEXT:  Unrolling: size 50 < threshold 16384.
; USER-COUNT-EXCEED-NEXT:  Exiting block %for.body: TripCount=16, TripMultiple=0, BreakoutTrip=0
; USER-COUNT-EXCEED-NEXT:COMPLETELY UNROLLING loop %for.body with trip count 16!
; USER-COUNT-EXCEED-NEXT:remark: <unknown>:0:0: completely unrolled loop with 16 iterations

define void @user_count_exceed(ptr %A) {
entry:
  br label %for.body

for.body:
  %i = phi i32 [ 0, %entry ], [ %inc, %for.body ]
  %arrayidx = getelementptr inbounds i32, ptr %A, i32 %i
  store i32 %i, ptr %arrayidx
  %inc = add i32 %i, 1
  %cmp = icmp ult i32 %inc, 16
  br i1 %cmp, label %for.body, label %exit

exit:
  ret void
}

; CHECK-LABEL:Loop Unroll: F[inline_prevents_unroll] Loop %for.body (depth=1)
; CHECK-NEXT:Loop Size = 8
; CHECK-NEXT: Not unrolling loop with inlinable calls.
; CHECK-NEXT:remark: <unknown>:0:0: unable to unroll loop: contains inlinable calls

; Internal function with single use - this is an inline candidate
define internal i32 @single_use_helper(i32 %x) {
  %add = add i32 %x, 42
  ret i32 %add
}

define i32 @inline_prevents_unroll(ptr %A) {
entry:
  br label %for.body

for.body:
  %i = phi i32 [ 0, %entry ], [ %inc, %for.body ]
  %sum = phi i32 [ 0, %entry ], [ %add, %for.body ]
  %arrayidx = getelementptr inbounds i32, ptr %A, i32 %i
  %load = load i32, ptr %arrayidx
  %helper_result = call i32 @single_use_helper(i32 %load)
  %add = add i32 %sum, %helper_result
  %inc = add i32 %i, 1
  %cmp = icmp ult i32 %inc, 10
  br i1 %cmp, label %for.body, label %exit, !llvm.loop !11

exit:
  ret i32 %add
}

; NO-REMAINDER-LABEL:Loop Unroll: F[small_max_trip_count] Loop %for.body (depth=1)
; NO-REMAINDER-NEXT:Loop Size = 5
; NO-REMAINDER-NEXT: Computing unroll count: TripCount=0, MaxTripCount=5, TripMultiple=1
; NO-REMAINDER-NEXT: Explicit unroll requested: pragma-count(4)
; NO-REMAINDER-NEXT: Trying pragma unroll...
; NO-REMAINDER-NEXT:  Not unrolling with pragma count 4: remainder not allowed, count does not divide trip multiple 1.
; NO-REMAINDER-NEXT:remark: <unknown>:0:0: unable to unroll loop with count 4: remainder loop is restricted and count does not divide trip multiple 1
; NO-REMAINDER-NEXT: Trying full unroll...
; NO-REMAINDER-NEXT: Trying upper-bound unroll...
; NO-REMAINDER-NEXT: Trying loop peeling...
; NO-REMAINDER-NEXT: Trying partial unroll...
; NO-REMAINDER-NEXT: Trying runtime unroll...
; NO-REMAINDER-NEXT: Not runtime unrolling: max trip count {{[0-9]+}} is small (< 8) and not forced.
; NO-REMAINDER-NEXT:remark: <unknown>:0:0: unable to runtime unroll loop: max trip count {{[0-9]+}} is too small (< {{[0-9]+}})
; NO-REMAINDER-NEXT:Not unrolling: no viable strategy found.

define void @small_max_trip_count(ptr %A, i32 %n) {
entry:
  ; Clamp n to max of 5, so MaxTripCount will be 5 (< default MaxUpperBound of 8)
  %clamped = call i32 @llvm.umin.i32(i32 %n, i32 5)
  %cmp.entry = icmp ugt i32 %clamped, 0
  br i1 %cmp.entry, label %for.body, label %exit

for.body:
  %i = phi i32 [ 0, %entry ], [ %inc, %for.body ]
  %arrayidx = getelementptr inbounds i32, ptr %A, i32 %i
  store i32 %i, ptr %arrayidx
  %inc = add nuw nsw i32 %i, 1
  %cmp = icmp ult i32 %inc, %clamped
  br i1 %cmp, label %for.body, label %exit, !llvm.loop !13

exit:
  ret void
}

declare i32 @llvm.umin.i32(i32, i32)

; NO-STRATEGY-LABEL:Loop Unroll: F[no_strategy_pragma] Loop %for.body (depth=1)
; NO-STRATEGY-NEXT:Loop Size = 5
; NO-STRATEGY-NEXT: Computing unroll count: TripCount=0, MaxTripCount={{[0-9]+}}, TripMultiple=1
; NO-STRATEGY-NEXT: Explicit unroll requested: pragma-enable
; NO-STRATEGY-NEXT: Trying pragma unroll...
; NO-STRATEGY-NEXT: Trying full unroll...
; NO-STRATEGY-NEXT: Trying upper-bound unroll...
; NO-STRATEGY-NEXT: Trying loop peeling...
; NO-STRATEGY-NEXT: Trying partial unroll...
; NO-STRATEGY-NEXT: Trying runtime unroll...
; NO-STRATEGY-NEXT:Not unrolling: no viable strategy found.
; NO-STRATEGY-NEXT:remark: <unknown>:0:0: unable to unroll loop: no viable unroll count found

define void @no_strategy_pragma(ptr %A, i32 %n) {
entry:
  %cmp.entry = icmp ugt i32 %n, 0
  br i1 %cmp.entry, label %for.body, label %exit

for.body:
  %i = phi i32 [ 0, %entry ], [ %inc, %for.body ]
  %arrayidx = getelementptr inbounds i32, ptr %A, i32 %i
  store i32 %i, ptr %arrayidx
  %inc = add nuw nsw i32 %i, 1
  %cmp = icmp ult i32 %inc, %n
  br i1 %cmp, label %for.body, label %exit, !llvm.loop !12

exit:
  ret void
}

; We get contradictory remarks here: full unroll is blocked by -unroll-full-max-count=10,
; but partial unroll picks count=20 (the full trip count) anyway. This is a test artifact.
; MAX-COUNT-10-LABEL:Loop Unroll: F[tc_exceeds_max_ore] Loop %for.body (depth=1)
; MAX-COUNT-10-NEXT:Loop Size = 6
; MAX-COUNT-10-NEXT: Computing unroll count: TripCount=20, MaxTripCount=0, TripMultiple=20
; MAX-COUNT-10-NEXT: Explicit unroll requested: pragma-full
; MAX-COUNT-10-NEXT: Trying pragma unroll...
; MAX-COUNT-10-NEXT:   Won't unroll; trip count is too large.
; MAX-COUNT-10-NEXT:remark: <unknown>:0:0: unable to fully unroll loop: trip count 20 exceeds limit 10
; MAX-COUNT-10-NEXT: Trying full unroll...
; MAX-COUNT-10-NEXT:  Not unrolling: trip count 20 exceeds max count 10.
; MAX-COUNT-10-NEXT:remark: <unknown>:0:0: unable to fully unroll loop: trip count 20 exceeds maximum full unroll count 10
; MAX-COUNT-10-NEXT: Trying upper-bound unroll...
; MAX-COUNT-10-NEXT: Trying loop peeling...
; MAX-COUNT-10-NEXT: Trying partial unroll...
; MAX-COUNT-10-NEXT:   Partially unrolling with count: 20
; MAX-COUNT-10-NEXT:  Exiting block %for.body: TripCount=20, TripMultiple=0, BreakoutTrip=0
; MAX-COUNT-10-NEXT:COMPLETELY UNROLLING loop %for.body with trip count 20!
; MAX-COUNT-10-NEXT:remark: <unknown>:0:0: completely unrolled loop with 20 iterations

define i32 @tc_exceeds_max_ore(ptr %A) {
entry:
  br label %for.body

for.body:
  %i = phi i32 [ 0, %entry ], [ %inc, %for.body ]
  %sum = phi i32 [ 0, %entry ], [ %add, %for.body ]
  %arrayidx = getelementptr inbounds i32, ptr %A, i32 %i
  %load = load i32, ptr %arrayidx
  %add = add i32 %sum, %load
  %inc = add i32 %i, 1
  %cmp = icmp ult i32 %inc, 20
  br i1 %cmp, label %for.body, label %exit, !llvm.loop !0

exit:
  ret i32 %add
}

; CHECK-LABEL:Loop Unroll: F[caller_with_inlined_loop] Loop %for.body.i (depth=1)
; CHECK-NEXT:remark: inlined.c:5:3: loop is from inlined function; call site is at caller.c:10
; CHECK-NEXT:Loop Size = 6
; CHECK-NEXT: Computing unroll count: TripCount=4, MaxTripCount=0, TripMultiple=4
; CHECK-NEXT: Explicit unroll requested: pragma-enable
; CHECK-NEXT: Trying pragma unroll...
; CHECK-NEXT: Trying full unroll...
; CHECK-NEXT:  Unrolling: size 18 < threshold 16384.
; CHECK-NEXT:  Exiting block %for.body.i: TripCount=4, TripMultiple=0, BreakoutTrip=0
; CHECK-NEXT:COMPLETELY UNROLLING loop %for.body.i with trip count 4!
; CHECK-NEXT:remark: inlined.c:5:3: completely unrolled loop with 4 iterations

define i32 @caller_with_inlined_loop(ptr %A) !dbg !20 {
entry:
  br label %for.body.i, !dbg !21

for.body.i:
  %i = phi i32 [ 0, %entry ], [ %inc, %for.body.i ]
  %sum = phi i32 [ 0, %entry ], [ %add, %for.body.i ]
  %arrayidx = getelementptr inbounds i32, ptr %A, i32 %i, !dbg !22
  %load = load i32, ptr %arrayidx, !dbg !22
  %add = add i32 %sum, %load, !dbg !22
  %inc = add i32 %i, 1, !dbg !22
  %cmp = icmp ult i32 %inc, 4, !dbg !22
  br i1 %cmp, label %for.body.i, label %exit, !dbg !22, !llvm.loop !23

exit:
  ret i32 %add, !dbg !21
}

; Same contradiction pattern as above: the low -unroll-threshold=20 -pragma-unroll-threshold=20
; cause cost analysis to reject full unroll, but partial unroll uses different heuristics and
; picks count=8 (the full trip count). Only happens with these artificial test flags.
; COST-NOT-PROFITABLE-LABEL:Loop Unroll: F[cost_not_profitable] Loop %for.body (depth=1)
; COST-NOT-PROFITABLE-NEXT:Loop Size = 14
; COST-NOT-PROFITABLE-NEXT: Computing unroll count: TripCount=8, MaxTripCount=0, TripMultiple=8
; COST-NOT-PROFITABLE-NEXT: Explicit unroll requested: pragma-enable
; COST-NOT-PROFITABLE-NEXT: Trying pragma unroll...
; COST-NOT-PROFITABLE-NEXT: Trying full unroll...
; COST-NOT-PROFITABLE-NEXT:  Unrolled size {{[0-9]+}} exceeds threshold {{[0-9]+}}; checking for cost benefit.
; COST-NOT-PROFITABLE-NEXT:    Starting LoopUnroll profitability analysis...
; COST-NOT-PROFITABLE:    Analysis finished:
; COST-NOT-PROFITABLE-NEXT:    UnrolledCost: {{[0-9]+}}, RolledDynamicCost: {{[0-9]+}}
; COST-NOT-PROFITABLE-NEXT:  Not unrolling: cost {{[0-9]+}} >= boosted threshold {{[0-9]+}}.
; COST-NOT-PROFITABLE-NEXT: Trying upper-bound unroll...
; COST-NOT-PROFITABLE-NEXT: Trying loop peeling...
; COST-NOT-PROFITABLE-NEXT: Trying partial unroll...
; COST-NOT-PROFITABLE-NEXT:   Partially unrolling with count: 8
; COST-NOT-PROFITABLE-NEXT:  Exiting block %for.body: TripCount=8, TripMultiple=0, BreakoutTrip=0
; COST-NOT-PROFITABLE-NEXT:COMPLETELY UNROLLING loop %for.body with trip count 8!
; COST-NOT-PROFITABLE-NEXT:remark: <unknown>:0:0: completely unrolled loop with 8 iterations

define i32 @cost_not_profitable(ptr %A, ptr %B, ptr %C) {
entry:
  br label %for.body

for.body:
  %i = phi i32 [ 0, %entry ], [ %inc, %for.body ]
  %sum = phi i32 [ 0, %entry ], [ %add3, %for.body ]
  %arrayidx = getelementptr inbounds i32, ptr %A, i32 %i
  %load1 = load i32, ptr %arrayidx
  %arrayidx2 = getelementptr inbounds i32, ptr %B, i32 %i
  %load2 = load i32, ptr %arrayidx2
  %arrayidx3 = getelementptr inbounds i32, ptr %C, i32 %i
  %load3 = load i32, ptr %arrayidx3
  %mul1 = mul i32 %load1, %load2
  %mul2 = mul i32 %mul1, %load3
  %add1 = add i32 %sum, %mul2
  %add2 = add i32 %add1, %load1
  %add3 = add i32 %add2, %load2
  %inc = add i32 %i, 1
  %cmp = icmp ult i32 %inc, 8
  br i1 %cmp, label %for.body, label %exit, !llvm.loop !1

exit:
  ret i32 %add3
}

; UPPER-BOUND-HEURISTIC-LABEL:Loop Unroll: F[upper_bound_heuristic] Loop %for.body (depth=1)
; UPPER-BOUND-HEURISTIC-NEXT:Loop Size = 6
; UPPER-BOUND-HEURISTIC-NEXT: Computing unroll count: TripCount=0, MaxTripCount=3, TripMultiple=1
; UPPER-BOUND-HEURISTIC-NEXT: Trying pragma unroll...
; UPPER-BOUND-HEURISTIC-NEXT: Trying full unroll...
; UPPER-BOUND-HEURISTIC-NEXT: Trying upper-bound unroll...
; UPPER-BOUND-HEURISTIC-NEXT:  Unrolling: size {{[0-9]+}} < threshold {{[0-9]+}}.
; UPPER-BOUND-HEURISTIC-NEXT:Attempting full unroll with upper bound trip count 3
; UPPER-BOUND-HEURISTIC-NEXT:remark: <unknown>:0:0: attempting full unroll using upper bound trip count 3
; UPPER-BOUND-HEURISTIC-NEXT:  Exiting block %for.body: TripCount=0, TripMultiple=1, BreakoutTrip=1
; UPPER-BOUND-HEURISTIC-NEXT:COMPLETELY UNROLLING loop %for.body with trip count 3!
; UPPER-BOUND-HEURISTIC-NEXT:remark: <unknown>:0:0: completely unrolled loop with 3 iterations

define i32 @upper_bound_heuristic(ptr %A, i32 %n) {
entry:
  ; Clamp n to max of 3, so MaxTripCount will be 3
  %n.clamped = and i32 %n, 3
  br label %for.body

for.body:
  %i = phi i32 [ 0, %entry ], [ %inc, %for.body ]
  %sum = phi i32 [ 0, %entry ], [ %add, %for.body ]
  %arrayidx = getelementptr inbounds i32, ptr %A, i32 %i
  %load = load i32, ptr %arrayidx
  %add = add i32 %sum, %load
  %inc = add i32 %i, 1
  %cmp = icmp ult i32 %inc, %n.clamped
  br i1 %cmp, label %for.body, label %exit

exit:
  ret i32 %add
}

; FULL-COST-NOT-PROFITABLE-LABEL:Loop Unroll: F[pragma_full_cost_not_profitable] Loop %for.body (depth=1)
; FULL-COST-NOT-PROFITABLE-NEXT:Loop Size = 9
; FULL-COST-NOT-PROFITABLE-NEXT: Computing unroll count: TripCount=10, MaxTripCount=0, TripMultiple=10
; FULL-COST-NOT-PROFITABLE-NEXT: Explicit unroll requested: pragma-full
; FULL-COST-NOT-PROFITABLE-NEXT: Trying pragma unroll...
; FULL-COST-NOT-PROFITABLE-NEXT:   Won't unroll; trip count is too large.
; FULL-COST-NOT-PROFITABLE-NEXT:remark: <unknown>:0:0: unable to fully unroll loop: trip count 10 exceeds limit 8
; FULL-COST-NOT-PROFITABLE-NEXT: Trying full unroll...
; FULL-COST-NOT-PROFITABLE-NEXT:  Unrolled size {{[0-9]+}} exceeds threshold 20; checking for cost benefit.
; FULL-COST-NOT-PROFITABLE-NEXT:    Starting LoopUnroll profitability analysis...
; FULL-COST-NOT-PROFITABLE:    Analysis finished:
; FULL-COST-NOT-PROFITABLE-NEXT:    UnrolledCost: {{[0-9]+}}, RolledDynamicCost: {{[0-9]+}}
; FULL-COST-NOT-PROFITABLE-NEXT:  Not unrolling: cost {{[0-9]+}} >= boosted threshold {{[0-9]+}}.
; FULL-COST-NOT-PROFITABLE-NEXT:remark: <unknown>:0:0: unable to fully unroll loop: estimated unrolled cost {{[0-9]+}} exceeds boosted threshold {{[0-9]+}}
; FULL-COST-NOT-PROFITABLE-NEXT: Trying upper-bound unroll...
; FULL-COST-NOT-PROFITABLE-NEXT: Trying loop peeling...
; FULL-COST-NOT-PROFITABLE-NEXT: Trying partial unroll...
; FULL-COST-NOT-PROFITABLE-NEXT:   Partially unrolling with count: 10
; FULL-COST-NOT-PROFITABLE-NEXT:  Exiting block %for.body: TripCount=10, TripMultiple=0, BreakoutTrip=0
; FULL-COST-NOT-PROFITABLE-NEXT:COMPLETELY UNROLLING loop %for.body with trip count 10!
; FULL-COST-NOT-PROFITABLE-NEXT:remark: <unknown>:0:0: completely unrolled loop with 10 iterations

define i32 @pragma_full_cost_not_profitable(ptr %A, ptr %B) {
entry:
  br label %for.body

for.body:
  %i = phi i32 [ 0, %entry ], [ %inc, %for.body ]
  %sum = phi i32 [ 0, %entry ], [ %add, %for.body ]
  %arrayidx = getelementptr inbounds i32, ptr %A, i32 %i
  %load = load i32, ptr %arrayidx
  %arrayidx2 = getelementptr inbounds i32, ptr %B, i32 %i
  %load2 = load i32, ptr %arrayidx2
  %mul = mul i32 %load, %load2
  %add = add i32 %sum, %mul
  %inc = add i32 %i, 1
  %cmp = icmp ult i32 %inc, 10
  br i1 %cmp, label %for.body, label %exit, !llvm.loop !0

exit:
  ret i32 %add
}

; =============================================================================
; Below are regression tests for edge cases in loop unrolling remarks.
; =============================================================================

; Test that a loop with multiple exits where one has a known trip count and
; another has an unknown trip count is NOT labeled as upper-bound unroll.
; CHECK-LABEL:Loop Unroll: F[multi_exit_known_and_unknown] Loop %for.header (depth=1)
; CHECK-NEXT:Loop Size = 8
; CHECK-NEXT: Computing unroll count: TripCount=5, MaxTripCount=0, TripMultiple=5
; CHECK-NEXT: Trying pragma unroll...
; CHECK-NEXT: Trying full unroll...
; CHECK-NEXT:  Unrolling: size {{[0-9]+}} < threshold {{[0-9]+}}.
; CHECK-NEXT:  Exiting block %for.header: TripCount=5, TripMultiple=0, BreakoutTrip=0
; CHECK-NEXT:  Exiting block %for.body: TripCount=0, TripMultiple=1, BreakoutTrip=1
; Note: This is a full unroll (not upper-bound) because we have a known trip count exit.
; CHECK-NEXT:COMPLETELY UNROLLING loop %for.header with trip count 5!
; CHECK-NEXT:remark: <unknown>:0:0: completely unrolled loop with 5 iterations

define i32 @multi_exit_known_and_unknown(ptr %A, i1 %cond) {
entry:
  br label %for.header

for.header:
  %i = phi i32 [ 0, %entry ], [ %inc, %for.latch ]
  %sum = phi i32 [ 0, %entry ], [ %add, %for.latch ]
  ; This exit has known trip count = 5
  %cmp = icmp ult i32 %i, 4
  br i1 %cmp, label %for.body, label %exit

for.body:
  %arrayidx = getelementptr inbounds i32, ptr %A, i32 %i
  %load = load i32, ptr %arrayidx
  %add = add i32 %sum, %load
  ; This exit has unknown trip count (depends on runtime condition)
  br i1 %cond, label %for.latch, label %exit

for.latch:
  %inc = add i32 %i, 1
  br label %for.header

exit:
  %result = phi i32 [ %sum, %for.header ], [ %add, %for.body ]
  ret i32 %result
}

; Test header-exiting while-style loop with partial unroll and remainder.
; The latch is NOT the exiting block (unconditional branch), but the header is.
; PARTIAL-UNROLL-LABEL:Loop Unroll: F[header_exit_with_remainder] Loop %while.header (depth=1)
; PARTIAL-UNROLL-NEXT:Loop Size = 8
; PARTIAL-UNROLL-NEXT: Computing unroll count: TripCount=11, MaxTripCount=0, TripMultiple=11
; PARTIAL-UNROLL-NEXT: Explicit unroll requested: user-count
; PARTIAL-UNROLL-NEXT: Trying pragma unroll...
; PARTIAL-UNROLL-NEXT:  Unrolling with user-specified count: 4
; PARTIAL-UNROLL-NEXT:Attempting unroll by factor 4 with remainder loop (trip count 11)
; PARTIAL-UNROLL-NEXT:remark: <unknown>:0:0: attempting unroll by factor 4 with remainder loop (trip count 11)
; PARTIAL-UNROLL-NEXT:  Exiting block %while.header: TripCount=11, TripMultiple=0, BreakoutTrip=3
; Note: Should say "with remainder" even though the latch is not the exiting block.
; PARTIAL-UNROLL-NEXT:UNROLLING loop %while.header by 4!
; PARTIAL-UNROLL-NEXT:remark: <unknown>:0:0: unrolled loop by a factor of 4

define i32 @header_exit_with_remainder(ptr %A) {
entry:
  br label %while.header

while.header:
  %i = phi i32 [ 0, %entry ], [ %inc, %while.latch ]
  %sum = phi i32 [ 0, %entry ], [ %add, %while.latch ]
  ; Exit is in the header (while-style loop)
  %cmp = icmp ult i32 %i, 10
  br i1 %cmp, label %while.body, label %exit

while.body:
  %arrayidx = getelementptr inbounds i32, ptr %A, i32 %i
  %load = load i32, ptr %arrayidx
  %add = add i32 %sum, %load
  br label %while.latch

while.latch:
  ; Latch is NOT an exiting block - unconditional branch
  %inc = add i32 %i, 1
  br label %while.header

exit:
  ret i32 %sum
}

; CHECK-LABEL:Loop Unroll: F[switch_exit_full_unroll_bug] Loop %loop (depth=1)
; CHECK-NEXT:Loop Size = 6
; CHECK-NEXT: Computing unroll count: TripCount=5, MaxTripCount=0, TripMultiple=5
; CHECK-NEXT: Trying pragma unroll...
; CHECK-NEXT: Trying full unroll...
; CHECK-NEXT:  Unrolling: size {{[0-9]+}} < threshold {{[0-9]+}}.
; CHECK-NEXT:COMPLETELY UNROLLING loop %loop with trip count 5!
; CHECK-NEXT:remark: <unknown>:0:0: completely unrolled loop with 5 iterations

define i32 @switch_exit_full_unroll_bug(ptr %A) {
entry:
  br label %loop

loop:
  %i = phi i32 [ 0, %entry ], [ %inc, %loop.latch ]
  %sum = phi i32 [ 0, %entry ], [ %add, %loop.latch ]
  %arrayidx = getelementptr inbounds i32, ptr %A, i32 %i
  %load = load i32, ptr %arrayidx
  %add = add i32 %sum, %load
  %inc = add i32 %i, 1
  ; Switch exit - not a branch, so won't be in ExitInfos
  switch i32 %i, label %loop.latch [
    i32 4, label %exit
  ]

loop.latch:
  br label %loop

exit:
  ret i32 %add
}

; PARTIAL-UNROLL-LABEL:Loop Unroll: F[switch_exit_partial_remainder_bug] Loop %loop (depth=1)
; PARTIAL-UNROLL-NEXT:Loop Size = 6
; PARTIAL-UNROLL-NEXT: Computing unroll count: TripCount=11, MaxTripCount=0, TripMultiple=11
; PARTIAL-UNROLL-NEXT: Explicit unroll requested: user-count
; PARTIAL-UNROLL-NEXT: Trying pragma unroll...
; PARTIAL-UNROLL-NEXT:  Unrolling with user-specified count: 4
; PARTIAL-UNROLL-NEXT:Attempting unroll by factor 4 with remainder loop (trip count 11)
; PARTIAL-UNROLL-NEXT:remark: <unknown>:0:0: attempting unroll by factor 4 with remainder loop (trip count 11)
; PARTIAL-UNROLL-NEXT:UNROLLING loop %loop by 4!
; PARTIAL-UNROLL-NEXT:remark: <unknown>:0:0: unrolled loop by a factor of 4

define i32 @switch_exit_partial_remainder_bug(ptr %A) {
entry:
  br label %loop

loop:
  %i = phi i32 [ 0, %entry ], [ %inc, %loop.latch ]
  %sum = phi i32 [ 0, %entry ], [ %add, %loop.latch ]
  %arrayidx = getelementptr inbounds i32, ptr %A, i32 %i
  %load = load i32, ptr %arrayidx
  %add = add i32 %sum, %load
  %inc = add i32 %i, 1
  ; Switch exit with trip count 11
  switch i32 %i, label %loop.latch [
    i32 10, label %exit
  ]

loop.latch:
  br label %loop

exit:
  ret i32 %add
}

; Metadata definitions
!0 = distinct !{!0, !5}
!1 = distinct !{!1, !6}
!2 = distinct !{!2, !7}
!3 = distinct !{!3, !8}
!4 = distinct !{!4, !9}
!5 = !{!"llvm.loop.unroll.full"}
!6 = !{!"llvm.loop.unroll.enable"}
!7 = !{!"llvm.loop.unroll.count", i32 4}
!8 = !{!"llvm.loop.unroll.disable"}
!9 = !{!"llvm.loop.unroll.runtime.disable"}
!10 = distinct !{!10, !7}
!11 = distinct !{!11, !6}
!12 = distinct !{!12, !6}
!13 = distinct !{!13, !7}

; Debug info for inlined loop test
!llvm.dbg.cu = !{!15}
!llvm.module.flags = !{!19}

!15 = distinct !DICompileUnit(language: DW_LANG_C99, file: !16, producer: "clang", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug)
!16 = !DIFile(filename: "caller.c", directory: "/tmp")
!17 = !DIFile(filename: "inlined.c", directory: "/tmp")
!18 = distinct !DISubprogram(name: "inlined_func", scope: !17, file: !17, line: 1, type: !24, isLocal: false, isDefinition: true, scopeLine: 1, isOptimized: true, unit: !15)
!19 = !{i32 2, !"Debug Info Version", i32 3}
!20 = distinct !DISubprogram(name: "caller_with_inlined_loop", scope: !16, file: !16, line: 8, type: !24, isLocal: false, isDefinition: true, scopeLine: 8, isOptimized: true, unit: !15)
!21 = !DILocation(line: 10, column: 3, scope: !20)
!22 = !DILocation(line: 5, column: 3, scope: !18, inlinedAt: !21)
!23 = distinct !{!23, !22, !6}
!24 = !DISubroutineType(types: !25)
!25 = !{null}
