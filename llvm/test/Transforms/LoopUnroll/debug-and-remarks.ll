; RUN: opt -disable-output -passes=loop-unroll -debug-only=loop-unroll -pass-remarks=loop-unroll \
; RUN:     -pass-remarks-missed=loop-unroll -pass-remarks-analysis=loop-unroll < %s 2>&1 \
; RUN:     | FileCheck %s --match-full-lines --strict-whitespace
; RUN: opt -disable-output -passes=loop-unroll -debug-only=loop-unroll -pass-remarks=loop-unroll \
; RUN:     -pass-remarks-missed=loop-unroll -pass-remarks-analysis=loop-unroll -unroll-allow-partial < %s 2>&1 \
; RUN:     | FileCheck %s --match-full-lines --strict-whitespace --check-prefix=PARTIAL-ALLOW
; RUN: opt -disable-output -passes=loop-unroll -debug-only=loop-unroll -pass-remarks=loop-unroll \
; RUN:     -pass-remarks-missed=loop-unroll -pass-remarks-analysis=loop-unroll -unroll-count=4 < %s 2>&1 \
; RUN:     | FileCheck %s --match-full-lines --strict-whitespace --check-prefix=USER-COUNT
; RUN: opt -disable-output -passes=loop-unroll -debug-only=loop-unroll -pass-remarks=loop-unroll \
; RUN:     -pass-remarks-missed=loop-unroll -pass-remarks-analysis=loop-unroll -unroll-count=9999 < %s 2>&1 \
; RUN:     | FileCheck %s --match-full-lines --strict-whitespace --check-prefix=USER-COUNT-EXCEED
; RUN: opt -disable-output -passes=loop-unroll -debug-only=loop-unroll -pass-remarks=loop-unroll \
; RUN:     -pass-remarks-missed=loop-unroll -pass-remarks-analysis=loop-unroll -unroll-peel-count=2 < %s 2>&1 \
; RUN:     | FileCheck %s --match-full-lines --strict-whitespace --check-prefix=EXPLICIT-PEEL
; RUN: opt -disable-output -passes=loop-unroll -debug-only=loop-unroll -pass-remarks=loop-unroll \
; RUN:     -pass-remarks-missed=loop-unroll -pass-remarks-analysis=loop-unroll -unroll-threshold=0 < %s 2>&1 \
; RUN:     | FileCheck %s --match-full-lines --strict-whitespace --check-prefix=ZERO-THRESH
; RUN: opt -disable-output -passes=loop-unroll -debug-only=loop-unroll -pass-remarks=loop-unroll \
; RUN:     -pass-remarks-missed=loop-unroll -pass-remarks-analysis=loop-unroll -unroll-full-max-count=2 < %s 2>&1 \
; RUN:     | FileCheck %s --match-full-lines --strict-whitespace --check-prefix=MAX-COUNT
; RUN: opt -disable-output -passes=loop-unroll -debug-only=loop-unroll -pass-remarks=loop-unroll \
; RUN:     -pass-remarks-missed=loop-unroll -pass-remarks-analysis=loop-unroll -unroll-allow-partial -unroll-partial-threshold=4 < %s 2>&1 \
; RUN:     | FileCheck %s --match-full-lines --strict-whitespace --check-prefix=PARTIAL-NOPROFIT
; RUN: opt -disable-output -passes=loop-unroll -debug-only=loop-unroll -pass-remarks=loop-unroll \
; RUN:     -pass-remarks-missed=loop-unroll -pass-remarks-analysis=loop-unroll -unroll-allow-remainder=false < %s 2>&1 \
; RUN:     | FileCheck %s --match-full-lines --strict-whitespace --check-prefix=PRAGMA-NOREMAINDER
; RUN: opt -disable-output -passes=loop-unroll -debug-only=loop-unroll -pass-remarks=loop-unroll \
; RUN:     -pass-remarks-missed=loop-unroll -pass-remarks-analysis=loop-unroll -unroll-remainder < %s 2>&1 \
; RUN:     | FileCheck %s --match-full-lines --strict-whitespace --check-prefix=REMAINDER
; RUN: opt -disable-output -passes=loop-unroll -debug-only=loop-unroll -pass-remarks=loop-unroll \
; RUN:     -pass-remarks-missed=loop-unroll -pass-remarks-analysis=loop-unroll -unroll-partial-threshold=9 < %s 2>&1 \
; RUN:     | FileCheck %s --match-full-lines --strict-whitespace --check-prefix=RUNTIME-NOPROFIT


; REQUIRES: asserts

; CHECK-LABEL:Loop Unroll: F[pragma_full_unroll_unknown_tc] Loop %for.body (depth=1)
; CHECK-NEXT:Loop Size = 6
; CHECK-NEXT: Computing unroll count: TripCount=0, MaxTripCount=2147483647, TripMultiple=1
; CHECK-NEXT: Explicit unroll requested: pragma-full
; CHECK-NEXT: Trying pragma unroll...
; CHECK-NEXT:  Not fully unrolling: unknown trip count.
; CHECK-NEXT:remark: <unknown>:0:0: may be unable to fully unroll loop: trip count is unknown
; CHECK-NEXT: Trying full unroll...
; CHECK-NEXT: Trying upper-bound unroll...
; CHECK-NEXT: Trying loop peeling...
; CHECK-NEXT: Trying partial unroll...
; CHECK-NEXT: Trying runtime unroll...
; CHECK-NEXT:  Will not try to unroll loop with runtime trip count because -unroll-runtime not given
; CHECK-NEXT: Not unrolling: no viable strategy found.
; CHECK-NEXT:remark: <unknown>:0:0: unable to unroll loop: no viable unroll count found

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

; CHECK-LABEL:Loop Unroll: F[full_unroll_cost_exceeds] Loop %for.body (depth=1)
; CHECK-NEXT:Loop Size = 6
; CHECK-NEXT: Computing unroll count: TripCount=100, MaxTripCount=0, TripMultiple=100
; CHECK-NEXT: Trying pragma unroll...
; CHECK-NEXT: Trying full unroll...
; CHECK-NEXT:  Unrolled size {{[0-9]+}} exceeds threshold {{[0-9]+}}; checking for cost benefit.
; CHECK-NEXT:   Not analyzing loop cost: trip count too large.
; CHECK-NEXT: Trying upper-bound unroll...
; CHECK-NEXT: Trying loop peeling...
; CHECK-NEXT: Trying partial unroll...
; CHECK-NEXT:  Will not try to unroll partially because -unroll-allow-partial not given
; CHECK-NEXT: Not unrolling: no viable strategy found.

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
  call void @noduplicate_func()
  %add = add i32 %sum, %load
  %inc = add i32 %i, 1
  %cmp = icmp ult i32 %inc, 8
  br i1 %cmp, label %for.body, label %exit, !llvm.loop !1

exit:
  ret i32 %add
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

; CHECK-LABEL:Loop Unroll: F[inline_prevents_unroll] Loop %for.body (depth=1)
; CHECK-NEXT:Loop Size = 8
; CHECK-NEXT: Not unrolling loop with inlinable calls.
; CHECK-NEXT:remark: <unknown>:0:0: unable to unroll loop: contains inlinable calls

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
  br i1 %cmp, label %for.body, label %exit, !llvm.loop !2

exit:
  ret i32 %add
}

; CHECK-LABEL:Loop Unroll: F[full_unroll_profitability_analysis] Loop %for.body (depth=1)
; CHECK-NEXT:Loop Size = {{[0-9]+}}
; CHECK-NEXT: Computing unroll count: TripCount=10, MaxTripCount=0, TripMultiple=10
; CHECK-NEXT: Trying pragma unroll...
; CHECK-NEXT: Trying full unroll...
; CHECK-NEXT:  Unrolled size {{[0-9]+}} exceeds threshold {{[0-9]+}}; checking for cost benefit.
; CHECK-NEXT:   Starting LoopUnroll profitability analysis...
; CHECK-NEXT:   Analyzing iteration 0
; CHECK-NEXT:   Analyzing iteration 1
; CHECK-NEXT:   Analyzing iteration 2
; CHECK-NEXT:   Analyzing iteration 3
; CHECK-NEXT:   Analyzing iteration 4
; CHECK-NEXT:   Analyzing iteration 5
; CHECK-NEXT:   Analyzing iteration 6
; CHECK-NEXT:   Analyzing iteration 7
; CHECK-NEXT:   Analyzing iteration 8
; CHECK-NEXT:   Analyzing iteration 9
; CHECK:   Analysis finished:
; CHECK-NEXT:   UnrolledCost: {{[0-9]+}}, RolledDynamicCost: {{[0-9]+}}
; CHECK-NEXT:  Profitable after cost analysis.
; CHECK-NEXT:  Exiting block %for.body: TripCount=10, TripMultiple=0, BreakoutTrip=0
; CHECK-NEXT:COMPLETELY UNROLLING loop %for.body with trip count 10!
; CHECK-NEXT:remark: <unknown>:0:0: completely unrolled loop with 10 iterations

define i32 @full_unroll_profitability_analysis(ptr %A, ptr %B) {
entry:
  br label %for.body

for.body:
  %i = phi i32 [ 0, %entry ], [ %inc, %for.body ]
  %sum = phi i32 [ 0, %entry ], [ %result, %for.body ]
  %idxA = getelementptr inbounds i32, ptr %A, i32 %i
  %loadA = load i32, ptr %idxA
  %idxB = getelementptr inbounds i32, ptr %B, i32 %i
  %loadB = load i32, ptr %idxB
  %mul1 = mul i32 %loadA, %loadB
  %add1 = add i32 %mul1, %loadA
  %mul2 = mul i32 %add1, %loadB
  %sub1 = sub i32 %mul2, %loadA
  %add2 = add i32 %sub1, %loadB
  %mul3 = mul i32 %add2, %loadA
  %sub2 = sub i32 %mul3, %loadB
  %xor1 = xor i32 %sub2, %loadA
  %or1 = or i32 %xor1, %loadB
  %result = add i32 %sum, %or1
  %inc = add i32 %i, 1
  %cmp = icmp ult i32 %inc, 10
  br i1 %cmp, label %for.body, label %exit

exit:
  ret i32 %result
}

; CHECK-LABEL:Loop Unroll: F[cost_exceed_boosted_threshold] Loop %for.body (depth=1)
; CHECK-NEXT:Loop Size = {{[0-9]+}}
; CHECK-NEXT: Computing unroll count: TripCount=10, MaxTripCount=0, TripMultiple=10
; CHECK-NEXT: Trying pragma unroll...
; CHECK-NEXT: Trying full unroll...
; CHECK-NEXT:  Unrolled size {{[0-9]+}} exceeds threshold {{[0-9]+}}; checking for cost benefit.
; CHECK-NEXT:   Starting LoopUnroll profitability analysis...
; CHECK-NEXT:   Analyzing iteration 0
; CHECK-NEXT:   Analyzing iteration 1
; CHECK-NEXT:   Analyzing iteration 2
; CHECK-NEXT:   Analyzing iteration 3
; CHECK-NEXT:   Analyzing iteration 4
; CHECK-NEXT:   Analyzing iteration 5
; CHECK-NEXT:   Analyzing iteration 6
; CHECK-NEXT:   Analyzing iteration 7
; CHECK-NEXT:   Analyzing iteration 8
; CHECK-NEXT:   Analyzing iteration 9
; CHECK:   Analysis finished:
; CHECK-NEXT:   UnrolledCost: {{[0-9]+}}, RolledDynamicCost: {{[0-9]+}}
; CHECK-NEXT:  Not unrolling: cost {{[0-9]+}} >= boosted threshold {{[0-9]+}}.
; CHECK-NEXT: Trying upper-bound unroll...
; CHECK-NEXT: Trying loop peeling...
; CHECK-NEXT: Trying partial unroll...
; CHECK-NEXT:  Will not try to unroll partially because -unroll-allow-partial not given
; CHECK-NEXT: Not unrolling: no viable strategy found.

define i32 @cost_exceed_boosted_threshold(ptr %A, ptr %B) {
entry:
  br label %for.body

for.body:
  %i = phi i32 [ 0, %entry ], [ %inc, %for.body ]
  %sum = phi i32 [ 0, %entry ], [ %result, %for.body ]
  %idxA = getelementptr inbounds i32, ptr %A, i32 %i
  %loadA = load i32, ptr %idxA
  %idxB = getelementptr inbounds i32, ptr %B, i32 %i
  %loadB = load i32, ptr %idxB
  %mul1 = mul i32 %loadA, %loadB
  %add1 = add i32 %mul1, %loadA
  %mul2 = mul i32 %add1, %loadB
  %sub1 = sub i32 %mul2, %loadA
  %add2 = add i32 %sub1, %loadB
  %mul3 = mul i32 %add2, %loadA
  %sub2 = sub i32 %mul3, %loadB
  %xor1 = xor i32 %sub2, %loadA
  %or1 = or i32 %xor1, %loadB
  %and1 = and i32 %or1, %loadA
  %shl1 = shl i32 %and1, 2
  %ashr1 = ashr i32 %shl1, 1
  %mul4 = mul i32 %ashr1, %loadB
  %add3 = add i32 %mul4, %loadA
  %xor2 = xor i32 %add3, %loadB
  %result = add i32 %sum, %xor2
  %inc = add i32 %i, 1
  %cmp = icmp ult i32 %inc, 10
  br i1 %cmp, label %for.body, label %exit

exit:
  ret i32 %result
}

; CHECK-LABEL:Loop Unroll: F[full_unroll_size_under_threshold] Loop %for.body (depth=1)
; CHECK-NEXT:Loop Size = 6
; CHECK-NEXT: Computing unroll count: TripCount=4, MaxTripCount=0, TripMultiple=4
; CHECK-NEXT: Trying pragma unroll...
; CHECK-NEXT: Trying full unroll...
; CHECK-NEXT:  Unrolling: size {{[0-9]+}} < threshold {{[0-9]+}}.
; CHECK-NEXT:  Exiting block %for.body: TripCount=4, TripMultiple=0, BreakoutTrip=0
; CHECK-NEXT:COMPLETELY UNROLLING loop %for.body with trip count 4!
; CHECK-NEXT:remark: <unknown>:0:0: completely unrolled loop with 4 iterations

define i32 @full_unroll_size_under_threshold(ptr %A) {
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

; CHECK-LABEL:Loop Unroll: F[pragma_full_known_tc] Loop %for.body (depth=1)
; CHECK-NEXT:Loop Size = 6
; CHECK-NEXT: Computing unroll count: TripCount=6, MaxTripCount=0, TripMultiple=6
; CHECK-NEXT: Explicit unroll requested: pragma-full
; CHECK-NEXT: Trying pragma unroll...
; CHECK-NEXT:  Fully unrolling with trip count: 6.
; CHECK-NEXT:  Exiting block %for.body: TripCount=6, TripMultiple=0, BreakoutTrip=0
; CHECK-NEXT:COMPLETELY UNROLLING loop %for.body with trip count 6!
; CHECK-NEXT:remark: <unknown>:0:0: completely unrolled loop with 6 iterations

define i32 @pragma_full_known_tc(ptr %A) {
entry:
  br label %for.body

for.body:
  %i = phi i32 [ 0, %entry ], [ %inc, %for.body ]
  %sum = phi i32 [ 0, %entry ], [ %add, %for.body ]
  %arrayidx = getelementptr inbounds i32, ptr %A, i32 %i
  %load = load i32, ptr %arrayidx
  %add = add i32 %sum, %load
  %inc = add i32 %i, 1
  %cmp = icmp ult i32 %inc, 6
  br i1 %cmp, label %for.body, label %exit, !llvm.loop !0

exit:
  ret i32 %add
}

; CHECK-LABEL:Loop Unroll: F[pragma_count_unroll] Loop %for.body (depth=1)
; CHECK-NEXT:Loop Size = 6
; CHECK-NEXT: Computing unroll count: TripCount=12, MaxTripCount=0, TripMultiple=12
; CHECK-NEXT: Explicit unroll requested: pragma-count(3)
; CHECK-NEXT: Trying pragma unroll...
; CHECK-NEXT:  Unrolling with pragma count: 3.
; CHECK-NEXT:  Exiting block %for.body: TripCount=12, TripMultiple=0, BreakoutTrip=0
; CHECK-NEXT:UNROLLING loop %for.body by 3!
; CHECK-NEXT:remark: <unknown>:0:0: unrolled loop by a factor of 3

define i32 @pragma_count_unroll(ptr %A) {
entry:
  br label %for.body

for.body:
  %i = phi i32 [ 0, %entry ], [ %inc, %for.body ]
  %sum = phi i32 [ 0, %entry ], [ %add, %for.body ]
  %arrayidx = getelementptr inbounds i32, ptr %A, i32 %i
  %load = load i32, ptr %arrayidx
  %add = add i32 %sum, %load
  %inc = add i32 %i, 1
  %cmp = icmp ult i32 %inc, 12
  br i1 %cmp, label %for.body, label %exit, !llvm.loop !5

exit:
  ret i32 %add
}

; CHECK-LABEL:Loop Unroll: F[no_viable_strategy] Loop %for.body (depth=1)
; CHECK-NEXT:Loop Size = 6
; CHECK-NEXT: Computing unroll count: TripCount=0, MaxTripCount=2147483647, TripMultiple=1
; CHECK-NEXT: Trying pragma unroll...
; CHECK-NEXT: Trying full unroll...
; CHECK-NEXT: Trying upper-bound unroll...
; CHECK-NEXT: Trying loop peeling...
; CHECK-NEXT: Trying partial unroll...
; CHECK-NEXT: Trying runtime unroll...
; CHECK-NEXT:  Will not try to unroll loop with runtime trip count because -unroll-runtime not given
; CHECK-NEXT: Not unrolling: no viable strategy found.

define i32 @no_viable_strategy(ptr %A, i32 %n) {
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

; CHECK-LABEL:Loop Unroll: F[disabled_by_metadata] Loop %for.body (depth=1)
; CHECK-NEXT: Not unrolling: transformation disabled by metadata.

define i32 @disabled_by_metadata(ptr %A) {
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
  br i1 %cmp, label %for.body, label %exit, !llvm.loop !7

exit:
  ret i32 %add
}

; CHECK-LABEL:Loop Unroll: F[upper_bound_unroll] Loop %for.body (depth=1)
; CHECK-NEXT:Loop Size = 6
; CHECK-NEXT: Computing unroll count: TripCount=0, MaxTripCount=3, TripMultiple=1
; CHECK-NEXT: Explicit unroll requested: pragma-enable
; CHECK-NEXT: Trying pragma unroll...
; CHECK-NEXT:  Unrolling with max trip count: 3.
; CHECK-NEXT:  Exiting block %for.body: TripCount=0, TripMultiple=1, BreakoutTrip=1
; CHECK-NEXT:COMPLETELY UNROLLING loop %for.body with trip count 3!
; CHECK-NEXT:remark: <unknown>:0:0: completely unrolled loop with 3 iterations

define i32 @upper_bound_unroll(ptr %A, i32 %n) {
entry:
  %masked = and i32 %n, 3
  %cmp.entry = icmp sgt i32 %masked, 0
  br i1 %cmp.entry, label %for.body, label %exit

for.body:
  %i = phi i32 [ 0, %entry ], [ %inc, %for.body ]
  %sum = phi i32 [ 0, %entry ], [ %add, %for.body ]
  %arrayidx = getelementptr inbounds i32, ptr %A, i32 %i
  %load = load i32, ptr %arrayidx
  %add = add i32 %sum, %load
  %inc = add i32 %i, 1
  %cmp = icmp slt i32 %inc, %masked
  br i1 %cmp, label %for.body, label %exit, !llvm.loop !9

exit:
  %result = phi i32 [ 0, %entry ], [ %add, %for.body ]
  ret i32 %result
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
; CHECK-NEXT:  Not runtime unrolling: max trip count 3 is small (<= 8) and not forced.
; CHECK-NEXT: Not unrolling: no viable strategy found.

define i32 @runtime_small_max_tc(ptr %A, i32 %n) {
entry:
  %masked = and i32 %n, 3
  %cmp.entry = icmp sgt i32 %masked, 0
  br i1 %cmp.entry, label %for.body, label %exit

for.body:
  %i = phi i32 [ 0, %entry ], [ %inc, %for.body ]
  %sum = phi i32 [ 0, %entry ], [ %add, %for.body ]
  %arrayidx = getelementptr inbounds i32, ptr %A, i32 %i
  %load = load i32, ptr %arrayidx
  %add = add i32 %sum, %load
  %inc = add i32 %i, 1
  %cmp = icmp slt i32 %inc, %masked
  br i1 %cmp, label %for.body, label %exit

exit:
  %result = phi i32 [ 0, %entry ], [ %add, %for.body ]
  ret i32 %result
}

; CHECK-LABEL:Loop Unroll: F[runtime_unroll_disabled_pragma] Loop %for.body (depth=1)
; CHECK-NEXT:Loop Size = 6
; CHECK-NEXT: Computing unroll count: TripCount=0, MaxTripCount=2147483647, TripMultiple=1
; CHECK-NEXT: Trying pragma unroll...
; CHECK-NEXT: Trying full unroll...
; CHECK-NEXT: Trying upper-bound unroll...
; CHECK-NEXT: Trying loop peeling...
; CHECK-NEXT: Trying partial unroll...
; CHECK-NEXT: Trying runtime unroll...
; CHECK-NEXT:  Not runtime unrolling: disabled by pragma.
; CHECK-NEXT: Not unrolling: no viable strategy found.

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
  br i1 %cmp, label %for.body, label %exit, !llvm.loop !11

exit:
  %result = phi i32 [ 0, %entry ], [ %add, %for.body ]
  ret i32 %result
}

; CHECK-LABEL:Loop Unroll: F[heuristic_peel] Loop %for.header (depth=1)
; CHECK-NEXT:Loop Size = 9
; CHECK-NEXT: Computing unroll count: TripCount=0, MaxTripCount=2147483647, TripMultiple=1
; CHECK-NEXT: Trying pragma unroll...
; CHECK-NEXT: Trying full unroll...
; CHECK-NEXT: Trying upper-bound unroll...
; CHECK-NEXT: Trying loop peeling...
; CHECK-NEXT:  Peeling with count: 1.
; CHECK-NEXT:PEELING loop %for.header with iteration count 1!
; CHECK-NEXT:remark: <unknown>:0:0: peeled loop by 1 iterations

declare void @foo()

define void @heuristic_peel(ptr %A, i32 %n) {
entry:
  %cmp.entry = icmp sgt i32 %n, 0
  br i1 %cmp.entry, label %for.header, label %exit

for.header:
  %i = phi i32 [ 0, %entry ], [ %inc, %for.latch ]
  %cmp.zero = icmp eq i32 %i, 0
  br i1 %cmp.zero, label %then, label %for.latch

then:
  call void @foo()
  br label %for.latch

for.latch:
  %arrayidx = getelementptr inbounds i32, ptr %A, i32 %i
  %load = load i32, ptr %arrayidx
  %inc = add i32 %i, 1
  %cmp = icmp slt i32 %inc, %n
  br i1 %cmp, label %for.header, label %exit

exit:
  ret void
}

; CHECK-LABEL:Loop Unroll: F[runtime_unroll_simple] Loop %for.body (depth=1)
; CHECK-NEXT:Loop Size = 6
; CHECK-NEXT: Computing unroll count: TripCount=0, MaxTripCount=2147483647, TripMultiple=1
; CHECK-NEXT: Explicit unroll requested: pragma-enable
; CHECK-NEXT: Trying pragma unroll...
; CHECK-NEXT: Trying full unroll...
; CHECK-NEXT: Trying upper-bound unroll...
; CHECK-NEXT: Trying loop peeling...
; CHECK-NEXT: Trying partial unroll...
; CHECK-NEXT: Trying runtime unroll...
; CHECK-NEXT:  Runtime unrolling with count: 8
; CHECK-NEXT:  Exiting block %for.body: TripCount=0, TripMultiple=1, BreakoutTrip=1
; CHECK:UNROLLING loop %for.body by 8 with run-time trip count!
; CHECK-NEXT:remark: <unknown>:0:0: unrolled loop by a factor of 8 with run-time trip count
;
; REMAINDER-LABEL:Loop Unroll: F[runtime_unroll_simple] Loop %for.body (depth=1)
; REMAINDER-NEXT:Loop Size = 6
; REMAINDER-NEXT: Computing unroll count: TripCount=0, MaxTripCount=2147483647, TripMultiple=1
; REMAINDER-NEXT: Explicit unroll requested: pragma-enable
; REMAINDER-NEXT: Trying pragma unroll...
; REMAINDER-NEXT: Trying full unroll...
; REMAINDER-NEXT: Trying upper-bound unroll...
; REMAINDER-NEXT: Trying loop peeling...
; REMAINDER-NEXT: Trying partial unroll...
; REMAINDER-NEXT: Trying runtime unroll...
; REMAINDER-NEXT:  Runtime unrolling with count: 8
; REMAINDER-NEXT:  Exiting block %for.body: TripCount=0, TripMultiple=1, BreakoutTrip=1
; REMAINDER:UNROLLING loop %for.body by 8 with run-time trip count (remainder unrolled)!
; REMAINDER-NEXT:remark: <unknown>:0:0: unrolled loop by a factor of 8 with run-time trip count (remainder unrolled)
;
; RUNTIME-NOPROFIT-LABEL:Loop Unroll: F[runtime_unroll_simple] Loop %for.body (depth=1)
; RUNTIME-NOPROFIT-NEXT:Loop Size = 6
; RUNTIME-NOPROFIT-NEXT: Computing unroll count: TripCount=0, MaxTripCount=2147483647, TripMultiple=1
; RUNTIME-NOPROFIT-NEXT: Explicit unroll requested: pragma-enable
; RUNTIME-NOPROFIT-NEXT: Trying pragma unroll...
; RUNTIME-NOPROFIT-NEXT: Trying full unroll...
; RUNTIME-NOPROFIT-NEXT: Trying upper-bound unroll...
; RUNTIME-NOPROFIT-NEXT: Trying loop peeling...
; RUNTIME-NOPROFIT-NEXT: Trying partial unroll...
; RUNTIME-NOPROFIT-NEXT: Trying runtime unroll...
; RUNTIME-NOPROFIT-NOT:  Runtime unrolling with count:
; RUNTIME-NOPROFIT: Not unrolling: no viable strategy found.

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
  br i1 %cmp, label %for.body, label %exit, !llvm.loop !13

exit:
  %result = phi i32 [ 0, %entry ], [ %add, %for.body ]
  ret i32 %result
}

; PARTIAL-ALLOW-LABEL:Loop Unroll: F[partial_unroll_cost_analysis] Loop %for.body (depth=1)
; PARTIAL-ALLOW-NEXT:Loop Size = 6
; PARTIAL-ALLOW-NEXT: Computing unroll count: TripCount=200, MaxTripCount=0, TripMultiple=200
; PARTIAL-ALLOW-NEXT: Trying pragma unroll...
; PARTIAL-ALLOW-NEXT: Trying full unroll...
; PARTIAL-ALLOW-NEXT:  Unrolled size {{[0-9]+}} exceeds threshold {{[0-9]+}}; checking for cost benefit.
; PARTIAL-ALLOW-NEXT:   Not analyzing loop cost: trip count too large.
; PARTIAL-ALLOW-NEXT: Trying upper-bound unroll...
; PARTIAL-ALLOW-NEXT: Trying loop peeling...
; PARTIAL-ALLOW-NEXT: Trying partial unroll...
; PARTIAL-ALLOW-NEXT:  Unrolled size exceeds threshold; reducing count from {{[0-9]+}} to {{[0-9]+}}.
; PARTIAL-ALLOW-NEXT:  Partially unrolling with count: {{[0-9]+}}
; PARTIAL-ALLOW-NEXT:  Exiting block %for.body: TripCount=200, TripMultiple=0, BreakoutTrip=0
; PARTIAL-ALLOW-NEXT:UNROLLING loop %for.body by {{[0-9]+}}!
; PARTIAL-ALLOW-NEXT:remark: <unknown>:0:0: unrolled loop by a factor of {{[0-9]+}}

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

; CHECK-LABEL:Loop Unroll: F[pragma_full_tc_too_large] Loop %for.body (depth=1)
; CHECK-NEXT:Loop Size = 6
; CHECK-NEXT: Computing unroll count: TripCount=1000001, MaxTripCount=0, TripMultiple=1000001
; CHECK-NEXT: Explicit unroll requested: pragma-full
; CHECK-NEXT: Trying pragma unroll...
; CHECK-NEXT:  Won't unroll; trip count is too large.
; CHECK-NEXT:remark: <unknown>:0:0: may be unable to fully unroll loop: trip count 1000001 exceeds limit 1000000
; CHECK-NEXT: Trying full unroll...
; CHECK-NEXT:  Unrolled size {{[0-9]+}} exceeds threshold {{[0-9]+}}; checking for cost benefit.
; CHECK-NEXT:   Not analyzing loop cost: trip count too large.
; CHECK-NEXT: Trying upper-bound unroll...
; CHECK-NEXT: Trying loop peeling...
; CHECK-NEXT: Trying partial unroll...
; CHECK-NEXT:  Unrolled size exceeds threshold; reducing count from {{[0-9]+}} to {{[0-9]+}}.
; CHECK-NEXT:  Partially unrolling with count: {{[0-9]+}}
; CHECK-NEXT:  Exiting block %for.body: TripCount=1000001, TripMultiple=0, BreakoutTrip=0
; CHECK-NEXT:UNROLLING loop %for.body by {{[0-9]+}}!
; CHECK-NEXT:remark: <unknown>:0:0: unrolled loop by a factor of {{[0-9]+}}
; CHECK-NEXT:remark: <unknown>:0:0: unable to fully unroll loop as directed; unrolled by factor {{[0-9]+}}

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
  %cmp = icmp ult i32 %inc, 1000001
  br i1 %cmp, label %for.body, label %exit, !llvm.loop !12

exit:
  ret i32 %add
}

; CHECK-LABEL:Loop Unroll: F[nested_loop_cost] Loop %inner (depth=2)
; CHECK-NEXT:Loop Size = 6
; CHECK-NEXT: Computing unroll count: TripCount=0, MaxTripCount=2147483647, TripMultiple=1
; CHECK-NEXT: Trying pragma unroll...
; CHECK-NEXT: Trying full unroll...
; CHECK-NEXT: Trying upper-bound unroll...
; CHECK-NEXT: Trying loop peeling...
; CHECK-NEXT: Trying partial unroll...
; CHECK-NEXT: Trying runtime unroll...
; CHECK-NEXT:  Will not try to unroll loop with runtime trip count because -unroll-runtime not given
; CHECK-NEXT: Not unrolling: no viable strategy found.
; CHECK-LABEL:Loop Unroll: F[nested_loop_cost] Loop %outer (depth=1)
; CHECK-NEXT:Loop Size = {{[0-9]+}}
; CHECK-NEXT: Computing unroll count: TripCount=10, MaxTripCount=0, TripMultiple=10
; CHECK-NEXT: Trying pragma unroll...
; CHECK-NEXT: Trying full unroll...
; CHECK-NEXT:  Unrolled size {{[0-9]+}} exceeds threshold {{[0-9]+}}; checking for cost benefit.
; CHECK-NEXT:   Not analyzing loop cost: not an innermost loop.
; CHECK-NEXT: Trying upper-bound unroll...
; CHECK-NEXT: Trying loop peeling...
; CHECK-NEXT: Trying partial unroll...
; CHECK-NEXT:  Will not try to unroll partially because -unroll-allow-partial not given
; CHECK-NEXT: Not unrolling: no viable strategy found.

define i32 @nested_loop_cost(ptr %A, i32 %n) {
entry:
  br label %outer

outer:
  %i = phi i32 [ 0, %entry ], [ %i.next, %inner.exit ]
  %sum.outer = phi i32 [ 0, %entry ], [ %sum.inner.lcssa, %inner.exit ]
  %idxA = getelementptr inbounds i32, ptr %A, i32 %i
  %loadA = load i32, ptr %idxA
  %mul1 = mul i32 %loadA, %sum.outer
  %add1 = add i32 %mul1, %loadA
  %mul2 = mul i32 %add1, %loadA
  %sub1 = sub i32 %mul2, %loadA
  %add2 = add i32 %sub1, %loadA
  %mul3 = mul i32 %add2, %loadA
  %sub2 = sub i32 %mul3, %loadA
  %xor1 = xor i32 %sub2, %loadA
  %or1 = or i32 %xor1, %loadA
  %outer.sum = add i32 %sum.outer, %or1
  br label %inner

inner:
  %j = phi i32 [ 0, %outer ], [ %j.next, %inner ]
  %sum.inner = phi i32 [ %outer.sum, %outer ], [ %inner.add, %inner ]
  %idxB = getelementptr inbounds i32, ptr %A, i32 %j
  %loadB = load i32, ptr %idxB
  %inner.add = add i32 %sum.inner, %loadB
  %j.next = add i32 %j, 1
  %inner.cmp = icmp slt i32 %j.next, %n
  br i1 %inner.cmp, label %inner, label %inner.exit

inner.exit:
  %sum.inner.lcssa = phi i32 [ %inner.add, %inner ]
  %i.next = add i32 %i, 1
  %outer.cmp = icmp ult i32 %i.next, 10
  br i1 %outer.cmp, label %outer, label %exit

exit:
  ret i32 %sum.inner.lcssa
}

; USER-COUNT-LABEL:Loop Unroll: F[user_count_unroll] Loop %for.body (depth=1)
; USER-COUNT-NEXT:Loop Size = 6
; USER-COUNT-NEXT: Computing unroll count: TripCount=12, MaxTripCount=0, TripMultiple=12
; USER-COUNT-NEXT: Explicit unroll requested: user-count
; USER-COUNT-NEXT: Trying pragma unroll...
; USER-COUNT-NEXT:  Unrolling with user-specified count: 4.
; USER-COUNT-NEXT:  Exiting block %for.body: TripCount=12, TripMultiple=0, BreakoutTrip=0
; USER-COUNT-NEXT:UNROLLING loop %for.body by 4!
; USER-COUNT-NEXT:remark: <unknown>:0:0: unrolled loop by a factor of 4
;
; USER-COUNT-EXCEED-LABEL:Loop Unroll: F[user_count_unroll] Loop %for.body (depth=1)
; USER-COUNT-EXCEED-NEXT:Loop Size = 6
; USER-COUNT-EXCEED-NEXT: Computing unroll count: TripCount=12, MaxTripCount=0, TripMultiple=12
; USER-COUNT-EXCEED-NEXT: Explicit unroll requested: user-count
; USER-COUNT-EXCEED-NEXT: Trying pragma unroll...
; USER-COUNT-EXCEED-NEXT:  Not unrolling with user count 9999: exceeds threshold.
; USER-COUNT-EXCEED-NEXT: Trying full unroll...
; USER-COUNT-EXCEED-NEXT:  Unrolling: size {{[0-9]+}} < threshold {{[0-9]+}}.
; USER-COUNT-EXCEED-NEXT:  Exiting block %for.body: TripCount=12, TripMultiple=0, BreakoutTrip=0
; USER-COUNT-EXCEED-NEXT:COMPLETELY UNROLLING loop %for.body with trip count 12!
; USER-COUNT-EXCEED-NEXT:remark: <unknown>:0:0: completely unrolled loop with 12 iterations

define i32 @user_count_unroll(ptr %A) {
entry:
  br label %for.body

for.body:
  %i = phi i32 [ 0, %entry ], [ %inc, %for.body ]
  %sum = phi i32 [ 0, %entry ], [ %add, %for.body ]
  %arrayidx = getelementptr inbounds i32, ptr %A, i32 %i
  %load = load i32, ptr %arrayidx
  %add = add i32 %sum, %load
  %inc = add i32 %i, 1
  %cmp = icmp ult i32 %inc, 12
  br i1 %cmp, label %for.body, label %exit

exit:
  ret i32 %add
}

; EXPLICIT-PEEL-LABEL:Loop Unroll: F[explicit_peel_count] Loop %for.body (depth=1)
; EXPLICIT-PEEL-NEXT:Loop Size = 6
; EXPLICIT-PEEL-NEXT: Computing unroll count: TripCount=0, MaxTripCount=2147483647, TripMultiple=1
; EXPLICIT-PEEL-NEXT:  Using explicit peel count: 2.
; EXPLICIT-PEEL-NEXT:PEELING loop %for.body with iteration count 2!
; EXPLICIT-PEEL-NEXT:remark: <unknown>:0:0: peeled loop by 2 iterations

define i32 @explicit_peel_count(ptr %A, i32 %n) {
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

; ZERO-THRESH-LABEL:Loop Unroll: F[zero_thresh_unroll] Loop %for.body (depth=1)
; ZERO-THRESH-NEXT: Not unrolling: all thresholds are zero.
; ZERO-THRESH-NEXT:remark: <unknown>:0:0: unable to unroll loop: unroll threshold is zero

define i32 @zero_thresh_unroll(ptr %A) {
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
  br i1 %cmp, label %for.body, label %exit, !llvm.loop !16

exit:
  ret i32 %add
}

; MAX-COUNT-LABEL:Loop Unroll: F[max_count_unroll] Loop %for.body (depth=1)
; MAX-COUNT-NEXT:Loop Size = 6
; MAX-COUNT-NEXT: Computing unroll count: TripCount=10, MaxTripCount=0, TripMultiple=10
; MAX-COUNT-NEXT: Trying pragma unroll...
; MAX-COUNT-NEXT: Trying full unroll...
; MAX-COUNT-NEXT:  Not unrolling: trip count 10 exceeds max count 2.
; MAX-COUNT-NEXT: Trying upper-bound unroll...
; MAX-COUNT-NEXT: Trying loop peeling...
; MAX-COUNT-NEXT: Trying partial unroll...
; MAX-COUNT-NEXT:  Will not try to unroll partially because -unroll-allow-partial not given
; MAX-COUNT-NEXT: Not unrolling: no viable strategy found.

define i32 @max_count_unroll(ptr %A) {
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

; PARTIAL-NOPROFIT-LABEL:Loop Unroll: F[partial_no_profit] Loop %for.body (depth=1)
; PARTIAL-NOPROFIT-NEXT:Loop Size = 6
; PARTIAL-NOPROFIT-NEXT: Computing unroll count: TripCount=200, MaxTripCount=0, TripMultiple=200
; PARTIAL-NOPROFIT-NEXT: Trying pragma unroll...
; PARTIAL-NOPROFIT-NEXT: Trying full unroll...
; PARTIAL-NOPROFIT-NEXT:  Unrolled size {{[0-9]+}} exceeds threshold {{[0-9]+}}; checking for cost benefit.
; PARTIAL-NOPROFIT-NEXT:   Not analyzing loop cost: trip count too large.
; PARTIAL-NOPROFIT-NEXT: Trying upper-bound unroll...
; PARTIAL-NOPROFIT-NEXT: Trying loop peeling...
; PARTIAL-NOPROFIT-NEXT: Trying partial unroll...
; PARTIAL-NOPROFIT-NEXT:  Unrolled size exceeds threshold; reducing count from {{[0-9]+}} to {{[0-9]+}}.
; PARTIAL-NOPROFIT-NEXT:  Will not partially unroll: no profitable count.
; PARTIAL-NOPROFIT-NEXT:  Partially unrolling with count: 0
; PARTIAL-NOPROFIT-NEXT: Not unrolling: no viable strategy found.

define i32 @partial_no_profit(ptr %A) {
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

; PRAGMA-NOREMAINDER-LABEL:Loop Unroll: F[pragma_count_no_remainder] Loop %for.body (depth=1)
; PRAGMA-NOREMAINDER-NEXT:Loop Size = 6
; PRAGMA-NOREMAINDER-NEXT: Computing unroll count: TripCount=10, MaxTripCount=0, TripMultiple=10
; PRAGMA-NOREMAINDER-NEXT: Explicit unroll requested: pragma-count(3)
; PRAGMA-NOREMAINDER-NEXT: Trying pragma unroll...
; PRAGMA-NOREMAINDER-NEXT:  Not unrolling with pragma count 3: remainder not allowed, count does not divide trip multiple 10.
; PRAGMA-NOREMAINDER-NEXT:remark: <unknown>:0:0: may be unable to unroll loop with count 3: remainder loop is not allowed and count does not divide trip multiple 10
; PRAGMA-NOREMAINDER-NEXT: Trying full unroll...
; PRAGMA-NOREMAINDER-NEXT:  Unrolling: size {{[0-9]+}} < threshold {{[0-9]+}}.
; PRAGMA-NOREMAINDER-NEXT:  Exiting block %for.body: TripCount=10, TripMultiple=0, BreakoutTrip=0
; PRAGMA-NOREMAINDER-NEXT:COMPLETELY UNROLLING loop %for.body with trip count 10!
; PRAGMA-NOREMAINDER-NEXT:remark: <unknown>:0:0: completely unrolled loop with 10 iterations
; PRAGMA-NOREMAINDER-NEXT:remark: <unknown>:0:0: unable to unroll loop with requested count 3; unrolled by factor 10

define i32 @pragma_count_no_remainder(ptr %A) {
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
  br i1 %cmp, label %for.body, label %exit, !llvm.loop !14

exit:
  ret i32 %add
}

; CHECK-LABEL:Loop Unroll: F[header_address_taken] Loop %for.body (depth=1)
; CHECK-NEXT:Loop Size = 6
; CHECK-NEXT: Computing unroll count: TripCount=4, MaxTripCount=0, TripMultiple=4
; CHECK-NEXT: Explicit unroll requested: pragma-enable
; CHECK-NEXT: Trying pragma unroll...
; CHECK-NEXT: Trying full unroll...
; CHECK-NEXT:  Unrolling: size {{[0-9]+}} < threshold {{[0-9]+}}.
; CHECK-NEXT:  Won't unroll loop: address of header block is taken.
; CHECK-NEXT: Failed to unroll loop as explicitly requested.
; CHECK-NEXT:remark: <unknown>:0:0: failed to unroll loop as explicitly requested

define i32 @header_address_taken(ptr %A) {
entry:
  store ptr blockaddress(@header_address_taken, %for.body), ptr %A
  br label %for.body

for.body:
  %i = phi i32 [ 0, %entry ], [ %inc, %for.body ]
  %sum = phi i32 [ 0, %entry ], [ %add, %for.body ]
  %arrayidx = getelementptr inbounds i32, ptr %A, i32 %i
  %load = load i32, ptr %arrayidx
  %add = add i32 %sum, %load
  %inc = add i32 %i, 1
  %cmp = icmp ult i32 %inc, 4
  br i1 %cmp, label %for.body, label %exit, !llvm.loop !15

exit:
  ret i32 %add
}

!0 = distinct !{!0, !3}
!1 = distinct !{!1, !4}
!2 = distinct !{!2, !4}
!3 = !{!"llvm.loop.unroll.full"}
!4 = !{!"llvm.loop.unroll.enable"}
!5 = distinct !{!5, !6}
!6 = !{!"llvm.loop.unroll.count", i32 3}
!7 = distinct !{!7, !8}
!8 = !{!"llvm.loop.unroll.disable"}
!9 = distinct !{!9, !4}
!10 = !{!"llvm.loop.unroll.runtime.disable"}
!11 = distinct !{!11, !10}
!12 = distinct !{!12, !3}
!13 = distinct !{!13, !4}
!14 = distinct !{!14, !6}
!15 = distinct !{!15, !4}
!16 = distinct !{!16, !4}
