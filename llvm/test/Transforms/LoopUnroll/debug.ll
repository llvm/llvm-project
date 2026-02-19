; RUN: opt -disable-output -passes=loop-unroll -debug-only=loop-unroll < %s 2>&1 | FileCheck %s --match-full-lines --strict-whitespace
; RUN: opt -disable-output -passes=loop-unroll -debug-only=loop-unroll -pass-remarks-missed=loop-unroll -unroll-allow-partial -pragma-unroll-threshold=100 -pragma-unroll-full-max-iterations=50 < %s 2>&1 | FileCheck %s --check-prefix=ALLOW-PARTIAL --match-full-lines --strict-whitespace
; RUN: opt -disable-output -passes=loop-unroll -debug-only=loop-unroll -pass-remarks-missed=loop-unroll -unroll-partial-threshold=6 -pragma-unroll-threshold=6 -pragma-unroll-full-max-iterations=50 < %s 2>&1 | FileCheck %s --check-prefix=PATH2 --match-full-lines --strict-whitespace

; REQUIRES: asserts

; CHECK-LABEL:Loop Unroll: F[pragma_full_unroll_unknown_tc] Loop %for.body
; CHECK-NEXT:Loop Size = 6
; CHECK-NEXT:  will not try to unroll loop with runtime trip count -unroll-runtime not given

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

; CHECK-LABEL:Loop Unroll: F[full_unroll_cost_exceeds] Loop %for.body
; CHECK-NEXT:Loop Size = 6
; CHECK-NEXT:   will not try to unroll partially because -unroll-allow-partial not given

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

; CHECK-LABEL:Loop Unroll: F[extended_convergence] Loop %for.body
; CHECK-NEXT: Convergence prevents unrolling.
; CHECK-NEXT: Loop not considered unrollable.

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

; CHECK-LABEL:Loop Unroll: F[noduplicate_prevents_unroll] Loop %for.body
; CHECK-NEXT: Non-duplicatable blocks prevent unrolling.
; CHECK-NEXT: Loop not considered unrollable.

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

; CHECK-LABEL:Loop Unroll: F[indirectbr_loop] Loop %for.body
; CHECK-NEXT: Not unrolling loop which is not in loop-simplify form.

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

; CHECK-LABEL:Loop Unroll: F[inline_prevents_unroll] Loop %for.body
; CHECK-NEXT:Loop Size = 8
; CHECK-NEXT: Not unrolling loop with inlinable calls.

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

; CHECK-LABEL:Loop Unroll: F[full_unroll_profitability_analysis] Loop %for.body
; CHECK-NEXT:Loop Size = {{[0-9]+}}
; CHECK-NEXT:    Starting LoopUnroll profitability analysis...
; CHECK-NEXT:    Analyzing iteration 0
; CHECK-NEXT:    Analyzing iteration 1
; CHECK-NEXT:    Analyzing iteration 2
; CHECK-NEXT:    Analyzing iteration 3
; CHECK-NEXT:    Analyzing iteration 4
; CHECK-NEXT:    Analyzing iteration 5
; CHECK-NEXT:    Analyzing iteration 6
; CHECK-NEXT:    Analyzing iteration 7
; CHECK-NEXT:    Analyzing iteration 8
; CHECK-NEXT:    Analyzing iteration 9
; CHECK:    Analysis finished:
; CHECK-NEXT:    UnrolledCost: {{[0-9]+}}, RolledDynamicCost: {{[0-9]+}}
; CHECK-NEXT:  Exiting block %for.body: TripCount=10, TripMultiple=0, BreakoutTrip=0
; CHECK-NEXT:COMPLETELY UNROLLING loop %for.body with trip count 10!

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

; ALLOW-PARTIAL-LABEL:Loop Unroll: F[pragma_full_partial] Loop %for.body
; ALLOW-PARTIAL-NEXT:Loop Size = 6
; ALLOW-PARTIAL:   Won't unroll; trip count is too large
; ALLOW-PARTIAL:   partially unrolling with count: 25
; ALLOW-PARTIAL-NEXT:remark: <unknown>:0:0: Unable to fully unroll loop as directed by unroll pragma(full) because unrolled size is too large.

; PATH2-LABEL:Loop Unroll: F[pragma_full_partial] Loop %for.body
; PATH2-NEXT:Loop Size = 6
; PATH2:   Won't unroll; trip count is too large
; PATH2:   partially unrolling with count: 0
; PATH2-NEXT:remark: <unknown>:0:0: Unable to unroll loop as directed by unroll pragma(enable) because unrolled size is too large.

define i32 @pragma_full_partial(ptr %A) {
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
  br i1 %cmp, label %for.body, label %exit, !llvm.loop !5

exit:
  ret i32 %add
}

; ALLOW-PARTIAL-LABEL:Loop Unroll: F[pragma_enable_partial] Loop %for.body
; ALLOW-PARTIAL-NEXT:Loop Size = 6
; ALLOW-PARTIAL:   partially unrolling with count: 25
; ALLOW-PARTIAL-NOT:remark: {{.*}}Unable to
; ALLOW-PARTIAL:UNROLLING loop %for.body by 25!

; PATH2-LABEL:Loop Unroll: F[pragma_enable_partial] Loop %for.body
; PATH2-NEXT:Loop Size = 6
; PATH2:   partially unrolling with count: 0
; PATH2-NEXT:remark: <unknown>:0:0: Unable to unroll loop as directed by unroll pragma(enable) because unrolled size is too large.

define i32 @pragma_enable_partial(ptr %A) {
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
  br i1 %cmp, label %for.body, label %exit, !llvm.loop !6

exit:
  ret i32 %add
}

!0 = distinct !{!0, !3}
!1 = distinct !{!1, !4}
!2 = distinct !{!2, !4}
!3 = !{!"llvm.loop.unroll.full"}
!4 = !{!"llvm.loop.unroll.enable"}
!5 = distinct !{!5, !3}
!6 = distinct !{!6, !4}
