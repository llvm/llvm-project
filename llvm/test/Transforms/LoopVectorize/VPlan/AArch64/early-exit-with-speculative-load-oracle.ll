; RUN: opt -p loop-vectorize -force-vector-width=4 -disable-output \
; RUN:     -vplan-print-after=printOptimizedVPlan %s 2>&1 | FileCheck %s --allow-empty
; CHECK-NOT: VPlan for loop in

target triple = "arm64-apple-macosx"

@G = external global [1024 x i8]

define i64 @find_first_eq_const(ptr %A, i64 %n) {
;
entry:
  br label %loop.header

loop.header:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %loop.latch ]
  %gep.A = getelementptr inbounds i8, ptr %A, i64 %iv
  %ld.A = load i8, ptr %gep.A, align 1
  %cmp = icmp eq i8 %ld.A, 42
  br i1 %cmp, label %early.exit, label %loop.latch

loop.latch:
  %iv.next = add nuw nsw i64 %iv, 1
  %ec = icmp ne i64 %iv.next, %n
  br i1 %ec, label %loop.header, label %exit

early.exit:
  ret i64 %iv

exit:
  ret i64 -1
}

define i64 @find_first_eq_live_in(ptr %A, i64 %n, i8 %val) {
;
entry:
  br label %loop.header

loop.header:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %loop.latch ]
  %gep.A = getelementptr inbounds i8, ptr %A, i64 %iv
  %ld.A = load i8, ptr %gep.A, align 1
  %cmp = icmp eq i8 %ld.A, %val
  br i1 %cmp, label %early.exit, label %loop.latch

loop.latch:
  %iv.next = add nuw nsw i64 %iv, 1
  %ec = icmp ne i64 %iv.next, %n
  br i1 %ec, label %loop.header, label %exit

early.exit:
  ret i64 %iv

exit:
  ret i64 -1
}

define i32 @i32_induction(ptr %A, ptr %B, i32 %n) {
;
entry:
  br label %loop.header

loop.header:
  %iv = phi i32 [ 0, %entry ], [ %iv.next, %loop.latch ]
  %gep.A = getelementptr inbounds i8, ptr %A, i32 %iv
  %ld.A = load i8, ptr %gep.A, align 1
  %gep.B = getelementptr inbounds i8, ptr %B, i32 %iv
  %ld.B = load i8, ptr %gep.B, align 1
  %cmp = icmp ne i8 %ld.A, %ld.B
  br i1 %cmp, label %early.exit, label %loop.latch

loop.latch:
  %iv.next = add nuw nsw i32 %iv, 1
  %ec = icmp ne i32 %iv.next, %n
  br i1 %ec, label %loop.header, label %exit

early.exit:
  ret i32 %iv

exit:
  ret i32 -1
}

; Covers replaying scalar binops, div/rem with a constant divisor and freeze.
define i64 @binop_chain(ptr %A, ptr %B, ptr %C, i64 %n) {
;
entry:
  br label %loop.header

loop.header:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %loop.latch ]
  %gep.A = getelementptr inbounds i8, ptr %A, i64 %iv
  %ld.A = load i8, ptr %gep.A, align 1
  %gep.B = getelementptr inbounds i8, ptr %B, i64 %iv
  %ld.B = load i8, ptr %gep.B, align 1
  %gep.C = getelementptr inbounds i8, ptr %C, i64 %iv
  %ld.C = load i8, ptr %gep.C, align 1
  %add = add i8 %ld.A, %ld.B
  %sub = sub i8 %add, %ld.C
  %mul = mul i8 %sub, 3
  %xor = xor i8 %mul, 7
  %div = udiv i8 %xor, 3
  %fr = freeze i8 %div
  %cmp = icmp eq i8 %fr, 0
  br i1 %cmp, label %early.exit, label %loop.latch

loop.latch:
  %iv.next = add nuw nsw i64 %iv, 1
  %ec = icmp ne i64 %iv.next, %n
  br i1 %ec, label %loop.header, label %exit

early.exit:
  ret i64 %iv

exit:
  ret i64 -1
}

define i64 @predicated_exit_condition(ptr %mask, ptr %A, ptr %B, i64 %n) {
entry:
  br label %loop.header

loop.header:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %loop.latch ]
  %gep.mask = getelementptr inbounds i8, ptr %mask, i64 %iv
  %ld.mask = load i8, ptr %gep.mask, align 1
  %mask.cmp = icmp ne i8 %ld.mask, 0
  br i1 %mask.cmp, label %check, label %loop.latch

check:
  %gep.A = getelementptr inbounds i8, ptr %A, i64 %iv
  %ld.A = load i8, ptr %gep.A, align 1
  %gep.B = getelementptr inbounds i8, ptr %B, i64 %iv
  %ld.B = load i8, ptr %gep.B, align 1
  %cmp = icmp ne i8 %ld.A, %ld.B
  br i1 %cmp, label %early.exit, label %loop.latch

loop.latch:
  %iv.next = add nuw nsw i64 %iv, 1
  %ec = icmp ne i64 %iv.next, %n
  br i1 %ec, label %loop.header, label %exit

early.exit:
  ret i64 %iv

exit:
  ret i64 -1
}

; As @predicated_exit_condition, but the loads are on the arms of a diamond.
define i64 @phi_merge_before_exit_condition(ptr %mask, ptr %A, ptr %B, i64 %n) {
entry:
  br label %loop.header

loop.header:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %loop.latch ]
  %gep.mask = getelementptr inbounds i8, ptr %mask, i64 %iv
  %ld.mask = load i8, ptr %gep.mask, align 1
  %mask.cmp = icmp ne i8 %ld.mask, 0
  br i1 %mask.cmp, label %if.true, label %if.false

if.true:
  %gep.A = getelementptr inbounds i8, ptr %A, i64 %iv
  %ld.A = load i8, ptr %gep.A, align 1
  br label %merge

if.false:
  %gep.B = getelementptr inbounds i8, ptr %B, i64 %iv
  %ld.B = load i8, ptr %gep.B, align 1
  br label %merge

merge:
  %val = phi i8 [ %ld.B, %if.false ], [ %ld.A, %if.true ]
  %cmp = icmp eq i8 %val, 0
  br i1 %cmp, label %early.exit, label %loop.latch

loop.latch:
  %iv.next = add nuw nsw i64 %iv, 1
  %ec = icmp ne i64 %iv.next, %n
  br i1 %ec, label %loop.header, label %exit

early.exit:
  ret i64 %iv

exit:
  ret i64 -1
}

define i64 @derived_induction_start(ptr %A, ptr %B, i64 %n) {
;
entry:
  br label %loop.header

loop.header:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %loop.latch ]
  %iv.off = phi i64 [ 10, %entry ], [ %iv.off.next, %loop.latch ]
  %gep.A = getelementptr inbounds i8, ptr %A, i64 %iv.off
  %ld.A = load i8, ptr %gep.A, align 1
  %gep.B = getelementptr inbounds i8, ptr %B, i64 %iv.off
  %ld.B = load i8, ptr %gep.B, align 1
  %cmp = icmp ne i8 %ld.A, %ld.B
  br i1 %cmp, label %early.exit, label %loop.latch

loop.latch:
  %iv.next = add nuw nsw i64 %iv, 1
  %iv.off.next = add nuw nsw i64 %iv.off, 1
  %ec = icmp ne i64 %iv.next, %n
  br i1 %ec, label %loop.header, label %exit

early.exit:
  ret i64 %iv

exit:
  ret i64 -1
}

; Test with induction with step 2.
define i64 @derived_induction_step(ptr %A, i64 %n) {
;
entry:
  br label %loop.header

loop.header:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %loop.latch ]
  %iv.2 = phi i64 [ 0, %entry ], [ %iv.2.next, %loop.latch ]
  %gep.A = getelementptr inbounds i8, ptr %A, i64 %iv
  %ld.A = load i8, ptr %gep.A, align 1
  %trunc = trunc i64 %iv.2 to i8
  %cmp = icmp ne i8 %ld.A, %trunc
  br i1 %cmp, label %early.exit, label %loop.latch

loop.latch:
  %iv.next = add nuw nsw i64 %iv, 1
  %iv.2.next = add nuw nsw i64 %iv.2, 2
  %ec = icmp ne i64 %iv.next, %n
  br i1 %ec, label %loop.header, label %exit

early.exit:
  ret i64 %iv

exit:
  ret i64 -1
}

; Dead recipes are dropped before the oracle plan is built.
define i64 @dead_recipes_in_loop(ptr %A, i64 %n) {
;
entry:
  br label %loop.header

loop.header:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %loop.latch ]
  %dead.call = call i8 @llvm.abs.i8(i8 0, i1 false)
  %dead.gep = getelementptr i8, ptr null, i64 8
  %gep.A = getelementptr inbounds i8, ptr %A, i64 %iv
  %ld.A = load i8, ptr %gep.A, align 1
  %cmp = icmp eq i8 %ld.A, 42
  br i1 %cmp, label %early.exit, label %loop.latch

loop.latch:
  %iv.next = add nuw nsw i64 %iv, 1
  %ec = icmp ne i64 %iv.next, %n
  br i1 %ec, label %loop.header, label %exit

early.exit:
  ret i64 %iv

exit:
  ret i64 -1
}

; The oracle only reads memory through its arguments, so @G is passed in.
define i64 @global_base(i64 %n) {
;
entry:
  br label %loop.header

loop.header:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %loop.latch ]
  %gep = getelementptr inbounds i8, ptr @G, i64 %iv
  %l = load i8, ptr %gep, align 1
  %c = icmp eq i8 %l, 42
  br i1 %c, label %early.exit, label %loop.latch

loop.latch:
  %iv.next = add nuw nsw i64 %iv, 1
  %ec = icmp ne i64 %iv.next, %n
  br i1 %ec, label %loop.header, label %exit

early.exit:
  ret i64 %iv

exit:
  ret i64 -1
}

; TODO: support calls in the oracle.
define i64 @call_in_exit_condition(ptr %A, i64 %n) {
entry:
  br label %loop.header

loop.header:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %loop.latch ]
  %gep.A = getelementptr inbounds i8, ptr %A, i64 %iv
  %ld.A = load i8, ptr %gep.A, align 1
  %abs = call i8 @llvm.abs.i8(i8 %ld.A, i1 false)
  %cmp = icmp sgt i8 %abs, 42
  br i1 %cmp, label %early.exit, label %loop.latch

loop.latch:
  %iv.next = add nuw nsw i64 %iv, 1
  %ec = icmp ne i64 %iv.next, %n
  br i1 %ec, label %loop.header, label %exit

early.exit:
  ret i64 %iv

exit:
  ret i64 -1
}

; TODO: support fcmp in the oracle.
define i64 @fcmp_in_exit_condition(ptr %A, i64 %n) {
entry:
  br label %loop.header

loop.header:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %loop.latch ]
  %gep.A = getelementptr inbounds float, ptr %A, i64 %iv
  %ld.A = load float, ptr %gep.A, align 4
  %cmp = fcmp uge float %ld.A, 0.000000e+00
  br i1 %cmp, label %early.exit, label %loop.latch

loop.latch:
  %iv.next = add nuw nsw i64 %iv, 1
  %ec = icmp ne i64 %iv.next, %n
  br i1 %ec, label %loop.header, label %exit

early.exit:
  ret i64 %iv

exit:
  ret i64 -1
}

; TODO: support unary operations in the oracle.
define i64 @fneg_in_exit_condition(ptr %A, i64 %n) {
entry:
  br label %loop.header

loop.header:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %loop.latch ]
  %gep.A = getelementptr inbounds float, ptr %A, i64 %iv
  %ld.A = load float, ptr %gep.A, align 4
  %fneg = fneg float %ld.A
  %cmp = fcmp ugt float %fneg, 0.000000e+00
  br i1 %cmp, label %early.exit, label %loop.latch

loop.latch:
  %iv.next = add nuw nsw i64 %iv, 1
  %ec = icmp ne i64 %iv.next, %n
  br i1 %ec, label %loop.header, label %exit

early.exit:
  ret i64 %iv

exit:
  ret i64 -1
}

; TODO: support pointer inductions in the oracle.
define i64 @pointer_induction(ptr %begin, ptr %end) {
entry:
  %is.empty = icmp eq ptr %begin, %end
  br i1 %is.empty, label %exit, label %loop.header

loop.header:
  %ptr = phi ptr [ %begin, %entry ], [ %ptr.next, %loop.latch ]
  %ld = load i8, ptr %ptr, align 1
  %cmp = icmp eq i8 %ld, 42
  br i1 %cmp, label %early.exit, label %loop.latch

loop.latch:
  %ptr.next = getelementptr inbounds i8, ptr %ptr, i64 1
  %ec = icmp eq ptr %ptr.next, %end
  br i1 %ec, label %exit, label %loop.header

early.exit:
  %iv = ptrtoint ptr %ptr to i64
  ret i64 %iv

exit:
  ret i64 -1
}

define i64 @early_exit_runtime_induction_step(ptr %A, i64 %n, i64 %s) {
entry:
  %step = mul i64 %s, 4
  br label %loop.header

loop.header:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %loop.latch ]
  %j = phi i64 [ 0, %entry ], [ %j.next, %loop.latch ]
  %gep = getelementptr inbounds i32, ptr %A, i64 %iv
  %l = load i32, ptr %gep, align 4
  %c = icmp eq i32 %l, 42
  br i1 %c, label %early.exit, label %loop.latch

loop.latch:
  %iv.next = add nuw nsw i64 %iv, 1
  %j.next = add nuw nsw i64 %j, %step
  %ec = icmp eq i64 %iv.next, %n
  br i1 %ec, label %exit, label %loop.header

early.exit:
  ret i64 %j

exit:
  ret i64 -1
}
