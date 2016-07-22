; RUN: llc < %s -mtriple=aarch64-linux-gnu | FileCheck %s

; marked as external to prevent possible optimizations
@a = external global i32
@b = external global i32
@c = external global i32
@d = external global i32

; (a > 10 && b == c) || (a >= 10 && b == d)
define i32 @combine_gt_ge_10() #0 {
; CHECK-LABEL: combine_gt_ge_10
; CHECK: cmp
; CHECK: b.le
; CHECK: ret
; CHECK-NOT: cmp
; CHECK: b.lt
entry:
  %0 = load i32, i32* @a, align 4
  %cmp = icmp sgt i32 %0, 10
  br i1 %cmp, label %land.lhs.true, label %lor.lhs.false

land.lhs.true:                                    ; preds = %entry
  %1 = load i32, i32* @b, align 4
  %2 = load i32, i32* @c, align 4
  %cmp1 = icmp eq i32 %1, %2
  br i1 %cmp1, label %return, label %land.lhs.true3

lor.lhs.false:                                    ; preds = %entry
  %cmp2 = icmp sgt i32 %0, 9
  br i1 %cmp2, label %land.lhs.true3, label %if.end

land.lhs.true3:                                   ; preds = %lor.lhs.false, %land.lhs.true
  %3 = load i32, i32* @b, align 4
  %4 = load i32, i32* @d, align 4
  %cmp4 = icmp eq i32 %3, %4
  br i1 %cmp4, label %return, label %if.end

if.end:                                           ; preds = %land.lhs.true3, %lor.lhs.false
  br label %return

return:                                           ; preds = %if.end, %land.lhs.true3, %land.lhs.true
  %retval.0 = phi i32 [ 0, %if.end ], [ 1, %land.lhs.true3 ], [ 1, %land.lhs.true ]
  ret i32 %retval.0
}

; (a > 5 && b == c) || (a < 5 && b == d)
define i32 @combine_gt_lt_5() #0 {
; CHECK-LABEL: combine_gt_lt_5
; CHECK: cmp
; CHECK: b.le
; CHECK: ret
; CHECK-NOT: cmp
; CHECK: b.ge
entry:
  %0 = load i32, i32* @a, align 4
  %cmp = icmp sgt i32 %0, 5
  br i1 %cmp, label %land.lhs.true, label %lor.lhs.false

land.lhs.true:                                    ; preds = %entry
  %1 = load i32, i32* @b, align 4
  %2 = load i32, i32* @c, align 4
  %cmp1 = icmp eq i32 %1, %2
  br i1 %cmp1, label %return, label %if.end

lor.lhs.false:                                    ; preds = %entry
  %cmp2 = icmp slt i32 %0, 5
  br i1 %cmp2, label %land.lhs.true3, label %if.end

land.lhs.true3:                                   ; preds = %lor.lhs.false
  %3 = load i32, i32* @b, align 4
  %4 = load i32, i32* @d, align 4
  %cmp4 = icmp eq i32 %3, %4
  br i1 %cmp4, label %return, label %if.end

if.end:                                           ; preds = %land.lhs.true3, %lor.lhs.false, %land.lhs.true
  br label %return

return:                                           ; preds = %if.end, %land.lhs.true3, %land.lhs.true
  %retval.0 = phi i32 [ 0, %if.end ], [ 1, %land.lhs.true3 ], [ 1, %land.lhs.true ]
  ret i32 %retval.0
}

; (a < 5 && b == c) || (a <= 5 && b == d)
define i32 @combine_lt_ge_5() #0 {
; CHECK-LABEL: combine_lt_ge_5
; CHECK: cmp
; CHECK: b.ge
; CHECK: ret
; CHECK-NOT: cmp
; CHECK: b.gt
entry:
  %0 = load i32, i32* @a, align 4
  %cmp = icmp slt i32 %0, 5
  br i1 %cmp, label %land.lhs.true, label %lor.lhs.false

land.lhs.true:                                    ; preds = %entry
  %1 = load i32, i32* @b, align 4
  %2 = load i32, i32* @c, align 4
  %cmp1 = icmp eq i32 %1, %2
  br i1 %cmp1, label %return, label %land.lhs.true3

lor.lhs.false:                                    ; preds = %entry
  %cmp2 = icmp slt i32 %0, 6
  br i1 %cmp2, label %land.lhs.true3, label %if.end

land.lhs.true3:                                   ; preds = %lor.lhs.false, %land.lhs.true
  %3 = load i32, i32* @b, align 4
  %4 = load i32, i32* @d, align 4
  %cmp4 = icmp eq i32 %3, %4
  br i1 %cmp4, label %return, label %if.end

if.end:                                           ; preds = %land.lhs.true3, %lor.lhs.false
  br label %return

return:                                           ; preds = %if.end, %land.lhs.true3, %land.lhs.true
  %retval.0 = phi i32 [ 0, %if.end ], [ 1, %land.lhs.true3 ], [ 1, %land.lhs.true ]
  ret i32 %retval.0
}

; (a < 5 && b == c) || (a > 5 && b == d)
define i32 @combine_lt_gt_5() #0 {
; CHECK-LABEL: combine_lt_gt_5
; CHECK: cmp
; CHECK: b.ge
; CHECK: ret
; CHECK-NOT: cmp
; CHECK: b.le
entry:
  %0 = load i32, i32* @a, align 4
  %cmp = icmp slt i32 %0, 5
  br i1 %cmp, label %land.lhs.true, label %lor.lhs.false

land.lhs.true:                                    ; preds = %entry
  %1 = load i32, i32* @b, align 4
  %2 = load i32, i32* @c, align 4
  %cmp1 = icmp eq i32 %1, %2
  br i1 %cmp1, label %return, label %if.end

lor.lhs.false:                                    ; preds = %entry
  %cmp2 = icmp sgt i32 %0, 5
  br i1 %cmp2, label %land.lhs.true3, label %if.end

land.lhs.true3:                                   ; preds = %lor.lhs.false
  %3 = load i32, i32* @b, align 4
  %4 = load i32, i32* @d, align 4
  %cmp4 = icmp eq i32 %3, %4
  br i1 %cmp4, label %return, label %if.end

if.end:                                           ; preds = %land.lhs.true3, %lor.lhs.false, %land.lhs.true
  br label %return

return:                                           ; preds = %if.end, %land.lhs.true3, %land.lhs.true
  %retval.0 = phi i32 [ 0, %if.end ], [ 1, %land.lhs.true3 ], [ 1, %land.lhs.true ]
  ret i32 %retval.0
}

; (a > -5 && b == c) || (a < -5 && b == d)
define i32 @combine_gt_lt_n5() #0 {
; CHECK-LABEL: combine_gt_lt_n5
; CHECK: cmn
; CHECK: b.le
; CHECK: ret
; CHECK-NOT: cmn
; CHECK: b.ge
entry:
  %0 = load i32, i32* @a, align 4
  %cmp = icmp sgt i32 %0, -5
  br i1 %cmp, label %land.lhs.true, label %lor.lhs.false

land.lhs.true:                                    ; preds = %entry
  %1 = load i32, i32* @b, align 4
  %2 = load i32, i32* @c, align 4
  %cmp1 = icmp eq i32 %1, %2
  br i1 %cmp1, label %return, label %if.end

lor.lhs.false:                                    ; preds = %entry
  %cmp2 = icmp slt i32 %0, -5
  br i1 %cmp2, label %land.lhs.true3, label %if.end

land.lhs.true3:                                   ; preds = %lor.lhs.false
  %3 = load i32, i32* @b, align 4
  %4 = load i32, i32* @d, align 4
  %cmp4 = icmp eq i32 %3, %4
  br i1 %cmp4, label %return, label %if.end

if.end:                                           ; preds = %land.lhs.true3, %lor.lhs.false, %land.lhs.true
  br label %return

return:                                           ; preds = %if.end, %land.lhs.true3, %land.lhs.true
  %retval.0 = phi i32 [ 0, %if.end ], [ 1, %land.lhs.true3 ], [ 1, %land.lhs.true ]
  ret i32 %retval.0
}

; (a < -5 && b == c) || (a > -5 && b == d)
define i32 @combine_lt_gt_n5() #0 {
; CHECK-LABEL: combine_lt_gt_n5
; CHECK: cmn
; CHECK: b.ge
; CHECK: ret
; CHECK-NOT: cmn
; CHECK: b.le
entry:
  %0 = load i32, i32* @a, align 4
  %cmp = icmp slt i32 %0, -5
  br i1 %cmp, label %land.lhs.true, label %lor.lhs.false

land.lhs.true:                                    ; preds = %entry
  %1 = load i32, i32* @b, align 4
  %2 = load i32, i32* @c, align 4
  %cmp1 = icmp eq i32 %1, %2
  br i1 %cmp1, label %return, label %if.end

lor.lhs.false:                                    ; preds = %entry
  %cmp2 = icmp sgt i32 %0, -5
  br i1 %cmp2, label %land.lhs.true3, label %if.end

land.lhs.true3:                                   ; preds = %lor.lhs.false
  %3 = load i32, i32* @b, align 4
  %4 = load i32, i32* @d, align 4
  %cmp4 = icmp eq i32 %3, %4
  br i1 %cmp4, label %return, label %if.end

if.end:                                           ; preds = %land.lhs.true3, %lor.lhs.false, %land.lhs.true
  br label %return

return:                                           ; preds = %if.end, %land.lhs.true3, %land.lhs.true
  %retval.0 = phi i32 [ 0, %if.end ], [ 1, %land.lhs.true3 ], [ 1, %land.lhs.true ]
  ret i32 %retval.0
}

%struct.Struct = type { i64, i64 }

@glob = internal unnamed_addr global %struct.Struct* null, align 8

declare %struct.Struct* @Update(%struct.Struct*) #1

; no checks for this case, it just should be processed without errors
define void @combine_non_adjacent_cmp_br(%struct.Struct* nocapture readonly %hdCall) #0 {
entry:
  %size = getelementptr inbounds %struct.Struct, %struct.Struct* %hdCall, i64 0, i32 0
  %0 = load i64, i64* %size, align 8
  br label %land.rhs

land.rhs:
  %rp.06 = phi i64 [ %0, %entry ], [ %sub, %while.body ]
  %1 = load i64, i64* inttoptr (i64 24 to i64*), align 8
  %cmp2 = icmp sgt i64 %1, 0
  br i1 %cmp2, label %while.body, label %while.end

while.body:
  %2 = load %struct.Struct*, %struct.Struct** @glob, align 8
  %call = tail call %struct.Struct* @Update(%struct.Struct* %2) #2
  %sub = add nsw i64 %rp.06, -2
  %cmp = icmp slt i64 %0, %rp.06
  br i1 %cmp, label %land.rhs, label %while.end

while.end:
  ret void
}

; undefined external to prevent possible optimizations
declare void @do_something() #1

define i32 @do_nothing_if_resultant_opcodes_would_differ() #0 {
; CHECK-LABEL: do_nothing_if_resultant_opcodes_would_differ
; CHECK: cmn
; CHECK: b.gt
; CHECK: cmp
; CHECK: b.gt
entry:
  %0 = load i32, i32* @a, align 4
  %cmp4 = icmp slt i32 %0, -1
  br i1 %cmp4, label %while.body.preheader, label %while.end

while.body.preheader:                             ; preds = %entry
  br label %while.body

while.body:                                       ; preds = %while.body, %while.body.preheader
  %i.05 = phi i32 [ %inc, %while.body ], [ %0, %while.body.preheader ]
  tail call void @do_something() #2
  %inc = add nsw i32 %i.05, 1
  %cmp = icmp slt i32 %i.05, 0
  br i1 %cmp, label %while.body, label %while.cond.while.end_crit_edge

while.cond.while.end_crit_edge:                   ; preds = %while.body
  %.pre = load i32, i32* @a, align 4
  br label %while.end

while.end:                                        ; preds = %while.cond.while.end_crit_edge, %entry
  %1 = phi i32 [ %.pre, %while.cond.while.end_crit_edge ], [ %0, %entry ]
  %cmp1 = icmp slt i32 %1, 2
  br i1 %cmp1, label %land.lhs.true, label %if.end

land.lhs.true:                                    ; preds = %while.end
  %2 = load i32, i32* @b, align 4
  %3 = load i32, i32* @d, align 4
  %cmp2 = icmp eq i32 %2, %3
  br i1 %cmp2, label %return, label %if.end

if.end:                                           ; preds = %land.lhs.true, %while.end
  br label %return

return:                                           ; preds = %if.end, %land.lhs.true
  %retval.0 = phi i32 [ 0, %if.end ], [ 123, %land.lhs.true ]
  ret i32 %retval.0
}

define i32 @do_nothing_if_compares_can_not_be_adjusted_to_each_other() #0 {
; CHECK-LABEL: do_nothing_if_compares_can_not_be_adjusted_to_each_other
; CHECK: cmp
; CHECK: b.gt
; CHECK: cmn
; CHECK: b.lt
entry:
  %0 = load i32, i32* @a, align 4
  %cmp4 = icmp slt i32 %0, 1
  br i1 %cmp4, label %while.body.preheader, label %while.end

while.body.preheader:                             ; preds = %entry
  br label %while.body

while.body:                                       ; preds = %while.body, %while.body.preheader
  %i.05 = phi i32 [ %inc, %while.body ], [ %0, %while.body.preheader ]
  tail call void @do_something() #2
  %inc = add nsw i32 %i.05, 1
  %cmp = icmp slt i32 %i.05, 0
  br i1 %cmp, label %while.body, label %while.end.loopexit

while.end.loopexit:                               ; preds = %while.body
  br label %while.end

while.end:                                        ; preds = %while.end.loopexit, %entry
  %1 = load i32, i32* @c, align 4
  %cmp1 = icmp sgt i32 %1, -3
  br i1 %cmp1, label %land.lhs.true, label %if.end

land.lhs.true:                                    ; preds = %while.end
  %2 = load i32, i32* @b, align 4
  %3 = load i32, i32* @d, align 4
  %cmp2 = icmp eq i32 %2, %3
  br i1 %cmp2, label %return, label %if.end

if.end:                                           ; preds = %land.lhs.true, %while.end
  br label %return

return:                                           ; preds = %if.end, %land.lhs.true
  %retval.0 = phi i32 [ 0, %if.end ], [ 123, %land.lhs.true ]
  ret i32 %retval.0
}

; Test in the following case, we don't hit 'cmp' and trigger a false positive
; cmp  w19, #0
; cinc w0, w19, gt
; ...
; fcmp d8, #0.0
; b.gt .LBB0_5

define i32 @fcmpri(i32 %argc, i8** nocapture readonly %argv) {

; CHECK-LABEL: fcmpri:
; CHECK: cmp w0, #2
; CHECK: b.lt .LBB9_3
; CHECK-NOT: cmp w0, #1
; CHECK-NOT: b.le .LBB9_3

; CHECK-LABEL-DAG: .LBB9_3
; CHECK: cmp w19, #0
; CHECK: fcmp d8, #0.0
; CHECK-NOT: cmp w19, #1
; CHECK-NOT: b.ge .LBB9_5

entry:
  %cmp = icmp sgt i32 %argc, 1
  br i1 %cmp, label %land.lhs.true, label %if.end

land.lhs.true:                                    ; preds = %entry
  %arrayidx = getelementptr inbounds i8*, i8** %argv, i64 1
  %0 = load i8*, i8** %arrayidx, align 8
  %cmp1 = icmp eq i8* %0, null
  br i1 %cmp1, label %if.end, label %return

if.end:                                           ; preds = %land.lhs.true, %entry
  %call = call i32 @zoo(i32 1)
  %call2 = call double @yoo(i32 -1)
  %cmp4 = icmp sgt i32 %call, 0
  %add = zext i1 %cmp4 to i32
  %cond = add nsw i32 %add, %call
  %call7 = call i32 @xoo(i32 %cond, i32 2)
  %cmp9 = fcmp ogt double %call2, 0.000000e+00
  br i1 %cmp9, label %cond.end14, label %cond.false12

cond.false12:                                     ; preds = %if.end
  %sub = fadd fast double %call2, -1.000000e+00
  br label %cond.end14

cond.end14:                                       ; preds = %if.end, %cond.false12
  %cond15 = phi double [ %sub, %cond.false12 ], [ %call2, %if.end ]
  %call16 = call i32 @woo(double %cond15, double -2.000000e+00)
  br label %return

return:                                           ; preds = %land.lhs.true, %cond.end14
  %retval.0 = phi i32 [ 4, %cond.end14 ], [ 3, %land.lhs.true ]
  ret i32 %retval.0
}

define void @cmp_shifted(i32 %in, i32 %lhs, i32 %rhs) {
; CHECK-LABEL: cmp_shifted:
; CHECK: cmp w0, #2, lsl #12
; [...]
; CHECK: cmp w0, #1

  %tst_low = icmp sgt i32 %in, 8191
  br i1 %tst_low, label %true, label %false

true:
  call i32 @zoo(i32 128)
  ret void

false:
  %tst = icmp sgt i32 %in, 0
  br i1 %tst, label %truer, label %falser

truer:
  call i32 @zoo(i32 42)
  ret void

falser:
  call i32 @zoo(i32 1)
  ret void
}

define i32 @combine_gt_ge_sel(i64 %v, i64* %p) #0 {
; CHECK-LABEL: combine_gt_ge_sel
; CHECK: ldr [[reg1:w[0-9]*]],
; CHECK: cmp [[reg1]], #0
; CHECK: csel {{.*}}, gt
entry:
  %0 = load i32, i32* @a, align 4
  %cmp = icmp sgt i32 %0, 0
  %m = select i1 %cmp, i64 %v, i64 0
  store i64 %m, i64* %p
  br i1 %cmp, label %lor.lhs.false, label %land.lhs.true

land.lhs.true:                                    ; preds = %entry
  %1 = load i32, i32* @b, align 4
  %2 = load i32, i32* @c, align 4
  %cmp1 = icmp eq i32 %1, %2
  br i1 %cmp1, label %return, label %land.lhs.true3

lor.lhs.false:                                    ; preds = %entry
  %cmp2 = icmp sgt i32 %0, 1
  br i1 %cmp2, label %land.lhs.true3, label %if.end

land.lhs.true3:                                   ; preds = %lor.lhs.false, %land.lhs.true
  %3 = load i32, i32* @b, align 4
  %4 = load i32, i32* @d, align 4
  %cmp4 = icmp eq i32 %3, %4
  br i1 %cmp4, label %return, label %if.end

if.end:                                           ; preds = %land.lhs.true3, %lor.lhs.false
  br label %return

return:                                           ; preds = %if.end, %land.lhs.true3, %land.lhs.true
  %retval.0 = phi i32 [ 0, %if.end ], [ 1, %land.lhs.true3 ], [ 1, %land.lhs.true ]
  ret i32 %retval.0
}

declare i32 @zoo(i32)

declare double @yoo(i32)

declare i32 @xoo(i32, i32)

declare i32 @woo(double, double)
