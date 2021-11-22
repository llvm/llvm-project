; RUN: opt -analyze -scalar-evolution %s -enable-new-pm=0 | FileCheck %s
; RUN: opt -passes='print<scalar-evolution>' -disable-output %s 2>&1 | FileCheck %s

; Test cases that require rewriting zext SCEV expression with infomration from
; the loop guards.

define void @rewrite_zext(i32 %n) {
; CHECK-LABEL: Determining loop execution counts for: @rewrite_zext
; CHECK-NEXT:  Loop %loop: backedge-taken count is ((-8 + (8 * ((zext i32 %n to i64) /u 8))<nuw><nsw>)<nsw> /u 8)
; CHECK-NEXT:  Loop %loop: max backedge-taken count is 2
; CHECK-NEXT:  Loop %loop: Predicated backedge-taken count is ((-8 + (8 * ((zext i32 %n to i64) /u 8))<nuw><nsw>)<nsw> /u 8)
; CHECK-NEXT:  Predicates:
; CHECK:        Loop %loop: Trip multiple is 1
;
entry:
  %ext = zext i32 %n to i64
  %cmp5 = icmp ule i64 %ext, 24
  br i1 %cmp5, label %check, label %exit

check:                                 ; preds = %entry
  %min.iters.check = icmp ult i64 %ext, 8
  %n.vec = and i64 %ext, -8
  br i1 %min.iters.check, label %exit, label %loop

loop:
  %index = phi i64 [ 0, %check ], [ %index.next, %loop ]
  %index.next = add nuw nsw i64 %index, 8
  %ec = icmp eq i64 %index.next, %n.vec
  br i1 %ec, label %exit, label %loop

exit:
  ret void
}

; Test case from PR40961.
define i32 @rewrite_zext_min_max(i32 %N, i32* %arr) {
; CHECK-LABEL:  Determining loop execution counts for: @rewrite_zext_min_max
; CHECK-NEXT:   Loop %loop: backedge-taken count is ((-4 + (4 * ((zext i32 (16 umin %N) to i64) /u 4))<nuw><nsw>)<nsw> /u 4)
; CHECK-NEXT:   Loop %loop: max backedge-taken count is 3
; CHECK-NEXT:   Loop %loop: Predicated backedge-taken count is ((-4 + (4 * ((zext i32 (16 umin %N) to i64) /u 4))<nuw><nsw>)<nsw> /u 4)
; CHECK-NEXT:   Predicates:
; CHECK:         Loop %loop: Trip multiple is 1
;
entry:
  %umin = call i32 @llvm.umin.i32(i32 %N, i32 16)
  %ext = zext i32 %umin to i64
  %min.iters.check = icmp ult i64 %ext, 4
  br i1 %min.iters.check, label %exit, label %loop.ph

loop.ph:
  %n.vec = and i64 %ext, 28
  br label %loop

; %n.vec is [4, 16) and a multiple of 4.
loop:
  %index = phi i64 [ 0, %loop.ph ], [ %index.next, %loop ]
  %gep = getelementptr inbounds i32, i32* %arr, i64 %index
  store i32 0, i32* %gep
  %index.next = add nuw i64 %index, 4
  %ec = icmp eq i64 %index.next, %n.vec
  br i1 %ec, label %exit, label %loop

exit:
  ret i32 0
}

; Test case from PR52464. applyLoopGuards needs to apply information about %and
; to %ext, which requires rewriting the zext.
define i32 @rewrite_zext_with_info_from_icmp_ne(i32 %N) {
; CHECK-LABEL: Determining loop execution counts for: @rewrite_zext_with_info_from_icmp_ne
; CHECK-NEXT:  Loop %loop: backedge-taken count is 0
; CHECK-NEXT:  Loop %loop: max backedge-taken count is 0
; CHECK-NEXT:  Loop %loop: Predicated backedge-taken count is 0
; CHECK-NEXT:   Predicates:
; CHECK-EMPTY:
; CHECK-NEXT:  Loop %loop: Trip multiple is 1
;
entry:
  %and = and i32 %N, 3
  %cmp6.not = icmp eq i32 %and, 0
  br i1 %cmp6.not, label %exit, label %loop.ph

loop.ph:
  %and.sub.1 = add nsw i32 %and, -1
  %ext = zext i32 %and.sub.1 to i64
  %n.rnd.up = add nuw nsw i64 %ext, 4
  %n.vec = and i64 %n.rnd.up, 8589934588
  br label %loop

loop:
  %iv = phi i64 [ 0, %loop.ph ], [ %iv.next, %loop ]
  %iv.next = add i64 %iv, 4
  call void @use(i64 %iv.next)
  %ec = icmp eq i64 %iv.next, %n.vec
  br i1 %ec, label %exit, label %loop

exit:
  ret i32 0
}

; Similar to @rewrite_zext_with_info_from_icmp_ne, but the loop is not guarded by %and != 0,
; hence the subsequent subtraction may yield a negative number.
define i32 @rewrite_zext_no_icmp_ne(i32 %N) {
; CHECK-LABEL: Determining loop execution counts for: @rewrite_zext_no_icmp_ne
; CHECK-NEXT:  Loop %loop: backedge-taken count is ((-4 + (4 * ((4 + (zext i32 (-1 + (zext i2 (trunc i32 %N to i2) to i32))<nsw> to i64))<nuw><nsw> /u 4))<nuw><nsw>)<nsw> /u 4)
; CHECK-NEXT:  Loop %loop: max backedge-taken count is 1073741823
; CHECK-NEXT:  Loop %loop: Predicated backedge-taken count is ((-4 + (4 * ((4 + (zext i32 (-1 + (zext i2 (trunc i32 %N to i2) to i32))<nsw> to i64))<nuw><nsw> /u 4))<nuw><nsw>)<nsw> /u 4)
; CHECK-NEXT:   Predicates:
; CHECK-EMPTY:
; CHECK-NEXT:  Loop %loop: Trip multiple is 1
;
entry:
  %and = and i32 %N, 3
  br label %loop.ph

loop.ph:
  %and.sub.1 = add nsw i32 %and, -1
  %ext = zext i32 %and.sub.1 to i64
  %n.rnd.up = add nuw nsw i64 %ext, 4
  %n.vec = and i64 %n.rnd.up, 8589934588
  br label %loop

loop:
  %iv = phi i64 [ 0, %loop.ph ], [ %iv.next, %loop ]
  %iv.next = add i64 %iv, 4
  call void @use(i64 %iv.next)
  %ec = icmp eq i64 %iv.next, %n.vec
  br i1 %ec, label %exit, label %loop

exit:
  ret i32 0
}

; Make sure no information is lost for conditions on both %n and (zext %n).
define void @rewrite_zext_and_base_1(i32 %n) {
; CHECK-LABEL: Determining loop execution counts for: @rewrite_zext_and_base
; CHECK-NEXT:  Loop %loop: backedge-taken count is ((-8 + (8 * ((zext i32 %n to i64) /u 8))<nuw><nsw>)<nsw> /u 8)
; CHECK-NEXT:  Loop %loop: max backedge-taken count is 3
; CHECK-NEXT:  Loop %loop: Predicated backedge-taken count is ((-8 + (8 * ((zext i32 %n to i64) /u 8))<nuw><nsw>)<nsw> /u 8)
; CHECK-NEXT:  Predicates:
; CHECK:        Loop %loop: Trip multiple is 1
;
entry:
  %ext = zext i32 %n to i64
  %cmp5 = icmp ule i64 %ext, 48
  br i1 %cmp5, label %check.1, label %exit

check.1:
  %cmp.2 = icmp ule i32 %n, 32
  br i1 %cmp.2, label %check, label %exit


check:                                 ; preds = %entry
  %min.iters.check = icmp ult i64 %ext, 8
  %n.vec = and i64 %ext, -8
  br i1 %min.iters.check, label %exit, label %loop

loop:
  %index = phi i64 [ 0, %check ], [ %index.next, %loop ]
  %index.next = add nuw nsw i64 %index, 8
  %ec = icmp eq i64 %index.next, %n.vec
  br i1 %ec, label %exit, label %loop

exit:
  ret void
}

; Make sure no information is lost for conditions on both %n and (zext %n).
define void @rewrite_zext_and_base_2(i32 %n) {
; CHECK-LABEL: Determining loop execution counts for: @rewrite_zext_and_base
; CHECK-NEXT:  Loop %loop: backedge-taken count is ((-8 + (8 * ((zext i32 %n to i64) /u 8))<nuw><nsw>)<nsw> /u 8)
; CHECK-NEXT:  Loop %loop: max backedge-taken count is 3
; CHECK-NEXT:  Loop %loop: Predicated backedge-taken count is ((-8 + (8 * ((zext i32 %n to i64) /u 8))<nuw><nsw>)<nsw> /u 8)
; CHECK-NEXT:  Predicates:
; CHECK:        Loop %loop: Trip multiple is 1
;
entry:
  %ext = zext i32 %n to i64
  %cmp5 = icmp ule i64 %ext, 32
  br i1 %cmp5, label %check.1, label %exit

check.1:
  %cmp.2 = icmp ule i32 %n, 48
  br i1 %cmp.2, label %check, label %exit

check:                                 ; preds = %entry
  %min.iters.check = icmp ult i64 %ext, 8
  %n.vec = and i64 %ext, -8
  br i1 %min.iters.check, label %exit, label %loop

loop:
  %index = phi i64 [ 0, %check ], [ %index.next, %loop ]
  %index.next = add nuw nsw i64 %index, 8
  %ec = icmp eq i64 %index.next, %n.vec
  br i1 %ec, label %exit, label %loop

exit:
  ret void
}

define void @guard_pessimizes_analysis_step2(i1 %c, i32 %N) {
; CHECK-LABEL: 'guard_pessimizes_analysis_step2'
; CHECK:       Determining loop execution counts for: @guard_pessimizes_analysis_step2
; CHECK-NEXT:  Loop %loop: backedge-taken count is ((14 + (-1 * %init)<nsw>)<nsw> /u 2)
; CHECK-NEXT:  Loop %loop: max backedge-taken count is 6
; CHECK-NEXT:  Loop %loop: Predicated backedge-taken count is ((14 + (-1 * %init)<nsw>)<nsw> /u 2)
; CHECK-NEXT:   Predicates:
; CHECK-EMPTY:
; CHECK-NEXT:  Loop %loop: Trip multiple is 1
;
entry:
  %N.ext = zext i32 %N to i64
  br i1 %c, label %bb1, label %guard

bb1:
  br label %guard

guard:
  %init = phi i64 [ 2, %entry ], [ 4, %bb1 ]
  %c.1 = icmp ult i64 %init, %N.ext
  br i1 %c.1, label %loop.ph, label %exit

loop.ph:
  br label %loop

loop:
  %iv = phi i64 [ %iv.next, %loop ], [ %init, %loop.ph ]
  %iv.next = add i64 %iv, 2
  %exitcond = icmp eq i64 %iv.next, 16
  br i1 %exitcond, label %exit, label %loop

exit:
  ret void
}
declare void @use(i64)

declare i32 @llvm.umin.i32(i32, i32)
