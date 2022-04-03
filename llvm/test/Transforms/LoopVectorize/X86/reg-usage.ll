; RUN: opt < %s -debug-only=loop-vectorize -loop-vectorize -vectorizer-maximize-bandwidth -mtriple=x86_64-unknown-linux -S 2>&1 | FileCheck %s
; RUN: opt < %s -debug-only=loop-vectorize -loop-vectorize -vectorizer-maximize-bandwidth -mtriple=x86_64-unknown-linux -mattr=+avx512f -S 2>&1 | FileCheck %s --check-prefix=AVX512F
; REQUIRES: asserts

@a = global [1024 x i8] zeroinitializer, align 16
@b = global [1024 x i8] zeroinitializer, align 16

define i32 @foo() {
; This function has a loop of SAD pattern. Here we check when VF = 16 the
; register usage doesn't exceed 16.
;
; CHECK-LABEL: foo
; CHECK:      LV(REG): VF = 8
; CHECK-NEXT: LV(REG): Found max usage: 2 item
; CHECK-NEXT: LV(REG): RegisterClass: Generic::ScalarRC, 2 registers
; CHECK-NEXT: LV(REG): RegisterClass: Generic::VectorRC, 7 registers
; CHECK-NEXT: LV(REG): Found invariant usage: 0 item
; CHECK:      LV(REG): VF = 16
; CHECK-NEXT: LV(REG): Found max usage: 2 item
; CHECK-NEXT: LV(REG): RegisterClass: Generic::ScalarRC, 2 registers
; CHECK-NEXT: LV(REG): RegisterClass: Generic::VectorRC, 13 registers
; CHECK-NEXT: LV(REG): Found invariant usage: 0 item

entry:
  br label %for.body

for.cond.cleanup:
  %add.lcssa = phi i32 [ %add, %for.body ]
  ret i32 %add.lcssa

for.body:
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %s.015 = phi i32 [ 0, %entry ], [ %add, %for.body ]
  %arrayidx = getelementptr inbounds [1024 x i8], [1024 x i8]* @a, i64 0, i64 %indvars.iv
  %0 = load i8, i8* %arrayidx, align 1
  %conv = zext i8 %0 to i32
  %arrayidx2 = getelementptr inbounds [1024 x i8], [1024 x i8]* @b, i64 0, i64 %indvars.iv
  %1 = load i8, i8* %arrayidx2, align 1
  %conv3 = zext i8 %1 to i32
  %sub = sub nsw i32 %conv, %conv3
  %ispos = icmp sgt i32 %sub, -1
  %neg = sub nsw i32 0, %sub
  %2 = select i1 %ispos, i32 %sub, i32 %neg
  %add = add nsw i32 %2, %s.015
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, 1024
  br i1 %exitcond, label %for.cond.cleanup, label %for.body
}

define i32 @goo() {
; For indvars.iv used in a computating chain only feeding into getelementptr or cmp,
; it will not have vector version and the vector register usage will not exceed the
; available vector register number.
; CHECK-LABEL: goo
; CHECK:      LV(REG): VF = 8
; CHECK-NEXT: LV(REG): Found max usage: 2 item
; CHECK-NEXT: LV(REG): RegisterClass: Generic::ScalarRC, 2 registers
; CHECK-NEXT: LV(REG): RegisterClass: Generic::VectorRC, 7 registers
; CHECK-NEXT: LV(REG): Found invariant usage: 0 item
; CHECK:      LV(REG): VF = 16
; CHECK-NEXT: LV(REG): Found max usage: 2 item
; CHECK-NEXT: LV(REG): RegisterClass: Generic::ScalarRC, 2 registers
; CHECK-NEXT: LV(REG): RegisterClass: Generic::VectorRC, 13 registers
; CHECK-NEXT: LV(REG): Found invariant usage: 0 item
entry:
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body
  %add.lcssa = phi i32 [ %add, %for.body ]
  ret i32 %add.lcssa

for.body:                                         ; preds = %for.body, %entry
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %s.015 = phi i32 [ 0, %entry ], [ %add, %for.body ]
  %tmp1 = add nsw i64 %indvars.iv, 3
  %arrayidx = getelementptr inbounds [1024 x i8], [1024 x i8]* @a, i64 0, i64 %tmp1
  %tmp = load i8, i8* %arrayidx, align 1
  %conv = zext i8 %tmp to i32
  %tmp2 = add nsw i64 %indvars.iv, 2
  %arrayidx2 = getelementptr inbounds [1024 x i8], [1024 x i8]* @b, i64 0, i64 %tmp2
  %tmp3 = load i8, i8* %arrayidx2, align 1
  %conv3 = zext i8 %tmp3 to i32
  %sub = sub nsw i32 %conv, %conv3
  %ispos = icmp sgt i32 %sub, -1
  %neg = sub nsw i32 0, %sub
  %tmp4 = select i1 %ispos, i32 %sub, i32 %neg
  %add = add nsw i32 %tmp4, %s.015
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, 1024
  br i1 %exitcond, label %for.cond.cleanup, label %for.body
}

define i64 @bar(i64* nocapture %a) {
; CHECK-LABEL: bar
; CHECK:       LV(REG): VF = 2
; CHECK-NEXT: LV(REG): Found max usage: 2 item
; CHECK-NEXT: LV(REG): RegisterClass: Generic::VectorRC, 3 registers
; CHECK-NEXT: LV(REG): RegisterClass: Generic::ScalarRC, 1 registers
; CHECK-NEXT: LV(REG): Found invariant usage: 0 item

entry:
  br label %for.body

for.cond.cleanup:
  %add2.lcssa = phi i64 [ %add2, %for.body ]
  ret i64 %add2.lcssa

for.body:
  %i.012 = phi i64 [ 0, %entry ], [ %inc, %for.body ]
  %s.011 = phi i64 [ 0, %entry ], [ %add2, %for.body ]
  %arrayidx = getelementptr inbounds i64, i64* %a, i64 %i.012
  %0 = load i64, i64* %arrayidx, align 8
  %add = add nsw i64 %0, %i.012
  store i64 %add, i64* %arrayidx, align 8
  %add2 = add nsw i64 %add, %s.011
  %inc = add nuw nsw i64 %i.012, 1
  %exitcond = icmp eq i64 %inc, 1024
  br i1 %exitcond, label %for.cond.cleanup, label %for.body
}

@d = external global [0 x i64], align 8
@e = external global [0 x i32], align 4
@c = external global [0 x i32], align 4

define void @hoo(i32 %n) {
; For c[i] = e[d[i]] in the loop, e[d[i]] is not consecutive but its index %tmp can
; be gathered into a vector. For VF == 16, the vector version of %tmp will be <16 x i64>
; so the max usage of AVX512 vector register will be 2.
; AVX512F-LABEL: bar
; AVX512F:       LV(REG): VF = 16
; AVX512F-CHECK: LV(REG): Found max usage: 2 item
; AVX512F-CHECK: LV(REG): RegisterClass: Generic::ScalarRC, 2 registers
; AVX512F-CHECK: LV(REG): RegisterClass: Generic::VectorRC, 2 registers
; AVX512F-CHECK: LV(REG): Found invariant usage: 0 item

entry:
  br label %for.body

for.body:                                         ; preds = %for.body, %entry
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds [0 x i64], [0 x i64]* @d, i64 0, i64 %indvars.iv
  %tmp = load i64, i64* %arrayidx, align 8
  %arrayidx1 = getelementptr inbounds [0 x i32], [0 x i32]* @e, i64 0, i64 %tmp
  %tmp1 = load i32, i32* %arrayidx1, align 4
  %arrayidx3 = getelementptr inbounds [0 x i32], [0 x i32]* @c, i64 0, i64 %indvars.iv
  store i32 %tmp1, i32* %arrayidx3, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, 10000
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body
  ret void
}
