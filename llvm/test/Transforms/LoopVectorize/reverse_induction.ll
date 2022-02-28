; RUN: opt < %s -loop-vectorize -force-vector-interleave=2 -force-vector-width=4 -S | FileCheck %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"

; Make sure consecutive vector generates correct negative indices.
; PR15882

; CHECK: %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
; CHECK: [[OFFSET_IDX:%.+]] = sub i64 %startval, %index
; CHECK: %[[a0:.+]] = add i64 [[OFFSET_IDX]], 0
; CHECK: %[[a4:.+]] = add i64 [[OFFSET_IDX]], -4

define i32 @reverse_induction_i64(i64 %startval, i32 * %ptr) {
entry:
  br label %for.body

for.body:
  %add.i7 = phi i64 [ %startval, %entry ], [ %add.i, %for.body ]
  %i.06 = phi i32 [ 0, %entry ], [ %inc4, %for.body ]
  %redux5 = phi i32 [ 0, %entry ], [ %inc.redux, %for.body ]
  %add.i = add i64 %add.i7, -1
  %kind_.i = getelementptr inbounds i32, i32* %ptr, i64 %add.i
  %tmp.i1 = load i32, i32* %kind_.i, align 4
  %inc.redux = add i32 %tmp.i1, %redux5
  %inc4 = add i32 %i.06, 1
  %exitcond = icmp ne i32 %inc4, 1024
  br i1 %exitcond, label %for.body, label %loopend

loopend:
  ret i32 %inc.redux
}

; CHECK-LABEL: @reverse_induction_i128(
; CHECK: %index = phi i128 [ 0, %vector.ph ], [ %index.next, %vector.body ]
; CHECK: [[OFFSET_IDX:%.+]] = sub i128 %startval, %index
; CHECK: %[[a0:.+]] = add i128 [[OFFSET_IDX]], 0
; CHECK: %[[a4:.+]] = add i128 [[OFFSET_IDX]], -4

define i32 @reverse_induction_i128(i128 %startval, i32 * %ptr) {
entry:
  br label %for.body

for.body:
  %add.i7 = phi i128 [ %startval, %entry ], [ %add.i, %for.body ]
  %i.06 = phi i32 [ 0, %entry ], [ %inc4, %for.body ]
  %redux5 = phi i32 [ 0, %entry ], [ %inc.redux, %for.body ]
  %add.i = add i128 %add.i7, -1
  %kind_.i = getelementptr inbounds i32, i32* %ptr, i128 %add.i
  %tmp.i1 = load i32, i32* %kind_.i, align 4
  %inc.redux = add i32 %tmp.i1, %redux5
  %inc4 = add i32 %i.06, 1
  %exitcond = icmp ne i32 %inc4, 1024
  br i1 %exitcond, label %for.body, label %loopend

loopend:
  ret i32 %inc.redux
}

; CHECK-LABEL: @reverse_induction_i16(
; CHECK: %index = phi i32 [ 0, %vector.ph ], [ %index.next, %vector.body ]
; CHECK: [[OFFSET_IDX:%.+]] = sub i16 %startval, {{.*}}
; CHECK: %[[a0:.+]] = add i16 [[OFFSET_IDX]], 0
; CHECK: %[[a4:.+]] = add i16 [[OFFSET_IDX]], -4

define i32 @reverse_induction_i16(i16 %startval, i32 * %ptr) {
entry:
  br label %for.body

for.body:
  %add.i7 = phi i16 [ %startval, %entry ], [ %add.i, %for.body ]
  %i.06 = phi i32 [ 0, %entry ], [ %inc4, %for.body ]
  %redux5 = phi i32 [ 0, %entry ], [ %inc.redux, %for.body ]
  %add.i = add i16 %add.i7, -1
  %kind_.i = getelementptr inbounds i32, i32* %ptr, i16 %add.i
  %tmp.i1 = load i32, i32* %kind_.i, align 4
  %inc.redux = add i32 %tmp.i1, %redux5
  %inc4 = add i32 %i.06, 1
  %exitcond = icmp ne i32 %inc4, 1024
  br i1 %exitcond, label %for.body, label %loopend

loopend:
  ret i32 %inc.redux
}


@a = common global [1024 x i32] zeroinitializer, align 16

; We incorrectly transformed this loop into an empty one because we left the
; induction variable in i8 type and truncated the exit value 1024 to 0.
; int a[1024];
;
; void fail() {
;   int reverse_induction = 1023;
;   unsigned char forward_induction = 0;
;   while ((reverse_induction) >= 0) {
;     forward_induction++;
;     a[reverse_induction] = forward_induction;
;     --reverse_induction;
;   }
; }

; CHECK-LABEL: @reverse_forward_induction_i64_i8(
; CHECK: %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
; CHECK: [[OFFSET_IDX:%.+]] = sub i64 1023, %index
; CHECK: %[[a0:.+]] = add i64 [[OFFSET_IDX]], 0
; CHECK: %[[a4:.+]] = add i64 [[OFFSET_IDX]], -4

define void @reverse_forward_induction_i64_i8() {
entry:
  br label %while.body

while.body:
  %indvars.iv = phi i64 [ 1023, %entry ], [ %indvars.iv.next, %while.body ]
  %forward_induction.05 = phi i8 [ 0, %entry ], [ %inc, %while.body ]
  %inc = add i8 %forward_induction.05, 1
  %conv = zext i8 %inc to i32
  %arrayidx = getelementptr inbounds [1024 x i32], [1024 x i32]* @a, i64 0, i64 %indvars.iv
  store i32 %conv, i32* %arrayidx, align 4
  %indvars.iv.next = add i64 %indvars.iv, -1
  %0 = trunc i64 %indvars.iv to i32
  %cmp = icmp sgt i32 %0, 0
  br i1 %cmp, label %while.body, label %while.end

while.end:
  ret void
}

; CHECK-LABEL: @reverse_forward_induction_i64_i8_signed(
; CHECK: %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
; CHECK: [[OFFSET_IDX:%.+]] = sub i64 1023, %index
; CHECK: %[[a0:.+]] = add i64 [[OFFSET_IDX]], 0
; CHECK: %[[a4:.+]] = add i64 [[OFFSET_IDX]], -4

define void @reverse_forward_induction_i64_i8_signed() {
entry:
  br label %while.body

while.body:
  %indvars.iv = phi i64 [ 1023, %entry ], [ %indvars.iv.next, %while.body ]
  %forward_induction.05 = phi i8 [ -127, %entry ], [ %inc, %while.body ]
  %inc = add i8 %forward_induction.05, 1
  %conv = sext i8 %inc to i32
  %arrayidx = getelementptr inbounds [1024 x i32], [1024 x i32]* @a, i64 0, i64 %indvars.iv
  store i32 %conv, i32* %arrayidx, align 4
  %indvars.iv.next = add i64 %indvars.iv, -1
  %0 = trunc i64 %indvars.iv to i32
  %cmp = icmp sgt i32 %0, 0
  br i1 %cmp, label %while.body, label %while.end

while.end:
  ret void
}
