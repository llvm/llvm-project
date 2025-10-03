; RUN: opt < %s -passes=loop-interchange -cache-line-size=64 -pass-remarks-output=%t -disable-output \
; RUN:     -verify-dom-info -verify-loop-info -verify-loop-lcssa
; RUN: FileCheck -input-file=%t %s

; Check that the loops aren't exchanged if there is a reduction of
; non-reassociative floating-point addition.
;
; float sum = 0;
; for (int i = 0; i < 2; i++)
;   for (int j = 0; j < 2; j++)
;     sum += A[j][i];

; CHECK:      --- !Missed
; CHECK-NEXT: Pass:            loop-interchange
; CHECK-NEXT: Name:            UnsupportedPHIOuter
; CHECK-NEXT: Function:        reduction_fadd
define void @reduction_fadd(ptr %A) {
entry:
  br label %for.i.header

for.i.header:
  %i = phi i32 [ 0, %entry ], [ %i.inc, %for.i.latch ]
  %sum.i = phi float [ 0.0, %entry ], [ %sum.i.lcssa, %for.i.latch ]
  br label %for.j

for.j:
  %j = phi i32 [ 0, %for.i.header ], [ %j.inc, %for.j ]
  %sum.j = phi float [ %sum.i, %for.i.header ], [ %sum.j.next, %for.j ]
  %idx = getelementptr inbounds [2 x [2 x i32]], ptr %A, i32 0, i32 %j, i32 %i
  %a = load float, ptr %idx, align 4
  %sum.j.next = fadd float %sum.j, %a
  %j.inc = add i32 %j, 1
  %cmp.j = icmp slt i32 %j.inc, 2
  br i1 %cmp.j, label %for.j, label %for.i.latch

for.i.latch:
  %sum.i.lcssa = phi float [ %sum.j.next, %for.j ]
  %i.inc = add i32 %i, 1
  %cmp.i = icmp slt i32 %i.inc, 2
  br i1 %cmp.i, label %for.i.header, label %exit

exit:
  ret void
}

; Check that the interchange is legal if the floating-point addition is marked
; as reassoc.
;
; CHECK:      --- !Pass
; CHECK-NEXT: Pass:            loop-interchange
; CHECK-NEXT: Name:            Interchanged
; CHECK-NEXT: Function:        reduction_reassoc_fadd
define void @reduction_reassoc_fadd(ptr %A) {
entry:
  br label %for.i.header

for.i.header:
  %i = phi i32 [ 0, %entry ], [ %i.inc, %for.i.latch ]
  %sum.i = phi float [ 0.0, %entry ], [ %sum.i.lcssa, %for.i.latch ]
  br label %for.j

for.j:
  %j = phi i32 [ 0, %for.i.header ], [ %j.inc, %for.j ]
  %sum.j = phi float [ %sum.i, %for.i.header ], [ %sum.j.next, %for.j ]
  %idx = getelementptr inbounds [2 x [2 x i32]], ptr %A, i32 0, i32 %j, i32 %i
  %a = load float, ptr %idx, align 4
  %sum.j.next = fadd reassoc float %sum.j, %a
  %j.inc = add i32 %j, 1
  %cmp.j = icmp slt i32 %j.inc, 2
  br i1 %cmp.j, label %for.j, label %for.i.latch

for.i.latch:
  %sum.i.lcssa = phi float [ %sum.j.next, %for.j ]
  %i.inc = add i32 %i, 1
  %cmp.i = icmp slt i32 %i.inc, 2
  br i1 %cmp.i, label %for.i.header, label %exit

exit:
  ret void
}

; FIXME: Is it really legal to interchange the loops when
; both reassoc and ninf are set?
; Check that the interchange is legal if the floating-point addition is marked
; as reassoc.
;
; CHECK:      --- !Pass
; CHECK-NEXT: Pass:            loop-interchange
; CHECK-NEXT: Name:            Interchanged
; CHECK-NEXT: Function:        reduction_reassoc_ninf_fadd
define void @reduction_reassoc_ninf_fadd(ptr %A) {
entry:
  br label %for.i.header

for.i.header:
  %i = phi i32 [ 0, %entry ], [ %i.inc, %for.i.latch ]
  %sum.i = phi float [ 0.0, %entry ], [ %sum.i.lcssa, %for.i.latch ]
  br label %for.j

for.j:
  %j = phi i32 [ 0, %for.i.header ], [ %j.inc, %for.j ]
  %sum.j = phi float [ %sum.i, %for.i.header ], [ %sum.j.next, %for.j ]
  %idx = getelementptr inbounds [2 x [2 x i32]], ptr %A, i32 0, i32 %j, i32 %i
  %a = load float, ptr %idx, align 4
  %sum.j.next = fadd reassoc ninf float %sum.j, %a
  %j.inc = add i32 %j, 1
  %cmp.j = icmp slt i32 %j.inc, 2
  br i1 %cmp.j, label %for.j, label %for.i.latch

for.i.latch:
  %sum.i.lcssa = phi float [ %sum.j.next, %for.j ]
  %i.inc = add i32 %i, 1
  %cmp.i = icmp slt i32 %i.inc, 2
  br i1 %cmp.i, label %for.i.header, label %exit

exit:
  ret void
}

; Check that the loops aren't exchanged if there is a reduction of
; non-reassociative floating-point multiplication.
;
; float prod = 1;
; for (int i = 0; i < 2; i++)
;   for (int j = 0; j < 2; j++)
;     prod *= A[j][i];

; CHECK:      --- !Missed
; CHECK-NEXT: Pass:            loop-interchange
; CHECK-NEXT: Name:            UnsupportedPHIOuter
; CHECK-NEXT: Function:        reduction_fmul
define void @reduction_fmul(ptr %A) {
entry:
  br label %for.i.header

for.i.header:
  %i = phi i32 [ 0, %entry ], [ %i.inc, %for.i.latch ]
  %prod.i = phi float [ 1.0, %entry ], [ %prod.i.lcssa, %for.i.latch ]
  br label %for.j

for.j:
  %j = phi i32 [ 0, %for.i.header ], [ %j.inc, %for.j ]
  %prod.j = phi float [ %prod.i, %for.i.header ], [ %prod.j.next, %for.j ]
  %idx = getelementptr inbounds [2 x [2 x i32]], ptr %A, i32 0, i32 %j, i32 %i
  %a = load float, ptr %idx, align 4
  %prod.j.next = fmul float %prod.j, %a
  %j.inc = add i32 %j, 1
  %cmp.j = icmp slt i32 %j.inc, 2
  br i1 %cmp.j, label %for.j, label %for.i.latch

for.i.latch:
  %prod.i.lcssa = phi float [ %prod.j.next, %for.j ]
  %i.inc = add i32 %i, 1
  %cmp.i = icmp slt i32 %i.inc, 2
  br i1 %cmp.i, label %for.i.header, label %exit

exit:
  ret void
}

; Check that the interchange is legal if the floating-point multiplication is
; marked as reassoc.
;
; CHECK:      --- !Pass
; CHECK-NEXT: Pass:            loop-interchange
; CHECK-NEXT: Name:            Interchanged
; CHECK-NEXT: Function:        reduction_reassoc_fmul
define void @reduction_reassoc_fmul(ptr %A) {
entry:
  br label %for.i.header

for.i.header:
  %i = phi i32 [ 0, %entry ], [ %i.inc, %for.i.latch ]
  %prod.i = phi float [ 1.0, %entry ], [ %prod.i.lcssa, %for.i.latch ]
  br label %for.j

for.j:
  %j = phi i32 [ 0, %for.i.header ], [ %j.inc, %for.j ]
  %prod.j = phi float [ %prod.i, %for.i.header ], [ %prod.j.next, %for.j ]
  %idx = getelementptr inbounds [2 x [2 x i32]], ptr %A, i32 0, i32 %j, i32 %i
  %a = load float, ptr %idx, align 4
  %prod.j.next = fmul reassoc float %prod.j, %a
  %j.inc = add i32 %j, 1
  %cmp.j = icmp slt i32 %j.inc, 2
  br i1 %cmp.j, label %for.j, label %for.i.latch

for.i.latch:
  %prod.i.lcssa = phi float [ %prod.j.next, %for.j ]
  %i.inc = add i32 %i, 1
  %cmp.i = icmp slt i32 %i.inc, 2
  br i1 %cmp.i, label %for.i.header, label %exit

exit:
  ret void
}

; Check that the loops aren't exchanged if there is a reduction of
; non-reassociative floating-point fmuladd.
;
; float fmuladd = 0;
; for (int i = 0; i < 2; i++)
;   for (int j = 0; j < 2; j++)
;     fmuladd += A[j][i] * B[j][i];

; CHECK:      --- !Missed
; CHECK-NEXT: Pass:            loop-interchange
; CHECK-NEXT: Name:            UnsupportedPHIOuter
; CHECK-NEXT: Function:        reduction_fmuladd
define void @reduction_fmuladd(ptr %A, ptr %B) {
entry:
  br label %for.i.header

for.i.header:
  %i = phi i32 [ 0, %entry ], [ %i.inc, %for.i.latch ]
  %fmuladd.i = phi float [ 1.0, %entry ], [ %fmuladd.i.lcssa, %for.i.latch ]
  br label %for.j

for.j:
  %j = phi i32 [ 0, %for.i.header ], [ %j.inc, %for.j ]
  %fmuladd.j = phi float [ %fmuladd.i, %for.i.header ], [ %fmuladd.j.next, %for.j ]
  %idx.a = getelementptr inbounds [2 x [2 x i32]], ptr %A, i32 0, i32 %j, i32 %i
  %idx.b = getelementptr inbounds [2 x [2 x i32]], ptr %B, i32 0, i32 %j, i32 %i
  %a = load float, ptr %idx.a, align 4
  %b = load float, ptr %idx.b, align 4
  %fmuladd.j.next = call float @llvm.fmuladd.f32(float %a, float %b, float %fmuladd.j)
  %j.inc = add i32 %j, 1
  %cmp.j = icmp slt i32 %j.inc, 2
  br i1 %cmp.j, label %for.j, label %for.i.latch

for.i.latch:
  %fmuladd.i.lcssa = phi float [ %fmuladd.j.next, %for.j ]
  %i.inc = add i32 %i, 1
  %cmp.i = icmp slt i32 %i.inc, 2
  br i1 %cmp.i, label %for.i.header, label %exit

exit:
  ret void
}

; Check that the interchange is legal if the floating-point fmuladd is marked
; as reassoc.
;
; CHECK:      --- !Pass
; CHECK-NEXT: Pass:            loop-interchange
; CHECK-NEXT: Name:            Interchanged
; CHECK-NEXT: Function:        reduction_reassoc_fmuladd
define void @reduction_reassoc_fmuladd(ptr %A, ptr %B) {
entry:
  br label %for.i.header

for.i.header:
  %i = phi i32 [ 0, %entry ], [ %i.inc, %for.i.latch ]
  %fmuladd.i = phi float [ 1.0, %entry ], [ %fmuladd.i.lcssa, %for.i.latch ]
  br label %for.j

for.j:
  %j = phi i32 [ 0, %for.i.header ], [ %j.inc, %for.j ]
  %fmuladd.j = phi float [ %fmuladd.i, %for.i.header ], [ %fmuladd.j.next, %for.j ]
  %idx.a = getelementptr inbounds [2 x [2 x i32]], ptr %A, i32 0, i32 %j, i32 %i
  %idx.b = getelementptr inbounds [2 x [2 x i32]], ptr %B, i32 0, i32 %j, i32 %i
  %a = load float, ptr %idx.a, align 4
  %b = load float, ptr %idx.b, align 4
  %fmuladd.j.next = call reassoc float @llvm.fmuladd.f32(float %a, float %b, float %fmuladd.j)
  %j.inc = add i32 %j, 1
  %cmp.j = icmp slt i32 %j.inc, 2
  br i1 %cmp.j, label %for.j, label %for.i.latch

for.i.latch:
  %fmuladd.i.lcssa = phi float [ %fmuladd.j.next, %for.j ]
  %i.inc = add i32 %i, 1
  %cmp.i = icmp slt i32 %i.inc, 2
  br i1 %cmp.i, label %for.i.header, label %exit

exit:
  ret void
}

; Check that interchanging the loops is legal for the reassociative
; floating-point minimum.
;
; float fmin = init;
; for (int i = 0; i < 2; i++)
;   for (int j = 0; j < 2; j++)
;     fmin = (A[j][i] < fmin) ? A[j][i] : fmin;

; CHECK:      --- !Pass
; CHECK-NEXT: Pass:            loop-interchange
; CHECK-NEXT: Name:            Interchanged
; CHECK-NEXT: Function:        reduction_fmin
define void @reduction_fmin(ptr %A, float %init) {
entry:
  br label %for.i.header

for.i.header:
  %i = phi i32 [ 0, %entry ], [ %i.inc, %for.i.latch ]
  %fmin.i = phi float [ %init, %entry ], [ %fmin.i.lcssa, %for.i.latch ]
  br label %for.j

for.j:
  %j = phi i32 [ 0, %for.i.header ], [ %j.inc, %for.j ]
  %fmin.j = phi float [ %fmin.i, %for.i.header ], [ %fmin.j.next, %for.j ]
  %idx = getelementptr inbounds [2 x [2 x i32]], ptr %A, i32 0, i32 %j, i32 %i
  %a = load float, ptr %idx, align 4
  %cmp = fcmp nnan nsz olt float %a, %fmin.j
  %fmin.j.next = select nnan nsz i1 %cmp, float %a, float %fmin.j
  %j.inc = add i32 %j, 1
  %cmp.j = icmp slt i32 %j.inc, 2
  br i1 %cmp.j, label %for.j, label %for.i.latch

for.i.latch:
  %fmin.i.lcssa = phi float [ %fmin.j.next, %for.j ]
  %i.inc = add i32 %i, 1
  %cmp.i = icmp slt i32 %i.inc, 2
  br i1 %cmp.i, label %for.i.header, label %exit

exit:
  ret void
}


; Check that interchanging the loops is legal for the floating-point
; llvm.minimumnum.
;
; CHECK:      --- !Pass
; CHECK-NEXT: Pass:            loop-interchange
; CHECK-NEXT: Name:            Interchanged
; CHECK-NEXT: Function:        reduction_fmininumnum
define void @reduction_fmininumnum(ptr %A, float %init) {
entry:
  br label %for.i.header

for.i.header:
  %i = phi i32 [ 0, %entry ], [ %i.inc, %for.i.latch ]
  %fmin.i = phi float [ %init, %entry ], [ %fmin.i.lcssa, %for.i.latch ]
  br label %for.j

for.j:
  %j = phi i32 [ 0, %for.i.header ], [ %j.inc, %for.j ]
  %fmin.j = phi float [ %fmin.i, %for.i.header ], [ %fmin.j.next, %for.j ]
  %idx = getelementptr inbounds [2 x [2 x i32]], ptr %A, i32 0, i32 %j, i32 %i
  %a = load float, ptr %idx, align 4
  %fmin.j.next = call float @llvm.minimumnum.f32(float %a, float %fmin.j)
  %j.inc = add i32 %j, 1
  %cmp.j = icmp slt i32 %j.inc, 2
  br i1 %cmp.j, label %for.j, label %for.i.latch

for.i.latch:
  %fmin.i.lcssa = phi float [ %fmin.j.next, %for.j ]
  %i.inc = add i32 %i, 1
  %cmp.i = icmp slt i32 %i.inc, 2
  br i1 %cmp.i, label %for.i.header, label %exit

exit:
  ret void
}

; Check that interchanging the loops is legal for the reassociative
; floating-point maximum.
;
; float fmax = init;
; for (int i = 0; i < 2; i++)
;   for (int j = 0; j < 2; j++)
;     fmax = (A[j][i] > fmax) ? A[j][i] : fmax;

; CHECK:      --- !Pass
; CHECK-NEXT: Pass:            loop-interchange
; CHECK-NEXT: Name:            Interchanged
; CHECK-NEXT: Function:        reduction_fmax
define void @reduction_fmax(ptr %A, float %init) {
entry:
  br label %for.i.header

for.i.header:
  %i = phi i32 [ 0, %entry ], [ %i.inc, %for.i.latch ]
  %fmax.i = phi float [ %init, %entry ], [ %fmax.i.lcssa, %for.i.latch ]
  br label %for.j

for.j:
  %j = phi i32 [ 0, %for.i.header ], [ %j.inc, %for.j ]
  %fmax.j = phi float [ %fmax.i, %for.i.header ], [ %fmax.j.next, %for.j ]
  %idx = getelementptr inbounds [2 x [2 x i32]], ptr %A, i32 0, i32 %j, i32 %i
  %a = load float, ptr %idx, align 4
  %cmp = fcmp nnan nsz ogt float %a, %fmax.j
  %fmax.j.next = select nnan nsz i1 %cmp, float %a, float %fmax.j
  %j.inc = add i32 %j, 1
  %cmp.j = icmp slt i32 %j.inc, 2
  br i1 %cmp.j, label %for.j, label %for.i.latch

for.i.latch:
  %fmax.i.lcssa = phi float [ %fmax.j.next, %for.j ]
  %i.inc = add i32 %i, 1
  %cmp.i = icmp slt i32 %i.inc, 2
  br i1 %cmp.i, label %for.i.header, label %exit

exit:
  ret void
}

; Check that interchanging the loops is legal for the floating-point
; llvm.maximumnum.

; CHECK:      --- !Pass
; CHECK-NEXT: Pass:            loop-interchange
; CHECK-NEXT: Name:            Interchanged
; CHECK-NEXT: Function:        reduction_fmaxinumnum
define void @reduction_fmaxinumnum(ptr %A, float %init) {
entry:
  br label %for.i.header

for.i.header:
  %i = phi i32 [ 0, %entry ], [ %i.inc, %for.i.latch ]
  %fmax.i = phi float [ %init, %entry ], [ %fmax.i.lcssa, %for.i.latch ]
  br label %for.j

for.j:
  %j = phi i32 [ 0, %for.i.header ], [ %j.inc, %for.j ]
  %fmax.j = phi float [ %fmax.i, %for.i.header ], [ %fmax.j.next, %for.j ]
  %idx = getelementptr inbounds [2 x [2 x i32]], ptr %A, i32 0, i32 %j, i32 %i
  %a = load float, ptr %idx, align 4
  %fmax.j.next = call float @llvm.maximumnum.f32(float %a, float %fmax.j)
  %j.inc = add i32 %j, 1
  %cmp.j = icmp slt i32 %j.inc, 2
  br i1 %cmp.j, label %for.j, label %for.i.latch

for.i.latch:
  %fmax.i.lcssa = phi float [ %fmax.j.next, %for.j ]
  %i.inc = add i32 %i, 1
  %cmp.i = icmp slt i32 %i.inc, 2
  br i1 %cmp.i, label %for.i.header, label %exit

exit:
  ret void
}

declare float @llvm.fmuladd.f32(float %a, float %b, float %c)
declare float @llvm.minimumnum.f32(float %a, float %b)
declare float @llvm.maximumnum.f32(float %a, float %b)