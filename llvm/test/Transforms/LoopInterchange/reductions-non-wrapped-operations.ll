; RUN: opt < %s -passes=loop-interchange -cache-line-size=64 -pass-remarks-output=%t -disable-output \
; RUN:     -verify-dom-info -verify-loop-info -verify-loop-lcssa
; RUN: FileCheck -input-file=%t %s

; Check that interchanging the loops is legal for the bitwise-or reduction.
;
; int b_or = 0;
; for (int i = 0; i < 2; i++)
;   for (int j = 0; j < 2; j++)
;     b_or |= A[j][i];

; CHECK:      --- !Pass
; CHECK-NEXT: Pass:            loop-interchange
; CHECK-NEXT: Name:            Interchanged
; CHECK-NEXT: Function:        reduction_or
define void @reduction_or(ptr %A) {
entry:
  br label %for.i.header

for.i.header:
  %i = phi i32 [ 0, %entry ], [ %i.inc, %for.i.latch ]
  %or.i = phi i32 [ 0, %entry ], [ %or.i.lcssa, %for.i.latch ]
  br label %for.j

for.j:
  %j = phi i32 [ 0, %for.i.header ], [ %j.inc, %for.j ]
  %or.j = phi i32 [ %or.i, %for.i.header ], [ %or.j.next, %for.j ]
  %idx = getelementptr inbounds [2 x [2 x i32]], ptr %A, i32 0, i32 %j, i32 %i
  %a = load i32, ptr %idx, align 4
  %or.j.next = or i32 %or.j, %a
  %j.inc = add i32 %j, 1
  %cmp.j = icmp slt i32 %j.inc, 2
  br i1 %cmp.j, label %for.j, label %for.i.latch

for.i.latch:
  %or.i.lcssa = phi i32 [ %or.j.next, %for.j ]
  %i.inc = add i32 %i, 1
  %cmp.i = icmp slt i32 %i.inc, 2
  br i1 %cmp.i, label %for.i.header, label %exit

exit:
  ret void
}


; Check that interchanging the loops is legal for the bitwise-and reduction.
;
; int b_and = -1;
; for (int i = 0; i < 2; i++)
;   for (int j = 0; j < 2; j++)
;     b_and &= A[j][i];

; CHECK:      --- !Pass
; CHECK-NEXT: Pass:            loop-interchange
; CHECK-NEXT: Name:            Interchanged
; CHECK-NEXT: Function:        reduction_and
define void @reduction_and(ptr %A) {
entry:
  br label %for.i.header

for.i.header:
  %i = phi i32 [ 0, %entry ], [ %i.inc, %for.i.latch ]
  %and.i = phi i32 [ -1, %entry ], [ %and.i.lcssa, %for.i.latch ]
  br label %for.j

for.j:
  %j = phi i32 [ 0, %for.i.header ], [ %j.inc, %for.j ]
  %and.j = phi i32 [ %and.i, %for.i.header ], [ %and.j.next, %for.j ]
  %idx = getelementptr inbounds [2 x [2 x i32]], ptr %A, i32 0, i32 %j, i32 %i
  %a = load i32, ptr %idx, align 4
  %and.j.next = and i32 %and.j, %a
  %j.inc = add i32 %j, 1
  %cmp.j = icmp slt i32 %j.inc, 2
  br i1 %cmp.j, label %for.j, label %for.i.latch

for.i.latch:
  %and.i.lcssa = phi i32 [ %and.j.next, %for.j ]
  %i.inc = add i32 %i, 1
  %cmp.i = icmp slt i32 %i.inc, 2
  br i1 %cmp.i, label %for.i.header, label %exit

exit:
  ret void
}


; Check that interchanging the loops is legal for the bitwise-xor reduction.
;
; int b_xor = 0;
; for (int i = 0; i < 2; i++)
;   for (int j = 0; j < 2; j++)
;     b_xor ^= A[j][i];

; CHECK:      --- !Pass
; CHECK-NEXT: Pass:            loop-interchange
; CHECK-NEXT: Name:            Interchanged
; CHECK-NEXT: Function:        reduction_xor
define void @reduction_xor(ptr %A) {
entry:
  br label %for.i.header

for.i.header:
  %i = phi i32 [ 0, %entry ], [ %i.inc, %for.i.latch ]
  %xor.i = phi i32 [ 0, %entry ], [ %xor.i.lcssa, %for.i.latch ]
  br label %for.j

for.j:
  %j = phi i32 [ 0, %for.i.header ], [ %j.inc, %for.j ]
  %xor.j = phi i32 [ %xor.i, %for.i.header ], [ %xor.j.next, %for.j ]
  %idx = getelementptr inbounds [2 x [2 x i32]], ptr %A, i32 0, i32 %j, i32 %i
  %a = load i32, ptr %idx, align 4
  %xor.j.next = xor i32 %xor.j, %a
  %j.inc = add i32 %j, 1
  %cmp.j = icmp slt i32 %j.inc, 2
  br i1 %cmp.j, label %for.j, label %for.i.latch

for.i.latch:
  %xor.i.lcssa = phi i32 [ %xor.j.next, %for.j ]
  %i.inc = add i32 %i, 1
  %cmp.i = icmp slt i32 %i.inc, 2
  br i1 %cmp.i, label %for.i.header, label %exit

exit:
  ret void
}


; Check that interchanging the loops is legal for the signed-minimum reduction.
;
; int smin = init;
; for (int i = 0; i < 2; i++)
;   for (int j = 0; j < 2; j++)
;     smin = (A[j][i] < smin) ? A[j][i] : smin;

; CHECK:      --- !Pass
; CHECK-NEXT: Pass:            loop-interchange
; CHECK-NEXT: Name:            Interchanged
; CHECK-NEXT: Function:        reduction_smin
define void @reduction_smin(ptr %A, i32 %init) {
entry:
  br label %for.i.header

for.i.header:
  %i = phi i32 [ 0, %entry ], [ %i.inc, %for.i.latch ]
  %smin.i = phi i32 [ %init, %entry ], [ %smin.i.lcssa, %for.i.latch ]
  br label %for.j

for.j:
  %j = phi i32 [ 0, %for.i.header ], [ %j.inc, %for.j ]
  %smin.j = phi i32 [ %smin.i, %for.i.header ], [ %smin.j.next, %for.j ]
  %idx = getelementptr inbounds [2 x [2 x i32]], ptr %A, i32 0, i32 %j, i32 %i
  %a = load i32, ptr %idx, align 4
  %cmp = icmp slt i32 %a, %smin.j
  %smin.j.next = select i1 %cmp, i32 %a, i32 %smin.j
  %j.inc = add i32 %j, 1
  %cmp.j = icmp slt i32 %j.inc, 2
  br i1 %cmp.j, label %for.j, label %for.i.latch

for.i.latch:
  %smin.i.lcssa = phi i32 [ %smin.j.next, %for.j ]
  %i.inc = add i32 %i, 1
  %cmp.i = icmp slt i32 %i.inc, 2
  br i1 %cmp.i, label %for.i.header, label %exit

exit:
  ret void
}


; Check that interchanging the loops is legal for the signed-maximum reduction.
;
; int smax = init;
; for (int i = 0; i < 2; i++)
;   for (int j = 0; j < 2; j++)
;     smax = (A[j][i] > smax) ? A[j][i] : smax;

; CHECK:      --- !Pass
; CHECK-NEXT: Pass:            loop-interchange
; CHECK-NEXT: Name:            Interchanged
; CHECK-NEXT: Function:        reduction_smax
define void @reduction_smax(ptr %A, i32 %init) {
entry:
  br label %for.i.header

for.i.header:
  %i = phi i32 [ 0, %entry ], [ %i.inc, %for.i.latch ]
  %smax.i = phi i32 [ %init, %entry ], [ %smax.i.lcssa, %for.i.latch ]
  br label %for.j

for.j:
  %j = phi i32 [ 0, %for.i.header ], [ %j.inc, %for.j ]
  %smax.j = phi i32 [ %smax.i, %for.i.header ], [ %smax.j.next, %for.j ]
  %idx = getelementptr inbounds [2 x [2 x i32]], ptr %A, i32 0, i32 %j, i32 %i
  %a = load i32, ptr %idx, align 4
  %cmp = icmp sgt i32 %a, %smax.j
  %smax.j.next = select i1 %cmp, i32 %a, i32 %smax.j
  %j.inc = add i32 %j, 1
  %cmp.j = icmp slt i32 %j.inc, 2
  br i1 %cmp.j, label %for.j, label %for.i.latch

for.i.latch:
  %smax.i.lcssa = phi i32 [ %smax.j.next, %for.j ]
  %i.inc = add i32 %i, 1
  %cmp.i = icmp slt i32 %i.inc, 2
  br i1 %cmp.i, label %for.i.header, label %exit

exit:
  ret void
}


; Check that interchanging the loops is legal for the unsigned-minimum reduction.
;
; unsigned umin = init;
; for (int i = 0; i < 2; i++)
;   for (int j = 0; j < 2; j++)
;     umin = (A[j][i] < umin) ? A[j][i] : umin;

; CHECK:      --- !Pass
; CHECK-NEXT: Pass:            loop-interchange
; CHECK-NEXT: Name:            Interchanged
; CHECK-NEXT: Function:        reduction_umin
define void @reduction_umin(ptr %A, i32 %init) {
entry:
  br label %for.i.header

for.i.header:
  %i = phi i32 [ 0, %entry ], [ %i.inc, %for.i.latch ]
  %umin.i = phi i32 [ %init, %entry ], [ %umin.i.lcssa, %for.i.latch ]
  br label %for.j

for.j:
  %j = phi i32 [ 0, %for.i.header ], [ %j.inc, %for.j ]
  %umin.j = phi i32 [ %umin.i, %for.i.header ], [ %umin.j.next, %for.j ]
  %idx = getelementptr inbounds [2 x [2 x i32]], ptr %A, i32 0, i32 %j, i32 %i
  %a = load i32, ptr %idx, align 4
  %cmp = icmp ult i32 %a, %umin.j
  %umin.j.next = select i1 %cmp, i32 %a, i32 %umin.j
  %j.inc = add i32 %j, 1
  %cmp.j = icmp slt i32 %j.inc, 2
  br i1 %cmp.j, label %for.j, label %for.i.latch

for.i.latch:
  %umin.i.lcssa = phi i32 [ %umin.j.next, %for.j ]
  %i.inc = add i32 %i, 1
  %cmp.i = icmp slt i32 %i.inc, 2
  br i1 %cmp.i, label %for.i.header, label %exit

exit:
  ret void
}


; Check that interchanging the loops is legal for the unsigned-maximum reduction.
;
; unsigned umax = 0;
; for (int i = 0; i < 2; i++)
;   for (int j = 0; j < 2; j++)
;     smax = (A[j][i] > smax) ? A[j][i] : smax;

; CHECK:      --- !Pass
; CHECK-NEXT: Pass:            loop-interchange
; CHECK-NEXT: Name:            Interchanged
; CHECK-NEXT: Function:        reduction_umax
define void @reduction_umax(ptr %A) {
entry:
  br label %for.i.header

for.i.header:
  %i = phi i32 [ 0, %entry ], [ %i.inc, %for.i.latch ]
  %umax.i = phi i32 [ 0, %entry ], [ %umax.i.lcssa, %for.i.latch ]
  br label %for.j

for.j:
  %j = phi i32 [ 0, %for.i.header ], [ %j.inc, %for.j ]
  %umax.j = phi i32 [ %umax.i, %for.i.header ], [ %umax.j.next, %for.j ]
  %idx = getelementptr inbounds [2 x [2 x i32]], ptr %A, i32 0, i32 %j, i32 %i
  %a = load i32, ptr %idx, align 4
  %cmp = icmp ugt i32 %a, %umax.j
  %umax.j.next = select i1 %cmp, i32 %a, i32 %umax.j
  %j.inc = add i32 %j, 1
  %cmp.j = icmp slt i32 %j.inc, 2
  br i1 %cmp.j, label %for.j, label %for.i.latch

for.i.latch:
  %umax.i.lcssa = phi i32 [ %umax.j.next, %for.j ]
  %i.inc = add i32 %i, 1
  %cmp.i = icmp slt i32 %i.inc, 2
  br i1 %cmp.i, label %for.i.header, label %exit

exit:
  ret void
}


; Check that interchanging the loops is legal for the any-of reduction.
;
; int any_of = 0;
; for (int i = 0; i < 2; i++)
;   for (int j = 0; j < 2; j++)
;     any_of = (A[j][i] == 42) ? 1 : any_of;

; CHECK:      --- !Pass
; CHECK-NEXT: Pass:            loop-interchange
; CHECK-NEXT: Name:            Interchanged
; CHECK-NEXT: Function:        reduction_anyof
define void @reduction_anyof(ptr %A) {
entry:
  br label %for.i.header

for.i.header:
  %i = phi i32 [ 0, %entry ], [ %i.inc, %for.i.latch ]
  %anyof.i = phi i32 [ 0, %entry ], [ %anyof.i.lcssa, %for.i.latch ]
  br label %for.j

for.j:
  %j = phi i32 [ 0, %for.i.header ], [ %j.inc, %for.j ]
  %anyof.j = phi i32 [ %anyof.i, %for.i.header ], [ %anyof.j.next, %for.j ]
  %idx = getelementptr inbounds [2 x [2 x i32]], ptr %A, i32 0, i32 %j, i32 %i
  %a = load i32, ptr %idx, align 4
  %cmp = icmp eq i32 %a, 42
  %anyof.j.next = select i1 %cmp, i32 1, i32 %anyof.j
  %j.inc = add i32 %j, 1
  %cmp.j = icmp slt i32 %j.inc, 2
  br i1 %cmp.j, label %for.j, label %for.i.latch

for.i.latch:
  %anyof.i.lcssa = phi i32 [ %anyof.j.next, %for.j ]
  %i.inc = add i32 %i, 1
  %cmp.i = icmp slt i32 %i.inc, 2
  br i1 %cmp.i, label %for.i.header, label %exit

exit:
  ret void
}

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
