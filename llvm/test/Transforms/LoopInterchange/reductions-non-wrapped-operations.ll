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
