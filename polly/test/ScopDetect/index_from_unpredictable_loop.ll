; RUN: opt %loadPolly                        -polly-print-scops -disable-output < %s | FileCheck %s --check-prefix=AFFINE
; RUN: opt %loadPolly -polly-allow-nonaffine -polly-print-scops -disable-output < %s | FileCheck %s --check-prefix=NONAFFINE

; The SCoP contains a loop with multiple exit blocks (BBs after leaving
; the loop). The current implementation of deriving their domain derives
; only a common domain for all of the exit blocks. We disabled loops with
; multiple exit blocks until this is fixed.
; XFAIL: *

; The loop for.body => for.inc has an unpredictable iteration count could due to
; the undef start value that it is compared to. Therefore the array element
; %arrayidx101 that depends on that exit value cannot be affine.
; Derived from test-suite/MultiSource/Benchmarks/BitBench/uuencode/uuencode.c

define void @encode_line(i8* nocapture readonly %input, i32 %octets, i64 %p) {
entry:
  br i1 undef, label %for.body, label %for.end

for.body:
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.inc ], [ %p, %entry ]
  %octets.addr.02 = phi i32 [ undef, %for.inc ], [ %octets, %entry ]
  br i1 false, label %for.inc, label %if.else

if.else:
  %cond = icmp eq i32 %octets.addr.02, 2
  br i1 %cond, label %if.then84, label %for.end

if.then84:
  %0 = add nsw i64 %indvars.iv, 1
  %arrayidx101 = getelementptr inbounds i8, i8* %input, i64 %0
  store i8 42, i8* %arrayidx101, align 1
  br label %for.end

for.inc:
  %cmp = icmp sgt i32 %octets.addr.02, 3
  %indvars.iv.next = add nsw i64 %indvars.iv, 3
  br i1 %cmp, label %for.body, label %for.end

for.end:
  ret void
}

; AFFINE:       Region: %if.else---%for.end

; AFFINE:       Statements {
; AFFINE-NEXT:  	Stmt_if_then84
; AFFINE-NEXT:          Domain :=
; AFFINE-NEXT:              [octets, p_1, p] -> { Stmt_if_then84[] : octets = 2 };
; AFFINE-NEXT:          Schedule :=
; AFFINE-NEXT:              [octets, p_1, p] -> { Stmt_if_then84[] -> [] };
; AFFINE-NEXT:          MustWriteAccess :=	[Reduction Type: NONE] [Scalar: 0]
; AFFINE-NEXT:              [octets, p_1, p] -> { Stmt_if_then84[] -> MemRef_input[1 + p] };
; AFFINE-NEXT:  }

; NONAFFINE:      Region: %for.body---%for.end

; NONAFFINE:      Statements {
; NONAFFINE-NEXT: 	Stmt_for_body
; NONAFFINE-NEXT:         Domain :=
; NONAFFINE-NEXT:             [octets] -> { Stmt_for_body[0] };
; NONAFFINE-NEXT:         Schedule :=
; NONAFFINE-NEXT:             [octets] -> { Stmt_for_body[i0] -> [0, 0] };
; NONAFFINE-NEXT:         MustWriteAccess :=	[Reduction Type: NONE] [Scalar: 1]
; NONAFFINE-NEXT:             [octets] -> { Stmt_for_body[i0] -> MemRef_indvars_iv[] };
; NONAFFINE-NEXT: 	Stmt_if_then84
; NONAFFINE-NEXT:         Domain :=
; NONAFFINE-NEXT:             [octets] -> { Stmt_if_then84[] : octets = 2 };
; NONAFFINE-NEXT:         Schedule :=
; NONAFFINE-NEXT:             [octets] -> { Stmt_if_then84[] -> [1, 0] };
; NONAFFINE-NEXT:         ReadAccess :=	[Reduction Type: NONE] [Scalar: 1]
; NONAFFINE-NEXT:             [octets] -> { Stmt_if_then84[] -> MemRef_indvars_iv[] };
; NONAFFINE-NEXT:         MayWriteAccess :=	[Reduction Type: NONE] [Scalar: 0]
; NONAFFINE-NEXT:             [octets] -> { Stmt_if_then84[] -> MemRef_input[o0] };
; NONAFFINE-NEXT: }
