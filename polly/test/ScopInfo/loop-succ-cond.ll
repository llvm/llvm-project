; RUN: opt %loadPolly -polly-scops -analyze < %s | FileCheck %s
; RUN: opt %loadPolly -polly-codegen -S < %s

; Bugpoint-reduced from
; test-suite/SingleSource/Benchmarks/Adobe-C++/loop_unroll.cpp
;
; Check that the loop %loop_start does not narrow the domain (the domain when
; entering and leaving the loop must be identical)
;
; What happened in detail:
;
; 1) if.end5 checks %count whether there will be at least one iteration of
;    loop_start. The domain condition added to loop_start therefore is
;      [count] -> { [] : count > 0 }
; 2) The loop exit condition of loop_start tests whether
;    %indvars.iv.next64 == %0 (which is zext i32 %count to i64, NOT %count
;    itself). %0 and %count have to be treated as independent parameters. The
;    loop exit condition is
;      [p_0] -> { [i0] : i0 = p_0 - 1 }
; 3) Normalized loop induction variables are always non-negative. The domain
;    condition for this is loop
;      { [i0] : i0 >= 0 }
; 4) The intersection of all three sets (condition of executing/entering loop,
;    non-negative induction variables, loop exit condition) is
;      [count, p_0] -> { [i0] : count > 0 and i0 >= 0 and i0 = p_0 - 1 }
; 5) from which ISL can derive
;     [count, p_0] -> { [i0] : p_0 > 0 }
; 6) if.end5 is either executed when skipping the loop
;    (domain [count] -> { [] : count <= 0 })
;    or though the loop.
; 7) Assuming the loop is guaranteed to exit, Polly computes the after-the-loop
;    domain by taking the loop exit condition and projecting-out the induction
;    variable. This yields
;       [count, p_0] -> { [] : count > 0 and p_0 > 0 }
; 8) The disjunction of both cases, 6) and 7)
;    (the two incoming edges of if.end12) is
;      [count, p_0] -> { [] : count <= 0 or (count > 0 and p_0 > 0) }
; 9) Notice that if.end12 is logically _always_ executed in every scop
;    execution. Both cases of if.end5 will eventually land in if.end12

define void @func(i32 %count, float* %A) {
entry:
  %0 = zext i32 %count to i64
  br i1 undef, label %if.end5.preheader, label %for.end

if.end5.preheader:
  %cmp6 = icmp sgt i32 %count, 0
  br label %if.end5

if.end5:
  br i1 %cmp6, label %loop_start, label %if.end12

loop_start:
  %indvars.iv63 = phi i64 [ %indvars.iv.next64, %loop_start ], [ 0, %if.end5 ]
  %add8 = add i32 undef, undef
  %indvars.iv.next64 = add nuw nsw i64 %indvars.iv63, 1
  %cmp9 = icmp eq i64 %indvars.iv.next64, %0
  br i1 %cmp9, label %if.end12, label %loop_start

if.end12:
  store float 0.0, float* %A
  br label %for.end

for.end:
  ret void
}


; CHECK:      Statements {
; CHECK-NEXT:     Stmt_if_end12
; CHECK-NEXT:         Domain :=
; CHECK-NEXT:             [count, p_1] -> { Stmt_if_end12[] };
; CHECK-NEXT:         Schedule :=
; CHECK-NEXT:             [count, p_1] -> { Stmt_if_end12[] -> [] };
; CHECK-NEXT:         MustWriteAccess :=    [Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:             [count, p_1] -> { Stmt_if_end12[] -> MemRef_A[0] };
; CHECK-NEXT: }
