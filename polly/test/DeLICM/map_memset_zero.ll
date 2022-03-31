; RUN: opt %loadPolly -polly-stmt-granularity=bb -polly-print-delicm -disable-output < %s | FileCheck -match-full-lines %s
; RUN: opt %loadNPMPolly -polly-stmt-granularity=bb "-passes=scop(print<polly-delicm>)" -disable-output < %s | FileCheck -match-full-lines %s
;
; Check that PHI mapping works even in presence of a memset whose'
; zero value is used.
;
declare void @llvm.memset.p0i8.i64(i8* nocapture, i8, i64, i32, i1)

define void @func(i8* noalias nonnull %A) {
entry:
  br label %outer.for

outer.for:
  %j = phi i32 [0, %entry], [%j.inc, %outer.inc]
  %j.cmp = icmp slt i32 %j, 2
  br i1 %j.cmp, label %bodyA, label %outer.exit


    bodyA:
      %A_idx = getelementptr inbounds i8, i8* %A, i32 %j
      %cond = icmp eq i32 21, 21
      br i1 %cond, label %bodyB, label %bodyC

    bodyB:
      call void @llvm.memset.p0i8.i64(i8* %A_idx, i8 0, i64 1, i32 1, i1 false)
      br label %bodyC

    bodyC:
      %phi = phi i8 [1, %bodyA], [0, %bodyB]
      %a = load i8, i8* %A_idx
      store i8 %phi, i8* %A_idx
      br label %outer.inc


outer.inc:
  %j.inc = add nuw nsw i32 %j, 1
  br label %outer.for

outer.exit:
  br label %return

return:
  ret void
}


; CHECK: Statistics {
; CHECK:     Compatible overwrites: 1
; CHECK:     Overwrites mapped to:  1
; CHECK:     PHI scalars mapped:    1
; CHECK: }

; CHECK:      After accesses {
; CHECK-NEXT:     Stmt_bodyA
; CHECK-NEXT:             MustWriteAccess :=  [Reduction Type: NONE] [Scalar: 1]
; CHECK-NEXT:                 { Stmt_bodyA[i0] -> MemRef_phi__phi[] };
; CHECK-NEXT:            new: { Stmt_bodyA[i0] -> MemRef_A[o0] : false };
; CHECK-NEXT:     Stmt_bodyB
; CHECK-NEXT:             MustWriteAccess :=  [Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:                 { Stmt_bodyB[i0] -> MemRef_A[i0] };
; CHECK-NEXT:             MustWriteAccess :=  [Reduction Type: NONE] [Scalar: 1]
; CHECK-NEXT:                 { Stmt_bodyB[i0] -> MemRef_phi__phi[] };
; CHECK-NEXT:            new: { Stmt_bodyB[i0] -> MemRef_A[i0] };
; CHECK-NEXT:     Stmt_bodyC
; CHECK-NEXT:             ReadAccess :=       [Reduction Type: NONE] [Scalar: 1]
; CHECK-NEXT:                 { Stmt_bodyC[i0] -> MemRef_phi__phi[] };
; CHECK-NEXT:            new: { Stmt_bodyC[i0] -> MemRef_A[i0] };
; CHECK-NEXT:             ReadAccess :=       [Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:                 { Stmt_bodyC[i0] -> MemRef_A[i0] };
; CHECK-NEXT:             MustWriteAccess :=  [Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:                 { Stmt_bodyC[i0] -> MemRef_A[i0] };
; CHECK-NEXT: }
