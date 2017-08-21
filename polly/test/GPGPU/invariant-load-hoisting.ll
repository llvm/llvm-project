; RUN: opt %loadPolly -polly-scops -analyze \
; RUN: -polly-invariant-load-hoisting < %s | FileCheck %s -check-prefix=SCOP
;
; RUN: opt %loadPolly -polly-scops -S  -polly-invariant-load-hoisting \
; RUN: -polly-codegen-ppcg < %s | FileCheck %s -check-prefix=HOST-IR
;
; RUN: opt %loadPolly -polly-scops -analyze  -polly-invariant-load-hoisting \
; RUN: -polly-codegen-ppcg -polly-acc-dump-kernel-ir < %s | FileCheck %s -check-prefix=KERNEL-IR
;
; REQUIRES: pollyacc
;
; SCOP:       Function: f
; SCOP-NEXT:  Region: %entry.split---%for.end26
; SCOP-NEXT:  Max Loop Depth:  3
; SCOP-NEXT:  Invariant Accesses: {
; SCOP-NEXT:          ReadAccess :=	[Reduction Type: NONE] [Scalar: 0]
; SCOP-NEXT:              [n, tmp12] -> { Stmt_for_body6[i0, i1, i2] -> MemRef_invariant[0] };
; SCOP-NEXT:          Execution Context: [n, tmp12] -> {  : n > 0 }
; SCOP-NEXT:  }
; HOST-IR:      call void @polly_launchKernel(i8* %209, i32 %215, i32 1, i32 32, i32 1, i32 1, i8* %polly_launch_0_params_i8ptr)
; HOST-IR-NEXT: call void @polly_freeKernel(i8* %209)

; KERNEL-IR: define ptx_kernel void @FUNC_f_SCOP_0_KERNEL_0(i8 addrspace(1)* %MemRef_B, i8 addrspace(1)* %MemRef_A, i32 %n, i32 %tmp12, i32 %polly.preload.tmp21.merge)


; Check that we generate correct GPU code in case of invariant load hoisting.
;
;
;    static const int N = 3000;
;
;    void f(int A[N][N], int *invariant, int B[N][N], int n) {
;      for (int i = 0; i < n; i++) {
;        for (int j = 0; j < n; j++) {
;          for (int k = 0; k < n; k++) {
;
;            A[*invariant][k] = B[k][k];
;            A[k][*invariant] += B[k][k];
;          }
;        }
;      }
;    }
;

define void @f([3000 x i32]* %A, i32* %invariant, [3000 x i32]* %B, i32 %n) {
entry:
  br label %entry.split

entry.split:                                      ; preds = %entry
  %cmp6 = icmp sgt i32 %n, 0
  br i1 %cmp6, label %for.cond1.preheader.lr.ph, label %for.end26

for.cond1.preheader.lr.ph:                        ; preds = %entry.split
  br label %for.cond1.preheader

for.cond1.preheader:                              ; preds = %for.cond1.preheader.lr.ph, %for.inc24
  %i.07 = phi i32 [ 0, %for.cond1.preheader.lr.ph ], [ %inc25, %for.inc24 ]
  %cmp23 = icmp sgt i32 %n, 0
  br i1 %cmp23, label %for.cond4.preheader.lr.ph, label %for.inc24

for.cond4.preheader.lr.ph:                        ; preds = %for.cond1.preheader
  br label %for.cond4.preheader

for.cond4.preheader:                              ; preds = %for.cond4.preheader.lr.ph, %for.inc21
  %j.04 = phi i32 [ 0, %for.cond4.preheader.lr.ph ], [ %inc22, %for.inc21 ]
  %cmp51 = icmp sgt i32 %n, 0
  br i1 %cmp51, label %for.body6.lr.ph, label %for.inc21

for.body6.lr.ph:                                  ; preds = %for.cond4.preheader
  br label %for.body6

for.body6:                                        ; preds = %for.body6.lr.ph, %for.body6
  %k.02 = phi i32 [ 0, %for.body6.lr.ph ], [ %inc, %for.body6 ]
  %idxprom = sext i32 %k.02 to i64
  %idxprom7 = sext i32 %k.02 to i64
  %arrayidx8 = getelementptr inbounds [3000 x i32], [3000 x i32]* %B, i64 %idxprom, i64 %idxprom7
  %tmp9 = load i32, i32* %arrayidx8, align 4
  %tmp12 = load i32, i32* %invariant, align 4
  %idxprom9 = sext i32 %tmp12 to i64
  %idxprom11 = sext i32 %k.02 to i64
  %arrayidx12 = getelementptr inbounds [3000 x i32], [3000 x i32]* %A, i64 %idxprom9, i64 %idxprom11
  store i32 %tmp9, i32* %arrayidx12, align 4
  %idxprom13 = sext i32 %k.02 to i64
  %idxprom15 = sext i32 %k.02 to i64
  %arrayidx16 = getelementptr inbounds [3000 x i32], [3000 x i32]* %B, i64 %idxprom13, i64 %idxprom15
  %tmp17 = load i32, i32* %arrayidx16, align 4
  %idxprom17 = sext i32 %k.02 to i64
  %tmp21 = load i32, i32* %invariant, align 4
  %idxprom19 = sext i32 %tmp21 to i64
  %arrayidx20 = getelementptr inbounds [3000 x i32], [3000 x i32]* %A, i64 %idxprom17, i64 %idxprom19
  %tmp22 = load i32, i32* %arrayidx20, align 4
  %add = add nsw i32 %tmp22, %tmp17
  store i32 %add, i32* %arrayidx20, align 4
  %inc = add nuw nsw i32 %k.02, 1
  %cmp5 = icmp slt i32 %inc, %n
  br i1 %cmp5, label %for.body6, label %for.cond4.for.inc21_crit_edge

for.cond4.for.inc21_crit_edge:                    ; preds = %for.body6
  br label %for.inc21

for.inc21:                                        ; preds = %for.cond4.for.inc21_crit_edge, %for.cond4.preheader
  %inc22 = add nuw nsw i32 %j.04, 1
  %cmp2 = icmp slt i32 %inc22, %n
  br i1 %cmp2, label %for.cond4.preheader, label %for.cond1.for.inc24_crit_edge

for.cond1.for.inc24_crit_edge:                    ; preds = %for.inc21
  br label %for.inc24

for.inc24:                                        ; preds = %for.cond1.for.inc24_crit_edge, %for.cond1.preheader
  %inc25 = add nuw nsw i32 %i.07, 1
  %cmp = icmp slt i32 %inc25, %n
  br i1 %cmp, label %for.cond1.preheader, label %for.cond.for.end26_crit_edge

for.cond.for.end26_crit_edge:                     ; preds = %for.inc24
  br label %for.end26

for.end26:                                        ; preds = %for.cond.for.end26_crit_edge, %entry.split
  ret void
}
