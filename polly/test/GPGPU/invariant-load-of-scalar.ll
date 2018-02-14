; RUN: opt %loadPolly -polly-scops -polly-invariant-load-hoisting \
; RUN: -analyze < %s | \
; RUN: FileCheck -check-prefix=SCOP %s

; RUN: opt %loadPolly -polly-codegen-ppcg -polly-invariant-load-hoisting \
; RUN: -S < %s | \
; RUN: FileCheck -check-prefix=HOST-IR %s


; RUN: opt %loadPolly -polly-codegen-ppcg -polly-invariant-load-hoisting \
; RUN: -disable-output -polly-acc-dump-kernel-ir < %s | \
; RUN: FileCheck -check-prefix=KERNEL-IR %s

; REQUIRES: pollyacc

; Check that we offload invariant loads of scalars correctly.

; Check that invariant loads are present.
; SCOP:      Function: checkPrivatization
; SCOP-NEXT: Region: %entry.split---%for.end
; SCOP-NEXT: Max Loop Depth:  1
; SCOP-NEXT: Invariant Accesses: {
; SCOP-NEXT:         ReadAccess :=	[Reduction Type: NONE] [Scalar: 0]
; SCOP-NEXT:             [tmp, tmp2] -> { Stmt_entry_split[] -> MemRef_begin[0] };
; SCOP-NEXT:         Execution Context: [tmp, tmp2] -> {  :  }
; SCOP-NEXT:         ReadAccess :=	[Reduction Type: NONE] [Scalar: 0]
; SCOP-NEXT:             [tmp, tmp2] -> { Stmt_for_body[i0] -> MemRef_end[0] };
; SCOP-NEXT:         Execution Context: [tmp, tmp2] -> {  :  }
; SCOP-NEXT: }
;

; Check that we do not actually allocate arrays for %begin, %end, since they are
; invariant load hoisted.
; HOST-IR: %p_dev_array_MemRef_A = call i8* @polly_allocateMemoryForDevice
; HOST-IR-NOT: call i8* @polly_allocateMemoryForDevice

; Check that we send the invariant loaded scalars as parameters to the
; kernel function.
; KERNEL-IR: define ptx_kernel void @FUNC_checkPrivatization_SCOP_0_KERNEL_0
; KERNEL-IR-SAME: (i8 addrspace(1)* %MemRef_A, i32 %tmp,
; KERNEL-IR-SAME: i32 %tmp2, i32 %polly.access.begin.load,
; KERNEL-IR-SAME: i32 %polly.access.end.load)


; void checkScalarPointerOffload(int A[], int *begin, int *end) {
;     for(int i = *begin; i < *end; i++) {
;         A[i] = 10;
;     }
; }

target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.12.0"

define void @checkPrivatization(i32* %A, i32* %begin, i32* %end) {
entry:
  br label %entry.split

entry.split:                                      ; preds = %entry
  %tmp = load i32, i32* %begin, align 4
  %tmp21 = load i32, i32* %end, align 4
  %cmp3 = icmp slt i32 %tmp, %tmp21
  br i1 %cmp3, label %for.body.lr.ph, label %for.end

for.body.lr.ph:                                   ; preds = %entry.split
  %tmp1 = sext i32 %tmp to i64
  br label %for.body

for.body:                                         ; preds = %for.body.lr.ph, %for.body
  %indvars.iv4 = phi i64 [ %tmp1, %for.body.lr.ph ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds i32, i32* %A, i64 %indvars.iv4
  store i32 10, i32* %arrayidx, align 4
  %indvars.iv.next = add i64 %indvars.iv4, 1
  %tmp2 = load i32, i32* %end, align 4
  %tmp3 = sext i32 %tmp2 to i64
  %cmp = icmp slt i64 %indvars.iv.next, %tmp3
  br i1 %cmp, label %for.body, label %for.cond.for.end_crit_edge

for.cond.for.end_crit_edge:                       ; preds = %for.body
  br label %for.end

for.end:                                          ; preds = %for.cond.for.end_crit_edge, %entry.split
  ret void
}

