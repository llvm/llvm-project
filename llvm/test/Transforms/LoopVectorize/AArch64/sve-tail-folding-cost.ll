; REQUIRES: asserts
; RUN: opt -S -passes=loop-vectorize -prefer-predicate-over-epilogue=predicate-else-scalar-epilogue \
; RUN:   -debug-only=loop-vectorize < %s 2>%t | FileCheck %s
; RUN: cat %t | FileCheck --check-prefix=COST %s

target triple = "aarch64-unknown-linux-gnu"

; The uniform load of %d in the following loop triggers the special
; branch costing code in LoopVectorizationCostModel::getInstructionCost.
; However, this should only affect the fixed-width cost because for
; NEON it needs to scalarize the load, whereas for SVE it can use a predicated load.
; Because of how the LoopVectorizer annotates the load to need scalarization with
; predicated blocks, this leads to different costs for the branch instruction.
;
; NOTE: This test assumes we will never use a fixed-width VF due to
; the high cost of scalarizing the masked store, however this assumption may
; break in future if we permit the use of SVE loads and stores to perform the
; fixed-width operations.
define i32 @uniform_load(i64 %n, ptr readnone %c, ptr %d) #0 {
; CHECK-LABEL: @uniform_load(
; CHECK: call void @llvm.masked.store.nxv4f32.p0(<vscale x 4 x float>
entry:
  br label %for.body

for.body:                                         ; preds = %entry, %for.body
  %indvars.iv = phi i64 [ 1, %entry ], [ %indvars.iv.next, %for.body ]
  %load2 = load float, ptr %d, align 4
  %arrayidx2 = getelementptr inbounds float, ptr %c, i64 %indvars.iv
  store float %load2, ptr %arrayidx2, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, %n
  br i1 %exitcond.not, label %for.end, label %for.body

for.end:                                          ; preds = %for.body
  ret i32 0
}

; COST: LV: Checking a loop in 'simple_memset'
; COST: Cost of 4 for VF 2: EMIT{{.*}}active lane mask
; COST: Cost of 8 for VF 4: EMIT{{.*}}active lane mask
; COST: Cost of Invalid for VF vscale x 1: EMIT{{.*}}active lane mask
; COST: Cost of 1 for VF vscale x 2: EMIT{{.*}}active lane mask
; COST: Cost of 1 for VF vscale x 4: EMIT{{.*}}active lane mask

define void @simple_memset(i32 %val, ptr %ptr, i64 %n) #0 {
; CHECK-LABEL: @simple_memset(
; CHECK: call void @llvm.masked.store.nxv4i32.p0(<vscale x 4 x i32>
entry:
  br label %while.body

while.body:                                       ; preds = %while.body, %entry
  %index = phi i64 [ %index.next, %while.body ], [ 0, %entry ]
  %gep = getelementptr i32, ptr %ptr, i64 %index
  store i32 %val, ptr %gep
  %index.next = add nsw i64 %index, 1
  %cmp10 = icmp ult i64 %index.next, %n
  br i1 %cmp10, label %while.body, label %while.end.loopexit

while.end.loopexit:                               ; preds = %while.body
  ret void
}

attributes #0 = { vscale_range(1,16) "target-features"="+sve" }
