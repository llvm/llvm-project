; RUN: opt < %s -passes=loop-vectorize -scalable-vectorization=on \
; RUN:   -riscv-v-vector-bits-min=128 -riscv-v-vector-bits-max=128 \
; RUN:   -pass-remarks=loop-vectorize -pass-remarks-analysis=loop-vectorize \
; RUN:   -pass-remarks-missed=loop-vectorize -mtriple riscv64-linux-gnu \
; RUN:   -force-target-max-vector-interleave=2 -mattr=+v,+f -S 2>%t \
; RUN:   | FileCheck %s -check-prefix=CHECK
; RUN: cat %t | FileCheck %s -check-prefix=CHECK-REMARK

; Reduction can be vectorized

; ADD

; CHECK-REMARK: vectorized loop (vectorization width: vscale x 8, interleaved count: 2)
define i32 @add(ptr nocapture %a, ptr nocapture readonly %b, i64 %n) {
; CHECK-LABEL: @add
; CHECK: vector.body:
; CHECK: %[[LOAD1:.*]] = load <vscale x 8 x i32>
; CHECK: %[[LOAD2:.*]] = load <vscale x 8 x i32>
; CHECK: %[[ADD1:.*]] = add <vscale x 8 x i32> %[[LOAD1]]
; CHECK: %[[ADD2:.*]] = add <vscale x 8 x i32> %[[LOAD2]]
; CHECK: middle.block:
; CHECK: %[[ADD:.*]] = add <vscale x 8 x i32> %[[ADD2]], %[[ADD1]]
; CHECK-NEXT: call i32 @llvm.vector.reduce.add.nxv8i32(<vscale x 8 x i32> %[[ADD]])
entry:
  br label %for.body

for.body:                                         ; preds = %entry, %for.body
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %sum.07 = phi i32 [ 2, %entry ], [ %add, %for.body ]
  %arrayidx = getelementptr inbounds i32, ptr %a, i64 %iv
  %0 = load i32, ptr %arrayidx, align 4
  %add = add nsw i32 %0, %sum.07
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond.not = icmp eq i64 %iv.next, %n
  br i1 %exitcond.not, label %for.end, label %for.body, !llvm.loop !0

for.end:                                 ; preds = %for.body, %entry
  ret i32 %add
}

; OR

; CHECK-REMARK: vectorized loop (vectorization width: vscale x 8, interleaved count: 2)
define i32 @or(ptr nocapture %a, ptr nocapture readonly %b, i64 %n) {
; CHECK-LABEL: @or
; CHECK: vector.body:
; CHECK: %[[LOAD1:.*]] = load <vscale x 8 x i32>
; CHECK: %[[LOAD2:.*]] = load <vscale x 8 x i32>
; CHECK: %[[OR1:.*]] = or <vscale x 8 x i32> %[[LOAD1]]
; CHECK: %[[OR2:.*]] = or <vscale x 8 x i32> %[[LOAD2]]
; CHECK: middle.block:
; CHECK: %[[OR:.*]] = or <vscale x 8 x i32> %[[OR2]], %[[OR1]]
; CHECK-NEXT: call i32 @llvm.vector.reduce.or.nxv8i32(<vscale x 8 x i32> %[[OR]])
entry:
  br label %for.body

for.body:                                         ; preds = %entry, %for.body
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %sum.07 = phi i32 [ 2, %entry ], [ %or, %for.body ]
  %arrayidx = getelementptr inbounds i32, ptr %a, i64 %iv
  %0 = load i32, ptr %arrayidx, align 4
  %or = or i32 %0, %sum.07
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond.not = icmp eq i64 %iv.next, %n
  br i1 %exitcond.not, label %for.end, label %for.body, !llvm.loop !0

for.end:                                 ; preds = %for.body, %entry
  ret i32 %or
}

; AND

; CHECK-REMARK: vectorized loop (vectorization width: vscale x 8, interleaved count: 2)
define i32 @and(ptr nocapture %a, ptr nocapture readonly %b, i64 %n) {
; CHECK-LABEL: @and
; CHECK: vector.body:
; CHECK: %[[LOAD1:.*]] = load <vscale x 8 x i32>
; CHECK: %[[LOAD2:.*]] = load <vscale x 8 x i32>
; CHECK: %[[AND1:.*]] = and <vscale x 8 x i32> %[[LOAD1]]
; CHECK: %[[AND2:.*]] = and <vscale x 8 x i32> %[[LOAD2]]
; CHECK: middle.block:
; CHECK: %[[ABD:.*]] = and <vscale x 8 x i32> %[[ADD2]], %[[AND1]]
; CHECK-NEXT: call i32 @llvm.vector.reduce.and.nxv8i32(<vscale x 8 x i32> %[[ADD]])
entry:
  br label %for.body

for.body:                                         ; preds = %entry, %for.body
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %sum.07 = phi i32 [ 2, %entry ], [ %and, %for.body ]
  %arrayidx = getelementptr inbounds i32, ptr %a, i64 %iv
  %0 = load i32, ptr %arrayidx, align 4
  %and = and i32 %0, %sum.07
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond.not = icmp eq i64 %iv.next, %n
  br i1 %exitcond.not, label %for.end, label %for.body, !llvm.loop !0

for.end:                                 ; preds = %for.body, %entry
  ret i32 %and
}

; XOR

; CHECK-REMARK: vectorized loop (vectorization width: vscale x 8, interleaved count: 2)
define i32 @xor(ptr nocapture %a, ptr nocapture readonly %b, i64 %n) {
; CHECK-LABEL: @xor
; CHECK: vector.body:
; CHECK: %[[LOAD1:.*]] = load <vscale x 8 x i32>
; CHECK: %[[LOAD2:.*]] = load <vscale x 8 x i32>
; CHECK: %[[XOR1:.*]] = xor <vscale x 8 x i32> %[[LOAD1]]
; CHECK: %[[XOR2:.*]] = xor <vscale x 8 x i32> %[[LOAD2]]
; CHECK: middle.block:
; CHECK: %[[XOR:.*]] = xor <vscale x 8 x i32> %[[XOR2]], %[[XOR1]]
; CHECK-NEXT: call i32 @llvm.vector.reduce.xor.nxv8i32(<vscale x 8 x i32> %[[XOR]])
entry:
  br label %for.body

for.body:                                         ; preds = %entry, %for.body
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %sum.07 = phi i32 [ 2, %entry ], [ %xor, %for.body ]
  %arrayidx = getelementptr inbounds i32, ptr %a, i64 %iv
  %0 = load i32, ptr %arrayidx, align 4
  %xor = xor i32 %0, %sum.07
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond.not = icmp eq i64 %iv.next, %n
  br i1 %exitcond.not, label %for.end, label %for.body, !llvm.loop !0

for.end:                                 ; preds = %for.body, %entry
  ret i32 %xor
}

; CHECK-REMARK: vectorized loop (vectorization width: vscale x 8, interleaved count: 2)
; SMIN

define i32 @smin(ptr nocapture %a, ptr nocapture readonly %b, i64 %n) {
; CHECK-LABEL: @smin
; CHECK: vector.body:
; CHECK: %[[LOAD1:.*]] = load <vscale x 8 x i32>
; CHECK: %[[LOAD2:.*]] = load <vscale x 8 x i32>
; CHECK: %[[ICMP1:.*]] = icmp slt <vscale x 8 x i32> %[[LOAD1]]
; CHECK: %[[ICMP2:.*]] = icmp slt <vscale x 8 x i32> %[[LOAD2]]
; CHECK: %[[SEL1:.*]] = select <vscale x 8 x i1> %[[ICMP1]], <vscale x 8 x i32> %[[LOAD1]]
; CHECK: %[[SEL2:.*]] = select <vscale x 8 x i1> %[[ICMP2]], <vscale x 8 x i32> %[[LOAD2]]
; CHECK: middle.block:
; CHECK: %[[RDX:.*]] = call <vscale x 8 x i32> @llvm.smin.nxv8i32(<vscale x 8 x i32> %[[SEL1]], <vscale x 8 x i32> %[[SEL2]])
; CHECK-NEXT: call i32 @llvm.vector.reduce.smin.nxv8i32(<vscale x 8 x i32>  %[[RDX]])
entry:
  br label %for.body

for.body:                                         ; preds = %entry, %for.body
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %sum.010 = phi i32 [ 2, %entry ], [ %.sroa.speculated, %for.body ]
  %arrayidx = getelementptr inbounds i32, ptr %a, i64 %iv
  %0 = load i32, ptr %arrayidx, align 4
  %cmp.i = icmp slt i32 %0, %sum.010
  %.sroa.speculated = select i1 %cmp.i, i32 %0, i32 %sum.010
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond.not = icmp eq i64 %iv.next, %n
  br i1 %exitcond.not, label %for.end, label %for.body, !llvm.loop !0

for.end:
  ret i32 %.sroa.speculated
}

; CHECK-REMARK: vectorized loop (vectorization width: vscale x 8, interleaved count: 2)
; UMAX

define i32 @umax(ptr nocapture %a, ptr nocapture readonly %b, i64 %n) {
; CHECK-LABEL: @umax
; CHECK: vector.body:
; CHECK: %[[LOAD1:.*]] = load <vscale x 8 x i32>
; CHECK: %[[LOAD2:.*]] = load <vscale x 8 x i32>
; CHECK: %[[ICMP1:.*]] = icmp ugt <vscale x 8 x i32> %[[LOAD1]]
; CHECK: %[[ICMP2:.*]] = icmp ugt <vscale x 8 x i32> %[[LOAD2]]
; CHECK: %[[SEL1:.*]] = select <vscale x 8 x i1> %[[ICMP1]], <vscale x 8 x i32> %[[LOAD1]]
; CHECK: %[[SEL2:.*]] = select <vscale x 8 x i1> %[[ICMP2]], <vscale x 8 x i32> %[[LOAD2]]
; CHECK: middle.block:
; CHECK: %[[RDX:.*]] = call <vscale x 8 x i32> @llvm.umax.nxv8i32(<vscale x 8 x i32> %[[SEL1]], <vscale x 8 x i32> %[[SEL2]])
; CHECK-NEXT: call i32 @llvm.vector.reduce.umax.nxv8i32(<vscale x 8 x i32>  %[[RDX]])
entry:
  br label %for.body

for.body:                                         ; preds = %entry, %for.body
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %sum.010 = phi i32 [ 2, %entry ], [ %.sroa.speculated, %for.body ]
  %arrayidx = getelementptr inbounds i32, ptr %a, i64 %iv
  %0 = load i32, ptr %arrayidx, align 4
  %cmp.i = icmp ugt i32 %0, %sum.010
  %.sroa.speculated = select i1 %cmp.i, i32 %0, i32 %sum.010
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond.not = icmp eq i64 %iv.next, %n
  br i1 %exitcond.not, label %for.end, label %for.body, !llvm.loop !0

for.end:
  ret i32 %.sroa.speculated
}

; CHECK-REMARK: vectorized loop (vectorization width: vscale x 8, interleaved count: 2)
; FADD (FAST)

define float @fadd_fast(ptr noalias nocapture readonly %a, i64 %n) {
; CHECK-LABEL: @fadd_fast
; CHECK: vector.body:
; CHECK: %[[LOAD1:.*]] = load <vscale x 8 x float>
; CHECK: %[[LOAD2:.*]] = load <vscale x 8 x float>
; CHECK: %[[ADD1:.*]] = fadd fast <vscale x 8 x float> %[[LOAD1]]
; CHECK: %[[ADD2:.*]] = fadd fast <vscale x 8 x float> %[[LOAD2]]
; CHECK: middle.block:
; CHECK: %[[ADD:.*]] = fadd fast <vscale x 8 x float> %[[ADD2]], %[[ADD1]]
; CHECK-NEXT: call fast float @llvm.vector.reduce.fadd.nxv8f32(float -0.000000e+00, <vscale x 8 x float> %[[ADD]])
entry:
  br label %for.body

for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %sum.07 = phi float [ 0.000000e+00, %entry ], [ %add, %for.body ]
  %arrayidx = getelementptr inbounds float, ptr %a, i64 %iv
  %0 = load float, ptr %arrayidx, align 4
  %add = fadd fast float %0, %sum.07
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond.not = icmp eq i64 %iv.next, %n
  br i1 %exitcond.not, label %for.end, label %for.body, !llvm.loop !0

for.end:
  ret float %add
}

; CHECK-REMARK: Scalable vectorization not supported for the reduction operations found in this loop.
; CHECK-REMARK: vectorized loop (vectorization width: 16, interleaved count: 2)
define bfloat @fadd_fast_bfloat(ptr noalias nocapture readonly %a, i64 %n) {
; CHECK-LABEL: @fadd_fast_bfloat
; CHECK: vector.body:
; CHECK: %[[LOAD1:.*]] = load <16 x bfloat>
; CHECK: %[[LOAD2:.*]] = load <16 x bfloat>
; CHECK: %[[FADD1:.*]] = fadd fast <16 x bfloat> %[[LOAD1]]
; CHECK: %[[FADD2:.*]] = fadd fast <16 x bfloat> %[[LOAD2]]
; CHECK: middle.block:
; CHECK: %[[RDX:.*]] = fadd fast <16 x bfloat> %[[FADD2]], %[[FADD1]]
; CHECK: call fast bfloat @llvm.vector.reduce.fadd.v16bf16(bfloat 0xR8000, <16 x bfloat> %[[RDX]])
entry:
  br label %for.body

for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %sum.07 = phi bfloat [ 0.000000e+00, %entry ], [ %add, %for.body ]
  %arrayidx = getelementptr inbounds bfloat, ptr %a, i64 %iv
  %0 = load bfloat, ptr %arrayidx, align 4
  %add = fadd fast bfloat %0, %sum.07
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond.not = icmp eq i64 %iv.next, %n
  br i1 %exitcond.not, label %for.end, label %for.body, !llvm.loop !0

for.end:
  ret bfloat %add
}

; FMIN (FAST)

; CHECK-REMARK: vectorized loop (vectorization width: vscale x 8, interleaved count: 2)
define float @fmin_fast(ptr noalias nocapture readonly %a, i64 %n) #0 {
; CHECK-LABEL: @fmin_fast
; CHECK: vector.body:
; CHECK: %[[LOAD1:.*]] = load <vscale x 8 x float>
; CHECK: %[[LOAD2:.*]] = load <vscale x 8 x float>
; CHECK: %[[FCMP1:.*]] = fcmp olt <vscale x 8 x float> %[[LOAD1]]
; CHECK: %[[FCMP2:.*]] = fcmp olt <vscale x 8 x float> %[[LOAD2]]
; CHECK: %[[SEL1:.*]] = select <vscale x 8 x i1> %[[FCMP1]], <vscale x 8 x float> %[[LOAD1]]
; CHECK: %[[SEL2:.*]] = select <vscale x 8 x i1> %[[FCMP2]], <vscale x 8 x float> %[[LOAD2]]
; CHECK: middle.block:
; CHECK: %[[FCMP:.*]] = fcmp olt <vscale x 8 x float> %[[SEL1]], %[[SEL2]]
; CHECK-NEXT: %[[SEL:.*]] = select <vscale x 8 x i1> %[[FCMP]], <vscale x 8 x float> %[[SEL1]], <vscale x 8 x float> %[[SEL2]]
; CHECK-NEXT: call float @llvm.vector.reduce.fmin.nxv8f32(<vscale x 8 x float> %[[SEL]])
entry:
  br label %for.body

for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %sum.07 = phi float [ 0.000000e+00, %entry ], [ %.sroa.speculated, %for.body ]
  %arrayidx = getelementptr inbounds float, ptr %a, i64 %iv
  %0 = load float, ptr %arrayidx, align 4
  %cmp.i = fcmp olt float %0, %sum.07
  %.sroa.speculated = select i1 %cmp.i, float %0, float %sum.07
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond.not = icmp eq i64 %iv.next, %n
  br i1 %exitcond.not, label %for.end, label %for.body, !llvm.loop !0

for.end:
  ret float %.sroa.speculated
}

; FMAX (FAST)

; CHECK-REMARK: vectorized loop (vectorization width: vscale x 8, interleaved count: 2)
define float @fmax_fast(ptr noalias nocapture readonly %a, i64 %n) #0 {
; CHECK-LABEL: @fmax_fast
; CHECK: vector.body:
; CHECK: %[[LOAD1:.*]] = load <vscale x 8 x float>
; CHECK: %[[LOAD2:.*]] = load <vscale x 8 x float>
; CHECK: %[[FCMP1:.*]] = fcmp fast ogt <vscale x 8 x float> %[[LOAD1]]
; CHECK: %[[FCMP2:.*]] = fcmp fast ogt <vscale x 8 x float> %[[LOAD2]]
; CHECK: %[[SEL1:.*]] = select <vscale x 8 x i1> %[[FCMP1]], <vscale x 8 x float> %[[LOAD1]]
; CHECK: %[[SEL2:.*]] = select <vscale x 8 x i1> %[[FCMP2]], <vscale x 8 x float> %[[LOAD2]]
; CHECK: middle.block:
; CHECK: %[[FCMP:.*]] = fcmp fast ogt <vscale x 8 x float> %[[SEL1]], %[[SEL2]]
; CHECK-NEXT: %[[SEL:.*]] = select fast <vscale x 8 x i1> %[[FCMP]], <vscale x 8 x float> %[[SEL1]], <vscale x 8 x float> %[[SEL2]]
; CHECK-NEXT: call fast float @llvm.vector.reduce.fmax.nxv8f32(<vscale x 8 x float> %[[SEL]])
entry:
  br label %for.body

for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %sum.07 = phi float [ 0.000000e+00, %entry ], [ %.sroa.speculated, %for.body ]
  %arrayidx = getelementptr inbounds float, ptr %a, i64 %iv
  %0 = load float, ptr %arrayidx, align 4
  %cmp.i = fcmp fast ogt float %0, %sum.07
  %.sroa.speculated = select i1 %cmp.i, float %0, float %sum.07
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond.not = icmp eq i64 %iv.next, %n
  br i1 %exitcond.not, label %for.end, label %for.body, !llvm.loop !0

for.end:
  ret float %.sroa.speculated
}

; Reduction cannot be vectorized

; MUL

; CHECK-REMARK: Scalable vectorization not supported for the reduction operations found in this loop.
; CHECK-REMARK: vectorized loop (vectorization width: 8, interleaved count: 2)
define i32 @mul(ptr nocapture %a, ptr nocapture readonly %b, i64 %n) {
; CHECK-LABEL: @mul
; CHECK: vector.body:
; CHECK: %[[LOAD1:.*]] = load <8 x i32>
; CHECK: %[[LOAD2:.*]] = load <8 x i32>
; CHECK: %[[MUL1:.*]] = mul <8 x i32> %[[LOAD1]]
; CHECK: %[[MUL2:.*]] = mul <8 x i32> %[[LOAD2]]
; CHECK: middle.block:
; CHECK: %[[RDX:.*]] = mul <8 x i32> %[[MUL2]], %[[MUL1]]
; CHECK: call i32 @llvm.vector.reduce.mul.v8i32(<8 x i32> %[[RDX]])
entry:
  br label %for.body

for.body:                                         ; preds = %entry, %for.body
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %sum.07 = phi i32 [ 2, %entry ], [ %mul, %for.body ]
  %arrayidx = getelementptr inbounds i32, ptr %a, i64 %iv
  %0 = load i32, ptr %arrayidx, align 4
  %mul = mul nsw i32 %0, %sum.07
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond.not = icmp eq i64 %iv.next, %n
  br i1 %exitcond.not, label %for.end, label %for.body, !llvm.loop !0

for.end:                                 ; preds = %for.body, %entry
  ret i32 %mul
}

; Note: This test was added to ensure we always check the legality of reductions (and emit a warning if necessary) before checking for memory dependencies
; CHECK-REMARK: Scalable vectorization not supported for the reduction operations found in this loop.
; CHECK-REMARK: vectorized loop (vectorization width: 8, interleaved count: 2)
define i32 @memory_dependence(ptr noalias nocapture %a, ptr noalias nocapture readonly %b, i64 %n) {
; CHECK-LABEL: @memory_dependence
; CHECK: vector.body:
; CHECK: %[[LOAD1:.*]] = load <8 x i32>
; CHECK: %[[LOAD2:.*]] = load <8 x i32>
; CHECK: %[[LOAD3:.*]] = load <8 x i32>
; CHECK: %[[LOAD4:.*]] = load <8 x i32>
; CHECK: %[[ADD1:.*]] = add nsw <8 x i32> %[[LOAD3]], %[[LOAD1]]
; CHECK: %[[ADD2:.*]] = add nsw <8 x i32> %[[LOAD4]], %[[LOAD2]]
; CHECK: %[[MUL1:.*]] = mul <8 x i32> %[[LOAD3]]
; CHECK: %[[MUL2:.*]] = mul <8 x i32> %[[LOAD4]]
; CHECK: middle.block:
; CHECK: %[[RDX:.*]] = mul <8 x i32> %[[MUL2]], %[[MUL1]]
; CHECK: call i32 @llvm.vector.reduce.mul.v8i32(<8 x i32> %[[RDX]])
entry:
  br label %for.body

for.body:
  %i = phi i64 [ %inc, %for.body ], [ 0, %entry ]
  %sum = phi i32 [ %mul, %for.body ], [ 2, %entry ]
  %arrayidx = getelementptr inbounds i32, ptr %a, i64 %i
  %0 = load i32, ptr %arrayidx, align 4
  %arrayidx1 = getelementptr inbounds i32, ptr %b, i64 %i
  %1 = load i32, ptr %arrayidx1, align 4
  %add = add nsw i32 %1, %0
  %add2 = add nuw nsw i64 %i, 32
  %arrayidx3 = getelementptr inbounds i32, ptr %a, i64 %add2
  store i32 %add, ptr %arrayidx3, align 4
  %mul = mul nsw i32 %1, %sum
  %inc = add nuw nsw i64 %i, 1
  %exitcond.not = icmp eq i64 %inc, %n
  br i1 %exitcond.not, label %for.end, label %for.body, !llvm.loop !0

for.end:
  ret i32 %mul
}

; CHECK-REMARK: vectorized loop (vectorization width: vscale x 4, interleaved count: 2)
define float @fmuladd(ptr %a, ptr %b, i64 %n) {
; CHECK-LABEL: @fmuladd(
; CHECK: vector.body:
; CHECK: [[WIDE_LOAD:%.*]] = load <vscale x 4 x float>
; CHECK: [[WIDE_LOAD2:%.*]] = load <vscale x 4 x float>
; CHECK: [[WIDE_LOAD3:%.*]] = load <vscale x 4 x float>
; CHECK: [[WIDE_LOAD4:%.*]] = load <vscale x 4 x float>
; CHECK: [[MULADD1:%.*]] = call reassoc <vscale x 4 x float> @llvm.fmuladd.nxv4f32(<vscale x 4 x float> [[WIDE_LOAD]], <vscale x 4 x float> [[WIDE_LOAD3]],
; CHECK: [[MULADD2:%.*]] = call reassoc <vscale x 4 x float> @llvm.fmuladd.nxv4f32(<vscale x 4 x float> [[WIDE_LOAD2]], <vscale x 4 x float> [[WIDE_LOAD4]],
; CHECK: middle.block:
; CHECK: [[BIN_RDX:%.*]] = fadd reassoc <vscale x 4 x float> [[MULADD2]], [[MULADD1]]
; CHECK: call reassoc float @llvm.vector.reduce.fadd.nxv4f32(float -0.000000e+00, <vscale x 4 x float> [[BIN_RDX]])
;
entry:
  br label %for.body

for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %sum.07 = phi float [ 0.000000e+00, %entry ], [ %muladd, %for.body ]
  %arrayidx = getelementptr inbounds float, ptr %a, i64 %iv
  %0 = load float, ptr %arrayidx, align 4
  %arrayidx2 = getelementptr inbounds float, ptr %b, i64 %iv
  %1 = load float, ptr %arrayidx2, align 4
  %muladd = tail call reassoc float @llvm.fmuladd.f32(float %0, float %1, float %sum.07)
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond.not = icmp eq i64 %iv.next, %n
  br i1 %exitcond.not, label %for.end, label %for.body, !llvm.loop !1

for.end:
  ret float %muladd
}

declare float @llvm.fmuladd.f32(float, float, float)

attributes #0 = { "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" }

!0 = distinct !{!0, !1, !2, !3, !4}
!1 = !{!"llvm.loop.vectorize.width", i32 8}
!2 = !{!"llvm.loop.vectorize.scalable.enable", i1 true}
!3 = !{!"llvm.loop.interleave.count", i32 2}
!4 = !{!"llvm.loop.vectorize.enable", i1 true}
