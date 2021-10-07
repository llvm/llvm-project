; RUN: opt %s -S -riscv-gather-scatter-lowering -mtriple=riscv64 -mattr=+m,+experimental-v -riscv-v-vector-bits-min=256 | FileCheck %s
; RUN: llc < %s -mtriple=riscv64 -mattr=+m,+experimental-v -riscv-v-vector-bits-min=256 | FileCheck %s --check-prefix=CHECK-ASM

%struct.foo = type { i32, i32, i32, i32 }

; void gather(signed char * __restrict  A, signed char * __restrict B) {
;   for (int i = 0; i != 1024; ++i)
;       A[i] += B[i * 5];
; }
define void @gather(i8* noalias nocapture %A, i8* noalias nocapture readonly %B) {
; CHECK-LABEL: @gather(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    br label [[VECTOR_BODY:%.*]]
; CHECK:       vector.body:
; CHECK-NEXT:    [[INDEX:%.*]] = phi i64 [ 0, [[ENTRY:%.*]] ], [ [[INDEX_NEXT:%.*]], [[VECTOR_BODY]] ]
; CHECK-NEXT:    [[VEC_IND_SCALAR:%.*]] = phi i64 [ 0, [[ENTRY]] ], [ [[VEC_IND_NEXT_SCALAR:%.*]], [[VECTOR_BODY]] ]
; CHECK-NEXT:    [[TMP0:%.*]] = getelementptr i8, i8* [[B:%.*]], i64 [[VEC_IND_SCALAR]]
; CHECK-NEXT:    [[WIDE_MASKED_GATHER:%.*]] = call <32 x i8> @llvm.riscv.masked.strided.load.v32i8.p0i8.i64(<32 x i8> undef, i8* [[TMP0]], i64 5, <32 x i1> <i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true>)
; CHECK-NEXT:    [[TMP1:%.*]] = getelementptr inbounds i8, i8* [[A:%.*]], i64 [[INDEX]]
; CHECK-NEXT:    [[TMP2:%.*]] = bitcast i8* [[TMP1]] to <32 x i8>*
; CHECK-NEXT:    [[WIDE_LOAD:%.*]] = load <32 x i8>, <32 x i8>* [[TMP2]], align 1
; CHECK-NEXT:    [[TMP3:%.*]] = add <32 x i8> [[WIDE_LOAD]], [[WIDE_MASKED_GATHER]]
; CHECK-NEXT:    [[TMP4:%.*]] = bitcast i8* [[TMP1]] to <32 x i8>*
; CHECK-NEXT:    store <32 x i8> [[TMP3]], <32 x i8>* [[TMP4]], align 1
; CHECK-NEXT:    [[INDEX_NEXT]] = add nuw i64 [[INDEX]], 32
; CHECK-NEXT:    [[VEC_IND_NEXT_SCALAR]] = add i64 [[VEC_IND_SCALAR]], 160
; CHECK-NEXT:    [[TMP5:%.*]] = icmp eq i64 [[INDEX_NEXT]], 1024
; CHECK-NEXT:    br i1 [[TMP5]], label [[FOR_COND_CLEANUP:%.*]], label [[VECTOR_BODY]]
; CHECK:       for.cond.cleanup:
; CHECK-NEXT:    ret void
;
; CHECK-ASM-LABEL: gather:
; CHECK-ASM:       # %bb.0: # %entry
; CHECK-ASM-NEXT:    mv a2, zero
; CHECK-ASM-NEXT:    addi a6, zero, 32
; CHECK-ASM-NEXT:    addi a4, zero, 5
; CHECK-ASM-NEXT:    addi a5, zero, 1024
; CHECK-ASM-NEXT:  .LBB0_1: # %vector.body
; CHECK-ASM-NEXT:    # =>This Inner Loop Header: Depth=1
; CHECK-ASM-NEXT:    vsetvli zero, a6, e8, m1, ta, mu
; CHECK-ASM-NEXT:    vlse8.v v25, (a1), a4
; CHECK-ASM-NEXT:    add a3, a0, a2
; CHECK-ASM-NEXT:    vle8.v v26, (a3)
; CHECK-ASM-NEXT:    vadd.vv v25, v26, v25
; CHECK-ASM-NEXT:    vse8.v v25, (a3)
; CHECK-ASM-NEXT:    addi a2, a2, 32
; CHECK-ASM-NEXT:    addi a1, a1, 160
; CHECK-ASM-NEXT:    bne a2, a5, .LBB0_1
; CHECK-ASM-NEXT:  # %bb.2: # %for.cond.cleanup
; CHECK-ASM-NEXT:    ret
entry:
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %entry
  %index = phi i64 [ 0, %entry ], [ %index.next, %vector.body ]
  %vec.ind = phi <32 x i64> [ <i64 0, i64 1, i64 2, i64 3, i64 4, i64 5, i64 6, i64 7, i64 8, i64 9, i64 10, i64 11, i64 12, i64 13, i64 14, i64 15, i64 16, i64 17, i64 18, i64 19, i64 20, i64 21, i64 22, i64 23, i64 24, i64 25, i64 26, i64 27, i64 28, i64 29, i64 30, i64 31>, %entry ], [ %vec.ind.next, %vector.body ]
  %0 = mul nuw nsw <32 x i64> %vec.ind, <i64 5, i64 5, i64 5, i64 5, i64 5, i64 5, i64 5, i64 5, i64 5, i64 5, i64 5, i64 5, i64 5, i64 5, i64 5, i64 5, i64 5, i64 5, i64 5, i64 5, i64 5, i64 5, i64 5, i64 5, i64 5, i64 5, i64 5, i64 5, i64 5, i64 5, i64 5, i64 5>
  %1 = getelementptr inbounds i8, i8* %B, <32 x i64> %0
  %wide.masked.gather = call <32 x i8> @llvm.masked.gather.v32i8.v32p0i8(<32 x i8*> %1, i32 1, <32 x i1> <i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true>, <32 x i8> undef)
  %2 = getelementptr inbounds i8, i8* %A, i64 %index
  %3 = bitcast i8* %2 to <32 x i8>*
  %wide.load = load <32 x i8>, <32 x i8>* %3, align 1
  %4 = add <32 x i8> %wide.load, %wide.masked.gather
  %5 = bitcast i8* %2 to <32 x i8>*
  store <32 x i8> %4, <32 x i8>* %5, align 1
  %index.next = add nuw i64 %index, 32
  %vec.ind.next = add <32 x i64> %vec.ind, <i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32>
  %6 = icmp eq i64 %index.next, 1024
  br i1 %6, label %for.cond.cleanup, label %vector.body

for.cond.cleanup:                                 ; preds = %vector.body
  ret void
}

define void @gather_masked(i8* noalias nocapture %A, i8* noalias nocapture readonly %B, <32 x i8> %maskedoff) {
; CHECK-LABEL: @gather_masked(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    br label [[VECTOR_BODY:%.*]]
; CHECK:       vector.body:
; CHECK-NEXT:    [[INDEX:%.*]] = phi i64 [ 0, [[ENTRY:%.*]] ], [ [[INDEX_NEXT:%.*]], [[VECTOR_BODY]] ]
; CHECK-NEXT:    [[VEC_IND_SCALAR:%.*]] = phi i64 [ 0, [[ENTRY]] ], [ [[VEC_IND_NEXT_SCALAR:%.*]], [[VECTOR_BODY]] ]
; CHECK-NEXT:    [[TMP0:%.*]] = getelementptr i8, i8* [[B:%.*]], i64 [[VEC_IND_SCALAR]]
; CHECK-NEXT:    [[WIDE_MASKED_GATHER:%.*]] = call <32 x i8> @llvm.riscv.masked.strided.load.v32i8.p0i8.i64(<32 x i8> [[MASKEDOFF:%.*]], i8* [[TMP0]], i64 5, <32 x i1> <i1 true, i1 false, i1 false, i1 true, i1 false, i1 true, i1 true, i1 false, i1 true, i1 true, i1 false, i1 false, i1 true, i1 false, i1 true, i1 false, i1 true, i1 false, i1 true, i1 true, i1 false, i1 true, i1 false, i1 false, i1 false, i1 false, i1 false, i1 false, i1 true, i1 true, i1 true, i1 true>)
; CHECK-NEXT:    [[TMP1:%.*]] = getelementptr inbounds i8, i8* [[A:%.*]], i64 [[INDEX]]
; CHECK-NEXT:    [[TMP2:%.*]] = bitcast i8* [[TMP1]] to <32 x i8>*
; CHECK-NEXT:    [[WIDE_LOAD:%.*]] = load <32 x i8>, <32 x i8>* [[TMP2]], align 1
; CHECK-NEXT:    [[TMP3:%.*]] = add <32 x i8> [[WIDE_LOAD]], [[WIDE_MASKED_GATHER]]
; CHECK-NEXT:    [[TMP4:%.*]] = bitcast i8* [[TMP1]] to <32 x i8>*
; CHECK-NEXT:    store <32 x i8> [[TMP3]], <32 x i8>* [[TMP4]], align 1
; CHECK-NEXT:    [[INDEX_NEXT]] = add nuw i64 [[INDEX]], 32
; CHECK-NEXT:    [[VEC_IND_NEXT_SCALAR]] = add i64 [[VEC_IND_SCALAR]], 160
; CHECK-NEXT:    [[TMP5:%.*]] = icmp eq i64 [[INDEX_NEXT]], 1024
; CHECK-NEXT:    br i1 [[TMP5]], label [[FOR_COND_CLEANUP:%.*]], label [[VECTOR_BODY]]
; CHECK:       for.cond.cleanup:
; CHECK-NEXT:    ret void
;
; CHECK-ASM-LABEL: gather_masked:
; CHECK-ASM:       # %bb.0: # %entry
; CHECK-ASM-NEXT:    mv a2, zero
; CHECK-ASM-NEXT:    lui a3, 983765
; CHECK-ASM-NEXT:    addiw a3, a3, 873
; CHECK-ASM-NEXT:    vsetivli zero, 1, e32, mf2, ta, mu
; CHECK-ASM-NEXT:    vmv.s.x v0, a3
; CHECK-ASM-NEXT:    addi a6, zero, 32
; CHECK-ASM-NEXT:    addi a4, zero, 5
; CHECK-ASM-NEXT:    addi a5, zero, 1024
; CHECK-ASM-NEXT:  .LBB1_1: # %vector.body
; CHECK-ASM-NEXT:    # =>This Inner Loop Header: Depth=1
; CHECK-ASM-NEXT:    vsetvli zero, a6, e8, m1, ta, mu
; CHECK-ASM-NEXT:    vmv1r.v v25, v8
; CHECK-ASM-NEXT:    vlse8.v v25, (a1), a4, v0.t
; CHECK-ASM-NEXT:    add a3, a0, a2
; CHECK-ASM-NEXT:    vle8.v v26, (a3)
; CHECK-ASM-NEXT:    vadd.vv v25, v26, v25
; CHECK-ASM-NEXT:    vse8.v v25, (a3)
; CHECK-ASM-NEXT:    addi a2, a2, 32
; CHECK-ASM-NEXT:    addi a1, a1, 160
; CHECK-ASM-NEXT:    bne a2, a5, .LBB1_1
; CHECK-ASM-NEXT:  # %bb.2: # %for.cond.cleanup
; CHECK-ASM-NEXT:    ret
entry:
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %entry
  %index = phi i64 [ 0, %entry ], [ %index.next, %vector.body ]
  %vec.ind = phi <32 x i64> [ <i64 0, i64 1, i64 2, i64 3, i64 4, i64 5, i64 6, i64 7, i64 8, i64 9, i64 10, i64 11, i64 12, i64 13, i64 14, i64 15, i64 16, i64 17, i64 18, i64 19, i64 20, i64 21, i64 22, i64 23, i64 24, i64 25, i64 26, i64 27, i64 28, i64 29, i64 30, i64 31>, %entry ], [ %vec.ind.next, %vector.body ]
  %0 = mul nuw nsw <32 x i64> %vec.ind, <i64 5, i64 5, i64 5, i64 5, i64 5, i64 5, i64 5, i64 5, i64 5, i64 5, i64 5, i64 5, i64 5, i64 5, i64 5, i64 5, i64 5, i64 5, i64 5, i64 5, i64 5, i64 5, i64 5, i64 5, i64 5, i64 5, i64 5, i64 5, i64 5, i64 5, i64 5, i64 5>
  %1 = getelementptr inbounds i8, i8* %B, <32 x i64> %0
  %wide.masked.gather = call <32 x i8> @llvm.masked.gather.v32i8.v32p0i8(<32 x i8*> %1, i32 1, <32 x i1> <i1 true, i1 false, i1 false, i1 true, i1 false, i1 true, i1 true, i1 false, i1 true, i1 true, i1 false, i1 false, i1 true, i1 false, i1 true, i1 false, i1 true, i1 false, i1 true, i1 true, i1 false, i1 true, i1 false, i1 false, i1 false, i1 false, i1 false, i1 false, i1 true, i1 true, i1 true, i1 true>, <32 x i8> %maskedoff)
  %2 = getelementptr inbounds i8, i8* %A, i64 %index
  %3 = bitcast i8* %2 to <32 x i8>*
  %wide.load = load <32 x i8>, <32 x i8>* %3, align 1
  %4 = add <32 x i8> %wide.load, %wide.masked.gather
  %5 = bitcast i8* %2 to <32 x i8>*
  store <32 x i8> %4, <32 x i8>* %5, align 1
  %index.next = add nuw i64 %index, 32
  %vec.ind.next = add <32 x i64> %vec.ind, <i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32>
  %6 = icmp eq i64 %index.next, 1024
  br i1 %6, label %for.cond.cleanup, label %vector.body

for.cond.cleanup:                                 ; preds = %vector.body
  ret void
}

define void @gather_negative_stride(i8* noalias nocapture %A, i8* noalias nocapture readonly %B) {
; CHECK-LABEL: @gather_negative_stride(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    br label [[VECTOR_BODY:%.*]]
; CHECK:       vector.body:
; CHECK-NEXT:    [[INDEX:%.*]] = phi i64 [ 0, [[ENTRY:%.*]] ], [ [[INDEX_NEXT:%.*]], [[VECTOR_BODY]] ]
; CHECK-NEXT:    [[VEC_IND_SCALAR:%.*]] = phi i64 [ 155, [[ENTRY]] ], [ [[VEC_IND_NEXT_SCALAR:%.*]], [[VECTOR_BODY]] ]
; CHECK-NEXT:    [[TMP0:%.*]] = getelementptr i8, i8* [[B:%.*]], i64 [[VEC_IND_SCALAR]]
; CHECK-NEXT:    [[WIDE_MASKED_GATHER:%.*]] = call <32 x i8> @llvm.riscv.masked.strided.load.v32i8.p0i8.i64(<32 x i8> undef, i8* [[TMP0]], i64 -5, <32 x i1> <i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true>)
; CHECK-NEXT:    [[TMP1:%.*]] = getelementptr inbounds i8, i8* [[A:%.*]], i64 [[INDEX]]
; CHECK-NEXT:    [[TMP2:%.*]] = bitcast i8* [[TMP1]] to <32 x i8>*
; CHECK-NEXT:    [[WIDE_LOAD:%.*]] = load <32 x i8>, <32 x i8>* [[TMP2]], align 1
; CHECK-NEXT:    [[TMP3:%.*]] = add <32 x i8> [[WIDE_LOAD]], [[WIDE_MASKED_GATHER]]
; CHECK-NEXT:    [[TMP4:%.*]] = bitcast i8* [[TMP1]] to <32 x i8>*
; CHECK-NEXT:    store <32 x i8> [[TMP3]], <32 x i8>* [[TMP4]], align 1
; CHECK-NEXT:    [[INDEX_NEXT]] = add nuw i64 [[INDEX]], 32
; CHECK-NEXT:    [[VEC_IND_NEXT_SCALAR]] = add i64 [[VEC_IND_SCALAR]], 160
; CHECK-NEXT:    [[TMP5:%.*]] = icmp eq i64 [[INDEX_NEXT]], 1024
; CHECK-NEXT:    br i1 [[TMP5]], label [[FOR_COND_CLEANUP:%.*]], label [[VECTOR_BODY]]
; CHECK:       for.cond.cleanup:
; CHECK-NEXT:    ret void
;
; CHECK-ASM-LABEL: gather_negative_stride:
; CHECK-ASM:       # %bb.0: # %entry
; CHECK-ASM-NEXT:    mv a2, zero
; CHECK-ASM-NEXT:    addi a1, a1, 155
; CHECK-ASM-NEXT:    addi a6, zero, 32
; CHECK-ASM-NEXT:    addi a4, zero, -5
; CHECK-ASM-NEXT:    addi a5, zero, 1024
; CHECK-ASM-NEXT:  .LBB2_1: # %vector.body
; CHECK-ASM-NEXT:    # =>This Inner Loop Header: Depth=1
; CHECK-ASM-NEXT:    vsetvli zero, a6, e8, m1, ta, mu
; CHECK-ASM-NEXT:    vlse8.v v25, (a1), a4
; CHECK-ASM-NEXT:    add a3, a0, a2
; CHECK-ASM-NEXT:    vle8.v v26, (a3)
; CHECK-ASM-NEXT:    vadd.vv v25, v26, v25
; CHECK-ASM-NEXT:    vse8.v v25, (a3)
; CHECK-ASM-NEXT:    addi a2, a2, 32
; CHECK-ASM-NEXT:    addi a1, a1, 160
; CHECK-ASM-NEXT:    bne a2, a5, .LBB2_1
; CHECK-ASM-NEXT:  # %bb.2: # %for.cond.cleanup
; CHECK-ASM-NEXT:    ret
entry:
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %entry
  %index = phi i64 [ 0, %entry ], [ %index.next, %vector.body ]
  %vec.ind = phi <32 x i64> [ <i64 31, i64 30, i64 29, i64 28, i64 27, i64 26, i64 25, i64 24, i64 23, i64 22, i64 21, i64 20, i64 19, i64 18, i64 17, i64 16, i64 15, i64 14, i64 13, i64 12, i64 11, i64 10, i64 9, i64 8, i64 7, i64 6, i64 5, i64 4, i64 3, i64 2, i64 1, i64 0>, %entry ], [ %vec.ind.next, %vector.body ]
  %0 = mul nuw nsw <32 x i64> %vec.ind, <i64 5, i64 5, i64 5, i64 5, i64 5, i64 5, i64 5, i64 5, i64 5, i64 5, i64 5, i64 5, i64 5, i64 5, i64 5, i64 5, i64 5, i64 5, i64 5, i64 5, i64 5, i64 5, i64 5, i64 5, i64 5, i64 5, i64 5, i64 5, i64 5, i64 5, i64 5, i64 5>
  %1 = getelementptr inbounds i8, i8* %B, <32 x i64> %0
  %wide.masked.gather = call <32 x i8> @llvm.masked.gather.v32i8.v32p0i8(<32 x i8*> %1, i32 1, <32 x i1> <i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true>, <32 x i8> undef)
  %2 = getelementptr inbounds i8, i8* %A, i64 %index
  %3 = bitcast i8* %2 to <32 x i8>*
  %wide.load = load <32 x i8>, <32 x i8>* %3, align 1
  %4 = add <32 x i8> %wide.load, %wide.masked.gather
  %5 = bitcast i8* %2 to <32 x i8>*
  store <32 x i8> %4, <32 x i8>* %5, align 1
  %index.next = add nuw i64 %index, 32
  %vec.ind.next = add <32 x i64> %vec.ind, <i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32>
  %6 = icmp eq i64 %index.next, 1024
  br i1 %6, label %for.cond.cleanup, label %vector.body

for.cond.cleanup:                                 ; preds = %vector.body
  ret void
}

define void @gather_zero_stride(i8* noalias nocapture %A, i8* noalias nocapture readonly %B) {
; CHECK-LABEL: @gather_zero_stride(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    br label [[VECTOR_BODY:%.*]]
; CHECK:       vector.body:
; CHECK-NEXT:    [[INDEX:%.*]] = phi i64 [ 0, [[ENTRY:%.*]] ], [ [[INDEX_NEXT:%.*]], [[VECTOR_BODY]] ]
; CHECK-NEXT:    [[VEC_IND_SCALAR:%.*]] = phi i64 [ 0, [[ENTRY]] ], [ [[VEC_IND_NEXT_SCALAR:%.*]], [[VECTOR_BODY]] ]
; CHECK-NEXT:    [[TMP0:%.*]] = getelementptr i8, i8* [[B:%.*]], i64 [[VEC_IND_SCALAR]]
; CHECK-NEXT:    [[WIDE_MASKED_GATHER:%.*]] = call <32 x i8> @llvm.riscv.masked.strided.load.v32i8.p0i8.i64(<32 x i8> undef, i8* [[TMP0]], i64 0, <32 x i1> <i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true>)
; CHECK-NEXT:    [[TMP1:%.*]] = getelementptr inbounds i8, i8* [[A:%.*]], i64 [[INDEX]]
; CHECK-NEXT:    [[TMP2:%.*]] = bitcast i8* [[TMP1]] to <32 x i8>*
; CHECK-NEXT:    [[WIDE_LOAD:%.*]] = load <32 x i8>, <32 x i8>* [[TMP2]], align 1
; CHECK-NEXT:    [[TMP3:%.*]] = add <32 x i8> [[WIDE_LOAD]], [[WIDE_MASKED_GATHER]]
; CHECK-NEXT:    [[TMP4:%.*]] = bitcast i8* [[TMP1]] to <32 x i8>*
; CHECK-NEXT:    store <32 x i8> [[TMP3]], <32 x i8>* [[TMP4]], align 1
; CHECK-NEXT:    [[INDEX_NEXT]] = add nuw i64 [[INDEX]], 32
; CHECK-NEXT:    [[VEC_IND_NEXT_SCALAR]] = add i64 [[VEC_IND_SCALAR]], 160
; CHECK-NEXT:    [[TMP5:%.*]] = icmp eq i64 [[INDEX_NEXT]], 1024
; CHECK-NEXT:    br i1 [[TMP5]], label [[FOR_COND_CLEANUP:%.*]], label [[VECTOR_BODY]]
; CHECK:       for.cond.cleanup:
; CHECK-NEXT:    ret void
;
; CHECK-ASM-LABEL: gather_zero_stride:
; CHECK-ASM:       # %bb.0: # %entry
; CHECK-ASM-NEXT:    mv a2, zero
; CHECK-ASM-NEXT:    addi a3, zero, 32
; CHECK-ASM-NEXT:    addi a4, zero, 1024
; CHECK-ASM-NEXT:  .LBB3_1: # %vector.body
; CHECK-ASM-NEXT:    # =>This Inner Loop Header: Depth=1
; CHECK-ASM-NEXT:    vsetvli zero, a3, e8, m1, ta, mu
; CHECK-ASM-NEXT:    vlse8.v v25, (a1), zero
; CHECK-ASM-NEXT:    add a5, a0, a2
; CHECK-ASM-NEXT:    vle8.v v26, (a5)
; CHECK-ASM-NEXT:    vadd.vv v25, v26, v25
; CHECK-ASM-NEXT:    vse8.v v25, (a5)
; CHECK-ASM-NEXT:    addi a2, a2, 32
; CHECK-ASM-NEXT:    addi a1, a1, 160
; CHECK-ASM-NEXT:    bne a2, a4, .LBB3_1
; CHECK-ASM-NEXT:  # %bb.2: # %for.cond.cleanup
; CHECK-ASM-NEXT:    ret
entry:
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %entry
  %index = phi i64 [ 0, %entry ], [ %index.next, %vector.body ]
  %vec.ind = phi <32 x i64> [ zeroinitializer, %entry ], [ %vec.ind.next, %vector.body ]
  %0 = mul nuw nsw <32 x i64> %vec.ind, <i64 5, i64 5, i64 5, i64 5, i64 5, i64 5, i64 5, i64 5, i64 5, i64 5, i64 5, i64 5, i64 5, i64 5, i64 5, i64 5, i64 5, i64 5, i64 5, i64 5, i64 5, i64 5, i64 5, i64 5, i64 5, i64 5, i64 5, i64 5, i64 5, i64 5, i64 5, i64 5>
  %1 = getelementptr inbounds i8, i8* %B, <32 x i64> %0
  %wide.masked.gather = call <32 x i8> @llvm.masked.gather.v32i8.v32p0i8(<32 x i8*> %1, i32 1, <32 x i1> <i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true>, <32 x i8> undef)
  %2 = getelementptr inbounds i8, i8* %A, i64 %index
  %3 = bitcast i8* %2 to <32 x i8>*
  %wide.load = load <32 x i8>, <32 x i8>* %3, align 1
  %4 = add <32 x i8> %wide.load, %wide.masked.gather
  %5 = bitcast i8* %2 to <32 x i8>*
  store <32 x i8> %4, <32 x i8>* %5, align 1
  %index.next = add nuw i64 %index, 32
  %vec.ind.next = add <32 x i64> %vec.ind, <i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32>
  %6 = icmp eq i64 %index.next, 1024
  br i1 %6, label %for.cond.cleanup, label %vector.body

for.cond.cleanup:                                 ; preds = %vector.body
  ret void
}

;void scatter(signed char * __restrict  A, signed char * __restrict B) {
;  for (int i = 0; i < 1024; ++i)
;      A[i * 5] += B[i];
;}
define void @scatter(i8* noalias nocapture %A, i8* noalias nocapture readonly %B) {
; CHECK-LABEL: @scatter(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    br label [[VECTOR_BODY:%.*]]
; CHECK:       vector.body:
; CHECK-NEXT:    [[INDEX:%.*]] = phi i64 [ 0, [[ENTRY:%.*]] ], [ [[INDEX_NEXT:%.*]], [[VECTOR_BODY]] ]
; CHECK-NEXT:    [[VEC_IND_SCALAR:%.*]] = phi i64 [ 0, [[ENTRY]] ], [ [[VEC_IND_NEXT_SCALAR:%.*]], [[VECTOR_BODY]] ]
; CHECK-NEXT:    [[VEC_IND_SCALAR1:%.*]] = phi i64 [ 0, [[ENTRY]] ], [ [[VEC_IND_NEXT_SCALAR2:%.*]], [[VECTOR_BODY]] ]
; CHECK-NEXT:    [[TMP0:%.*]] = getelementptr inbounds i8, i8* [[B:%.*]], i64 [[INDEX]]
; CHECK-NEXT:    [[TMP1:%.*]] = bitcast i8* [[TMP0]] to <32 x i8>*
; CHECK-NEXT:    [[WIDE_LOAD:%.*]] = load <32 x i8>, <32 x i8>* [[TMP1]], align 1
; CHECK-NEXT:    [[TMP2:%.*]] = getelementptr i8, i8* [[A:%.*]], i64 [[VEC_IND_SCALAR]]
; CHECK-NEXT:    [[TMP3:%.*]] = getelementptr i8, i8* [[A]], i64 [[VEC_IND_SCALAR1]]
; CHECK-NEXT:    [[WIDE_MASKED_GATHER:%.*]] = call <32 x i8> @llvm.riscv.masked.strided.load.v32i8.p0i8.i64(<32 x i8> undef, i8* [[TMP2]], i64 5, <32 x i1> <i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true>)
; CHECK-NEXT:    [[TMP4:%.*]] = add <32 x i8> [[WIDE_MASKED_GATHER]], [[WIDE_LOAD]]
; CHECK-NEXT:    call void @llvm.riscv.masked.strided.store.v32i8.p0i8.i64(<32 x i8> [[TMP4]], i8* [[TMP3]], i64 5, <32 x i1> <i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true>)
; CHECK-NEXT:    [[INDEX_NEXT]] = add nuw i64 [[INDEX]], 32
; CHECK-NEXT:    [[VEC_IND_NEXT_SCALAR]] = add i64 [[VEC_IND_SCALAR]], 160
; CHECK-NEXT:    [[VEC_IND_NEXT_SCALAR2]] = add i64 [[VEC_IND_SCALAR1]], 160
; CHECK-NEXT:    [[TMP5:%.*]] = icmp eq i64 [[INDEX_NEXT]], 1024
; CHECK-NEXT:    br i1 [[TMP5]], label [[FOR_COND_CLEANUP:%.*]], label [[VECTOR_BODY]]
; CHECK:       for.cond.cleanup:
; CHECK-NEXT:    ret void
;
; CHECK-ASM-LABEL: scatter:
; CHECK-ASM:       # %bb.0: # %entry
; CHECK-ASM-NEXT:    mv a2, zero
; CHECK-ASM-NEXT:    addi a6, zero, 32
; CHECK-ASM-NEXT:    addi a4, zero, 5
; CHECK-ASM-NEXT:    addi a5, zero, 1024
; CHECK-ASM-NEXT:  .LBB4_1: # %vector.body
; CHECK-ASM-NEXT:    # =>This Inner Loop Header: Depth=1
; CHECK-ASM-NEXT:    add a3, a1, a2
; CHECK-ASM-NEXT:    vsetvli zero, a6, e8, m1, ta, mu
; CHECK-ASM-NEXT:    vle8.v v25, (a3)
; CHECK-ASM-NEXT:    vlse8.v v26, (a0), a4
; CHECK-ASM-NEXT:    vadd.vv v25, v26, v25
; CHECK-ASM-NEXT:    vsse8.v v25, (a0), a4
; CHECK-ASM-NEXT:    addi a2, a2, 32
; CHECK-ASM-NEXT:    addi a0, a0, 160
; CHECK-ASM-NEXT:    bne a2, a5, .LBB4_1
; CHECK-ASM-NEXT:  # %bb.2: # %for.cond.cleanup
; CHECK-ASM-NEXT:    ret
entry:
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %entry
  %index = phi i64 [ 0, %entry ], [ %index.next, %vector.body ]
  %vec.ind = phi <32 x i64> [ <i64 0, i64 1, i64 2, i64 3, i64 4, i64 5, i64 6, i64 7, i64 8, i64 9, i64 10, i64 11, i64 12, i64 13, i64 14, i64 15, i64 16, i64 17, i64 18, i64 19, i64 20, i64 21, i64 22, i64 23, i64 24, i64 25, i64 26, i64 27, i64 28, i64 29, i64 30, i64 31>, %entry ], [ %vec.ind.next, %vector.body ]
  %0 = getelementptr inbounds i8, i8* %B, i64 %index
  %1 = bitcast i8* %0 to <32 x i8>*
  %wide.load = load <32 x i8>, <32 x i8>* %1, align 1
  %2 = mul nuw nsw <32 x i64> %vec.ind, <i64 5, i64 5, i64 5, i64 5, i64 5, i64 5, i64 5, i64 5, i64 5, i64 5, i64 5, i64 5, i64 5, i64 5, i64 5, i64 5, i64 5, i64 5, i64 5, i64 5, i64 5, i64 5, i64 5, i64 5, i64 5, i64 5, i64 5, i64 5, i64 5, i64 5, i64 5, i64 5>
  %3 = getelementptr inbounds i8, i8* %A, <32 x i64> %2
  %wide.masked.gather = call <32 x i8> @llvm.masked.gather.v32i8.v32p0i8(<32 x i8*> %3, i32 1, <32 x i1> <i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true>, <32 x i8> undef)
  %4 = add <32 x i8> %wide.masked.gather, %wide.load
  call void @llvm.masked.scatter.v32i8.v32p0i8(<32 x i8> %4, <32 x i8*> %3, i32 1, <32 x i1> <i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true>)
  %index.next = add nuw i64 %index, 32
  %vec.ind.next = add <32 x i64> %vec.ind, <i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32>
  %5 = icmp eq i64 %index.next, 1024
  br i1 %5, label %for.cond.cleanup, label %vector.body

for.cond.cleanup:                                 ; preds = %vector.body
  ret void
}

define void @scatter_masked(i8* noalias nocapture %A, i8* noalias nocapture readonly %B, <32 x i8> %maskedoff) {
; CHECK-LABEL: @scatter_masked(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    br label [[VECTOR_BODY:%.*]]
; CHECK:       vector.body:
; CHECK-NEXT:    [[INDEX:%.*]] = phi i64 [ 0, [[ENTRY:%.*]] ], [ [[INDEX_NEXT:%.*]], [[VECTOR_BODY]] ]
; CHECK-NEXT:    [[VEC_IND_SCALAR:%.*]] = phi i64 [ 0, [[ENTRY]] ], [ [[VEC_IND_NEXT_SCALAR:%.*]], [[VECTOR_BODY]] ]
; CHECK-NEXT:    [[VEC_IND_SCALAR1:%.*]] = phi i64 [ 0, [[ENTRY]] ], [ [[VEC_IND_NEXT_SCALAR2:%.*]], [[VECTOR_BODY]] ]
; CHECK-NEXT:    [[TMP0:%.*]] = getelementptr inbounds i8, i8* [[B:%.*]], i64 [[INDEX]]
; CHECK-NEXT:    [[TMP1:%.*]] = bitcast i8* [[TMP0]] to <32 x i8>*
; CHECK-NEXT:    [[WIDE_LOAD:%.*]] = load <32 x i8>, <32 x i8>* [[TMP1]], align 1
; CHECK-NEXT:    [[TMP2:%.*]] = getelementptr i8, i8* [[A:%.*]], i64 [[VEC_IND_SCALAR]]
; CHECK-NEXT:    [[TMP3:%.*]] = getelementptr i8, i8* [[A]], i64 [[VEC_IND_SCALAR1]]
; CHECK-NEXT:    [[WIDE_MASKED_GATHER:%.*]] = call <32 x i8> @llvm.riscv.masked.strided.load.v32i8.p0i8.i64(<32 x i8> [[MASKEDOFF:%.*]], i8* [[TMP2]], i64 5, <32 x i1> <i1 true, i1 false, i1 false, i1 true, i1 false, i1 true, i1 true, i1 false, i1 true, i1 true, i1 false, i1 false, i1 true, i1 false, i1 true, i1 false, i1 true, i1 false, i1 true, i1 true, i1 false, i1 true, i1 false, i1 false, i1 false, i1 false, i1 false, i1 false, i1 true, i1 true, i1 true, i1 true>)
; CHECK-NEXT:    [[TMP4:%.*]] = add <32 x i8> [[WIDE_MASKED_GATHER]], [[WIDE_LOAD]]
; CHECK-NEXT:    call void @llvm.riscv.masked.strided.store.v32i8.p0i8.i64(<32 x i8> [[TMP4]], i8* [[TMP3]], i64 5, <32 x i1> <i1 true, i1 false, i1 false, i1 true, i1 false, i1 true, i1 true, i1 false, i1 true, i1 true, i1 false, i1 false, i1 true, i1 false, i1 true, i1 false, i1 true, i1 false, i1 true, i1 true, i1 false, i1 true, i1 false, i1 false, i1 false, i1 false, i1 false, i1 false, i1 true, i1 true, i1 true, i1 true>)
; CHECK-NEXT:    [[INDEX_NEXT]] = add nuw i64 [[INDEX]], 32
; CHECK-NEXT:    [[VEC_IND_NEXT_SCALAR]] = add i64 [[VEC_IND_SCALAR]], 160
; CHECK-NEXT:    [[VEC_IND_NEXT_SCALAR2]] = add i64 [[VEC_IND_SCALAR1]], 160
; CHECK-NEXT:    [[TMP5:%.*]] = icmp eq i64 [[INDEX_NEXT]], 1024
; CHECK-NEXT:    br i1 [[TMP5]], label [[FOR_COND_CLEANUP:%.*]], label [[VECTOR_BODY]]
; CHECK:       for.cond.cleanup:
; CHECK-NEXT:    ret void
;
; CHECK-ASM-LABEL: scatter_masked:
; CHECK-ASM:       # %bb.0: # %entry
; CHECK-ASM-NEXT:    mv a2, zero
; CHECK-ASM-NEXT:    addi a6, zero, 32
; CHECK-ASM-NEXT:    lui a4, 983765
; CHECK-ASM-NEXT:    addiw a4, a4, 873
; CHECK-ASM-NEXT:    vsetivli zero, 1, e32, mf2, ta, mu
; CHECK-ASM-NEXT:    vmv.s.x v0, a4
; CHECK-ASM-NEXT:    addi a4, zero, 5
; CHECK-ASM-NEXT:    addi a5, zero, 1024
; CHECK-ASM-NEXT:  .LBB5_1: # %vector.body
; CHECK-ASM-NEXT:    # =>This Inner Loop Header: Depth=1
; CHECK-ASM-NEXT:    add a3, a1, a2
; CHECK-ASM-NEXT:    vsetvli zero, a6, e8, m1, ta, mu
; CHECK-ASM-NEXT:    vle8.v v25, (a3)
; CHECK-ASM-NEXT:    vmv1r.v v26, v8
; CHECK-ASM-NEXT:    vlse8.v v26, (a0), a4, v0.t
; CHECK-ASM-NEXT:    vadd.vv v25, v26, v25
; CHECK-ASM-NEXT:    vsse8.v v25, (a0), a4, v0.t
; CHECK-ASM-NEXT:    addi a2, a2, 32
; CHECK-ASM-NEXT:    addi a0, a0, 160
; CHECK-ASM-NEXT:    bne a2, a5, .LBB5_1
; CHECK-ASM-NEXT:  # %bb.2: # %for.cond.cleanup
; CHECK-ASM-NEXT:    ret
entry:
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %entry
  %index = phi i64 [ 0, %entry ], [ %index.next, %vector.body ]
  %vec.ind = phi <32 x i64> [ <i64 0, i64 1, i64 2, i64 3, i64 4, i64 5, i64 6, i64 7, i64 8, i64 9, i64 10, i64 11, i64 12, i64 13, i64 14, i64 15, i64 16, i64 17, i64 18, i64 19, i64 20, i64 21, i64 22, i64 23, i64 24, i64 25, i64 26, i64 27, i64 28, i64 29, i64 30, i64 31>, %entry ], [ %vec.ind.next, %vector.body ]
  %0 = getelementptr inbounds i8, i8* %B, i64 %index
  %1 = bitcast i8* %0 to <32 x i8>*
  %wide.load = load <32 x i8>, <32 x i8>* %1, align 1
  %2 = mul nuw nsw <32 x i64> %vec.ind, <i64 5, i64 5, i64 5, i64 5, i64 5, i64 5, i64 5, i64 5, i64 5, i64 5, i64 5, i64 5, i64 5, i64 5, i64 5, i64 5, i64 5, i64 5, i64 5, i64 5, i64 5, i64 5, i64 5, i64 5, i64 5, i64 5, i64 5, i64 5, i64 5, i64 5, i64 5, i64 5>
  %3 = getelementptr inbounds i8, i8* %A, <32 x i64> %2
  %wide.masked.gather = call <32 x i8> @llvm.masked.gather.v32i8.v32p0i8(<32 x i8*> %3, i32 1, <32 x i1> <i1 true, i1 false, i1 false, i1 true, i1 false, i1 true, i1 true, i1 false, i1 true, i1 true, i1 false, i1 false, i1 true, i1 false, i1 true, i1 false, i1 true, i1 false, i1 true, i1 true, i1 false, i1 true, i1 false, i1 false, i1 false, i1 false, i1 false, i1 false, i1 true, i1 true, i1 true, i1 true>, <32 x i8> %maskedoff)
  %4 = add <32 x i8> %wide.masked.gather, %wide.load
  call void @llvm.masked.scatter.v32i8.v32p0i8(<32 x i8> %4, <32 x i8*> %3, i32 1, <32 x i1> <i1 true, i1 false, i1 false, i1 true, i1 false, i1 true, i1 true, i1 false, i1 true, i1 true, i1 false, i1 false, i1 true, i1 false, i1 true, i1 false, i1 true, i1 false, i1 true, i1 true, i1 false, i1 true, i1 false, i1 false, i1 false, i1 false, i1 false, i1 false, i1 true, i1 true, i1 true, i1 true>)
  %index.next = add nuw i64 %index, 32
  %vec.ind.next = add <32 x i64> %vec.ind, <i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32>
  %5 = icmp eq i64 %index.next, 1024
  br i1 %5, label %for.cond.cleanup, label %vector.body

for.cond.cleanup:                                 ; preds = %vector.body
  ret void
}

; void gather_pow2(signed char * __restrict  A, signed char * __restrict B) {
;   for (int i = 0; i != 1024; ++i)
;       A[i] += B[i * 4];
; }
define void @gather_pow2(i32* noalias nocapture %A, i32* noalias nocapture readonly %B) {
; CHECK-LABEL: @gather_pow2(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    br label [[VECTOR_BODY:%.*]]
; CHECK:       vector.body:
; CHECK-NEXT:    [[INDEX:%.*]] = phi i64 [ 0, [[ENTRY:%.*]] ], [ [[INDEX_NEXT:%.*]], [[VECTOR_BODY]] ]
; CHECK-NEXT:    [[VEC_IND_SCALAR:%.*]] = phi i64 [ 0, [[ENTRY]] ], [ [[VEC_IND_NEXT_SCALAR:%.*]], [[VECTOR_BODY]] ]
; CHECK-NEXT:    [[TMP0:%.*]] = getelementptr i32, i32* [[B:%.*]], i64 [[VEC_IND_SCALAR]]
; CHECK-NEXT:    [[TMP1:%.*]] = bitcast i32* [[TMP0]] to i8*
; CHECK-NEXT:    [[WIDE_MASKED_GATHER:%.*]] = call <8 x i32> @llvm.riscv.masked.strided.load.v8i32.p0i8.i64(<8 x i32> undef, i8* [[TMP1]], i64 16, <8 x i1> <i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true>)
; CHECK-NEXT:    [[TMP2:%.*]] = getelementptr inbounds i32, i32* [[A:%.*]], i64 [[INDEX]]
; CHECK-NEXT:    [[TMP3:%.*]] = bitcast i32* [[TMP2]] to <8 x i32>*
; CHECK-NEXT:    [[WIDE_LOAD:%.*]] = load <8 x i32>, <8 x i32>* [[TMP3]], align 1
; CHECK-NEXT:    [[TMP4:%.*]] = add <8 x i32> [[WIDE_LOAD]], [[WIDE_MASKED_GATHER]]
; CHECK-NEXT:    [[TMP5:%.*]] = bitcast i32* [[TMP2]] to <8 x i32>*
; CHECK-NEXT:    store <8 x i32> [[TMP4]], <8 x i32>* [[TMP5]], align 1
; CHECK-NEXT:    [[INDEX_NEXT]] = add nuw i64 [[INDEX]], 8
; CHECK-NEXT:    [[VEC_IND_NEXT_SCALAR]] = add i64 [[VEC_IND_SCALAR]], 32
; CHECK-NEXT:    [[TMP6:%.*]] = icmp eq i64 [[INDEX_NEXT]], 1024
; CHECK-NEXT:    br i1 [[TMP6]], label [[FOR_COND_CLEANUP:%.*]], label [[VECTOR_BODY]]
; CHECK:       for.cond.cleanup:
; CHECK-NEXT:    ret void
;
; CHECK-ASM-LABEL: gather_pow2:
; CHECK-ASM:       # %bb.0: # %entry
; CHECK-ASM-NEXT:    addi a2, zero, 1024
; CHECK-ASM-NEXT:    addi a3, zero, 16
; CHECK-ASM-NEXT:    addi a4, zero, 32
; CHECK-ASM-NEXT:  .LBB6_1: # %vector.body
; CHECK-ASM-NEXT:    # =>This Inner Loop Header: Depth=1
; CHECK-ASM-NEXT:    vsetivli zero, 8, e32, m1, ta, mu
; CHECK-ASM-NEXT:    vlse32.v v25, (a1), a3
; CHECK-ASM-NEXT:    vsetvli zero, a4, e8, m1, ta, mu
; CHECK-ASM-NEXT:    vle8.v v26, (a0)
; CHECK-ASM-NEXT:    vsetivli zero, 8, e32, m1, ta, mu
; CHECK-ASM-NEXT:    vadd.vv v25, v26, v25
; CHECK-ASM-NEXT:    vsetvli zero, a4, e8, m1, ta, mu
; CHECK-ASM-NEXT:    vse8.v v25, (a0)
; CHECK-ASM-NEXT:    addi a2, a2, -8
; CHECK-ASM-NEXT:    addi a0, a0, 32
; CHECK-ASM-NEXT:    addi a1, a1, 128
; CHECK-ASM-NEXT:    bnez a2, .LBB6_1
; CHECK-ASM-NEXT:  # %bb.2: # %for.cond.cleanup
; CHECK-ASM-NEXT:    ret
entry:
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %entry
  %index = phi i64 [ 0, %entry ], [ %index.next, %vector.body ]
  %vec.ind = phi <8 x i64> [ <i64 0, i64 1, i64 2, i64 3, i64 4, i64 5, i64 6, i64 7>, %entry ], [ %vec.ind.next, %vector.body ]
  %0 = shl nsw <8 x i64> %vec.ind, <i64 2, i64 2, i64 2, i64 2, i64 2, i64 2, i64 2, i64 2>
  %1 = getelementptr inbounds i32, i32* %B, <8 x i64> %0
  %wide.masked.gather = call <8 x i32> @llvm.masked.gather.v8i32.v8p0i32(<8 x i32*> %1, i32 4, <8 x i1> <i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true>, <8 x i32> undef)
  %2 = getelementptr inbounds i32, i32* %A, i64 %index
  %3 = bitcast i32* %2 to <8 x i32>*
  %wide.load = load <8 x i32>, <8 x i32>* %3, align 1
  %4 = add <8 x i32> %wide.load, %wide.masked.gather
  %5 = bitcast i32* %2 to <8 x i32>*
  store <8 x i32> %4, <8 x i32>* %5, align 1
  %index.next = add nuw i64 %index, 8
  %vec.ind.next = add <8 x i64> %vec.ind, <i64 8, i64 8, i64 8, i64 8, i64 8, i64 8, i64 8, i64 8>
  %6 = icmp eq i64 %index.next, 1024
  br i1 %6, label %for.cond.cleanup, label %vector.body

for.cond.cleanup:                                 ; preds = %vector.body
  ret void
}

;void scatter_pow2(signed char * __restrict  A, signed char * __restrict B) {
;  for (int i = 0; i < 1024; ++i)
;      A[i * 4] += B[i];
;}
define void @scatter_pow2(i32* noalias nocapture %A, i32* noalias nocapture readonly %B) {
; CHECK-LABEL: @scatter_pow2(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    br label [[VECTOR_BODY:%.*]]
; CHECK:       vector.body:
; CHECK-NEXT:    [[INDEX:%.*]] = phi i64 [ 0, [[ENTRY:%.*]] ], [ [[INDEX_NEXT:%.*]], [[VECTOR_BODY]] ]
; CHECK-NEXT:    [[VEC_IND_SCALAR:%.*]] = phi i64 [ 0, [[ENTRY]] ], [ [[VEC_IND_NEXT_SCALAR:%.*]], [[VECTOR_BODY]] ]
; CHECK-NEXT:    [[VEC_IND_SCALAR1:%.*]] = phi i64 [ 0, [[ENTRY]] ], [ [[VEC_IND_NEXT_SCALAR2:%.*]], [[VECTOR_BODY]] ]
; CHECK-NEXT:    [[TMP0:%.*]] = getelementptr inbounds i32, i32* [[B:%.*]], i64 [[INDEX]]
; CHECK-NEXT:    [[TMP1:%.*]] = bitcast i32* [[TMP0]] to <8 x i32>*
; CHECK-NEXT:    [[WIDE_LOAD:%.*]] = load <8 x i32>, <8 x i32>* [[TMP1]], align 1
; CHECK-NEXT:    [[TMP2:%.*]] = getelementptr i32, i32* [[A:%.*]], i64 [[VEC_IND_SCALAR]]
; CHECK-NEXT:    [[TMP3:%.*]] = bitcast i32* [[TMP2]] to i8*
; CHECK-NEXT:    [[TMP4:%.*]] = getelementptr i32, i32* [[A]], i64 [[VEC_IND_SCALAR1]]
; CHECK-NEXT:    [[TMP5:%.*]] = bitcast i32* [[TMP4]] to i8*
; CHECK-NEXT:    [[WIDE_MASKED_GATHER:%.*]] = call <8 x i32> @llvm.riscv.masked.strided.load.v8i32.p0i8.i64(<8 x i32> undef, i8* [[TMP3]], i64 16, <8 x i1> <i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true>)
; CHECK-NEXT:    [[TMP6:%.*]] = add <8 x i32> [[WIDE_MASKED_GATHER]], [[WIDE_LOAD]]
; CHECK-NEXT:    call void @llvm.riscv.masked.strided.store.v8i32.p0i8.i64(<8 x i32> [[TMP6]], i8* [[TMP5]], i64 16, <8 x i1> <i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true>)
; CHECK-NEXT:    [[INDEX_NEXT]] = add nuw i64 [[INDEX]], 8
; CHECK-NEXT:    [[VEC_IND_NEXT_SCALAR]] = add i64 [[VEC_IND_SCALAR]], 32
; CHECK-NEXT:    [[VEC_IND_NEXT_SCALAR2]] = add i64 [[VEC_IND_SCALAR1]], 32
; CHECK-NEXT:    [[TMP7:%.*]] = icmp eq i64 [[INDEX_NEXT]], 1024
; CHECK-NEXT:    br i1 [[TMP7]], label [[FOR_COND_CLEANUP:%.*]], label [[VECTOR_BODY]]
; CHECK:       for.cond.cleanup:
; CHECK-NEXT:    ret void
;
; CHECK-ASM-LABEL: scatter_pow2:
; CHECK-ASM:       # %bb.0: # %entry
; CHECK-ASM-NEXT:    addi a2, zero, 1024
; CHECK-ASM-NEXT:    addi a3, zero, 32
; CHECK-ASM-NEXT:    addi a4, zero, 16
; CHECK-ASM-NEXT:  .LBB7_1: # %vector.body
; CHECK-ASM-NEXT:    # =>This Inner Loop Header: Depth=1
; CHECK-ASM-NEXT:    vsetvli zero, a3, e8, m1, ta, mu
; CHECK-ASM-NEXT:    vle8.v v25, (a1)
; CHECK-ASM-NEXT:    vsetivli zero, 8, e32, m1, ta, mu
; CHECK-ASM-NEXT:    vlse32.v v26, (a0), a4
; CHECK-ASM-NEXT:    vadd.vv v25, v26, v25
; CHECK-ASM-NEXT:    vsse32.v v25, (a0), a4
; CHECK-ASM-NEXT:    addi a2, a2, -8
; CHECK-ASM-NEXT:    addi a1, a1, 32
; CHECK-ASM-NEXT:    addi a0, a0, 128
; CHECK-ASM-NEXT:    bnez a2, .LBB7_1
; CHECK-ASM-NEXT:  # %bb.2: # %for.cond.cleanup
; CHECK-ASM-NEXT:    ret
entry:
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %entry
  %index = phi i64 [ 0, %entry ], [ %index.next, %vector.body ]
  %vec.ind = phi <8 x i64> [ <i64 0, i64 1, i64 2, i64 3, i64 4, i64 5, i64 6, i64 7>, %entry ], [ %vec.ind.next, %vector.body ]
  %0 = getelementptr inbounds i32, i32* %B, i64 %index
  %1 = bitcast i32* %0 to <8 x i32>*
  %wide.load = load <8 x i32>, <8 x i32>* %1, align 1
  %2 = shl nuw nsw <8 x i64> %vec.ind, <i64 2, i64 2, i64 2, i64 2, i64 2, i64 2, i64 2, i64 2>
  %3 = getelementptr inbounds i32, i32* %A, <8 x i64> %2
  %wide.masked.gather = call <8 x i32> @llvm.masked.gather.v8i32.v8p0i32(<8 x i32*> %3, i32 4, <8 x i1> <i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true>, <8 x i32> undef)
  %4 = add <8 x i32> %wide.masked.gather, %wide.load
  call void @llvm.masked.scatter.v8i32.v8p0i32(<8 x i32> %4, <8 x i32*> %3, i32 4, <8 x i1> <i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true>)
  %index.next = add nuw i64 %index, 8
  %vec.ind.next = add <8 x i64> %vec.ind, <i64 8, i64 8, i64 8, i64 8, i64 8, i64 8, i64 8, i64 8>
  %5 = icmp eq i64 %index.next, 1024
  br i1 %5, label %for.cond.cleanup, label %vector.body

for.cond.cleanup:                                 ; preds = %vector.body
  ret void
}

;struct foo {
;  int a, b, c, d;
;};
;
;void struct_gather(int * __restrict  A, struct foo * __restrict B) {
;  for (int i = 0; i < 1024; ++i)
;      A[i] += B[i].b;
;}
define void @struct_gather(i32* noalias nocapture %A, %struct.foo* noalias nocapture readonly %B) {
; CHECK-LABEL: @struct_gather(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    br label [[VECTOR_BODY:%.*]]
; CHECK:       vector.body:
; CHECK-NEXT:    [[INDEX:%.*]] = phi i64 [ 0, [[ENTRY:%.*]] ], [ [[INDEX_NEXT:%.*]], [[VECTOR_BODY]] ]
; CHECK-NEXT:    [[VEC_IND_SCALAR:%.*]] = phi i64 [ 0, [[ENTRY]] ], [ [[VEC_IND_NEXT_SCALAR:%.*]], [[VECTOR_BODY]] ]
; CHECK-NEXT:    [[VEC_IND_SCALAR1:%.*]] = phi i64 [ 8, [[ENTRY]] ], [ [[VEC_IND_NEXT_SCALAR2:%.*]], [[VECTOR_BODY]] ]
; CHECK-NEXT:    [[TMP0:%.*]] = getelementptr [[STRUCT_FOO:%.*]], %struct.foo* [[B:%.*]], i64 [[VEC_IND_SCALAR]], i32 1
; CHECK-NEXT:    [[TMP1:%.*]] = bitcast i32* [[TMP0]] to i8*
; CHECK-NEXT:    [[TMP2:%.*]] = getelementptr [[STRUCT_FOO]], %struct.foo* [[B]], i64 [[VEC_IND_SCALAR1]], i32 1
; CHECK-NEXT:    [[TMP3:%.*]] = bitcast i32* [[TMP2]] to i8*
; CHECK-NEXT:    [[WIDE_MASKED_GATHER:%.*]] = call <8 x i32> @llvm.riscv.masked.strided.load.v8i32.p0i8.i64(<8 x i32> undef, i8* [[TMP1]], i64 16, <8 x i1> <i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true>)
; CHECK-NEXT:    [[WIDE_MASKED_GATHER9:%.*]] = call <8 x i32> @llvm.riscv.masked.strided.load.v8i32.p0i8.i64(<8 x i32> undef, i8* [[TMP3]], i64 16, <8 x i1> <i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true>)
; CHECK-NEXT:    [[TMP4:%.*]] = getelementptr inbounds i32, i32* [[A:%.*]], i64 [[INDEX]]
; CHECK-NEXT:    [[TMP5:%.*]] = bitcast i32* [[TMP4]] to <8 x i32>*
; CHECK-NEXT:    [[WIDE_LOAD:%.*]] = load <8 x i32>, <8 x i32>* [[TMP5]], align 4
; CHECK-NEXT:    [[TMP6:%.*]] = getelementptr inbounds i32, i32* [[TMP4]], i64 8
; CHECK-NEXT:    [[TMP7:%.*]] = bitcast i32* [[TMP6]] to <8 x i32>*
; CHECK-NEXT:    [[WIDE_LOAD10:%.*]] = load <8 x i32>, <8 x i32>* [[TMP7]], align 4
; CHECK-NEXT:    [[TMP8:%.*]] = add nsw <8 x i32> [[WIDE_LOAD]], [[WIDE_MASKED_GATHER]]
; CHECK-NEXT:    [[TMP9:%.*]] = add nsw <8 x i32> [[WIDE_LOAD10]], [[WIDE_MASKED_GATHER9]]
; CHECK-NEXT:    [[TMP10:%.*]] = bitcast i32* [[TMP4]] to <8 x i32>*
; CHECK-NEXT:    store <8 x i32> [[TMP8]], <8 x i32>* [[TMP10]], align 4
; CHECK-NEXT:    [[TMP11:%.*]] = bitcast i32* [[TMP6]] to <8 x i32>*
; CHECK-NEXT:    store <8 x i32> [[TMP9]], <8 x i32>* [[TMP11]], align 4
; CHECK-NEXT:    [[INDEX_NEXT]] = add nuw i64 [[INDEX]], 16
; CHECK-NEXT:    [[VEC_IND_NEXT_SCALAR]] = add i64 [[VEC_IND_SCALAR]], 16
; CHECK-NEXT:    [[VEC_IND_NEXT_SCALAR2]] = add i64 [[VEC_IND_SCALAR1]], 16
; CHECK-NEXT:    [[TMP12:%.*]] = icmp eq i64 [[INDEX_NEXT]], 1024
; CHECK-NEXT:    br i1 [[TMP12]], label [[FOR_COND_CLEANUP:%.*]], label [[VECTOR_BODY]]
; CHECK:       for.cond.cleanup:
; CHECK-NEXT:    ret void
;
; CHECK-ASM-LABEL: struct_gather:
; CHECK-ASM:       # %bb.0: # %entry
; CHECK-ASM-NEXT:    addi a0, a0, 32
; CHECK-ASM-NEXT:    addi a1, a1, 132
; CHECK-ASM-NEXT:    addi a2, zero, 1024
; CHECK-ASM-NEXT:    addi a3, zero, 16
; CHECK-ASM-NEXT:  .LBB8_1: # %vector.body
; CHECK-ASM-NEXT:    # =>This Inner Loop Header: Depth=1
; CHECK-ASM-NEXT:    addi a4, a1, -128
; CHECK-ASM-NEXT:    vsetivli zero, 8, e32, m1, ta, mu
; CHECK-ASM-NEXT:    vlse32.v v25, (a4), a3
; CHECK-ASM-NEXT:    vlse32.v v26, (a1), a3
; CHECK-ASM-NEXT:    addi a4, a0, -32
; CHECK-ASM-NEXT:    vle32.v v27, (a4)
; CHECK-ASM-NEXT:    vle32.v v28, (a0)
; CHECK-ASM-NEXT:    vadd.vv v25, v27, v25
; CHECK-ASM-NEXT:    vadd.vv v26, v28, v26
; CHECK-ASM-NEXT:    vse32.v v25, (a4)
; CHECK-ASM-NEXT:    vse32.v v26, (a0)
; CHECK-ASM-NEXT:    addi a2, a2, -16
; CHECK-ASM-NEXT:    addi a0, a0, 64
; CHECK-ASM-NEXT:    addi a1, a1, 256
; CHECK-ASM-NEXT:    bnez a2, .LBB8_1
; CHECK-ASM-NEXT:  # %bb.2: # %for.cond.cleanup
; CHECK-ASM-NEXT:    ret
entry:
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %entry
  %index = phi i64 [ 0, %entry ], [ %index.next, %vector.body ]
  %vec.ind = phi <8 x i64> [ <i64 0, i64 1, i64 2, i64 3, i64 4, i64 5, i64 6, i64 7>, %entry ], [ %vec.ind.next, %vector.body ]
  %step.add = add <8 x i64> %vec.ind, <i64 8, i64 8, i64 8, i64 8, i64 8, i64 8, i64 8, i64 8>
  %0 = getelementptr inbounds %struct.foo, %struct.foo* %B, <8 x i64> %vec.ind, i32 1
  %1 = getelementptr inbounds %struct.foo, %struct.foo* %B, <8 x i64> %step.add, i32 1
  %wide.masked.gather = call <8 x i32> @llvm.masked.gather.v8i32.v8p0i32(<8 x i32*> %0, i32 4, <8 x i1> <i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true>, <8 x i32> undef)
  %wide.masked.gather9 = call <8 x i32> @llvm.masked.gather.v8i32.v8p0i32(<8 x i32*> %1, i32 4, <8 x i1> <i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true>, <8 x i32> undef)
  %2 = getelementptr inbounds i32, i32* %A, i64 %index
  %3 = bitcast i32* %2 to <8 x i32>*
  %wide.load = load <8 x i32>, <8 x i32>* %3, align 4
  %4 = getelementptr inbounds i32, i32* %2, i64 8
  %5 = bitcast i32* %4 to <8 x i32>*
  %wide.load10 = load <8 x i32>, <8 x i32>* %5, align 4
  %6 = add nsw <8 x i32> %wide.load, %wide.masked.gather
  %7 = add nsw <8 x i32> %wide.load10, %wide.masked.gather9
  %8 = bitcast i32* %2 to <8 x i32>*
  store <8 x i32> %6, <8 x i32>* %8, align 4
  %9 = bitcast i32* %4 to <8 x i32>*
  store <8 x i32> %7, <8 x i32>* %9, align 4
  %index.next = add nuw i64 %index, 16
  %vec.ind.next = add <8 x i64> %vec.ind, <i64 16, i64 16, i64 16, i64 16, i64 16, i64 16, i64 16, i64 16>
  %10 = icmp eq i64 %index.next, 1024
  br i1 %10, label %for.cond.cleanup, label %vector.body

for.cond.cleanup:                                 ; preds = %vector.body
  ret void
}

;void gather_unroll(int * __restrict  A, int * __restrict B) {
;  for (int i = 0; i < 1024; i+= 4 ) {
;    A[i] += B[i * 4];
;    A[i+1] += B[(i+1) * 4];
;    A[i+2] += B[(i+2) * 4];
;    A[i+3] += B[(i+3) * 4];
;  }
;}
define void @gather_unroll(i32* noalias nocapture %A, i32* noalias nocapture readonly %B) {
; CHECK-LABEL: @gather_unroll(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    br label [[VECTOR_BODY:%.*]]
; CHECK:       vector.body:
; CHECK-NEXT:    [[INDEX:%.*]] = phi i64 [ 0, [[ENTRY:%.*]] ], [ [[INDEX_NEXT:%.*]], [[VECTOR_BODY]] ]
; CHECK-NEXT:    [[VEC_IND_SCALAR:%.*]] = phi i64 [ 0, [[ENTRY]] ], [ [[VEC_IND_NEXT_SCALAR:%.*]], [[VECTOR_BODY]] ]
; CHECK-NEXT:    [[VEC_IND_SCALAR1:%.*]] = phi i64 [ 0, [[ENTRY]] ], [ [[VEC_IND_NEXT_SCALAR2:%.*]], [[VECTOR_BODY]] ]
; CHECK-NEXT:    [[VEC_IND_SCALAR3:%.*]] = phi i64 [ 4, [[ENTRY]] ], [ [[VEC_IND_NEXT_SCALAR4:%.*]], [[VECTOR_BODY]] ]
; CHECK-NEXT:    [[VEC_IND_SCALAR5:%.*]] = phi i64 [ 1, [[ENTRY]] ], [ [[VEC_IND_NEXT_SCALAR6:%.*]], [[VECTOR_BODY]] ]
; CHECK-NEXT:    [[VEC_IND_SCALAR7:%.*]] = phi i64 [ 8, [[ENTRY]] ], [ [[VEC_IND_NEXT_SCALAR8:%.*]], [[VECTOR_BODY]] ]
; CHECK-NEXT:    [[VEC_IND_SCALAR9:%.*]] = phi i64 [ 2, [[ENTRY]] ], [ [[VEC_IND_NEXT_SCALAR10:%.*]], [[VECTOR_BODY]] ]
; CHECK-NEXT:    [[VEC_IND_SCALAR11:%.*]] = phi i64 [ 12, [[ENTRY]] ], [ [[VEC_IND_NEXT_SCALAR12:%.*]], [[VECTOR_BODY]] ]
; CHECK-NEXT:    [[VEC_IND_SCALAR13:%.*]] = phi i64 [ 3, [[ENTRY]] ], [ [[VEC_IND_NEXT_SCALAR14:%.*]], [[VECTOR_BODY]] ]
; CHECK-NEXT:    [[VEC_IND_SCALAR15:%.*]] = phi i64 [ 0, [[ENTRY]] ], [ [[VEC_IND_NEXT_SCALAR16:%.*]], [[VECTOR_BODY]] ]
; CHECK-NEXT:    [[VEC_IND_SCALAR17:%.*]] = phi i64 [ 1, [[ENTRY]] ], [ [[VEC_IND_NEXT_SCALAR18:%.*]], [[VECTOR_BODY]] ]
; CHECK-NEXT:    [[VEC_IND_SCALAR19:%.*]] = phi i64 [ 2, [[ENTRY]] ], [ [[VEC_IND_NEXT_SCALAR20:%.*]], [[VECTOR_BODY]] ]
; CHECK-NEXT:    [[VEC_IND_SCALAR21:%.*]] = phi i64 [ 3, [[ENTRY]] ], [ [[VEC_IND_NEXT_SCALAR22:%.*]], [[VECTOR_BODY]] ]
; CHECK-NEXT:    [[TMP0:%.*]] = getelementptr i32, i32* [[B:%.*]], i64 [[VEC_IND_SCALAR]]
; CHECK-NEXT:    [[TMP1:%.*]] = bitcast i32* [[TMP0]] to i8*
; CHECK-NEXT:    [[WIDE_MASKED_GATHER:%.*]] = call <8 x i32> @llvm.riscv.masked.strided.load.v8i32.p0i8.i64(<8 x i32> undef, i8* [[TMP1]], i64 64, <8 x i1> <i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true>)
; CHECK-NEXT:    [[TMP2:%.*]] = getelementptr i32, i32* [[A:%.*]], i64 [[VEC_IND_SCALAR1]]
; CHECK-NEXT:    [[TMP3:%.*]] = bitcast i32* [[TMP2]] to i8*
; CHECK-NEXT:    [[TMP4:%.*]] = getelementptr i32, i32* [[A]], i64 [[VEC_IND_SCALAR15]]
; CHECK-NEXT:    [[TMP5:%.*]] = bitcast i32* [[TMP4]] to i8*
; CHECK-NEXT:    [[WIDE_MASKED_GATHER52:%.*]] = call <8 x i32> @llvm.riscv.masked.strided.load.v8i32.p0i8.i64(<8 x i32> undef, i8* [[TMP3]], i64 16, <8 x i1> <i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true>)
; CHECK-NEXT:    [[TMP6:%.*]] = add nsw <8 x i32> [[WIDE_MASKED_GATHER52]], [[WIDE_MASKED_GATHER]]
; CHECK-NEXT:    call void @llvm.riscv.masked.strided.store.v8i32.p0i8.i64(<8 x i32> [[TMP6]], i8* [[TMP5]], i64 16, <8 x i1> <i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true>)
; CHECK-NEXT:    [[TMP7:%.*]] = getelementptr i32, i32* [[B]], i64 [[VEC_IND_SCALAR3]]
; CHECK-NEXT:    [[TMP8:%.*]] = bitcast i32* [[TMP7]] to i8*
; CHECK-NEXT:    [[WIDE_MASKED_GATHER53:%.*]] = call <8 x i32> @llvm.riscv.masked.strided.load.v8i32.p0i8.i64(<8 x i32> undef, i8* [[TMP8]], i64 64, <8 x i1> <i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true>)
; CHECK-NEXT:    [[TMP9:%.*]] = getelementptr i32, i32* [[A]], i64 [[VEC_IND_SCALAR5]]
; CHECK-NEXT:    [[TMP10:%.*]] = bitcast i32* [[TMP9]] to i8*
; CHECK-NEXT:    [[TMP11:%.*]] = getelementptr i32, i32* [[A]], i64 [[VEC_IND_SCALAR17]]
; CHECK-NEXT:    [[TMP12:%.*]] = bitcast i32* [[TMP11]] to i8*
; CHECK-NEXT:    [[WIDE_MASKED_GATHER54:%.*]] = call <8 x i32> @llvm.riscv.masked.strided.load.v8i32.p0i8.i64(<8 x i32> undef, i8* [[TMP10]], i64 16, <8 x i1> <i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true>)
; CHECK-NEXT:    [[TMP13:%.*]] = add nsw <8 x i32> [[WIDE_MASKED_GATHER54]], [[WIDE_MASKED_GATHER53]]
; CHECK-NEXT:    call void @llvm.riscv.masked.strided.store.v8i32.p0i8.i64(<8 x i32> [[TMP13]], i8* [[TMP12]], i64 16, <8 x i1> <i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true>)
; CHECK-NEXT:    [[TMP14:%.*]] = getelementptr i32, i32* [[B]], i64 [[VEC_IND_SCALAR7]]
; CHECK-NEXT:    [[TMP15:%.*]] = bitcast i32* [[TMP14]] to i8*
; CHECK-NEXT:    [[WIDE_MASKED_GATHER55:%.*]] = call <8 x i32> @llvm.riscv.masked.strided.load.v8i32.p0i8.i64(<8 x i32> undef, i8* [[TMP15]], i64 64, <8 x i1> <i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true>)
; CHECK-NEXT:    [[TMP16:%.*]] = getelementptr i32, i32* [[A]], i64 [[VEC_IND_SCALAR9]]
; CHECK-NEXT:    [[TMP17:%.*]] = bitcast i32* [[TMP16]] to i8*
; CHECK-NEXT:    [[TMP18:%.*]] = getelementptr i32, i32* [[A]], i64 [[VEC_IND_SCALAR19]]
; CHECK-NEXT:    [[TMP19:%.*]] = bitcast i32* [[TMP18]] to i8*
; CHECK-NEXT:    [[WIDE_MASKED_GATHER56:%.*]] = call <8 x i32> @llvm.riscv.masked.strided.load.v8i32.p0i8.i64(<8 x i32> undef, i8* [[TMP17]], i64 16, <8 x i1> <i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true>)
; CHECK-NEXT:    [[TMP20:%.*]] = add nsw <8 x i32> [[WIDE_MASKED_GATHER56]], [[WIDE_MASKED_GATHER55]]
; CHECK-NEXT:    call void @llvm.riscv.masked.strided.store.v8i32.p0i8.i64(<8 x i32> [[TMP20]], i8* [[TMP19]], i64 16, <8 x i1> <i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true>)
; CHECK-NEXT:    [[TMP21:%.*]] = getelementptr i32, i32* [[B]], i64 [[VEC_IND_SCALAR11]]
; CHECK-NEXT:    [[TMP22:%.*]] = bitcast i32* [[TMP21]] to i8*
; CHECK-NEXT:    [[WIDE_MASKED_GATHER57:%.*]] = call <8 x i32> @llvm.riscv.masked.strided.load.v8i32.p0i8.i64(<8 x i32> undef, i8* [[TMP22]], i64 64, <8 x i1> <i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true>)
; CHECK-NEXT:    [[TMP23:%.*]] = getelementptr i32, i32* [[A]], i64 [[VEC_IND_SCALAR13]]
; CHECK-NEXT:    [[TMP24:%.*]] = bitcast i32* [[TMP23]] to i8*
; CHECK-NEXT:    [[TMP25:%.*]] = getelementptr i32, i32* [[A]], i64 [[VEC_IND_SCALAR21]]
; CHECK-NEXT:    [[TMP26:%.*]] = bitcast i32* [[TMP25]] to i8*
; CHECK-NEXT:    [[WIDE_MASKED_GATHER58:%.*]] = call <8 x i32> @llvm.riscv.masked.strided.load.v8i32.p0i8.i64(<8 x i32> undef, i8* [[TMP24]], i64 16, <8 x i1> <i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true>)
; CHECK-NEXT:    [[TMP27:%.*]] = add nsw <8 x i32> [[WIDE_MASKED_GATHER58]], [[WIDE_MASKED_GATHER57]]
; CHECK-NEXT:    call void @llvm.riscv.masked.strided.store.v8i32.p0i8.i64(<8 x i32> [[TMP27]], i8* [[TMP26]], i64 16, <8 x i1> <i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true>)
; CHECK-NEXT:    [[INDEX_NEXT]] = add nuw i64 [[INDEX]], 8
; CHECK-NEXT:    [[VEC_IND_NEXT_SCALAR]] = add i64 [[VEC_IND_SCALAR]], 128
; CHECK-NEXT:    [[VEC_IND_NEXT_SCALAR2]] = add i64 [[VEC_IND_SCALAR1]], 32
; CHECK-NEXT:    [[VEC_IND_NEXT_SCALAR4]] = add i64 [[VEC_IND_SCALAR3]], 128
; CHECK-NEXT:    [[VEC_IND_NEXT_SCALAR6]] = add i64 [[VEC_IND_SCALAR5]], 32
; CHECK-NEXT:    [[VEC_IND_NEXT_SCALAR8]] = add i64 [[VEC_IND_SCALAR7]], 128
; CHECK-NEXT:    [[VEC_IND_NEXT_SCALAR10]] = add i64 [[VEC_IND_SCALAR9]], 32
; CHECK-NEXT:    [[VEC_IND_NEXT_SCALAR12]] = add i64 [[VEC_IND_SCALAR11]], 128
; CHECK-NEXT:    [[VEC_IND_NEXT_SCALAR14]] = add i64 [[VEC_IND_SCALAR13]], 32
; CHECK-NEXT:    [[VEC_IND_NEXT_SCALAR16]] = add i64 [[VEC_IND_SCALAR15]], 32
; CHECK-NEXT:    [[VEC_IND_NEXT_SCALAR18]] = add i64 [[VEC_IND_SCALAR17]], 32
; CHECK-NEXT:    [[VEC_IND_NEXT_SCALAR20]] = add i64 [[VEC_IND_SCALAR19]], 32
; CHECK-NEXT:    [[VEC_IND_NEXT_SCALAR22]] = add i64 [[VEC_IND_SCALAR21]], 32
; CHECK-NEXT:    [[TMP28:%.*]] = icmp eq i64 [[INDEX_NEXT]], 256
; CHECK-NEXT:    br i1 [[TMP28]], label [[FOR_COND_CLEANUP:%.*]], label [[VECTOR_BODY]]
; CHECK:       for.cond.cleanup:
; CHECK-NEXT:    ret void
;
; CHECK-ASM-LABEL: gather_unroll:
; CHECK-ASM:       # %bb.0: # %entry
; CHECK-ASM-NEXT:    addi a2, zero, 256
; CHECK-ASM-NEXT:    addi a3, zero, 64
; CHECK-ASM-NEXT:    addi a4, zero, 16
; CHECK-ASM-NEXT:  .LBB9_1: # %vector.body
; CHECK-ASM-NEXT:    # =>This Inner Loop Header: Depth=1
; CHECK-ASM-NEXT:    vsetivli zero, 8, e32, m1, ta, mu
; CHECK-ASM-NEXT:    vlse32.v v25, (a1), a3
; CHECK-ASM-NEXT:    vlse32.v v26, (a0), a4
; CHECK-ASM-NEXT:    vadd.vv v25, v26, v25
; CHECK-ASM-NEXT:    vsse32.v v25, (a0), a4
; CHECK-ASM-NEXT:    addi a5, a1, 16
; CHECK-ASM-NEXT:    vlse32.v v25, (a5), a3
; CHECK-ASM-NEXT:    addi a5, a0, 4
; CHECK-ASM-NEXT:    vlse32.v v26, (a5), a4
; CHECK-ASM-NEXT:    vadd.vv v25, v26, v25
; CHECK-ASM-NEXT:    vsse32.v v25, (a5), a4
; CHECK-ASM-NEXT:    addi a5, a1, 32
; CHECK-ASM-NEXT:    vlse32.v v25, (a5), a3
; CHECK-ASM-NEXT:    addi a5, a0, 8
; CHECK-ASM-NEXT:    vlse32.v v26, (a5), a4
; CHECK-ASM-NEXT:    vadd.vv v25, v26, v25
; CHECK-ASM-NEXT:    vsse32.v v25, (a5), a4
; CHECK-ASM-NEXT:    addi a5, a1, 48
; CHECK-ASM-NEXT:    vlse32.v v25, (a5), a3
; CHECK-ASM-NEXT:    addi a5, a0, 12
; CHECK-ASM-NEXT:    vlse32.v v26, (a5), a4
; CHECK-ASM-NEXT:    vadd.vv v25, v26, v25
; CHECK-ASM-NEXT:    vsse32.v v25, (a5), a4
; CHECK-ASM-NEXT:    addi a2, a2, -8
; CHECK-ASM-NEXT:    addi a1, a1, 512
; CHECK-ASM-NEXT:    addi a0, a0, 128
; CHECK-ASM-NEXT:    bnez a2, .LBB9_1
; CHECK-ASM-NEXT:  # %bb.2: # %for.cond.cleanup
; CHECK-ASM-NEXT:    ret
entry:
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %entry
  %index = phi i64 [ 0, %entry ], [ %index.next, %vector.body ]
  %vec.ind = phi <8 x i64> [ <i64 0, i64 4, i64 8, i64 12, i64 16, i64 20, i64 24, i64 28>, %entry ], [ %vec.ind.next, %vector.body ]
  %0 = shl nuw nsw <8 x i64> %vec.ind, <i64 2, i64 2, i64 2, i64 2, i64 2, i64 2, i64 2, i64 2>
  %1 = getelementptr inbounds i32, i32* %B, <8 x i64> %0
  %wide.masked.gather = call <8 x i32> @llvm.masked.gather.v8i32.v8p0i32(<8 x i32*> %1, i32 4, <8 x i1> <i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true>, <8 x i32> undef)
  %2 = getelementptr inbounds i32, i32* %A, <8 x i64> %vec.ind
  %wide.masked.gather52 = call <8 x i32> @llvm.masked.gather.v8i32.v8p0i32(<8 x i32*> %2, i32 4, <8 x i1> <i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true>, <8 x i32> undef)
  %3 = add nsw <8 x i32> %wide.masked.gather52, %wide.masked.gather
  call void @llvm.masked.scatter.v8i32.v8p0i32(<8 x i32> %3, <8 x i32*> %2, i32 4, <8 x i1> <i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true>)
  %4 = or <8 x i64> %vec.ind, <i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1>
  %5 = shl nsw <8 x i64> %4, <i64 2, i64 2, i64 2, i64 2, i64 2, i64 2, i64 2, i64 2>
  %6 = getelementptr inbounds i32, i32* %B, <8 x i64> %5
  %wide.masked.gather53 = call <8 x i32> @llvm.masked.gather.v8i32.v8p0i32(<8 x i32*> %6, i32 4, <8 x i1> <i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true>, <8 x i32> undef)
  %7 = getelementptr inbounds i32, i32* %A, <8 x i64> %4
  %wide.masked.gather54 = call <8 x i32> @llvm.masked.gather.v8i32.v8p0i32(<8 x i32*> %7, i32 4, <8 x i1> <i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true>, <8 x i32> undef)
  %8 = add nsw <8 x i32> %wide.masked.gather54, %wide.masked.gather53
  call void @llvm.masked.scatter.v8i32.v8p0i32(<8 x i32> %8, <8 x i32*> %7, i32 4, <8 x i1> <i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true>)
  %9 = or <8 x i64> %vec.ind, <i64 2, i64 2, i64 2, i64 2, i64 2, i64 2, i64 2, i64 2>
  %10 = shl nsw <8 x i64> %9, <i64 2, i64 2, i64 2, i64 2, i64 2, i64 2, i64 2, i64 2>
  %11 = getelementptr inbounds i32, i32* %B, <8 x i64> %10
  %wide.masked.gather55 = call <8 x i32> @llvm.masked.gather.v8i32.v8p0i32(<8 x i32*> %11, i32 4, <8 x i1> <i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true>, <8 x i32> undef)
  %12 = getelementptr inbounds i32, i32* %A, <8 x i64> %9
  %wide.masked.gather56 = call <8 x i32> @llvm.masked.gather.v8i32.v8p0i32(<8 x i32*> %12, i32 4, <8 x i1> <i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true>, <8 x i32> undef)
  %13 = add nsw <8 x i32> %wide.masked.gather56, %wide.masked.gather55
  call void @llvm.masked.scatter.v8i32.v8p0i32(<8 x i32> %13, <8 x i32*> %12, i32 4, <8 x i1> <i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true>)
  %14 = or <8 x i64> %vec.ind, <i64 3, i64 3, i64 3, i64 3, i64 3, i64 3, i64 3, i64 3>
  %15 = shl nsw <8 x i64> %14, <i64 2, i64 2, i64 2, i64 2, i64 2, i64 2, i64 2, i64 2>
  %16 = getelementptr inbounds i32, i32* %B, <8 x i64> %15
  %wide.masked.gather57 = call <8 x i32> @llvm.masked.gather.v8i32.v8p0i32(<8 x i32*> %16, i32 4, <8 x i1> <i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true>, <8 x i32> undef)
  %17 = getelementptr inbounds i32, i32* %A, <8 x i64> %14
  %wide.masked.gather58 = call <8 x i32> @llvm.masked.gather.v8i32.v8p0i32(<8 x i32*> %17, i32 4, <8 x i1> <i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true>, <8 x i32> undef)
  %18 = add nsw <8 x i32> %wide.masked.gather58, %wide.masked.gather57
  call void @llvm.masked.scatter.v8i32.v8p0i32(<8 x i32> %18, <8 x i32*> %17, i32 4, <8 x i1> <i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true>)
  %index.next = add nuw i64 %index, 8
  %vec.ind.next = add <8 x i64> %vec.ind, <i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32>
  %19 = icmp eq i64 %index.next, 256
  br i1 %19, label %for.cond.cleanup, label %vector.body

for.cond.cleanup:                                 ; preds = %vector.body
  ret void
}

declare <32 x i8> @llvm.masked.gather.v32i8.v32p0i8(<32 x i8*>, i32 immarg, <32 x i1>, <32 x i8>)
declare <8 x i32> @llvm.masked.gather.v8i32.v8p0i32(<8 x i32*>, i32 immarg, <8 x i1>, <8 x i32>)
declare void @llvm.masked.scatter.v32i8.v32p0i8(<32 x i8>, <32 x i8*>, i32 immarg, <32 x i1>)
declare void @llvm.masked.scatter.v8i32.v8p0i32(<8 x i32>, <8 x i32*>, i32 immarg, <8 x i1>)

; Make sure we don't crash in getTgtMemIntrinsic for a vector of pointers.
define void @gather_of_pointers(i32** noalias nocapture %0, i32** noalias nocapture readonly %1) {
; CHECK-LABEL: @gather_of_pointers(
; CHECK-NEXT:    br label [[TMP3:%.*]]
; CHECK:       3:
; CHECK-NEXT:    [[TMP4:%.*]] = phi i64 [ 0, [[TMP2:%.*]] ], [ [[TMP15:%.*]], [[TMP3]] ]
; CHECK-NEXT:    [[DOTSCALAR:%.*]] = phi i64 [ 0, [[TMP2]] ], [ [[DOTSCALAR1:%.*]], [[TMP3]] ]
; CHECK-NEXT:    [[DOTSCALAR2:%.*]] = phi i64 [ 10, [[TMP2]] ], [ [[DOTSCALAR3:%.*]], [[TMP3]] ]
; CHECK-NEXT:    [[TMP5:%.*]] = getelementptr i32*, i32** [[TMP1:%.*]], i64 [[DOTSCALAR]]
; CHECK-NEXT:    [[TMP6:%.*]] = bitcast i32** [[TMP5]] to i8*
; CHECK-NEXT:    [[TMP7:%.*]] = getelementptr i32*, i32** [[TMP1]], i64 [[DOTSCALAR2]]
; CHECK-NEXT:    [[TMP8:%.*]] = bitcast i32** [[TMP7]] to i8*
; CHECK-NEXT:    [[TMP9:%.*]] = call <2 x i32*> @llvm.riscv.masked.strided.load.v2p0i32.p0i8.i64(<2 x i32*> undef, i8* [[TMP6]], i64 40, <2 x i1> <i1 true, i1 true>)
; CHECK-NEXT:    [[TMP10:%.*]] = call <2 x i32*> @llvm.riscv.masked.strided.load.v2p0i32.p0i8.i64(<2 x i32*> undef, i8* [[TMP8]], i64 40, <2 x i1> <i1 true, i1 true>)
; CHECK-NEXT:    [[TMP11:%.*]] = getelementptr inbounds i32*, i32** [[TMP0:%.*]], i64 [[TMP4]]
; CHECK-NEXT:    [[TMP12:%.*]] = bitcast i32** [[TMP11]] to <2 x i32*>*
; CHECK-NEXT:    store <2 x i32*> [[TMP9]], <2 x i32*>* [[TMP12]], align 8
; CHECK-NEXT:    [[TMP13:%.*]] = getelementptr inbounds i32*, i32** [[TMP11]], i64 2
; CHECK-NEXT:    [[TMP14:%.*]] = bitcast i32** [[TMP13]] to <2 x i32*>*
; CHECK-NEXT:    store <2 x i32*> [[TMP10]], <2 x i32*>* [[TMP14]], align 8
; CHECK-NEXT:    [[TMP15]] = add nuw i64 [[TMP4]], 4
; CHECK-NEXT:    [[DOTSCALAR1]] = add i64 [[DOTSCALAR]], 20
; CHECK-NEXT:    [[DOTSCALAR3]] = add i64 [[DOTSCALAR2]], 20
; CHECK-NEXT:    [[TMP16:%.*]] = icmp eq i64 [[TMP15]], 1024
; CHECK-NEXT:    br i1 [[TMP16]], label [[TMP17:%.*]], label [[TMP3]]
; CHECK:       17:
; CHECK-NEXT:    ret void
;
; CHECK-ASM-LABEL: gather_of_pointers:
; CHECK-ASM:       # %bb.0:
; CHECK-ASM-NEXT:    addi a0, a0, 16
; CHECK-ASM-NEXT:    addi a2, zero, 1024
; CHECK-ASM-NEXT:    addi a3, zero, 40
; CHECK-ASM-NEXT:  .LBB10_1: # =>This Inner Loop Header: Depth=1
; CHECK-ASM-NEXT:    addi a4, a1, 80
; CHECK-ASM-NEXT:    vsetivli zero, 2, e64, m1, ta, mu
; CHECK-ASM-NEXT:    vlse64.v v25, (a1), a3
; CHECK-ASM-NEXT:    vlse64.v v26, (a4), a3
; CHECK-ASM-NEXT:    addi a4, a0, -16
; CHECK-ASM-NEXT:    vse64.v v25, (a4)
; CHECK-ASM-NEXT:    vse64.v v26, (a0)
; CHECK-ASM-NEXT:    addi a2, a2, -4
; CHECK-ASM-NEXT:    addi a0, a0, 32
; CHECK-ASM-NEXT:    addi a1, a1, 160
; CHECK-ASM-NEXT:    bnez a2, .LBB10_1
; CHECK-ASM-NEXT:  # %bb.2:
; CHECK-ASM-NEXT:    ret
  br label %3

3:                                                ; preds = %3, %2
  %4 = phi i64 [ 0, %2 ], [ %17, %3 ]
  %5 = phi <2 x i64> [ <i64 0, i64 1>, %2 ], [ %18, %3 ]
  %6 = mul nuw nsw <2 x i64> %5, <i64 5, i64 5>
  %7 = mul <2 x i64> %5, <i64 5, i64 5>
  %8 = add <2 x i64> %7, <i64 10, i64 10>
  %9 = getelementptr inbounds i32*, i32** %1, <2 x i64> %6
  %10 = getelementptr inbounds i32*, i32** %1, <2 x i64> %8
  %11 = call <2 x i32*> @llvm.masked.gather.v2p0i32.v2p0p0i32(<2 x i32**> %9, i32 8, <2 x i1> <i1 true, i1 true>, <2 x i32*> undef)
  %12 = call <2 x i32*> @llvm.masked.gather.v2p0i32.v2p0p0i32(<2 x i32**> %10, i32 8, <2 x i1> <i1 true, i1 true>, <2 x i32*> undef)
  %13 = getelementptr inbounds i32*, i32** %0, i64 %4
  %14 = bitcast i32** %13 to <2 x i32*>*
  store <2 x i32*> %11, <2 x i32*>* %14, align 8
  %15 = getelementptr inbounds i32*, i32** %13, i64 2
  %16 = bitcast i32** %15 to <2 x i32*>*
  store <2 x i32*> %12, <2 x i32*>* %16, align 8
  %17 = add nuw i64 %4, 4
  %18 = add <2 x i64> %5, <i64 4, i64 4>
  %19 = icmp eq i64 %17, 1024
  br i1 %19, label %20, label %3

20:                                               ; preds = %3
  ret void
}

declare <2 x i32*> @llvm.masked.gather.v2p0i32.v2p0p0i32(<2 x i32**>, i32 immarg, <2 x i1>, <2 x i32*>)

; Make sure we don't crash in getTgtMemIntrinsic for a vector of pointers.
define void @scatter_of_pointers(i32** noalias nocapture %0, i32** noalias nocapture readonly %1) {
; CHECK-LABEL: @scatter_of_pointers(
; CHECK-NEXT:    br label [[TMP3:%.*]]
; CHECK:       3:
; CHECK-NEXT:    [[TMP4:%.*]] = phi i64 [ 0, [[TMP2:%.*]] ], [ [[TMP15:%.*]], [[TMP3]] ]
; CHECK-NEXT:    [[DOTSCALAR:%.*]] = phi i64 [ 0, [[TMP2]] ], [ [[DOTSCALAR1:%.*]], [[TMP3]] ]
; CHECK-NEXT:    [[DOTSCALAR2:%.*]] = phi i64 [ 10, [[TMP2]] ], [ [[DOTSCALAR3:%.*]], [[TMP3]] ]
; CHECK-NEXT:    [[TMP5:%.*]] = getelementptr inbounds i32*, i32** [[TMP1:%.*]], i64 [[TMP4]]
; CHECK-NEXT:    [[TMP6:%.*]] = bitcast i32** [[TMP5]] to <2 x i32*>*
; CHECK-NEXT:    [[TMP7:%.*]] = load <2 x i32*>, <2 x i32*>* [[TMP6]], align 8
; CHECK-NEXT:    [[TMP8:%.*]] = getelementptr inbounds i32*, i32** [[TMP5]], i64 2
; CHECK-NEXT:    [[TMP9:%.*]] = bitcast i32** [[TMP8]] to <2 x i32*>*
; CHECK-NEXT:    [[TMP10:%.*]] = load <2 x i32*>, <2 x i32*>* [[TMP9]], align 8
; CHECK-NEXT:    [[TMP11:%.*]] = getelementptr i32*, i32** [[TMP0:%.*]], i64 [[DOTSCALAR]]
; CHECK-NEXT:    [[TMP12:%.*]] = bitcast i32** [[TMP11]] to i8*
; CHECK-NEXT:    [[TMP13:%.*]] = getelementptr i32*, i32** [[TMP0]], i64 [[DOTSCALAR2]]
; CHECK-NEXT:    [[TMP14:%.*]] = bitcast i32** [[TMP13]] to i8*
; CHECK-NEXT:    call void @llvm.riscv.masked.strided.store.v2p0i32.p0i8.i64(<2 x i32*> [[TMP7]], i8* [[TMP12]], i64 40, <2 x i1> <i1 true, i1 true>)
; CHECK-NEXT:    call void @llvm.riscv.masked.strided.store.v2p0i32.p0i8.i64(<2 x i32*> [[TMP10]], i8* [[TMP14]], i64 40, <2 x i1> <i1 true, i1 true>)
; CHECK-NEXT:    [[TMP15]] = add nuw i64 [[TMP4]], 4
; CHECK-NEXT:    [[DOTSCALAR1]] = add i64 [[DOTSCALAR]], 20
; CHECK-NEXT:    [[DOTSCALAR3]] = add i64 [[DOTSCALAR2]], 20
; CHECK-NEXT:    [[TMP16:%.*]] = icmp eq i64 [[TMP15]], 1024
; CHECK-NEXT:    br i1 [[TMP16]], label [[TMP17:%.*]], label [[TMP3]]
; CHECK:       17:
; CHECK-NEXT:    ret void
;
; CHECK-ASM-LABEL: scatter_of_pointers:
; CHECK-ASM:       # %bb.0:
; CHECK-ASM-NEXT:    addi a1, a1, 16
; CHECK-ASM-NEXT:    addi a2, zero, 1024
; CHECK-ASM-NEXT:    addi a3, zero, 40
; CHECK-ASM-NEXT:  .LBB11_1: # =>This Inner Loop Header: Depth=1
; CHECK-ASM-NEXT:    addi a4, a1, -16
; CHECK-ASM-NEXT:    vsetivli zero, 2, e64, m1, ta, mu
; CHECK-ASM-NEXT:    vle64.v v25, (a4)
; CHECK-ASM-NEXT:    vle64.v v26, (a1)
; CHECK-ASM-NEXT:    addi a4, a0, 80
; CHECK-ASM-NEXT:    vsse64.v v25, (a0), a3
; CHECK-ASM-NEXT:    vsse64.v v26, (a4), a3
; CHECK-ASM-NEXT:    addi a2, a2, -4
; CHECK-ASM-NEXT:    addi a1, a1, 32
; CHECK-ASM-NEXT:    addi a0, a0, 160
; CHECK-ASM-NEXT:    bnez a2, .LBB11_1
; CHECK-ASM-NEXT:  # %bb.2:
; CHECK-ASM-NEXT:    ret
  br label %3

3:                                                ; preds = %3, %2
  %4 = phi i64 [ 0, %2 ], [ %17, %3 ]
  %5 = phi <2 x i64> [ <i64 0, i64 1>, %2 ], [ %18, %3 ]
  %6 = getelementptr inbounds i32*, i32** %1, i64 %4
  %7 = bitcast i32** %6 to <2 x i32*>*
  %8 = load <2 x i32*>, <2 x i32*>* %7, align 8
  %9 = getelementptr inbounds i32*, i32** %6, i64 2
  %10 = bitcast i32** %9 to <2 x i32*>*
  %11 = load <2 x i32*>, <2 x i32*>* %10, align 8
  %12 = mul nuw nsw <2 x i64> %5, <i64 5, i64 5>
  %13 = mul <2 x i64> %5, <i64 5, i64 5>
  %14 = add <2 x i64> %13, <i64 10, i64 10>
  %15 = getelementptr inbounds i32*, i32** %0, <2 x i64> %12
  %16 = getelementptr inbounds i32*, i32** %0, <2 x i64> %14
  call void @llvm.masked.scatter.v2p0i32.v2p0p0i32(<2 x i32*> %8, <2 x i32**> %15, i32 8, <2 x i1> <i1 true, i1 true>)
  call void @llvm.masked.scatter.v2p0i32.v2p0p0i32(<2 x i32*> %11, <2 x i32**> %16, i32 8, <2 x i1> <i1 true, i1 true>)
  %17 = add nuw i64 %4, 4
  %18 = add <2 x i64> %5, <i64 4, i64 4>
  %19 = icmp eq i64 %17, 1024
  br i1 %19, label %20, label %3

20:                                               ; preds = %3
  ret void
}

declare void @llvm.masked.scatter.v2p0i32.v2p0p0i32(<2 x i32*>, <2 x i32**>, i32 immarg, <2 x i1>)
