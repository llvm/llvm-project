; RUN: opt %s -S -riscv-gather-scatter-lowering -mtriple=riscv64 -mattr=+m,+v -riscv-v-vector-bits-min=256 | FileCheck %s
; RUN: llc < %s -mtriple=riscv64 -mattr=+m,+v -riscv-v-vector-bits-min=256 | FileCheck %s --check-prefix=CHECK-ASM

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
; CHECK-ASM-NEXT:    li a2, 0
; CHECK-ASM-NEXT:    li a3, 32
; CHECK-ASM-NEXT:    li a4, 5
; CHECK-ASM-NEXT:    li a5, 1024
; CHECK-ASM-NEXT:  .LBB0_1: # %vector.body
; CHECK-ASM-NEXT:    # =>This Inner Loop Header: Depth=1
; CHECK-ASM-NEXT:    vsetvli zero, a3, e8, m1, ta, mu
; CHECK-ASM-NEXT:    vlse8.v v8, (a1), a4
; CHECK-ASM-NEXT:    add a6, a0, a2
; CHECK-ASM-NEXT:    vle8.v v9, (a6)
; CHECK-ASM-NEXT:    vadd.vv v8, v9, v8
; CHECK-ASM-NEXT:    vse8.v v8, (a6)
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
; CHECK-ASM-NEXT:    li a2, 0
; CHECK-ASM-NEXT:    lui a3, 983765
; CHECK-ASM-NEXT:    addiw a3, a3, 873
; CHECK-ASM-NEXT:    vsetivli zero, 1, e32, mf2, ta, mu
; CHECK-ASM-NEXT:    vmv.s.x v0, a3
; CHECK-ASM-NEXT:    li a3, 32
; CHECK-ASM-NEXT:    li a4, 5
; CHECK-ASM-NEXT:    li a5, 1024
; CHECK-ASM-NEXT:  .LBB1_1: # %vector.body
; CHECK-ASM-NEXT:    # =>This Inner Loop Header: Depth=1
; CHECK-ASM-NEXT:    vsetvli zero, a3, e8, m1, ta, mu
; CHECK-ASM-NEXT:    vmv1r.v v9, v8
; CHECK-ASM-NEXT:    vlse8.v v9, (a1), a4, v0.t
; CHECK-ASM-NEXT:    add a6, a0, a2
; CHECK-ASM-NEXT:    vle8.v v10, (a6)
; CHECK-ASM-NEXT:    vadd.vv v9, v10, v9
; CHECK-ASM-NEXT:    vse8.v v9, (a6)
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
; CHECK-ASM-NEXT:    li a2, 0
; CHECK-ASM-NEXT:    addi a1, a1, 155
; CHECK-ASM-NEXT:    li a3, 32
; CHECK-ASM-NEXT:    li a4, -5
; CHECK-ASM-NEXT:    li a5, 1024
; CHECK-ASM-NEXT:  .LBB2_1: # %vector.body
; CHECK-ASM-NEXT:    # =>This Inner Loop Header: Depth=1
; CHECK-ASM-NEXT:    vsetvli zero, a3, e8, m1, ta, mu
; CHECK-ASM-NEXT:    vlse8.v v8, (a1), a4
; CHECK-ASM-NEXT:    add a6, a0, a2
; CHECK-ASM-NEXT:    vle8.v v9, (a6)
; CHECK-ASM-NEXT:    vadd.vv v8, v9, v8
; CHECK-ASM-NEXT:    vse8.v v8, (a6)
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
; CHECK-ASM-NEXT:    li a2, 0
; CHECK-ASM-NEXT:    li a3, 32
; CHECK-ASM-NEXT:    li a4, 1024
; CHECK-ASM-NEXT:  .LBB3_1: # %vector.body
; CHECK-ASM-NEXT:    # =>This Inner Loop Header: Depth=1
; CHECK-ASM-NEXT:    vsetvli zero, a3, e8, m1, ta, mu
; CHECK-ASM-NEXT:    vlse8.v v8, (a1), zero
; CHECK-ASM-NEXT:    add a5, a0, a2
; CHECK-ASM-NEXT:    vle8.v v9, (a5)
; CHECK-ASM-NEXT:    vadd.vv v8, v9, v8
; CHECK-ASM-NEXT:    vse8.v v8, (a5)
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
; CHECK-NEXT:    [[TMP0:%.*]] = getelementptr inbounds i8, i8* [[B:%.*]], i64 [[INDEX]]
; CHECK-NEXT:    [[TMP1:%.*]] = bitcast i8* [[TMP0]] to <32 x i8>*
; CHECK-NEXT:    [[WIDE_LOAD:%.*]] = load <32 x i8>, <32 x i8>* [[TMP1]], align 1
; CHECK-NEXT:    [[TMP2:%.*]] = getelementptr i8, i8* [[A:%.*]], i64 [[VEC_IND_SCALAR]]
; CHECK-NEXT:    [[WIDE_MASKED_GATHER:%.*]] = call <32 x i8> @llvm.riscv.masked.strided.load.v32i8.p0i8.i64(<32 x i8> undef, i8* [[TMP2]], i64 5, <32 x i1> <i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true>)
; CHECK-NEXT:    [[TMP3:%.*]] = add <32 x i8> [[WIDE_MASKED_GATHER]], [[WIDE_LOAD]]
; CHECK-NEXT:    call void @llvm.riscv.masked.strided.store.v32i8.p0i8.i64(<32 x i8> [[TMP3]], i8* [[TMP2]], i64 5, <32 x i1> <i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true>)
; CHECK-NEXT:    [[INDEX_NEXT]] = add nuw i64 [[INDEX]], 32
; CHECK-NEXT:    [[VEC_IND_NEXT_SCALAR]] = add i64 [[VEC_IND_SCALAR]], 160
; CHECK-NEXT:    [[TMP4:%.*]] = icmp eq i64 [[INDEX_NEXT]], 1024
; CHECK-NEXT:    br i1 [[TMP4]], label [[FOR_COND_CLEANUP:%.*]], label [[VECTOR_BODY]]
; CHECK:       for.cond.cleanup:
; CHECK-NEXT:    ret void
;
; CHECK-ASM-LABEL: scatter:
; CHECK-ASM:       # %bb.0: # %entry
; CHECK-ASM-NEXT:    li a2, 0
; CHECK-ASM-NEXT:    li a3, 32
; CHECK-ASM-NEXT:    li a4, 5
; CHECK-ASM-NEXT:    li a5, 1024
; CHECK-ASM-NEXT:  .LBB4_1: # %vector.body
; CHECK-ASM-NEXT:    # =>This Inner Loop Header: Depth=1
; CHECK-ASM-NEXT:    add a6, a1, a2
; CHECK-ASM-NEXT:    vsetvli zero, a3, e8, m1, ta, mu
; CHECK-ASM-NEXT:    vle8.v v8, (a6)
; CHECK-ASM-NEXT:    vlse8.v v9, (a0), a4
; CHECK-ASM-NEXT:    vadd.vv v8, v9, v8
; CHECK-ASM-NEXT:    vsse8.v v8, (a0), a4
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
; CHECK-NEXT:    [[TMP0:%.*]] = getelementptr inbounds i8, i8* [[B:%.*]], i64 [[INDEX]]
; CHECK-NEXT:    [[TMP1:%.*]] = bitcast i8* [[TMP0]] to <32 x i8>*
; CHECK-NEXT:    [[WIDE_LOAD:%.*]] = load <32 x i8>, <32 x i8>* [[TMP1]], align 1
; CHECK-NEXT:    [[TMP2:%.*]] = getelementptr i8, i8* [[A:%.*]], i64 [[VEC_IND_SCALAR]]
; CHECK-NEXT:    [[WIDE_MASKED_GATHER:%.*]] = call <32 x i8> @llvm.riscv.masked.strided.load.v32i8.p0i8.i64(<32 x i8> [[MASKEDOFF:%.*]], i8* [[TMP2]], i64 5, <32 x i1> <i1 true, i1 false, i1 false, i1 true, i1 false, i1 true, i1 true, i1 false, i1 true, i1 true, i1 false, i1 false, i1 true, i1 false, i1 true, i1 false, i1 true, i1 false, i1 true, i1 true, i1 false, i1 true, i1 false, i1 false, i1 false, i1 false, i1 false, i1 false, i1 true, i1 true, i1 true, i1 true>)
; CHECK-NEXT:    [[TMP3:%.*]] = add <32 x i8> [[WIDE_MASKED_GATHER]], [[WIDE_LOAD]]
; CHECK-NEXT:    call void @llvm.riscv.masked.strided.store.v32i8.p0i8.i64(<32 x i8> [[TMP3]], i8* [[TMP2]], i64 5, <32 x i1> <i1 true, i1 false, i1 false, i1 true, i1 false, i1 true, i1 true, i1 false, i1 true, i1 true, i1 false, i1 false, i1 true, i1 false, i1 true, i1 false, i1 true, i1 false, i1 true, i1 true, i1 false, i1 true, i1 false, i1 false, i1 false, i1 false, i1 false, i1 false, i1 true, i1 true, i1 true, i1 true>)
; CHECK-NEXT:    [[INDEX_NEXT]] = add nuw i64 [[INDEX]], 32
; CHECK-NEXT:    [[VEC_IND_NEXT_SCALAR]] = add i64 [[VEC_IND_SCALAR]], 160
; CHECK-NEXT:    [[TMP4:%.*]] = icmp eq i64 [[INDEX_NEXT]], 1024
; CHECK-NEXT:    br i1 [[TMP4]], label [[FOR_COND_CLEANUP:%.*]], label [[VECTOR_BODY]]
; CHECK:       for.cond.cleanup:
; CHECK-NEXT:    ret void
;
; CHECK-ASM-LABEL: scatter_masked:
; CHECK-ASM:       # %bb.0: # %entry
; CHECK-ASM-NEXT:    li a2, 0
; CHECK-ASM-NEXT:    li a3, 32
; CHECK-ASM-NEXT:    lui a4, 983765
; CHECK-ASM-NEXT:    addiw a4, a4, 873
; CHECK-ASM-NEXT:    vsetivli zero, 1, e32, mf2, ta, mu
; CHECK-ASM-NEXT:    vmv.s.x v0, a4
; CHECK-ASM-NEXT:    li a4, 5
; CHECK-ASM-NEXT:    li a5, 1024
; CHECK-ASM-NEXT:  .LBB5_1: # %vector.body
; CHECK-ASM-NEXT:    # =>This Inner Loop Header: Depth=1
; CHECK-ASM-NEXT:    add a6, a1, a2
; CHECK-ASM-NEXT:    vsetvli zero, a3, e8, m1, ta, mu
; CHECK-ASM-NEXT:    vle8.v v9, (a6)
; CHECK-ASM-NEXT:    vmv1r.v v10, v8
; CHECK-ASM-NEXT:    vlse8.v v10, (a0), a4, v0.t
; CHECK-ASM-NEXT:    vadd.vv v9, v10, v9
; CHECK-ASM-NEXT:    vsse8.v v9, (a0), a4, v0.t
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
; CHECK-NEXT:    [[WIDE_MASKED_GATHER:%.*]] = call <8 x i32> @llvm.riscv.masked.strided.load.v8i32.p0i32.i64(<8 x i32> undef, i32* [[TMP0]], i64 16, <8 x i1> <i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true>)
; CHECK-NEXT:    [[TMP1:%.*]] = getelementptr inbounds i32, i32* [[A:%.*]], i64 [[INDEX]]
; CHECK-NEXT:    [[TMP2:%.*]] = bitcast i32* [[TMP1]] to <8 x i32>*
; CHECK-NEXT:    [[WIDE_LOAD:%.*]] = load <8 x i32>, <8 x i32>* [[TMP2]], align 1
; CHECK-NEXT:    [[TMP3:%.*]] = add <8 x i32> [[WIDE_LOAD]], [[WIDE_MASKED_GATHER]]
; CHECK-NEXT:    [[TMP4:%.*]] = bitcast i32* [[TMP1]] to <8 x i32>*
; CHECK-NEXT:    store <8 x i32> [[TMP3]], <8 x i32>* [[TMP4]], align 1
; CHECK-NEXT:    [[INDEX_NEXT]] = add nuw i64 [[INDEX]], 8
; CHECK-NEXT:    [[VEC_IND_NEXT_SCALAR]] = add i64 [[VEC_IND_SCALAR]], 32
; CHECK-NEXT:    [[TMP5:%.*]] = icmp eq i64 [[INDEX_NEXT]], 1024
; CHECK-NEXT:    br i1 [[TMP5]], label [[FOR_COND_CLEANUP:%.*]], label [[VECTOR_BODY]]
; CHECK:       for.cond.cleanup:
; CHECK-NEXT:    ret void
;
; CHECK-ASM-LABEL: gather_pow2:
; CHECK-ASM:       # %bb.0: # %entry
; CHECK-ASM-NEXT:    li a2, 1024
; CHECK-ASM-NEXT:    li a3, 16
; CHECK-ASM-NEXT:    li a4, 32
; CHECK-ASM-NEXT:  .LBB6_1: # %vector.body
; CHECK-ASM-NEXT:    # =>This Inner Loop Header: Depth=1
; CHECK-ASM-NEXT:    vsetivli zero, 8, e32, m1, ta, mu
; CHECK-ASM-NEXT:    vlse32.v v8, (a1), a3
; CHECK-ASM-NEXT:    vsetvli zero, a4, e8, m1, ta, mu
; CHECK-ASM-NEXT:    vle8.v v9, (a0)
; CHECK-ASM-NEXT:    vsetivli zero, 8, e32, m1, ta, mu
; CHECK-ASM-NEXT:    vadd.vv v8, v9, v8
; CHECK-ASM-NEXT:    vsetvli zero, a4, e8, m1, ta, mu
; CHECK-ASM-NEXT:    vse8.v v8, (a0)
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
; CHECK-NEXT:    [[TMP0:%.*]] = getelementptr inbounds i32, i32* [[B:%.*]], i64 [[INDEX]]
; CHECK-NEXT:    [[TMP1:%.*]] = bitcast i32* [[TMP0]] to <8 x i32>*
; CHECK-NEXT:    [[WIDE_LOAD:%.*]] = load <8 x i32>, <8 x i32>* [[TMP1]], align 1
; CHECK-NEXT:    [[TMP2:%.*]] = getelementptr i32, i32* [[A:%.*]], i64 [[VEC_IND_SCALAR]]
; CHECK-NEXT:    [[WIDE_MASKED_GATHER:%.*]] = call <8 x i32> @llvm.riscv.masked.strided.load.v8i32.p0i32.i64(<8 x i32> undef, i32* [[TMP2]], i64 16, <8 x i1> <i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true>)
; CHECK-NEXT:    [[TMP3:%.*]] = add <8 x i32> [[WIDE_MASKED_GATHER]], [[WIDE_LOAD]]
; CHECK-NEXT:    call void @llvm.riscv.masked.strided.store.v8i32.p0i32.i64(<8 x i32> [[TMP3]], i32* [[TMP2]], i64 16, <8 x i1> <i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true>)
; CHECK-NEXT:    [[INDEX_NEXT]] = add nuw i64 [[INDEX]], 8
; CHECK-NEXT:    [[VEC_IND_NEXT_SCALAR]] = add i64 [[VEC_IND_SCALAR]], 32
; CHECK-NEXT:    [[TMP4:%.*]] = icmp eq i64 [[INDEX_NEXT]], 1024
; CHECK-NEXT:    br i1 [[TMP4]], label [[FOR_COND_CLEANUP:%.*]], label [[VECTOR_BODY]]
; CHECK:       for.cond.cleanup:
; CHECK-NEXT:    ret void
;
; CHECK-ASM-LABEL: scatter_pow2:
; CHECK-ASM:       # %bb.0: # %entry
; CHECK-ASM-NEXT:    li a2, 1024
; CHECK-ASM-NEXT:    li a3, 32
; CHECK-ASM-NEXT:    li a4, 16
; CHECK-ASM-NEXT:  .LBB7_1: # %vector.body
; CHECK-ASM-NEXT:    # =>This Inner Loop Header: Depth=1
; CHECK-ASM-NEXT:    vsetvli zero, a3, e8, m1, ta, mu
; CHECK-ASM-NEXT:    vle8.v v8, (a1)
; CHECK-ASM-NEXT:    vsetivli zero, 8, e32, m1, ta, mu
; CHECK-ASM-NEXT:    vlse32.v v9, (a0), a4
; CHECK-ASM-NEXT:    vadd.vv v8, v9, v8
; CHECK-ASM-NEXT:    vsse32.v v8, (a0), a4
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
; CHECK-NEXT:    [[TMP1:%.*]] = getelementptr [[STRUCT_FOO]], %struct.foo* [[B]], i64 [[VEC_IND_SCALAR1]], i32 1
; CHECK-NEXT:    [[WIDE_MASKED_GATHER:%.*]] = call <8 x i32> @llvm.riscv.masked.strided.load.v8i32.p0i32.i64(<8 x i32> undef, i32* [[TMP0]], i64 16, <8 x i1> <i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true>)
; CHECK-NEXT:    [[WIDE_MASKED_GATHER9:%.*]] = call <8 x i32> @llvm.riscv.masked.strided.load.v8i32.p0i32.i64(<8 x i32> undef, i32* [[TMP1]], i64 16, <8 x i1> <i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true>)
; CHECK-NEXT:    [[TMP2:%.*]] = getelementptr inbounds i32, i32* [[A:%.*]], i64 [[INDEX]]
; CHECK-NEXT:    [[TMP3:%.*]] = bitcast i32* [[TMP2]] to <8 x i32>*
; CHECK-NEXT:    [[WIDE_LOAD:%.*]] = load <8 x i32>, <8 x i32>* [[TMP3]], align 4
; CHECK-NEXT:    [[TMP4:%.*]] = getelementptr inbounds i32, i32* [[TMP2]], i64 8
; CHECK-NEXT:    [[TMP5:%.*]] = bitcast i32* [[TMP4]] to <8 x i32>*
; CHECK-NEXT:    [[WIDE_LOAD10:%.*]] = load <8 x i32>, <8 x i32>* [[TMP5]], align 4
; CHECK-NEXT:    [[TMP6:%.*]] = add nsw <8 x i32> [[WIDE_LOAD]], [[WIDE_MASKED_GATHER]]
; CHECK-NEXT:    [[TMP7:%.*]] = add nsw <8 x i32> [[WIDE_LOAD10]], [[WIDE_MASKED_GATHER9]]
; CHECK-NEXT:    [[TMP8:%.*]] = bitcast i32* [[TMP2]] to <8 x i32>*
; CHECK-NEXT:    store <8 x i32> [[TMP6]], <8 x i32>* [[TMP8]], align 4
; CHECK-NEXT:    [[TMP9:%.*]] = bitcast i32* [[TMP4]] to <8 x i32>*
; CHECK-NEXT:    store <8 x i32> [[TMP7]], <8 x i32>* [[TMP9]], align 4
; CHECK-NEXT:    [[INDEX_NEXT]] = add nuw i64 [[INDEX]], 16
; CHECK-NEXT:    [[VEC_IND_NEXT_SCALAR]] = add i64 [[VEC_IND_SCALAR]], 16
; CHECK-NEXT:    [[VEC_IND_NEXT_SCALAR2]] = add i64 [[VEC_IND_SCALAR1]], 16
; CHECK-NEXT:    [[TMP10:%.*]] = icmp eq i64 [[INDEX_NEXT]], 1024
; CHECK-NEXT:    br i1 [[TMP10]], label [[FOR_COND_CLEANUP:%.*]], label [[VECTOR_BODY]]
; CHECK:       for.cond.cleanup:
; CHECK-NEXT:    ret void
;
; CHECK-ASM-LABEL: struct_gather:
; CHECK-ASM:       # %bb.0: # %entry
; CHECK-ASM-NEXT:    addi a1, a1, 132
; CHECK-ASM-NEXT:    li a2, 1024
; CHECK-ASM-NEXT:    li a3, 16
; CHECK-ASM-NEXT:    vsetivli zero, 8, e32, m1, ta, mu
; CHECK-ASM-NEXT:  .LBB8_1: # %vector.body
; CHECK-ASM-NEXT:    # =>This Inner Loop Header: Depth=1
; CHECK-ASM-NEXT:    addi a4, a1, -128
; CHECK-ASM-NEXT:    vlse32.v v8, (a4), a3
; CHECK-ASM-NEXT:    vlse32.v v9, (a1), a3
; CHECK-ASM-NEXT:    vle32.v v10, (a0)
; CHECK-ASM-NEXT:    addi a4, a0, 32
; CHECK-ASM-NEXT:    vle32.v v11, (a4)
; CHECK-ASM-NEXT:    vadd.vv v8, v10, v8
; CHECK-ASM-NEXT:    vadd.vv v9, v11, v9
; CHECK-ASM-NEXT:    vse32.v v8, (a0)
; CHECK-ASM-NEXT:    vse32.v v9, (a4)
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
; CHECK-NEXT:    [[TMP0:%.*]] = getelementptr i32, i32* [[B:%.*]], i64 [[VEC_IND_SCALAR]]
; CHECK-NEXT:    [[WIDE_MASKED_GATHER:%.*]] = call <8 x i32> @llvm.riscv.masked.strided.load.v8i32.p0i32.i64(<8 x i32> undef, i32* [[TMP0]], i64 64, <8 x i1> <i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true>)
; CHECK-NEXT:    [[TMP1:%.*]] = getelementptr i32, i32* [[A:%.*]], i64 [[VEC_IND_SCALAR1]]
; CHECK-NEXT:    [[WIDE_MASKED_GATHER52:%.*]] = call <8 x i32> @llvm.riscv.masked.strided.load.v8i32.p0i32.i64(<8 x i32> undef, i32* [[TMP1]], i64 16, <8 x i1> <i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true>)
; CHECK-NEXT:    [[TMP2:%.*]] = add nsw <8 x i32> [[WIDE_MASKED_GATHER52]], [[WIDE_MASKED_GATHER]]
; CHECK-NEXT:    call void @llvm.riscv.masked.strided.store.v8i32.p0i32.i64(<8 x i32> [[TMP2]], i32* [[TMP1]], i64 16, <8 x i1> <i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true>)
; CHECK-NEXT:    [[TMP3:%.*]] = getelementptr i32, i32* [[B]], i64 [[VEC_IND_SCALAR3]]
; CHECK-NEXT:    [[WIDE_MASKED_GATHER53:%.*]] = call <8 x i32> @llvm.riscv.masked.strided.load.v8i32.p0i32.i64(<8 x i32> undef, i32* [[TMP3]], i64 64, <8 x i1> <i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true>)
; CHECK-NEXT:    [[TMP4:%.*]] = getelementptr i32, i32* [[A]], i64 [[VEC_IND_SCALAR5]]
; CHECK-NEXT:    [[WIDE_MASKED_GATHER54:%.*]] = call <8 x i32> @llvm.riscv.masked.strided.load.v8i32.p0i32.i64(<8 x i32> undef, i32* [[TMP4]], i64 16, <8 x i1> <i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true>)
; CHECK-NEXT:    [[TMP5:%.*]] = add nsw <8 x i32> [[WIDE_MASKED_GATHER54]], [[WIDE_MASKED_GATHER53]]
; CHECK-NEXT:    call void @llvm.riscv.masked.strided.store.v8i32.p0i32.i64(<8 x i32> [[TMP5]], i32* [[TMP4]], i64 16, <8 x i1> <i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true>)
; CHECK-NEXT:    [[TMP6:%.*]] = getelementptr i32, i32* [[B]], i64 [[VEC_IND_SCALAR7]]
; CHECK-NEXT:    [[WIDE_MASKED_GATHER55:%.*]] = call <8 x i32> @llvm.riscv.masked.strided.load.v8i32.p0i32.i64(<8 x i32> undef, i32* [[TMP6]], i64 64, <8 x i1> <i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true>)
; CHECK-NEXT:    [[TMP7:%.*]] = getelementptr i32, i32* [[A]], i64 [[VEC_IND_SCALAR9]]
; CHECK-NEXT:    [[WIDE_MASKED_GATHER56:%.*]] = call <8 x i32> @llvm.riscv.masked.strided.load.v8i32.p0i32.i64(<8 x i32> undef, i32* [[TMP7]], i64 16, <8 x i1> <i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true>)
; CHECK-NEXT:    [[TMP8:%.*]] = add nsw <8 x i32> [[WIDE_MASKED_GATHER56]], [[WIDE_MASKED_GATHER55]]
; CHECK-NEXT:    call void @llvm.riscv.masked.strided.store.v8i32.p0i32.i64(<8 x i32> [[TMP8]], i32* [[TMP7]], i64 16, <8 x i1> <i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true>)
; CHECK-NEXT:    [[TMP9:%.*]] = getelementptr i32, i32* [[B]], i64 [[VEC_IND_SCALAR11]]
; CHECK-NEXT:    [[WIDE_MASKED_GATHER57:%.*]] = call <8 x i32> @llvm.riscv.masked.strided.load.v8i32.p0i32.i64(<8 x i32> undef, i32* [[TMP9]], i64 64, <8 x i1> <i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true>)
; CHECK-NEXT:    [[TMP10:%.*]] = getelementptr i32, i32* [[A]], i64 [[VEC_IND_SCALAR13]]
; CHECK-NEXT:    [[WIDE_MASKED_GATHER58:%.*]] = call <8 x i32> @llvm.riscv.masked.strided.load.v8i32.p0i32.i64(<8 x i32> undef, i32* [[TMP10]], i64 16, <8 x i1> <i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true>)
; CHECK-NEXT:    [[TMP11:%.*]] = add nsw <8 x i32> [[WIDE_MASKED_GATHER58]], [[WIDE_MASKED_GATHER57]]
; CHECK-NEXT:    call void @llvm.riscv.masked.strided.store.v8i32.p0i32.i64(<8 x i32> [[TMP11]], i32* [[TMP10]], i64 16, <8 x i1> <i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true>)
; CHECK-NEXT:    [[INDEX_NEXT]] = add nuw i64 [[INDEX]], 8
; CHECK-NEXT:    [[VEC_IND_NEXT_SCALAR]] = add i64 [[VEC_IND_SCALAR]], 128
; CHECK-NEXT:    [[VEC_IND_NEXT_SCALAR2]] = add i64 [[VEC_IND_SCALAR1]], 32
; CHECK-NEXT:    [[VEC_IND_NEXT_SCALAR4]] = add i64 [[VEC_IND_SCALAR3]], 128
; CHECK-NEXT:    [[VEC_IND_NEXT_SCALAR6]] = add i64 [[VEC_IND_SCALAR5]], 32
; CHECK-NEXT:    [[VEC_IND_NEXT_SCALAR8]] = add i64 [[VEC_IND_SCALAR7]], 128
; CHECK-NEXT:    [[VEC_IND_NEXT_SCALAR10]] = add i64 [[VEC_IND_SCALAR9]], 32
; CHECK-NEXT:    [[VEC_IND_NEXT_SCALAR12]] = add i64 [[VEC_IND_SCALAR11]], 128
; CHECK-NEXT:    [[VEC_IND_NEXT_SCALAR14]] = add i64 [[VEC_IND_SCALAR13]], 32
; CHECK-NEXT:    [[TMP12:%.*]] = icmp eq i64 [[INDEX_NEXT]], 256
; CHECK-NEXT:    br i1 [[TMP12]], label [[FOR_COND_CLEANUP:%.*]], label [[VECTOR_BODY]]
; CHECK:       for.cond.cleanup:
; CHECK-NEXT:    ret void
;
; CHECK-ASM-LABEL: gather_unroll:
; CHECK-ASM:       # %bb.0: # %entry
; CHECK-ASM-NEXT:    li a2, 256
; CHECK-ASM-NEXT:    li a3, 64
; CHECK-ASM-NEXT:    li a4, 16
; CHECK-ASM-NEXT:    vsetivli zero, 8, e32, m1, ta, mu
; CHECK-ASM-NEXT:  .LBB9_1: # %vector.body
; CHECK-ASM-NEXT:    # =>This Inner Loop Header: Depth=1
; CHECK-ASM-NEXT:    vlse32.v v8, (a1), a3
; CHECK-ASM-NEXT:    vlse32.v v9, (a0), a4
; CHECK-ASM-NEXT:    vadd.vv v8, v9, v8
; CHECK-ASM-NEXT:    vsse32.v v8, (a0), a4
; CHECK-ASM-NEXT:    addi a5, a1, 16
; CHECK-ASM-NEXT:    vlse32.v v8, (a5), a3
; CHECK-ASM-NEXT:    addi a5, a0, 4
; CHECK-ASM-NEXT:    vlse32.v v9, (a5), a4
; CHECK-ASM-NEXT:    vadd.vv v8, v9, v8
; CHECK-ASM-NEXT:    vsse32.v v8, (a5), a4
; CHECK-ASM-NEXT:    addi a5, a1, 32
; CHECK-ASM-NEXT:    vlse32.v v8, (a5), a3
; CHECK-ASM-NEXT:    addi a5, a0, 8
; CHECK-ASM-NEXT:    vlse32.v v9, (a5), a4
; CHECK-ASM-NEXT:    vadd.vv v8, v9, v8
; CHECK-ASM-NEXT:    vsse32.v v8, (a5), a4
; CHECK-ASM-NEXT:    addi a5, a1, 48
; CHECK-ASM-NEXT:    vlse32.v v8, (a5), a3
; CHECK-ASM-NEXT:    addi a5, a0, 12
; CHECK-ASM-NEXT:    vlse32.v v9, (a5), a4
; CHECK-ASM-NEXT:    vadd.vv v8, v9, v8
; CHECK-ASM-NEXT:    vsse32.v v8, (a5), a4
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
; CHECK-NEXT:    [[TMP4:%.*]] = phi i64 [ 0, [[TMP2:%.*]] ], [ [[TMP13:%.*]], [[TMP3]] ]
; CHECK-NEXT:    [[DOTSCALAR:%.*]] = phi i64 [ 0, [[TMP2]] ], [ [[DOTSCALAR1:%.*]], [[TMP3]] ]
; CHECK-NEXT:    [[DOTSCALAR2:%.*]] = phi i64 [ 10, [[TMP2]] ], [ [[DOTSCALAR3:%.*]], [[TMP3]] ]
; CHECK-NEXT:    [[TMP5:%.*]] = getelementptr i32*, i32** [[TMP1:%.*]], i64 [[DOTSCALAR]]
; CHECK-NEXT:    [[TMP6:%.*]] = getelementptr i32*, i32** [[TMP1]], i64 [[DOTSCALAR2]]
; CHECK-NEXT:    [[TMP7:%.*]] = call <2 x i32*> @llvm.riscv.masked.strided.load.v2p0i32.p0p0i32.i64(<2 x i32*> undef, i32** [[TMP5]], i64 40, <2 x i1> <i1 true, i1 true>)
; CHECK-NEXT:    [[TMP8:%.*]] = call <2 x i32*> @llvm.riscv.masked.strided.load.v2p0i32.p0p0i32.i64(<2 x i32*> undef, i32** [[TMP6]], i64 40, <2 x i1> <i1 true, i1 true>)
; CHECK-NEXT:    [[TMP9:%.*]] = getelementptr inbounds i32*, i32** [[TMP0:%.*]], i64 [[TMP4]]
; CHECK-NEXT:    [[TMP10:%.*]] = bitcast i32** [[TMP9]] to <2 x i32*>*
; CHECK-NEXT:    store <2 x i32*> [[TMP7]], <2 x i32*>* [[TMP10]], align 8
; CHECK-NEXT:    [[TMP11:%.*]] = getelementptr inbounds i32*, i32** [[TMP9]], i64 2
; CHECK-NEXT:    [[TMP12:%.*]] = bitcast i32** [[TMP11]] to <2 x i32*>*
; CHECK-NEXT:    store <2 x i32*> [[TMP8]], <2 x i32*>* [[TMP12]], align 8
; CHECK-NEXT:    [[TMP13]] = add nuw i64 [[TMP4]], 4
; CHECK-NEXT:    [[DOTSCALAR1]] = add i64 [[DOTSCALAR]], 20
; CHECK-NEXT:    [[DOTSCALAR3]] = add i64 [[DOTSCALAR2]], 20
; CHECK-NEXT:    [[TMP14:%.*]] = icmp eq i64 [[TMP13]], 1024
; CHECK-NEXT:    br i1 [[TMP14]], label [[TMP15:%.*]], label [[TMP3]]
; CHECK:       15:
; CHECK-NEXT:    ret void
;
; CHECK-ASM-LABEL: gather_of_pointers:
; CHECK-ASM:       # %bb.0:
; CHECK-ASM-NEXT:    li a2, 1024
; CHECK-ASM-NEXT:    li a3, 40
; CHECK-ASM-NEXT:    vsetivli zero, 2, e64, m1, ta, mu
; CHECK-ASM-NEXT:  .LBB10_1: # =>This Inner Loop Header: Depth=1
; CHECK-ASM-NEXT:    vlse64.v v8, (a1), a3
; CHECK-ASM-NEXT:    addi a4, a1, 80
; CHECK-ASM-NEXT:    vlse64.v v9, (a4), a3
; CHECK-ASM-NEXT:    vse64.v v8, (a0)
; CHECK-ASM-NEXT:    addi a4, a0, 16
; CHECK-ASM-NEXT:    vse64.v v9, (a4)
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
; CHECK-NEXT:    [[TMP4:%.*]] = phi i64 [ 0, [[TMP2:%.*]] ], [ [[TMP13:%.*]], [[TMP3]] ]
; CHECK-NEXT:    [[DOTSCALAR:%.*]] = phi i64 [ 0, [[TMP2]] ], [ [[DOTSCALAR1:%.*]], [[TMP3]] ]
; CHECK-NEXT:    [[DOTSCALAR2:%.*]] = phi i64 [ 10, [[TMP2]] ], [ [[DOTSCALAR3:%.*]], [[TMP3]] ]
; CHECK-NEXT:    [[TMP5:%.*]] = getelementptr inbounds i32*, i32** [[TMP1:%.*]], i64 [[TMP4]]
; CHECK-NEXT:    [[TMP6:%.*]] = bitcast i32** [[TMP5]] to <2 x i32*>*
; CHECK-NEXT:    [[TMP7:%.*]] = load <2 x i32*>, <2 x i32*>* [[TMP6]], align 8
; CHECK-NEXT:    [[TMP8:%.*]] = getelementptr inbounds i32*, i32** [[TMP5]], i64 2
; CHECK-NEXT:    [[TMP9:%.*]] = bitcast i32** [[TMP8]] to <2 x i32*>*
; CHECK-NEXT:    [[TMP10:%.*]] = load <2 x i32*>, <2 x i32*>* [[TMP9]], align 8
; CHECK-NEXT:    [[TMP11:%.*]] = getelementptr i32*, i32** [[TMP0:%.*]], i64 [[DOTSCALAR]]
; CHECK-NEXT:    [[TMP12:%.*]] = getelementptr i32*, i32** [[TMP0]], i64 [[DOTSCALAR2]]
; CHECK-NEXT:    call void @llvm.riscv.masked.strided.store.v2p0i32.p0p0i32.i64(<2 x i32*> [[TMP7]], i32** [[TMP11]], i64 40, <2 x i1> <i1 true, i1 true>)
; CHECK-NEXT:    call void @llvm.riscv.masked.strided.store.v2p0i32.p0p0i32.i64(<2 x i32*> [[TMP10]], i32** [[TMP12]], i64 40, <2 x i1> <i1 true, i1 true>)
; CHECK-NEXT:    [[TMP13]] = add nuw i64 [[TMP4]], 4
; CHECK-NEXT:    [[DOTSCALAR1]] = add i64 [[DOTSCALAR]], 20
; CHECK-NEXT:    [[DOTSCALAR3]] = add i64 [[DOTSCALAR2]], 20
; CHECK-NEXT:    [[TMP14:%.*]] = icmp eq i64 [[TMP13]], 1024
; CHECK-NEXT:    br i1 [[TMP14]], label [[TMP15:%.*]], label [[TMP3]]
; CHECK:       15:
; CHECK-NEXT:    ret void
;
; CHECK-ASM-LABEL: scatter_of_pointers:
; CHECK-ASM:       # %bb.0:
; CHECK-ASM-NEXT:    li a2, 1024
; CHECK-ASM-NEXT:    li a3, 40
; CHECK-ASM-NEXT:    vsetivli zero, 2, e64, m1, ta, mu
; CHECK-ASM-NEXT:  .LBB11_1: # =>This Inner Loop Header: Depth=1
; CHECK-ASM-NEXT:    vle64.v v8, (a1)
; CHECK-ASM-NEXT:    addi a4, a1, 16
; CHECK-ASM-NEXT:    vle64.v v9, (a4)
; CHECK-ASM-NEXT:    addi a4, a0, 80
; CHECK-ASM-NEXT:    vsse64.v v8, (a0), a3
; CHECK-ASM-NEXT:    vsse64.v v9, (a4), a3
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

define void @strided_load_startval_add_with_splat(i8* noalias nocapture %0, i8* noalias nocapture readonly %1, i32 signext %2) {
; CHECK-LABEL: @strided_load_startval_add_with_splat(
; CHECK-NEXT:    [[TMP4:%.*]] = icmp eq i32 [[TMP2:%.*]], 1024
; CHECK-NEXT:    br i1 [[TMP4]], label [[TMP31:%.*]], label [[TMP5:%.*]]
; CHECK:       5:
; CHECK-NEXT:    [[TMP6:%.*]] = sext i32 [[TMP2]] to i64
; CHECK-NEXT:    [[TMP7:%.*]] = sub i32 1023, [[TMP2]]
; CHECK-NEXT:    [[TMP8:%.*]] = zext i32 [[TMP7]] to i64
; CHECK-NEXT:    [[TMP9:%.*]] = add nuw nsw i64 [[TMP8]], 1
; CHECK-NEXT:    [[TMP10:%.*]] = icmp ult i32 [[TMP7]], 31
; CHECK-NEXT:    br i1 [[TMP10]], label [[TMP29:%.*]], label [[TMP11:%.*]]
; CHECK:       11:
; CHECK-NEXT:    [[TMP12:%.*]] = and i64 [[TMP9]], 8589934560
; CHECK-NEXT:    [[TMP13:%.*]] = add nsw i64 [[TMP12]], [[TMP6]]
; CHECK-NEXT:    [[TMP14:%.*]] = add i64 0, [[TMP6]]
; CHECK-NEXT:    [[START:%.*]] = mul i64 [[TMP14]], 5
; CHECK-NEXT:    br label [[TMP15:%.*]]
; CHECK:       15:
; CHECK-NEXT:    [[TMP16:%.*]] = phi i64 [ 0, [[TMP11]] ], [ [[TMP25:%.*]], [[TMP15]] ]
; CHECK-NEXT:    [[DOTSCALAR:%.*]] = phi i64 [ [[START]], [[TMP11]] ], [ [[DOTSCALAR1:%.*]], [[TMP15]] ]
; CHECK-NEXT:    [[TMP17:%.*]] = add i64 [[TMP16]], [[TMP6]]
; CHECK-NEXT:    [[TMP18:%.*]] = getelementptr i8, i8* [[TMP1:%.*]], i64 [[DOTSCALAR]]
; CHECK-NEXT:    [[TMP19:%.*]] = call <32 x i8> @llvm.riscv.masked.strided.load.v32i8.p0i8.i64(<32 x i8> undef, i8* [[TMP18]], i64 5, <32 x i1> <i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true>)
; CHECK-NEXT:    [[TMP20:%.*]] = getelementptr inbounds i8, i8* [[TMP0:%.*]], i64 [[TMP17]]
; CHECK-NEXT:    [[TMP21:%.*]] = bitcast i8* [[TMP20]] to <32 x i8>*
; CHECK-NEXT:    [[TMP22:%.*]] = load <32 x i8>, <32 x i8>* [[TMP21]], align 1
; CHECK-NEXT:    [[TMP23:%.*]] = add <32 x i8> [[TMP22]], [[TMP19]]
; CHECK-NEXT:    [[TMP24:%.*]] = bitcast i8* [[TMP20]] to <32 x i8>*
; CHECK-NEXT:    store <32 x i8> [[TMP23]], <32 x i8>* [[TMP24]], align 1
; CHECK-NEXT:    [[TMP25]] = add nuw i64 [[TMP16]], 32
; CHECK-NEXT:    [[DOTSCALAR1]] = add i64 [[DOTSCALAR]], 160
; CHECK-NEXT:    [[TMP26:%.*]] = icmp eq i64 [[TMP25]], [[TMP12]]
; CHECK-NEXT:    br i1 [[TMP26]], label [[TMP27:%.*]], label [[TMP15]]
; CHECK:       27:
; CHECK-NEXT:    [[TMP28:%.*]] = icmp eq i64 [[TMP9]], [[TMP12]]
; CHECK-NEXT:    br i1 [[TMP28]], label [[TMP31]], label [[TMP29]]
; CHECK:       29:
; CHECK-NEXT:    [[TMP30:%.*]] = phi i64 [ [[TMP6]], [[TMP5]] ], [ [[TMP13]], [[TMP27]] ]
; CHECK-NEXT:    br label [[TMP32:%.*]]
; CHECK:       31:
; CHECK-NEXT:    ret void
; CHECK:       32:
; CHECK-NEXT:    [[TMP33:%.*]] = phi i64 [ [[TMP40:%.*]], [[TMP32]] ], [ [[TMP30]], [[TMP29]] ]
; CHECK-NEXT:    [[TMP34:%.*]] = mul nsw i64 [[TMP33]], 5
; CHECK-NEXT:    [[TMP35:%.*]] = getelementptr inbounds i8, i8* [[TMP1]], i64 [[TMP34]]
; CHECK-NEXT:    [[TMP36:%.*]] = load i8, i8* [[TMP35]], align 1
; CHECK-NEXT:    [[TMP37:%.*]] = getelementptr inbounds i8, i8* [[TMP0]], i64 [[TMP33]]
; CHECK-NEXT:    [[TMP38:%.*]] = load i8, i8* [[TMP37]], align 1
; CHECK-NEXT:    [[TMP39:%.*]] = add i8 [[TMP38]], [[TMP36]]
; CHECK-NEXT:    store i8 [[TMP39]], i8* [[TMP37]], align 1
; CHECK-NEXT:    [[TMP40]] = add nsw i64 [[TMP33]], 1
; CHECK-NEXT:    [[TMP41:%.*]] = trunc i64 [[TMP40]] to i32
; CHECK-NEXT:    [[TMP42:%.*]] = icmp eq i32 [[TMP41]], 1024
; CHECK-NEXT:    br i1 [[TMP42]], label [[TMP31]], label [[TMP32]]
;
; CHECK-ASM-LABEL: strided_load_startval_add_with_splat:
; CHECK-ASM:       # %bb.0:
; CHECK-ASM-NEXT:    li a3, 1024
; CHECK-ASM-NEXT:    beq a2, a3, .LBB12_7
; CHECK-ASM-NEXT:  # %bb.1:
; CHECK-ASM-NEXT:    li a3, 1023
; CHECK-ASM-NEXT:    subw a4, a3, a2
; CHECK-ASM-NEXT:    li a5, 31
; CHECK-ASM-NEXT:    mv a3, a2
; CHECK-ASM-NEXT:    bltu a4, a5, .LBB12_5
; CHECK-ASM-NEXT:  # %bb.2:
; CHECK-ASM-NEXT:    slli a3, a4, 32
; CHECK-ASM-NEXT:    srli a3, a3, 32
; CHECK-ASM-NEXT:    addi a4, a3, 1
; CHECK-ASM-NEXT:    andi a5, a4, -32
; CHECK-ASM-NEXT:    add a3, a5, a2
; CHECK-ASM-NEXT:    slli a6, a2, 2
; CHECK-ASM-NEXT:    add a6, a6, a2
; CHECK-ASM-NEXT:    add a2, a0, a2
; CHECK-ASM-NEXT:    add a6, a1, a6
; CHECK-ASM-NEXT:    li a7, 32
; CHECK-ASM-NEXT:    li t0, 5
; CHECK-ASM-NEXT:    mv t1, a5
; CHECK-ASM-NEXT:  .LBB12_3: # =>This Inner Loop Header: Depth=1
; CHECK-ASM-NEXT:    vsetvli zero, a7, e8, m1, ta, mu
; CHECK-ASM-NEXT:    vlse8.v v8, (a6), t0
; CHECK-ASM-NEXT:    vle8.v v9, (a2)
; CHECK-ASM-NEXT:    vadd.vv v8, v9, v8
; CHECK-ASM-NEXT:    vse8.v v8, (a2)
; CHECK-ASM-NEXT:    addi t1, t1, -32
; CHECK-ASM-NEXT:    addi a2, a2, 32
; CHECK-ASM-NEXT:    addi a6, a6, 160
; CHECK-ASM-NEXT:    bnez t1, .LBB12_3
; CHECK-ASM-NEXT:  # %bb.4:
; CHECK-ASM-NEXT:    beq a4, a5, .LBB12_7
; CHECK-ASM-NEXT:  .LBB12_5:
; CHECK-ASM-NEXT:    slli a2, a3, 2
; CHECK-ASM-NEXT:    add a2, a2, a3
; CHECK-ASM-NEXT:    add a1, a1, a2
; CHECK-ASM-NEXT:    li a2, 1024
; CHECK-ASM-NEXT:  .LBB12_6: # =>This Inner Loop Header: Depth=1
; CHECK-ASM-NEXT:    lb a4, 0(a1)
; CHECK-ASM-NEXT:    add a5, a0, a3
; CHECK-ASM-NEXT:    lb a6, 0(a5)
; CHECK-ASM-NEXT:    addw a4, a6, a4
; CHECK-ASM-NEXT:    sb a4, 0(a5)
; CHECK-ASM-NEXT:    addiw a4, a3, 1
; CHECK-ASM-NEXT:    addi a3, a3, 1
; CHECK-ASM-NEXT:    addi a1, a1, 5
; CHECK-ASM-NEXT:    bne a4, a2, .LBB12_6
; CHECK-ASM-NEXT:  .LBB12_7:
; CHECK-ASM-NEXT:    ret
  %4 = icmp eq i32 %2, 1024
  br i1 %4, label %36, label %5

5:                                                ; preds = %3
  %6 = sext i32 %2 to i64
  %7 = sub i32 1023, %2
  %8 = zext i32 %7 to i64
  %9 = add nuw nsw i64 %8, 1
  %10 = icmp ult i32 %7, 31
  br i1 %10, label %34, label %11

11:                                               ; preds = %5
  %12 = and i64 %9, 8589934560
  %13 = add nsw i64 %12, %6
  %14 = insertelement <32 x i64> poison, i64 %6, i64 0
  %15 = shufflevector <32 x i64> %14, <32 x i64> poison, <32 x i32> zeroinitializer
  %16 = add <32 x i64> %15, <i64 0, i64 1, i64 2, i64 3, i64 4, i64 5, i64 6, i64 7, i64 8, i64 9, i64 10, i64 11, i64 12, i64 13, i64 14, i64 15, i64 16, i64 17, i64 18, i64 19, i64 20, i64 21, i64 22, i64 23, i64 24, i64 25, i64 26, i64 27, i64 28, i64 29, i64 30, i64 31>
  br label %17

17:                                               ; preds = %17, %11
  %18 = phi i64 [ 0, %11 ], [ %29, %17 ]
  %19 = phi <32 x i64> [ %16, %11 ], [ %30, %17 ]
  %20 = add i64 %18, %6
  %21 = mul nsw <32 x i64> %19, <i64 5, i64 5, i64 5, i64 5, i64 5, i64 5, i64 5, i64 5, i64 5, i64 5, i64 5, i64 5, i64 5, i64 5, i64 5, i64 5, i64 5, i64 5, i64 5, i64 5, i64 5, i64 5, i64 5, i64 5, i64 5, i64 5, i64 5, i64 5, i64 5, i64 5, i64 5, i64 5>
  %22 = getelementptr inbounds i8, i8* %1, <32 x i64> %21
  %23 = call <32 x i8> @llvm.masked.gather.v32i8.v32p0i8(<32 x i8*> %22, i32 1, <32 x i1> <i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true>, <32 x i8> undef)
  %24 = getelementptr inbounds i8, i8* %0, i64 %20
  %25 = bitcast i8* %24 to <32 x i8>*
  %26 = load <32 x i8>, <32 x i8>* %25, align 1
  %27 = add <32 x i8> %26, %23
  %28 = bitcast i8* %24 to <32 x i8>*
  store <32 x i8> %27, <32 x i8>* %28, align 1
  %29 = add nuw i64 %18, 32
  %30 = add <32 x i64> %19, <i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32>
  %31 = icmp eq i64 %29, %12
  br i1 %31, label %32, label %17

32:                                               ; preds = %17
  %33 = icmp eq i64 %9, %12
  br i1 %33, label %36, label %34

34:                                               ; preds = %5, %32
  %35 = phi i64 [ %6, %5 ], [ %13, %32 ]
  br label %37

36:                                               ; preds = %37, %32, %3
  ret void

37:                                               ; preds = %34, %37
  %38 = phi i64 [ %45, %37 ], [ %35, %34 ]
  %39 = mul nsw i64 %38, 5
  %40 = getelementptr inbounds i8, i8* %1, i64 %39
  %41 = load i8, i8* %40, align 1
  %42 = getelementptr inbounds i8, i8* %0, i64 %38
  %43 = load i8, i8* %42, align 1
  %44 = add i8 %43, %41
  store i8 %44, i8* %42, align 1
  %45 = add nsw i64 %38, 1
  %46 = trunc i64 %45 to i32
  %47 = icmp eq i32 %46, 1024
  br i1 %47, label %36, label %37
}
