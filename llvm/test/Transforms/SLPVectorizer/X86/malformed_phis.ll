; RUN: opt -S -slp-vectorizer < %s | FileCheck %s
; RUN: opt -S -passes=slp-vectorizer < %s | FileCheck %s

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128-ni:1-p2:32:8:8:32-ni:2"
target triple = "x86_64-unknown-linux-gnu"

; Make sure we do not generate malformed phis not in the beginning of block.
define void @test() #0 {
; CHECK-LABEL: @test(
; CHECK-NEXT:  bb:
; CHECK-NEXT:    br label [[BB1:%.*]]
; CHECK:       bb1:
; CHECK-NEXT:    [[TMP:%.*]] = phi i32 [ undef, [[BB1]] ], [ undef, [[BB:%.*]] ]
; CHECK-NEXT:    [[TMP2:%.*]] = phi i32 [ [[OP_RDX:%.*]], [[BB1]] ], [ undef, [[BB]] ]
; CHECK-NEXT:    [[TMP0:%.*]] = insertelement <16 x i32> poison, i32 [[TMP]], i32 0
; CHECK-NEXT:    [[TMP1:%.*]] = insertelement <16 x i32> [[TMP0]], i32 [[TMP]], i32 1
; CHECK-NEXT:    [[TMP2:%.*]] = insertelement <16 x i32> [[TMP1]], i32 [[TMP]], i32 2
; CHECK-NEXT:    [[TMP3:%.*]] = insertelement <16 x i32> [[TMP2]], i32 [[TMP]], i32 3
; CHECK-NEXT:    [[TMP4:%.*]] = insertelement <16 x i32> [[TMP3]], i32 [[TMP]], i32 4
; CHECK-NEXT:    [[TMP5:%.*]] = insertelement <16 x i32> [[TMP4]], i32 [[TMP]], i32 5
; CHECK-NEXT:    [[TMP6:%.*]] = insertelement <16 x i32> [[TMP5]], i32 [[TMP]], i32 6
; CHECK-NEXT:    [[TMP7:%.*]] = insertelement <16 x i32> [[TMP6]], i32 [[TMP]], i32 7
; CHECK-NEXT:    [[TMP8:%.*]] = insertelement <16 x i32> [[TMP7]], i32 [[TMP]], i32 8
; CHECK-NEXT:    [[TMP9:%.*]] = insertelement <16 x i32> [[TMP8]], i32 [[TMP]], i32 9
; CHECK-NEXT:    [[TMP10:%.*]] = insertelement <16 x i32> [[TMP9]], i32 [[TMP]], i32 10
; CHECK-NEXT:    [[TMP11:%.*]] = insertelement <16 x i32> [[TMP10]], i32 [[TMP]], i32 11
; CHECK-NEXT:    [[TMP12:%.*]] = insertelement <16 x i32> [[TMP11]], i32 [[TMP]], i32 12
; CHECK-NEXT:    [[TMP13:%.*]] = insertelement <16 x i32> [[TMP12]], i32 [[TMP]], i32 13
; CHECK-NEXT:    [[TMP14:%.*]] = insertelement <16 x i32> [[TMP13]], i32 [[TMP]], i32 14
; CHECK-NEXT:    [[TMP15:%.*]] = insertelement <16 x i32> [[TMP14]], i32 [[TMP]], i32 15
; CHECK-NEXT:    [[TMP16:%.*]] = call i32 @llvm.vector.reduce.mul.v16i32(<16 x i32> [[TMP15]])
; CHECK-NEXT:    [[OP_RDX]] = mul i32 [[TMP16]], undef
; CHECK-NEXT:    br label [[BB1]]
;
bb:
  br label %bb1

bb1:                                              ; preds = %bb1, %bb
  %tmp = phi i32 [ undef, %bb1 ], [ undef, %bb ]
  %tmp2 = phi i32 [ %tmp18, %bb1 ], [ undef, %bb ]
  %tmp3 = mul i32 undef, %tmp
  %tmp4 = mul i32 %tmp3, %tmp
  %tmp5 = mul i32 %tmp4, %tmp
  %tmp6 = mul i32 %tmp5, %tmp
  %tmp7 = mul i32 %tmp6, %tmp
  %tmp8 = mul i32 %tmp7, %tmp
  %tmp9 = mul i32 %tmp8, %tmp
  %tmp10 = mul i32 %tmp9, %tmp
  %tmp11 = mul i32 %tmp10, %tmp
  %tmp12 = mul i32 %tmp11, %tmp
  %tmp13 = mul i32 %tmp12, %tmp
  %tmp14 = mul i32 %tmp13, %tmp
  %tmp15 = mul i32 %tmp14, %tmp
  %tmp16 = mul i32 %tmp15, %tmp
  %tmp17 = mul i32 %tmp16, %tmp
  %tmp18 = mul i32 %tmp17, %tmp
  br label %bb1
}

define void @test_2(i8 addrspace(1)* %arg, i32 %arg1) #0 {
; CHECK-LABEL: @test_2(
; CHECK-NEXT:  bb:
; CHECK-NEXT:    br label [[BB2:%.*]]
; CHECK:       bb2:
; CHECK-NEXT:    [[TMP:%.*]] = phi i32 [ undef, [[BB:%.*]] ], [ undef, [[BB2]] ]
; CHECK-NEXT:    [[TMP3:%.*]] = phi i32 [ 0, [[BB]] ], [ undef, [[BB2]] ]
; CHECK-NEXT:    [[TMP0:%.*]] = insertelement <8 x i32> poison, i32 [[TMP]], i32 0
; CHECK-NEXT:    [[TMP1:%.*]] = insertelement <8 x i32> [[TMP0]], i32 [[TMP]], i32 1
; CHECK-NEXT:    [[TMP2:%.*]] = insertelement <8 x i32> [[TMP1]], i32 [[TMP]], i32 2
; CHECK-NEXT:    [[TMP3:%.*]] = insertelement <8 x i32> [[TMP2]], i32 [[TMP]], i32 3
; CHECK-NEXT:    [[TMP4:%.*]] = insertelement <8 x i32> [[TMP3]], i32 [[TMP]], i32 4
; CHECK-NEXT:    [[TMP5:%.*]] = insertelement <8 x i32> [[TMP4]], i32 [[TMP]], i32 5
; CHECK-NEXT:    [[TMP6:%.*]] = insertelement <8 x i32> [[TMP5]], i32 [[TMP]], i32 6
; CHECK-NEXT:    [[TMP7:%.*]] = insertelement <8 x i32> [[TMP6]], i32 [[TMP]], i32 7
; CHECK-NEXT:    [[TMP8:%.*]] = call i32 @llvm.vector.reduce.add.v8i32(<8 x i32> undef)
; CHECK-NEXT:    [[TMP9:%.*]] = call i32 @llvm.vector.reduce.add.v8i32(<8 x i32> [[TMP7]])
; CHECK-NEXT:    [[OP_RDX:%.*]] = add i32 [[TMP8]], [[TMP9]]
; CHECK-NEXT:    [[OP_RDX1:%.*]] = add i32 [[OP_RDX]], undef
; CHECK-NEXT:    call void @use(i32 [[OP_RDX1]])
; CHECK-NEXT:    br label [[BB2]]
;
bb:
  br label %bb2

bb2:                                              ; preds = %bb2, %bb
  %tmp = phi i32 [ undef, %bb ], [ undef, %bb2 ]
  %tmp3 = phi i32 [ 0, %bb ], [ undef, %bb2 ]
  %tmp4 = add i32 %tmp, undef
  %tmp5 = add i32 undef, %tmp4
  %tmp6 = add i32 %tmp, %tmp5
  %tmp7 = add i32 undef, %tmp6
  %tmp8 = add i32 %tmp, %tmp7
  %tmp9 = add i32 undef, %tmp8
  %tmp10 = add i32 %tmp, %tmp9
  %tmp11 = add i32 undef, %tmp10
  %tmp12 = add i32 %tmp, %tmp11
  %tmp13 = add i32 undef, %tmp12
  %tmp14 = add i32 %tmp, %tmp13
  %tmp15 = add i32 undef, %tmp14
  %tmp16 = add i32 %tmp, %tmp15
  %tmp17 = add i32 undef, %tmp16
  %tmp18 = add i32 %tmp, %tmp17
  %tmp19 = add i32 undef, %tmp18
  call void @use(i32 %tmp19)
  br label %bb2
}

; Make sure we don't crash.
define i64 @test_3() #0 {
; CHECK-LABEL: @test_3(
bb:
  br label %bb1

bb1:                                              ; preds = %bb
  br label %bb3

bb2:                                              ; No predecessors!
  br label %bb3

bb3:                                              ; preds = %bb2, %bb1
  %tmp = phi i32 [ undef, %bb1 ], [ undef, %bb2 ]
  %tmp4 = phi i32 [ undef, %bb1 ], [ undef, %bb2 ]
  %tmp5 = mul i32 %tmp, %tmp4
  %tmp6 = mul i32 %tmp5, %tmp4
  %tmp7 = mul i32 %tmp6, %tmp4
  %tmp8 = mul i32 %tmp7, %tmp4
  %tmp9 = mul i32 %tmp8, %tmp4
  %tmp10 = mul i32 %tmp9, %tmp4
  %tmp11 = mul i32 %tmp10, %tmp4
  %tmp12 = mul i32 %tmp11, %tmp4
  %tmp13 = mul i32 %tmp12, %tmp4
  %tmp14 = mul i32 %tmp13, %tmp4
  %tmp15 = mul i32 %tmp14, %tmp4
  %tmp16 = mul i32 %tmp15, %tmp4
  %tmp17 = mul i32 %tmp16, %tmp4
  %tmp18 = mul i32 %tmp17, %tmp4
  %tmp19 = mul i32 %tmp18, %tmp4
  %tmp20 = mul i32 %tmp19, %tmp4
  %tmp21 = mul i32 %tmp20, %tmp4
  %tmp22 = mul i32 %tmp21, %tmp4
  %tmp23 = mul i32 %tmp22, %tmp4
  %tmp24 = mul i32 %tmp23, %tmp4
  %tmp25 = mul i32 %tmp24, %tmp4
  %tmp26 = mul i32 %tmp25, %tmp4
  %tmp27 = mul i32 %tmp26, %tmp4
  %tmp28 = mul i32 %tmp27, %tmp4
  %tmp29 = mul i32 %tmp28, %tmp4
  %tmp30 = mul i32 %tmp29, %tmp4
  %tmp31 = mul i32 %tmp30, %tmp4
  %tmp32 = mul i32 %tmp31, %tmp4
  %tmp33 = mul i32 %tmp32, %tmp4
  %tmp34 = mul i32 %tmp33, %tmp4
  %tmp35 = mul i32 %tmp34, %tmp4
  %tmp36 = mul i32 %tmp35, %tmp4
  %tmp37 = mul i32 %tmp36, %tmp4
  %tmp38 = mul i32 %tmp37, %tmp4
  %tmp39 = mul i32 %tmp38, %tmp4
  %tmp40 = mul i32 %tmp39, %tmp4
  %tmp41 = mul i32 %tmp40, %tmp4
  %tmp42 = mul i32 %tmp41, %tmp4
  %tmp43 = mul i32 %tmp42, %tmp4
  %tmp44 = mul i32 %tmp43, %tmp4
  %tmp45 = mul i32 %tmp44, %tmp4
  %tmp46 = mul i32 %tmp45, %tmp4
  %tmp47 = mul i32 %tmp46, %tmp4
  %tmp48 = mul i32 %tmp47, %tmp4
  %tmp49 = mul i32 %tmp48, %tmp4
  %tmp50 = mul i32 %tmp49, %tmp4
  %tmp51 = mul i32 %tmp50, %tmp4
  %tmp52 = mul i32 %tmp51, %tmp4
  %tmp53 = mul i32 %tmp52, %tmp4
  %tmp54 = mul i32 %tmp53, %tmp4
  %tmp55 = mul i32 %tmp54, %tmp4
  %tmp56 = mul i32 %tmp55, %tmp4
  %tmp57 = mul i32 %tmp56, %tmp4
  %tmp58 = mul i32 %tmp57, %tmp4
  %tmp59 = mul i32 %tmp58, %tmp4
  %tmp60 = mul i32 %tmp59, %tmp4
  %tmp61 = mul i32 %tmp60, %tmp4
  %tmp62 = mul i32 %tmp61, %tmp4
  %tmp63 = mul i32 %tmp62, %tmp4
  %tmp64 = add i32 undef, %tmp63
  %tmp65 = sext i32 %tmp64 to i64
  ret i64 %tmp65
}

declare void @use(i32) #0

attributes #0 = { "target-features"="+sse4.1" }
