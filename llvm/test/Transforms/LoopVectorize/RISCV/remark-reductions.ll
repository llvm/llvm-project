; RUN: opt < %s -mtriple=riscv64 -mattr=+v -p loop-vectorize -pass-remarks-analysis=loop-vectorize -S 2>&1 | FileCheck %s

; CHECK: remark: <unknown>:0:0: the cost-model indicates that interleaving is not beneficial
define float @s311(float %a_0, float %s311_sum) {
; CHECK-LABEL: define float @s311(
; CHECK-SAME: float [[A_0:%.*]], float [[S311_SUM:%.*]]) #[[ATTR0:[0-9]+]] {
; CHECK-NEXT:  [[ENTRY:.*]]:
; CHECK-NEXT:    [[TMP0:%.*]] = call i32 @llvm.vscale.i32()
; CHECK-NEXT:    [[TMP1:%.*]] = mul i32 [[TMP0]], 4
; CHECK-NEXT:    [[MIN_ITERS_CHECK:%.*]] = icmp ult i32 1200, [[TMP1]]
; CHECK-NEXT:    br i1 [[MIN_ITERS_CHECK]], label %[[SCALAR_PH:.*]], label %[[VECTOR_PH:.*]]
; CHECK:       [[VECTOR_PH]]:
; CHECK-NEXT:    [[TMP2:%.*]] = call i32 @llvm.vscale.i32()
; CHECK-NEXT:    [[TMP3:%.*]] = mul i32 [[TMP2]], 4
; CHECK-NEXT:    [[N_MOD_VF:%.*]] = urem i32 1200, [[TMP3]]
; CHECK-NEXT:    [[N_VEC:%.*]] = sub i32 1200, [[N_MOD_VF]]
; CHECK-NEXT:    [[TMP4:%.*]] = call i32 @llvm.vscale.i32()
; CHECK-NEXT:    [[TMP5:%.*]] = mul i32 [[TMP4]], 4
; CHECK-NEXT:    [[BROADCAST_SPLATINSERT:%.*]] = insertelement <vscale x 4 x float> poison, float [[A_0]], i64 0
; CHECK-NEXT:    [[BROADCAST_SPLAT:%.*]] = shufflevector <vscale x 4 x float> [[BROADCAST_SPLATINSERT]], <vscale x 4 x float> poison, <vscale x 4 x i32> zeroinitializer
; CHECK-NEXT:    br label %[[VECTOR_BODY:.*]]
; CHECK:       [[VECTOR_BODY]]:
; CHECK-NEXT:    [[INDEX:%.*]] = phi i32 [ 0, %[[VECTOR_PH]] ], [ [[INDEX_NEXT:%.*]], %[[VECTOR_BODY]] ]
; CHECK-NEXT:    [[VEC_PHI:%.*]] = phi float [ [[S311_SUM]], %[[VECTOR_PH]] ], [ [[TMP6:%.*]], %[[VECTOR_BODY]] ]
; CHECK-NEXT:    [[TMP6]] = call float @llvm.vector.reduce.fadd.nxv4f32(float [[VEC_PHI]], <vscale x 4 x float> [[BROADCAST_SPLAT]])
; CHECK-NEXT:    [[INDEX_NEXT]] = add nuw i32 [[INDEX]], [[TMP5]]
; CHECK-NEXT:    [[TMP7:%.*]] = icmp eq i32 [[INDEX_NEXT]], [[N_VEC]]
; CHECK-NEXT:    br i1 [[TMP7]], label %[[MIDDLE_BLOCK:.*]], label %[[VECTOR_BODY]], !llvm.loop [[LOOP0:![0-9]+]]
; CHECK:       [[MIDDLE_BLOCK]]:
; CHECK-NEXT:    [[CMP_N:%.*]] = icmp eq i32 1200, [[N_VEC]]
; CHECK-NEXT:    br i1 [[CMP_N]], label %[[EXIT:.*]], label %[[SCALAR_PH]]
; CHECK:       [[SCALAR_PH]]:
; CHECK-NEXT:    [[BC_RESUME_VAL:%.*]] = phi i32 [ [[N_VEC]], %[[MIDDLE_BLOCK]] ], [ 0, %[[ENTRY]] ]
; CHECK-NEXT:    [[BC_MERGE_RDX:%.*]] = phi float [ [[TMP6]], %[[MIDDLE_BLOCK]] ], [ [[S311_SUM]], %[[ENTRY]] ]
; CHECK-NEXT:    br label %[[LOOP:.*]]
; CHECK:       [[LOOP]]:
; CHECK-NEXT:    [[IV:%.*]] = phi i32 [ [[BC_RESUME_VAL]], %[[SCALAR_PH]] ], [ [[IV_NEXT:%.*]], %[[LOOP]] ]
; CHECK-NEXT:    [[RED:%.*]] = phi float [ [[BC_MERGE_RDX]], %[[SCALAR_PH]] ], [ [[RED_NEXT:%.*]], %[[LOOP]] ]
; CHECK-NEXT:    [[RED_NEXT]] = fadd float [[A_0]], [[RED]]
; CHECK-NEXT:    [[IV_NEXT]] = add nuw nsw i32 [[IV]], 1
; CHECK-NEXT:    [[EXITCOND:%.*]] = icmp eq i32 [[IV_NEXT]], 1200
; CHECK-NEXT:    br i1 [[EXITCOND]], label %[[EXIT]], label %[[LOOP]], !llvm.loop [[LOOP3:![0-9]+]]
; CHECK:       [[EXIT]]:
; CHECK-NEXT:    [[RED_LCSSA:%.*]] = phi float [ [[RED_NEXT]], %[[LOOP]] ], [ [[TMP6]], %[[MIDDLE_BLOCK]] ]
; CHECK-NEXT:    ret float [[RED_LCSSA]]
;
entry:
  br label %loop

loop:
  %iv = phi i32 [ 0, %entry ], [ %iv.next, %loop ]
  %red = phi float [ %s311_sum, %entry ], [ %red.next, %loop ]
  %red.next = fadd float %a_0, %red
  %iv.next = add nuw nsw i32 %iv, 1
  %exitcond = icmp eq i32 %iv.next, 1200
  br i1 %exitcond, label %exit, label %loop

exit:
  %red.lcssa = phi float [ %red.next, %loop ]
  ret float %red.lcssa
}
;.
; CHECK: [[LOOP0]] = distinct !{[[LOOP0]], [[META1:![0-9]+]], [[META2:![0-9]+]]}
; CHECK: [[META1]] = !{!"llvm.loop.isvectorized", i32 1}
; CHECK: [[META2]] = !{!"llvm.loop.unroll.runtime.disable"}
; CHECK: [[LOOP3]] = distinct !{[[LOOP3]], [[META2]], [[META1]]}
;.
