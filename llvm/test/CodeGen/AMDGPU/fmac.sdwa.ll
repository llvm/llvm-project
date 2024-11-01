; RUN: llc -march=amdgcn -mcpu=gfx1010 -verify-machineinstrs < %s | FileCheck -check-prefixes=GCN,GFX1010 %s
; RUN: llc -march=amdgcn -mcpu=gfx1100 -verify-machineinstrs < %s | FileCheck -check-prefixes=GCN,GFX1010 %s

; GCN-LABEL: {{^}}addMul2D:
; GFX1010: v_fmac_f16
; GFX1010: v_fmac_f16
define hidden <4 x half> @addMul2D(ptr nocapture readonly %arg, ptr addrspace(4) nocapture readonly %arg1, <2 x i32> %arg2, i32 %arg3) local_unnamed_addr #0 {
bb:
  %tmp = extractelement <2 x i32> %arg2, i64 1
  %tmp4 = icmp sgt i32 %tmp, 0
  br i1 %tmp4, label %bb5, label %bb36

bb5:                                              ; preds = %bb
  %tmp6 = extractelement <2 x i32> %arg2, i64 0
  %tmp7 = icmp sgt i32 %tmp6, 0
  br label %bb8

bb8:                                              ; preds = %bb32, %bb5
  %tmp9 = phi <4 x half> [ zeroinitializer, %bb5 ], [ %tmp33, %bb32 ]
  %tmp10 = phi i32 [ 0, %bb5 ], [ %tmp34, %bb32 ]
  br i1 %tmp7, label %bb11, label %bb32

bb11:                                             ; preds = %bb8
  %tmp12 = mul nsw i32 %tmp10, %arg3
  %tmp13 = mul nsw i32 %tmp10, %tmp6
  br label %bb14

bb14:                                             ; preds = %bb14, %bb11
  %tmp15 = phi <4 x half> [ %tmp9, %bb11 ], [ %tmp29, %bb14 ]
  %tmp16 = phi i32 [ 0, %bb11 ], [ %tmp30, %bb14 ]
  %tmp17 = add nsw i32 %tmp16, %tmp12
  %tmp18 = sext i32 %tmp17 to i64
  %tmp19 = getelementptr inbounds <4 x i8>, ptr %arg, i64 %tmp18
  %tmp20 = load <4 x i8>, ptr %tmp19, align 4
  %tmp21 = tail call <4 x half> @_Z13convert_half4Dv4_h(<4 x i8> %tmp20)
  %tmp22 = add nsw i32 %tmp16, %tmp13
  %tmp23 = sext i32 %tmp22 to i64
  %tmp24 = getelementptr inbounds float, ptr addrspace(4) %arg1, i64 %tmp23
  %tmp25 = load float, ptr addrspace(4) %tmp24, align 4
  %tmp26 = fptrunc float %tmp25 to half
  %tmp27 = insertelement <4 x half> undef, half %tmp26, i32 0
  %tmp28 = shufflevector <4 x half> %tmp27, <4 x half> undef, <4 x i32> zeroinitializer
  %vec.A.0 = extractelement <4 x half> %tmp21, i32 0
  %vec.B.0 = extractelement <4 x half> %tmp28, i32 0
  %vec.C.0 = extractelement <4 x half> %tmp15, i32 0
  %vec.res.0 = tail call half @llvm.fmuladd.f16(half %vec.A.0, half %vec.B.0, half %vec.C.0)
  %vec.A.1 = extractelement <4 x half> %tmp21, i32 1
  %vec.B.1 = extractelement <4 x half> %tmp28, i32 1
  %vec.C.1 = extractelement <4 x half> %tmp15, i32 1
  %vec.res.1 = tail call half @llvm.fmuladd.f16(half %vec.A.1, half %vec.B.1, half %vec.C.1)
  %vec.A.2 = extractelement <4 x half> %tmp21, i32 2
  %vec.B.2 = extractelement <4 x half> %tmp28, i32 2
  %vec.C.2 = extractelement <4 x half> %tmp15, i32 2
  %vec.res.2 = tail call half @llvm.fmuladd.f16(half %vec.A.2, half %vec.B.2, half %vec.C.2)
  %vec.A.3 = extractelement <4 x half> %tmp21, i32 3
  %vec.B.3 = extractelement <4 x half> %tmp28, i32 3
  %vec.C.3 = extractelement <4 x half> %tmp15, i32 3
  %vec.res.3 = tail call half @llvm.fmuladd.f16(half %vec.A.3, half %vec.B.3, half %vec.C.3)
  %full.res.0 = insertelement <4 x half> undef, half %vec.res.0, i32 0
  %full.res.1 = insertelement <4 x half> %full.res.0, half %vec.res.1, i32 1
  %full.res.2 = insertelement <4 x half> %full.res.1, half %vec.res.2, i32 2
  %tmp29 = insertelement <4 x half> %full.res.2, half %vec.res.3, i32 3
  %tmp30 = add nuw nsw i32 %tmp16, 1
  %tmp31 = icmp eq i32 %tmp30, %tmp6
  br i1 %tmp31, label %bb32, label %bb14

bb32:                                             ; preds = %bb14, %bb8
  %tmp33 = phi <4 x half> [ %tmp9, %bb8 ], [ %tmp29, %bb14 ]
  %tmp34 = add nuw nsw i32 %tmp10, 1
  %tmp35 = icmp eq i32 %tmp34, %tmp
  br i1 %tmp35, label %bb36, label %bb8

bb36:                                             ; preds = %bb32, %bb
  %tmp37 = phi <4 x half> [ zeroinitializer, %bb ], [ %tmp33, %bb32 ]
  ret <4 x half> %tmp37
}

; Function Attrs: norecurse nounwind readnone
define linkonce_odr hidden <4 x half> @_Z13convert_half4Dv4_h(<4 x i8> %arg) local_unnamed_addr #1 {
bb:
  %tmp = extractelement <4 x i8> %arg, i64 0
  %tmp1 = uitofp i8 %tmp to half
  %tmp2 = insertelement <4 x half> undef, half %tmp1, i32 0
  %tmp3 = extractelement <4 x i8> %arg, i64 1
  %tmp4 = uitofp i8 %tmp3 to half
  %tmp5 = insertelement <4 x half> %tmp2, half %tmp4, i32 1
  %tmp6 = extractelement <4 x i8> %arg, i64 2
  %tmp7 = uitofp i8 %tmp6 to half
  %tmp8 = insertelement <4 x half> %tmp5, half %tmp7, i32 2
  %tmp9 = extractelement <4 x i8> %arg, i64 3
  %tmp10 = uitofp i8 %tmp9 to half
  %tmp11 = insertelement <4 x half> %tmp8, half %tmp10, i32 3
  ret <4 x half> %tmp11
}

declare half @llvm.fmuladd.f16(half, half, half)

attributes #0 = { convergent nounwind readonly}
attributes #1 = { norecurse nounwind readnone }
