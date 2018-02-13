; RUN: llc < %s -mtriple=amdgcn--amdhsa -mcpu=fiji -mattr=-flat-for-global | FileCheck --check-prefix=GCN %s

; Check that the waitcnt insertion algorithm correctly propagates wait counts
; from before a loop to the loop header.

; GCN-LABEL: {{^}}testKernel
; GCN: BB0_1:
; GCN: s_waitcnt vmcnt(0) lgkmcnt(0)
; GCN-NEXT: v_cmp_eq_f32_e64
; GCN: s_waitcnt vmcnt(0) lgkmcnt(0)
; GCN-NEXT: v_cmp_eq_f32_e32
; GCN: s_waitcnt vmcnt(0) lgkmcnt(0)
; GCN-NEXT: v_cmp_eq_f32_e32

@data_generic = addrspace(1) global [100 x float] [float 0.000000e+00, float 0x3FB99999A0000000, float 0x3FC99999A0000000, float 0x3FD3333340000000, float 0x3FD99999A0000000, float 5.000000e-01, float 0x3FE3333340000000, float 0x3FE6666660000000, float 0x3FE99999A0000000, float 0x3FECCCCCC0000000, float 1.000000e+00, float 0x3FF19999A0000000, float 0x3FF3333340000000, float 0x3FF4CCCCC0000000, float 0x3FF6666660000000, float 1.500000e+00, float 0x3FF99999A0000000, float 0x3FFB333340000000, float 0x3FFCCCCCC0000000, float 0x3FFE666660000000, float 2.000000e+00, float 0x4000CCCCC0000000, float 0x40019999A0000000, float 0x4002666660000000, float 0x4003333340000000, float 2.500000e+00, float 0x4004CCCCC0000000, float 0x40059999A0000000, float 0x4006666660000000, float 0x4007333340000000, float 3.000000e+00, float 0x4008CCCCC0000000, float 0x40099999A0000000, float 0x400A666660000000, float 0x400B333340000000, float 3.500000e+00, float 0x400CCCCCC0000000, float 0x400D9999A0000000, float 0x400E666660000000, float 0x400F333340000000, float 4.000000e+00, float 0x4010666660000000, float 0x4010CCCCC0000000, float 0x4011333340000000, float 0x40119999A0000000, float 4.500000e+00, float 0x4012666660000000, float 0x4012CCCCC0000000, float 0x4013333340000000, float 0x40139999A0000000, float 5.000000e+00, float 0x4014666660000000, float 0x4014CCCCC0000000, float 0x4015333340000000, float 0x40159999A0000000, float 5.500000e+00, float 0x4016666660000000, float 0x4016CCCCC0000000, float 0x4017333340000000, float 0x40179999A0000000, float 6.000000e+00, float 0x4018666660000000, float 0x4018CCCCC0000000, float 0x4019333340000000, float 0x40199999A0000000, float 6.500000e+00, float 0x401A666660000000, float 0x401ACCCCC0000000, float 0x401B333340000000, float 0x401B9999A0000000, float 7.000000e+00, float 0x401C666660000000, float 0x401CCCCCC0000000, float 0x401D333340000000, float 0x401D9999A0000000, float 7.500000e+00, float 0x401E666660000000, float 0x401ECCCCC0000000, float 0x401F333340000000, float 0x401F9999A0000000, float 8.000000e+00, float 0x4020333340000000, float 0x4020666660000000, float 0x40209999A0000000, float 0x4020CCCCC0000000, float 8.500000e+00, float 0x4021333340000000, float 0x4021666660000000, float 0x40219999A0000000, float 0x4021CCCCC0000000, float 9.000000e+00, float 0x4022333340000000, float 0x4022666660000000, float 0x40229999A0000000, float 0x4022CCCCC0000000, float 9.500000e+00, float 0x4023333340000000, float 0x4023666660000000, float 0x40239999A0000000, float 0x4023CCCCC0000000], align 4
@data_reference = addrspace(1) global [100 x float] [float 0.000000e+00, float 0x3FB99999A0000000, float 0x3FC99999A0000000, float 0x3FD3333340000000, float 0x3FD99999A0000000, float 5.000000e-01, float 0x3FE3333340000000, float 0x3FE6666660000000, float 0x3FE99999A0000000, float 0x3FECCCCCC0000000, float 1.000000e+00, float 0x3FF19999A0000000, float 0x3FF3333340000000, float 0x3FF4CCCCC0000000, float 0x3FF6666660000000, float 1.500000e+00, float 0x3FF99999A0000000, float 0x3FFB333340000000, float 0x3FFCCCCCC0000000, float 0x3FFE666660000000, float 2.000000e+00, float 0x4000CCCCC0000000, float 0x40019999A0000000, float 0x4002666660000000, float 0x4003333340000000, float 2.500000e+00, float 0x4004CCCCC0000000, float 0x40059999A0000000, float 0x4006666660000000, float 0x4007333340000000, float 3.000000e+00, float 0x4008CCCCC0000000, float 0x40099999A0000000, float 0x400A666660000000, float 0x400B333340000000, float 3.500000e+00, float 0x400CCCCCC0000000, float 0x400D9999A0000000, float 0x400E666660000000, float 0x400F333340000000, float 4.000000e+00, float 0x4010666660000000, float 0x4010CCCCC0000000, float 0x4011333340000000, float 0x40119999A0000000, float 4.500000e+00, float 0x4012666660000000, float 0x4012CCCCC0000000, float 0x4013333340000000, float 0x40139999A0000000, float 5.000000e+00, float 0x4014666660000000, float 0x4014CCCCC0000000, float 0x4015333340000000, float 0x40159999A0000000, float 5.500000e+00, float 0x4016666660000000, float 0x4016CCCCC0000000, float 0x4017333340000000, float 0x40179999A0000000, float 6.000000e+00, float 0x4018666660000000, float 0x4018CCCCC0000000, float 0x4019333340000000, float 0x40199999A0000000, float 6.500000e+00, float 0x401A666660000000, float 0x401ACCCCC0000000, float 0x401B333340000000, float 0x401B9999A0000000, float 7.000000e+00, float 0x401C666660000000, float 0x401CCCCCC0000000, float 0x401D333340000000, float 0x401D9999A0000000, float 7.500000e+00, float 0x401E666660000000, float 0x401ECCCCC0000000, float 0x401F333340000000, float 0x401F9999A0000000, float 8.000000e+00, float 0x4020333340000000, float 0x4020666660000000, float 0x40209999A0000000, float 0x4020CCCCC0000000, float 8.500000e+00, float 0x4021333340000000, float 0x4021666660000000, float 0x40219999A0000000, float 0x4021CCCCC0000000, float 9.000000e+00, float 0x4022333340000000, float 0x4022666660000000, float 0x40229999A0000000, float 0x4022CCCCC0000000, float 9.500000e+00, float 0x4023333340000000, float 0x4023666660000000, float 0x40239999A0000000, float 0x4023CCCCC0000000], align 4

define amdgpu_kernel void @testKernel(i32 addrspace(1)* nocapture %arg) local_unnamed_addr #0 {
bb:
  store <2 x float> <float 1.000000e+00, float 1.000000e+00>, <2 x float>* bitcast (float* getelementptr ([100 x float], [100 x float]* addrspacecast ([100 x float] addrspace(1)* @data_generic to [100 x float]*), i64 0, i64 4) to <2 x float>*), align 4
  store <2 x float> <float 1.000000e+00, float 1.000000e+00>, <2 x float>* bitcast (float* getelementptr ([100 x float], [100 x float]* addrspacecast ([100 x float] addrspace(1)* @data_reference to [100 x float]*), i64 0, i64 4) to <2 x float>*), align 4
  br label %bb18

bb1:                                              ; preds = %bb18
  %tmp = tail call i8 addrspace(4)* @llvm.amdgcn.dispatch.ptr()
  %tmp2 = tail call i32 @llvm.amdgcn.workitem.id.x()
  %tmp3 = tail call i32 @llvm.amdgcn.workgroup.id.x()
  %tmp4 = getelementptr inbounds i8, i8 addrspace(4)* %tmp, i64 4
  %tmp5 = bitcast i8 addrspace(4)* %tmp4 to i16 addrspace(4)*
  %tmp6 = load i16, i16 addrspace(4)* %tmp5, align 4
  %tmp7 = zext i16 %tmp6 to i32
  %tmp8 = mul i32 %tmp3, %tmp7
  %tmp9 = add i32 %tmp8, %tmp2
  %tmp10 = tail call i8 addrspace(4)* @llvm.amdgcn.implicitarg.ptr()
  %tmp11 = zext i32 %tmp9 to i64
  %tmp12 = bitcast i8 addrspace(4)* %tmp10 to i64 addrspace(4)*
  %tmp13 = load i64, i64 addrspace(4)* %tmp12, align 8
  %tmp14 = add i64 %tmp13, %tmp11
  %tmp15 = zext i1 %tmp99 to i32
  %tmp16 = and i64 %tmp14, 4294967295
  %tmp17 = getelementptr inbounds i32, i32 addrspace(1)* %arg, i64 %tmp16
  store i32 %tmp15, i32 addrspace(1)* %tmp17, align 4
  ret void

bb18:                                             ; preds = %bb18, %bb
  %tmp19 = phi i64 [ 0, %bb ], [ %tmp102, %bb18 ]
  %tmp20 = phi i32 [ 0, %bb ], [ %tmp100, %bb18 ]
  %tmp21 = phi i1 [ true, %bb ], [ %tmp99, %bb18 ]
  %tmp22 = getelementptr inbounds [100 x float], [100 x float] addrspace(1)* @data_generic, i64 0, i64 %tmp19
  %tmp23 = load float, float addrspace(1)* %tmp22, align 4
  %tmp24 = getelementptr inbounds [100 x float], [100 x float] addrspace(1)* @data_reference, i64 0, i64 %tmp19
  %tmp25 = load float, float addrspace(1)* %tmp24, align 4
  %tmp26 = fcmp oeq float %tmp23, %tmp25
  %tmp27 = and i1 %tmp21, %tmp26
  %tmp28 = or i32 %tmp20, 1
  %tmp29 = sext i32 %tmp28 to i64
  %tmp30 = getelementptr inbounds [100 x float], [100 x float] addrspace(1)* @data_generic, i64 0, i64 %tmp29
  %tmp31 = load float, float addrspace(1)* %tmp30, align 4
  %tmp32 = getelementptr inbounds [100 x float], [100 x float] addrspace(1)* @data_reference, i64 0, i64 %tmp29
  %tmp33 = load float, float addrspace(1)* %tmp32, align 4
  %tmp34 = fcmp oeq float %tmp31, %tmp33
  %tmp35 = and i1 %tmp27, %tmp34
  %tmp36 = add nuw nsw i32 %tmp20, 2
  %tmp37 = sext i32 %tmp36 to i64
  %tmp38 = getelementptr inbounds [100 x float], [100 x float] addrspace(1)* @data_generic, i64 0, i64 %tmp37
  %tmp39 = load float, float addrspace(1)* %tmp38, align 4
  %tmp40 = getelementptr inbounds [100 x float], [100 x float] addrspace(1)* @data_reference, i64 0, i64 %tmp37
  %tmp41 = load float, float addrspace(1)* %tmp40, align 4
  %tmp42 = fcmp oeq float %tmp39, %tmp41
  %tmp43 = and i1 %tmp35, %tmp42
  %tmp44 = add nuw nsw i32 %tmp20, 3
  %tmp45 = sext i32 %tmp44 to i64
  %tmp46 = getelementptr inbounds [100 x float], [100 x float] addrspace(1)* @data_generic, i64 0, i64 %tmp45
  %tmp47 = load float, float addrspace(1)* %tmp46, align 4
  %tmp48 = getelementptr inbounds [100 x float], [100 x float] addrspace(1)* @data_reference, i64 0, i64 %tmp45
  %tmp49 = load float, float addrspace(1)* %tmp48, align 4
  %tmp50 = fcmp oeq float %tmp47, %tmp49
  %tmp51 = and i1 %tmp43, %tmp50
  %tmp52 = add nuw nsw i32 %tmp20, 4
  %tmp53 = sext i32 %tmp52 to i64
  %tmp54 = getelementptr inbounds [100 x float], [100 x float] addrspace(1)* @data_generic, i64 0, i64 %tmp53
  %tmp55 = load float, float addrspace(1)* %tmp54, align 4
  %tmp56 = getelementptr inbounds [100 x float], [100 x float] addrspace(1)* @data_reference, i64 0, i64 %tmp53
  %tmp57 = load float, float addrspace(1)* %tmp56, align 4
  %tmp58 = fcmp oeq float %tmp55, %tmp57
  %tmp59 = and i1 %tmp51, %tmp58
  %tmp60 = add nuw nsw i32 %tmp20, 5
  %tmp61 = sext i32 %tmp60 to i64
  %tmp62 = getelementptr inbounds [100 x float], [100 x float] addrspace(1)* @data_generic, i64 0, i64 %tmp61
  %tmp63 = load float, float addrspace(1)* %tmp62, align 4
  %tmp64 = getelementptr inbounds [100 x float], [100 x float] addrspace(1)* @data_reference, i64 0, i64 %tmp61
  %tmp65 = load float, float addrspace(1)* %tmp64, align 4
  %tmp66 = fcmp oeq float %tmp63, %tmp65
  %tmp67 = and i1 %tmp59, %tmp66
  %tmp68 = add nuw nsw i32 %tmp20, 6
  %tmp69 = sext i32 %tmp68 to i64
  %tmp70 = getelementptr inbounds [100 x float], [100 x float] addrspace(1)* @data_generic, i64 0, i64 %tmp69
  %tmp71 = load float, float addrspace(1)* %tmp70, align 4
  %tmp72 = getelementptr inbounds [100 x float], [100 x float] addrspace(1)* @data_reference, i64 0, i64 %tmp69
  %tmp73 = load float, float addrspace(1)* %tmp72, align 4
  %tmp74 = fcmp oeq float %tmp71, %tmp73
  %tmp75 = and i1 %tmp67, %tmp74
  %tmp76 = add nuw nsw i32 %tmp20, 7
  %tmp77 = sext i32 %tmp76 to i64
  %tmp78 = getelementptr inbounds [100 x float], [100 x float] addrspace(1)* @data_generic, i64 0, i64 %tmp77
  %tmp79 = load float, float addrspace(1)* %tmp78, align 4
  %tmp80 = getelementptr inbounds [100 x float], [100 x float] addrspace(1)* @data_reference, i64 0, i64 %tmp77
  %tmp81 = load float, float addrspace(1)* %tmp80, align 4
  %tmp82 = fcmp oeq float %tmp79, %tmp81
  %tmp83 = and i1 %tmp75, %tmp82
  %tmp84 = add nuw nsw i32 %tmp20, 8
  %tmp85 = sext i32 %tmp84 to i64
  %tmp86 = getelementptr inbounds [100 x float], [100 x float] addrspace(1)* @data_generic, i64 0, i64 %tmp85
  %tmp87 = load float, float addrspace(1)* %tmp86, align 4
  %tmp88 = getelementptr inbounds [100 x float], [100 x float] addrspace(1)* @data_reference, i64 0, i64 %tmp85
  %tmp89 = load float, float addrspace(1)* %tmp88, align 4
  %tmp90 = fcmp oeq float %tmp87, %tmp89
  %tmp91 = and i1 %tmp83, %tmp90
  %tmp92 = add nuw nsw i32 %tmp20, 9
  %tmp93 = sext i32 %tmp92 to i64
  %tmp94 = getelementptr inbounds [100 x float], [100 x float] addrspace(1)* @data_generic, i64 0, i64 %tmp93
  %tmp95 = load float, float addrspace(1)* %tmp94, align 4
  %tmp96 = getelementptr inbounds [100 x float], [100 x float] addrspace(1)* @data_reference, i64 0, i64 %tmp93
  %tmp97 = load float, float addrspace(1)* %tmp96, align 4
  %tmp98 = fcmp oeq float %tmp95, %tmp97
  %tmp99 = and i1 %tmp91, %tmp98
  %tmp100 = add nuw nsw i32 %tmp20, 10
  %tmp101 = icmp eq i32 %tmp100, 100
  %tmp102 = sext i32 %tmp100 to i64
  br i1 %tmp101, label %bb1, label %bb18
}

; Function Attrs: nounwind readnone speculatable
declare i8 addrspace(4)* @llvm.amdgcn.dispatch.ptr() #1

; Function Attrs: nounwind readnone speculatable
declare i32 @llvm.amdgcn.workitem.id.x() #1

; Function Attrs: nounwind readnone speculatable
declare i32 @llvm.amdgcn.workgroup.id.x() #1

; Function Attrs: nounwind readnone speculatable
declare i8 addrspace(4)* @llvm.amdgcn.implicitarg.ptr() #1

attributes #0 = { "target-cpu"="fiji" "target-features"="-flat-for-global" }
attributes #1 = { nounwind readnone speculatable }
