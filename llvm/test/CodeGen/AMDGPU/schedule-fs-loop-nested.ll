; RUN: llc -mtriple=r600 -mcpu=cayman -stress-sched -verify-misched -verify-machineinstrs < %s
; REQUIRES: asserts

define amdgpu_kernel void @main() #0 {
main_body:
  %tmp = load <4 x float>, ptr addrspace(9) null
  %tmp5 = extractelement <4 x float> %tmp, i32 3
  %tmp6 = fptosi float %tmp5 to i32
  %tmp7 = bitcast i32 %tmp6 to float
  %tmp8 = bitcast float %tmp7 to i32
  %tmp9 = sdiv i32 %tmp8, 4
  %tmp10 = bitcast i32 %tmp9 to float
  %tmp11 = bitcast float %tmp10 to i32
  %tmp12 = mul i32 %tmp11, 4
  %tmp13 = bitcast i32 %tmp12 to float
  %tmp14 = bitcast float %tmp13 to i32
  %tmp15 = sub i32 0, %tmp14
  %tmp16 = bitcast i32 %tmp15 to float
  %tmp17 = bitcast float %tmp7 to i32
  %tmp18 = bitcast float %tmp16 to i32
  %tmp19 = add i32 %tmp17, %tmp18
  %tmp20 = bitcast i32 %tmp19 to float
  %tmp21 = load <4 x float>, ptr addrspace(9) null
  %tmp22 = extractelement <4 x float> %tmp21, i32 0
  %tmp23 = load <4 x float>, ptr addrspace(9) null
  %tmp24 = extractelement <4 x float> %tmp23, i32 1
  %tmp25 = load <4 x float>, ptr addrspace(9) null
  %tmp26 = extractelement <4 x float> %tmp25, i32 2
  br label %LOOP

LOOP:                                             ; preds = %IF31, %main_body
  %temp12.0 = phi float [ 0.000000e+00, %main_body ], [ %tmp47, %IF31 ]
  %temp6.0 = phi float [ %tmp26, %main_body ], [ %temp6.1, %IF31 ]
  %temp5.0 = phi float [ %tmp24, %main_body ], [ %temp5.1, %IF31 ]
  %temp4.0 = phi float [ %tmp22, %main_body ], [ %temp4.1, %IF31 ]
  %tmp27 = bitcast float %temp12.0 to i32
  %tmp28 = bitcast float %tmp10 to i32
  %tmp29 = icmp sge i32 %tmp27, %tmp28
  %tmp30 = sext i1 %tmp29 to i32
  %tmp31 = bitcast i32 %tmp30 to float
  %tmp32 = bitcast float %tmp31 to i32
  %tmp33 = icmp ne i32 %tmp32, 0
  br i1 %tmp33, label %IF, label %LOOP29

IF:                                               ; preds = %LOOP
  %max.0.i = call float @llvm.maxnum.f32(float %temp4.0, float 0.000000e+00)
  %clamp.i = call float @llvm.minnum.f32(float %max.0.i, float 1.000000e+00)
  %max.0.i3 = call float @llvm.maxnum.f32(float %temp5.0, float 0.000000e+00)
  %clamp.i4 = call float @llvm.minnum.f32(float %max.0.i3, float 1.000000e+00)
  %max.0.i1 = call float @llvm.maxnum.f32(float %temp6.0, float 0.000000e+00)
  %clamp.i2 = call float @llvm.minnum.f32(float %max.0.i1, float 1.000000e+00)
  %tmp34 = insertelement <4 x float> undef, float %clamp.i, i32 0
  %tmp35 = insertelement <4 x float> %tmp34, float %clamp.i4, i32 1
  %tmp36 = insertelement <4 x float> %tmp35, float %clamp.i2, i32 2
  %tmp37 = insertelement <4 x float> %tmp36, float 1.000000e+00, i32 3
  call void @llvm.r600.store.swizzle(<4 x float> %tmp37, i32 0, i32 0)
  ret void

LOOP29:                                           ; preds = %ENDIF30, %LOOP
  %temp6.1 = phi float [ %temp4.1, %ENDIF30 ], [ %temp6.0, %LOOP ]
  %temp5.1 = phi float [ %temp6.1, %ENDIF30 ], [ %temp5.0, %LOOP ]
  %temp4.1 = phi float [ %temp5.1, %ENDIF30 ], [ %temp4.0, %LOOP ]
  %temp20.0 = phi float [ %tmp50, %ENDIF30 ], [ 0.000000e+00, %LOOP ]
  %tmp38 = bitcast float %temp20.0 to i32
  %tmp39 = bitcast float %tmp20 to i32
  %tmp40 = icmp sge i32 %tmp38, %tmp39
  %tmp41 = sext i1 %tmp40 to i32
  %tmp42 = bitcast i32 %tmp41 to float
  %tmp43 = bitcast float %tmp42 to i32
  %tmp44 = icmp ne i32 %tmp43, 0
  br i1 %tmp44, label %IF31, label %ENDIF30

IF31:                                             ; preds = %LOOP29
  %tmp45 = bitcast float %temp12.0 to i32
  %tmp46 = add i32 %tmp45, 1
  %tmp47 = bitcast i32 %tmp46 to float
  br label %LOOP

ENDIF30:                                          ; preds = %LOOP29
  %tmp48 = bitcast float %temp20.0 to i32
  %tmp49 = add i32 %tmp48, 1
  %tmp50 = bitcast i32 %tmp49 to float
  br label %LOOP29
}

declare void @llvm.r600.store.swizzle(<4 x float>, i32, i32) #0
declare float @llvm.minnum.f32(float, float) #1
declare float @llvm.maxnum.f32(float, float) #1

attributes #0 = { nounwind }
attributes #1 = { nounwind readnone }
