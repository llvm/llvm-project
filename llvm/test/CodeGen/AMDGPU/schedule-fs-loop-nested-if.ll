; RUN: llc -march=r600 -mcpu=cayman -stress-sched -verify-misched -verify-machineinstrs < %s
; REQUIRES: asserts

define amdgpu_vs void @main(<4 x float> inreg %reg0, <4 x float> inreg %reg1) #0 {
main_body:
  %tmp = extractelement <4 x float> %reg1, i32 0
  %tmp5 = extractelement <4 x float> %reg1, i32 1
  %tmp6 = extractelement <4 x float> %reg1, i32 2
  %tmp7 = extractelement <4 x float> %reg1, i32 3
  %tmp8 = fcmp ult float %tmp5, 0.000000e+00
  %tmp9 = select i1 %tmp8, float 1.000000e+00, float 0.000000e+00
  %tmp10 = fsub float -0.000000e+00, %tmp9
  %tmp11 = fptosi float %tmp10 to i32
  %tmp12 = bitcast i32 %tmp11 to float
  %tmp13 = fcmp ult float %tmp, 5.700000e+01
  %tmp14 = select i1 %tmp13, float 1.000000e+00, float 0.000000e+00
  %tmp15 = fsub float -0.000000e+00, %tmp14
  %tmp16 = fptosi float %tmp15 to i32
  %tmp17 = bitcast i32 %tmp16 to float
  %tmp18 = bitcast float %tmp12 to i32
  %tmp19 = bitcast float %tmp17 to i32
  %tmp20 = and i32 %tmp18, %tmp19
  %tmp21 = bitcast i32 %tmp20 to float
  %tmp22 = bitcast float %tmp21 to i32
  %tmp23 = icmp ne i32 %tmp22, 0
  %tmp24 = fcmp ult float %tmp, 0.000000e+00
  %tmp25 = select i1 %tmp24, float 1.000000e+00, float 0.000000e+00
  %tmp26 = fsub float -0.000000e+00, %tmp25
  %tmp27 = fptosi float %tmp26 to i32
  %tmp28 = bitcast i32 %tmp27 to float
  %tmp29 = bitcast float %tmp28 to i32
  %tmp30 = icmp ne i32 %tmp29, 0
  br i1 %tmp23, label %IF, label %ELSE

IF:                                               ; preds = %main_body
  %. = select i1 %tmp30, float 0.000000e+00, float 1.000000e+00
  %.18 = select i1 %tmp30, float 1.000000e+00, float 0.000000e+00
  br label %ENDIF

ELSE:                                             ; preds = %main_body
  br i1 %tmp30, label %ENDIF, label %ELSE17

ENDIF:                                            ; preds = %ELSE17, %ELSE, %IF
  %temp1.0 = phi float [ %., %IF ], [ %tmp48, %ELSE17 ], [ 0.000000e+00, %ELSE ]
  %temp2.0 = phi float [ 0.000000e+00, %IF ], [ %tmp49, %ELSE17 ], [ 1.000000e+00, %ELSE ]
  %temp.0 = phi float [ %.18, %IF ], [ %tmp47, %ELSE17 ], [ 0.000000e+00, %ELSE ]
  %max.0.i = call float @llvm.maxnum.f32(float %temp.0, float 0.000000e+00)
  %clamp.i = call float @llvm.minnum.f32(float %max.0.i, float 1.000000e+00)
  %max.0.i3 = call float @llvm.maxnum.f32(float %temp1.0, float 0.000000e+00)
  %clamp.i4 = call float @llvm.minnum.f32(float %max.0.i3, float 1.000000e+00)
  %max.0.i1 = call float @llvm.maxnum.f32(float %temp2.0, float 0.000000e+00)
  %clamp.i2 = call float @llvm.minnum.f32(float %max.0.i1, float 1.000000e+00)
  %tmp31 = insertelement <4 x float> undef, float %clamp.i, i32 0
  %tmp32 = insertelement <4 x float> %tmp31, float %clamp.i4, i32 1
  %tmp33 = insertelement <4 x float> %tmp32, float %clamp.i2, i32 2
  %tmp34 = insertelement <4 x float> %tmp33, float 1.000000e+00, i32 3
  call void @llvm.r600.store.swizzle(<4 x float> %tmp34, i32 0, i32 0)
  ret void

ELSE17:                                           ; preds = %ELSE
  %tmp35 = fadd float 0.000000e+00, 0x3FC99999A0000000
  %tmp36 = fadd float 0.000000e+00, 0x3FC99999A0000000
  %tmp37 = fadd float 0.000000e+00, 0x3FC99999A0000000
  %tmp38 = fadd float %tmp35, 0x3FC99999A0000000
  %tmp39 = fadd float %tmp36, 0x3FC99999A0000000
  %tmp40 = fadd float %tmp37, 0x3FC99999A0000000
  %tmp41 = fadd float %tmp38, 0x3FC99999A0000000
  %tmp42 = fadd float %tmp39, 0x3FC99999A0000000
  %tmp43 = fadd float %tmp40, 0x3FC99999A0000000
  %tmp44 = fadd float %tmp41, 0x3FC99999A0000000
  %tmp45 = fadd float %tmp42, 0x3FC99999A0000000
  %tmp46 = fadd float %tmp43, 0x3FC99999A0000000
  %tmp47 = fadd float %tmp44, 0x3FC99999A0000000
  %tmp48 = fadd float %tmp45, 0x3FC99999A0000000
  %tmp49 = fadd float %tmp46, 0x3FC99999A0000000
  br label %ENDIF
}

declare float @llvm.minnum.f32(float, float) #1
declare float @llvm.maxnum.f32(float, float) #1

declare void @llvm.r600.store.swizzle(<4 x float>, i32, i32)

attributes #0 = { nounwind }
attributes #1 = { nounwind readnone }
