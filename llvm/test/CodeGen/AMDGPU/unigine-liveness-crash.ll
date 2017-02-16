; RUN: llc -march=amdgcn -verify-machineinstrs < %s | FileCheck %s
;
; This test used to crash with the following assertion:
; llc: include/llvm/ADT/IntervalMap.h:632: unsigned int llvm::IntervalMapImpl::LeafNode<llvm::SlotIndex, llvm::LiveInterval *, 8, llvm::IntervalMapInfo<llvm::SlotIndex> >::insertFrom(unsigned int &, unsigned int, KeyT, KeyT, ValT) [KeyT = llvm::SlotIndex, ValT = llvm::LiveInterval *, N = 8, Traits = llvm::IntervalMapInfo<llvm::SlotIndex>]: Assertion `(i == Size || Traits::stopLess(b, start(i))) && "Overlapping insert"' failed.
;
; This was related to incorrectly calculating subregister live ranges
; (i.e. live interval subranges): subregister defs are not uses for that
; purpose.
;
; Check for a valid output.
; CHECK: image_sample_c
define amdgpu_ps <{ i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, float, float, float, float, float, float, float, float, float, float, float, float, float, float }> @main([17 x <16 x i8>] addrspace(2)* byval dereferenceable(18446744073709551615) %arg, [16 x <16 x i8>] addrspace(2)* byval dereferenceable(18446744073709551615) %arg1, [32 x <8 x i32>] addrspace(2)* byval dereferenceable(18446744073709551615) %arg2, [16 x <8 x i32>] addrspace(2)* byval dereferenceable(18446744073709551615) %arg3, [16 x <4 x i32>] addrspace(2)* byval dereferenceable(18446744073709551615) %arg4, float inreg %arg5, i32 inreg %arg6, <2 x i32> %arg7, <2 x i32> %arg8, <2 x i32> %arg9, <3 x i32> %arg10, <2 x i32> %arg11, <2 x i32> %arg12, <2 x i32> %arg13, float %arg14, float %arg15, float %arg16, float %arg17, float %arg18, i32 %arg19, i32 %arg20, float %arg21, i32 %arg22) #0 {
main_body:
  %i.i = extractelement <2 x i32> %arg8, i32 0
  %j.i = extractelement <2 x i32> %arg8, i32 1
  %i.f.i = bitcast i32 %i.i to float
  %j.f.i = bitcast i32 %j.i to float
  %p1.i = call float @llvm.amdgcn.interp.p1(float %i.f.i, i32 3, i32 4, i32 %arg6) #2
  %p2.i = call float @llvm.amdgcn.interp.p2(float %p1.i, float %j.f.i, i32 3, i32 4, i32 %arg6) #2
  %tmp23 = call <4 x float> @llvm.SI.image.sample.v2i32(<2 x i32> undef, <8 x i32> undef, <4 x i32> undef, i32 15, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0)
  %tmp24 = extractelement <4 x float> %tmp23, i32 3
  %tmp25 = fmul float %tmp24, undef
  %tmp26 = fmul float undef, %p2.i
  %tmp27 = fadd float %tmp26, undef
  %tmp28 = bitcast float %tmp27 to i32
  %tmp29 = insertelement <4 x i32> undef, i32 %tmp28, i32 0
  %tmp30 = insertelement <4 x i32> %tmp29, i32 0, i32 1
  %tmp31 = insertelement <4 x i32> %tmp30, i32 undef, i32 2
  %tmp32 = call <4 x float> @llvm.SI.image.sample.c.v4i32(<4 x i32> %tmp31, <8 x i32> undef, <4 x i32> undef, i32 15, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0)
  %tmp33 = extractelement <4 x float> %tmp32, i32 0
  %tmp34 = fadd float undef, %tmp33
  %tmp35 = fadd float %tmp34, undef
  %tmp36 = fadd float %tmp35, undef
  %tmp37 = fadd float %tmp36, undef
  %tmp38 = fadd float %tmp37, undef
  %tmp39 = call <4 x float> @llvm.SI.image.sample.v4i32(<4 x i32> undef, <8 x i32> undef, <4 x i32> undef, i32 15, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0)
  %tmp40 = extractelement <4 x float> %tmp39, i32 0
  %tmp41 = extractelement <4 x float> %tmp39, i32 1
  %tmp42 = extractelement <4 x float> %tmp39, i32 2
  %tmp43 = extractelement <4 x float> %tmp39, i32 3
  %tmp44 = fmul float %tmp40, undef
  %tmp45 = fmul float %tmp41, undef
  %tmp46 = fmul float %tmp42, undef
  %tmp47 = fmul float %tmp43, undef
  %tmp48 = fadd float undef, %tmp44
  %tmp49 = fadd float undef, %tmp45
  %tmp50 = bitcast float %tmp27 to i32
  %tmp51 = bitcast float %tmp48 to i32
  %tmp52 = bitcast float %tmp49 to i32
  %tmp53 = insertelement <4 x i32> undef, i32 %tmp50, i32 0
  %tmp54 = insertelement <4 x i32> %tmp53, i32 %tmp51, i32 1
  %tmp55 = insertelement <4 x i32> %tmp54, i32 %tmp52, i32 2
  %tmp56 = call <4 x float> @llvm.SI.image.sample.c.v4i32(<4 x i32> %tmp55, <8 x i32> undef, <4 x i32> undef, i32 15, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0)
  %tmp57 = extractelement <4 x float> %tmp56, i32 0
  %tmp58 = fadd float %tmp38, %tmp57
  %tmp59 = fadd float undef, %tmp46
  %tmp60 = fadd float undef, %tmp47
  %tmp61 = bitcast float %tmp59 to i32
  %tmp62 = bitcast float %tmp60 to i32
  %tmp63 = insertelement <4 x i32> undef, i32 %tmp61, i32 1
  %tmp64 = insertelement <4 x i32> %tmp63, i32 %tmp62, i32 2
  %tmp65 = call <4 x float> @llvm.SI.image.sample.c.v4i32(<4 x i32> %tmp64, <8 x i32> undef, <4 x i32> undef, i32 15, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0)
  %tmp66 = extractelement <4 x float> %tmp65, i32 0
  %tmp67 = fadd float %tmp58, %tmp66
  %tmp68 = fmul float %tmp67, 1.250000e-01
  %tmp69 = fmul float %tmp68, undef
  %tmp70 = fcmp une float %tmp69, 0.000000e+00
  br i1 %tmp70, label %IF26, label %ENDIF25

IF26:                                             ; preds = %main_body
  %tmp71 = bitcast float %tmp27 to i32
  %tmp72 = insertelement <4 x i32> undef, i32 %tmp71, i32 0
  br label %LOOP

ENDIF25:                                          ; preds = %IF29, %main_body
  %.4 = phi float [ %tmp84, %IF29 ], [ %tmp68, %main_body ]
  %tmp73 = fadd float %.4, undef
  %tmp74 = call float @llvm.AMDGPU.clamp.(float %tmp73, float 0.000000e+00, float 1.000000e+00)
  %tmp75 = fmul float undef, %tmp74
  %tmp76 = fmul float %tmp75, undef
  %tmp77 = fadd float %tmp76, undef
  %tmp78 = insertvalue <{ i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, float, float, float, float, float, float, float, float, float, float, float, float, float, float }> undef, float %tmp77, 11
  %tmp79 = insertvalue <{ i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, float, float, float, float, float, float, float, float, float, float, float, float, float, float }> %tmp78, float undef, 12
  %tmp80 = insertvalue <{ i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, float, float, float, float, float, float, float, float, float, float, float, float, float, float }> %tmp79, float undef, 13
  %tmp81 = insertvalue <{ i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, float, float, float, float, float, float, float, float, float, float, float, float, float, float }> %tmp80, float %tmp25, 14
  %tmp82 = insertvalue <{ i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, float, float, float, float, float, float, float, float, float, float, float, float, float, float }> %tmp81, float undef, 15
  %tmp83 = insertvalue <{ i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, float, float, float, float, float, float, float, float, float, float, float, float, float, float }> %tmp82, float %arg21, 24
  ret <{ i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, float, float, float, float, float, float, float, float, float, float, float, float, float, float }> %tmp83

LOOP:                                             ; preds = %ENDIF28, %IF26
  %.5 = phi float [ undef, %IF26 ], [ %tmp89, %ENDIF28 ]
  br i1 false, label %IF29, label %ENDIF28

IF29:                                             ; preds = %LOOP
  %tmp84 = fmul float %.5, 3.125000e-02
  br label %ENDIF25

ENDIF28:                                          ; preds = %LOOP
  %tmp85 = insertelement <4 x i32> %tmp72, i32 undef, i32 1
  %tmp86 = insertelement <4 x i32> %tmp85, i32 undef, i32 2
  %tmp87 = call <4 x float> @llvm.SI.image.sample.c.v4i32(<4 x i32> %tmp86, <8 x i32> undef, <4 x i32> undef, i32 15, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0)
  %tmp88 = extractelement <4 x float> %tmp87, i32 0
  %tmp89 = fadd float undef, %tmp88
  br label %LOOP
}

; Function Attrs: nounwind readnone
declare float @llvm.AMDGPU.clamp.(float, float, float) #1

; Function Attrs: nounwind readnone
declare <4 x float> @llvm.SI.image.sample.v2i32(<2 x i32>, <8 x i32>, <4 x i32>, i32, i32, i32, i32, i32, i32, i32, i32) #1

; Function Attrs: nounwind readnone
declare <4 x float> @llvm.SI.image.sample.v4i32(<4 x i32>, <8 x i32>, <4 x i32>, i32, i32, i32, i32, i32, i32, i32, i32) #1

; Function Attrs: nounwind readnone
declare <4 x float> @llvm.SI.image.sample.c.v4i32(<4 x i32>, <8 x i32>, <4 x i32>, i32, i32, i32, i32, i32, i32, i32, i32) #1

; Function Attrs: nounwind readnone
declare float @llvm.amdgcn.interp.p1(float, i32, i32, i32) #1

; Function Attrs: nounwind readnone
declare float @llvm.amdgcn.interp.p2(float, float, i32, i32, i32) #1

attributes #0 = { "InitialPSInputAddr"="36983" "target-cpu"="tonga" }
attributes #1 = { nounwind readnone }
attributes #2 = { nounwind }
