; RUN: llc -march=amdgcn < %s | FileCheck %s
; REQUIRES: asserts
;
; This testcase used to cause the following crash:
;
; *** Couldn't join subrange!
;
; UNREACHABLE executed at lib/CodeGen/RegisterCoalescer.cpp:2666!
;
; The insertelement instructions became subregister definitions: one virtual
; register was defined and re-defined by one group of the consecutive insert-
; elements, and another was defined by the second group.
; Since a copy between the two full registers was present in the program,
; the coalescer tried to merge them. The join algorithm for the main range
; decided that it was correct to do so, while the subrange join unexpectedly
; failed. This was caused by the live interval subranges not being computed
; correctly: subregister defs are not uses for the purpose of subranges.
;
; Test for a valid output:
; CHECK: image_sample_c_d_o

target triple = "amdgcn--"

define amdgpu_ps <{ i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, float, float, float, float, float, float, float, float, float, float, float, float, float, float }> @main([17 x <16 x i8>] addrspace(2)* byval dereferenceable(18446744073709551615) %arg, [16 x <16 x i8>] addrspace(2)* byval dereferenceable(18446744073709551615) %arg1, [32 x <8 x i32>] addrspace(2)* byval dereferenceable(18446744073709551615) %arg2, [16 x <8 x i32>] addrspace(2)* byval dereferenceable(18446744073709551615) %arg3, [16 x <4 x i32>] addrspace(2)* byval dereferenceable(18446744073709551615) %arg4, float inreg %arg5, i32 inreg %arg6, <2 x i32> %arg7, <2 x i32> %arg8, <2 x i32> %arg9, <3 x i32> %arg10, <2 x i32> %arg11, <2 x i32> %arg12, <2 x i32> %arg13, float %arg14, float %arg15, float %arg16, float %arg17, float %arg18, i32 %arg19, i32 %arg20, float %arg21, i32 %arg22) #0 {
main_body:
  %tmp = call float @llvm.SI.fs.interp(i32 3, i32 0, i32 %arg6, <2 x i32> %arg8)
  %tmp23 = fadd float %tmp, 0xBFA99999A0000000
  %tmp24 = fadd float %tmp, 0x3FA99999A0000000
  %tmp25 = bitcast float %tmp23 to i32
  %tmp26 = insertelement <16 x i32> <i32 212739, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>, i32 %tmp25, i32 1
  %tmp27 = insertelement <16 x i32> %tmp26, i32 undef, i32 2
  %tmp28 = insertelement <16 x i32> %tmp27, i32 undef, i32 3
  %tmp29 = insertelement <16 x i32> %tmp28, i32 undef, i32 4
  %tmp30 = insertelement <16 x i32> %tmp29, i32 0, i32 5
  %tmp31 = insertelement <16 x i32> %tmp30, i32 undef, i32 6
  %tmp32 = insertelement <16 x i32> %tmp31, i32 undef, i32 7
  %tmp33 = insertelement <16 x i32> %tmp32, i32 undef, i32 8
  %tmp34 = call <4 x float> @llvm.SI.image.sample.c.d.o.v16i32(<16 x i32> %tmp33, <8 x i32> undef, <4 x i32> undef, i32 15, i32 0, i32 0, i32 1, i32 0, i32 0, i32 0, i32 0)
  %tmp35 = extractelement <4 x float> %tmp34, i32 0
  %tmp36 = bitcast float %tmp24 to i32
  %tmp37 = insertelement <16 x i32> <i32 212739, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>, i32 %tmp36, i32 1
  %tmp38 = insertelement <16 x i32> %tmp37, i32 undef, i32 2
  %tmp39 = insertelement <16 x i32> %tmp38, i32 undef, i32 3
  %tmp40 = insertelement <16 x i32> %tmp39, i32 undef, i32 4
  %tmp41 = insertelement <16 x i32> %tmp40, i32 0, i32 5
  %tmp42 = insertelement <16 x i32> %tmp41, i32 undef, i32 6
  %tmp43 = insertelement <16 x i32> %tmp42, i32 undef, i32 7
  %tmp44 = insertelement <16 x i32> %tmp43, i32 undef, i32 8
  %tmp45 = call <4 x float> @llvm.SI.image.sample.c.d.o.v16i32(<16 x i32> %tmp44, <8 x i32> undef, <4 x i32> undef, i32 15, i32 0, i32 0, i32 1, i32 0, i32 0, i32 0, i32 0)
  %tmp46 = extractelement <4 x float> %tmp45, i32 0
  %tmp47 = fmul float %tmp35, %tmp46
  %tmp48 = insertvalue <{ i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, float, float, float, float, float, float, float, float, float, float, float, float, float, float }> undef, float %tmp47, 14
  %tmp49 = insertvalue <{ i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, float, float, float, float, float, float, float, float, float, float, float, float, float, float }> %tmp48, float %arg21, 24
  ret <{ i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, float, float, float, float, float, float, float, float, float, float, float, float, float, float }> %tmp49
}

declare float @llvm.SI.fs.interp(i32, i32, i32, <2 x i32>) #1
declare float @llvm.SI.load.const(<16 x i8>, i32) #1
declare <4 x float> @llvm.SI.image.sample.c.d.o.v16i32(<16 x i32>, <8 x i32>, <4 x i32>, i32, i32, i32, i32, i32, i32, i32, i32) #1

attributes #0 = { "InitialPSInputAddr"="36983" "target-cpu"="tonga" }
attributes #1 = { nounwind readnone }
