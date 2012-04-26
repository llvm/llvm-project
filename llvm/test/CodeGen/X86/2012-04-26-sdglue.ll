; RUN: llc < %s -mtriple=x86_64-apple-darwin -mcpu=core-avx2 -mattr=+avx | FileCheck %s
; rdar://11314175: SD Scheduler, BuildSchedUnits assert:
;                  N->getNodeId() == -1 && "Node already inserted!

; It's hard to test for the ISEL condition because CodeGen optimizes
; away the bugpointed code. Just ensure the basics are still there.
;CHECK: func:
;CHECK: vmovups
;CHECK: vpshufd
;CHECK: vpshufd
;CHECK: vmulps
;CHECK: vmulps
;CHECK: ret

define void @func() nounwind ssp {
  %tmp = load <4 x float>* null, align 1
  %tmp14 = getelementptr <4 x float>* null, i32 2
  %tmp15 = load <4 x float>* %tmp14, align 1
  %tmp16 = shufflevector <4 x float> %tmp, <4 x float> <float 0.000000e+00, float undef, float undef, float undef>, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 4, i32 4, i32 4>
  %tmp17 = call <8 x float> @llvm.x86.avx.vinsertf128.ps.256(<8 x float> %tmp16, <4 x float> undef, i8 1)
  %tmp18 = bitcast <4 x float> %tmp to <16 x i8>
  %tmp19 = shufflevector <16 x i8> %tmp18, <16 x i8> undef, <16 x i32> <i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19>
  %tmp20 = bitcast <16 x i8> %tmp19 to <4 x float>
  %tmp21 = bitcast <4 x float> %tmp15 to <16 x i8>
  %tmp22 = shufflevector <16 x i8> undef, <16 x i8> %tmp21, <16 x i32> <i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19>
  %tmp23 = bitcast <16 x i8> %tmp22 to <4 x float>
  %tmp24 = shufflevector <4 x float> %tmp20, <4 x float> <float 0.000000e+00, float undef, float undef, float undef>, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 4, i32 4, i32 4>
  %tmp25 = call <8 x float> @llvm.x86.avx.vinsertf128.ps.256(<8 x float> %tmp24, <4 x float> %tmp23, i8 1)
  %tmp26 = fmul <8 x float> %tmp17, undef
  %tmp27 = fmul <8 x float> %tmp25, undef
  %tmp28 = fadd <8 x float> %tmp26, %tmp27
  %tmp29 = fadd <8 x float> %tmp28, undef
  %tmp30 = shufflevector <8 x float> %tmp29, <8 x float> undef, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  %tmp31 = fmul <4 x float> undef, %tmp30
  %tmp32 = call <8 x float> @llvm.x86.avx.vinsertf128.ps.256(<8 x float> zeroinitializer, <4 x float> %tmp31, i8 1)
  %tmp33 = fadd <8 x float> undef, %tmp32
  %tmp34 = call <8 x float> @llvm.x86.avx.hadd.ps.256(<8 x float> %tmp33, <8 x float> undef) nounwind
  %tmp35 = fsub <8 x float> %tmp34, undef
  %tmp36 = call <8 x float> @llvm.x86.avx.hadd.ps.256(<8 x float> zeroinitializer, <8 x float> %tmp35) nounwind
  store <8 x float> %tmp36, <8 x float>* undef, align 32
  ret void
}

declare <8 x float> @llvm.x86.avx.vinsertf128.ps.256(<8 x float>, <4 x float>, i8) nounwind readnone

declare <8 x float> @llvm.x86.avx.hadd.ps.256(<8 x float>, <8 x float>) nounwind readnone
