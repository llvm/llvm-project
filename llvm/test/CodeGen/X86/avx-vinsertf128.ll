; RUN: llc < %s -mtriple=x86_64-apple-darwin -mcpu=corei7-avx -mattr=+avx | FileCheck %s
; RUN: llc < %s -mtriple=x86_64-apple-darwin -mcpu=corei7-avx -mattr=+avx | FileCheck -check-prefix=CHECK-SSE %s

; CHECK-NOT: vunpck
; CHECK: vinsertf128 $1
define <8 x float> @A(<8 x float> %a) nounwind uwtable readnone ssp {
entry:
  %shuffle = shufflevector <8 x float> %a, <8 x float> undef, <8 x i32> <i32 8, i32 8, i32 8, i32 8, i32 0, i32 1, i32 2, i32 3>
  ret <8 x float> %shuffle
}

; CHECK-NOT: vunpck
; CHECK: vinsertf128 $1
define <4 x double> @B(<4 x double> %a) nounwind uwtable readnone ssp {
entry:
  %shuffle = shufflevector <4 x double> %a, <4 x double> undef, <4 x i32> <i32 4, i32 4, i32 0, i32 1>
  ret <4 x double> %shuffle
}

declare <2 x double> @llvm.x86.sse2.min.pd(<2 x double>, <2 x double>) nounwind readnone

declare <2 x double> @llvm.x86.sse2.min.sd(<2 x double>, <2 x double>) nounwind readnone

; Just check that no crash happens
; CHECK-SSE: _insert_crash
define void @insert_crash() nounwind {
allocas:
  %v1.i.i451 = shufflevector <4 x double> zeroinitializer, <4 x double> undef, <4 x i32> <i32 2, i32 3, i32 undef, i32 undef>
  %ret_0a.i.i.i452 = shufflevector <4 x double> %v1.i.i451, <4 x double> undef, <2 x i32> <i32 0, i32 1>
  %vret_0.i.i.i454 = tail call <2 x double> @llvm.x86.sse2.min.pd(<2 x double> %ret_0a.i.i.i452, <2 x double> undef) nounwind
  %ret_val.i.i.i463 = tail call <2 x double> @llvm.x86.sse2.min.sd(<2 x double> %vret_0.i.i.i454, <2 x double> undef) nounwind
  %ret.i1.i.i464 = extractelement <2 x double> %ret_val.i.i.i463, i32 0
  %double2float = fptrunc double %ret.i1.i.i464 to float
  %smearinsert50 = insertelement <4 x float> undef, float %double2float, i32 3
  %blendAsInt.i503 = bitcast <4 x float> %smearinsert50 to <4 x i32>
  store <4 x i32> %blendAsInt.i503, <4 x i32>* undef, align 4
  ret void
}

;; DAG Combine must remove useless vinsertf128 instructions

; CHECK: DAGCombineA
; CHECK-NOT: vinsertf128 $1
define <4 x i32> @DAGCombineA(<4 x i32> %v1) nounwind readonly {
  %1 = shufflevector <4 x i32> %v1, <4 x i32> undef, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %2 = shufflevector <8 x i32> %1, <8 x i32> undef, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  ret <4 x i32> %2
}

; CHECK: DAGCombineB
; CHECK: vpaddd %xmm
; CHECK-NOT: vinsertf128  $1
; CHECK: vpaddd %xmm
define <8 x i32> @DAGCombineB(<8 x i32> %v1, <8 x i32> %v2) nounwind readonly {
  %1 = add <8 x i32> %v1, %v2
  %2 = add <8 x i32> %1, %v1
  ret <8 x i32> %2
}

; CHECK: insert_pd
define <4 x double> @insert_pd(<4 x double> %a0, <2 x double> %a1) {
; CHECK: vinsertf128
%res = call <4 x double> @llvm.x86.avx.vinsertf128.pd.256(<4 x double> %a0, <2 x double> %a1, i8 0)
ret <4 x double> %res
}

; CHECK: insert_undef_pd
define <4 x double> @insert_undef_pd(<4 x double> %a0, <2 x double> %a1) {
; CHECK: vmovaps	%ymm1, %ymm0
%res = call <4 x double> @llvm.x86.avx.vinsertf128.pd.256(<4 x double> undef, <2 x double> %a1, i8 0)
ret <4 x double> %res
}
declare <4 x double> @llvm.x86.avx.vinsertf128.pd.256(<4 x double>, <2 x double>, i8) nounwind readnone


; CHECK: insert_ps
define <8 x float> @insert_ps(<8 x float> %a0, <4 x float> %a1) {
; CHECK: vinsertf128
%res = call <8 x float> @llvm.x86.avx.vinsertf128.ps.256(<8 x float> %a0, <4 x float> %a1, i8 0)
ret <8 x float> %res
}

; CHECK: insert_undef_ps
define <8 x float> @insert_undef_ps(<8 x float> %a0, <4 x float> %a1) {
; CHECK: vmovaps	%ymm1, %ymm0
%res = call <8 x float> @llvm.x86.avx.vinsertf128.ps.256(<8 x float> undef, <4 x float> %a1, i8 0)
ret <8 x float> %res
}
declare <8 x float> @llvm.x86.avx.vinsertf128.ps.256(<8 x float>, <4 x float>, i8) nounwind readnone


; CHECK: insert_si
define <8 x i32> @insert_si(<8 x i32> %a0, <4 x i32> %a1) {
; CHECK: vinsertf128
%res = call <8 x i32> @llvm.x86.avx.vinsertf128.si.256(<8 x i32> %a0, <4 x i32> %a1, i8 0)
ret <8 x i32> %res
}

; CHECK: insert_undef_si
define <8 x i32> @insert_undef_si(<8 x i32> %a0, <4 x i32> %a1) {
; CHECK: vmovaps	%ymm1, %ymm0
%res = call <8 x i32> @llvm.x86.avx.vinsertf128.si.256(<8 x i32> undef, <4 x i32> %a1, i8 0)
ret <8 x i32> %res
}
declare <8 x i32> @llvm.x86.avx.vinsertf128.si.256(<8 x i32>, <4 x i32>, i8) nounwind readnone

; rdar://10643481
; CHECK: vinsertf128_combine
define <8 x float> @vinsertf128_combine(float* nocapture %f) nounwind uwtable readonly ssp {
; CHECK-NOT: vmovaps
; CHECK: vinsertf128
entry:
  %add.ptr = getelementptr inbounds float* %f, i64 4
  %0 = bitcast float* %add.ptr to <4 x float>*
  %1 = load <4 x float>* %0, align 16
  %2 = tail call <8 x float> @llvm.x86.avx.vinsertf128.ps.256(<8 x float> undef, <4 x float> %1, i8 1)
  ret <8 x float> %2
}

; rdar://11076953
; CHECK: vinsertf128_ucombine
define <8 x float> @vinsertf128_ucombine(float* nocapture %f) nounwind uwtable readonly ssp {
; CHECK-NOT: vmovups
; CHECK: vinsertf128
entry:
  %add.ptr = getelementptr inbounds float* %f, i64 4
  %0 = bitcast float* %add.ptr to <4 x float>*
  %1 = load <4 x float>* %0, align 8
  %2 = tail call <8 x float> @llvm.x86.avx.vinsertf128.ps.256(<8 x float> undef, <4 x float> %1, i8 1)
  ret <8 x float> %2
}
