; RUN: opt -S -mtriple=amdgcn-- -amdgpu-codegenprepare %s | FileCheck %s
; RUN: opt -S -amdgpu-codegenprepare %s | FileCheck -check-prefix=NOOP %s
; Make sure this doesn't crash with no triple

; NOOP-LABEL: @noop_fdiv_fpmath(
; NOOP: %md.25ulp = fdiv float %a, %b, !fpmath !0
define amdgpu_kernel void @noop_fdiv_fpmath(float addrspace(1)* %out, float %a, float %b) #3 {
  %md.25ulp = fdiv float %a, %b, !fpmath !0
  store volatile float %md.25ulp, float addrspace(1)* %out
  ret void
}

; CHECK-LABEL: @fdiv_fpmath(
; CHECK: %no.md = fdiv float %a, %b{{$}}
; CHECK: %md.half.ulp = fdiv float %a, %b, !fpmath !1
; CHECK: %md.1ulp = fdiv float %a, %b, !fpmath !2
; CHECK: %md.25ulp = call float @llvm.amdgcn.fdiv.fast(float %a, float %b), !fpmath !0
; CHECK: %md.3ulp = call float @llvm.amdgcn.fdiv.fast(float %a, float %b), !fpmath !3
; CHECK: %[[FAST_RCP:[0-9]+]] = call fast float @llvm.amdgcn.rcp.f32(float %b)
; CHECK: %fast.md.25ulp = fmul fast float %a, %[[FAST_RCP]], !fpmath !0
; CHECK: %[[ARCP_RCP:[0-9]+]] = call arcp float @llvm.amdgcn.rcp.f32(float %b)
; CHECK: arcp.md.25ulp = fmul arcp float %a, %[[ARCP_RCP]], !fpmath !0
define amdgpu_kernel void @fdiv_fpmath(float addrspace(1)* %out, float %a, float %b) #1 {
  %no.md = fdiv float %a, %b
  store volatile float %no.md, float addrspace(1)* %out

  %md.half.ulp = fdiv float %a, %b, !fpmath !1
  store volatile float %md.half.ulp, float addrspace(1)* %out

  %md.1ulp = fdiv float %a, %b, !fpmath !2
  store volatile float %md.1ulp, float addrspace(1)* %out

  %md.25ulp = fdiv float %a, %b, !fpmath !0
  store volatile float %md.25ulp, float addrspace(1)* %out

  %md.3ulp = fdiv float %a, %b, !fpmath !3
  store volatile float %md.3ulp, float addrspace(1)* %out

  %fast.md.25ulp = fdiv fast float %a, %b, !fpmath !0
  store volatile float %fast.md.25ulp, float addrspace(1)* %out

  %arcp.md.25ulp = fdiv arcp float %a, %b, !fpmath !0
  store volatile float %arcp.md.25ulp, float addrspace(1)* %out

  ret void
}

; CHECK-LABEL: @rcp_fdiv_fpmath(
; CHECK: %no.md = fdiv float 1.000000e+00, %x{{$}}
; CHECK: %md.25ulp = call float @llvm.amdgcn.rcp.f32(float %x)
; CHECK: %md.half.ulp = fdiv float 1.000000e+00, %x, !fpmath !1
; CHECK: %arcp.no.md = fdiv arcp float 1.000000e+00, %x
; CHECK: %arcp.25ulp = call arcp float @llvm.amdgcn.rcp.f32(float %x)
; CHECK: %fast.no.md = call fast float @llvm.amdgcn.rcp.f32(float %x)
; CHECK: %fast.25ulp = call fast float @llvm.amdgcn.rcp.f32(float %x)
define amdgpu_kernel void @rcp_fdiv_fpmath(float addrspace(1)* %out, float %x) #1 {
  %no.md = fdiv float 1.0, %x
  store volatile float %no.md, float addrspace(1)* %out

  %md.25ulp = fdiv float 1.0, %x, !fpmath !0
  store volatile float %md.25ulp, float addrspace(1)* %out

  %md.half.ulp = fdiv float 1.0, %x, !fpmath !1
  store volatile float %md.half.ulp, float addrspace(1)* %out

  %arcp.no.md = fdiv arcp float 1.0, %x
  store volatile float %arcp.no.md, float addrspace(1)* %out

  %arcp.25ulp = fdiv arcp float 1.0, %x, !fpmath !0
  store volatile float %arcp.25ulp, float addrspace(1)* %out

  %fast.no.md = fdiv fast float 1.0, %x
  store volatile float %fast.no.md, float addrspace(1)* %out

  %fast.25ulp = fdiv fast float 1.0, %x, !fpmath !0
  store volatile float %fast.25ulp, float addrspace(1)* %out

  ret void
}

; CHECK-LABEL: @rcp_fdiv_arcp_denormal(
; CHECK: %arcp.low.accuracy = call arcp float @llvm.amdgcn.fdiv.fast(float 1.000000e+00, float %x), !fpmath !0
; CHECK: %arcp.high.accuracy = fdiv arcp float 1.000000e+00, %x, !fpmath !2
; CHECK: %arcp.low.afn = call arcp afn float @llvm.amdgcn.rcp.f32(float %x)
; CHECK: %arcp.high.afn = call arcp afn float @llvm.amdgcn.rcp.f32(float %x)
define amdgpu_kernel void @rcp_fdiv_arcp_denormal(float addrspace(1)* %out, float %x) #2 {

  %arcp.low.accuracy = fdiv arcp float 1.0, %x, !fpmath !0
  store volatile float %arcp.low.accuracy, float addrspace(1)* %out

  %arcp.high.accuracy = fdiv arcp float 1.0, %x, !fpmath !2
  store volatile float %arcp.high.accuracy, float addrspace(1)* %out

  %arcp.low.afn = fdiv arcp afn float 1.0, %x, !fpmath !0
  store volatile float %arcp.low.afn, float addrspace(1)* %out

  %arcp.high.afn = fdiv arcp afn float 1.0, %x, !fpmath !2
  store volatile float %arcp.high.afn, float addrspace(1)* %out

  ret void
}

; CHECK-LABEL: @fdiv_fpmath_vector(
; CHECK: %[[NO_A0:[0-9]+]] = extractelement <2 x float> %a, i64 0
; CHECK: %[[NO_B0:[0-9]+]] = extractelement <2 x float> %b, i64 0
; CHECK: %[[NO_FDIV0:[0-9]+]] = fdiv float %[[NO_A0]], %[[NO_B0]]
; CHECK: %[[NO_INS0:[0-9]+]] = insertelement <2 x float> undef, float %[[NO_FDIV0]], i64 0
; CHECK: %[[NO_A1:[0-9]+]] = extractelement <2 x float> %a, i64 1
; CHECK: %[[NO_B1:[0-9]+]] = extractelement <2 x float> %b, i64 1
; CHECK: %[[NO_FDIV1:[0-9]+]] = fdiv float %[[NO_A1]], %[[NO_B1]]
; CHECK: %no.md = insertelement <2 x float> %[[NO_INS0]], float %[[NO_FDIV1]], i64 1
; CHECK: store volatile <2 x float> %no.md, <2 x float> addrspace(1)* %out

; CHECK: %[[HALF_A0:[0-9]+]] = extractelement <2 x float> %a, i64 0
; CHECK: %[[HALF_B0:[0-9]+]] = extractelement <2 x float> %b, i64 0
; CHECK: %[[HALF_FDIV0:[0-9]+]] = fdiv float %[[HALF_A0]], %[[HALF_B0]], !fpmath !1
; CHECK: %[[HALF_INS0:[0-9]+]] = insertelement <2 x float> undef, float %[[HALF_FDIV0]], i64 0
; CHECK: %[[HALF_A1:[0-9]+]] = extractelement <2 x float> %a, i64 1
; CHECK: %[[HALF_B1:[0-9]+]] = extractelement <2 x float> %b, i64 1
; CHECK: %[[HALF_FDIV1:[0-9]+]] = fdiv float %[[HALF_A1]], %[[HALF_B1]], !fpmath !1
; CHECK: %md.half.ulp = insertelement <2 x float> %[[HALF_INS0]], float %[[HALF_FDIV1]], i64 1
; CHECK: store volatile <2 x float> %md.half.ulp, <2 x float> addrspace(1)* %out

; CHECK: %[[ONE_A0:[0-9]+]] = extractelement <2 x float> %a, i64 0
; CHECK: %[[ONE_B0:[0-9]+]] = extractelement <2 x float> %b, i64 0
; CHECK: %[[ONE_FDIV0:[0-9]+]] = fdiv float %[[ONE_A0]], %[[ONE_B0]], !fpmath !2
; CHECK: %[[ONE_INS0:[0-9]+]] = insertelement <2 x float> undef, float %[[ONE_FDIV0]], i64 0
; CHECK: %[[ONE_A1:[0-9]+]] = extractelement <2 x float> %a, i64 1
; CHECK: %[[ONE_B1:[0-9]+]] = extractelement <2 x float> %b, i64 1
; CHECK: %[[ONE_FDIV1:[0-9]+]] = fdiv float %[[ONE_A1]], %[[ONE_B1]], !fpmath !2
; CHECK: %md.1ulp = insertelement <2 x float> %[[ONE_INS0]], float %[[ONE_FDIV1]], i64 1
; CHECK: store volatile <2 x float> %md.1ulp, <2 x float> addrspace(1)* %out

; CHECK: %[[A0:[0-9]+]] = extractelement <2 x float> %a, i64 0
; CHECK: %[[B0:[0-9]+]] = extractelement <2 x float> %b, i64 0
; CHECK: %[[FDIV0:[0-9]+]] = call float @llvm.amdgcn.fdiv.fast(float %[[A0]], float %[[B0]]), !fpmath !0
; CHECK: %[[INS0:[0-9]+]] = insertelement <2 x float> undef, float %[[FDIV0]], i64 0
; CHECK: %[[A1:[0-9]+]] = extractelement <2 x float> %a, i64 1
; CHECK: %[[B1:[0-9]+]] = extractelement <2 x float> %b, i64 1
; CHECK: %[[FDIV1:[0-9]+]] = call float @llvm.amdgcn.fdiv.fast(float %[[A1]], float %[[B1]]), !fpmath !0
; CHECK: %md.25ulp = insertelement <2 x float> %[[INS0]], float %[[FDIV1]], i64 1
define amdgpu_kernel void @fdiv_fpmath_vector(<2 x float> addrspace(1)* %out, <2 x float> %a, <2 x float> %b) #1 {
  %no.md = fdiv <2 x float> %a, %b
  store volatile <2 x float> %no.md, <2 x float> addrspace(1)* %out

  %md.half.ulp = fdiv <2 x float> %a, %b, !fpmath !1
  store volatile <2 x float> %md.half.ulp, <2 x float> addrspace(1)* %out

  %md.1ulp = fdiv <2 x float> %a, %b, !fpmath !2
  store volatile <2 x float> %md.1ulp, <2 x float> addrspace(1)* %out

  %md.25ulp = fdiv <2 x float> %a, %b, !fpmath !0
  store volatile <2 x float> %md.25ulp, <2 x float> addrspace(1)* %out

  ret void
}

; CHECK-LABEL: @rcp_fdiv_fpmath_vector(
; CHECK: %[[NO0:[0-9]+]] =  extractelement <2 x float> %x, i64 0
; CHECK: %[[NO_FDIV0:[0-9]+]] = fdiv float 1.000000e+00, %[[NO0]]
; CHECK: %[[NO_INS0:[0-9]+]] = insertelement <2 x float> undef, float %[[NO_FDIV0]], i64 0
; CHECK: %[[NO1:[0-9]+]] = extractelement <2 x float> %x, i64 1
; CHECK: %[[NO_FDIV1:[0-9]+]] = fdiv float 1.000000e+00, %[[NO1]]
; CHECK: %no.md = insertelement <2 x float> %[[NO_INS0]], float %[[NO_FDIV1]], i64 1
; CHECK: store volatile <2 x float> %no.md, <2 x float> addrspace(1)* %out

; CHECK: %[[HALF0:[0-9]+]] =  extractelement <2 x float> %x, i64 0
; CHECK: %[[HALF_FDIV0:[0-9]+]] = fdiv float 1.000000e+00, %[[HALF0]], !fpmath !1
; CHECK: %[[HALF_INS0:[0-9]+]] = insertelement <2 x float> undef, float %[[HALF_FDIV0]], i64 0
; CHECK: %[[HALF1:[0-9]+]] = extractelement <2 x float> %x, i64 1
; CHECK: %[[HALF_FDIV1:[0-9]+]] =  fdiv float 1.000000e+00, %[[HALF1]], !fpmath !1
; CHECK: %md.half.ulp = insertelement <2 x float> %[[HALF_INS0]], float %[[HALF_FDIV1]], i64 1
; CHECK: store volatile <2 x float> %md.half.ulp, <2 x float> addrspace(1)* %out

; CHECK: %[[ARCP_NO0:[0-9]+]] =  extractelement <2 x float> %x, i64 0
; CHECK: %[[ARCP_NO_FDIV0:[0-9]+]] = fdiv arcp float 1.000000e+00, %[[ARCP_NO0]]
; CHECK: %[[ARCP_NO_INS0:[0-9]+]] = insertelement <2 x float> undef, float %[[ARCP_NO_FDIV0]], i64 0
; CHECK: %[[ARCP_NO1:[0-9]+]] = extractelement <2 x float> %x, i64 1
; CHECK: %[[ARCP_NO_FDIV1:[0-9]+]] =  fdiv arcp float 1.000000e+00, %[[ARCP_NO1]]
; CHECK: %arcp.no.md = insertelement <2 x float> %[[ARCP_NO_INS0]], float %[[ARCP_NO_FDIV1]], i64 1
; CHECK: store volatile <2 x float> %arcp.no.md, <2 x float> addrspace(1)* %out

; CHECK: %[[FAST_NO0:[0-9]+]] =  extractelement <2 x float> %x, i64 0
; CHECK: %[[FAST_NO_RCP0:[0-9]+]] = call fast float @llvm.amdgcn.rcp.f32(float %[[FAST_NO0]])
; CHECK: %[[FAST_NO_INS0:[0-9]+]] = insertelement <2 x float> undef, float %[[FAST_NO_RCP0]], i64 0
; CHECK: %[[FAST_NO1:[0-9]+]] = extractelement <2 x float> %x, i64 1
; CHECK: %[[FAST_NO_RCP1:[0-9]+]] =  call fast float @llvm.amdgcn.rcp.f32(float %[[FAST_NO1]])
; CHECK: %fast.no.md = insertelement <2 x float> %[[FAST_NO_INS0]], float %[[FAST_NO_RCP1]], i64 1
; CHECK: store volatile <2 x float> %fast.no.md, <2 x float> addrspace(1)* %out

; CHECK: %[[ARCP_250:[0-9]+]] =  extractelement <2 x float> %x, i64 0
; CHECK: %[[ARCP_25_RCP0:[0-9]+]] = call arcp float @llvm.amdgcn.rcp.f32(float %[[ARCP_250]])
; CHECK: %[[ARCP_25_INS0:[0-9]+]] = insertelement <2 x float> undef, float %[[ARCP_25_RCP0]], i64 0
; CHECK: %[[ARCP_251:[0-9]+]] = extractelement <2 x float> %x, i64 1
; CHECK: %[[ARCP_25_RCP1:[0-9]+]] =  call arcp float @llvm.amdgcn.rcp.f32(float %[[ARCP_251]])
; CHECK: %arcp.25ulp = insertelement <2 x float> %[[ARCP_25_INS0]], float %[[ARCP_25_RCP1]], i64 1
; CHECK: store volatile <2 x float> %arcp.25ulp, <2 x float> addrspace(1)* %out

; CHECK: %[[FAST_250:[0-9]+]] =  extractelement <2 x float> %x, i64 0
; CHECK: %[[FAST_25_RCP0:[0-9]+]] = call fast float @llvm.amdgcn.rcp.f32(float %[[FAST_250]])
; CHECK: %[[FAST_25_INS0:[0-9]+]] = insertelement <2 x float> undef, float %[[FAST_25_RCP0]], i64 0
; CHECK: %[[FAST_251:[0-9]+]] = extractelement <2 x float> %x, i64 1
; CHECK: %[[FAST_25_RCP1:[0-9]+]] =  call fast float @llvm.amdgcn.rcp.f32(float %[[FAST_251]])
; CHECK: %fast.25ulp = insertelement <2 x float> %[[FAST_25_INS0]], float %[[FAST_25_RCP1]], i64 1
; CHECK: store volatile <2 x float> %fast.25ulp, <2 x float> addrspace(1)* %out
define amdgpu_kernel void @rcp_fdiv_fpmath_vector(<2 x float> addrspace(1)* %out, <2 x float> %x) #1 {
  %no.md = fdiv <2 x float> <float 1.0, float 1.0>, %x
  store volatile <2 x float> %no.md, <2 x float> addrspace(1)* %out

  %md.half.ulp = fdiv <2 x float> <float 1.0, float 1.0>, %x, !fpmath !1
  store volatile <2 x float> %md.half.ulp, <2 x float> addrspace(1)* %out

  %arcp.no.md = fdiv arcp <2 x float> <float 1.0, float 1.0>, %x
  store volatile <2 x float> %arcp.no.md, <2 x float> addrspace(1)* %out

  %fast.no.md = fdiv fast <2 x float> <float 1.0, float 1.0>, %x
  store volatile <2 x float> %fast.no.md, <2 x float> addrspace(1)* %out

  %arcp.25ulp = fdiv arcp <2 x float> <float 1.0, float 1.0>, %x, !fpmath !0
  store volatile <2 x float> %arcp.25ulp, <2 x float> addrspace(1)* %out

  %fast.25ulp = fdiv fast <2 x float> <float 1.0, float 1.0>, %x, !fpmath !0
  store volatile <2 x float> %fast.25ulp, <2 x float> addrspace(1)* %out

  ret void
}

; CHECK-LABEL: @rcp_fdiv_fpmath_vector_nonsplat(
; CHECK: %[[NO0:[0-9]+]] =  extractelement <2 x float> %x, i64 0
; CHECK: %[[NO_FDIV0:[0-9]+]] = fdiv float 1.000000e+00, %[[NO0]]
; CHECK: %[[NO_INS0:[0-9]+]] = insertelement <2 x float> undef, float %[[NO_FDIV0]], i64 0
; CHECK: %[[NO1:[0-9]+]] = extractelement <2 x float> %x, i64 1
; CHECK: %[[NO_FDIV1:[0-9]+]] = fdiv float 2.000000e+00, %[[NO1]]
; CHECK: %no.md = insertelement <2 x float> %[[NO_INS0]], float %[[NO_FDIV1]], i64 1
; CHECK: store volatile <2 x float> %no.md, <2 x float> addrspace(1)* %out

; CHECK: %[[ARCP_NO0:[0-9]+]] =  extractelement <2 x float> %x, i64 0
; CHECK: %[[ARCP_NO_FDIV0:[0-9]+]] = fdiv arcp float 1.000000e+00, %[[ARCP_NO0]]
; CHECK: %[[ARCP_NO_INS0:[0-9]+]] = insertelement <2 x float> undef, float %[[ARCP_NO_FDIV0]], i64 0
; CHECK: %[[ARCP_NO1:[0-9]+]] = extractelement <2 x float> %x, i64 1
; CHECK: %[[ARCP_NO_FDIV1:[0-9]+]] =  fdiv arcp float 2.000000e+00, %[[ARCP_NO1]]
; CHECK: %arcp.no.md = insertelement <2 x float> %[[ARCP_NO_INS0]], float %[[ARCP_NO_FDIV1]], i64 1
; CHECK: store volatile <2 x float> %arcp.no.md, <2 x float> addrspace(1)* %out

; CHECK: %[[FAST_NO0:[0-9]+]] =  extractelement <2 x float> %x, i64 0
; CHECK: %[[FAST_NO_RCP0:[0-9]+]] = call fast float @llvm.amdgcn.rcp.f32(float %[[FAST_NO0]])
; CHECK: %[[FAST_NO_INS0:[0-9]+]] = insertelement <2 x float> undef, float %[[FAST_NO_RCP0]], i64 0
; CHECK: %[[FAST_NO1:[0-9]+]] = extractelement <2 x float> %x, i64 1
; CHECK: %[[FAST_NO_RCP1:[0-9]+]] =  call fast float @llvm.amdgcn.rcp.f32(float %[[FAST_NO1]])
; CHECK: %[[FAST_NO_MUL1:[0-9]+]] = fmul fast float 2.000000e+00, %[[FAST_NO_RCP1]]
; CHECK: %fast.no.md = insertelement <2 x float> %[[FAST_NO_INS0]], float %[[FAST_NO_MUL1]], i64 1
; CHECK: store volatile <2 x float> %fast.no.md, <2 x float> addrspace(1)* %out

; CHECK: %[[ARCP_250:[0-9]+]] =  extractelement <2 x float> %x, i64 0
; CHECK: %[[ARCP_25_RCP0:[0-9]+]] = call arcp float @llvm.amdgcn.rcp.f32(float %[[ARCP_250]])
; CHECK: %[[ARCP_25_INS0:[0-9]+]] = insertelement <2 x float> undef, float %[[ARCP_25_RCP0]], i64 0
; CHECK: %[[ARCP_251:[0-9]+]] = extractelement <2 x float> %x, i64 1
; CHECK: %[[ARCP_25_RCP1:[0-9]+]] =  call arcp float @llvm.amdgcn.rcp.f32(float %[[ARCP_251]])
; CHECK: %[[ARCP_25_MUL1:[0-9]+]] = fmul arcp float 2.000000e+00, %[[ARCP_25_RCP1]]
; CHECK: %arcp.25ulp = insertelement <2 x float> %[[ARCP_25_INS0]], float %[[ARCP_25_MUL1]], i64 1
; CHECK: store volatile <2 x float> %arcp.25ulp, <2 x float> addrspace(1)* %out

; CHECK: %[[FAST_250:[0-9]+]] =  extractelement <2 x float> %x, i64 0
; CHECK: %[[FAST_25_RCP0:[0-9]+]] = call fast float @llvm.amdgcn.rcp.f32(float %[[FAST_250]])
; CHECK: %[[FAST_25_INS0:[0-9]+]] = insertelement <2 x float> undef, float %[[FAST_25_RCP0]], i64 0
; CHECK: %[[FAST_251:[0-9]+]] = extractelement <2 x float> %x, i64 1
; CHECK: %[[FAST_25_RCP1:[0-9]+]] =  call fast float @llvm.amdgcn.rcp.f32(float %[[FAST_251]])
; CHECK: %[[FAST_25_MUL1:[0-9]+]] = fmul fast float 2.000000e+00, %[[FAST_25_RCP1]]
; CHECK: %fast.25ulp = insertelement <2 x float> %[[FAST_25_INS0]], float %[[FAST_25_MUL1]], i64 1
; CHECK: store volatile <2 x float> %fast.25ulp, <2 x float> addrspace(1)* %out
define amdgpu_kernel void @rcp_fdiv_fpmath_vector_nonsplat(<2 x float> addrspace(1)* %out, <2 x float> %x) #1 {
  %no.md = fdiv <2 x float> <float 1.0, float 2.0>, %x
  store volatile <2 x float> %no.md, <2 x float> addrspace(1)* %out

  %arcp.no.md = fdiv arcp <2 x float> <float 1.0, float 2.0>, %x
  store volatile <2 x float> %arcp.no.md, <2 x float> addrspace(1)* %out

  %fast.no.md = fdiv fast <2 x float> <float 1.0, float 2.0>, %x
  store volatile <2 x float> %fast.no.md, <2 x float> addrspace(1)* %out

  %arcp.25ulp = fdiv arcp <2 x float> <float 1.0, float 2.0>, %x, !fpmath !0
  store volatile <2 x float> %arcp.25ulp, <2 x float> addrspace(1)* %out

  %fast.25ulp = fdiv fast <2 x float> <float 1.0, float 2.0>, %x, !fpmath !0
  store volatile <2 x float> %fast.25ulp, <2 x float> addrspace(1)* %out

  ret void
}

; CHECK-LABEL: @rcp_fdiv_fpmath_vector_partial_constant(
; CHECK: %[[ARCP_A0:[0-9]+]] = extractelement <2 x float> %x.insert, i64 0
; CHECK: %[[ARCP_B0:[0-9]+]] = extractelement <2 x float> %y, i64 0
; CHECK: %[[ARCP_RCP0:[0-9]+]] = call arcp float @llvm.amdgcn.rcp.f32(float %[[ARCP_B0]])
; CHECK: %[[ARCP_MUL0:[0-9]+]] = fmul arcp float %[[ARCP_A0]], %[[ARCP_RCP0]], !fpmath !0
; CHECK: %[[ARCP_INS0:[0-9]+]] = insertelement <2 x float> undef, float %[[ARCP_MUL0]], i64 0
; CHECK: %[[ARCP_A1:[0-9]+]] = extractelement <2 x float> %x.insert, i64 1
; CHECK: %[[ARCP_B1:[0-9]+]] = extractelement <2 x float> %y, i64 1
; CHECK: %[[ARCP_RCP1:[0-9]+]] = call arcp float @llvm.amdgcn.rcp.f32(float %[[ARCP_B1]])
; CHECK: %[[ARCP_MUL1:[0-9]+]] = fmul arcp float %[[ARCP_A1]], %[[ARCP_RCP1]], !fpmath !0
; CHECK: %arcp.25ulp = insertelement <2 x float> %[[ARCP_INS0]], float %[[ARCP_MUL1]], i64 1
; CHECK: store volatile <2 x float> %arcp.25ulp

; CHECK: %[[FAST_A0:[0-9]+]] = extractelement <2 x float> %x.insert, i64 0
; CHECK: %[[FAST_B0:[0-9]+]] = extractelement <2 x float> %y, i64 0
; CHECK: %[[FAST_RCP0:[0-9]+]] = call fast float @llvm.amdgcn.rcp.f32(float %[[FAST_B0]])
; CHECK: %[[FAST_MUL0:[0-9]+]] = fmul fast float %[[FAST_A0]], %[[FAST_RCP0]], !fpmath !0
; CHECK: %[[FAST_INS0:[0-9]+]] = insertelement <2 x float> undef, float %[[FAST_MUL0]], i64 0
; CHECK: %[[FAST_A1:[0-9]+]] = extractelement <2 x float> %x.insert, i64 1
; CHECK: %[[FAST_B1:[0-9]+]] = extractelement <2 x float> %y, i64 1
; CHECK: %[[FAST_RCP1:[0-9]+]] = call fast float @llvm.amdgcn.rcp.f32(float %[[FAST_B1]])
; CHECK: %[[FAST_MUL1:[0-9]+]] = fmul fast float %[[FAST_A1]], %[[FAST_RCP1]], !fpmath !0
; CHECK: %fast.25ulp = insertelement <2 x float> %[[FAST_INS0]], float %[[FAST_MUL1]], i64 1
; CHECK: store volatile <2 x float> %fast.25ulp
define amdgpu_kernel void @rcp_fdiv_fpmath_vector_partial_constant(<2 x float> addrspace(1)* %out, <2 x float> %x, <2 x float> %y) #1 {
  %x.insert = insertelement <2 x float> %x, float 1.0, i32 0

  %arcp.25ulp = fdiv arcp <2 x float> %x.insert, %y, !fpmath !0
  store volatile <2 x float> %arcp.25ulp, <2 x float> addrspace(1)* %out

  %fast.25ulp = fdiv fast <2 x float> %x.insert, %y, !fpmath !0
  store volatile <2 x float> %fast.25ulp, <2 x float> addrspace(1)* %out

  ret void
}

; CHECK-LABEL: @fdiv_fpmath_f32_denormals(
; CHECK: %no.md = fdiv float %a, %b{{$}}
; CHECK: %md.half.ulp = fdiv float %a, %b, !fpmath !1
; CHECK: %md.1ulp = fdiv float %a, %b, !fpmath !2
; CHECK: %md.25ulp = fdiv float %a, %b, !fpmath !0
; CHECK: %md.3ulp = fdiv float %a, %b, !fpmath !3
; CHECK: %[[RCP_FAST:[0-9]+]] = call fast float @llvm.amdgcn.rcp.f32(float %b)
; CHECK: %fast.md.25ulp = fmul fast float %a, %[[RCP_FAST]], !fpmath !0
; CHECK: %arcp.md.25ulp  = fdiv arcp float %a, %b, !fpmath !0
define amdgpu_kernel void @fdiv_fpmath_f32_denormals(float addrspace(1)* %out, float %a, float %b) #2 {
  %no.md = fdiv float %a, %b
  store volatile float %no.md, float addrspace(1)* %out

  %md.half.ulp = fdiv float %a, %b, !fpmath !1
  store volatile float %md.half.ulp, float addrspace(1)* %out

  %md.1ulp = fdiv float %a, %b, !fpmath !2
  store volatile float %md.1ulp, float addrspace(1)* %out

  %md.25ulp = fdiv float %a, %b, !fpmath !0
  store volatile float %md.25ulp, float addrspace(1)* %out

  %md.3ulp = fdiv float %a, %b, !fpmath !3
  store volatile float %md.3ulp, float addrspace(1)* %out

  %fast.md.25ulp = fdiv fast float %a, %b, !fpmath !0
  store volatile float %fast.md.25ulp, float addrspace(1)* %out

  %arcp.md.25ulp = fdiv arcp float %a, %b, !fpmath !0
  store volatile float %arcp.md.25ulp, float addrspace(1)* %out

  ret void
}

attributes #0 = { nounwind optnone noinline }
attributes #1 = { nounwind }
attributes #2 = { nounwind "target-features"="+fp32-denormals" }

; CHECK: !0 = !{float 2.500000e+00}
; CHECK: !1 = !{float 5.000000e-01}
; CHECK: !2 = !{float 1.000000e+00}
; CHECK: !3 = !{float 3.000000e+00}

!0 = !{float 2.500000e+00}
!1 = !{float 5.000000e-01}
!2 = !{float 1.000000e+00}
!3 = !{float 3.000000e+00}
