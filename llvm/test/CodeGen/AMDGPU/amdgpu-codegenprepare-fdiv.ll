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
; CHECK: %md.half.ulp = fdiv float %a, %b
; CHECK: %md.1ulp = fdiv float %a, %b
; CHECK: %md.25ulp = call float @llvm.amdgcn.fdiv.fast(float %a, float %b)
; CHECK: %md.3ulp = call float @llvm.amdgcn.fdiv.fast(float %a, float %b)
; CHECK: %[[FAST_RCP:[0-9]+]] = call fast float @llvm.amdgcn.rcp.f32(float %b)
; CHECK: %fast.md.25ulp = fmul fast float %a, %[[FAST_RCP]]
; CHECK: %[[AFN_RCP:[0-9]+]] = call afn float @llvm.amdgcn.rcp.f32(float %b)
; CHECK: afn.md.25ulp = fmul afn float %a, %[[AFN_RCP]]
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

  %afn.md.25ulp = fdiv afn float %a, %b, !fpmath !0
  store volatile float %afn.md.25ulp, float addrspace(1)* %out

  ret void
}

; CHECK-LABEL: @rcp_fdiv_fpmath(
; CHECK: %no.md = fdiv float 1.000000e+00, %x{{$}}
; CHECK: %md.25ulp = call float @llvm.amdgcn.rcp.f32(float %x)
; CHECK: %md.half.ulp = fdiv float 1.000000e+00, %x
; CHECK: %afn.no.md = call afn float @llvm.amdgcn.rcp.f32(float %x)
; CHECK: %afn.25ulp = call afn float @llvm.amdgcn.rcp.f32(float %x)
; CHECK: %fast.no.md = call fast float @llvm.amdgcn.rcp.f32(float %x)
; CHECK: %fast.25ulp = call fast float @llvm.amdgcn.rcp.f32(float %x)
define amdgpu_kernel void @rcp_fdiv_fpmath(float addrspace(1)* %out, float %x) #1 {
  %no.md = fdiv float 1.0, %x
  store volatile float %no.md, float addrspace(1)* %out

  %md.25ulp = fdiv float 1.0, %x, !fpmath !0
  store volatile float %md.25ulp, float addrspace(1)* %out

  %md.half.ulp = fdiv float 1.0, %x, !fpmath !1
  store volatile float %md.half.ulp, float addrspace(1)* %out

  %afn.no.md = fdiv afn float 1.0, %x
  store volatile float %afn.no.md, float addrspace(1)* %out

  %afn.25ulp = fdiv afn float 1.0, %x, !fpmath !0
  store volatile float %afn.25ulp, float addrspace(1)* %out

  %fast.no.md = fdiv fast float 1.0, %x
  store volatile float %fast.no.md, float addrspace(1)* %out

  %fast.25ulp = fdiv fast float 1.0, %x, !fpmath !0
  store volatile float %fast.25ulp, float addrspace(1)* %out

  ret void
}

; CHECK-LABEL: @fdiv_fpmath_vector(
; CHECK: %[[NO_A0:[0-9]+]] = extractelement <2 x float> %a, i64 0
; CHECK: %[[NO_B0:[0-9]+]] = extractelement <2 x float> %b, i64 0
; CHECK: %[[NO_FDIV0:[0-9]+]] = fdiv float %[[NO_A0]], %[[NO_B0]]
; CHECK: %[[NO_INS0:[0-9]+]] = insertelement <2 x float> poison, float %[[NO_FDIV0]], i64 0
; CHECK: %[[NO_A1:[0-9]+]] = extractelement <2 x float> %a, i64 1
; CHECK: %[[NO_B1:[0-9]+]] = extractelement <2 x float> %b, i64 1
; CHECK: %[[NO_FDIV1:[0-9]+]] = fdiv float %[[NO_A1]], %[[NO_B1]]
; CHECK: %no.md = insertelement <2 x float> %[[NO_INS0]], float %[[NO_FDIV1]], i64 1
; CHECK: store volatile <2 x float> %no.md, <2 x float> addrspace(1)* %out

; CHECK: %[[HALF_A0:[0-9]+]] = extractelement <2 x float> %a, i64 0
; CHECK: %[[HALF_B0:[0-9]+]] = extractelement <2 x float> %b, i64 0
; CHECK: %[[HALF_FDIV0:[0-9]+]] = fdiv float %[[HALF_A0]], %[[HALF_B0]]
; CHECK: %[[HALF_INS0:[0-9]+]] = insertelement <2 x float> poison, float %[[HALF_FDIV0]], i64 0
; CHECK: %[[HALF_A1:[0-9]+]] = extractelement <2 x float> %a, i64 1
; CHECK: %[[HALF_B1:[0-9]+]] = extractelement <2 x float> %b, i64 1
; CHECK: %[[HALF_FDIV1:[0-9]+]] = fdiv float %[[HALF_A1]], %[[HALF_B1]]
; CHECK: %md.half.ulp = insertelement <2 x float> %[[HALF_INS0]], float %[[HALF_FDIV1]], i64 1
; CHECK: store volatile <2 x float> %md.half.ulp, <2 x float> addrspace(1)* %out

; CHECK: %[[ONE_A0:[0-9]+]] = extractelement <2 x float> %a, i64 0
; CHECK: %[[ONE_B0:[0-9]+]] = extractelement <2 x float> %b, i64 0
; CHECK: %[[ONE_FDIV0:[0-9]+]] = fdiv float %[[ONE_A0]], %[[ONE_B0]]
; CHECK: %[[ONE_INS0:[0-9]+]] = insertelement <2 x float> poison, float %[[ONE_FDIV0]], i64 0
; CHECK: %[[ONE_A1:[0-9]+]] = extractelement <2 x float> %a, i64 1
; CHECK: %[[ONE_B1:[0-9]+]] = extractelement <2 x float> %b, i64 1
; CHECK: %[[ONE_FDIV1:[0-9]+]] = fdiv float %[[ONE_A1]], %[[ONE_B1]]
; CHECK: %md.1ulp = insertelement <2 x float> %[[ONE_INS0]], float %[[ONE_FDIV1]], i64 1
; CHECK: store volatile <2 x float> %md.1ulp, <2 x float> addrspace(1)* %out

; CHECK: %[[A0:[0-9]+]] = extractelement <2 x float> %a, i64 0
; CHECK: %[[B0:[0-9]+]] = extractelement <2 x float> %b, i64 0
; CHECK: %[[FDIV0:[0-9]+]] = call float @llvm.amdgcn.fdiv.fast(float %[[A0]], float %[[B0]])
; CHECK: %[[INS0:[0-9]+]] = insertelement <2 x float> poison, float %[[FDIV0]], i64 0
; CHECK: %[[A1:[0-9]+]] = extractelement <2 x float> %a, i64 1
; CHECK: %[[B1:[0-9]+]] = extractelement <2 x float> %b, i64 1
; CHECK: %[[FDIV1:[0-9]+]] = call float @llvm.amdgcn.fdiv.fast(float %[[A1]], float %[[B1]])
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
; CHECK: %[[NO_INS0:[0-9]+]] = insertelement <2 x float> poison, float %[[NO_FDIV0]], i64 0
; CHECK: %[[NO1:[0-9]+]] = extractelement <2 x float> %x, i64 1
; CHECK: %[[NO_FDIV1:[0-9]+]] = fdiv float 1.000000e+00, %[[NO1]]
; CHECK: %no.md = insertelement <2 x float> %[[NO_INS0]], float %[[NO_FDIV1]], i64 1
; CHECK: store volatile <2 x float> %no.md, <2 x float> addrspace(1)* %out

; CHECK: %[[HALF0:[0-9]+]] =  extractelement <2 x float> %x, i64 0
; CHECK: %[[HALF_FDIV0:[0-9]+]] = fdiv float 1.000000e+00, %[[HALF0]]
; CHECK: %[[HALF_INS0:[0-9]+]] = insertelement <2 x float> poison, float %[[HALF_FDIV0]], i64 0
; CHECK: %[[HALF1:[0-9]+]] = extractelement <2 x float> %x, i64 1
; CHECK: %[[HALF_FDIV1:[0-9]+]] =  fdiv float 1.000000e+00, %[[HALF1]]
; CHECK: %md.half.ulp = insertelement <2 x float> %[[HALF_INS0]], float %[[HALF_FDIV1]], i64 1
; CHECK: store volatile <2 x float> %md.half.ulp, <2 x float> addrspace(1)* %out

; CHECK: %[[AFN_NO0:[0-9]+]] =  extractelement <2 x float> %x, i64 0
; CHECK: %[[AFN_NO_FDIV0:[0-9]+]] = call afn float @llvm.amdgcn.rcp.f32(float %[[AFN_NO0]])
; CHECK: %[[AFN_NO_INS0:[0-9]+]] = insertelement <2 x float> poison, float %[[AFN_NO_FDIV0]], i64 0
; CHECK: %[[AFN_NO1:[0-9]+]] = extractelement <2 x float> %x, i64 1
; CHECK: %[[AFN_NO_FDIV1:[0-9]+]] =  call afn float @llvm.amdgcn.rcp.f32(float %[[AFN_NO1]])
; CHECK: %afn.no.md = insertelement <2 x float> %[[AFN_NO_INS0]], float %[[AFN_NO_FDIV1]], i64 1
; CHECK: store volatile <2 x float> %afn.no.md, <2 x float> addrspace(1)* %out

; CHECK: %[[FAST_NO0:[0-9]+]] =  extractelement <2 x float> %x, i64 0
; CHECK: %[[FAST_NO_RCP0:[0-9]+]] = call fast float @llvm.amdgcn.rcp.f32(float %[[FAST_NO0]])
; CHECK: %[[FAST_NO_INS0:[0-9]+]] = insertelement <2 x float> poison, float %[[FAST_NO_RCP0]], i64 0
; CHECK: %[[FAST_NO1:[0-9]+]] = extractelement <2 x float> %x, i64 1
; CHECK: %[[FAST_NO_RCP1:[0-9]+]] =  call fast float @llvm.amdgcn.rcp.f32(float %[[FAST_NO1]])
; CHECK: %fast.no.md = insertelement <2 x float> %[[FAST_NO_INS0]], float %[[FAST_NO_RCP1]], i64 1
; CHECK: store volatile <2 x float> %fast.no.md, <2 x float> addrspace(1)* %out

; CHECK: %[[AFN_250:[0-9]+]] =  extractelement <2 x float> %x, i64 0
; CHECK: %[[AFN_25_RCP0:[0-9]+]] = call afn float @llvm.amdgcn.rcp.f32(float %[[AFN_250]])
; CHECK: %[[AFN_25_INS0:[0-9]+]] = insertelement <2 x float> poison, float %[[AFN_25_RCP0]], i64 0
; CHECK: %[[AFN_251:[0-9]+]] = extractelement <2 x float> %x, i64 1
; CHECK: %[[AFN_25_RCP1:[0-9]+]] =  call afn float @llvm.amdgcn.rcp.f32(float %[[AFN_251]])
; CHECK: %afn.25ulp = insertelement <2 x float> %[[AFN_25_INS0]], float %[[AFN_25_RCP1]], i64 1
; CHECK: store volatile <2 x float> %afn.25ulp, <2 x float> addrspace(1)* %out

; CHECK: %[[FAST_250:[0-9]+]] =  extractelement <2 x float> %x, i64 0
; CHECK: %[[FAST_25_RCP0:[0-9]+]] = call fast float @llvm.amdgcn.rcp.f32(float %[[FAST_250]])
; CHECK: %[[FAST_25_INS0:[0-9]+]] = insertelement <2 x float> poison, float %[[FAST_25_RCP0]], i64 0
; CHECK: %[[FAST_251:[0-9]+]] = extractelement <2 x float> %x, i64 1
; CHECK: %[[FAST_25_RCP1:[0-9]+]] =  call fast float @llvm.amdgcn.rcp.f32(float %[[FAST_251]])
; CHECK: %fast.25ulp = insertelement <2 x float> %[[FAST_25_INS0]], float %[[FAST_25_RCP1]], i64 1
; CHECK: store volatile <2 x float> %fast.25ulp, <2 x float> addrspace(1)* %out
define amdgpu_kernel void @rcp_fdiv_fpmath_vector(<2 x float> addrspace(1)* %out, <2 x float> %x) #1 {
  %no.md = fdiv <2 x float> <float 1.0, float 1.0>, %x
  store volatile <2 x float> %no.md, <2 x float> addrspace(1)* %out

  %md.half.ulp = fdiv <2 x float> <float 1.0, float 1.0>, %x, !fpmath !1
  store volatile <2 x float> %md.half.ulp, <2 x float> addrspace(1)* %out

  %afn.no.md = fdiv afn <2 x float> <float 1.0, float 1.0>, %x
  store volatile <2 x float> %afn.no.md, <2 x float> addrspace(1)* %out

  %fast.no.md = fdiv fast <2 x float> <float 1.0, float 1.0>, %x
  store volatile <2 x float> %fast.no.md, <2 x float> addrspace(1)* %out

  %afn.25ulp = fdiv afn <2 x float> <float 1.0, float 1.0>, %x, !fpmath !0
  store volatile <2 x float> %afn.25ulp, <2 x float> addrspace(1)* %out

  %fast.25ulp = fdiv fast <2 x float> <float 1.0, float 1.0>, %x, !fpmath !0
  store volatile <2 x float> %fast.25ulp, <2 x float> addrspace(1)* %out

  ret void
}

; CHECK-LABEL: @rcp_fdiv_fpmath_vector_nonsplat(
; CHECK: %[[NO0:[0-9]+]] =  extractelement <2 x float> %x, i64 0
; CHECK: %[[NO_FDIV0:[0-9]+]] = fdiv float 1.000000e+00, %[[NO0]]
; CHECK: %[[NO_INS0:[0-9]+]] = insertelement <2 x float> poison, float %[[NO_FDIV0]], i64 0
; CHECK: %[[NO1:[0-9]+]] = extractelement <2 x float> %x, i64 1
; CHECK: %[[NO_FDIV1:[0-9]+]] = fdiv float 2.000000e+00, %[[NO1]]
; CHECK: %no.md = insertelement <2 x float> %[[NO_INS0]], float %[[NO_FDIV1]], i64 1
; CHECK: store volatile <2 x float> %no.md, <2 x float> addrspace(1)* %out

; CHECK: %[[AFN_NO0:[0-9]+]] =  extractelement <2 x float> %x, i64 0
; CHECK: %[[AFN_NO_FDIV0:[0-9]+]] = call afn float @llvm.amdgcn.rcp.f32(float %[[AFN_NO0]])
; CHECK: %[[AFN_NO_INS0:[0-9]+]] = insertelement <2 x float> poison, float %[[AFN_NO_FDIV0]], i64 0
; CHECK: %[[AFN_NO1:[0-9]+]] = extractelement <2 x float> %x, i64 1
; CHECK: %[[AFN_NO_FDIV1:[0-9]+]] =  call afn float @llvm.amdgcn.rcp.f32(float %[[AFN_NO1]])
; CHECK: %[[AFN_NO_MUL1:[0-9]+]] = fmul afn float 2.000000e+00, %[[AFN_NO_FDIV1]]
; CHECK: %afn.no.md = insertelement <2 x float> %[[AFN_NO_INS0]], float %[[AFN_NO_MUL1]], i64 1
; CHECK: store volatile <2 x float> %afn.no.md, <2 x float> addrspace(1)* %out

; CHECK: %[[FAST_NO0:[0-9]+]] =  extractelement <2 x float> %x, i64 0
; CHECK: %[[FAST_NO_RCP0:[0-9]+]] = call fast float @llvm.amdgcn.rcp.f32(float %[[FAST_NO0]])
; CHECK: %[[FAST_NO_INS0:[0-9]+]] = insertelement <2 x float> poison, float %[[FAST_NO_RCP0]], i64 0
; CHECK: %[[FAST_NO1:[0-9]+]] = extractelement <2 x float> %x, i64 1
; CHECK: %[[FAST_NO_RCP1:[0-9]+]] =  call fast float @llvm.amdgcn.rcp.f32(float %[[FAST_NO1]])
; CHECK: %[[FAST_NO_MUL1:[0-9]+]] = fmul fast float 2.000000e+00, %[[FAST_NO_RCP1]]
; CHECK: %fast.no.md = insertelement <2 x float> %[[FAST_NO_INS0]], float %[[FAST_NO_MUL1]], i64 1
; CHECK: store volatile <2 x float> %fast.no.md, <2 x float> addrspace(1)* %out

; CHECK: %[[AFN_250:[0-9]+]] =  extractelement <2 x float> %x, i64 0
; CHECK: %[[AFN_25_RCP0:[0-9]+]] = call afn float @llvm.amdgcn.rcp.f32(float %[[AFN_250]])
; CHECK: %[[AFN_25_INS0:[0-9]+]] = insertelement <2 x float> poison, float %[[AFN_25_RCP0]], i64 0
; CHECK: %[[AFN_251:[0-9]+]] = extractelement <2 x float> %x, i64 1
; CHECK: %[[AFN_25_RCP1:[0-9]+]] =  call afn float @llvm.amdgcn.rcp.f32(float %[[AFN_251]])
; CHECK: %[[AFN_25_MUL1:[0-9]+]] = fmul afn float 2.000000e+00, %[[AFN_25_RCP1]]
; CHECK: %afn.25ulp = insertelement <2 x float> %[[AFN_25_INS0]], float %[[AFN_25_MUL1]], i64 1
; CHECK: store volatile <2 x float> %afn.25ulp, <2 x float> addrspace(1)* %out

; CHECK: %[[FAST_250:[0-9]+]] =  extractelement <2 x float> %x, i64 0
; CHECK: %[[FAST_25_RCP0:[0-9]+]] = call fast float @llvm.amdgcn.rcp.f32(float %[[FAST_250]])
; CHECK: %[[FAST_25_INS0:[0-9]+]] = insertelement <2 x float> poison, float %[[FAST_25_RCP0]], i64 0
; CHECK: %[[FAST_251:[0-9]+]] = extractelement <2 x float> %x, i64 1
; CHECK: %[[FAST_25_RCP1:[0-9]+]] =  call fast float @llvm.amdgcn.rcp.f32(float %[[FAST_251]])
; CHECK: %[[FAST_25_MUL1:[0-9]+]] = fmul fast float 2.000000e+00, %[[FAST_25_RCP1]]
; CHECK: %fast.25ulp = insertelement <2 x float> %[[FAST_25_INS0]], float %[[FAST_25_MUL1]], i64 1
; CHECK: store volatile <2 x float> %fast.25ulp, <2 x float> addrspace(1)* %out
define amdgpu_kernel void @rcp_fdiv_fpmath_vector_nonsplat(<2 x float> addrspace(1)* %out, <2 x float> %x) #1 {
  %no.md = fdiv <2 x float> <float 1.0, float 2.0>, %x
  store volatile <2 x float> %no.md, <2 x float> addrspace(1)* %out

  %afn.no.md = fdiv afn <2 x float> <float 1.0, float 2.0>, %x
  store volatile <2 x float> %afn.no.md, <2 x float> addrspace(1)* %out

  %fast.no.md = fdiv fast <2 x float> <float 1.0, float 2.0>, %x
  store volatile <2 x float> %fast.no.md, <2 x float> addrspace(1)* %out

  %afn.25ulp = fdiv afn <2 x float> <float 1.0, float 2.0>, %x, !fpmath !0
  store volatile <2 x float> %afn.25ulp, <2 x float> addrspace(1)* %out

  %fast.25ulp = fdiv fast <2 x float> <float 1.0, float 2.0>, %x, !fpmath !0
  store volatile <2 x float> %fast.25ulp, <2 x float> addrspace(1)* %out

  ret void
}

; CHECK-LABEL: @rcp_fdiv_fpmath_vector_partial_constant(
; CHECK: %[[AFN_A0:[0-9]+]] = extractelement <2 x float> %x.insert, i64 0
; CHECK: %[[AFN_B0:[0-9]+]] = extractelement <2 x float> %y, i64 0
; CHECK: %[[AFN_RCP0:[0-9]+]] = call afn float @llvm.amdgcn.rcp.f32(float %[[AFN_B0]])
; CHECK: %[[AFN_MUL0:[0-9]+]] = fmul afn float %[[AFN_A0]], %[[AFN_RCP0]]
; CHECK: %[[AFN_INS0:[0-9]+]] = insertelement <2 x float> poison, float %[[AFN_MUL0]], i64 0
; CHECK: %[[AFN_A1:[0-9]+]] = extractelement <2 x float> %x.insert, i64 1
; CHECK: %[[AFN_B1:[0-9]+]] = extractelement <2 x float> %y, i64 1
; CHECK: %[[AFN_RCP1:[0-9]+]] = call afn float @llvm.amdgcn.rcp.f32(float %[[AFN_B1]])
; CHECK: %[[AFN_MUL1:[0-9]+]] = fmul afn float %[[AFN_A1]], %[[AFN_RCP1]]
; CHECK: %afn.25ulp = insertelement <2 x float> %[[AFN_INS0]], float %[[AFN_MUL1]], i64 1
; CHECK: store volatile <2 x float> %afn.25ulp

; CHECK: %[[FAST_A0:[0-9]+]] = extractelement <2 x float> %x.insert, i64 0
; CHECK: %[[FAST_B0:[0-9]+]] = extractelement <2 x float> %y, i64 0
; CHECK: %[[FAST_RCP0:[0-9]+]] = call fast float @llvm.amdgcn.rcp.f32(float %[[FAST_B0]])
; CHECK: %[[FAST_MUL0:[0-9]+]] = fmul fast float %[[FAST_A0]], %[[FAST_RCP0]]
; CHECK: %[[FAST_INS0:[0-9]+]] = insertelement <2 x float> poison, float %[[FAST_MUL0]], i64 0
; CHECK: %[[FAST_A1:[0-9]+]] = extractelement <2 x float> %x.insert, i64 1
; CHECK: %[[FAST_B1:[0-9]+]] = extractelement <2 x float> %y, i64 1
; CHECK: %[[FAST_RCP1:[0-9]+]] = call fast float @llvm.amdgcn.rcp.f32(float %[[FAST_B1]])
; CHECK: %[[FAST_MUL1:[0-9]+]] = fmul fast float %[[FAST_A1]], %[[FAST_RCP1]]
; CHECK: %fast.25ulp = insertelement <2 x float> %[[FAST_INS0]], float %[[FAST_MUL1]], i64 1
; CHECK: store volatile <2 x float> %fast.25ulp
define amdgpu_kernel void @rcp_fdiv_fpmath_vector_partial_constant(<2 x float> addrspace(1)* %out, <2 x float> %x, <2 x float> %y) #1 {
  %x.insert = insertelement <2 x float> %x, float 1.0, i32 0

  %afn.25ulp = fdiv afn <2 x float> %x.insert, %y, !fpmath !0
  store volatile <2 x float> %afn.25ulp, <2 x float> addrspace(1)* %out

  %fast.25ulp = fdiv fast <2 x float> %x.insert, %y, !fpmath !0
  store volatile <2 x float> %fast.25ulp, <2 x float> addrspace(1)* %out

  ret void
}

; CHECK-LABEL: @fdiv_fpmath_f32_denormals(
; CHECK: %no.md = fdiv float %a, %b{{$}}
; CHECK: %md.half.ulp = fdiv float %a, %b
; CHECK: %md.1ulp = fdiv float %a, %b
; CHECK: %md.25ulp = fdiv float %a, %b
; CHECK: %md.3ulp = fdiv float %a, %b
; CHECK: %[[RCP_FAST:[0-9]+]] = call fast float @llvm.amdgcn.rcp.f32(float %b)
; CHECK: %fast.md.25ulp = fmul fast float %a, %[[RCP_FAST]]
; CHECK: %[[RCP_AFN:[0-9]+]] = call afn float @llvm.amdgcn.rcp.f32(float %b)
; CHECK: %afn.md.25ulp  = fmul afn float %a, %[[RCP_AFN]]
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

  %afn.md.25ulp = fdiv afn float %a, %b, !fpmath !0
  store volatile float %afn.md.25ulp, float addrspace(1)* %out

  ret void
}

attributes #0 = { nounwind optnone noinline }
attributes #1 = { nounwind "denormal-fp-math-f32"="preserve-sign,preserve-sign" }
attributes #2 = { nounwind "denormal-fp-math-f32"="ieee,ieee" }

!0 = !{float 2.500000e+00}
!1 = !{float 5.000000e-01}
!2 = !{float 1.000000e+00}
!3 = !{float 3.000000e+00}
