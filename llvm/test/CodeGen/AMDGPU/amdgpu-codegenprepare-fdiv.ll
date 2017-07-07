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
; CHECK: %fast.md.25ulp = fdiv fast float %a, %b, !fpmath !0
; CHECK: arcp.md.25ulp = fdiv arcp float %a, %b, !fpmath !0
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
; CHECK: %md.25ulp = fdiv float 1.000000e+00, %x, !fpmath !0
; CHECK: %md.half.ulp = fdiv float 1.000000e+00, %x, !fpmath !1
; CHECK: %arcp.no.md = fdiv arcp float 1.000000e+00, %x{{$}}
; CHECK: %arcp.25ulp = fdiv arcp float 1.000000e+00, %x, !fpmath !0
; CHECK: %fast.no.md = fdiv fast float 1.000000e+00, %x{{$}}
; CHECK: %fast.25ulp = fdiv fast float 1.000000e+00, %x, !fpmath !0
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

; CHECK-LABEL: @fdiv_fpmath_vector(
; CHECK: %no.md = fdiv <2 x float> %a, %b{{$}}
; CHECK: %md.half.ulp = fdiv <2 x float> %a, %b, !fpmath !1
; CHECK: %md.1ulp = fdiv <2 x float> %a, %b, !fpmath !2

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
; CHECK: %no.md = fdiv <2 x float> <float 1.000000e+00, float 1.000000e+00>, %x{{$}}
; CHECK: %md.half.ulp = fdiv <2 x float> <float 1.000000e+00, float 1.000000e+00>, %x, !fpmath !1
; CHECK: %arcp.no.md = fdiv arcp <2 x float> <float 1.000000e+00, float 1.000000e+00>, %x{{$}}
; CHECK: %fast.no.md = fdiv fast <2 x float> <float 1.000000e+00, float 1.000000e+00>, %x{{$}}
; CHECK: %arcp.25ulp = fdiv arcp <2 x float> <float 1.000000e+00, float 1.000000e+00>, %x, !fpmath !0
; CHECK: %fast.25ulp = fdiv fast <2 x float> <float 1.000000e+00, float 1.000000e+00>, %x, !fpmath !0
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
; CHECK: %no.md = fdiv <2 x float> <float 1.000000e+00, float 2.000000e+00>, %x
; CHECK: %arcp.no.md = fdiv arcp <2 x float> <float 1.000000e+00, float 2.000000e+00>, %x
; CHECK: %fast.no.md = fdiv fast <2 x float> <float 1.000000e+00, float 2.000000e+00>, %x{{$}}
; CHECK: %arcp.25ulp = fdiv arcp <2 x float> <float 1.000000e+00, float 2.000000e+00>, %x, !fpmath !0
; CHECK: %fast.25ulp = fdiv fast <2 x float> <float 1.000000e+00, float 2.000000e+00>, %x, !fpmath !0
; CHECK: store volatile <2 x float> %fast.25ulp
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

; FIXME: Should be able to get fdiv for 1.0 component
; CHECK-LABEL: @rcp_fdiv_fpmath_vector_partial_constant(
; CHECK: %arcp.25ulp = fdiv arcp <2 x float> %x.insert, %y, !fpmath !0
; CHECK: store volatile <2 x float> %arcp.25ulp

; CHECK: %fast.25ulp = fdiv fast <2 x float> %x.insert, %y, !fpmath !0
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
; CHECK: %fast.md.25ulp = fdiv fast float %a, %b, !fpmath !0
; CHECK: %arcp.md.25ulp = fdiv arcp float %a, %b, !fpmath !0
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
