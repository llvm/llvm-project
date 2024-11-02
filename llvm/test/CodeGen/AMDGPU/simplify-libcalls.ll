; RUN: opt -S -O1 -mtriple=amdgcn-- -amdgpu-simplify-libcall < %s | FileCheck -enable-var-scope -check-prefix=GCN -check-prefix=GCN-POSTLINK %s
; RUN: opt -S -O1 -mtriple=amdgcn-- -amdgpu-simplify-libcall -amdgpu-prelink -amdgpu-enable-ocl-mangling-mismatch-workaround=0 <%s | FileCheck -enable-var-scope -check-prefix=GCN -check-prefix=GCN-PRELINK %s
; RUN: opt -S -O1 -mtriple=amdgcn-- -amdgpu-use-native -amdgpu-prelink < %s | FileCheck -enable-var-scope -check-prefix=GCN -check-prefix=GCN-NATIVE %s
; RUN: opt -S -passes='default<O1>' -mtriple=amdgcn-- -amdgpu-simplify-libcall < %s | FileCheck -enable-var-scope -check-prefix=GCN -check-prefix=GCN-POSTLINK %s
; RUN: opt -S -passes='default<O1>' -mtriple=amdgcn-- -amdgpu-simplify-libcall -amdgpu-prelink -amdgpu-enable-ocl-mangling-mismatch-workaround=0 <%s | FileCheck -enable-var-scope -check-prefix=GCN -check-prefix=GCN-PRELINK %s
; RUN: opt -S -passes='default<O1>' -mtriple=amdgcn-- -amdgpu-use-native -amdgpu-prelink < %s | FileCheck -enable-var-scope -check-prefix=GCN -check-prefix=GCN-NATIVE %s

; GCN-LABEL: {{^}}define amdgpu_kernel void @test_sincos
; GCN-POSTLINK: call fast float @_Z3sinf(
; GCN-POSTLINK: call fast float @_Z3cosf(
; GCN-PRELINK: call fast float @_Z6sincosfPf(
; GCN-NATIVE: call fast float @_Z10native_sinf(
; GCN-NATIVE: call fast float @_Z10native_cosf(
define amdgpu_kernel void @test_sincos(float addrspace(1)* nocapture %a) {
entry:
  %tmp = load float, float addrspace(1)* %a, align 4
  %call = call fast float @_Z3sinf(float %tmp)
  store float %call, float addrspace(1)* %a, align 4
  %call2 = call fast float @_Z3cosf(float %tmp)
  %arrayidx3 = getelementptr inbounds float, float addrspace(1)* %a, i64 1
  store float %call2, float addrspace(1)* %arrayidx3, align 4
  ret void
}

declare float @_Z3sinf(float)

declare float @_Z3cosf(float)

; GCN-LABEL: {{^}}define amdgpu_kernel void @test_sincos_v2
; GCN-POSTLINK: call fast <2 x float> @_Z3sinDv2_f(
; GCN-POSTLINK: call fast <2 x float> @_Z3cosDv2_f(
; GCN-PRELINK: call fast <2 x float> @_Z6sincosDv2_fPS_(
; GCN-NATIVE: call fast <2 x float> @_Z10native_sinDv2_f(
; GCN-NATIVE: call fast <2 x float> @_Z10native_cosDv2_f(
define amdgpu_kernel void @test_sincos_v2(<2 x float> addrspace(1)* nocapture %a) {
entry:
  %tmp = load <2 x float>, <2 x float> addrspace(1)* %a, align 8
  %call = call fast <2 x float> @_Z3sinDv2_f(<2 x float> %tmp)
  store <2 x float> %call, <2 x float> addrspace(1)* %a, align 8
  %call2 = call fast <2 x float> @_Z3cosDv2_f(<2 x float> %tmp)
  %arrayidx3 = getelementptr inbounds <2 x float>, <2 x float> addrspace(1)* %a, i64 1
  store <2 x float> %call2, <2 x float> addrspace(1)* %arrayidx3, align 8
  ret void
}

declare <2 x float> @_Z3sinDv2_f(<2 x float>)

declare <2 x float> @_Z3cosDv2_f(<2 x float>)

; GCN-LABEL: {{^}}define amdgpu_kernel void @test_sincos_v3
; GCN-POSTLINK: call fast <3 x float> @_Z3sinDv3_f(
; GCN-POSTLINK: call fast <3 x float> @_Z3cosDv3_f(
; GCN-PRELINK: call fast <3 x float> @_Z6sincosDv3_fPS_(
; GCN-NATIVE: call fast <3 x float> @_Z10native_sinDv3_f(
; GCN-NATIVE: call fast <3 x float> @_Z10native_cosDv3_f(
define amdgpu_kernel void @test_sincos_v3(<3 x float> addrspace(1)* nocapture %a) {
entry:
  %castToVec4 = bitcast <3 x float> addrspace(1)* %a to <4 x float> addrspace(1)*
  %loadVec4 = load <4 x float>, <4 x float> addrspace(1)* %castToVec4, align 16
  %extractVec4 = shufflevector <4 x float> %loadVec4, <4 x float> undef, <3 x i32> <i32 0, i32 1, i32 2>
  %call = call fast <3 x float> @_Z3sinDv3_f(<3 x float> %extractVec4)
  %extractVec6 = shufflevector <3 x float> %call, <3 x float> undef, <4 x i32> <i32 0, i32 1, i32 2, i32 undef>
  store <4 x float> %extractVec6, <4 x float> addrspace(1)* %castToVec4, align 16
  %call11 = call fast <3 x float> @_Z3cosDv3_f(<3 x float> %extractVec4)
  %arrayidx12 = getelementptr inbounds <3 x float>, <3 x float> addrspace(1)* %a, i64 1
  %extractVec13 = shufflevector <3 x float> %call11, <3 x float> undef, <4 x i32> <i32 0, i32 1, i32 2, i32 undef>
  %storetmp14 = bitcast <3 x float> addrspace(1)* %arrayidx12 to <4 x float> addrspace(1)*
  store <4 x float> %extractVec13, <4 x float> addrspace(1)* %storetmp14, align 16
  ret void
}

declare <3 x float> @_Z3sinDv3_f(<3 x float>)

declare <3 x float> @_Z3cosDv3_f(<3 x float>)

; GCN-LABEL: {{^}}define amdgpu_kernel void @test_sincos_v4
; GCN-POSTLINK: call fast <4 x float> @_Z3sinDv4_f(
; GCN-POSTLINK: call fast <4 x float> @_Z3cosDv4_f(
; GCN-PRELINK: call fast <4 x float> @_Z6sincosDv4_fPS_(
; GCN-NATIVE: call fast <4 x float> @_Z10native_sinDv4_f(
; GCN-NATIVE: call fast <4 x float> @_Z10native_cosDv4_f(
define amdgpu_kernel void @test_sincos_v4(<4 x float> addrspace(1)* nocapture %a) {
entry:
  %tmp = load <4 x float>, <4 x float> addrspace(1)* %a, align 16
  %call = call fast <4 x float> @_Z3sinDv4_f(<4 x float> %tmp)
  store <4 x float> %call, <4 x float> addrspace(1)* %a, align 16
  %call2 = call fast <4 x float> @_Z3cosDv4_f(<4 x float> %tmp)
  %arrayidx3 = getelementptr inbounds <4 x float>, <4 x float> addrspace(1)* %a, i64 1
  store <4 x float> %call2, <4 x float> addrspace(1)* %arrayidx3, align 16
  ret void
}

declare <4 x float> @_Z3sinDv4_f(<4 x float>)

declare <4 x float> @_Z3cosDv4_f(<4 x float>)

; GCN-LABEL: {{^}}define amdgpu_kernel void @test_sincos_v8
; GCN-POSTLINK: call fast <8 x float> @_Z3sinDv8_f(
; GCN-POSTLINK: call fast <8 x float> @_Z3cosDv8_f(
; GCN-PRELINK: call fast <8 x float> @_Z6sincosDv8_fPS_(
; GCN-NATIVE: call fast <8 x float> @_Z10native_sinDv8_f(
; GCN-NATIVE: call fast <8 x float> @_Z10native_cosDv8_f(
define amdgpu_kernel void @test_sincos_v8(<8 x float> addrspace(1)* nocapture %a) {
entry:
  %tmp = load <8 x float>, <8 x float> addrspace(1)* %a, align 32
  %call = call fast <8 x float> @_Z3sinDv8_f(<8 x float> %tmp)
  store <8 x float> %call, <8 x float> addrspace(1)* %a, align 32
  %call2 = call fast <8 x float> @_Z3cosDv8_f(<8 x float> %tmp)
  %arrayidx3 = getelementptr inbounds <8 x float>, <8 x float> addrspace(1)* %a, i64 1
  store <8 x float> %call2, <8 x float> addrspace(1)* %arrayidx3, align 32
  ret void
}

declare <8 x float> @_Z3sinDv8_f(<8 x float>)

declare <8 x float> @_Z3cosDv8_f(<8 x float>)

; GCN-LABEL: {{^}}define amdgpu_kernel void @test_sincos_v16
; GCN-POSTLINK: call fast <16 x float> @_Z3sinDv16_f(
; GCN-POSTLINK: call fast <16 x float> @_Z3cosDv16_f(
; GCN-PRELINK: call fast <16 x float> @_Z6sincosDv16_fPS_(
; GCN-NATIVE: call fast <16 x float> @_Z10native_sinDv16_f(
; GCN-NATIVE: call fast <16 x float> @_Z10native_cosDv16_f(
define amdgpu_kernel void @test_sincos_v16(<16 x float> addrspace(1)* nocapture %a) {
entry:
  %tmp = load <16 x float>, <16 x float> addrspace(1)* %a, align 64
  %call = call fast <16 x float> @_Z3sinDv16_f(<16 x float> %tmp)
  store <16 x float> %call, <16 x float> addrspace(1)* %a, align 64
  %call2 = call fast <16 x float> @_Z3cosDv16_f(<16 x float> %tmp)
  %arrayidx3 = getelementptr inbounds <16 x float>, <16 x float> addrspace(1)* %a, i64 1
  store <16 x float> %call2, <16 x float> addrspace(1)* %arrayidx3, align 64
  ret void
}

declare <16 x float> @_Z3sinDv16_f(<16 x float>)

declare <16 x float> @_Z3cosDv16_f(<16 x float>)

; GCN-LABEL: {{^}}define amdgpu_kernel void @test_native_recip
; GCN: store float 0x3FD5555560000000, float addrspace(1)* %a
define amdgpu_kernel void @test_native_recip(float addrspace(1)* nocapture %a) {
entry:
  %call = call fast float @_Z12native_recipf(float 3.000000e+00)
  store float %call, float addrspace(1)* %a, align 4
  ret void
}

declare float @_Z12native_recipf(float)

; GCN-LABEL: {{^}}define amdgpu_kernel void @test_half_recip
; GCN: store float 0x3FD5555560000000, float addrspace(1)* %a
define amdgpu_kernel void @test_half_recip(float addrspace(1)* nocapture %a) {
entry:
  %call = call fast float @_Z10half_recipf(float 3.000000e+00)
  store float %call, float addrspace(1)* %a, align 4
  ret void
}

declare float @_Z10half_recipf(float)

; GCN-LABEL: {{^}}define amdgpu_kernel void @test_native_divide
; GCN: fmul fast float %tmp, 0x3FD5555560000000
define amdgpu_kernel void @test_native_divide(float addrspace(1)* nocapture %a) {
entry:
  %tmp = load float, float addrspace(1)* %a, align 4
  %call = call fast float @_Z13native_divideff(float %tmp, float 3.000000e+00)
  store float %call, float addrspace(1)* %a, align 4
  ret void
}

declare float @_Z13native_divideff(float, float)

; GCN-LABEL: {{^}}define amdgpu_kernel void @test_half_divide
; GCN: fmul fast float %tmp, 0x3FD5555560000000
define amdgpu_kernel void @test_half_divide(float addrspace(1)* nocapture %a) {
entry:
  %tmp = load float, float addrspace(1)* %a, align 4
  %call = call fast float @_Z11half_divideff(float %tmp, float 3.000000e+00)
  store float %call, float addrspace(1)* %a, align 4
  ret void
}

declare float @_Z11half_divideff(float, float)

; GCN-LABEL: {{^}}define amdgpu_kernel void @test_pow_0f
; GCN: store float 1.000000e+00, float addrspace(1)* %a
define amdgpu_kernel void @test_pow_0f(float addrspace(1)* nocapture %a) {
entry:
  %tmp = load float, float addrspace(1)* %a, align 4
  %call = call fast float @_Z3powff(float %tmp, float 0.000000e+00)
  store float %call, float addrspace(1)* %a, align 4
  ret void
}

declare float @_Z3powff(float, float)

; GCN-LABEL: {{^}}define amdgpu_kernel void @test_pow_0i
; GCN: store float 1.000000e+00, float addrspace(1)* %a
define amdgpu_kernel void @test_pow_0i(float addrspace(1)* nocapture %a) {
entry:
  %tmp = load float, float addrspace(1)* %a, align 4
  %call = call fast float @_Z3powff(float %tmp, float 0.000000e+00)
  store float %call, float addrspace(1)* %a, align 4
  ret void
}

; GCN-LABEL: {{^}}define amdgpu_kernel void @test_pow_1f
; GCN: %tmp = load float, float addrspace(1)* %arrayidx, align 4
; GCN: store float %tmp, float addrspace(1)* %a, align 4
define amdgpu_kernel void @test_pow_1f(float addrspace(1)* nocapture %a) {
entry:
  %arrayidx = getelementptr inbounds float, float addrspace(1)* %a, i64 1
  %tmp = load float, float addrspace(1)* %arrayidx, align 4
  %call = call fast float @_Z3powff(float %tmp, float 1.000000e+00)
  store float %call, float addrspace(1)* %a, align 4
  ret void
}

; GCN-LABEL: {{^}}define amdgpu_kernel void @test_pow_1i
; GCN: %tmp = load float, float addrspace(1)* %arrayidx, align 4
; GCN: store float %tmp, float addrspace(1)* %a, align 4
define amdgpu_kernel void @test_pow_1i(float addrspace(1)* nocapture %a) {
entry:
  %arrayidx = getelementptr inbounds float, float addrspace(1)* %a, i64 1
  %tmp = load float, float addrspace(1)* %arrayidx, align 4
  %call = call fast float @_Z3powff(float %tmp, float 1.000000e+00)
  store float %call, float addrspace(1)* %a, align 4
  ret void
}

; GCN-LABEL: {{^}}define amdgpu_kernel void @test_pow_2f
; GCN: %tmp = load float, float addrspace(1)* %a, align 4
; GCN: %__pow2 = fmul fast float %tmp, %tmp
define amdgpu_kernel void @test_pow_2f(float addrspace(1)* nocapture %a) {
entry:
  %tmp = load float, float addrspace(1)* %a, align 4
  %call = call fast float @_Z3powff(float %tmp, float 2.000000e+00)
  store float %call, float addrspace(1)* %a, align 4
  ret void
}

; GCN-LABEL: {{^}}define amdgpu_kernel void @test_pow_2i
; GCN: %tmp = load float, float addrspace(1)* %a, align 4
; GCN: %__pow2 = fmul fast float %tmp, %tmp
define amdgpu_kernel void @test_pow_2i(float addrspace(1)* nocapture %a) {
entry:
  %tmp = load float, float addrspace(1)* %a, align 4
  %call = call fast float @_Z3powff(float %tmp, float 2.000000e+00)
  store float %call, float addrspace(1)* %a, align 4
  ret void
}

; GCN-LABEL: {{^}}define amdgpu_kernel void @test_pow_m1f
; GCN: %tmp = load float, float addrspace(1)* %arrayidx, align 4
; GCN: %__powrecip = fdiv fast float 1.000000e+00, %tmp
define amdgpu_kernel void @test_pow_m1f(float addrspace(1)* nocapture %a) {
entry:
  %arrayidx = getelementptr inbounds float, float addrspace(1)* %a, i64 1
  %tmp = load float, float addrspace(1)* %arrayidx, align 4
  %call = call fast float @_Z3powff(float %tmp, float -1.000000e+00)
  store float %call, float addrspace(1)* %a, align 4
  ret void
}

; GCN-LABEL: {{^}}define amdgpu_kernel void @test_pow_m1i
; GCN: %tmp = load float, float addrspace(1)* %arrayidx, align 4
; GCN: %__powrecip = fdiv fast float 1.000000e+00, %tmp
define amdgpu_kernel void @test_pow_m1i(float addrspace(1)* nocapture %a) {
entry:
  %arrayidx = getelementptr inbounds float, float addrspace(1)* %a, i64 1
  %tmp = load float, float addrspace(1)* %arrayidx, align 4
  %call = call fast float @_Z3powff(float %tmp, float -1.000000e+00)
  store float %call, float addrspace(1)* %a, align 4
  ret void
}

; GCN-LABEL: {{^}}define amdgpu_kernel void @test_pow_half
; GCN-POSTLINK: call fast float @_Z3powff(float %tmp, float 5.000000e-01)
; GCN-PRELINK: %__pow2sqrt = tail call fast float @_Z4sqrtf(float %tmp)
define amdgpu_kernel void @test_pow_half(float addrspace(1)* nocapture %a) {
entry:
  %arrayidx = getelementptr inbounds float, float addrspace(1)* %a, i64 1
  %tmp = load float, float addrspace(1)* %arrayidx, align 4
  %call = call fast float @_Z3powff(float %tmp, float 5.000000e-01)
  store float %call, float addrspace(1)* %a, align 4
  ret void
}

; GCN-LABEL: {{^}}define amdgpu_kernel void @test_pow_mhalf
; GCN-POSTLINK: call fast float @_Z3powff(float %tmp, float -5.000000e-01)
; GCN-PRELINK: %__pow2rsqrt = tail call fast float @_Z5rsqrtf(float %tmp)
define amdgpu_kernel void @test_pow_mhalf(float addrspace(1)* nocapture %a) {
entry:
  %arrayidx = getelementptr inbounds float, float addrspace(1)* %a, i64 1
  %tmp = load float, float addrspace(1)* %arrayidx, align 4
  %call = call fast float @_Z3powff(float %tmp, float -5.000000e-01)
  store float %call, float addrspace(1)* %a, align 4
  ret void
}

; GCN-LABEL: {{^}}define amdgpu_kernel void @test_pow_c
; GCN: %__powx2 = fmul fast float %tmp, %tmp
; GCN: %__powx21 = fmul fast float %__powx2, %__powx2
; GCN: %__powx22 = fmul fast float %__powx2, %tmp
; GCN: %[[r0:.*]] = fmul fast float %__powx21, %__powx21
; GCN: %__powprod3 = fmul fast float %[[r0]], %__powx22
define amdgpu_kernel void @test_pow_c(float addrspace(1)* nocapture %a) {
entry:
  %arrayidx = getelementptr inbounds float, float addrspace(1)* %a, i64 1
  %tmp = load float, float addrspace(1)* %arrayidx, align 4
  %call = call fast float @_Z3powff(float %tmp, float 1.100000e+01)
  store float %call, float addrspace(1)* %a, align 4
  ret void
}

; GCN-LABEL: {{^}}define amdgpu_kernel void @test_powr_c
; GCN: %__powx2 = fmul fast float %tmp, %tmp
; GCN: %__powx21 = fmul fast float %__powx2, %__powx2
; GCN: %__powx22 = fmul fast float %__powx2, %tmp
; GCN: %[[r0:.*]] = fmul fast float %__powx21, %__powx21
; GCN: %__powprod3 = fmul fast float %[[r0]], %__powx22
define amdgpu_kernel void @test_powr_c(float addrspace(1)* nocapture %a) {
entry:
  %arrayidx = getelementptr inbounds float, float addrspace(1)* %a, i64 1
  %tmp = load float, float addrspace(1)* %arrayidx, align 4
  %call = call fast float @_Z4powrff(float %tmp, float 1.100000e+01)
  store float %call, float addrspace(1)* %a, align 4
  ret void
}

declare float @_Z4powrff(float, float)

; GCN-LABEL: {{^}}define amdgpu_kernel void @test_pown_c
; GCN: %__powx2 = fmul fast float %tmp, %tmp
; GCN: %__powx21 = fmul fast float %__powx2, %__powx2
; GCN: %__powx22 = fmul fast float %__powx2, %tmp
; GCN: %[[r0:.*]] = fmul fast float %__powx21, %__powx21
; GCN: %__powprod3 = fmul fast float %[[r0]], %__powx22
define amdgpu_kernel void @test_pown_c(float addrspace(1)* nocapture %a) {
entry:
  %arrayidx = getelementptr inbounds float, float addrspace(1)* %a, i64 1
  %tmp = load float, float addrspace(1)* %arrayidx, align 4
  %call = call fast float @_Z4pownfi(float %tmp, i32 11)
  store float %call, float addrspace(1)* %a, align 4
  ret void
}

declare float @_Z4pownfi(float, i32)

; GCN-LABEL: {{^}}define amdgpu_kernel void @test_pow
; GCN-POSTLINK: call fast float @_Z3powff(float %tmp, float 1.013000e+03)
; GCN-PRELINK: %__fabs = tail call fast float @_Z4fabsf(float %tmp)
; GCN-PRELINK: %__log2 = tail call fast float @_Z4log2f(float %__fabs)
; GCN-PRELINK: %__ylogx = fmul fast float %__log2, 1.013000e+03
; GCN-PRELINK: %__exp2 = tail call fast float @_Z4exp2f(float %__ylogx)
; GCN-PRELINK: %[[r0:.*]] = bitcast float %tmp to i32
; GCN-PRELINK: %__pow_sign = and i32 %[[r0]], -2147483648
; GCN-PRELINK: %[[r1:.*]] = bitcast float %__exp2 to i32
; GCN-PRELINK: %[[r2:.*]] = or i32 %__pow_sign, %[[r1]]
; GCN-PRELINK: %[[r3:.*]] = bitcast float addrspace(1)* %a to i32 addrspace(1)*
; GCN-PRELINK: store i32 %[[r2]], i32 addrspace(1)* %[[r3]], align 4
define amdgpu_kernel void @test_pow(float addrspace(1)* nocapture %a) {
entry:
  %tmp = load float, float addrspace(1)* %a, align 4
  %call = call fast float @_Z3powff(float %tmp, float 1.013000e+03)
  store float %call, float addrspace(1)* %a, align 4
  ret void
}

; GCN-LABEL: {{^}}define amdgpu_kernel void @test_powr
; GCN-POSTLINK: call fast float @_Z4powrff(float %tmp, float %tmp1)
; GCN-PRELINK: %__log2 = tail call fast float @_Z4log2f(float %tmp)
; GCN-PRELINK: %__ylogx = fmul fast float %__log2, %tmp1
; GCN-PRELINK: %__exp2 = tail call fast float @_Z4exp2f(float %__ylogx)
; GCN-PRELINK: store float %__exp2, float addrspace(1)* %a, align 4
; GCN-NATIVE:  %__log2 = tail call fast float @_Z11native_log2f(float %tmp)
; GCN-NATIVE:  %__ylogx = fmul fast float %__log2, %tmp1
; GCN-NATIVE:  %__exp2 = tail call fast float @_Z11native_exp2f(float %__ylogx)
; GCN-NATIVE:  store float %__exp2, float addrspace(1)* %a, align 4
define amdgpu_kernel void @test_powr(float addrspace(1)* nocapture %a) {
entry:
  %tmp = load float, float addrspace(1)* %a, align 4
  %arrayidx1 = getelementptr inbounds float, float addrspace(1)* %a, i64 1
  %tmp1 = load float, float addrspace(1)* %arrayidx1, align 4
  %call = call fast float @_Z4powrff(float %tmp, float %tmp1)
  store float %call, float addrspace(1)* %a, align 4
  ret void
}

; GCN-LABEL: {{^}}define amdgpu_kernel void @test_pown
; GCN-POSTLINK: call fast float @_Z4pownfi(float %tmp, i32 %conv)
; GCN-PRELINK: %conv = fptosi float %tmp1 to i32
; GCN-PRELINK: %__fabs = tail call fast float @_Z4fabsf(float %tmp)
; GCN-PRELINK: %__log2 = tail call fast float @_Z4log2f(float %__fabs)
; GCN-PRELINK: %pownI2F = sitofp i32 %conv to float
; GCN-PRELINK: %__ylogx = fmul fast float %__log2, %pownI2F
; GCN-PRELINK: %__exp2 = tail call fast float @_Z4exp2f(float %__ylogx)
; GCN-PRELINK: %__yeven = shl i32 %conv, 31
; GCN-PRELINK: %[[r0:.*]] = bitcast float %tmp to i32
; GCN-PRELINK: %__pow_sign = and i32 %__yeven, %[[r0]]
; GCN-PRELINK: %[[r1:.*]] = bitcast float %__exp2 to i32
; GCN-PRELINK: %[[r2:.*]] = or i32 %__pow_sign, %[[r1]]
; GCN-PRELINK: %[[r3:.*]] = bitcast float addrspace(1)* %a to i32 addrspace(1)*
; GCN-PRELINK: store i32 %[[r2]], i32 addrspace(1)* %[[r3]], align 4
define amdgpu_kernel void @test_pown(float addrspace(1)* nocapture %a) {
entry:
  %tmp = load float, float addrspace(1)* %a, align 4
  %arrayidx1 = getelementptr inbounds float, float addrspace(1)* %a, i64 1
  %tmp1 = load float, float addrspace(1)* %arrayidx1, align 4
  %conv = fptosi float %tmp1 to i32
  %call = call fast float @_Z4pownfi(float %tmp, i32 %conv)
  store float %call, float addrspace(1)* %a, align 4
  ret void
}

; GCN-LABEL: {{^}}define amdgpu_kernel void @test_rootn_1
; GCN: %tmp = load float, float addrspace(1)* %arrayidx, align 4
; GCN: store float %tmp, float addrspace(1)* %a, align 4
define amdgpu_kernel void @test_rootn_1(float addrspace(1)* nocapture %a) {
entry:
  %arrayidx = getelementptr inbounds float, float addrspace(1)* %a, i64 1
  %tmp = load float, float addrspace(1)* %arrayidx, align 4
  %call = call fast float @_Z5rootnfi(float %tmp, i32 1)
  store float %call, float addrspace(1)* %a, align 4
  ret void
}

declare float @_Z5rootnfi(float, i32)

; GCN-LABEL: {{^}}define amdgpu_kernel void @test_rootn_2
; GCN-POSTLINK: call fast float @_Z5rootnfi(float %tmp, i32 2)
; GCN-PRELINK: %__rootn2sqrt = tail call fast float @_Z4sqrtf(float %tmp)
define amdgpu_kernel void @test_rootn_2(float addrspace(1)* nocapture %a) {
entry:
  %tmp = load float, float addrspace(1)* %a, align 4
  %call = call fast float @_Z5rootnfi(float %tmp, i32 2)
  store float %call, float addrspace(1)* %a, align 4
  ret void
}

; GCN-LABEL: {{^}}define amdgpu_kernel void @test_rootn_3
; GCN-POSTLINK: call fast float @_Z5rootnfi(float %tmp, i32 3)
; GCN-PRELINK: %__rootn2cbrt = tail call fast float @_Z4cbrtf(float %tmp)
define amdgpu_kernel void @test_rootn_3(float addrspace(1)* nocapture %a) {
entry:
  %tmp = load float, float addrspace(1)* %a, align 4
  %call = call fast float @_Z5rootnfi(float %tmp, i32 3)
  store float %call, float addrspace(1)* %a, align 4
  ret void
}

; GCN-LABEL: {{^}}define amdgpu_kernel void @test_rootn_m1
; GCN: fdiv fast float 1.000000e+00, %tmp
define amdgpu_kernel void @test_rootn_m1(float addrspace(1)* nocapture %a) {
entry:
  %tmp = load float, float addrspace(1)* %a, align 4
  %call = call fast float @_Z5rootnfi(float %tmp, i32 -1)
  store float %call, float addrspace(1)* %a, align 4
  ret void
}

; GCN-LABEL: {{^}}define amdgpu_kernel void @test_rootn_m2
; GCN-POSTLINK: call fast float @_Z5rootnfi(float %tmp, i32 -2)
; GCN-PRELINK: %__rootn2rsqrt = tail call fast float @_Z5rsqrtf(float %tmp)
define amdgpu_kernel void @test_rootn_m2(float addrspace(1)* nocapture %a) {
entry:
  %tmp = load float, float addrspace(1)* %a, align 4
  %call = call fast float @_Z5rootnfi(float %tmp, i32 -2)
  store float %call, float addrspace(1)* %a, align 4
  ret void
}

; GCN-LABEL: {{^}}define amdgpu_kernel void @test_fma_0x
; GCN: store float %y, float addrspace(1)* %a
define amdgpu_kernel void @test_fma_0x(float addrspace(1)* nocapture %a, float %y) {
entry:
  %tmp = load float, float addrspace(1)* %a, align 4
  %call = call fast float @_Z3fmafff(float 0.000000e+00, float %tmp, float %y)
  store float %call, float addrspace(1)* %a, align 4
  ret void
}

declare float @_Z3fmafff(float, float, float)

; GCN-LABEL: {{^}}define amdgpu_kernel void @test_fma_x0
; GCN: store float %y, float addrspace(1)* %a
define amdgpu_kernel void @test_fma_x0(float addrspace(1)* nocapture %a, float %y) {
entry:
  %tmp = load float, float addrspace(1)* %a, align 4
  %call = call fast float @_Z3fmafff(float %tmp, float 0.000000e+00, float %y)
  store float %call, float addrspace(1)* %a, align 4
  ret void
}

; GCN-LABEL: {{^}}define amdgpu_kernel void @test_mad_0x
; GCN: store float %y, float addrspace(1)* %a
define amdgpu_kernel void @test_mad_0x(float addrspace(1)* nocapture %a, float %y) {
entry:
  %tmp = load float, float addrspace(1)* %a, align 4
  %call = call fast float @_Z3madfff(float 0.000000e+00, float %tmp, float %y)
  store float %call, float addrspace(1)* %a, align 4
  ret void
}

declare float @_Z3madfff(float, float, float)

; GCN-LABEL: {{^}}define amdgpu_kernel void @test_mad_x0
; GCN: store float %y, float addrspace(1)* %a
define amdgpu_kernel void @test_mad_x0(float addrspace(1)* nocapture %a, float %y) {
entry:
  %tmp = load float, float addrspace(1)* %a, align 4
  %call = call fast float @_Z3madfff(float %tmp, float 0.000000e+00, float %y)
  store float %call, float addrspace(1)* %a, align 4
  ret void
}

; GCN-LABEL: {{^}}define amdgpu_kernel void @test_fma_x1y
; GCN: %fmaadd = fadd fast float %tmp, %y
define amdgpu_kernel void @test_fma_x1y(float addrspace(1)* nocapture %a, float %y) {
entry:
  %tmp = load float, float addrspace(1)* %a, align 4
  %call = call fast float @_Z3fmafff(float %tmp, float 1.000000e+00, float %y)
  store float %call, float addrspace(1)* %a, align 4
  ret void
}

; GCN-LABEL: {{^}}define amdgpu_kernel void @test_fma_1xy
; GCN: %fmaadd = fadd fast float %tmp, %y
define amdgpu_kernel void @test_fma_1xy(float addrspace(1)* nocapture %a, float %y) {
entry:
  %tmp = load float, float addrspace(1)* %a, align 4
  %call = call fast float @_Z3fmafff(float 1.000000e+00, float %tmp, float %y)
  store float %call, float addrspace(1)* %a, align 4
  ret void
}

; GCN-LABEL: {{^}}define amdgpu_kernel void @test_fma_xy0
; GCN: %fmamul = fmul fast float %tmp1, %tmp
define amdgpu_kernel void @test_fma_xy0(float addrspace(1)* nocapture %a) {
entry:
  %arrayidx = getelementptr inbounds float, float addrspace(1)* %a, i64 1
  %tmp = load float, float addrspace(1)* %arrayidx, align 4
  %tmp1 = load float, float addrspace(1)* %a, align 4
  %call = call fast float @_Z3fmafff(float %tmp, float %tmp1, float 0.000000e+00)
  store float %call, float addrspace(1)* %a, align 4
  ret void
}

; GCN-LABEL: {{^}}define amdgpu_kernel void @test_use_native_exp
; GCN-NATIVE: call fast float @_Z10native_expf(float %tmp)
define amdgpu_kernel void @test_use_native_exp(float addrspace(1)* nocapture %a) {
entry:
  %tmp = load float, float addrspace(1)* %a, align 4
  %call = call fast float @_Z3expf(float %tmp)
  store float %call, float addrspace(1)* %a, align 4
  ret void
}

declare float @_Z3expf(float)

; GCN-LABEL: {{^}}define amdgpu_kernel void @test_use_native_exp2
; GCN-NATIVE: call fast float @_Z11native_exp2f(float %tmp)
define amdgpu_kernel void @test_use_native_exp2(float addrspace(1)* nocapture %a) {
entry:
  %tmp = load float, float addrspace(1)* %a, align 4
  %call = call fast float @_Z4exp2f(float %tmp)
  store float %call, float addrspace(1)* %a, align 4
  ret void
}

declare float @_Z4exp2f(float)

; GCN-LABEL: {{^}}define amdgpu_kernel void @test_use_native_exp10
; GCN-NATIVE: call fast float @_Z12native_exp10f(float %tmp)
define amdgpu_kernel void @test_use_native_exp10(float addrspace(1)* nocapture %a) {
entry:
  %tmp = load float, float addrspace(1)* %a, align 4
  %call = call fast float @_Z5exp10f(float %tmp)
  store float %call, float addrspace(1)* %a, align 4
  ret void
}

declare float @_Z5exp10f(float)

; GCN-LABEL: {{^}}define amdgpu_kernel void @test_use_native_log
; GCN-NATIVE: call fast float @_Z10native_logf(float %tmp)
define amdgpu_kernel void @test_use_native_log(float addrspace(1)* nocapture %a) {
entry:
  %tmp = load float, float addrspace(1)* %a, align 4
  %call = call fast float @_Z3logf(float %tmp)
  store float %call, float addrspace(1)* %a, align 4
  ret void
}

declare float @_Z3logf(float)

; GCN-LABEL: {{^}}define amdgpu_kernel void @test_use_native_log2
; GCN-NATIVE: call fast float @_Z11native_log2f(float %tmp)
define amdgpu_kernel void @test_use_native_log2(float addrspace(1)* nocapture %a) {
entry:
  %tmp = load float, float addrspace(1)* %a, align 4
  %call = call fast float @_Z4log2f(float %tmp)
  store float %call, float addrspace(1)* %a, align 4
  ret void
}

declare float @_Z4log2f(float)

; GCN-LABEL: {{^}}define amdgpu_kernel void @test_use_native_log10
; GCN-NATIVE: call fast float @_Z12native_log10f(float %tmp)
define amdgpu_kernel void @test_use_native_log10(float addrspace(1)* nocapture %a) {
entry:
  %tmp = load float, float addrspace(1)* %a, align 4
  %call = call fast float @_Z5log10f(float %tmp)
  store float %call, float addrspace(1)* %a, align 4
  ret void
}

declare float @_Z5log10f(float)

; GCN-LABEL: {{^}}define amdgpu_kernel void @test_use_native_powr
; GCN-NATIVE: %tmp1 = load float, float addrspace(1)* %arrayidx1, align 4
; GCN-NATIVE: %__log2 = tail call fast float @_Z11native_log2f(float %tmp)
; GCN-NATIVE: %__ylogx = fmul fast float %__log2, %tmp1
; GCN-NATIVE: %__exp2 = tail call fast float @_Z11native_exp2f(float %__ylogx)
; GCN-NATIVE: store float %__exp2, float addrspace(1)* %a, align 4
define amdgpu_kernel void @test_use_native_powr(float addrspace(1)* nocapture %a) {
entry:
  %tmp = load float, float addrspace(1)* %a, align 4
  %arrayidx1 = getelementptr inbounds float, float addrspace(1)* %a, i64 1
  %tmp1 = load float, float addrspace(1)* %arrayidx1, align 4
  %call = call fast float @_Z4powrff(float %tmp, float %tmp1)
  store float %call, float addrspace(1)* %a, align 4
  ret void
}

; GCN-LABEL: {{^}}define amdgpu_kernel void @test_use_native_sqrt
; GCN-NATIVE: call fast float @_Z11native_sqrtf(float %tmp)
define amdgpu_kernel void @test_use_native_sqrt(float addrspace(1)* nocapture %a) {
entry:
  %tmp = load float, float addrspace(1)* %a, align 4
  %call = call fast float @_Z4sqrtf(float %tmp)
  store float %call, float addrspace(1)* %a, align 4
  ret void
}

; GCN-LABEL: {{^}}define amdgpu_kernel void @test_dont_use_native_sqrt_fast_f64
; GCN: call fast double @_Z4sqrtd(double %tmp)
define amdgpu_kernel void @test_dont_use_native_sqrt_fast_f64(double addrspace(1)* nocapture %a) {
entry:
  %tmp = load double, double addrspace(1)* %a, align 8
  %call = call fast double @_Z4sqrtd(double %tmp)
  store double %call, double addrspace(1)* %a, align 8
  ret void
}

declare float @_Z4sqrtf(float)
declare double @_Z4sqrtd(double)

; GCN-LABEL: {{^}}define amdgpu_kernel void @test_use_native_rsqrt
; GCN-NATIVE: call fast float @_Z12native_rsqrtf(float %tmp)
define amdgpu_kernel void @test_use_native_rsqrt(float addrspace(1)* nocapture %a) {
entry:
  %tmp = load float, float addrspace(1)* %a, align 4
  %call = call fast float @_Z5rsqrtf(float %tmp)
  store float %call, float addrspace(1)* %a, align 4
  ret void
}

declare float @_Z5rsqrtf(float)

; GCN-LABEL: {{^}}define amdgpu_kernel void @test_use_native_tan
; GCN-NATIVE: call fast float @_Z10native_tanf(float %tmp)
define amdgpu_kernel void @test_use_native_tan(float addrspace(1)* nocapture %a) {
entry:
  %tmp = load float, float addrspace(1)* %a, align 4
  %call = call fast float @_Z3tanf(float %tmp)
  store float %call, float addrspace(1)* %a, align 4
  ret void
}

declare float @_Z3tanf(float)

; GCN-LABEL: {{^}}define amdgpu_kernel void @test_use_native_sincos
; GCN-NATIVE: call float @_Z10native_sinf(float %tmp)
; GCN-NATIVE: call float @_Z10native_cosf(float %tmp)
define amdgpu_kernel void @test_use_native_sincos(float addrspace(1)* %a) {
entry:
  %tmp = load float, float addrspace(1)* %a, align 4
  %arrayidx1 = getelementptr inbounds float, float addrspace(1)* %a, i64 1
  %tmp1 = addrspacecast float addrspace(1)* %arrayidx1 to float*
  %call = call fast float @_Z6sincosfPf(float %tmp, float* %tmp1)
  store float %call, float addrspace(1)* %a, align 4
  ret void
}

declare float @_Z6sincosfPf(float, float*)

%opencl.pipe_t = type opaque
%opencl.reserve_id_t = type opaque

; GCN-LABEL: {{^}}define amdgpu_kernel void @test_read_pipe(%opencl.pipe_t addrspace(1)* %p, i32 addrspace(1)* %ptr)
; GCN-PRELINK: call i32 @__read_pipe_2_4(%opencl.pipe_t addrspace(1)* %{{.*}}, i32* %{{.*}}) #[[$NOUNWIND:[0-9]+]]
; GCN-PRELINK: call i32 @__read_pipe_4_4(%opencl.pipe_t addrspace(1)* %{{.*}}, %opencl.reserve_id_t addrspace(5)* %{{.*}}, i32 2, i32* %{{.*}}) #[[$NOUNWIND]]
define amdgpu_kernel void @test_read_pipe(%opencl.pipe_t addrspace(1)* %p, i32 addrspace(1)* %ptr) local_unnamed_addr {
entry:
  %tmp = bitcast i32 addrspace(1)* %ptr to i8 addrspace(1)*
  %tmp1 = addrspacecast i8 addrspace(1)* %tmp to i8*
  %tmp2 = call i32 @__read_pipe_2(%opencl.pipe_t addrspace(1)* %p, i8* %tmp1, i32 4, i32 4) #0
  %tmp3 = call %opencl.reserve_id_t addrspace(5)* @__reserve_read_pipe(%opencl.pipe_t addrspace(1)* %p, i32 2, i32 4, i32 4)
  %tmp4 = call i32 @__read_pipe_4(%opencl.pipe_t addrspace(1)* %p, %opencl.reserve_id_t addrspace(5)* %tmp3, i32 2, i8* %tmp1, i32 4, i32 4) #0
  call void @__commit_read_pipe(%opencl.pipe_t addrspace(1)* %p, %opencl.reserve_id_t addrspace(5)* %tmp3, i32 4, i32 4)
  ret void
}

declare i32 @__read_pipe_2(%opencl.pipe_t addrspace(1)*, i8*, i32, i32)

declare %opencl.reserve_id_t addrspace(5)* @__reserve_read_pipe(%opencl.pipe_t addrspace(1)*, i32, i32, i32)

declare i32 @__read_pipe_4(%opencl.pipe_t addrspace(1)*, %opencl.reserve_id_t addrspace(5)*, i32, i8*, i32, i32)

declare void @__commit_read_pipe(%opencl.pipe_t addrspace(1)*, %opencl.reserve_id_t addrspace(5)*, i32, i32)

; GCN-LABEL: {{^}}define amdgpu_kernel void @test_write_pipe(%opencl.pipe_t addrspace(1)* %p, i32 addrspace(1)* %ptr)
; GCN-PRELINK: call i32 @__write_pipe_2_4(%opencl.pipe_t addrspace(1)* %{{.*}}, i32* %{{.*}}) #[[$NOUNWIND]]
; GCN-PRELINK: call i32 @__write_pipe_4_4(%opencl.pipe_t addrspace(1)* %{{.*}}, %opencl.reserve_id_t addrspace(5)* %{{.*}}, i32 2, i32* %{{.*}}) #[[$NOUNWIND]]
define amdgpu_kernel void @test_write_pipe(%opencl.pipe_t addrspace(1)* %p, i32 addrspace(1)* %ptr) local_unnamed_addr {
entry:
  %tmp = bitcast i32 addrspace(1)* %ptr to i8 addrspace(1)*
  %tmp1 = addrspacecast i8 addrspace(1)* %tmp to i8*
  %tmp2 = call i32 @__write_pipe_2(%opencl.pipe_t addrspace(1)* %p, i8* %tmp1, i32 4, i32 4) #0
  %tmp3 = call %opencl.reserve_id_t addrspace(5)* @__reserve_write_pipe(%opencl.pipe_t addrspace(1)* %p, i32 2, i32 4, i32 4) #0
  %tmp4 = call i32 @__write_pipe_4(%opencl.pipe_t addrspace(1)* %p, %opencl.reserve_id_t addrspace(5)* %tmp3, i32 2, i8* %tmp1, i32 4, i32 4) #0
  call void @__commit_write_pipe(%opencl.pipe_t addrspace(1)* %p, %opencl.reserve_id_t addrspace(5)* %tmp3, i32 4, i32 4) #0
  ret void
}

declare i32 @__write_pipe_2(%opencl.pipe_t addrspace(1)*, i8*, i32, i32) local_unnamed_addr

declare %opencl.reserve_id_t addrspace(5)* @__reserve_write_pipe(%opencl.pipe_t addrspace(1)*, i32, i32, i32) local_unnamed_addr

declare i32 @__write_pipe_4(%opencl.pipe_t addrspace(1)*, %opencl.reserve_id_t addrspace(5)*, i32, i8*, i32, i32) local_unnamed_addr

declare void @__commit_write_pipe(%opencl.pipe_t addrspace(1)*, %opencl.reserve_id_t addrspace(5)*, i32, i32) local_unnamed_addr

%struct.S = type { [100 x i32] }

; GCN-LABEL: {{^}}define amdgpu_kernel void @test_pipe_size
; GCN-PRELINK: call i32 @__read_pipe_2_1(%opencl.pipe_t addrspace(1)* %{{.*}} i8* %{{.*}}) #[[$NOUNWIND]]
; GCN-PRELINK: call i32 @__read_pipe_2_2(%opencl.pipe_t addrspace(1)* %{{.*}} i16* %{{.*}}) #[[$NOUNWIND]]
; GCN-PRELINK: call i32 @__read_pipe_2_4(%opencl.pipe_t addrspace(1)* %{{.*}} i32* %{{.*}}) #[[$NOUNWIND]]
; GCN-PRELINK: call i32 @__read_pipe_2_8(%opencl.pipe_t addrspace(1)* %{{.*}} i64* %{{.*}}) #[[$NOUNWIND]]
; GCN-PRELINK: call i32 @__read_pipe_2_16(%opencl.pipe_t addrspace(1)* %{{.*}}, <2 x i64>* %{{.*}}) #[[$NOUNWIND]]
; GCN-PRELINK: call i32 @__read_pipe_2_32(%opencl.pipe_t addrspace(1)* %{{.*}}, <4 x i64>* %{{.*}} #[[$NOUNWIND]]
; GCN-PRELINK: call i32 @__read_pipe_2_64(%opencl.pipe_t addrspace(1)* %{{.*}}, <8 x i64>* %{{.*}} #[[$NOUNWIND]]
; GCN-PRELINK: call i32 @__read_pipe_2_128(%opencl.pipe_t addrspace(1)* %{{.*}}, <16 x i64>* %{{.*}} #[[$NOUNWIND]]
; GCN-PRELINK: call i32 @__read_pipe_2(%opencl.pipe_t addrspace(1)* %{{.*}}, i8* %{{.*}} i32 400, i32 4) #[[$NOUNWIND]]
define amdgpu_kernel void @test_pipe_size(%opencl.pipe_t addrspace(1)* %p1, i8 addrspace(1)* %ptr1, %opencl.pipe_t addrspace(1)* %p2, i16 addrspace(1)* %ptr2, %opencl.pipe_t addrspace(1)* %p4, i32 addrspace(1)* %ptr4, %opencl.pipe_t addrspace(1)* %p8, i64 addrspace(1)* %ptr8, %opencl.pipe_t addrspace(1)* %p16, <2 x i64> addrspace(1)* %ptr16, %opencl.pipe_t addrspace(1)* %p32, <4 x i64> addrspace(1)* %ptr32, %opencl.pipe_t addrspace(1)* %p64, <8 x i64> addrspace(1)* %ptr64, %opencl.pipe_t addrspace(1)* %p128, <16 x i64> addrspace(1)* %ptr128, %opencl.pipe_t addrspace(1)* %pu, %struct.S addrspace(1)* %ptru) local_unnamed_addr #0 {
entry:
  %tmp = addrspacecast i8 addrspace(1)* %ptr1 to i8*
  %tmp1 = call i32 @__read_pipe_2(%opencl.pipe_t addrspace(1)* %p1, i8* %tmp, i32 1, i32 1) #0
  %tmp2 = bitcast i16 addrspace(1)* %ptr2 to i8 addrspace(1)*
  %tmp3 = addrspacecast i8 addrspace(1)* %tmp2 to i8*
  %tmp4 = call i32 @__read_pipe_2(%opencl.pipe_t addrspace(1)* %p2, i8* %tmp3, i32 2, i32 2) #0
  %tmp5 = bitcast i32 addrspace(1)* %ptr4 to i8 addrspace(1)*
  %tmp6 = addrspacecast i8 addrspace(1)* %tmp5 to i8*
  %tmp7 = call i32 @__read_pipe_2(%opencl.pipe_t addrspace(1)* %p4, i8* %tmp6, i32 4, i32 4) #0
  %tmp8 = bitcast i64 addrspace(1)* %ptr8 to i8 addrspace(1)*
  %tmp9 = addrspacecast i8 addrspace(1)* %tmp8 to i8*
  %tmp10 = call i32 @__read_pipe_2(%opencl.pipe_t addrspace(1)* %p8, i8* %tmp9, i32 8, i32 8) #0
  %tmp11 = bitcast <2 x i64> addrspace(1)* %ptr16 to i8 addrspace(1)*
  %tmp12 = addrspacecast i8 addrspace(1)* %tmp11 to i8*
  %tmp13 = call i32 @__read_pipe_2(%opencl.pipe_t addrspace(1)* %p16, i8* %tmp12, i32 16, i32 16) #0
  %tmp14 = bitcast <4 x i64> addrspace(1)* %ptr32 to i8 addrspace(1)*
  %tmp15 = addrspacecast i8 addrspace(1)* %tmp14 to i8*
  %tmp16 = call i32 @__read_pipe_2(%opencl.pipe_t addrspace(1)* %p32, i8* %tmp15, i32 32, i32 32) #0
  %tmp17 = bitcast <8 x i64> addrspace(1)* %ptr64 to i8 addrspace(1)*
  %tmp18 = addrspacecast i8 addrspace(1)* %tmp17 to i8*
  %tmp19 = call i32 @__read_pipe_2(%opencl.pipe_t addrspace(1)* %p64, i8* %tmp18, i32 64, i32 64) #0
  %tmp20 = bitcast <16 x i64> addrspace(1)* %ptr128 to i8 addrspace(1)*
  %tmp21 = addrspacecast i8 addrspace(1)* %tmp20 to i8*
  %tmp22 = call i32 @__read_pipe_2(%opencl.pipe_t addrspace(1)* %p128, i8* %tmp21, i32 128, i32 128) #0
  %tmp23 = bitcast %struct.S addrspace(1)* %ptru to i8 addrspace(1)*
  %tmp24 = addrspacecast i8 addrspace(1)* %tmp23 to i8*
  %tmp25 = call i32 @__read_pipe_2(%opencl.pipe_t addrspace(1)* %pu, i8* %tmp24, i32 400, i32 4) #0
  ret void
}

; GCN-PRELINK: declare float @_Z4fabsf(float) local_unnamed_addr #[[$NOUNWIND_READONLY:[0-9]+]]
; GCN-PRELINK: declare float @_Z4cbrtf(float) local_unnamed_addr #[[$NOUNWIND_READONLY]]
; GCN-PRELINK: declare float @_Z11native_sqrtf(float) local_unnamed_addr #[[$NOUNWIND_READONLY]]

; GCN-PRELINK: attributes #[[$NOUNWIND]] = { nounwind }
; GCN-PRELINK: attributes #[[$NOUNWIND_READONLY]] = { nofree nounwind memory(read) }
attributes #0 = { nounwind }
