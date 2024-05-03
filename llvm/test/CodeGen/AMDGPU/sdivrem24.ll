; RUN: llc -mtriple=amdgcn < %s | FileCheck -check-prefix=SI -check-prefix=FUNC %s
; RUN: llc -mtriple=amdgcn -mcpu=tonga -mattr=-flat-for-global < %s | FileCheck -check-prefix=SI -check-prefix=FUNC %s
; RUN: llc -mtriple=r600 -mcpu=redwood < %s | FileCheck -check-prefix=EG -check-prefix=FUNC %s

; FUNC-LABEL: {{^}}sdiv24_i8:
; SI: v_cvt_f32_i32
; SI: v_cvt_f32_i32
; SI: v_rcp_iflag_f32
; SI: v_cvt_i32_f32

; EG: INT_TO_FLT
; EG-DAG: INT_TO_FLT
; EG-DAG: RECIP_IEEE
; EG: FLT_TO_INT
define amdgpu_kernel void @sdiv24_i8(ptr addrspace(1) %out, ptr addrspace(1) %in) {
  %den_ptr = getelementptr i8, ptr addrspace(1) %in, i8 1
  %num = load i8, ptr addrspace(1) %in
  %den = load i8, ptr addrspace(1) %den_ptr
  %result = sdiv i8 %num, %den
  store i8 %result, ptr addrspace(1) %out
  ret void
}

; FUNC-LABEL: {{^}}sdiv24_i16:
; SI: v_cvt_f32_i32
; SI: v_cvt_f32_i32
; SI: v_rcp_iflag_f32
; SI: v_cvt_i32_f32

; EG: INT_TO_FLT
; EG-DAG: INT_TO_FLT
; EG-DAG: RECIP_IEEE
; EG: FLT_TO_INT
define amdgpu_kernel void @sdiv24_i16(ptr addrspace(1) %out, ptr addrspace(1) %in) {
  %den_ptr = getelementptr i16, ptr addrspace(1) %in, i16 1
  %num = load i16, ptr addrspace(1) %in, align 2
  %den = load i16, ptr addrspace(1) %den_ptr, align 2
  %result = sdiv i16 %num, %den
  store i16 %result, ptr addrspace(1) %out, align 2
  ret void
}

; FUNC-LABEL: {{^}}sdiv24_i32:
; SI: v_cvt_f32_i32
; SI: v_cvt_f32_i32
; SI: v_rcp_iflag_f32
; SI: v_cvt_i32_f32

; EG: INT_TO_FLT
; EG-DAG: INT_TO_FLT
; EG-DAG: RECIP_IEEE
; EG: FLT_TO_INT
define amdgpu_kernel void @sdiv24_i32(ptr addrspace(1) %out, ptr addrspace(1) %in) {
  %den_ptr = getelementptr i32, ptr addrspace(1) %in, i32 1
  %num = load i32, ptr addrspace(1) %in, align 4
  %den = load i32, ptr addrspace(1) %den_ptr, align 4
  %num.i24.0 = shl i32 %num, 8
  %den.i24.0 = shl i32 %den, 8
  %num.i24 = ashr i32 %num.i24.0, 8
  %den.i24 = ashr i32 %den.i24.0, 8
  %result = sdiv i32 %num.i24, %den.i24
  store i32 %result, ptr addrspace(1) %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}sdiv25_i32:
; SI-NOT: v_cvt_f32_i32
; SI-NOT: v_rcp_f32

; EG-NOT: INT_TO_FLT
; EG-NOT: RECIP_IEEE
define amdgpu_kernel void @sdiv25_i32(ptr addrspace(1) %out, ptr addrspace(1) %in) {
  %den_ptr = getelementptr i32, ptr addrspace(1) %in, i32 1
  %num = load i32, ptr addrspace(1) %in, align 4
  %den = load i32, ptr addrspace(1) %den_ptr, align 4
  %num.i24.0 = shl i32 %num, 7
  %den.i24.0 = shl i32 %den, 7
  %num.i24 = ashr i32 %num.i24.0, 7
  %den.i24 = ashr i32 %den.i24.0, 7
  %result = sdiv i32 %num.i24, %den.i24
  store i32 %result, ptr addrspace(1) %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}test_no_sdiv24_i32_1:
; SI-NOT: v_cvt_f32_i32
; SI-NOT: v_rcp_f32

; EG-NOT: INT_TO_FLT
; EG-NOT: RECIP_IEEE
define amdgpu_kernel void @test_no_sdiv24_i32_1(ptr addrspace(1) %out, ptr addrspace(1) %in) {
  %den_ptr = getelementptr i32, ptr addrspace(1) %in, i32 1
  %num = load i32, ptr addrspace(1) %in, align 4
  %den = load i32, ptr addrspace(1) %den_ptr, align 4
  %num.i24.0 = shl i32 %num, 8
  %den.i24.0 = shl i32 %den, 7
  %num.i24 = ashr i32 %num.i24.0, 8
  %den.i24 = ashr i32 %den.i24.0, 7
  %result = sdiv i32 %num.i24, %den.i24
  store i32 %result, ptr addrspace(1) %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}test_no_sdiv24_i32_2:
; SI-NOT: v_cvt_f32_i32
; SI-NOT: v_rcp_f32

; EG-NOT: INT_TO_FLT
; EG-NOT: RECIP_IEEE
define amdgpu_kernel void @test_no_sdiv24_i32_2(ptr addrspace(1) %out, ptr addrspace(1) %in) {
  %den_ptr = getelementptr i32, ptr addrspace(1) %in, i32 1
  %num = load i32, ptr addrspace(1) %in, align 4
  %den = load i32, ptr addrspace(1) %den_ptr, align 4
  %num.i24.0 = shl i32 %num, 7
  %den.i24.0 = shl i32 %den, 8
  %num.i24 = ashr i32 %num.i24.0, 7
  %den.i24 = ashr i32 %den.i24.0, 8
  %result = sdiv i32 %num.i24, %den.i24
  store i32 %result, ptr addrspace(1) %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}srem24_i8:
; SI: v_cvt_f32_i32
; SI: v_cvt_f32_i32
; SI: v_rcp_iflag_f32
; SI: v_cvt_i32_f32

; EG: INT_TO_FLT
; EG-DAG: INT_TO_FLT
; EG-DAG: RECIP_IEEE
; EG: FLT_TO_INT
define amdgpu_kernel void @srem24_i8(ptr addrspace(1) %out, ptr addrspace(1) %in) {
  %den_ptr = getelementptr i8, ptr addrspace(1) %in, i8 1
  %num = load i8, ptr addrspace(1) %in
  %den = load i8, ptr addrspace(1) %den_ptr
  %result = srem i8 %num, %den
  store i8 %result, ptr addrspace(1) %out
  ret void
}

; FUNC-LABEL: {{^}}srem24_i16:
; SI: v_cvt_f32_i32
; SI: v_cvt_f32_i32
; SI: v_rcp_iflag_f32
; SI: v_cvt_i32_f32

; EG: INT_TO_FLT
; EG-DAG: INT_TO_FLT
; EG-DAG: RECIP_IEEE
; EG: FLT_TO_INT
define amdgpu_kernel void @srem24_i16(ptr addrspace(1) %out, ptr addrspace(1) %in) {
  %den_ptr = getelementptr i16, ptr addrspace(1) %in, i16 1
  %num = load i16, ptr addrspace(1) %in, align 2
  %den = load i16, ptr addrspace(1) %den_ptr, align 2
  %result = srem i16 %num, %den
  store i16 %result, ptr addrspace(1) %out, align 2
  ret void
}

; FUNC-LABEL: {{^}}srem24_i32:
; SI: v_cvt_f32_i32
; SI: v_cvt_f32_i32
; SI: v_rcp_iflag_f32
; SI: v_cvt_i32_f32

; EG: INT_TO_FLT
; EG-DAG: INT_TO_FLT
; EG-DAG: RECIP_IEEE
; EG: FLT_TO_INT
define amdgpu_kernel void @srem24_i32(ptr addrspace(1) %out, ptr addrspace(1) %in) {
  %den_ptr = getelementptr i32, ptr addrspace(1) %in, i32 1
  %num = load i32, ptr addrspace(1) %in, align 4
  %den = load i32, ptr addrspace(1) %den_ptr, align 4
  %num.i24.0 = shl i32 %num, 8
  %den.i24.0 = shl i32 %den, 8
  %num.i24 = ashr i32 %num.i24.0, 8
  %den.i24 = ashr i32 %den.i24.0, 8
  %result = srem i32 %num.i24, %den.i24
  store i32 %result, ptr addrspace(1) %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}no_srem25_i32:
; SI-NOT: v_cvt_f32_i32
; SI-NOT: v_rcp_f32

; EG-NOT: INT_TO_FLT
; EG-NOT: RECIP_IEEE
define amdgpu_kernel void @no_srem25_i32(ptr addrspace(1) %out, ptr addrspace(1) %in) {
  %den_ptr = getelementptr i32, ptr addrspace(1) %in, i32 1
  %num = load i32, ptr addrspace(1) %in, align 4
  %den = load i32, ptr addrspace(1) %den_ptr, align 4
  %num.i24.0 = shl i32 %num, 7
  %den.i24.0 = shl i32 %den, 7
  %num.i24 = ashr i32 %num.i24.0, 7
  %den.i24 = ashr i32 %den.i24.0, 7
  %result = srem i32 %num.i24, %den.i24
  store i32 %result, ptr addrspace(1) %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}no_sdiv25_i24_i25_i32:
; SI-NOT: v_cvt_f32_i32
; SI-NOT: v_rcp_f32

; EG-NOT: INT_TO_FLT
; EG-NOT: RECIP_IEEE
define amdgpu_kernel void @no_sdiv25_i24_i25_i32(ptr addrspace(1) %out, ptr addrspace(1) %in) {
  %den_ptr = getelementptr i32, ptr addrspace(1) %in, i32 1
  %num = load i32, ptr addrspace(1) %in, align 4
  %den = load i32, ptr addrspace(1) %den_ptr, align 4
  %num.i24.0 = shl i32 %num, 8
  %den.i25.0 = shl i32 %den, 7
  %num.i24 = ashr i32 %num.i24.0, 8
  %den.i25 = ashr i32 %den.i25.0, 7
  %result = sdiv i32 %num.i24, %den.i25
  store i32 %result, ptr addrspace(1) %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}no_sdiv25_i25_i24_i32:
; SI-NOT: v_cvt_f32_i32
; SI-NOT: v_rcp_f32

; EG-NOT: INT_TO_FLT
; EG-NOT: RECIP_IEEE
define amdgpu_kernel void @no_sdiv25_i25_i24_i32(ptr addrspace(1) %out, ptr addrspace(1) %in) {
  %den_ptr = getelementptr i32, ptr addrspace(1) %in, i32 1
  %num = load i32, ptr addrspace(1) %in, align 4
  %den = load i32, ptr addrspace(1) %den_ptr, align 4
  %num.i25.0 = shl i32 %num, 7
  %den.i24.0 = shl i32 %den, 8
  %num.i25 = ashr i32 %num.i25.0, 7
  %den.i24 = ashr i32 %den.i24.0, 8
  %result = sdiv i32 %num.i25, %den.i24
  store i32 %result, ptr addrspace(1) %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}no_srem25_i24_i25_i32:
; SI-NOT: v_cvt_f32_i32
; SI-NOT: v_rcp_f32

; EG-NOT: INT_TO_FLT
; EG-NOT: RECIP_IEEE
define amdgpu_kernel void @no_srem25_i24_i25_i32(ptr addrspace(1) %out, ptr addrspace(1) %in) {
  %den_ptr = getelementptr i32, ptr addrspace(1) %in, i32 1
  %num = load i32, ptr addrspace(1) %in, align 4
  %den = load i32, ptr addrspace(1) %den_ptr, align 4
  %num.i24.0 = shl i32 %num, 8
  %den.i25.0 = shl i32 %den, 7
  %num.i24 = ashr i32 %num.i24.0, 8
  %den.i25 = ashr i32 %den.i25.0, 7
  %result = srem i32 %num.i24, %den.i25
  store i32 %result, ptr addrspace(1) %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}no_srem25_i25_i24_i32:
; SI-NOT: v_cvt_f32_i32
; SI-NOT: v_rcp_f32

; EG-NOT: INT_TO_FLT
; EG-NOT: RECIP_IEEE
define amdgpu_kernel void @no_srem25_i25_i24_i32(ptr addrspace(1) %out, ptr addrspace(1) %in) {
  %den_ptr = getelementptr i32, ptr addrspace(1) %in, i32 1
  %num = load i32, ptr addrspace(1) %in, align 4
  %den = load i32, ptr addrspace(1) %den_ptr, align 4
  %num.i25.0 = shl i32 %num, 7
  %den.i24.0 = shl i32 %den, 8
  %num.i25 = ashr i32 %num.i25.0, 7
  %den.i24 = ashr i32 %den.i24.0, 8
  %result = srem i32 %num.i25, %den.i24
  store i32 %result, ptr addrspace(1) %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}srem25_i24_i11_i32:
; SI: v_cvt_f32_i32
; SI: v_rcp_iflag_f32
; SI: v_bfe_i32 v{{[0-9]+}}, v{{[0-9]+}}, 0, 24

; EG: INT_TO_FLT
; EG: RECIP_IEEE
define amdgpu_kernel void @srem25_i24_i11_i32(ptr addrspace(1) %out, ptr addrspace(1) %in) {
  %den_ptr = getelementptr i32, ptr addrspace(1) %in, i32 1
  %num = load i32, ptr addrspace(1) %in, align 4
  %den = load i32, ptr addrspace(1) %den_ptr, align 4
  %num.i24.0 = shl i32 %num, 8
  %den.i11.0 = shl i32 %den, 21
  %num.i24 = ashr i32 %num.i24.0, 8
  %den.i11 = ashr i32 %den.i11.0, 21
  %result = srem i32 %num.i24, %den.i11
  store i32 %result, ptr addrspace(1) %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}srem25_i11_i24_i32:
; SI: v_cvt_f32_i32
; SI: v_rcp_iflag_f32
; SI: v_bfe_i32 v{{[0-9]+}}, v{{[0-9]+}}, 0, 24

; EG: INT_TO_FLT
; EG: RECIP_IEEE
define amdgpu_kernel void @srem25_i11_i24_i32(ptr addrspace(1) %out, ptr addrspace(1) %in) {
  %den_ptr = getelementptr i32, ptr addrspace(1) %in, i32 1
  %num = load i32, ptr addrspace(1) %in, align 4
  %den = load i32, ptr addrspace(1) %den_ptr, align 4
  %num.i11.0 = shl i32 %num, 21
  %den.i24.0 = shl i32 %den, 8
  %num.i11 = ashr i32 %num.i11.0, 21
  %den.i24 = ashr i32 %den.i24.0, 8
  %result = srem i32 %num.i11, %den.i24
  store i32 %result, ptr addrspace(1) %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}srem25_i17_i12_i32:
; SI: v_cvt_f32_i32
; SI: v_rcp_iflag_f32
; SI: v_bfe_i32 v{{[0-9]+}}, v{{[0-9]+}}, 0, 17

; EG: INT_TO_FLT
; EG: RECIP_IEEE
define amdgpu_kernel void @srem25_i17_i12_i32(ptr addrspace(1) %out, ptr addrspace(1) %in) {
  %den_ptr = getelementptr i32, ptr addrspace(1) %in, i32 1
  %num = load i32, ptr addrspace(1) %in, align 4
  %den = load i32, ptr addrspace(1) %den_ptr, align 4
  %num.i17.0 = shl i32 %num, 15
  %den.i12.0 = shl i32 %den, 20
  %num.i17 = ashr i32 %num.i17.0, 15
  %den.i12 = ashr i32 %den.i12.0, 20
  %result = sdiv i32 %num.i17, %den.i12
  store i32 %result, ptr addrspace(1) %out, align 4
  ret void
}
