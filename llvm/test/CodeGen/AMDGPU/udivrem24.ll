; RUN: llc -march=amdgcn -verify-machineinstrs < %s | FileCheck -check-prefix=SI -check-prefix=FUNC %s
; RUN: llc -march=amdgcn -mcpu=tonga -mattr=-flat-for-global -verify-machineinstrs < %s | FileCheck -check-prefix=SI -check-prefix=FUNC %s
; RUN: llc -march=r600 -mcpu=redwood < %s | FileCheck -check-prefix=EG -check-prefix=FUNC %s

; FUNC-LABEL: {{^}}udiv24_i8:
; SI: v_cvt_f32_ubyte
; SI-DAG: v_cvt_f32_ubyte
; SI-DAG: v_rcp_iflag_f32
; SI: v_cvt_u32_f32

; EG: UINT_TO_FLT
; EG-DAG: UINT_TO_FLT
; EG-DAG: RECIP_IEEE
; EG: FLT_TO_UINT
define amdgpu_kernel void @udiv24_i8(ptr addrspace(1) %out, ptr addrspace(1) %in) {
  %den_ptr = getelementptr i8, ptr addrspace(1) %in, i8 1
  %num = load i8, ptr addrspace(1) %in
  %den = load i8, ptr addrspace(1) %den_ptr
  %result = udiv i8 %num, %den
  store i8 %result, ptr addrspace(1) %out
  ret void
}

; FUNC-LABEL: {{^}}udiv24_i8_denorm_flush_in_out:
; SI: v_cvt_f32_ubyte
; SI-DAG: v_cvt_f32_ubyte
; SI-DAG: v_rcp_iflag_f32
; SI: v_cvt_u32_f32

; EG: UINT_TO_FLT
; EG-DAG: UINT_TO_FLT
; EG-DAG: RECIP_IEEE
; EG: FLT_TO_UINT
define amdgpu_kernel void @udiv24_i8_denorm_flush_in_out(ptr addrspace(1) %out, ptr addrspace(1) %in) #0 {
  %den_ptr = getelementptr i8, ptr addrspace(1) %in, i8 1
  %num = load i8, ptr addrspace(1) %in
  %den = load i8, ptr addrspace(1) %den_ptr
  %result = udiv i8 %num, %den
  store i8 %result, ptr addrspace(1) %out
  ret void
}

; FUNC-LABEL: {{^}}udiv24_i8_denorm_flush_in:
; SI: v_cvt_f32_ubyte
; SI-DAG: v_cvt_f32_ubyte
; SI-DAG: v_rcp_iflag_f32
; SI: v_cvt_u32_f32

; EG: UINT_TO_FLT
; EG-DAG: UINT_TO_FLT
; EG-DAG: RECIP_IEEE
; EG: FLT_TO_UINT
define amdgpu_kernel void @udiv24_i8_denorm_flush_in(ptr addrspace(1) %out, ptr addrspace(1) %in) #1 {
  %den_ptr = getelementptr i8, ptr addrspace(1) %in, i8 1
  %num = load i8, ptr addrspace(1) %in
  %den = load i8, ptr addrspace(1) %den_ptr
  %result = udiv i8 %num, %den
  store i8 %result, ptr addrspace(1) %out
  ret void
}

; FUNC-LABEL: {{^}}udiv24_i8_denorm_flush_out:
; SI: v_cvt_f32_ubyte
; SI-DAG: v_cvt_f32_ubyte
; SI-DAG: v_rcp_iflag_f32
; SI: v_cvt_u32_f32

; EG: UINT_TO_FLT
; EG-DAG: UINT_TO_FLT
; EG-DAG: RECIP_IEEE
; EG: FLT_TO_UINT
define amdgpu_kernel void @udiv24_i8_denorm_flush_out(ptr addrspace(1) %out, ptr addrspace(1) %in) #2 {
  %den_ptr = getelementptr i8, ptr addrspace(1) %in, i8 1
  %num = load i8, ptr addrspace(1) %in
  %den = load i8, ptr addrspace(1) %den_ptr
  %result = udiv i8 %num, %den
  store i8 %result, ptr addrspace(1) %out
  ret void
}

; FUNC-LABEL: {{^}}udiv24_i16:
; SI: v_cvt_f32_u32
; SI: v_cvt_f32_u32
; SI: v_rcp_iflag_f32
; SI: v_cvt_u32_f32

; EG: UINT_TO_FLT
; EG-DAG: UINT_TO_FLT
; EG-DAG: RECIP_IEEE
; EG: FLT_TO_UINT
define amdgpu_kernel void @udiv24_i16(ptr addrspace(1) %out, ptr addrspace(1) %in) {
  %den_ptr = getelementptr i16, ptr addrspace(1) %in, i16 1
  %num = load i16, ptr addrspace(1) %in, align 2
  %den = load i16, ptr addrspace(1) %den_ptr, align 2
  %result = udiv i16 %num, %den
  store i16 %result, ptr addrspace(1) %out, align 2
  ret void
}

; FUNC-LABEL: {{^}}udiv23_i32:
; SI: v_cvt_f32_u32
; SI-DAG: v_cvt_f32_u32
; SI-DAG: v_rcp_iflag_f32
; SI: v_cvt_u32_f32

; EG: UINT_TO_FLT
; EG-DAG: UINT_TO_FLT
; EG-DAG: RECIP_IEEE
; EG: FLT_TO_UINT
define amdgpu_kernel void @udiv23_i32(ptr addrspace(1) %out, ptr addrspace(1) %in) {
  %den_ptr = getelementptr i32, ptr addrspace(1) %in, i32 1
  %num = load i32, ptr addrspace(1) %in, align 4
  %den = load i32, ptr addrspace(1) %den_ptr, align 4
  %num.i23.0 = shl i32 %num, 9
  %den.i23.0 = shl i32 %den, 9
  %num.i23 = lshr i32 %num.i23.0, 9
  %den.i23 = lshr i32 %den.i23.0, 9
  %result = udiv i32 %num.i23, %den.i23
  store i32 %result, ptr addrspace(1) %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}udiv24_i32:
; SI: v_rcp_iflag
; SI-NOT: v_rcp_f32
; EG-NOT: RECIP_IEEE
define amdgpu_kernel void @udiv24_i32(ptr addrspace(1) %out, ptr addrspace(1) %in) {
  %den_ptr = getelementptr i32, ptr addrspace(1) %in, i32 1
  %num = load i32, ptr addrspace(1) %in, align 4
  %den = load i32, ptr addrspace(1) %den_ptr, align 4
  %num.i24.0 = shl i32 %num, 8
  %den.i24.0 = shl i32 %den, 8
  %num.i24 = lshr i32 %num.i24.0, 8
  %den.i24 = lshr i32 %den.i24.0, 8
  %result = udiv i32 %num.i24, %den.i24
  store i32 %result, ptr addrspace(1) %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}no_udiv24_u23_u24_i32:
; SI: v_rcp_iflag
; SI-NOT: v_rcp_f32
; EG-NOT: RECIP_IEEE
define amdgpu_kernel void @no_udiv24_u23_u24_i32(ptr addrspace(1) %out, ptr addrspace(1) %in) {
  %den_ptr = getelementptr i32, ptr addrspace(1) %in, i32 1
  %num = load i32, ptr addrspace(1) %in, align 4
  %den = load i32, ptr addrspace(1) %den_ptr, align 4
  %num.i23.0 = shl i32 %num, 9
  %den.i24.0 = shl i32 %den, 8
  %num.i23 = lshr i32 %num.i23.0, 9
  %den.i24 = lshr i32 %den.i24.0, 8
  %result = udiv i32 %num.i23, %den.i24
  store i32 %result, ptr addrspace(1) %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}no_udiv24_u24_u23_i32:
; SI: v_rcp_iflag
; SI-NOT: v_rcp_f32
; EG-NOT: RECIP_IEEE
define amdgpu_kernel void @no_udiv24_u24_u23_i32(ptr addrspace(1) %out, ptr addrspace(1) %in) {
  %den_ptr = getelementptr i32, ptr addrspace(1) %in, i32 1
  %num = load i32, ptr addrspace(1) %in, align 4
  %den = load i32, ptr addrspace(1) %den_ptr, align 4
  %num.i24.0 = shl i32 %num, 8
  %den.i23.0 = shl i32 %den, 9
  %num.i24 = lshr i32 %num.i24.0, 8
  %den.i23 = lshr i32 %den.i23.0, 9
  %result = udiv i32 %num.i24, %den.i23
  store i32 %result, ptr addrspace(1) %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}udiv25_i32:
; RCP_IFLAG is for URECIP in the full 32b alg
; SI: v_rcp_iflag
; SI-NOT: v_rcp_f32

; EG-NOT: UINT_TO_FLT
; EG-NOT: RECIP_IEEE
define amdgpu_kernel void @udiv25_i32(ptr addrspace(1) %out, ptr addrspace(1) %in) {
  %den_ptr = getelementptr i32, ptr addrspace(1) %in, i32 1
  %num = load i32, ptr addrspace(1) %in, align 4
  %den = load i32, ptr addrspace(1) %den_ptr, align 4
  %num.i25.0 = shl i32 %num, 7
  %den.i25.0 = shl i32 %den, 7
  %num.i25 = lshr i32 %num.i25.0, 7
  %den.i25 = lshr i32 %den.i25.0, 7
  %result = udiv i32 %num.i25, %den.i25
  store i32 %result, ptr addrspace(1) %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}test_no_udiv24_i32_1:
; RCP_IFLAG is for URECIP in the full 32b alg
; SI: v_rcp_iflag
; SI-NOT: v_rcp_f32

; EG-NOT: UINT_TO_FLT
; EG-NOT: RECIP_IEEE
define amdgpu_kernel void @test_no_udiv24_i32_1(ptr addrspace(1) %out, ptr addrspace(1) %in) {
  %den_ptr = getelementptr i32, ptr addrspace(1) %in, i32 1
  %num = load i32, ptr addrspace(1) %in, align 4
  %den = load i32, ptr addrspace(1) %den_ptr, align 4
  %num.i24.0 = shl i32 %num, 8
  %den.i24.0 = shl i32 %den, 7
  %num.i24 = lshr i32 %num.i24.0, 8
  %den.i24 = lshr i32 %den.i24.0, 7
  %result = udiv i32 %num.i24, %den.i24
  store i32 %result, ptr addrspace(1) %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}test_no_udiv24_i32_2:
; RCP_IFLAG is for URECIP in the full 32b alg
; SI: v_rcp_iflag
; SI-NOT: v_rcp_f32

; EG-NOT: UINT_TO_FLT
; EG-NOT: RECIP_IEEE
define amdgpu_kernel void @test_no_udiv24_i32_2(ptr addrspace(1) %out, ptr addrspace(1) %in) {
  %den_ptr = getelementptr i32, ptr addrspace(1) %in, i32 1
  %num = load i32, ptr addrspace(1) %in, align 4
  %den = load i32, ptr addrspace(1) %den_ptr, align 4
  %num.i24.0 = shl i32 %num, 7
  %den.i24.0 = shl i32 %den, 8
  %num.i24 = lshr i32 %num.i24.0, 7
  %den.i24 = lshr i32 %den.i24.0, 8
  %result = udiv i32 %num.i24, %den.i24
  store i32 %result, ptr addrspace(1) %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}urem24_i8:
; SI: v_cvt_f32_ubyte
; SI-DAG: v_cvt_f32_ubyte
; SI-DAG: v_rcp_iflag_f32
; SI: v_cvt_u32_f32

; EG: UINT_TO_FLT
; EG-DAG: UINT_TO_FLT
; EG-DAG: RECIP_IEEE
; EG: FLT_TO_UINT
define amdgpu_kernel void @urem24_i8(ptr addrspace(1) %out, ptr addrspace(1) %in) {
  %den_ptr = getelementptr i8, ptr addrspace(1) %in, i8 1
  %num = load i8, ptr addrspace(1) %in
  %den = load i8, ptr addrspace(1) %den_ptr
  %result = urem i8 %num, %den
  store i8 %result, ptr addrspace(1) %out
  ret void
}

; FUNC-LABEL: {{^}}urem24_i16:
; SI: v_cvt_f32_u32
; SI: v_cvt_f32_u32
; SI: v_rcp_iflag_f32
; SI: v_cvt_u32_f32

; EG: UINT_TO_FLT
; EG-DAG: UINT_TO_FLT
; EG-DAG: RECIP_IEEE
; EG: FLT_TO_UINT
define amdgpu_kernel void @urem24_i16(ptr addrspace(1) %out, ptr addrspace(1) %in) {
  %den_ptr = getelementptr i16, ptr addrspace(1) %in, i16 1
  %num = load i16, ptr addrspace(1) %in, align 2
  %den = load i16, ptr addrspace(1) %den_ptr, align 2
  %result = urem i16 %num, %den
  store i16 %result, ptr addrspace(1) %out, align 2
  ret void
}

; FUNC-LABEL: {{^}}urem24_i32:
; SI-NOT: v_rcp_f32
; EG-NOT: RECIP_IEEE
define amdgpu_kernel void @urem24_i32(ptr addrspace(1) %out, ptr addrspace(1) %in) {
  %den_ptr = getelementptr i32, ptr addrspace(1) %in, i32 1
  %num = load i32, ptr addrspace(1) %in, align 4
  %den = load i32, ptr addrspace(1) %den_ptr, align 4
  %num.i24.0 = shl i32 %num, 8
  %den.i24.0 = shl i32 %den, 8
  %num.i24 = lshr i32 %num.i24.0, 8
  %den.i24 = lshr i32 %den.i24.0, 8
  %result = urem i32 %num.i24, %den.i24
  store i32 %result, ptr addrspace(1) %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}urem25_i32:
; RCP_IFLAG is for URECIP in the full 32b alg
; SI: v_rcp_iflag
; SI-NOT: v_rcp_f32

; EG-NOT: UINT_TO_FLT
; EG-NOT: RECIP_IEEE
define amdgpu_kernel void @urem25_i32(ptr addrspace(1) %out, ptr addrspace(1) %in) {
  %den_ptr = getelementptr i32, ptr addrspace(1) %in, i32 1
  %num = load i32, ptr addrspace(1) %in, align 4
  %den = load i32, ptr addrspace(1) %den_ptr, align 4
  %num.i24.0 = shl i32 %num, 7
  %den.i24.0 = shl i32 %den, 7
  %num.i24 = lshr i32 %num.i24.0, 7
  %den.i24 = lshr i32 %den.i24.0, 7
  %result = urem i32 %num.i24, %den.i24
  store i32 %result, ptr addrspace(1) %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}test_no_urem24_i32_1:
; RCP_IFLAG is for URECIP in the full 32b alg
; SI: v_rcp_iflag
; SI-NOT: v_rcp_f32

; EG-NOT: UINT_TO_FLT
; EG-NOT: RECIP_IEEE
define amdgpu_kernel void @test_no_urem24_i32_1(ptr addrspace(1) %out, ptr addrspace(1) %in) {
  %den_ptr = getelementptr i32, ptr addrspace(1) %in, i32 1
  %num = load i32, ptr addrspace(1) %in, align 4
  %den = load i32, ptr addrspace(1) %den_ptr, align 4
  %num.i24.0 = shl i32 %num, 8
  %den.i24.0 = shl i32 %den, 7
  %num.i24 = lshr i32 %num.i24.0, 8
  %den.i24 = lshr i32 %den.i24.0, 7
  %result = urem i32 %num.i24, %den.i24
  store i32 %result, ptr addrspace(1) %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}test_no_urem24_i32_2:
; RCP_IFLAG is for URECIP in the full 32b alg
; SI: v_rcp_iflag
; SI-NOT: v_rcp_f32

; EG-NOT: UINT_TO_FLT
; EG-NOT: RECIP_IEEE
define amdgpu_kernel void @test_no_urem24_i32_2(ptr addrspace(1) %out, ptr addrspace(1) %in) {
  %den_ptr = getelementptr i32, ptr addrspace(1) %in, i32 1
  %num = load i32, ptr addrspace(1) %in, align 4
  %den = load i32, ptr addrspace(1) %den_ptr, align 4
  %num.i24.0 = shl i32 %num, 7
  %den.i24.0 = shl i32 %den, 8
  %num.i24 = lshr i32 %num.i24.0, 7
  %den.i24 = lshr i32 %den.i24.0, 8
  %result = urem i32 %num.i24, %den.i24
  store i32 %result, ptr addrspace(1) %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}test_udiv24_u16_u23_i32:
; SI: v_rcp_iflag_f32
; SI: v_and_b32_e32 v{{[0-9]+}}, 0x7fffff,

; EG: RECIP_IEEE
define amdgpu_kernel void @test_udiv24_u16_u23_i32(ptr addrspace(1) %out, ptr addrspace(1) %in) {
  %den_ptr = getelementptr i32, ptr addrspace(1) %in, i32 1
  %num = load i32, ptr addrspace(1) %in, align 4
  %den = load i32, ptr addrspace(1) %den_ptr, align 4
  %num.i16.0 = shl i32 %num, 16
  %den.i23.0 = shl i32 %den, 9
  %num.i16 = lshr i32 %num.i16.0, 16
  %den.i23 = lshr i32 %den.i23.0, 9
  %result = udiv i32 %num.i16, %den.i23
  store i32 %result, ptr addrspace(1) %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}test_udiv24_u23_u16_i32:
; SI: v_rcp_iflag_f32
; SI: v_and_b32_e32 v{{[0-9]+}}, 0x7fffff,

; EG: RECIP_IEEE
define amdgpu_kernel void @test_udiv24_u23_u16_i32(ptr addrspace(1) %out, ptr addrspace(1) %in) {
  %den_ptr = getelementptr i32, ptr addrspace(1) %in, i32 1
  %num = load i32, ptr addrspace(1) %in, align 4
  %den = load i32, ptr addrspace(1) %den_ptr, align 4
  %num.i23.0 = shl i32 %num, 9
  %den.i16.0 = shl i32 %den, 16
  %num.i23 = lshr i32 %num.i23.0, 9
  %den.i16 = lshr i32 %den.i16.0, 16
  %result = udiv i32 %num.i23, %den.i16
  store i32 %result, ptr addrspace(1) %out, align 4
  ret void
}

attributes #0 = { "denormal-fp-math-f32"="preserve-sign,preserve-sign" }
attributes #1 = { "denormal-fp-math-f32"="ieee,preserve-sign" }
attributes #2 = { "denormal-fp-math-f32"="preserve-sign,ieee" }
