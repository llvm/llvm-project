; RUN: llc -march=amdgcn -mcpu=SI < %s | FileCheck -check-prefix=SI -check-prefix=FUNC %s
; RUN: llc -march=amdgcn -mcpu=tonga < %s | FileCheck -check-prefix=SI -check-prefix=FUNC %s
; RUN: llc -march=r600 -mcpu=redwood < %s | FileCheck -check-prefix=EG -check-prefix=FUNC %s

; FUNC-LABEL: {{^}}sdiv24_i8:
; SI: v_cvt_f32_i32
; SI: v_cvt_f32_i32
; SI: v_rcp_f32
; SI: v_cvt_i32_f32

; EG: INT_TO_FLT
; EG-DAG: INT_TO_FLT
; EG-DAG: RECIP_IEEE
; EG: FLT_TO_INT
define void @sdiv24_i8(i8 addrspace(1)* %out, i8 addrspace(1)* %in) {
  %den_ptr = getelementptr i8 addrspace(1)* %in, i8 1
  %num = load i8 addrspace(1) * %in
  %den = load i8 addrspace(1) * %den_ptr
  %result = sdiv i8 %num, %den
  store i8 %result, i8 addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}sdiv24_i16:
; SI: v_cvt_f32_i32
; SI: v_cvt_f32_i32
; SI: v_rcp_f32
; SI: v_cvt_i32_f32

; EG: INT_TO_FLT
; EG-DAG: INT_TO_FLT
; EG-DAG: RECIP_IEEE
; EG: FLT_TO_INT
define void @sdiv24_i16(i16 addrspace(1)* %out, i16 addrspace(1)* %in) {
  %den_ptr = getelementptr i16 addrspace(1)* %in, i16 1
  %num = load i16 addrspace(1) * %in, align 2
  %den = load i16 addrspace(1) * %den_ptr, align 2
  %result = sdiv i16 %num, %den
  store i16 %result, i16 addrspace(1)* %out, align 2
  ret void
}

; FUNC-LABEL: {{^}}sdiv24_i32:
; SI: v_cvt_f32_i32
; SI: v_cvt_f32_i32
; SI: v_rcp_f32
; SI: v_cvt_i32_f32

; EG: INT_TO_FLT
; EG-DAG: INT_TO_FLT
; EG-DAG: RECIP_IEEE
; EG: FLT_TO_INT
define void @sdiv24_i32(i32 addrspace(1)* %out, i32 addrspace(1)* %in) {
  %den_ptr = getelementptr i32 addrspace(1)* %in, i32 1
  %num = load i32 addrspace(1) * %in, align 4
  %den = load i32 addrspace(1) * %den_ptr, align 4
  %num.i24.0 = shl i32 %num, 8
  %den.i24.0 = shl i32 %den, 8
  %num.i24 = ashr i32 %num.i24.0, 8
  %den.i24 = ashr i32 %den.i24.0, 8
  %result = sdiv i32 %num.i24, %den.i24
  store i32 %result, i32 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}sdiv25_i32:
; SI-NOT: v_cvt_f32_i32
; SI-NOT: v_rcp_f32

; EG-NOT: INT_TO_FLT
; EG-NOT: RECIP_IEEE
define void @sdiv25_i32(i32 addrspace(1)* %out, i32 addrspace(1)* %in) {
  %den_ptr = getelementptr i32 addrspace(1)* %in, i32 1
  %num = load i32 addrspace(1) * %in, align 4
  %den = load i32 addrspace(1) * %den_ptr, align 4
  %num.i24.0 = shl i32 %num, 7
  %den.i24.0 = shl i32 %den, 7
  %num.i24 = ashr i32 %num.i24.0, 7
  %den.i24 = ashr i32 %den.i24.0, 7
  %result = sdiv i32 %num.i24, %den.i24
  store i32 %result, i32 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}test_no_sdiv24_i32_1:
; SI-NOT: v_cvt_f32_i32
; SI-NOT: v_rcp_f32

; EG-NOT: INT_TO_FLT
; EG-NOT: RECIP_IEEE
define void @test_no_sdiv24_i32_1(i32 addrspace(1)* %out, i32 addrspace(1)* %in) {
  %den_ptr = getelementptr i32 addrspace(1)* %in, i32 1
  %num = load i32 addrspace(1) * %in, align 4
  %den = load i32 addrspace(1) * %den_ptr, align 4
  %num.i24.0 = shl i32 %num, 8
  %den.i24.0 = shl i32 %den, 7
  %num.i24 = ashr i32 %num.i24.0, 8
  %den.i24 = ashr i32 %den.i24.0, 7
  %result = sdiv i32 %num.i24, %den.i24
  store i32 %result, i32 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}test_no_sdiv24_i32_2:
; SI-NOT: v_cvt_f32_i32
; SI-NOT: v_rcp_f32

; EG-NOT: INT_TO_FLT
; EG-NOT: RECIP_IEEE
define void @test_no_sdiv24_i32_2(i32 addrspace(1)* %out, i32 addrspace(1)* %in) {
  %den_ptr = getelementptr i32 addrspace(1)* %in, i32 1
  %num = load i32 addrspace(1) * %in, align 4
  %den = load i32 addrspace(1) * %den_ptr, align 4
  %num.i24.0 = shl i32 %num, 7
  %den.i24.0 = shl i32 %den, 8
  %num.i24 = ashr i32 %num.i24.0, 7
  %den.i24 = ashr i32 %den.i24.0, 8
  %result = sdiv i32 %num.i24, %den.i24
  store i32 %result, i32 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}srem24_i8:
; SI: v_cvt_f32_i32
; SI: v_cvt_f32_i32
; SI: v_rcp_f32
; SI: v_cvt_i32_f32

; EG: INT_TO_FLT
; EG-DAG: INT_TO_FLT
; EG-DAG: RECIP_IEEE
; EG: FLT_TO_INT
define void @srem24_i8(i8 addrspace(1)* %out, i8 addrspace(1)* %in) {
  %den_ptr = getelementptr i8 addrspace(1)* %in, i8 1
  %num = load i8 addrspace(1) * %in
  %den = load i8 addrspace(1) * %den_ptr
  %result = srem i8 %num, %den
  store i8 %result, i8 addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}srem24_i16:
; SI: v_cvt_f32_i32
; SI: v_cvt_f32_i32
; SI: v_rcp_f32
; SI: v_cvt_i32_f32

; EG: INT_TO_FLT
; EG-DAG: INT_TO_FLT
; EG-DAG: RECIP_IEEE
; EG: FLT_TO_INT
define void @srem24_i16(i16 addrspace(1)* %out, i16 addrspace(1)* %in) {
  %den_ptr = getelementptr i16 addrspace(1)* %in, i16 1
  %num = load i16 addrspace(1) * %in, align 2
  %den = load i16 addrspace(1) * %den_ptr, align 2
  %result = srem i16 %num, %den
  store i16 %result, i16 addrspace(1)* %out, align 2
  ret void
}

; FUNC-LABEL: {{^}}srem24_i32:
; SI: v_cvt_f32_i32
; SI: v_cvt_f32_i32
; SI: v_rcp_f32
; SI: v_cvt_i32_f32

; EG: INT_TO_FLT
; EG-DAG: INT_TO_FLT
; EG-DAG: RECIP_IEEE
; EG: FLT_TO_INT
define void @srem24_i32(i32 addrspace(1)* %out, i32 addrspace(1)* %in) {
  %den_ptr = getelementptr i32 addrspace(1)* %in, i32 1
  %num = load i32 addrspace(1) * %in, align 4
  %den = load i32 addrspace(1) * %den_ptr, align 4
  %num.i24.0 = shl i32 %num, 8
  %den.i24.0 = shl i32 %den, 8
  %num.i24 = ashr i32 %num.i24.0, 8
  %den.i24 = ashr i32 %den.i24.0, 8
  %result = srem i32 %num.i24, %den.i24
  store i32 %result, i32 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}srem25_i32:
; SI-NOT: v_cvt_f32_i32
; SI-NOT: v_rcp_f32

; EG-NOT: INT_TO_FLT
; EG-NOT: RECIP_IEEE
define void @srem25_i32(i32 addrspace(1)* %out, i32 addrspace(1)* %in) {
  %den_ptr = getelementptr i32 addrspace(1)* %in, i32 1
  %num = load i32 addrspace(1) * %in, align 4
  %den = load i32 addrspace(1) * %den_ptr, align 4
  %num.i24.0 = shl i32 %num, 7
  %den.i24.0 = shl i32 %den, 7
  %num.i24 = ashr i32 %num.i24.0, 7
  %den.i24 = ashr i32 %den.i24.0, 7
  %result = srem i32 %num.i24, %den.i24
  store i32 %result, i32 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}test_no_srem24_i32_1:
; SI-NOT: v_cvt_f32_i32
; SI-NOT: v_rcp_f32

; EG-NOT: INT_TO_FLT
; EG-NOT: RECIP_IEEE
define void @test_no_srem24_i32_1(i32 addrspace(1)* %out, i32 addrspace(1)* %in) {
  %den_ptr = getelementptr i32 addrspace(1)* %in, i32 1
  %num = load i32 addrspace(1) * %in, align 4
  %den = load i32 addrspace(1) * %den_ptr, align 4
  %num.i24.0 = shl i32 %num, 8
  %den.i24.0 = shl i32 %den, 7
  %num.i24 = ashr i32 %num.i24.0, 8
  %den.i24 = ashr i32 %den.i24.0, 7
  %result = srem i32 %num.i24, %den.i24
  store i32 %result, i32 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}test_no_srem24_i32_2:
; SI-NOT: v_cvt_f32_i32
; SI-NOT: v_rcp_f32

; EG-NOT: INT_TO_FLT
; EG-NOT: RECIP_IEEE
define void @test_no_srem24_i32_2(i32 addrspace(1)* %out, i32 addrspace(1)* %in) {
  %den_ptr = getelementptr i32 addrspace(1)* %in, i32 1
  %num = load i32 addrspace(1) * %in, align 4
  %den = load i32 addrspace(1) * %den_ptr, align 4
  %num.i24.0 = shl i32 %num, 7
  %den.i24.0 = shl i32 %den, 8
  %num.i24 = ashr i32 %num.i24.0, 7
  %den.i24 = ashr i32 %den.i24.0, 8
  %result = srem i32 %num.i24, %den.i24
  store i32 %result, i32 addrspace(1)* %out, align 4
  ret void
}
