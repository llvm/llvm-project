; RUN: llc -march=r600 -mcpu=cypress < %s | FileCheck -check-prefix=EG -check-prefix=FUNC %s

; This test just checks that the compiler doesn't crash.


; FUNC-LABEL: {{^}}i8ptr_v16i8ptr:
; EG: MEM_RAT_CACHELESS STORE_RAW [[DATA:T[0-9]+\.XYZW]], [[ST_PTR:T[0-9]+\.[XYZW]]]
; EG: VTX_READ_128 [[DATA]], [[LD_PTR:T[0-9]+\.[XYZW]]]
; EG-DAG: MOV {{[\* ]*}}[[LD_PTR]], KC0[2].Z
; EG-DAG: LSHR {{[\* ]*}}[[ST_PTR]], KC0[2].Y, literal
define void @i8ptr_v16i8ptr(<16 x i8> addrspace(1)* %out, i8 addrspace(1)* %in) {
entry:
  %0 = bitcast i8 addrspace(1)* %in to <16 x i8> addrspace(1)*
  %1 = load <16 x i8>, <16 x i8> addrspace(1)* %0
  store <16 x i8> %1, <16 x i8> addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}f32_to_v2i16:
; EG: MEM_RAT_CACHELESS STORE_RAW [[DATA:T[0-9]+\.[XYZW]]], [[ST_PTR:T[0-9]+\.[XYZW]]]
; EG: VTX_READ_32 [[DATA]], [[LD_PTR:T[0-9]+\.[XYZW]]]
; EG-DAG: MOV {{[\* ]*}}[[LD_PTR]], KC0[2].Z
; EG-DAG: LSHR {{[\* ]*}}[[ST_PTR]], KC0[2].Y, literal
define void @f32_to_v2i16(<2 x i16> addrspace(1)* %out, float addrspace(1)* %in) nounwind {
  %load = load float, float addrspace(1)* %in, align 4
  %bc = bitcast float %load to <2 x i16>
  store <2 x i16> %bc, <2 x i16> addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}v2i16_to_f32:
; EG: MEM_RAT_CACHELESS STORE_RAW [[DATA:T[0-9]+\.[XYZW]]], [[ST_PTR:T[0-9]+\.[XYZW]]]
; EG: VTX_READ_32 [[DATA]], [[LD_PTR:T[0-9]+\.[XYZW]]]
; EG-DAG: MOV {{[\* ]*}}[[LD_PTR]], KC0[2].Z
; EG-DAG: LSHR {{[\* ]*}}[[ST_PTR]], KC0[2].Y, literal
define void @v2i16_to_f32(float addrspace(1)* %out, <2 x i16> addrspace(1)* %in) nounwind {
  %load = load <2 x i16>, <2 x i16> addrspace(1)* %in, align 4
  %bc = bitcast <2 x i16> %load to float
  store float %bc, float addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}v4i8_to_i32:
; EG: MEM_RAT_CACHELESS STORE_RAW [[DATA:T[0-9]+\.[XYZW]]], [[ST_PTR:T[0-9]+\.[XYZW]]]
; EG: VTX_READ_32 [[DATA]], [[LD_PTR:T[0-9]+\.[XYZW]]]
; EG-DAG: MOV {{[\* ]*}}[[LD_PTR]], KC0[2].Z
; EG-DAG: LSHR {{[\* ]*}}[[ST_PTR]], KC0[2].Y, literal
define void @v4i8_to_i32(i32 addrspace(1)* %out, <4 x i8> addrspace(1)* %in) nounwind {
  %load = load <4 x i8>, <4 x i8> addrspace(1)* %in, align 4
  %bc = bitcast <4 x i8> %load to i32
  store i32 %bc, i32 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}i32_to_v4i8:
; EG: MEM_RAT_CACHELESS STORE_RAW [[DATA:T[0-9]+\.[XYZW]]], [[ST_PTR:T[0-9]+\.[XYZW]]]
; EG: VTX_READ_32 [[DATA]], [[LD_PTR:T[0-9]+\.[XYZW]]]
; EG-DAG: MOV {{[\* ]*}}[[LD_PTR]], KC0[2].Z
; EG-DAG: LSHR {{[\* ]*}}[[ST_PTR]], KC0[2].Y, literal
define void @i32_to_v4i8(<4 x i8> addrspace(1)* %out, i32 addrspace(1)* %in) nounwind {
  %load = load i32, i32 addrspace(1)* %in, align 4
  %bc = bitcast i32 %load to <4 x i8>
  store <4 x i8> %bc, <4 x i8> addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}v2i16_to_v4i8:
; EG: MEM_RAT_CACHELESS STORE_RAW [[DATA:T[0-9]+\.[XYZW]]], [[ST_PTR:T[0-9]+\.[XYZW]]]
; EG: VTX_READ_32 [[DATA]], [[LD_PTR:T[0-9]+\.[XYZW]]]
; EG-DAG: MOV {{[\* ]*}}[[LD_PTR]], KC0[2].Z
; EG-DAG: LSHR {{[\* ]*}}[[ST_PTR]], KC0[2].Y, literal
define void @v2i16_to_v4i8(<4 x i8> addrspace(1)* %out, <2 x i16> addrspace(1)* %in) nounwind {
  %load = load <2 x i16>, <2 x i16>  addrspace(1)* %in, align 4
  %bc = bitcast <2 x i16> %load to <4 x i8>
  store <4 x i8> %bc, <4 x i8> addrspace(1)* %out, align 4
  ret void
}

; This just checks for crash in BUILD_VECTOR/EXTRACT_ELEMENT combine
; the stack manipulation is tricky to follow
; TODO: This should only use one load
; FUNC-LABEL: {{^}}v4i16_extract_i8:
; EG: MEM_RAT MSKOR {{T[0-9]+\.XW}}, [[ST_PTR:T[0-9]+\.[XYZW]]]
; EG: VTX_READ_16
; EG: VTX_READ_16
; EG-DAG: BFE_UINT
; EG-DAG: LSHR {{[\* ]*}}[[ST_PTR]], KC0[2].Y, literal
define void @v4i16_extract_i8(i8 addrspace(1)* %out, <4 x i16> addrspace(1)* %in) nounwind {
  %load = load <4 x i16>, <4 x i16>  addrspace(1)* %in, align 2
  %bc = bitcast <4 x i16> %load to <8 x i8>
  %element = extractelement <8 x i8> %bc, i32 5
  store i8 %element, i8 addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}bitcast_v2i32_to_f64:
; EG: MEM_RAT_CACHELESS STORE_RAW [[DATA:T[0-9]+\.XY]], [[ST_PTR:T[0-9]+\.[XYZW]]]
; EG: VTX_READ_64 [[DATA]], [[LD_PTR:T[0-9]+\.[XYZW]]]
; EG-DAG: MOV {{[\* ]*}}[[LD_PTR]], KC0[2].Z
; EG-DAG: LSHR {{[\* ]*}}[[ST_PTR]], KC0[2].Y, literal
define void @bitcast_v2i32_to_f64(double addrspace(1)* %out, <2 x i32> addrspace(1)* %in) {
  %val = load <2 x i32>, <2 x i32> addrspace(1)* %in, align 8
  %bc = bitcast <2 x i32> %val to double
  store double %bc, double addrspace(1)* %out, align 8
  ret void
}

