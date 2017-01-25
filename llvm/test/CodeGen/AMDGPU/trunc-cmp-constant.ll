; RUN: llc -march=amdgcn -verify-machineinstrs < %s | FileCheck -check-prefix=SI -check-prefix=FUNC %s
; RUN: llc -march=amdgcn -mcpu=tonga -mattr=-flat-for-global -verify-machineinstrs < %s | FileCheck -check-prefix=SI -check-prefix=FUNC %s

declare i32 @llvm.amdgcn.workitem.id.x() nounwind readnone

; FUNC-LABEL {{^}}sextload_i1_to_i32_trunc_cmp_eq_0:
; SI: buffer_load_ubyte [[LOAD:v[0-9]+]]
; SI: v_and_b32_e32 [[TMP:v[0-9]+]], 1, [[LOAD]]
; SI: v_cmp_eq_u32_e32 vcc, 0, [[TMP]]{{$}}
; SI: v_cndmask_b32_e64
; SI: buffer_store_byte
define void @sextload_i1_to_i32_trunc_cmp_eq_0(i1 addrspace(1)* %out, i1 addrspace(1)* %in) nounwind {
  %load = load i1, i1 addrspace(1)* %in
  %ext = sext i1 %load to i32
  %cmp = icmp eq i32 %ext, 0
  store i1 %cmp, i1 addrspace(1)* %out
  ret void
}

; FIXME: The negate should be inverting the compare.
; FUNC-LABEL: {{^}}zextload_i1_to_i32_trunc_cmp_eq_0:
; SI: buffer_load_ubyte [[LOAD:v[0-9]+]]
; SI: v_and_b32_e32 [[TMP:v[0-9]+]], 1, [[LOAD]]
; SI: v_cmp_eq_u32_e32 vcc, 1, [[TMP]]{{$}}
; SI-NEXT: s_xor_b64 [[NEG:s\[[0-9]+:[0-9]+\]]], vcc, -1
; SI-NEXT: v_cndmask_b32_e64 [[RESULT:v[0-9]+]], 0, 1, [[NEG]]
; SI: buffer_store_byte [[RESULT]]
define void @zextload_i1_to_i32_trunc_cmp_eq_0(i1 addrspace(1)* %out, i1 addrspace(1)* %in) nounwind {
  %load = load i1, i1 addrspace(1)* %in
  %ext = zext i1 %load to i32
  %cmp = icmp eq i32 %ext, 0
  store i1 %cmp, i1 addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}sextload_i1_to_i32_trunc_cmp_eq_1:
; SI: v_mov_b32_e32 [[RESULT:v[0-9]+]], 0{{$}}
; SI: buffer_store_byte [[RESULT]]
define void @sextload_i1_to_i32_trunc_cmp_eq_1(i1 addrspace(1)* %out, i1 addrspace(1)* %in) nounwind {
  %load = load i1, i1 addrspace(1)* %in
  %ext = sext i1 %load to i32
  %cmp = icmp eq i32 %ext, 1
  store i1 %cmp, i1 addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}zextload_i1_to_i32_trunc_cmp_eq_1:
; SI: buffer_load_ubyte [[LOAD:v[0-9]+]]
; SI: v_and_b32_e32 [[RESULT:v[0-9]+]], 1, [[LOAD]]
; SI: buffer_store_byte [[RESULT]]
define void @zextload_i1_to_i32_trunc_cmp_eq_1(i1 addrspace(1)* %out, i1 addrspace(1)* %in) nounwind {
  %load = load i1, i1 addrspace(1)* %in
  %ext = zext i1 %load to i32
  %cmp = icmp eq i32 %ext, 1
  store i1 %cmp, i1 addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}sextload_i1_to_i32_trunc_cmp_eq_neg1:
; SI: buffer_load_ubyte [[LOAD:v[0-9]+]]
; SI: v_and_b32_e32 [[RESULT:v[0-9]+]], 1, [[LOAD]]
; SI: buffer_store_byte [[RESULT]]
define void @sextload_i1_to_i32_trunc_cmp_eq_neg1(i1 addrspace(1)* %out, i1 addrspace(1)* %in) nounwind {
  %load = load i1, i1 addrspace(1)* %in
  %ext = sext i1 %load to i32
  %cmp = icmp eq i32 %ext, -1
  store i1 %cmp, i1 addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}zextload_i1_to_i32_trunc_cmp_eq_neg1:
; SI: v_mov_b32_e32 [[RESULT:v[0-9]+]], 0{{$}}
; SI: buffer_store_byte [[RESULT]]
define void @zextload_i1_to_i32_trunc_cmp_eq_neg1(i1 addrspace(1)* %out, i1 addrspace(1)* %in) nounwind {
  %load = load i1, i1 addrspace(1)* %in
  %ext = zext i1 %load to i32
  %cmp = icmp eq i32 %ext, -1
  store i1 %cmp, i1 addrspace(1)* %out
  ret void
}


; FUNC-LABEL {{^}}sextload_i1_to_i32_trunc_cmp_ne_0:
; SI: buffer_load_ubyte [[LOAD:v[0-9]+]]
; SI: v_and_b32_e32 [[TMP:v[0-9]+]], 1, [[LOAD]]
; SI: buffer_store_byte [[RESULT]]
define void @sextload_i1_to_i32_trunc_cmp_ne_0(i1 addrspace(1)* %out, i1 addrspace(1)* %in) nounwind {
  %load = load i1, i1 addrspace(1)* %in
  %ext = sext i1 %load to i32
  %cmp = icmp ne i32 %ext, 0
  store i1 %cmp, i1 addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}zextload_i1_to_i32_trunc_cmp_ne_0:
; SI: buffer_load_ubyte [[LOAD:v[0-9]+]]
; SI: v_and_b32_e32 [[TMP:v[0-9]+]], 1, [[LOAD]]
; SI: buffer_store_byte [[RESULT]]
define void @zextload_i1_to_i32_trunc_cmp_ne_0(i1 addrspace(1)* %out, i1 addrspace(1)* %in) nounwind {
  %load = load i1, i1 addrspace(1)* %in
  %ext = zext i1 %load to i32
  %cmp = icmp ne i32 %ext, 0
  store i1 %cmp, i1 addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}sextload_i1_to_i32_trunc_cmp_ne_1:
; SI: v_mov_b32_e32 [[RESULT:v[0-9]+]], 1{{$}}
; SI: buffer_store_byte [[RESULT]]
define void @sextload_i1_to_i32_trunc_cmp_ne_1(i1 addrspace(1)* %out, i1 addrspace(1)* %in) nounwind {
  %load = load i1, i1 addrspace(1)* %in
  %ext = sext i1 %load to i32
  %cmp = icmp ne i32 %ext, 1
  store i1 %cmp, i1 addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}zextload_i1_to_i32_trunc_cmp_ne_1:
; SI: buffer_load_ubyte [[LOAD:v[0-9]+]]
; SI: v_and_b32_e32 [[TMP:v[0-9]+]], 1, [[LOAD]]
; SI: v_cmp_eq_u32_e32 vcc, 1, [[TMP]]{{$}}
; SI-NEXT: s_xor_b64 [[NEG:s\[[0-9]+:[0-9]+\]]], vcc, -1
; SI-NEXT: v_cndmask_b32_e64 [[RESULT:v[0-9]+]], 0, 1, [[NEG]]
; SI: buffer_store_byte [[RESULT]]
define void @zextload_i1_to_i32_trunc_cmp_ne_1(i1 addrspace(1)* %out, i1 addrspace(1)* %in) nounwind {
  %load = load i1, i1 addrspace(1)* %in
  %ext = zext i1 %load to i32
  %cmp = icmp ne i32 %ext, 1
  store i1 %cmp, i1 addrspace(1)* %out
  ret void
}

; FIXME: This should be one compare.
; FUNC-LABEL: {{^}}sextload_i1_to_i32_trunc_cmp_ne_neg1:
; XSI: buffer_load_ubyte [[LOAD:v[0-9]+]]
; XSI: v_and_b32_e32 [[TMP:v[0-9]+]], 1, [[LOAD]]
; XSI: v_cmp_eq_u32_e64 [[CMP0:s\[[0-9]+:[0-9]+\]]], [[TMP]], 0{{$}}
; XSI-NEXT: v_cndmask_b32_e64 [[RESULT:v[0-9]+]], 0, 1, [[CMP0]]
; XSI-NEXT: buffer_store_byte [[RESULT]]
define void @sextload_i1_to_i32_trunc_cmp_ne_neg1(i1 addrspace(1)* %out, i1 addrspace(1)* %in) nounwind {
  %load = load i1, i1 addrspace(1)* %in
  %ext = sext i1 %load to i32
  %cmp = icmp ne i32 %ext, -1
  store i1 %cmp, i1 addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}zextload_i1_to_i32_trunc_cmp_ne_neg1:
; SI: v_mov_b32_e32 [[RESULT:v[0-9]+]], 1{{$}}
; SI: buffer_store_byte [[RESULT]]
define void @zextload_i1_to_i32_trunc_cmp_ne_neg1(i1 addrspace(1)* %out, i1 addrspace(1)* %in) nounwind {
  %load = load i1, i1 addrspace(1)* %in
  %ext = zext i1 %load to i32
  %cmp = icmp ne i32 %ext, -1
  store i1 %cmp, i1 addrspace(1)* %out
  ret void
}

; FIXME: Need to handle non-uniform case for function below (load without gep).
; FUNC-LABEL: {{^}}masked_load_i1_to_i32_trunc_cmp_ne_neg1:
; SI: {{buffer|flat}}_load_sbyte [[LOAD:v[0-9]+]]
; SI: v_cmp_ne_u32_e32 vcc, -1, [[LOAD]]{{$}}
; SI-NEXT: v_cndmask_b32_e64
; SI: {{buffer|flat}}_store_byte
define void @masked_load_i1_to_i32_trunc_cmp_ne_neg1(i1 addrspace(1)* %out, i8 addrspace(1)* %in) nounwind {
  %tid.x = call i32 @llvm.amdgcn.workitem.id.x()
  %in.ptr = getelementptr i8, i8 addrspace(1)* %in, i32 %tid.x
  %load = load i8, i8 addrspace(1)* %in.ptr
  %masked = and i8 %load, 255
  %ext = sext i8 %masked to i32
  %cmp = icmp ne i32 %ext, -1
  store i1 %cmp, i1 addrspace(1)* %out
  ret void
}
