; RUN: llc -march=amdgcn -mcpu=SI -verify-machineinstrs < %s | FileCheck -check-prefix=SI -check-prefix=FUNC %s
; RUN: llc -march=amdgcn -mcpu=tonga -verify-machineinstrs < %s | FileCheck -check-prefix=SI -check-prefix=FUNC %s

declare i16 @llvm.bitreverse.i16(i16) #1
declare i32 @llvm.bitreverse.i32(i32) #1
declare i64 @llvm.bitreverse.i64(i64) #1

declare <2 x i32> @llvm.bitreverse.v2i32(<2 x i32>) #1
declare <4 x i32> @llvm.bitreverse.v4i32(<4 x i32>) #1

declare <2 x i64> @llvm.bitreverse.v2i64(<2 x i64>) #1
declare <4 x i64> @llvm.bitreverse.v4i64(<4 x i64>) #1

declare i32 @llvm.AMDGPU.brev(i32) #1

; FUNC-LABEL: {{^}}s_brev_i16:
; SI: s_brev_b32
define void @s_brev_i16(i16 addrspace(1)* noalias %out, i16 %val) #0 {
  %brev = call i16 @llvm.bitreverse.i16(i16 %val) #1
  store i16 %brev, i16 addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}v_brev_i16:
; SI: v_bfrev_b32_e32
define void @v_brev_i16(i16 addrspace(1)* noalias %out, i16 addrspace(1)* noalias %valptr) #0 {
  %val = load i16, i16 addrspace(1)* %valptr
  %brev = call i16 @llvm.bitreverse.i16(i16 %val) #1
  store i16 %brev, i16 addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}s_brev_i32:
; SI: s_load_dword [[VAL:s[0-9]+]],
; SI: s_brev_b32 [[SRESULT:s[0-9]+]], [[VAL]]
; SI: v_mov_b32_e32 [[VRESULT:v[0-9]+]], [[SRESULT]]
; SI: buffer_store_dword [[VRESULT]],
; SI: s_endpgm
define void @s_brev_i32(i32 addrspace(1)* noalias %out, i32 %val) #0 {
  %brev = call i32 @llvm.bitreverse.i32(i32 %val) #1
  store i32 %brev, i32 addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}v_brev_i32:
; SI: buffer_load_dword [[VAL:v[0-9]+]],
; SI: v_bfrev_b32_e32 [[RESULT:v[0-9]+]], [[VAL]]
; SI: buffer_store_dword [[RESULT]],
; SI: s_endpgm
define void @v_brev_i32(i32 addrspace(1)* noalias %out, i32 addrspace(1)* noalias %valptr) #0 {
  %val = load i32, i32 addrspace(1)* %valptr
  %brev = call i32 @llvm.bitreverse.i32(i32 %val) #1
  store i32 %brev, i32 addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}s_brev_v2i32:
; SI: s_brev_b32
; SI: s_brev_b32
define void @s_brev_v2i32(<2 x i32> addrspace(1)* noalias %out, <2 x i32> %val) #0 {
  %brev = call <2 x i32> @llvm.bitreverse.v2i32(<2 x i32> %val) #1
  store <2 x i32> %brev, <2 x i32> addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}v_brev_v2i32:
; SI: v_bfrev_b32_e32
; SI: v_bfrev_b32_e32
define void @v_brev_v2i32(<2 x i32> addrspace(1)* noalias %out, <2 x i32> addrspace(1)* noalias %valptr) #0 {
  %val = load <2 x i32>, <2 x i32> addrspace(1)* %valptr
  %brev = call <2 x i32> @llvm.bitreverse.v2i32(<2 x i32> %val) #1
  store <2 x i32> %brev, <2 x i32> addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}s_brev_i64:
define void @s_brev_i64(i64 addrspace(1)* noalias %out, i64 %val) #0 {
  %brev = call i64 @llvm.bitreverse.i64(i64 %val) #1
  store i64 %brev, i64 addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}v_brev_i64:
define void @v_brev_i64(i64 addrspace(1)* noalias %out, i64 addrspace(1)* noalias %valptr) #0 {
  %val = load i64, i64 addrspace(1)* %valptr
  %brev = call i64 @llvm.bitreverse.i64(i64 %val) #1
  store i64 %brev, i64 addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}s_brev_v2i64:
define void @s_brev_v2i64(<2 x i64> addrspace(1)* noalias %out, <2 x i64> %val) #0 {
  %brev = call <2 x i64> @llvm.bitreverse.v2i64(<2 x i64> %val) #1
  store <2 x i64> %brev, <2 x i64> addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}v_brev_v2i64:
define void @v_brev_v2i64(<2 x i64> addrspace(1)* noalias %out, <2 x i64> addrspace(1)* noalias %valptr) #0 {
  %val = load <2 x i64>, <2 x i64> addrspace(1)* %valptr
  %brev = call <2 x i64> @llvm.bitreverse.v2i64(<2 x i64> %val) #1
  store <2 x i64> %brev, <2 x i64> addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}legacy_s_brev_i32:
; SI: s_brev_b32
define void @legacy_s_brev_i32(i32 addrspace(1)* noalias %out, i32 %val) nounwind {
  %brev = call i32 @llvm.AMDGPU.brev(i32 %val) #1
  store i32 %brev, i32 addrspace(1)* %out
  ret void
}

attributes #0 = { nounwind }
attributes #1 = { nounwind readnone }
