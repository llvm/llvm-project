; RUN: llc -march=amdgcn -verify-machineinstrs < %s | FileCheck -check-prefix=SI -check-prefix=FUNC %s
; RUN: llc -march=amdgcn -mcpu=tonga -verify-machineinstrs < %s | FileCheck -check-prefix=SI -check-prefix=FUNC %s

; XXX - Why the packing?
; FUNC-LABEL: {{^}}scalar_to_vector_v2i32:
; SI: buffer_load_dword [[VAL:v[0-9]+]],
; SI: v_lshrrev_b32_e32 [[SHR:v[0-9]+]], 16, [[VAL]]
; SI: v_lshlrev_b32_e32 [[SHL:v[0-9]+]], 16, [[SHR]]
; SI: v_or_b32_e32 v[[OR:[0-9]+]], [[SHL]], [[SHR]]
; SI: v_mov_b32_e32 v[[COPY:[0-9]+]], v[[OR]]
; SI: buffer_store_dwordx2 v{{\[}}[[OR]]:[[COPY]]{{\]}}
define void @scalar_to_vector_v2i32(<4 x i16> addrspace(1)* %out, i32 addrspace(1)* %in) nounwind {
  %tmp1 = load i32, i32 addrspace(1)* %in, align 4
  %bc = bitcast i32 %tmp1 to <2 x i16>
  %tmp2 = shufflevector <2 x i16> %bc, <2 x i16> undef, <4 x i32> <i32 1, i32 1, i32 1, i32 1>
  store <4 x i16> %tmp2, <4 x i16> addrspace(1)* %out, align 8
  ret void
}

; FUNC-LABEL: {{^}}scalar_to_vector_v2f32:
; SI: buffer_load_dword [[VAL:v[0-9]+]],
; SI: v_lshrrev_b32_e32 [[RESULT:v[0-9]+]], 16, [[VAL]]
; SI: buffer_store_dwordx2
define void @scalar_to_vector_v2f32(<4 x i16> addrspace(1)* %out, float addrspace(1)* %in) nounwind {
  %tmp1 = load float, float addrspace(1)* %in, align 4
  %bc = bitcast float %tmp1 to <2 x i16>
  %tmp2 = shufflevector <2 x i16> %bc, <2 x i16> undef, <4 x i32> <i32 1, i32 1, i32 1, i32 1>
  store <4 x i16> %tmp2, <4 x i16> addrspace(1)* %out, align 8
  ret void
}

; Getting a SCALAR_TO_VECTOR seems to be tricky. These cases managed
; to produce one, but for some reason never made it to selection.


; define void @scalar_to_vector_test2(<8 x i8> addrspace(1)* %out, i32 addrspace(1)* %in) nounwind {
;   %tmp1 = load i32, i32 addrspace(1)* %in, align 4
;   %bc = bitcast i32 %tmp1 to <4 x i8>

;   %tmp2 = shufflevector <4 x i8> %bc, <4 x i8> undef, <8 x i32> <i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1>
;   store <8 x i8> %tmp2, <8 x i8> addrspace(1)* %out, align 4
;   ret void
; }

; define void @scalar_to_vector_test3(<4 x i32> addrspace(1)* %out) nounwind {
;   %newvec0 = insertelement <2 x i64> undef, i64 12345, i32 0
;   %newvec1 = insertelement <2 x i64> %newvec0, i64 undef, i32 1
;   %bc = bitcast <2 x i64> %newvec1 to <4 x i32>
;   %add = add <4 x i32> %bc, <i32 1, i32 2, i32 3, i32 4>
;   store <4 x i32> %add, <4 x i32> addrspace(1)* %out, align 16
;   ret void
; }

; define void @scalar_to_vector_test4(<8 x i16> addrspace(1)* %out) nounwind {
;   %newvec0 = insertelement <4 x i32> undef, i32 12345, i32 0
;   %bc = bitcast <4 x i32> %newvec0 to <8 x i16>
;   %add = add <8 x i16> %bc, <i16 1, i16 2, i16 3, i16 4, i16 1, i16 2, i16 3, i16 4>
;   store <8 x i16> %add, <8 x i16> addrspace(1)* %out, align 16
;   ret void
; }

; define void @scalar_to_vector_test5(<4 x i16> addrspace(1)* %out) nounwind {
;   %newvec0 = insertelement <2 x i32> undef, i32 12345, i32 0
;   %bc = bitcast <2 x i32> %newvec0 to <4 x i16>
;   %add = add <4 x i16> %bc, <i16 1, i16 2, i16 3, i16 4>
;   store <4 x i16> %add, <4 x i16> addrspace(1)* %out, align 16
;   ret void
; }

; define void @scalar_to_vector_test6(<4 x i16> addrspace(1)* %out) nounwind {
;   %newvec0 = insertelement <2 x i32> undef, i32 12345, i32 0
;   %bc = bitcast <2 x i32> %newvec0 to <4 x i16>
;   %add = add <4 x i16> %bc, <i16 1, i16 2, i16 3, i16 4>
;   store <4 x i16> %add, <4 x i16> addrspace(1)* %out, align 16
;   ret void
; }
