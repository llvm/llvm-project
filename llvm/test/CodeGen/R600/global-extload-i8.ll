; RUN: llc -march=r600 -mcpu=SI -verify-machineinstrs< %s | FileCheck -check-prefix=SI -check-prefix=FUNC %s
; RUN: llc -march=r600 -mcpu=tonga -verify-machineinstrs< %s | FileCheck -check-prefix=SI -check-prefix=FUNC %s
; RUN: llc -march=r600 -mcpu=cypress < %s | FileCheck -check-prefix=EG -check-prefix=FUNC %s

; FUNC-LABEL: {{^}}zextload_global_i8_to_i32:
; SI: buffer_load_ubyte
; SI: buffer_store_dword
; SI: s_endpgm
define void @zextload_global_i8_to_i32(i32 addrspace(1)* %out, i8 addrspace(1)* %in) nounwind {
  %a = load i8 addrspace(1)* %in
  %ext = zext i8 %a to i32
  store i32 %ext, i32 addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}sextload_global_i8_to_i32:
; SI: buffer_load_sbyte
; SI: buffer_store_dword
; SI: s_endpgm
define void @sextload_global_i8_to_i32(i32 addrspace(1)* %out, i8 addrspace(1)* %in) nounwind {
  %a = load i8 addrspace(1)* %in
  %ext = sext i8 %a to i32
  store i32 %ext, i32 addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}zextload_global_v1i8_to_v1i32:
; SI: s_endpgm
define void @zextload_global_v1i8_to_v1i32(<1 x i32> addrspace(1)* %out, <1 x i8> addrspace(1)* nocapture %in) nounwind {
  %load = load <1 x i8> addrspace(1)* %in
  %ext = zext <1 x i8> %load to <1 x i32>
  store <1 x i32> %ext, <1 x i32> addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}sextload_global_v1i8_to_v1i32:
; SI: s_endpgm
define void @sextload_global_v1i8_to_v1i32(<1 x i32> addrspace(1)* %out, <1 x i8> addrspace(1)* nocapture %in) nounwind {
  %load = load <1 x i8> addrspace(1)* %in
  %ext = sext <1 x i8> %load to <1 x i32>
  store <1 x i32> %ext, <1 x i32> addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}zextload_global_v2i8_to_v2i32:
; SI: s_endpgm
define void @zextload_global_v2i8_to_v2i32(<2 x i32> addrspace(1)* %out, <2 x i8> addrspace(1)* nocapture %in) nounwind {
  %load = load <2 x i8> addrspace(1)* %in
  %ext = zext <2 x i8> %load to <2 x i32>
  store <2 x i32> %ext, <2 x i32> addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}sextload_global_v2i8_to_v2i32:
; SI: s_endpgm
define void @sextload_global_v2i8_to_v2i32(<2 x i32> addrspace(1)* %out, <2 x i8> addrspace(1)* nocapture %in) nounwind {
  %load = load <2 x i8> addrspace(1)* %in
  %ext = sext <2 x i8> %load to <2 x i32>
  store <2 x i32> %ext, <2 x i32> addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}zextload_global_v4i8_to_v4i32:
; SI: s_endpgm
define void @zextload_global_v4i8_to_v4i32(<4 x i32> addrspace(1)* %out, <4 x i8> addrspace(1)* nocapture %in) nounwind {
  %load = load <4 x i8> addrspace(1)* %in
  %ext = zext <4 x i8> %load to <4 x i32>
  store <4 x i32> %ext, <4 x i32> addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}sextload_global_v4i8_to_v4i32:
; SI: s_endpgm
define void @sextload_global_v4i8_to_v4i32(<4 x i32> addrspace(1)* %out, <4 x i8> addrspace(1)* nocapture %in) nounwind {
  %load = load <4 x i8> addrspace(1)* %in
  %ext = sext <4 x i8> %load to <4 x i32>
  store <4 x i32> %ext, <4 x i32> addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}zextload_global_v8i8_to_v8i32:
; SI: s_endpgm
define void @zextload_global_v8i8_to_v8i32(<8 x i32> addrspace(1)* %out, <8 x i8> addrspace(1)* nocapture %in) nounwind {
  %load = load <8 x i8> addrspace(1)* %in
  %ext = zext <8 x i8> %load to <8 x i32>
  store <8 x i32> %ext, <8 x i32> addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}sextload_global_v8i8_to_v8i32:
; SI: s_endpgm
define void @sextload_global_v8i8_to_v8i32(<8 x i32> addrspace(1)* %out, <8 x i8> addrspace(1)* nocapture %in) nounwind {
  %load = load <8 x i8> addrspace(1)* %in
  %ext = sext <8 x i8> %load to <8 x i32>
  store <8 x i32> %ext, <8 x i32> addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}zextload_global_v16i8_to_v16i32:
; SI: s_endpgm
define void @zextload_global_v16i8_to_v16i32(<16 x i32> addrspace(1)* %out, <16 x i8> addrspace(1)* nocapture %in) nounwind {
  %load = load <16 x i8> addrspace(1)* %in
  %ext = zext <16 x i8> %load to <16 x i32>
  store <16 x i32> %ext, <16 x i32> addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}sextload_global_v16i8_to_v16i32:
; SI: s_endpgm
define void @sextload_global_v16i8_to_v16i32(<16 x i32> addrspace(1)* %out, <16 x i8> addrspace(1)* nocapture %in) nounwind {
  %load = load <16 x i8> addrspace(1)* %in
  %ext = sext <16 x i8> %load to <16 x i32>
  store <16 x i32> %ext, <16 x i32> addrspace(1)* %out
  ret void
}

; XFUNC-LABEL: {{^}}zextload_global_v32i8_to_v32i32:
; XSI: s_endpgm
; define void @zextload_global_v32i8_to_v32i32(<32 x i32> addrspace(1)* %out, <32 x i8> addrspace(1)* nocapture %in) nounwind {
;   %load = load <32 x i8> addrspace(1)* %in
;   %ext = zext <32 x i8> %load to <32 x i32>
;   store <32 x i32> %ext, <32 x i32> addrspace(1)* %out
;   ret void
; }

; XFUNC-LABEL: {{^}}sextload_global_v32i8_to_v32i32:
; XSI: s_endpgm
; define void @sextload_global_v32i8_to_v32i32(<32 x i32> addrspace(1)* %out, <32 x i8> addrspace(1)* nocapture %in) nounwind {
;   %load = load <32 x i8> addrspace(1)* %in
;   %ext = sext <32 x i8> %load to <32 x i32>
;   store <32 x i32> %ext, <32 x i32> addrspace(1)* %out
;   ret void
; }

; XFUNC-LABEL: {{^}}zextload_global_v64i8_to_v64i32:
; XSI: s_endpgm
; define void @zextload_global_v64i8_to_v64i32(<64 x i32> addrspace(1)* %out, <64 x i8> addrspace(1)* nocapture %in) nounwind {
;   %load = load <64 x i8> addrspace(1)* %in
;   %ext = zext <64 x i8> %load to <64 x i32>
;   store <64 x i32> %ext, <64 x i32> addrspace(1)* %out
;   ret void
; }

; XFUNC-LABEL: {{^}}sextload_global_v64i8_to_v64i32:
; XSI: s_endpgm
; define void @sextload_global_v64i8_to_v64i32(<64 x i32> addrspace(1)* %out, <64 x i8> addrspace(1)* nocapture %in) nounwind {
;   %load = load <64 x i8> addrspace(1)* %in
;   %ext = sext <64 x i8> %load to <64 x i32>
;   store <64 x i32> %ext, <64 x i32> addrspace(1)* %out
;   ret void
; }

; FUNC-LABEL: {{^}}zextload_global_i8_to_i64:
; SI: buffer_load_ubyte v[[LO:[0-9]+]],
; SI: v_mov_b32_e32 v[[HI:[0-9]+]], 0{{$}}
; SI: buffer_store_dwordx2 v{{\[}}[[LO]]:[[HI]]]
define void @zextload_global_i8_to_i64(i64 addrspace(1)* %out, i8 addrspace(1)* %in) nounwind {
  %a = load i8 addrspace(1)* %in
  %ext = zext i8 %a to i64
  store i64 %ext, i64 addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}sextload_global_i8_to_i64:
; SI: buffer_load_sbyte [[LOAD:v[0-9]+]],
; SI: v_ashrrev_i32_e32 v{{[0-9]+}}, 31, [[LOAD]]
; SI: buffer_store_dwordx2
define void @sextload_global_i8_to_i64(i64 addrspace(1)* %out, i8 addrspace(1)* %in) nounwind {
  %a = load i8 addrspace(1)* %in
  %ext = sext i8 %a to i64
  store i64 %ext, i64 addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}zextload_global_v1i8_to_v1i64:
; SI: s_endpgm
define void @zextload_global_v1i8_to_v1i64(<1 x i64> addrspace(1)* %out, <1 x i8> addrspace(1)* nocapture %in) nounwind {
  %load = load <1 x i8> addrspace(1)* %in
  %ext = zext <1 x i8> %load to <1 x i64>
  store <1 x i64> %ext, <1 x i64> addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}sextload_global_v1i8_to_v1i64:
; SI: s_endpgm
define void @sextload_global_v1i8_to_v1i64(<1 x i64> addrspace(1)* %out, <1 x i8> addrspace(1)* nocapture %in) nounwind {
  %load = load <1 x i8> addrspace(1)* %in
  %ext = sext <1 x i8> %load to <1 x i64>
  store <1 x i64> %ext, <1 x i64> addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}zextload_global_v2i8_to_v2i64:
; SI: s_endpgm
define void @zextload_global_v2i8_to_v2i64(<2 x i64> addrspace(1)* %out, <2 x i8> addrspace(1)* nocapture %in) nounwind {
  %load = load <2 x i8> addrspace(1)* %in
  %ext = zext <2 x i8> %load to <2 x i64>
  store <2 x i64> %ext, <2 x i64> addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}sextload_global_v2i8_to_v2i64:
; SI: s_endpgm
define void @sextload_global_v2i8_to_v2i64(<2 x i64> addrspace(1)* %out, <2 x i8> addrspace(1)* nocapture %in) nounwind {
  %load = load <2 x i8> addrspace(1)* %in
  %ext = sext <2 x i8> %load to <2 x i64>
  store <2 x i64> %ext, <2 x i64> addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}zextload_global_v4i8_to_v4i64:
; SI: s_endpgm
define void @zextload_global_v4i8_to_v4i64(<4 x i64> addrspace(1)* %out, <4 x i8> addrspace(1)* nocapture %in) nounwind {
  %load = load <4 x i8> addrspace(1)* %in
  %ext = zext <4 x i8> %load to <4 x i64>
  store <4 x i64> %ext, <4 x i64> addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}sextload_global_v4i8_to_v4i64:
; SI: s_endpgm
define void @sextload_global_v4i8_to_v4i64(<4 x i64> addrspace(1)* %out, <4 x i8> addrspace(1)* nocapture %in) nounwind {
  %load = load <4 x i8> addrspace(1)* %in
  %ext = sext <4 x i8> %load to <4 x i64>
  store <4 x i64> %ext, <4 x i64> addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}zextload_global_v8i8_to_v8i64:
; SI: s_endpgm
define void @zextload_global_v8i8_to_v8i64(<8 x i64> addrspace(1)* %out, <8 x i8> addrspace(1)* nocapture %in) nounwind {
  %load = load <8 x i8> addrspace(1)* %in
  %ext = zext <8 x i8> %load to <8 x i64>
  store <8 x i64> %ext, <8 x i64> addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}sextload_global_v8i8_to_v8i64:
; SI: s_endpgm
define void @sextload_global_v8i8_to_v8i64(<8 x i64> addrspace(1)* %out, <8 x i8> addrspace(1)* nocapture %in) nounwind {
  %load = load <8 x i8> addrspace(1)* %in
  %ext = sext <8 x i8> %load to <8 x i64>
  store <8 x i64> %ext, <8 x i64> addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}zextload_global_v16i8_to_v16i64:
; SI: s_endpgm
define void @zextload_global_v16i8_to_v16i64(<16 x i64> addrspace(1)* %out, <16 x i8> addrspace(1)* nocapture %in) nounwind {
  %load = load <16 x i8> addrspace(1)* %in
  %ext = zext <16 x i8> %load to <16 x i64>
  store <16 x i64> %ext, <16 x i64> addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}sextload_global_v16i8_to_v16i64:
; SI: s_endpgm
define void @sextload_global_v16i8_to_v16i64(<16 x i64> addrspace(1)* %out, <16 x i8> addrspace(1)* nocapture %in) nounwind {
  %load = load <16 x i8> addrspace(1)* %in
  %ext = sext <16 x i8> %load to <16 x i64>
  store <16 x i64> %ext, <16 x i64> addrspace(1)* %out
  ret void
}

; XFUNC-LABEL: {{^}}zextload_global_v32i8_to_v32i64:
; XSI: s_endpgm
; define void @zextload_global_v32i8_to_v32i64(<32 x i64> addrspace(1)* %out, <32 x i8> addrspace(1)* nocapture %in) nounwind {
;   %load = load <32 x i8> addrspace(1)* %in
;   %ext = zext <32 x i8> %load to <32 x i64>
;   store <32 x i64> %ext, <32 x i64> addrspace(1)* %out
;   ret void
; }

; XFUNC-LABEL: {{^}}sextload_global_v32i8_to_v32i64:
; XSI: s_endpgm
; define void @sextload_global_v32i8_to_v32i64(<32 x i64> addrspace(1)* %out, <32 x i8> addrspace(1)* nocapture %in) nounwind {
;   %load = load <32 x i8> addrspace(1)* %in
;   %ext = sext <32 x i8> %load to <32 x i64>
;   store <32 x i64> %ext, <32 x i64> addrspace(1)* %out
;   ret void
; }

; XFUNC-LABEL: {{^}}zextload_global_v64i8_to_v64i64:
; XSI: s_endpgm
; define void @zextload_global_v64i8_to_v64i64(<64 x i64> addrspace(1)* %out, <64 x i8> addrspace(1)* nocapture %in) nounwind {
;   %load = load <64 x i8> addrspace(1)* %in
;   %ext = zext <64 x i8> %load to <64 x i64>
;   store <64 x i64> %ext, <64 x i64> addrspace(1)* %out
;   ret void
; }

; XFUNC-LABEL: {{^}}sextload_global_v64i8_to_v64i64:
; XSI: s_endpgm
; define void @sextload_global_v64i8_to_v64i64(<64 x i64> addrspace(1)* %out, <64 x i8> addrspace(1)* nocapture %in) nounwind {
;   %load = load <64 x i8> addrspace(1)* %in
;   %ext = sext <64 x i8> %load to <64 x i64>
;   store <64 x i64> %ext, <64 x i64> addrspace(1)* %out
;   ret void
; }
