; RUN: llc -march=amdgcn -mcpu=verde -verify-machineinstrs < %s | FileCheck -check-prefix=GCN -check-prefix=FUNC %s
; RUN: llc -march=amdgcn -mcpu=tonga -verify-machineinstrs < %s | FileCheck -check-prefix=GCN -check-prefix=FUNC %s
; RUN: llc -march=r600 -mcpu=redwood < %s | FileCheck -check-prefix=EG -check-prefix=FUNC %s
; RUN: llc -march=r600 -mcpu=cayman < %s | FileCheck -check-prefix=CM -check-prefix=FUNC %s

; FUNC-LABEL: {{^}}store_local_i1:
; EG: LDS_BYTE_WRITE
; GCN: ds_write_b8
define void @store_local_i1(i1 addrspace(3)* %out) {
entry:
  store i1 true, i1 addrspace(3)* %out
  ret void
}

; FUNC-LABEL: {{^}}store_local_i8:
; EG: LDS_BYTE_WRITE

; GCN: ds_write_b8
define void @store_local_i8(i8 addrspace(3)* %out, i8 %in) {
  store i8 %in, i8 addrspace(3)* %out
  ret void
}

; FUNC-LABEL: {{^}}store_local_i16:
; EG: LDS_SHORT_WRITE

; GCN: ds_write_b16
define void @store_local_i16(i16 addrspace(3)* %out, i16 %in) {
  store i16 %in, i16 addrspace(3)* %out
  ret void
}

; FUNC-LABEL: {{^}}store_local_v2i16:
; EG: LDS_WRITE

; CM: LDS_WRITE

; GCN: ds_write_b32
define void @store_local_v2i16(<2 x i16> addrspace(3)* %out, <2 x i16> %in) {
entry:
  store <2 x i16> %in, <2 x i16> addrspace(3)* %out
  ret void
}

; FUNC-LABEL: {{^}}store_local_v4i8:
; EG: LDS_WRITE

; CM: LDS_WRITE

; GCN: ds_write_b32
define void @store_local_v4i8(<4 x i8> addrspace(3)* %out, <4 x i8> %in) {
entry:
  store <4 x i8> %in, <4 x i8> addrspace(3)* %out
  ret void
}

; FUNC-LABEL: {{^}}store_local_v2i32:
; EG: LDS_WRITE
; EG: LDS_WRITE

; CM: LDS_WRITE
; CM: LDS_WRITE

; GCN: ds_write_b64
define void @store_local_v2i32(<2 x i32> addrspace(3)* %out, <2 x i32> %in) {
entry:
  store <2 x i32> %in, <2 x i32> addrspace(3)* %out
  ret void
}

; FUNC-LABEL: {{^}}store_local_v4i32:
; EG: LDS_WRITE
; EG: LDS_WRITE
; EG: LDS_WRITE
; EG: LDS_WRITE

; CM: LDS_WRITE
; CM: LDS_WRITE
; CM: LDS_WRITE
; CM: LDS_WRITE

; GCN: ds_write2_b64
define void @store_local_v4i32(<4 x i32> addrspace(3)* %out, <4 x i32> %in) {
entry:
  store <4 x i32> %in, <4 x i32> addrspace(3)* %out
  ret void
}

; FUNC-LABEL: {{^}}store_local_v4i32_align4:
; EG: LDS_WRITE
; EG: LDS_WRITE
; EG: LDS_WRITE
; EG: LDS_WRITE

; CM: LDS_WRITE
; CM: LDS_WRITE
; CM: LDS_WRITE
; CM: LDS_WRITE

; GCN: ds_write2_b32
; GCN: ds_write2_b32
define void @store_local_v4i32_align4(<4 x i32> addrspace(3)* %out, <4 x i32> %in) {
entry:
  store <4 x i32> %in, <4 x i32> addrspace(3)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}store_local_i64_i8:
; EG: LDS_BYTE_WRITE
; GCN: ds_write_b8
define void @store_local_i64_i8(i8 addrspace(3)* %out, i64 %in) {
entry:
  %0 = trunc i64 %in to i8
  store i8 %0, i8 addrspace(3)* %out
  ret void
}

; FUNC-LABEL: {{^}}store_local_i64_i16:
; EG: LDS_SHORT_WRITE
; GCN: ds_write_b16
define void @store_local_i64_i16(i16 addrspace(3)* %out, i64 %in) {
entry:
  %0 = trunc i64 %in to i16
  store i16 %0, i16 addrspace(3)* %out
  ret void
}
