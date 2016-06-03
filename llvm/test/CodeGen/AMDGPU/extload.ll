; RUN: llc -march=amdgcn -verify-machineinstrs < %s | FileCheck -check-prefix=SI-NOHSA -check-prefix=FUNC %s
; RUN: llc -mtriple=amdgcn-amdhsa -mcpu=kaveri -verify-machineinstrs < %s | FileCheck -check-prefix=FUNC -check-prefix=CI-HSA -check-prefix=SI %s
; RUN: llc -march=amdgcn -mcpu=tonga -verify-machineinstrs < %s | FileCheck -check-prefix=SI-NOHSA -check-prefix=FUNC %s
; RUN: llc -march=r600 -mcpu=redwood < %s | FileCheck -check-prefix=EG -check-prefix=FUNC %s

; FIXME: This seems to not ever actually become an extload
; FUNC-LABEL: {{^}}global_anyext_load_i8:
; GCN: buffer_load_dword v{{[0-9]+}}
; GCN: buffer_store_dword v{{[0-9]+}}

; EG: MEM_RAT_CACHELESS STORE_RAW [[VAL:T[0-9]+.[XYZW]]],
; EG: VTX_READ_32 [[VAL]]
define void @global_anyext_load_i8(i8 addrspace(1)* nocapture noalias %out, i8 addrspace(1)* nocapture noalias %src) nounwind {
  %cast = bitcast i8 addrspace(1)* %src to i32 addrspace(1)*
  %load = load i32, i32 addrspace(1)* %cast
  %x = bitcast i32 %load to <4 x i8>
  %castOut = bitcast i8 addrspace(1)* %out to <4 x i8> addrspace(1)*
  store <4 x i8> %x, <4 x i8> addrspace(1)* %castOut
  ret void
}

; FUNC-LABEL: {{^}}global_anyext_load_i16:
; GCN: buffer_load_dword v{{[0-9]+}}
; GCN: buffer_store_dword v{{[0-9]+}}

; EG: MEM_RAT_CACHELESS STORE_RAW [[VAL:T[0-9]+.[XYZW]]],
; EG: VTX_READ_32 [[VAL]]
define void @global_anyext_load_i16(i16 addrspace(1)* nocapture noalias %out, i16 addrspace(1)* nocapture noalias %src) nounwind {
  %cast = bitcast i16 addrspace(1)* %src to i32 addrspace(1)*
  %load = load i32, i32 addrspace(1)* %cast
  %x = bitcast i32 %load to <2 x i16>
  %castOut = bitcast i16 addrspace(1)* %out to <2 x i16> addrspace(1)*
  store <2 x i16> %x, <2 x i16> addrspace(1)* %castOut
  ret void
}

; FUNC-LABEL: {{^}}local_anyext_load_i8:
; GCN: ds_read_b32 v{{[0-9]+}}
; GCN: ds_write_b32 v{{[0-9]+}}

; EG: LDS_READ_RET {{.*}}, [[VAL:T[0-9]+.[XYZW]]]
; EG: LDS_WRITE * [[VAL]]
define void @local_anyext_load_i8(i8 addrspace(3)* nocapture noalias %out, i8 addrspace(3)* nocapture noalias %src) nounwind {
  %cast = bitcast i8 addrspace(3)* %src to i32 addrspace(3)*
  %load = load i32, i32 addrspace(3)* %cast
  %x = bitcast i32 %load to <4 x i8>
  %castOut = bitcast i8 addrspace(3)* %out to <4 x i8> addrspace(3)*
  store <4 x i8> %x, <4 x i8> addrspace(3)* %castOut
  ret void
}

; FUNC-LABEL: {{^}}local_anyext_load_i16:
; GCN: ds_read_b32 v{{[0-9]+}}
; GCN: ds_write_b32 v{{[0-9]+}}

; EG: LDS_READ_RET {{.*}}, [[VAL:T[0-9]+.[XYZW]]]
; EG: LDS_WRITE * [[VAL]]
define void @local_anyext_load_i16(i16 addrspace(3)* nocapture noalias %out, i16 addrspace(3)* nocapture noalias %src) nounwind {
  %cast = bitcast i16 addrspace(3)* %src to i32 addrspace(3)*
  %load = load i32, i32 addrspace(3)* %cast
  %x = bitcast i32 %load to <2 x i16>
  %castOut = bitcast i16 addrspace(3)* %out to <2 x i16> addrspace(3)*
  store <2 x i16> %x, <2 x i16> addrspace(3)* %castOut
  ret void
}
