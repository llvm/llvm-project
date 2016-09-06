; RUN: llc -march=amdgcn -verify-machineinstrs < %s | FileCheck -check-prefix=SI -check-prefix=FUNC %s
; RUN: llc -march=amdgcn -mcpu=tonga -verify-machineinstrs < %s | FileCheck -check-prefix=SI -check-prefix=FUNC %s

; FUNC-LABEL: {{^}}extract_vector_elt_v2i16:
; GCN: buffer_load_ushort
; GCN: buffer_load_ushort
; GCN: buffer_store_short
; GCN: buffer_store_short
define void @extract_vector_elt_v2i16(i16 addrspace(1)* %out, <2 x i16> %foo) #0 {
  %p0 = extractelement <2 x i16> %foo, i32 0
  %p1 = extractelement <2 x i16> %foo, i32 1
  %out1 = getelementptr i16, i16 addrspace(1)* %out, i32 10
  store i16 %p1, i16 addrspace(1)* %out, align 2
  store i16 %p0, i16 addrspace(1)* %out1, align 2
  ret void
}

; FUNC-LABEL: {{^}}extract_vector_elt_v3i16:
; GCN: buffer_load_ushort
; GCN: buffer_store_short
; GCN: buffer_store_short
define void @extract_vector_elt_v3i16(i16 addrspace(1)* %out, <3 x i16> %foo) #0 {
  %p0 = extractelement <3 x i16> %foo, i32 0
  %p1 = extractelement <3 x i16> %foo, i32 2
  %out1 = getelementptr i16, i16 addrspace(1)* %out, i32 1
  store i16 %p1, i16 addrspace(1)* %out, align 2
  store i16 %p0, i16 addrspace(1)* %out1, align 2
  ret void
}

; FUNC-LABEL: {{^}}extract_vector_elt_v4i16:
; GCN: buffer_load_ushort
; GCN: buffer_load_ushort
; GCN: buffer_store_short
; GCN: buffer_store_short
define void @extract_vector_elt_v4i16(i16 addrspace(1)* %out, <4 x i16> %foo) #0 {
  %p0 = extractelement <4 x i16> %foo, i32 0
  %p1 = extractelement <4 x i16> %foo, i32 2
  %out1 = getelementptr i16, i16 addrspace(1)* %out, i32 10
  store i16 %p1, i16 addrspace(1)* %out, align 2
  store i16 %p0, i16 addrspace(1)* %out1, align 2
  ret void
}


; FUNC-LABEL: {{^}}dynamic_extract_vector_elt_v3i16:
; GCN: buffer_load_ushort
; GCN: buffer_load_ushort
; GCN: buffer_load_ushort

; GCN: buffer_store_short
; GCN: buffer_store_short
; GCN: buffer_store_short

; GCN: buffer_store_short
; GCN: buffer_load_ushort
; GCN: buffer_store_short
define void @dynamic_extract_vector_elt_v3i16(i16 addrspace(1)* %out, <3 x i16> %foo, i32 %idx) #0 {
  %p0 = extractelement <3 x i16> %foo, i32 %idx
  %out1 = getelementptr i16, i16 addrspace(1)* %out, i32 1
  store i16 %p0, i16 addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}dynamic_extract_vector_elt_v4i16:
; GCN: buffer_load_ushort
; GCN: buffer_load_ushort
; GCN: buffer_load_ushort
; GCN: buffer_load_ushort

; GCN: buffer_store_short
; GCN: buffer_store_short
; GCN: buffer_store_short
; GCN: buffer_store_short

; GCN: buffer_store_short
; GCN: buffer_load_ushort
; GCN: buffer_store_short
define void @dynamic_extract_vector_elt_v4i16(i16 addrspace(1)* %out, <4 x i16> %foo, i32 %idx) #0 {
  %p0 = extractelement <4 x i16> %foo, i32 %idx
  %out1 = getelementptr i16, i16 addrspace(1)* %out, i32 1
  store i16 %p0, i16 addrspace(1)* %out
  ret void
}

attributes #0 = { nounwind }
