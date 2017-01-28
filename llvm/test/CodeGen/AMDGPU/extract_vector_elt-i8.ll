; RUN: llc -march=amdgcn -verify-machineinstrs < %s | FileCheck -check-prefix=SI -check-prefix=FUNC %s
; RUN: llc -march=amdgcn -mcpu=tonga -mattr=-flat-for-global -verify-machineinstrs < %s | FileCheck -check-prefix=SI -check-prefix=FUNC %s

; FUNC-LABEL: {{^}}extract_vector_elt_v1i8:
; GCN: buffer_load_ubyte
; GCN: buffer_store_byte
define void @extract_vector_elt_v1i8(i8 addrspace(1)* %out, <1 x i8> %foo) #0 {
  %p0 = extractelement <1 x i8> %foo, i32 0
  store i8 %p0, i8 addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}extract_vector_elt_v2i8:
; GCN: buffer_load_ubyte
; GCN: buffer_load_ubyte
; GCN: buffer_store_byte
; GCN: buffer_store_byte
define void @extract_vector_elt_v2i8(i8 addrspace(1)* %out, <2 x i8> %foo) #0 {
  %p0 = extractelement <2 x i8> %foo, i32 0
  %p1 = extractelement <2 x i8> %foo, i32 1
  %out1 = getelementptr i8, i8 addrspace(1)* %out, i32 1
  store i8 %p1, i8 addrspace(1)* %out
  store i8 %p0, i8 addrspace(1)* %out1
  ret void
}

; FUNC-LABEL: {{^}}extract_vector_elt_v3i8:
; GCN: buffer_load_ubyte
; GCN: buffer_load_ubyte
; GCN: buffer_store_byte
; GCN: buffer_store_byte
define void @extract_vector_elt_v3i8(i8 addrspace(1)* %out, <3 x i8> %foo) #0 {
  %p0 = extractelement <3 x i8> %foo, i32 0
  %p1 = extractelement <3 x i8> %foo, i32 2
  %out1 = getelementptr i8, i8 addrspace(1)* %out, i32 1
  store i8 %p1, i8 addrspace(1)* %out
  store i8 %p0, i8 addrspace(1)* %out1
  ret void
}

; FUNC-LABEL: {{^}}extract_vector_elt_v4i8:
; GCN: buffer_load_ubyte
; GCN: buffer_load_ubyte
; GCN: buffer_store_byte
; GCN: buffer_store_byte
define void @extract_vector_elt_v4i8(i8 addrspace(1)* %out, <4 x i8> %foo) #0 {
  %p0 = extractelement <4 x i8> %foo, i32 0
  %p1 = extractelement <4 x i8> %foo, i32 2
  %out1 = getelementptr i8, i8 addrspace(1)* %out, i32 1
  store i8 %p1, i8 addrspace(1)* %out
  store i8 %p0, i8 addrspace(1)* %out1
  ret void
}

; FUNC-LABEL: {{^}}extract_vector_elt_v8i8:
; GCN: buffer_load_ubyte
; GCN: buffer_load_ubyte
; GCN: buffer_store_byte
; GCN: buffer_store_byte
define void @extract_vector_elt_v8i8(i8 addrspace(1)* %out, <8 x i8> %foo) #0 {
  %p0 = extractelement <8 x i8> %foo, i32 0
  %p1 = extractelement <8 x i8> %foo, i32 2
  %out1 = getelementptr i8, i8 addrspace(1)* %out, i32 1
  store i8 %p1, i8 addrspace(1)* %out
  store i8 %p0, i8 addrspace(1)* %out1
  ret void
}

; FUNC-LABEL: {{^}}extract_vector_elt_v16i8:
; GCN: buffer_load_ubyte
; GCN: buffer_load_ubyte
; GCN: buffer_store_byte
; GCN: buffer_store_byte
define void @extract_vector_elt_v16i8(i8 addrspace(1)* %out, <16 x i8> %foo) #0 {
  %p0 = extractelement <16 x i8> %foo, i32 0
  %p1 = extractelement <16 x i8> %foo, i32 2
  %out1 = getelementptr i8, i8 addrspace(1)* %out, i32 1
  store i8 %p1, i8 addrspace(1)* %out
  store i8 %p0, i8 addrspace(1)* %out1
  ret void
}

; FUNC-LABEL: {{^}}extract_vector_elt_v32i8:
; GCN: buffer_load_ubyte
; GCN: buffer_load_ubyte
; GCN: buffer_store_byte
; GCN: buffer_store_byte
define void @extract_vector_elt_v32i8(i8 addrspace(1)* %out, <32 x i8> %foo) #0 {
  %p0 = extractelement <32 x i8> %foo, i32 0
  %p1 = extractelement <32 x i8> %foo, i32 2
  %out1 = getelementptr i8, i8 addrspace(1)* %out, i32 1
  store i8 %p1, i8 addrspace(1)* %out
  store i8 %p0, i8 addrspace(1)* %out1
  ret void
}

; FUNC-LABEL: {{^}}extract_vector_elt_v64i8:
; GCN: buffer_load_ubyte
; GCN: buffer_load_ubyte
; GCN: buffer_store_byte
; GCN: buffer_store_byte
define void @extract_vector_elt_v64i8(i8 addrspace(1)* %out, <64 x i8> %foo) #0 {
  %p0 = extractelement <64 x i8> %foo, i32 0
  %p1 = extractelement <64 x i8> %foo, i32 2
  %out1 = getelementptr i8, i8 addrspace(1)* %out, i32 1
  store i8 %p1, i8 addrspace(1)* %out
  store i8 %p0, i8 addrspace(1)* %out1
  ret void
}

; FUNC-LABEL: {{^}}dynamic_extract_vector_elt_v3i8:
; GCN: buffer_load_ubyte
; GCN: buffer_load_ubyte
; GCN: buffer_load_ubyte

; GCN: buffer_store_byte
; GCN: buffer_store_byte
; GCN: buffer_store_byte

; GCN: buffer_store_byte
; GCN: buffer_load_ubyte
; GCN: buffer_store_byte
define void @dynamic_extract_vector_elt_v3i8(i8 addrspace(1)* %out, <3 x i8> %foo, i32 %idx) #0 {
  %p0 = extractelement <3 x i8> %foo, i32 %idx
  %out1 = getelementptr i8, i8 addrspace(1)* %out, i32 1
  store i8 %p0, i8 addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}dynamic_extract_vector_elt_v4i8:
; GCN: buffer_load_ubyte
; GCN: buffer_load_ubyte
; GCN: buffer_load_ubyte
; GCN: buffer_load_ubyte

; GCN: buffer_store_byte
; GCN: buffer_store_byte
; GCN: buffer_store_byte
; GCN: buffer_store_byte

; GCN: buffer_store_byte
; GCN: buffer_load_ubyte
; GCN: buffer_store_byte
define void @dynamic_extract_vector_elt_v4i8(i8 addrspace(1)* %out, <4 x i8> %foo, i32 %idx) #0 {
  %p0 = extractelement <4 x i8> %foo, i32 %idx
  %out1 = getelementptr i8, i8 addrspace(1)* %out, i32 1
  store i8 %p0, i8 addrspace(1)* %out
  ret void
}

attributes #0 = { nounwind }
