; RUN: llc -march=amdgcn -mcpu=tahiti < %s | FileCheck -check-prefix=SI -check-prefix=FUNC %s
; RUN: llc -march=amdgcn -mcpu=tonga < %s | FileCheck -check-prefix=SI -check-prefix=FUNC %s

; FUNC-LABEL: {{^}}test_copy_v4i8:
; SI: buffer_load_dword [[REG:v[0-9]+]]
; SI: buffer_store_dword [[REG]]
; SI: s_endpgm
define void @test_copy_v4i8(<4 x i8> addrspace(1)* %out, <4 x i8> addrspace(1)* %in) nounwind {
  %val = load <4 x i8>, <4 x i8> addrspace(1)* %in, align 4
  store <4 x i8> %val, <4 x i8> addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}test_copy_v4i8_x2:
; SI: buffer_load_dword [[REG:v[0-9]+]]
; SI: buffer_store_dword [[REG]]
; SI: buffer_store_dword [[REG]]
; SI: s_endpgm
define void @test_copy_v4i8_x2(<4 x i8> addrspace(1)* %out0, <4 x i8> addrspace(1)* %out1, <4 x i8> addrspace(1)* %in) nounwind {
  %val = load <4 x i8>, <4 x i8> addrspace(1)* %in, align 4
  store <4 x i8> %val, <4 x i8> addrspace(1)* %out0, align 4
  store <4 x i8> %val, <4 x i8> addrspace(1)* %out1, align 4
  ret void
}

; FUNC-LABEL: {{^}}test_copy_v4i8_x3:
; SI: buffer_load_dword [[REG:v[0-9]+]]
; SI: buffer_store_dword [[REG]]
; SI: buffer_store_dword [[REG]]
; SI: buffer_store_dword [[REG]]
; SI: s_endpgm
define void @test_copy_v4i8_x3(<4 x i8> addrspace(1)* %out0, <4 x i8> addrspace(1)* %out1, <4 x i8> addrspace(1)* %out2, <4 x i8> addrspace(1)* %in) nounwind {
  %val = load <4 x i8>, <4 x i8> addrspace(1)* %in, align 4
  store <4 x i8> %val, <4 x i8> addrspace(1)* %out0, align 4
  store <4 x i8> %val, <4 x i8> addrspace(1)* %out1, align 4
  store <4 x i8> %val, <4 x i8> addrspace(1)* %out2, align 4
  ret void
}

; FUNC-LABEL: {{^}}test_copy_v4i8_x4:
; SI: buffer_load_dword [[REG:v[0-9]+]]
; SI: buffer_store_dword [[REG]]
; SI: buffer_store_dword [[REG]]
; SI: buffer_store_dword [[REG]]
; SI: buffer_store_dword [[REG]]
; SI: s_endpgm
define void @test_copy_v4i8_x4(<4 x i8> addrspace(1)* %out0, <4 x i8> addrspace(1)* %out1, <4 x i8> addrspace(1)* %out2, <4 x i8> addrspace(1)* %out3, <4 x i8> addrspace(1)* %in) nounwind {
  %val = load <4 x i8>, <4 x i8> addrspace(1)* %in, align 4
  store <4 x i8> %val, <4 x i8> addrspace(1)* %out0, align 4
  store <4 x i8> %val, <4 x i8> addrspace(1)* %out1, align 4
  store <4 x i8> %val, <4 x i8> addrspace(1)* %out2, align 4
  store <4 x i8> %val, <4 x i8> addrspace(1)* %out3, align 4
  ret void
}

; FUNC-LABEL: {{^}}test_copy_v4i8_extra_use:
; SI: buffer_load_ubyte
; SI: buffer_load_ubyte
; SI: buffer_load_ubyte
; SI: buffer_load_ubyte
; SI-DAG: v_add
; SI-DAG: v_add
; SI-DAG: v_add
; SI-DAG: v_add
; SI-DAG: buffer_store_byte
; SI-DAG: buffer_store_byte
; SI-DAG: buffer_store_byte
; SI-DAG: buffer_store_byte
; SI-DAG: buffer_store_byte
; SI-DAG: buffer_store_byte
; SI-DAG: buffer_store_byte
; SI-DAG: buffer_store_byte

; After scalarizing v4i8 loads is fixed.
; XSI: buffer_load_dword
; XSI: V_BFE
; XSI: V_ADD
; XSI: V_ADD
; XSI: V_ADD
; XSI: buffer_store_dword
; XSI: buffer_store_dword

; SI: s_endpgm
define void @test_copy_v4i8_extra_use(<4 x i8> addrspace(1)* %out0, <4 x i8> addrspace(1)* %out1, <4 x i8> addrspace(1)* %in) nounwind {
  %val = load <4 x i8>, <4 x i8> addrspace(1)* %in, align 4
  %add = add <4 x i8> %val, <i8 9, i8 9, i8 9, i8 9>
  store <4 x i8> %val, <4 x i8> addrspace(1)* %out0, align 4
  store <4 x i8> %add, <4 x i8> addrspace(1)* %out1, align 4
  ret void
}

; FUNC-LABEL: {{^}}test_copy_v4i8_x2_extra_use:
; SI: buffer_load_ubyte
; SI: buffer_load_ubyte
; SI: buffer_load_ubyte
; SI: buffer_load_ubyte
; SI-DAG: v_add
; SI-DAG: v_add
; SI-DAG: v_add
; SI-DAG: v_add
; SI-DAG: buffer_store_byte
; SI-DAG: buffer_store_byte
; SI-DAG: buffer_store_byte
; SI-DAG: buffer_store_byte
; SI-DAG: buffer_store_byte
; SI-DAG: buffer_store_byte
; SI-DAG: buffer_store_byte
; SI-DAG: buffer_store_byte
; SI-DAG: buffer_store_byte
; SI-DAG: buffer_store_byte
; SI-DAG: buffer_store_byte
; SI-DAG: buffer_store_byte

; XSI: buffer_load_dword
; XSI: BFE
; XSI: buffer_store_dword
; XSI: V_ADD
; XSI: buffer_store_dword
; XSI-NEXT: buffer_store_dword

; SI: s_endpgm
define void @test_copy_v4i8_x2_extra_use(<4 x i8> addrspace(1)* %out0, <4 x i8> addrspace(1)* %out1, <4 x i8> addrspace(1)* %out2, <4 x i8> addrspace(1)* %in) nounwind {
  %val = load <4 x i8>, <4 x i8> addrspace(1)* %in, align 4
  %add = add <4 x i8> %val, <i8 9, i8 9, i8 9, i8 9>
  store <4 x i8> %val, <4 x i8> addrspace(1)* %out0, align 4
  store <4 x i8> %add, <4 x i8> addrspace(1)* %out1, align 4
  store <4 x i8> %val, <4 x i8> addrspace(1)* %out2, align 4
  ret void
}

; FUNC-LABEL: {{^}}test_copy_v3i8:
; SI-NOT: bfe
; SI-NOT: bfi
; SI: s_endpgm
define void @test_copy_v3i8(<3 x i8> addrspace(1)* %out, <3 x i8> addrspace(1)* %in) nounwind {
  %val = load <3 x i8>, <3 x i8> addrspace(1)* %in, align 4
  store <3 x i8> %val, <3 x i8> addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}test_copy_v4i8_volatile_load:
; SI: buffer_load_ubyte
; SI: buffer_load_ubyte
; SI: buffer_load_ubyte
; SI: buffer_load_ubyte
; SI: s_endpgm
define void @test_copy_v4i8_volatile_load(<4 x i8> addrspace(1)* %out, <4 x i8> addrspace(1)* %in) nounwind {
  %val = load volatile <4 x i8>, <4 x i8> addrspace(1)* %in, align 4
  store <4 x i8> %val, <4 x i8> addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}test_copy_v4i8_volatile_store:
; SI: buffer_load_ubyte
; SI: buffer_load_ubyte
; SI: buffer_load_ubyte
; SI: buffer_load_ubyte
; SI: buffer_store_byte
; SI: buffer_store_byte
; SI: buffer_store_byte
; SI: buffer_store_byte
; SI: s_endpgm
define void @test_copy_v4i8_volatile_store(<4 x i8> addrspace(1)* %out, <4 x i8> addrspace(1)* %in) nounwind {
  %val = load <4 x i8>, <4 x i8> addrspace(1)* %in, align 4
  store volatile <4 x i8> %val, <4 x i8> addrspace(1)* %out, align 4
  ret void
}
