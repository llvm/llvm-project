; RUN: llc -march=r600 -mcpu=redwood < %s | FileCheck -check-prefix=EG -check-prefix=FUNC %s
; RUN: llc -march=amdgcn -mcpu=verde -verify-machineinstrs < %s | FileCheck -check-prefix=SI -check-prefix=FUNC %s
; RUN: llc -march=amdgcn -mcpu=tonga -verify-machineinstrs < %s | FileCheck -check-prefix=SI -check-prefix=FUNC %s

; FUNC-LABEL: {{^}}test2:
; EG: AND_INT {{\*? *}}T{{[0-9]+\.[XYZW], T[0-9]+\.[XYZW], T[0-9]+\.[XYZW]}}
; EG: AND_INT {{\*? *}}T{{[0-9]+\.[XYZW], T[0-9]+\.[XYZW], T[0-9]+\.[XYZW]}}

; SI: v_and_b32_e32 v{{[0-9]+, v[0-9]+, v[0-9]+}}
; SI: v_and_b32_e32 v{{[0-9]+, v[0-9]+, v[0-9]+}}

define void @test2(<2 x i32> addrspace(1)* %out, <2 x i32> addrspace(1)* %in) {
  %b_ptr = getelementptr <2 x i32> addrspace(1)* %in, i32 1
  %a = load <2 x i32> addrspace(1) * %in
  %b = load <2 x i32> addrspace(1) * %b_ptr
  %result = and <2 x i32> %a, %b
  store <2 x i32> %result, <2 x i32> addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}test4:
; EG: AND_INT {{\** *}}T{{[0-9]+\.[XYZW], T[0-9]+\.[XYZW], T[0-9]+\.[XYZW]}}
; EG: AND_INT {{\** *}}T{{[0-9]+\.[XYZW], T[0-9]+\.[XYZW], T[0-9]+\.[XYZW]}}
; EG: AND_INT {{\** *}}T{{[0-9]+\.[XYZW], T[0-9]+\.[XYZW], T[0-9]+\.[XYZW]}}
; EG: AND_INT {{\** *}}T{{[0-9]+\.[XYZW], T[0-9]+\.[XYZW], T[0-9]+\.[XYZW]}}

; SI: v_and_b32_e32 v{{[0-9]+, v[0-9]+, v[0-9]+}}
; SI: v_and_b32_e32 v{{[0-9]+, v[0-9]+, v[0-9]+}}
; SI: v_and_b32_e32 v{{[0-9]+, v[0-9]+, v[0-9]+}}
; SI: v_and_b32_e32 v{{[0-9]+, v[0-9]+, v[0-9]+}}

define void @test4(<4 x i32> addrspace(1)* %out, <4 x i32> addrspace(1)* %in) {
  %b_ptr = getelementptr <4 x i32> addrspace(1)* %in, i32 1
  %a = load <4 x i32> addrspace(1) * %in
  %b = load <4 x i32> addrspace(1) * %b_ptr
  %result = and <4 x i32> %a, %b
  store <4 x i32> %result, <4 x i32> addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}s_and_i32:
; SI: s_and_b32
define void @s_and_i32(i32 addrspace(1)* %out, i32 %a, i32 %b) {
  %and = and i32 %a, %b
  store i32 %and, i32 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}s_and_constant_i32:
; SI: s_and_b32 s{{[0-9]+}}, s{{[0-9]+}}, 0x12d687
define void @s_and_constant_i32(i32 addrspace(1)* %out, i32 %a) {
  %and = and i32 %a, 1234567
  store i32 %and, i32 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}v_and_i32:
; SI: v_and_b32
define void @v_and_i32(i32 addrspace(1)* %out, i32 addrspace(1)* %aptr, i32 addrspace(1)* %bptr) {
  %a = load i32 addrspace(1)* %aptr, align 4
  %b = load i32 addrspace(1)* %bptr, align 4
  %and = and i32 %a, %b
  store i32 %and, i32 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}v_and_constant_i32:
; SI: v_and_b32
define void @v_and_constant_i32(i32 addrspace(1)* %out, i32 addrspace(1)* %aptr) {
  %a = load i32 addrspace(1)* %aptr, align 4
  %and = and i32 %a, 1234567
  store i32 %and, i32 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}s_and_i64:
; SI: s_and_b64
define void @s_and_i64(i64 addrspace(1)* %out, i64 %a, i64 %b) {
  %and = and i64 %a, %b
  store i64 %and, i64 addrspace(1)* %out, align 8
  ret void
}

; FIXME: Should use SGPRs
; FUNC-LABEL: {{^}}s_and_i1:
; SI: v_and_b32
define void @s_and_i1(i1 addrspace(1)* %out, i1 %a, i1 %b) {
  %and = and i1 %a, %b
  store i1 %and, i1 addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}s_and_constant_i64:
; SI: s_and_b64
define void @s_and_constant_i64(i64 addrspace(1)* %out, i64 %a) {
  %and = and i64 %a, 281474976710655
  store i64 %and, i64 addrspace(1)* %out, align 8
  ret void
}

; FUNC-LABEL: {{^}}v_and_i64:
; SI: v_and_b32
; SI: v_and_b32
define void @v_and_i64(i64 addrspace(1)* %out, i64 addrspace(1)* %aptr, i64 addrspace(1)* %bptr) {
  %a = load i64 addrspace(1)* %aptr, align 8
  %b = load i64 addrspace(1)* %bptr, align 8
  %and = and i64 %a, %b
  store i64 %and, i64 addrspace(1)* %out, align 8
  ret void
}

; FUNC-LABEL: {{^}}v_and_i64_br:
; SI: v_and_b32
; SI: v_and_b32
define void @v_and_i64_br(i64 addrspace(1)* %out, i64 addrspace(1)* %aptr, i64 addrspace(1)* %bptr, i32 %cond) {
entry:
  %tmp0 = icmp eq i32 %cond, 0
  br i1 %tmp0, label %if, label %endif

if:
  %a = load i64 addrspace(1)* %aptr, align 8
  %b = load i64 addrspace(1)* %bptr, align 8
  %and = and i64 %a, %b
  br label %endif

endif:
  %tmp1 = phi i64 [%and, %if], [0, %entry]
  store i64 %tmp1, i64 addrspace(1)* %out, align 8
  ret void
}

; FUNC-LABEL: {{^}}v_and_constant_i64:
; SI: v_and_b32_e32 {{v[0-9]+}}, {{s[0-9]+}}, {{v[0-9]+}}
; SI: v_and_b32_e32 {{v[0-9]+}}, {{s[0-9]+}}, {{v[0-9]+}}
define void @v_and_constant_i64(i64 addrspace(1)* %out, i64 addrspace(1)* %aptr) {
  %a = load i64 addrspace(1)* %aptr, align 8
  %and = and i64 %a, 1234567
  store i64 %and, i64 addrspace(1)* %out, align 8
  ret void
}

; FIXME: Replace and 0 with mov 0
; FUNC-LABEL: {{^}}v_and_inline_imm_i64:
; SI: v_and_b32_e32 {{v[0-9]+}}, 64, {{v[0-9]+}}
; SI: v_and_b32_e32 {{v[0-9]+}}, 0, {{v[0-9]+}}
define void @v_and_inline_imm_i64(i64 addrspace(1)* %out, i64 addrspace(1)* %aptr) {
  %a = load i64 addrspace(1)* %aptr, align 8
  %and = and i64 %a, 64
  store i64 %and, i64 addrspace(1)* %out, align 8
  ret void
}

; FUNC-LABEL: {{^}}s_and_inline_imm_i64:
; SI: s_and_b64 s{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 64
define void @s_and_inline_imm_i64(i64 addrspace(1)* %out, i64 addrspace(1)* %aptr, i64 %a) {
  %and = and i64 %a, 64
  store i64 %and, i64 addrspace(1)* %out, align 8
  ret void
}
