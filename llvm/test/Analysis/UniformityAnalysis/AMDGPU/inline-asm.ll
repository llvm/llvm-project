; RUN: opt -mtriple amdgcn-unknown-amdhsa -mcpu=tahiti -passes='print<uniformity>' -disable-output %s 2>&1 | FileCheck %s
; RUN: opt -mtriple amdgcn-unknown-amdhsa -mcpu=gfx908 -passes='print<uniformity>' -disable-output %s 2>&1 | FileCheck %s
; Make sure nothing crashes on targets with or without AGPRs

; CHECK-LABEL: for function 'inline_asm_1_sgpr_virtreg_output':
; CHECK-NOT: DIVERGENT
define i32 @inline_asm_1_sgpr_virtreg_output() {
  %sgpr = call i32 asm "s_mov_b32 $0, 0", "=s"()
  ret i32 %sgpr
}

; CHECK-LABEL: for function 'inline_asm_1_sgpr_physreg_output':
; CHECK-NOT: DIVERGENT
define i32 @inline_asm_1_sgpr_physreg_output() {
  %sgpr = call i32 asm "s_mov_b32 s0, 0", "={s0}"()
  ret i32 %sgpr
}

; CHECK-LABEL: for function 'inline_asm_1_vgpr_virtreg_output':
; CHECK: DIVERGENT: %vgpr = call i32 asm "v_mov_b32 $0, 0", "=v"()
define i32 @inline_asm_1_vgpr_virtreg_output() {
  %vgpr = call i32 asm "v_mov_b32 $0, 0", "=v"()
  ret i32 %vgpr
}

; CHECK-LABEL: for function 'inline_asm_1_vgpr_physreg_output':
; CHECK: DIVERGENT: %vgpr = call i32 asm "v_mov_b32 v0, 0", "={v0}"()
define i32 @inline_asm_1_vgpr_physreg_output() {
  %vgpr = call i32 asm "v_mov_b32 v0, 0", "={v0}"()
  ret i32 %vgpr
}

; CHECK-LABEL: for function 'inline_asm_1_agpr_virtreg_output':
; CHECK: DIVERGENT: %vgpr = call i32 asm "; def $0", "=a"()
define i32 @inline_asm_1_agpr_virtreg_output() {
  %vgpr = call i32 asm "; def $0", "=a"()
  ret i32 %vgpr
}

; CHECK-LABEL: for function 'inline_asm_1_agpr_physreg_output':
; CHECK: DIVERGENT: %vgpr = call i32 asm "; def a0", "={a0}"()
define i32 @inline_asm_1_agpr_physreg_output() {
  %vgpr = call i32 asm "; def a0", "={a0}"()
  ret i32 %vgpr
}

; CHECK-LABEL: for function 'inline_asm_2_sgpr_virtreg_output':
; CHECK-NOT: DIVERGENT
define void @inline_asm_2_sgpr_virtreg_output() {
  %asm = call { i32, i32 } asm "; def $0, $1", "=s,=s"()
  %sgpr0 = extractvalue { i32, i32 } %asm, 0
  %sgpr1 = extractvalue { i32, i32 } %asm, 1
  store i32 %sgpr0, ptr addrspace(1) undef
  store i32 %sgpr1, ptr addrspace(1) undef
  ret void
}

; One output is SGPR, one is VGPR. Infer divergent for the aggregate, but uniform on the SGPR extract
; CHECK-LABEL: for function 'inline_asm_sgpr_vgpr_virtreg_output':
; CHECK: DIVERGENT:       %asm = call { i32, i32 } asm "; def $0, $1", "=s,=v"()
; CHECK-NEXT: {{^[ \t]+}}%sgpr = extractvalue { i32, i32 } %asm, 0
; CHECK-NEXT: DIVERGENT:       %vgpr = extractvalue { i32, i32 } %asm, 1
define void @inline_asm_sgpr_vgpr_virtreg_output() {
  %asm = call { i32, i32 } asm "; def $0, $1", "=s,=v"()
  %sgpr = extractvalue { i32, i32 } %asm, 0
  %vgpr = extractvalue { i32, i32 } %asm, 1
  store i32 %sgpr, ptr addrspace(1) undef
  store i32 %vgpr, ptr addrspace(1) undef
  ret void
}

; CHECK-LABEL: for function 'inline_asm_vgpr_sgpr_virtreg_output':
; CHECK: DIVERGENT:       %asm = call { i32, i32 } asm "; def $0, $1", "=v,=s"()
; CHECK-NEXT: DIVERGENT:       %vgpr = extractvalue { i32, i32 } %asm, 0
; CHECK-NEXT: {{^[ \t]+}}%sgpr = extractvalue { i32, i32 } %asm, 1
define void @inline_asm_vgpr_sgpr_virtreg_output() {
  %asm = call { i32, i32 } asm "; def $0, $1", "=v,=s"()
  %vgpr = extractvalue { i32, i32 } %asm, 0
  %sgpr = extractvalue { i32, i32 } %asm, 1
  store i32 %vgpr, ptr addrspace(1) undef
  store i32 %sgpr, ptr addrspace(1) undef
  ret void
}

; Have an extra output constraint
; CHECK-LABEL: for function 'multi_sgpr_inline_asm_output_input_constraint':
; CHECK-NOT: DIVERGENT
define void @multi_sgpr_inline_asm_output_input_constraint() {
  %asm = call { i32, i32 } asm "; def $0, $1", "=s,=s,s"(i32 1234)
  %sgpr0 = extractvalue { i32, i32 } %asm, 0
  %sgpr1 = extractvalue { i32, i32 } %asm, 1
  store i32 %sgpr0, ptr addrspace(1) undef
  store i32 %sgpr1, ptr addrspace(1) undef
  ret void
}

; CHECK-LABEL: for function 'inline_asm_vgpr_sgpr_virtreg_output_input_constraint':
; CHECK: DIVERGENT:       %asm = call { i32, i32 } asm "; def $0, $1", "=v,=s,v"(i32 1234)
; CHECK-NEXT: DIVERGENT:       %vgpr = extractvalue { i32, i32 } %asm, 0
; CHECK-NEXT: {{^[ \t]+}}%sgpr = extractvalue { i32, i32 } %asm, 1
define void @inline_asm_vgpr_sgpr_virtreg_output_input_constraint() {
  %asm = call { i32, i32 } asm "; def $0, $1", "=v,=s,v"(i32 1234)
  %vgpr = extractvalue { i32, i32 } %asm, 0
  %sgpr = extractvalue { i32, i32 } %asm, 1
  store i32 %vgpr, ptr addrspace(1) undef
  store i32 %sgpr, ptr addrspace(1) undef
  ret void
}
