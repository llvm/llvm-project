; RUN: llc < %s -march=amdgcn -mcpu=SI -verify-machineinstrs | FileCheck %s
; RUN: llc < %s -march=amdgcn -mcpu=tonga -verify-machineinstrs | FileCheck %s

; CHECK: {{^}}inline_asm:
; CHECK: s_endpgm
; CHECK: s_endpgm
define void @inline_asm(i32 addrspace(1)* %out) {
entry:
  store i32 5, i32 addrspace(1)* %out
  call void asm sideeffect "s_endpgm", ""()
  ret void
}

; CHECK: {{^}}inline_asm_shader:
; CHECK: s_endpgm
; CHECK: s_endpgm
define void @inline_asm_shader() #0 {
entry:
  call void asm sideeffect "s_endpgm", ""()
  ret void
}

attributes #0 = { "ShaderType"="0" }


; CHECK: {{^}}branch_on_asm:
; Make sure inline assembly is treted as divergent.
; CHECK: s_mov_b32 s{{[0-9]+}}, 0
; CHECK: s_and_saveexec_b64
define void @branch_on_asm(i32 addrspace(1)* %out) {
	%zero = call i32 asm "s_mov_b32 $0, 0", "=s"()
	%cmp = icmp eq i32 %zero, 0
	br i1 %cmp, label %if, label %endif

if:
	store i32 0, i32 addrspace(1)* %out
	br label %endif

endif:
  ret void
}
