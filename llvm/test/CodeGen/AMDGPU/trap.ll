; RUN: llc -mtriple=amdgcn--amdhsa -verify-machineinstrs < %s 2>&1 | FileCheck -check-prefix=GCN %s

declare void @llvm.trap() #0

; GCN-LABEL: {{^}}trap:
; GCN: v_mov_b32_e32 v0, 1
; GCN: s_mov_b64 s[0:1], s[4:5]
; GCN: s_trap 1
define void @trap() {
  call void @llvm.trap()
  ret void
}

attributes #0 = { nounwind noreturn }
