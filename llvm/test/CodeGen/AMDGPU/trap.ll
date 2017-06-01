; RUN: llc -march=amdgcn -verify-machineinstrs < %s 2>&1 | FileCheck -check-prefix=GCN %s

; GCN: warning: <unknown>:0:0: in function trap void (): trap handler not supported

declare void @llvm.trap() #0

; GCN-LABEL: {{^}}trap:
; GCN: s_endpgm
; GCN-NEXT: s_endpgm
define void @trap() {
  call void @llvm.trap()
  ret void
}

attributes #0 = { nounwind noreturn }
