; RUN: llc < %s -mtriple=amdgcn -mcpu=tonga -show-mc-encoding -global-isel | FileCheck --check-prefix=GCN %s

; GCN-LABEL: vs_epilog
; GCN: s_endpgm

define amdgpu_vs void @vs_epilog() {
main_body:
  ret void
}
