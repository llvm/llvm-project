; RUN: llc < %s -march=amdgcn -mcpu=tonga -show-mc-encoding -verify-machineinstrs -global-isel | FileCheck --check-prefix=GCN %s

; REQUIRES: global-isel

; GCN-LABEL: vs_epilog
; GCN: s_endpgm

define amdgpu_vs void @vs_epilog() {
main_body:
  ret void
}
