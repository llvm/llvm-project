; RUN: llc < %s -mtriple=mips -mcpu=mips2 | FileCheck %s -check-prefix=ALL

; Address spaces 1-255 are software defined.
define ptr @cast(ptr %arg) {
  %1 = addrspacecast ptr %arg to ptr addrspace(1)
  %2 = addrspacecast ptr addrspace(1) %1 to ptr addrspace(2)
  %3 = addrspacecast ptr addrspace(2) %2 to ptr addrspace(0)
  ret ptr %3
}

; ALL-LABEL: cast:
; ALL:           move   $2, $4
