; RUN: llc < %s -march=mips -mcpu=mips2 | FileCheck %s -check-prefix=ALL

; Address spaces 1-255 are software defined.
define i32* @cast(i32 *%arg) {
  %1 = addrspacecast i32* %arg to i32 addrspace(1)*
  %2 = addrspacecast i32 addrspace(1)* %1 to i32 addrspace(2)*
  %3 = addrspacecast i32 addrspace(2)* %2 to i32 addrspace(0)*
  ret i32* %3
}

; ALL-LABEL: cast:
; ALL:           move   $2, $4
