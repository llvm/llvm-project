; RUN: llc -mtriple=hexagon -filetype=obj < %s | llvm-objdump --no-print-imm-hex -d -r - | FileCheck %s

declare void @bar()

define void @foo() {
call void @bar()
ret void
}

; CHECK: { call 0
; CHECK: 00000000:  R_HEX_B22_PCREL
; CHECK:   allocframe(#0)
; CHECK: { dealloc_return }
