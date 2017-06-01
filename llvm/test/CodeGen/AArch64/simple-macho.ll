; RUN: llc -mtriple=arm64-macho -o - %s | FileCheck %s
; RUN: llc -mtriple=arm64-macho -filetype=obj -o %t %s
; RUN: llvm-objdump -triple=arm64-macho -d %t | FileCheck --check-prefix=CHECK-OBJ %s

define void @foo() {
; CHECK-LABEL: _foo:
; CHECK: ret

; CHECK-OBJ: 0: c0 03 5f d6 ret

  ret void
}
