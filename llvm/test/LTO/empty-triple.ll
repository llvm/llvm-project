; RUN: llvm-as < %s >%t1
; RUN: llvm-lto -exported-symbol=main -filetype=asm -o - %t1  2>&1 | FileCheck %s

; The test is to check the triple is set to default one when it's empty.
; Otherwise, an error will be raised by llvm-lto.

; CHECK-LABEL: main
; CHECK-NOT: error
define void @main() {
entry:
  ret void
}
