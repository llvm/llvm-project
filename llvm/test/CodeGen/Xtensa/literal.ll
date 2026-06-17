; RUN: llc -mtriple=xtensa -function-sections --mcpu=esp32 --filetype=obj < %s\
; RUN: | llvm-objdump -r -s --triple=xtensa --mcpu=esp32 - | FileCheck %s

; CHECK-LABEL: Contents of section .literal.func:
; CHECK:       0000 88130000                             ....

define i32 @func() {
  ret i32 5000
}
