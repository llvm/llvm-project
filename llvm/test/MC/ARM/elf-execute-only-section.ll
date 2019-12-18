; RUN: llc < %s -mtriple=thumbv8m.base-eabi -mattr=+execute-only -filetype=obj %s -o - | \
; RUN: llvm-readelf -S | FileCheck %s
; RUN: llc < %s -mtriple=thumbv8m.main-eabi -mattr=+execute-only -filetype=obj %s -o - | \
; RUN: llvm-readelf -S | FileCheck %s
; RUN: llc < %s -mtriple=thumbv7m-eabi -mattr=+execute-only -filetype=obj %s -o - | \
; RUN: llvm-readelf -S | FileCheck %s

; CHECK-NOT: {{.text[ ]+PROGBITS[ ]+[0-9]+ [0-9]+ [0-9]+ [0-9]+ AX[^y]}}
; CHECK: {{.text[ ]+PROGBITS[ ]+[0-9]+ [0-9]+ [0-9]+ [0-9]+ AXy}}
define void @test_func() {
entry:
  ret void
}
