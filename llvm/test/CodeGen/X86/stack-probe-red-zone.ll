; RUN: llc -mtriple=x86_64-pc-linux-gnu < %s -o - | FileCheck %s

; Ensure that red zone usage occurs.
define signext i8 @testStackProbesOff() {
  %array = alloca [40096 x i8], align 16
  ret i8 0

; CHECK-LABEL:  testStackProbesOff:
; CHECK:        subq $39976, %rsp # imm = 0x9C28
}

; Ensure stack probes do not result in red zone usage.
define void @testStackProbesOn() "probe-stack"="__probestack" {
  %array = alloca [40096 x i8], align 16
  ret void

; CHECK-LABEL:  testStackProbesOn:
; CHECK:        movl $40104, %eax # imm = 0x9CA8
; CHECK-NEXT:   callq __probestack
; CHECK-NEXT:   subq %rax, %rsp
}
