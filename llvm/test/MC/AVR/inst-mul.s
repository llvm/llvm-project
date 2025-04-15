; RUN: llvm-mc -triple avr -mattr=mul -show-encoding < %s | FileCheck %s
; RUN: llvm-mc -filetype=obj -triple avr -mattr=mul < %s \
; RUN:     | llvm-objdump -dr --mattr=mul - | FileCheck --check-prefix=INST %s

foo:
  mul r0,  r15
  mul r15, r0
  mul r16, r31
  mul r31, r16

; CHECK: mul r0,  r15               ; encoding: [0x0f,0x9c]
; CHECK: mul r15, r0                ; encoding: [0xf0,0x9c]
; CHECK: mul r16, r31               ; encoding: [0x0f,0x9f]
; CHECK: mul r31, r16               ; encoding: [0xf0,0x9f]

; INST:  mul r0,  r15
; INST:  mul r15, r0
; INST:  mul r16, r31
; INST:  mul r31, r16
