; RUN: llvm-mc -triple avr -mattr=spm,spmx -show-encoding < %s | FileCheck %s
; RUN: llvm-mc -filetype=obj -triple avr -mattr=spm,spmx < %s | llvm-objdump -d --mattr=spm,spmx - | FileCheck -check-prefix=CHECK-INST %s


foo:

  spm
  spm Z+

; CHECK: spm                  ; encoding: [0xe8,0x95]
; CHECK: spm Z+               ; encoding: [0xf8,0x95]

; CHECK-INST: spm
; CHECK-INST: spm Z+
