; RUN: llvm-mc -triple avr -mattr=spm,spmx -show-encoding < %s | FileCheck %s


foo:

  spm
  spm Z+

; CHECK: spm                  ; encoding: [0xe8,0x95]
; CHECK: spm Z+               ; encoding: [0xf8,0x95]
