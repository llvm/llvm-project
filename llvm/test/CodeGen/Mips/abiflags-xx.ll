; RUN: llc -filetype=asm -mtriple mipsel-unknown-linux -mcpu=mips32 -mattr=fpxx %s -o - | FileCheck %s

; CHECK: .nan    legacy
; CHECK: .module fp=xx

