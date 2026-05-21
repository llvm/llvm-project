; RUN: llvm-as < %s | llvm-dis | FileCheck %s

@x = global fp128 0xL01
; CHECK: @x = global fp128 f0x00000000000000010000000000000000
