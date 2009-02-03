; RUN: llvm-as < %s | llvm-dis | grep 0x36A0000000000000
@A = global float 0x36A0000000000000            ; <float*> [#uses=0]
