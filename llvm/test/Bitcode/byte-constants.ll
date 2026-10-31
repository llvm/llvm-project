; RUN: llvm-as < %s | llvm-dis | FileCheck %s

; CHECK: @byte.highbit = constant b8 -1
@byte.highbit = constant b8 255

; CHECK: @byte.vector.highbit = constant <2 x b8> <b8 -128, b8 -1>
@byte.vector.highbit = constant <2 x b8> <b8 128, b8 255>
