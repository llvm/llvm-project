; RUN: llvm-link -S %s %p/Inputs/comdat14.ll -o - | FileCheck %s

$c = comdat any

@v = global i32 0, comdat ($c)

; CHECK: @v = global i32 0, comdat($c)
; CHECK: @v3 = external global i32
; CHECK: @v2 = external dllexport global i32
