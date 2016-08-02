; RUN: llvm-link -S %s %p/Inputs/comdat14.ll -o - | FileCheck %s

$c = comdat any

@v = global i32 0, comdat ($c)

; CHECK: @v = global i32 0, comdat($c)
; CHECK: @v2 = extern_weak dllexport global i32
; CHECK: @v3 = extern_weak global i32
