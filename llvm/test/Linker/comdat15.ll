; RUN: llvm-link -S %s %p/Inputs/comdat15.ll -o - | FileCheck %s

$a1 = comdat any
@bar = global i32 0, comdat($a1)

; CHECK: @bar = global i32 0, comdat($a1)
; CHECK: @baz = private global i32 42, comdat($a1)
; CHECK: @a1 = internal alias i32, i32* @baz

