; RUN: llvm-link -S %s %p/Inputs/ctors2.ll -o - | FileCheck %s

$foo = comdat any
@foo = global i8 0, comdat

; CHECK: @llvm.global_ctors = appending global [0 x { i32, void ()*, i8* }] zeroinitializer
; CHECK: @foo = global i8 0, comdat
