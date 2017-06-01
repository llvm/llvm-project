; RUN: llvm-link -S %s %p/Inputs/ctors3.ll -o - | FileCheck %s

$foo = comdat any
%t = type { i8 }
@foo = global %t zeroinitializer, comdat

; CHECK: @foo = global %t zeroinitializer, comdat
; CHECK: @llvm.global_ctors = appending global [0 x { i32, void ()*, i8* }] zeroinitializer
