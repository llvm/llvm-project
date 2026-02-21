; RUN: opt -S -dxil-prepare < %s | FileCheck %s

; Test that dxil-prepare handles llvm.global_ctors with zeroinitializer
; (which is not a ConstantArray) without crashing.
; Fixes https://github.com/llvm/llvm-project/issues/178993

target triple = "dxil-unknown-shadermodel6.7-library"

; An empty global_ctors array uses zeroinitializer, not ConstantArray
@llvm.global_ctors = appending global [0 x { i32, ptr, ptr }] zeroinitializer

; CHECK: define void @main()
define void @main() {
entry:
  ret void
}
