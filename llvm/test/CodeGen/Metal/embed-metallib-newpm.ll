; Drive MetalEmbedderPass via the new pass manager. The embedder serializes
; the module into a .metallib byte blob held in a private global with section
; ".metallib".
; RUN: opt -mtriple=air -passes=metal-embed -S %s | FileCheck %s

; CHECK: @metal.metallib = private constant [{{[0-9]+}} x i8]
; CHECK-SAME: section ".metallib"
; CHECK: @llvm.compiler.used = appending global

define void @kernel() {
  ret void
}
