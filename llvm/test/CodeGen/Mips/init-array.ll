; RUN: llc -mtriple mipsel-unknown-linux < %s | FileCheck  %s

target triple = "mipsel-unknown-linux"

@llvm.global_ctors = appending global [1 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 65535, ptr @test, ptr null }]
; CHECK: .section
; CHECK: .init_array
; CHECK-NOT: .ctors
; CHECK: .4byte test

define internal void @test() section ".text.startup" {
entry:
  ret void
}
