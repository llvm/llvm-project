; RUN: llc -mtriple mipsel-unknown-linux < %s | FileCheck  %s

target triple = "mipsel-unknown-linux"

@llvm.global_ctors = appending global [1 x { i32, void ()*, i8* }] [{ i32, void ()*, i8* } { i32 65535, void ()* @test, i8* null }]
; CHECK: .section
; CHECK: .init_array
; CHECK-NOT: .ctors
; CHECK: .4byte test

define internal void @test() section ".text.startup" {
entry:
  ret void
}
