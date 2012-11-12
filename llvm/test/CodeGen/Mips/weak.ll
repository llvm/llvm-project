; RUN: llc -march=mips < %s | FileCheck %s

@t = common global i32 (...)* null, align 4

define void @f() nounwind {
entry:
  store i32 (...)* @test_weak, i32 (...)** @t, align 4
  ret void
}

; CHECK: .weak test_weak
declare extern_weak i32 @test_weak(...)
