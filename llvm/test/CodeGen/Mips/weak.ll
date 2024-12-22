; RUN: llc -march=mips < %s | FileCheck %s

@t = common global ptr null, align 4

define void @f() nounwind {
entry:
  store ptr @test_weak, ptr @t, align 4
  ret void
}

; CHECK: .weak test_weak
declare extern_weak i32 @test_weak(...)
