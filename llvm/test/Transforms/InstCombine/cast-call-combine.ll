; RUN: opt < %s -instcombine -always-inline -S | FileCheck %s

define internal void @foo(ptr) alwaysinline {
  ret void
}

define void @bar() noinline noreturn {
  unreachable
}

define void @test() {
  br i1 false, label %then, label %else

then:
  call void @bar()
  unreachable

else:
  ; CHECK-NOT: call
  call void @foo (ptr null)
  ret void
}

