; RUN: opt < %s -always-inline -instcombine -S | FileCheck %s

define internal void @foo(i16*) alwaysinline {
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
  call void bitcast (void (i16*)* @foo to void (i8*)*) (i8* null)
  ret void
}

