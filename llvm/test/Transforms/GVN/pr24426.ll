; RUN: opt < %s -passes=memcpyopt,mldst-motion,gvn -S | FileCheck %s

declare void @check(i8)

declare void @write(ptr %res)

define void @test1() {
  %1 = alloca [10 x i8]
  call void @write(ptr %1)
  %2 = load i8, ptr %1

; CHECK-NOT: undef
  call void @check(i8 %2)

  ret void
}

