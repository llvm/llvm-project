; RUN: opt < %s -memcpyopt -mldst-motion -gvn -S | FileCheck %s

declare void @check(i8)

declare void @write(i8* %res)

define void @test1() {
  %1 = alloca [10 x i8]
  %2 = bitcast [10 x i8]* %1 to i8*
  call void @write(i8* %2)
  %3 = load i8, i8* %2

; CHECK-NOT: undef
  call void @check(i8 %3)

  ret void
}

