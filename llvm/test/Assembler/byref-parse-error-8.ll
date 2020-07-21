; RUN: not llvm-as < %s 2>&1 | FileCheck %s

; CHECK: <stdin>:[[@LINE+1]]:27: error: invalid use of parameter-only attribute on a function{{$}}
define void @test_byref() byref {
  ret void
}
