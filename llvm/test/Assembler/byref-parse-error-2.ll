; RUN: not llvm-as < %s 2>&1 | FileCheck %s

; CHECK: <stdin>:[[@LINE+1]]:35: error: expected type{{$}}
define void @test_byref(ptr byref()) {
  ret void
}

