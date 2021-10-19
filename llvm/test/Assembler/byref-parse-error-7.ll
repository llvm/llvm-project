; RUN: not llvm-as < %s 2>&1 | FileCheck %s

; CHECK: <stdin>:[[@LINE+1]]:14: error: expected type{{$}}
define byref(8) i8* @test_byref() {
  ret void
}
