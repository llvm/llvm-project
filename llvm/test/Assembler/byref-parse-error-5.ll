; RUN: not llvm-as < %s 2>&1 | FileCheck %s

; CHECK: <stdin>:[[@LINE+1]]:14: error: expected '('{{$}}
define byref ptr @test_byref() {
  ret void
}
