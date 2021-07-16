; RUN: not llvm-as < %s 2>&1 | FileCheck %s

; CHECK: <stdin>:[[@LINE+1]]:32: error: expected '('{{$}}
define void @test_byref() byref=4 {
  ret void
}
