; RUN: not llvm-as < %s 2>&1 | FileCheck %s

; CHECK: <stdin>:[[@LINE+1]]:33: error: expected '('{{$}}
define void @test_byref() byref {
  ret void
}
