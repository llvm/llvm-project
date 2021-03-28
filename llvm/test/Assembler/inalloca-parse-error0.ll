; RUN: not llvm-as < %s 2>&1 | FileCheck %s

; CHECK: <stdin>:[[@LINE+1]]:40: error: expected '('{{$}}
define void @test_inalloca(i8* inalloca) {
  ret void
}
