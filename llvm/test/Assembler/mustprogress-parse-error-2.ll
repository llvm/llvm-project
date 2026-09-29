; RUN: not llvm-as < %s 2>&1 | FileCheck %s

; CHECK: [[@LINE+1]]:51: error: expected '{' in function body
define ptr @test_mustprogress(i8 %a) mustprogress 8 {
  ret void
}

