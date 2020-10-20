; RUN: not llvm-as < %s 2>&1 | FileCheck %s

; CHECK: [[@LINE+1]]:35: error: invalid use of function-only attribute 
define void @test_mustprogress(i8 mustprogress %a) {
  ret void
}

