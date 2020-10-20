; RUN: not llvm-as < %s 2>&1 | FileCheck %s

; CHECK: [[@LINE+1]]:8: error: invalid use of function-only attribute 
define mustprogress void @test_mustprogress(i8 %a) {
  ret void
}

