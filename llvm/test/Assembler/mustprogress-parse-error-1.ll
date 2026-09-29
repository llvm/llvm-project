; RUN: not llvm-as < %s 2>&1 | FileCheck %s

; CHECK: [[@LINE+1]]:8: error: this attribute does not apply to return values
define mustprogress void @test_mustprogress(i8 %a) {
  ret void
}

