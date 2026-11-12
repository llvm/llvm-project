; RUN: not llvm-as < %s 2>&1 | FileCheck %s
; CHECK: <stdin>:4:26: error: expected comma after getelementptr's type
define void @test(ptr %t) {
  %x = getelementptr ptr %t, i32 0
  ret void
}

