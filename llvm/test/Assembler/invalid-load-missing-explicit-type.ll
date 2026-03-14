; RUN: not llvm-as < %s 2>&1 | FileCheck %s
; CHECK: <stdin>:4:17: error: expected comma after load's type
define void @test(ptr %t) {
  %x = load ptr %t
  ret void
}
