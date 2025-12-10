; RUN: llvm-dis -o - %s.bc | FileCheck %s

; CHECK: define void @test_dead_on_return_autoupgrade(ptr dead_on_return(18446744073709551615) %p) {

define void @test_dead_on_return_autoupgrade(ptr dead_on_return %p) {
  ret void
}
