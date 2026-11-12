; RUN: llvm-dis -o - %s.bc | FileCheck %s

; CHECK: define void @test_dead_on_return_autoupgrade(ptr dead_on_return %p) !guid !{{[0-9]+}} {

define void @test_dead_on_return_autoupgrade(ptr dead_on_return %p) {
  ret void
}
