; RUN: not llvm-as < %s -disable-output 2>&1 | FileCheck %s
; CHECK: error: '%x' defined with type 'i32' but expected 'float'
define void @test(i32 %x) {
  ret void

  uselistorder float %x, { 1, 0 }
}
