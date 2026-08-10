; RUN: llvm-link %s -S -o - | FileCheck %s

$foo = comdat largest
define internal void @foo() comdat($foo) {
  ret void
}

; CHECK-NOT: foo
