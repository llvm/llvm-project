; Test that "personality" attributes are correctly updated when cloning modules.
; RUN: llvm-split -o %t %s
; RUN: llvm-dis -o - %t0 | FileCheck --check-prefix=CHECK0 %s
; RUN: llvm-dis -o - %t1 | FileCheck --check-prefix=CHECK1 %s

; CHECK0: define void @foo()
; CHECK1: declare void @foo()
define void @foo() {
  ret void
}

; CHECK0: declare void @bar()
; CHECK0-NOT: personality
; CHECK1: define void @bar() personality ptr @foo
define void @bar() personality ptr @foo
{
  ret void
}
