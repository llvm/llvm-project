; RUN: llvm-split -o %t %s
; RUN: llvm-dis -o - %t0 | FileCheck --check-prefix=CHECK0 %s
; RUN: llvm-dis -o - %t1 | FileCheck --check-prefix=CHECK1 %s

$foo = comdat any

; CHECK0: define void @foo()
; CHECK1: declare void @foo()
define void @foo() comdat {
  call void @bar()
  ret void
}

; CHECK0: define void @bar()
; CHECK1: declare void @bar()
define void @bar() comdat($foo) {
  call void @foo()
  ret void
}
