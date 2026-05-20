; RUN: llvm-split -enable-split-module-CG=true -j1 -o %t %s
; RUN: llvm-dis -o - %t0 | FileCheck --check-prefix=CHECK0 %s

; CHECK0: define void @foo()
; CHECK0: define void @bar()

define void @foo() {
  call void @bar()
  ret void
}
define void @bar() {
  ret void
}
