; RUN: llvm-split -enable-split-module-CG=true -j2 -o %t %s
; RUN: llvm-dis -o - %t0 | FileCheck --check-prefix=CHECK0 %s
; RUN: llvm-dis -o - %t1 | FileCheck --check-prefix=CHECK1 %s

; CHECK0-DAG: declare void @foo()
; CHECK0-DAG: define void @bar()
; CHECK0-DAG: declare void @call_foo()
; CHECK0-DAG: define void @call_bar()

; CHECK1-DAG: define void @foo()
; CHECK1-DAG: declare void @bar()
; CHECK1-DAG: define void @call_foo()
; CHECK1-DAG: declare void @call_bar()

define void @foo() {
entry:
  call void @call_foo()
  ret void
}

define void @bar() {
entry:
  ret void
}

define void @call_foo() {
entry:
  call void @foo()
  ret void
}

define void @call_bar() {
entry:
  call void @bar()
  ret void
}
