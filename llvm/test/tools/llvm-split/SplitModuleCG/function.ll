; RUN: llvm-split -enable-call-graph-split-module=true -j2 -o %t %s
; RUN: llvm-dis -o - %t0 | FileCheck --check-prefix=CHECK0 %s
; RUN: llvm-dis -o - %t1 | FileCheck --check-prefix=CHECK1 %s

; Test basic call graph based module splitting — functions are grouped into
; partitions by their call relationships.

; CHECK0-DAG: declare dso_local void @foo()
; CHECK0-DAG: define void @bar()
; CHECK0-DAG: declare void @func_a()
; CHECK0-DAG: define void @func_b()
; CHECK1-DAG: define internal void @foo()
; CHECK1-DAG: define available_externally void @bar()
; CHECK1-DAG: define void @func_a()
; CHECK1-DAG: declare void @func_b()

define internal void @foo() {
entry:
  ret void
}

define void @bar() {
entry:
  ret void
}

define void @func_a() {
entry:
  call void @foo()
  call void @bar()
  ret void
}

define void @func_b() {
entry:
  call void @bar()
  ret void
}
