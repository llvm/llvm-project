; RUN: llvm-split -enable-split-module-CG=true -j2 -o %t %s
; RUN: llvm-dis -o - %t0 | FileCheck --check-prefix=CHECK0 %s
; RUN: llvm-dis -o - %t1 | FileCheck --check-prefix=CHECK1 %s

; CHECK0-DAG: $group = comdat any
; CHECK0-DAG: @my_var = global i32 42, comdat($group)
; CHECK0-DAG: define void @foo() comdat($group)
; CHECK0-DAG: define void @bar() comdat($group)
; CHECK0-DAG: define void @call_foo()
; CHECK0-DAG: declare void @call_bar()
; CHECK1-DAG: @my_var = available_externally global i32 42
; CHECK1-DAG: define available_externally void @foo()
; CHECK1-DAG: define available_externally void @bar()
; CHECK1-DAG: declare void @call_foo()
; CHECK1-DAG: define void @call_bar()

$group = comdat any

@my_var =  global i32 42, comdat($group)

define void @foo() comdat($group) {
entry:
  ret void
}

define void @bar() comdat($group) {
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


