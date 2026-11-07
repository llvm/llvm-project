; Test that an internal function called by multiple roots is externalized
; and renamed with a .llvm.<suffix> suffix in all partitions.

; RUN: llvm-split -enable-call-graph-split-module=true -j2 -o %t %s
; RUN: llvm-dis -o - %t0 | FileCheck --check-prefix=CHECK0 %s
; RUN: llvm-dis -o - %t1 | FileCheck --check-prefix=CHECK1 %s

; CHECK0-DAG: define hidden void @helper.llvm.{{[0-9a-f]+}}()
; CHECK1-DAG: define available_externally hidden  void @helper.llvm.{{[0-9a-f]+}}()

define internal void @helper() {
  ret void
}

define void @caller1() {
  call void @helper()
  ret void
}

define void @caller2() {
  call void @helper()
  ret void
}
