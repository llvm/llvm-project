; RUN: llvm-split -j2 -o %t %s
; RUN: llvm-dis -o - %t0 | FileCheck --check-prefix=CHECK0 %s
; RUN: llvm-dis -o - %t1 | FileCheck --check-prefix=CHECK1 %s

; CHECK0-DAG: @foo_a.ifunc = ifunc void (), ptr @foo_a.resolver
; CHECK1-DAG: declare void @foo_a.ifunc()

@foo_a.ifunc = ifunc void (), ptr @foo_a.resolver

; CHECK0-DAG: define hidden void @foo.impl()
; CHECK1-DAG: declare hidden void @foo.impl()

define internal void @foo.impl() {
entry:
  ret void
}

; CHECK0-DAG: define hidden ptr @foo_a.resolver()
; CHECK1-DAG: declare hidden ptr @foo_a.resolver()

define internal ptr @foo_a.resolver() {
entry:
  ret ptr @foo.impl
}

; CHECK0-DAG: declare void @bar_a()
; CHECK1-DAG: define void @bar_a()

define void @bar_a() {
entry:
  call void @foo_a.ifunc()
  ret void
}
