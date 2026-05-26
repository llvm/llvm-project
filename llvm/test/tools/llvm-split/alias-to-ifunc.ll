; RUN: llvm-split -j2 -o %t %s
; RUN: llvm-dis -o - %t0 | FileCheck --check-prefix=CHECK0 %s
; RUN: llvm-dis -o - %t1 | FileCheck --check-prefix=CHECK1 %s

; CHECK0-DAG: @alias_foo = alias void (), ptr @foo_a.ifunc
; CHECK0-DAG: @foo_a.ifunc = ifunc void (), ptr @foo_a.resolver
; CHECK0-DAG: define hidden ptr @foo_a.resolver()
; CHECK1-DAG: declare void @alias_foo()
; CHECK1-DAG: declare void @foo_a.ifunc()
; CHECK1-DAG: declare hidden ptr @foo_a.resolver()

@alias_foo = alias void (), ptr @foo_a.ifunc
@foo_a.ifunc = ifunc void (), ptr @foo_a.resolver

define internal void @foo.impl() {
entry:
  ret void
}

define internal ptr @foo_a.resolver() {
entry:
  ret ptr @foo.impl
}

define void @bar_a() {
entry:
  call void @alias_foo()
  ret void
}
