; Test handling of weak_odr and linkonce_odr functions across partitions:
; ODR definitions are safe to downgrade to available_externally in duplicate
; partitions, while interposable linkages (weak/linkonce non-ODR) must keep
; real definitions everywhere.

; RUN: llvm-split -enable-call-graph-split-module=true -j2 -o %t %s
; RUN: llvm-dis -o - %t0 | FileCheck --check-prefix=CHECK0 %s
; RUN: llvm-dis -o - %t1 | FileCheck --check-prefix=CHECK1 %s

; ODR function defined in both partitions: the first partition keeps the real
; definition, the second gets an available_externally copy.
; CHECK0-DAG: define weak_odr void @odr_func()
; CHECK1-DAG: define available_externally void @odr_func()

; Interposable functions must NOT be downgraded: every partition keeps a real
; definition since the linker may pick any of them.
; CHECK0-DAG: define weak void @weak_func()
; CHECK1-DAG: define weak void @weak_func()
; CHECK0-DAG: define linkonce void @linkonce_func()
; CHECK1-DAG: define linkonce void @linkonce_func()

define weak_odr void @odr_func() {
  ret void
}

define weak void @weak_func() {
  ret void
}

define linkonce void @linkonce_func() {
  ret void
}

define void @caller1() {
  call void @odr_func()
  call void @linkonce_func()
  call void @weak_func()
  ret void
}

define void @caller2() {
  call void @odr_func()
  call void @linkonce_func()
  call void @weak_func()
  ret void
}
