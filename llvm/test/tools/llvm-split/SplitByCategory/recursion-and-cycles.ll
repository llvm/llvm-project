; Check that Module splitting produces correct output in the presence of cycles
; in the dependency graph.

; RUN: llvm-split -split-by-category=kernel -S < %s -o %t
; RUN: FileCheck %s -input-file=%t_0.ll --check-prefix CHECK0 \
; RUN:     --implicit-check-not @gptr --implicit-check-not @kernel_A
; RUN: FileCheck %s -input-file=%t_1.ll --check-prefix CHECK1 \
; RUN:     --implicit-check-not @self --implicit-check-not @kernel_B

; CHECK0-DAG: define spir_func void @ping()
; CHECK0-DAG: define spir_func void @pong()
; CHECK0-DAG: define spir_func void @self()
; CHECK0-DAG: define spir_kernel void @kernel_B()

; CHECK1-DAG: @gptr = private global ptr @gptr
; CHECK1-DAG: define spir_func void @ping()
; CHECK1-DAG: define spir_func void @pong()
; CHECK1-DAG: define spir_kernel void @kernel_A()

@gptr = private global ptr @gptr

define spir_func void @ping() {
  call spir_func void @pong()
  ret void
}

define spir_func void @pong() {
  call spir_func void @ping()
  ret void
}

define spir_func void @self() {
  call spir_func void @self()
  ret void
}

define spir_kernel void @kernel_A() {
  call spir_func void @ping()
  %v = load ptr, ptr @gptr
  ret void
}

define spir_kernel void @kernel_B() {
  call spir_func void @ping()
  call spir_func void @self()
  ret void
}
