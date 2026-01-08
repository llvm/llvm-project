; Check that Module splitting can trace indirect calls through signatures.

; RUN: llvm-split -split-by-category=module-id -S < %s -o %t
; RUN: FileCheck %s -input-file=%t_0.ll --check-prefix CHECK0 \
; RUN:     --implicit-check-not @kernel_A --implicit-check-not @bbb
; RUN: FileCheck %s -input-file=%t_1.ll --check-prefix CHECK1 \
; RUN:     --implicit-check-not @kernel_B --implicit-check-not @ccc

; RUN: llvm-split -split-by-category=kernel -S < %s -o %t
; RUN: FileCheck %s -input-file=%t_0.ll --check-prefix CHECK0 \
; RUN:     --implicit-check-not @kernel_A
; RUN: FileCheck %s -input-file=%t_1.ll --check-prefix CHECK1 \
; RUN:     --implicit-check-not @kernel_B

; CHECK0-DAG: define spir_kernel void @kernel_B
; CHECK0-DAG: define spir_func void @aaa(i32 %0, i32 %1)
; CHECK0-DAG: define spir_func void @ccc(ptr %ptr)

; CHECK1-DAG: define spir_kernel void @kernel_A
; CHECK1-DAG: define spir_func void @aaa(i32 %0, i32 %1)
; CHECK1-DAG: define spir_func void @bbb(ptr %ptr)


define spir_func void @aaa(i32 %0, i32 %1) {
  ret void
}

define spir_func void @bbb(void (i32, i32)* %ptr) {
  call spir_func void %ptr(i32 0, i32 0)
  ret void
}

define spir_func void @ccc(void (i32, i32)* %ptr) {
  call spir_func void %ptr(i32 0, i32 0)
  ret void
}

define spir_kernel void @kernel_A() #0 {
  call spir_func void @bbb(void (i32, i32)* null)
  ret void
}

define spir_kernel void @kernel_B() #1 {
  call spir_func void @ccc(void (i32, i32)* null)
  ret void
}

attributes #0 = { "module-id"="TU1.cpp" }
attributes #1 = { "module-id"="TU2.cpp" }
