; Check that Module splitting can trace through more complex call stacks
; involving several nested indirect calls.

; RUN: llvm-split -split-by-category=module-id -S < %s -o %t
; RUN: FileCheck %s -input-file=%t_0.ll --check-prefix CHECK0 \
; RUN:     --implicit-check-not @foo --implicit-check-not @kernel_A \
; RUN:     --implicit-check-not @kernel_B --implicit-check-not @baz
; RUN: FileCheck %s -input-file=%t_1.ll --check-prefix CHECK1 \
; RUN:     --implicit-check-not @kernel_A --implicit-check-not @kernel_C
; RUN: FileCheck %s -input-file=%t_2.ll --check-prefix CHECK2 \
; RUN:     --implicit-check-not @foo --implicit-check-not @bar \
; RUN:     --implicit-check-not @BAZ --implicit-check-not @kernel_B \
; RUN:     --implicit-check-not @kernel_C

; RUN: llvm-split -split-by-category=kernel -S < %s -o %t
; RUN: FileCheck %s -input-file=%t_0.ll --check-prefix CHECK0 \
; RUN:     --implicit-check-not @foo --implicit-check-not @kernel_A \
; RUN:     --implicit-check-not @kernel_B
; RUN: FileCheck %s -input-file=%t_1.ll --check-prefix CHECK1 \
; RUN:     --implicit-check-not @kernel_A --implicit-check-not @kernel_C
; RUN: FileCheck %s -input-file=%t_2.ll --check-prefix CHECK2 \
; RUN:     --implicit-check-not @foo --implicit-check-not @bar \
; RUN:     --implicit-check-not @BAZ --implicit-check-not @kernel_B \
; RUN:     --implicit-check-not @kernel_C

; CHECK0-DAG: define spir_kernel void @kernel_C
; CHECK0-DAG: define spir_func i32 @bar
; CHECK0-DAG: define spir_func void @baz
; CHECK0-DAG: define spir_func void @BAZ

; CHECK1-DAG: define spir_kernel void @kernel_B
; CHECK1-DAG: define {{.*}}spir_func i32 @foo
; CHECK1-DAG: define spir_func i32 @bar
; CHECK1-DAG: define spir_func void @baz
; CHECK1-DAG: define spir_func void @BAZ

; CHECK2-DAG: define spir_kernel void @kernel_A
; CHECK2-DAG: define {{.*}}spir_func void @baz

define spir_func i32 @foo(i32 (i32, void ()*)* %ptr1, void ()* %ptr2) {
  %1 = call spir_func i32 %ptr1(i32 42, void ()* %ptr2)
  ret i32 %1
}

define spir_func i32 @bar(i32 %arg, void ()* %ptr) {
  call spir_func void %ptr()
  ret i32 %arg
}

define spir_func void @baz() {
  ret void
}

define spir_func void @BAZ() {
  ret void
}

define spir_kernel void @kernel_A() #0 {
  call spir_func void @baz()
  ret void
}

define spir_kernel void @kernel_B() #1 {
  call spir_func i32 @foo(i32 (i32, void ()*)* null, void ()* null)
  ret void
}

define spir_kernel void @kernel_C() #2 {
  call spir_func i32 @bar(i32 42, void ()* null)
  ret void
}

attributes #0 = { "module-id"="TU1.cpp" }
attributes #1 = { "module-id"="TU2.cpp" }
attributes #2 = { "module-id"="TU3.cpp" }
