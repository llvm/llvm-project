; We run nvvm-reflect (and then optimize) this module twice, once with metadata
; that enables precise sqrt, and again with metadata that disables it.

; RUN: cat %s > %t.noprec
; RUN: echo '!0 = !{i32 4, !"nvvm-reflect-prec-sqrt", i32 0}' >> %t.noprec
; RUN: opt %t.noprec -S -mtriple=nvptx-nvidia-cuda -passes='nvvm-reflect' \
; RUN:   | FileCheck %s --check-prefix=PREC_SQRT_0 --check-prefix=CHECK

; RUN: cat %s > %t.prec
; RUN: echo '!0 = !{i32 4, !"nvvm-reflect-prec-sqrt", i32 1}' >> %t.prec
; RUN: opt %t.prec -S -mtriple=nvptx-nvidia-cuda -passes='nvvm-reflect' \
; RUN:   | FileCheck %s --check-prefix=PREC_SQRT_1 --check-prefix=CHECK

@.str = private unnamed_addr constant [17 x i8] c"__CUDA_PREC_SQRT\00", align 1

declare i32 @__nvvm_reflect(ptr)

; CHECK-LABEL: @foo
define i32 @foo() {
  ; CHECK-NOT: call i32 @__nvvm_reflect
  %reflect = call i32 @__nvvm_reflect(ptr @.str)
  ; PREC_SQRT_0: ret i32 0
  ; PREC_SQRT_1: ret i32 1
  ret i32 %reflect
}

!llvm.module.flags = !{!0}
; A module flag is added to the end of this file by the RUN lines at the top.
