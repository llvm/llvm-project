; The test checks that Module splitting does not treat declarations as entry points.

; RUN: llvm-split -split-by-category=module-id -S < %s -o %t1
; RUN: FileCheck %s -input-file=%t1_0.ll --check-prefix CHECK-MODULE-ID0
; RUN: FileCheck %s -input-file=%t1_1.ll --check-prefix CHECK-MODULE-ID1

; RUN: llvm-split -split-by-category=kernel -S < %s -o %t2
; RUN: FileCheck %s -input-file=%t2_0.ll --check-prefix CHECK-PER-KERNEL0
; RUN: FileCheck %s -input-file=%t2_1.ll --check-prefix CHECK-PER-KERNEL1
; RUN: FileCheck %s -input-file=%t2_2.ll --check-prefix CHECK-PER-KERNEL2

; With module-id split, there should be two modules
; CHECK-MODULE-ID0-NOT: TU0
; CHECK-MODULE-ID0-NOT: TU1_kernel1
; CHECK-MODULE-ID0: TU1_kernel0
;
; CHECK-MODULE-ID1-NOT: TU1
; CHECK-MODULE-ID1: TU0_kernel0
; CHECK-MODULE-ID1: TU0_kernel1

; With per-kernel split, there should be three modules.
; CHECK-PER-KERNEL0-NOT: TU0
; CHECK-PER-KERNEL0-NOT: TU1_kernel1
; CHECK-PER-KERNEL0: TU1_kernel0
;
; CHECK-PER-KERNEL1-NOT: TU0_kernel0
; CHECK-PER-KERNEL1-NOT: TU1
; CHECK-PER-KERNEL1: TU0_kernel1
;
; CHECK-PER-KERNEL2-NOT: TU0_kernel1
; CHECK-PER-KERNEL2-NOT: TU1
; CHECK-PER-KERNEL2: TU0_kernel0


define spir_kernel void @TU0_kernel0() #0 {
entry:
  ret void
}

define spir_kernel void @TU0_kernel1() #0 {
entry:
  ret void
}

define spir_kernel void @TU1_kernel0() #1 {
  ret void
}

declare spir_kernel void @TU1_kernel1() #1

attributes #0 = { "module-id"="TU1.cpp" }
attributes #1 = { "module-id"="TU2.cpp" }
