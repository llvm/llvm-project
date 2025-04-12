; The test checks that Module splitting does not treat declarations as entry points.

; RUN: llvm-split -sycl-split=source -S < %s -o %t1
; RUN: FileCheck %s -input-file=%t1.table --check-prefix CHECK-PER-SOURCE-TABLE
; RUN: FileCheck %s -input-file=%t1_0.sym --check-prefix CHECK-PER-SOURCE-SYM0
; RUN: FileCheck %s -input-file=%t1_1.sym --check-prefix CHECK-PER-SOURCE-SYM1

; RUN: llvm-split -sycl-split=kernel -S < %s -o %t2
; RUN: FileCheck %s -input-file=%t2.table --check-prefix CHECK-PER-KERNEL-TABLE
; RUN: FileCheck %s -input-file=%t2_0.sym --check-prefix CHECK-PER-KERNEL-SYM0
; RUN: FileCheck %s -input-file=%t2_1.sym --check-prefix CHECK-PER-KERNEL-SYM1
; RUN: FileCheck %s -input-file=%t2_2.sym --check-prefix CHECK-PER-KERNEL-SYM2

; With per-source split, there should be two device images
; CHECK-PER-SOURCE-TABLE: [Code|Symbols]
; CHECK-PER-SOURCE-TABLE: {{.*}}_0.ll|{{.*}}_0.sym
; CHECK-PER-SOURCE-TABLE-NEXT: {{.*}}_1.ll|{{.*}}_1.sym
; CHECK-PER-SOURCE-TABLE-EMPTY:
;
; CHECK-PER-SOURCE-SYM0-NOT: TU1_kernel1
; CHECK-PER-SOURCE-SYM0: TU1_kernel0
; CHECK-PER-SOURCE-SYM0-EMPTY:
;
; CHECK-PER-SOURCE-SYM1-NOT: TU1_kernel1
; CHECK-PER-SOURCE-SYM1: TU0_kernel0
; CHECK-PER-SOURCE-SYM1-NEXT: TU0_kernel1
; CHECK-PER-SOURCE-SYM1-EMPTY:

; With per-kernel split, there should be three device images
; CHECK-PER-KERNEL-TABLE: [Code|Symbols]
; CHECK-PER-KERNEL-TABLE: {{.*}}_0.ll|{{.*}}_0.sym
; CHECK-PER-KERNEL-TABLE-NEXT: {{.*}}_1.ll|{{.*}}_1.sym
; CHECK-PER-KERNEL-TABLE-NEXT: {{.*}}_2.ll|{{.*}}_2.sym
; CHECK-PER-KERNEL-TABLE-EMPTY:
;
; CHECK-PER-KERNEL-SYM0-NOT: TU1_kernel1
; CHECK-PER-KERNEL-SYM0: TU1_kernel0
; CHECK-PER-KERNEL-SYM0-EMPTY:
;
; CHECK-PER-KERNEL-SYM1-NOT: TU1_kernel1
; CHECK-PER-KERNEL-SYM1: TU0_kernel1
; CHECK-PER-KERNEL-SYM1-EMPTY:
;
; CHECK-PER-KERNEL-SYM2-NOT: TU1_kernel1
; CHECK-PER-KERNEL-SYM2: TU0_kernel0
; CHECK-PER-KERNEL-SYM2-EMPTY:


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

attributes #0 = { "sycl-module-id"="TU1.cpp" }
attributes #1 = { "sycl-module-id"="TU2.cpp" }
