; The test checks that Module splitting does not treat declarations as entry points.

; RUN: llvm-split -sycl-split=kernel -S < %s -o %t2
; RUN: FileCheck %s -input-file=%t2.table --check-prefix CHECK-PER-KERNEL-TABLE
; RUN: FileCheck %s -input-file=%t2_0.sym --check-prefix CHECK-PER-KERNEL-SYM1
; RUN: FileCheck %s -input-file=%t2_1.sym --check-prefix CHECK-PER-KERNEL-SYM2
; RUN: FileCheck %s -input-file=%t2_2.sym --check-prefix CHECK-PER-KERNEL-SYM0

; With per-kernel split, there should be three device images
; CHECK-PER-KERNEL-TABLE: [Code|Symbols]
; CHECK-PER-KERNEL-TABLE: {{.*}}_0.ll|{{.*}}_0.sym
; CHECK-PER-KERNEL-TABLE-NEXT: {{.*}}_1.ll|{{.*}}_1.sym
; CHECK-PER-KERNEL-TABLE-NEXT: {{.*}}_2.ll|{{.*}}_2.sym
; CHECK-PER-KERNEL-TABLE-EMPTY:
;
; CHECK-PER-KERNEL-SYM0-NOT: _ZTS4mainE10TU1_kernel1
; CHECK-PER-KERNEL-SYM0: _ZTSZ4mainE10TU1_kernel0
; CHECK-PER-KERNEL-SYM0-EMPTY:
;
; CHECK-PER-KERNEL-SYM2-NOT: _ZTS4mainE10TU1_kernel1
; CHECK-PER-KERNEL-SYM2: _ZTSZ4mainE11TU0_kernel0
; CHECK-PER-KERNEL-SYM2-EMPTY:
;
; CHECK-PER-KERNEL-SYM1-NOT: _ZTS4mainE10TU1_kernel1
; CHECK-PER-KERNEL-SYM1: _ZTSZ4mainE11TU0_kernel1
; CHECK-PER-KERNEL-SYM1-EMPTY:

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir64-unknown-linux"

define spir_kernel void @_ZTSZ4mainE11TU0_kernel0() #0 {
entry:
  ret void
}

define spir_kernel void @_ZTSZ4mainE11TU0_kernel1() #0 {
entry:
  ret void
}

define spir_kernel void @_ZTSZ4mainE10TU1_kernel0() #1 {
  ret void
}

declare spir_kernel void @_ZTS4mainE10TU1_kernel1() #1

attributes #0 = { "sycl-module-id"="TU1.cpp" }
attributes #1 = { "sycl-module-id"="TU2.cpp" }

!opencl.spir.version = !{!0, !0}
!spirv.Source = !{!1, !1}
!0 = !{i32 1, i32 2}
!1 = !{i32 4, i32 100000}
