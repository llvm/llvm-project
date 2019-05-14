; RUN: llc -verify-machineinstrs -o - %s -mtriple=aarch64-none-linux-gnu -mcpu=cyclone | FileCheck %s
; RUN: llc -verify-machineinstrs -o - %s -mtriple=aarch64-none-linux-gnu -code-model=large -mcpu=cyclone | FileCheck --check-prefix=CHECK-LARGE %s
; RUN: llc -verify-machineinstrs -o - %s -mtriple=aarch64-none-none-eabi -code-model=tiny -mcpu=cyclone | FileCheck --check-prefix=CHECK-TINY %s
; RUN: llc -verify-machineinstrs < %s -mtriple=aarch64-none-linux-gnu -mattr=-fp-armv8 | FileCheck --check-prefix=CHECK-NOFP %s
; RUN: llc -verify-machineinstrs < %s -mtriple=aarch64-none-linux-gnu -code-model=large -mattr=-fp-armv8 | FileCheck --check-prefix=CHECK-NOFP-LARGE %s
; RUN: llc -verify-machineinstrs < %s -mtriple=aarch64-none-none-eabi -code-model=tiny -mattr=-fp-armv8 | FileCheck --check-prefix=CHECK-NOFP-TINY %s

@varfloat = global float 0.0
@vardouble = global double 0.0

define void @floating_lits() {
; CHECK-LABEL: floating_lits:

  %floatval = load float, float* @varfloat
  %newfloat = fadd float %floatval, 128.0
; CHECK: mov [[W128:w[0-9]+]], #1124073472
; CHECK: fmov [[LIT128:s[0-9]+]], [[W128]]
; CHECK-NOFP-NOT: ldr {{s[0-9]+}},

; CHECK-TINY: mov [[W128:w[0-9]+]], #1124073472
; CHECK-TINE: fmov [[LIT128:s[0-9]+]], [[W128]]
; CHECK-NOFP-TINY-NOT: ldr {{s[0-9]+}},

; CHECK-LARGE: mov [[W128:w[0-9]+]], #1124073472
; CHECK-LARGE: fmov [[LIT128:s[0-9]+]], [[W128]]
; CHECK-LARGE: fadd
; CHECK-NOFP-LARGE-NOT: ldr {{s[0-9]+}},
; CHECK-NOFP-LARGE-NOT: fadd

  store float %newfloat, float* @varfloat

  %doubleval = load double, double* @vardouble
  %newdouble = fadd double %doubleval, 129.0
; CHECK-NOFP-NOT: ldr {{d[0-9]+}},
; CHECK: mov  [[W129:x[0-9]+]], #35184372088832
; CHECK: movk [[W129]], #16480, lsl #48
; CHECK: fmov {{d[0-9]+}}, [[W129]]
; CHECK-NOFP-NOT: fadd

; CHECK-TINY: mov  [[W129:x[0-9]+]], #35184372088832
; CHECK-TINY: movk [[W129]], #16480, lsl #48
; CHECK-TINY: fmov {{d[0-9]+}}, [[W129]]
; CHECK-NOFP-TINY-NOT: ldr {{d[0-9]+}},
; CHECK-NOFP-TINY-NOT: fadd

; CHECK-LARGE: movz x[[LITADDR:[0-9]+]], #:abs_g0_nc:[[CURLIT:vardouble]]
; CHECK-LARGE: movk x[[LITADDR]], #:abs_g1_nc:[[CURLIT]]
; CHECK-LARGE: movk x[[LITADDR]], #:abs_g2_nc:[[CURLIT]]
; CHECK-LARGE: movk x[[LITADDR]], #:abs_g3:[[CURLIT]]
; CHECK-LARGE: ldr {{d[0-9]+}}, [x[[LITADDR]]]
; CHECK-NOFP-LARGE-NOT: ldr {{d[0-9]+}},

  store double %newdouble, double* @vardouble

  ret void
}
