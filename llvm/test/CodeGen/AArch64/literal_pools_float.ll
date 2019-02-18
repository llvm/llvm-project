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
; CHECK: adrp x[[LITBASE:[0-9]+]], [[CURLIT:.LCPI[0-9]+_[0-9]+]]
; CHECK: ldr [[LIT128:s[0-9]+]], [x[[LITBASE]], {{#?}}:lo12:[[CURLIT]]]
; CHECK-NOFP-NOT: ldr {{s[0-9]+}},

; CHECK-TINY: ldr [[LIT128:s[0-9]+]], [[CURLIT:.LCPI[0-9]+_[0-9]+]]
; CHECK-NOFP-TINY-NOT: ldr {{s[0-9]+}},

; CHECK-LARGE: movz x[[LITADDR:[0-9]+]], #:abs_g0_nc:[[CURLIT:.LCPI[0-9]+_[0-9]+]]
; CHECK-LARGE: movk x[[LITADDR]], #:abs_g1_nc:[[CURLIT]]
; CHECK-LARGE: movk x[[LITADDR]], #:abs_g2_nc:[[CURLIT]]
; CHECK-LARGE: movk x[[LITADDR]], #:abs_g3:[[CURLIT]]
; CHECK-LARGE: ldr {{s[0-9]+}}, [x[[LITADDR]]]
; CHECK-LARGE: fadd
; CHECK-NOFP-LARGE-NOT: ldr {{s[0-9]+}},
; CHECK-NOFP-LARGE-NOT: fadd

  store float %newfloat, float* @varfloat

  %doubleval = load double, double* @vardouble
  %newdouble = fadd double %doubleval, 129.0
; CHECK: adrp x[[LITBASE:[0-9]+]], [[CURLIT:.LCPI[0-9]+_[0-9]+]]
; CHECK: ldr [[LIT129:d[0-9]+]], [x[[LITBASE]], {{#?}}:lo12:[[CURLIT]]]
; CHECK-NOFP-NOT: ldr {{d[0-9]+}},
; CHECK-NOFP-NOT: fadd

; CHECK-TINY: ldr [[LIT129:d[0-9]+]], [[CURLIT:.LCPI[0-9]+_[0-9]+]]
; CHECK-NOFP-TINY-NOT: ldr {{d[0-9]+}},
; CHECK-NOFP-TINY-NOT: fadd

; CHECK-LARGE: movz x[[LITADDR:[0-9]+]], #:abs_g0_nc:[[CURLIT:.LCPI[0-9]+_[0-9]+]]
; CHECK-LARGE: movk x[[LITADDR]], #:abs_g1_nc:[[CURLIT]]
; CHECK-LARGE: movk x[[LITADDR]], #:abs_g2_nc:[[CURLIT]]
; CHECK-LARGE: movk x[[LITADDR]], #:abs_g3:[[CURLIT]]
; CHECK-LARGE: ldr {{d[0-9]+}}, [x[[LITADDR]]]
; CHECK-NOFP-LARGE-NOT: ldr {{d[0-9]+}},

  store double %newdouble, double* @vardouble

  ret void
}
