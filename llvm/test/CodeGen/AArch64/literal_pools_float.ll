; RUN: llc -verify-machineinstrs -o - %s -mtriple=aarch64-none-linux-gnu -mcpu=cyclone | FileCheck %s
; RUN: llc -verify-machineinstrs -o - %s -mtriple=aarch64-none-linux-gnu -code-model=large -mcpu=cyclone | FileCheck --check-prefix=CHECK-LARGE %s
; RUN: llc -verify-machineinstrs -o - %s -mtriple=aarch64-none-elf -code-model=tiny -mcpu=cyclone | FileCheck --check-prefix=CHECK-TINY %s
; RUN: llc -verify-machineinstrs < %s -mtriple=aarch64-none-linux-gnu -mattr=-fp-armv8 | FileCheck --check-prefix=CHECK-NOFP %s
; RUN: llc -verify-machineinstrs < %s -mtriple=aarch64-none-linux-gnu -code-model=large -mattr=-fp-armv8 | FileCheck --check-prefix=CHECK-NOFP-LARGE %s
; RUN: llc -verify-machineinstrs < %s -mtriple=aarch64-none-elf -code-model=tiny -mattr=-fp-armv8 | FileCheck --check-prefix=CHECK-NOFP-TINY %s

@varfloat = dso_local global float 0.0
@vardouble = dso_local global double 0.0

define dso_local void @floating_lits() optsize {
; CHECK-LABEL: floating_lits:

  %floatval = load float, ptr @varfloat
  %newfloat = fadd float %floatval, 511.0
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

  store float %newfloat, ptr @varfloat

  %doubleval = load double, ptr @vardouble
  %newdouble = fadd double %doubleval, 511.0
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

  store double %newdouble, ptr @vardouble

  ret void
}

define dso_local float @float_ret_optnone() optnone noinline {
; CHECK-LABEL: float_ret_optnone:

  ret float 0x3FB99999A0000000
; CHECK: adrp x[[LITBASE:[0-9]+]], [[CURLIT:.LCPI[0-9]+_[0-9]+]]
; CHECK: ldr [[LIT128:s[0-9]+]], [x[[LITBASE]], {{#?}}:lo12:[[CURLIT]]]

; In the large code model, FastISel cannot load from the constant pool.
; CHECK-LARGE-NOT: adrp
; CHECK-LARGE-NOT: ldr
}

define dso_local double @double_ret_optnone() optnone noinline {
; CHECK-LABEL: double_ret_optnone:

  ret double 0.1
; CHECK: adrp x[[LITBASE:[0-9]+]], [[CURLIT:.LCPI[0-9]+_[0-9]+]]
; CHECK: ldr [[LIT128:d[0-9]+]], [x[[LITBASE]], {{#?}}:lo12:[[CURLIT]]]

; In the large code model, FastISel cannot load from the constant pool.
; CHECK-LARGE-NOT: adrp
; CHECK-LARGE-NOT: ldr
}
