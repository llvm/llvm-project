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
; CHECK:       // %bb.0:
; CHECK-NEXT:    mov w8, #52429 // =0xcccd
; CHECK-NEXT:    movk w8, #15820, lsl #16
; CHECK-NEXT:    fmov s0, w8
; CHECK-NEXT:    ret
;
; CHECK-LARGE-LABEL: float_ret_optnone:
; CHECK-LARGE:       // %bb.0:
; CHECK-LARGE-NEXT:    mov w8, #52429 // =0xcccd
; CHECK-LARGE-NEXT:    movk w8, #15820, lsl #16
; CHECK-LARGE-NEXT:    fmov s0, w8
; CHECK-LARGE-NEXT:    ret
;
; CHECK-TINY-LABEL: float_ret_optnone:
; CHECK-TINY:       // %bb.0:
; CHECK-TINY-NEXT:    mov w8, #52429 // =0xcccd
; CHECK-TINY-NEXT:    movk w8, #15820, lsl #16
; CHECK-TINY-NEXT:    fmov s0, w8
; CHECK-TINY-NEXT:    ret
;
; CHECK-NOFP-LABEL: float_ret_optnone:
; CHECK-NOFP:       // %bb.0:
; CHECK-NOFP-NEXT:    mov w0, #52429 // =0xcccd
; CHECK-NOFP-NEXT:    movk w0, #15820, lsl #16
; CHECK-NOFP-NEXT:    ret
;
; CHECK-NOFP-LARGE-LABEL: float_ret_optnone:
; CHECK-NOFP-LARGE:       // %bb.0:
; CHECK-NOFP-LARGE-NEXT:    mov w0, #52429 // =0xcccd
; CHECK-NOFP-LARGE-NEXT:    movk w0, #15820, lsl #16
; CHECK-NOFP-LARGE-NEXT:    ret
;
; CHECK-NOFP-TINY-LABEL: float_ret_optnone:
; CHECK-NOFP-TINY:       // %bb.0:
; CHECK-NOFP-TINY-NEXT:    mov w0, #52429 // =0xcccd
; CHECK-NOFP-TINY-NEXT:    movk w0, #15820, lsl #16
; CHECK-NOFP-TINY-NEXT:    ret
  ret float 0x3FB99999A0000000
}

define dso_local double @double_ret_optnone() optnone noinline {
; CHECK-LABEL: double_ret_optnone:
; CHECK:       // %bb.0:
; CHECK-NEXT:    adrp x8, .LCPI2_0
; CHECK-NEXT:    ldr d0, [x8, :lo12:.LCPI2_0]
; CHECK-NEXT:    ret
;
; CHECK-LARGE-LABEL: double_ret_optnone:
; CHECK-LARGE:       // %bb.0:
; CHECK-LARGE-NEXT:    adrp x8, .LCPI2_0
; CHECK-LARGE-NEXT:    ldr d0, [x8, :lo12:.LCPI2_0]
; CHECK-LARGE-NEXT:    ret
;
; CHECK-TINY-LABEL: double_ret_optnone:
; CHECK-TINY:       // %bb.0:
; CHECK-TINY-NEXT:    ldr d0, .LCPI2_0
; CHECK-TINY-NEXT:    ret
;
; CHECK-NOFP-LABEL: double_ret_optnone:
; CHECK-NOFP:       // %bb.0:
; CHECK-NOFP-NEXT:    mov x0, #-7378697629483820647 // =0x9999999999999999
; CHECK-NOFP-NEXT:    movk x0, #39322
; CHECK-NOFP-NEXT:    movk x0, #16313, lsl #48
; CHECK-NOFP-NEXT:    ret
;
; CHECK-NOFP-LARGE-LABEL: double_ret_optnone:
; CHECK-NOFP-LARGE:       // %bb.0:
; CHECK-NOFP-LARGE-NEXT:    mov x0, #-7378697629483820647 // =0x9999999999999999
; CHECK-NOFP-LARGE-NEXT:    movk x0, #39322
; CHECK-NOFP-LARGE-NEXT:    movk x0, #16313, lsl #48
; CHECK-NOFP-LARGE-NEXT:    ret
;
; CHECK-NOFP-TINY-LABEL: double_ret_optnone:
; CHECK-NOFP-TINY:       // %bb.0:
; CHECK-NOFP-TINY-NEXT:    mov x0, #-7378697629483820647 // =0x9999999999999999
; CHECK-NOFP-TINY-NEXT:    movk x0, #39322
; CHECK-NOFP-TINY-NEXT:    movk x0, #16313, lsl #48
; CHECK-NOFP-TINY-NEXT:    ret
  ret double 0.1
}
