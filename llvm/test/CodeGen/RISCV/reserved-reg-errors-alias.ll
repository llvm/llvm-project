; RUN: not llc -mtriple=riscv32 -mattr=+zhinx,+zfinx,+zdinx,+reserve-x10 -target-abi=ilp32 < %s 2>&1 | FileCheck %s --check-prefixes=X10,RV32-X10
; RUN: not llc -mtriple=riscv64 -mattr=+zhinx,+zfinx,+zdinx,+reserve-x10 -target-abi=lp64 < %s 2>&1 | FileCheck %s --check-prefixes=X10,RV64-X10
; RUN: not llc -mtriple=riscv32 -mattr=+zdinx,+reserve-x11 -target-abi=ilp32 < %s 2>&1 | FileCheck %s --check-prefix=RV32-X11
; RUN: llc -mtriple=riscv64 -mattr=+zhinx,+zfinx,+zdinx,+reserve-x11 -target-abi=lp64 < %s

; Reserving an X register must also be diagnosed when the calling convention
; uses a sub-register (X10_H for half, X10_W for float on RV64) or a
; super-register (the X10_X11 GPRPair for double on RV32).

declare half @gh(half)
declare float @gf(float)
declare double @gd(double)

define half @ch(half %a) {
; X10: in function ch {{.*}} Argument register required, but has been reserved.
; X10: in function ch {{.*}} Return value register required, but has been reserved.
  %r = call half @gh(half %a)
  ret half %r
}

define float @cf(float %a) {
; X10: in function cf {{.*}} Argument register required, but has been reserved.
; X10: in function cf {{.*}} Return value register required, but has been reserved.
  %r = call float @gf(float %a)
  ret float %r
}

define double @cd(double %a) {
; RV32-X10: in function cd {{.*}} Argument register required, but has been reserved.
; RV32-X10: in function cd {{.*}} Return value register required, but has been reserved.
; RV64-X10: in function cd {{.*}} Argument register required, but has been reserved.
; RV64-X10: in function cd {{.*}} Return value register required, but has been reserved.
; RV32-X11: in function cd {{.*}} Argument register required, but has been reserved.
; RV32-X11: in function cd {{.*}} Return value register required, but has been reserved.
  %r = call double @gd(double %a)
  ret double %r
}
