; RUN: not llc -march=mipsel -mcpu=mips32r2 -fast-isel -mattr=+fp64 < %s \
; RUN:     -fast-isel-abort=3

; Check that FastISel aborts when we have 64bit FPU registers. FastISel currently
; supports AFGR64 only, which uses paired 32 bit registers.

define zeroext i1 @f(double %value) {
entry:
; CHECK-LABEL: f:
; CHECK: sdc1
  %value.addr = alloca double, align 8
  store double %value, double* %value.addr, align 8
  ret i1 false
}
