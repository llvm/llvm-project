; RUN: not llc -march=mipsel -mcpu=mips32r2 -mattr=+fp64 \
; RUN:         -O0 -relocation-model=pic -fast-isel-abort=3 < %s

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
