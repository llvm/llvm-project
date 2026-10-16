; RUN: not llc -mtriple=mipsel -mcpu=mips32r2 -mattr=+fp64 \
; RUN:         -O0 -relocation-model=pic -fast-isel-abort=3 -filetype=null %s 2>&1 | FileCheck %s

; Check that FastISel aborts when we have 64bit FPU registers. FastISel currently
; supports AFGR64 only, which uses paired 32 bit registers.

; CHECK: LLVM ERROR: FastISel didn't lower all arguments: i1 (double) (in function: f)

define zeroext i1 @f(double %value) {
entry:
  %value.addr = alloca double, align 8
  store double %value, ptr %value.addr, align 8
  ret i1 false
}
