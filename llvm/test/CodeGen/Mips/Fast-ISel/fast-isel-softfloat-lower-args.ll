; RUN: not llc -mtriple=mipsel -mcpu=mips32r2 -mattr=+soft-float \
; RUN:         -O0 -fast-isel-abort=3 -relocation-model=pic < %s 2>&1 | FileCheck %s

; Test that FastISel aborts instead of trying to lower arguments for soft-float.

; CHECK: LLVM ERROR: FastISel didn't lower all arguments: void (double) (in function: __signbit)
define void @__signbit(double %__x) {
entry:
  %__x.addr = alloca double, align 8
  store double %__x, ptr %__x.addr, align 8
  ret void
}
