; RUN: not llvm-as < %s -o /dev/null 2>&1 | FileCheck %s

; CHECK: Function takes x86_bsr but isn't an intrinsic
define void @takes_bsr(x86_bsr %b) {
  ret void
}
