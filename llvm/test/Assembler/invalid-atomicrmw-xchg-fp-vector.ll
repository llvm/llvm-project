; RUN: not llvm-as -disable-output %s 2>&1 | FileCheck %s

; CHECK: error: atomicrmw xchg operand must be an integer, floating point, or pointer type
define <2 x half> @fp_vector_atomicrmw(ptr %x, <2 x half> %val) {
  %atomic.xchg = atomicrmw xchg ptr %x, <2 x half> %val seq_cst
  ret <2 x half> %atomic.xchg
}
