; RUN: split-file %s %t --leading-lines
; RUN: not llvm-as < %t/scalable_fp_vector_atomicrmw_xchg.ll 2>&1 | FileCheck -check-prefix=ERR0 %s
; RUN: not llvm-as < %t/scalable_int_vector_atomicrmw_xchg.ll 2>&1 | FileCheck -check-prefix=ERR1 %s
; RUN: not llvm-as < %t/scalable_ptr_vector_atomicrmw_xchg.ll 2>&1 | FileCheck -check-prefix=ERR2 %s
; RUN: not llvm-as < %t/scalable_fp_vector_atomicrmw_fadd.ll 2>&1 | FileCheck -check-prefix=ERR3 %s
; RUN: not llvm-as < %t/scalable_int_vector_atomicrmw_add.ll 2>&1 | FileCheck -check-prefix=ERR4 %s

;--- scalable_fp_vector_atomicrmw_xchg.ll
define <vscale x 2 x half> @scalable_fp_vector_atomicrmw_xchg(ptr %x, <vscale x 2 x half> %val) {
; ERR0: :41: error: atomicrmw operand may not be scalable
  %atomic.xchg = atomicrmw xchg ptr %x, <vscale x 2 x half> %val seq_cst
  ret <vscale x 2 x half> %atomic.xchg
}

;--- scalable_int_vector_atomicrmw_xchg.ll
define <vscale x 2 x i16> @scalable_int_vector_atomicrmw_xchg(ptr %x, <vscale x 2 x i16> %val) {
; ERR1: :41: error: atomicrmw operand may not be scalable
  %atomic.xchg = atomicrmw xchg ptr %x, <vscale x 2 x i16> %val seq_cst
  ret <vscale x 2 x i16> %atomic.xchg
}

;--- scalable_ptr_vector_atomicrmw_xchg.ll
define <vscale x 2 x ptr> @scalable_ptr_vector_atomicrmw_xchg(ptr %x, <vscale x 2 x ptr> %val) {
; ERR2: :41: error: atomicrmw operand may not be scalable
  %atomic.xchg = atomicrmw xchg ptr %x, <vscale x 2 x ptr> %val seq_cst
  ret <vscale x 2 x ptr> %atomic.xchg
}

;--- scalable_fp_vector_atomicrmw_fadd.ll
define <vscale x 2 x half> @scalable_fp_vector_atomicrmw_fadd(ptr %x, <vscale x 2 x half> %val) {
; ERR3: :41: error: atomicrmw operand may not be scalable
  %atomic.fadd = atomicrmw fadd ptr %x, <vscale x 2 x half> %val seq_cst
  ret <vscale x 2 x half> %atomic.fadd
}

;--- scalable_int_vector_atomicrmw_add.ll
define <vscale x 2 x i16> @scalable_int_vector_atomicrmw_add(ptr %x, <vscale x 2 x i16> %val) {
; ERR4: :39: error: atomicrmw operand may not be scalable
  %atomic.add = atomicrmw add ptr %x, <vscale x 2 x i16> %val seq_cst
  ret <vscale x 2 x i16> %atomic.add
}
