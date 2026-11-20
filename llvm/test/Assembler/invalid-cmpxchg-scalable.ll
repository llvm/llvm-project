; RUN: split-file %s %t --leading-lines
; RUN: not llvm-as < %t/scalable_int_vector_cmpxchg.ll 2>&1 | FileCheck %s --check-prefix=INT
; RUN: not llvm-as < %t/scalable_fp_vector_cmpxchg.ll 2>&1 | FileCheck %s --check-prefix=FP
; RUN: not llvm-as < %t/scalable_ptr_vector_cmpxchg.ll 2>&1 | FileCheck %s --check-prefix=PTR

;--- scalable_int_vector_cmpxchg.ll
define void @scalable_int_vector_cmpxchg(ptr %p, <vscale x 2 x i16> %cmp, <vscale x 2 x i16> %new) {
; INT: error: cmpxchg operand may not be scalable
  %val_success = cmpxchg ptr %p, <vscale x 2 x i16> %cmp, <vscale x 2 x i16> %new seq_cst monotonic
  ret void
}

;--- scalable_fp_vector_cmpxchg.ll
define void @scalable_fp_vector_cmpxchg(ptr %p, <vscale x 2 x half> %cmp, <vscale x 2 x half> %new) {
; FP: error: cmpxchg operand may not be scalable
  %val_success = cmpxchg ptr %p, <vscale x 2 x half> %cmp, <vscale x 2 x half> %new seq_cst monotonic
  ret void
}

;--- scalable_ptr_vector_cmpxchg.ll
define void @scalable_ptr_vector_cmpxchg(ptr %p, <vscale x 2 x ptr> %cmp, <vscale x 2 x ptr> %new) {
; PTR: error: cmpxchg operand may not be scalable
  %val_success = cmpxchg ptr %p, <vscale x 2 x ptr> %cmp, <vscale x 2 x ptr> %new seq_cst monotonic
  ret void
}
