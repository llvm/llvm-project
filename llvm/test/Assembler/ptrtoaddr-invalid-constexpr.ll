;; Check all requirements on the ptrtoaddr constant expression operands
;; Most of these invalid cases are detected at parse time but some are only
;; detected at verification time (see Verifier::visitPtrToAddrInst())
; RUN: rm -rf %t && split-file --leading-lines %s %t

;--- src_vec_dst_no_vec.ll
; RUN: not llvm-as %t/src_vec_dst_no_vec.ll -o /dev/null 2>&1 | FileCheck -check-prefix=SRC_VEC_DST_NO_VEC %s --implicit-check-not="error:"
@g = global i64 ptrtoaddr (<2 x ptr> <ptr @g, ptr @g> to i64)
; SRC_VEC_DST_NO_VEC: [[#@LINE-1]]:17: error: invalid cast opcode for cast from '<2 x ptr>' to 'i64'

;--- src_no_vec_dst_vec.ll
; RUN: not llvm-as %t/src_no_vec_dst_vec.ll -o /dev/null 2>&1 | FileCheck -check-prefix=SRC_NO_VEC_DST_VEC %s --implicit-check-not="error:"
@g = global <2 x i64> ptrtoaddr (ptr @g to <2 x i64>)
; SRC_NO_VEC_DST_VEC: [[#@LINE-1]]:23: error: invalid cast opcode for cast from 'ptr' to '<2 x i64>'

;--- dst_not_int.ll
; RUN: not llvm-as %t/dst_not_int.ll -o /dev/null 2>&1 | FileCheck -check-prefix=DST_NOT_INT %s --implicit-check-not="error:"
@g = global float ptrtoaddr (ptr @g to float)
; DST_NOT_INT: [[#@LINE-1]]:19: error: invalid cast opcode for cast from 'ptr' to 'float'

;--- dst_not_int_vec.ll
; RUN: not llvm-as %t/dst_not_int_vec.ll -o /dev/null 2>&1 | FileCheck -check-prefix=DST_NOT_INT_VEC %s --implicit-check-not="error:"
@g = global <2 x float> ptrtoaddr (<2 x ptr> <ptr @g, ptr @g> to <2 x float>)
; DST_NOT_INT_VEC: [[#@LINE-1]]:25: error: invalid cast opcode for cast from '<2 x ptr>' to '<2 x float>'

;--- src_not_ptr.ll
; RUN: not llvm-as %t/src_not_ptr.ll -o /dev/null 2>&1 | FileCheck -check-prefix=SRC_NOT_PTR %s --implicit-check-not="error:"
@g = global i64 ptrtoaddr (i32 1 to i64)
; SRC_NOT_PTR: [[#@LINE-1]]:17: error: invalid cast opcode for cast from 'i32' to 'i64'

;--- src_not_ptr_vec.ll
; RUN: not llvm-as %t/src_not_ptr_vec.ll -o /dev/null 2>&1 | FileCheck -check-prefix=SRC_NOT_PTR_VEC %s --implicit-check-not="error:"
@g = global <2 x i64> ptrtoaddr (<2 x i32> <i32 1, i32 2> to <2 x i64>)
; SRC_NOT_PTR_VEC: [[#@LINE-1]]:23: error: invalid cast opcode for cast from '<2 x i32>' to '<2 x i64>'

;--- vec_src_fewer_elems.ll
; RUN: not llvm-as %t/vec_src_fewer_elems.ll -o /dev/null 2>&1 | FileCheck -check-prefix=VEC_SRC_FEWER_ELEMS %s --implicit-check-not="error:"
@g = global <4 x i64> ptrtoaddr (<2 x ptr> <ptr @g, ptr @g> to <4 x i64>)
; VEC_SRC_FEWER_ELEMS: [[#@LINE-1]]:23: error: invalid cast opcode for cast from '<2 x ptr>' to '<4 x i64>'

;--- vec_dst_fewer_elems.ll
; RUN: not llvm-as %t/vec_dst_fewer_elems.ll -o /dev/null 2>&1 | FileCheck -check-prefix=VEC_DST_FEWER_ELEMS %s --implicit-check-not="error:"
@g = global <2 x i64> ptrtoaddr (<4 x ptr> <ptr @g, ptr @g, ptr @g, ptr @g> to <2 x i64>)
; VEC_DST_FEWER_ELEMS: [[#@LINE-1]]:23: error: invalid cast opcode for cast from '<4 x ptr>' to '<2 x i64>'

;--- dst_not_addr_size.ll
; The following invalid IR is caught by the verifier, not the parser:
; RUN: llvm-as %t/dst_not_addr_size.ll --disable-output --disable-verify
; RUN: not llvm-as %t/dst_not_addr_size.ll -o /dev/null 2>&1 | FileCheck -check-prefix=DST_NOT_ADDR_SIZE %s --implicit-check-not="error:"
; DST_NOT_ADDR_SIZE: assembly parsed, but does not verify as correct!
@g = global i32 ptrtoaddr (ptr @g to i32)
; DST_NOT_ADDR_SIZE-NEXT: PtrToAddr result must be address width
; DST_NOT_ADDR_SIZE-NEXT: i32 ptrtoaddr (ptr @g to i32)
@g_vec = global <4 x i32> ptrtoaddr (<4 x ptr> <ptr @g, ptr @g, ptr @g, ptr @g> to <4 x i32>)
; TODO: Verifier.cpp does not visit ConstantVector/ConstantStruct values
; TODO-DST_NOT_ADDR_SIZE: PtrToAddr result must be address width
