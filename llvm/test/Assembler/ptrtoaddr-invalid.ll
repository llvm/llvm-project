;; Check all requirements on the ptrtoaddr instruction operands
;; Most of these invalid cases are detected at parse time but some are only
;; detected at verification time (see Verifier::visitPtrToAddrInst())
; RUN: rm -rf %t && split-file --leading-lines %s %t

;--- src_vec_dst_no_vec.ll
; RUN: not llvm-as %t/src_vec_dst_no_vec.ll -o /dev/null 2>&1 | FileCheck -check-prefix=SRC_VEC_DST_NO_VEC %s --implicit-check-not="error:"
define i64 @bad(<2 x ptr> %p) {
  %addr = ptrtoaddr <2 x ptr> %p to i64
  ; SRC_VEC_DST_NO_VEC: [[#@LINE-1]]:21: error: invalid cast opcode for cast from '<2 x ptr>' to 'i64'
  ret i64 %addr
}

;--- src_no_vec_dst_vec.ll
; RUN: not llvm-as %t/src_no_vec_dst_vec.ll -o /dev/null 2>&1 | FileCheck -check-prefix=SRC_NO_VEC_DST_VEC %s --implicit-check-not="error:"
define <2 x i64> @bad(ptr %p) {
  %addr = ptrtoaddr ptr %p to <2 x i64>
  ; SRC_NO_VEC_DST_VEC: [[#@LINE-1]]:21: error: invalid cast opcode for cast from 'ptr' to '<2 x i64>'
  ret <2 x i64> %addr
}

;--- dst_not_int.ll
; RUN: not llvm-as %t/dst_not_int.ll -o /dev/null 2>&1 | FileCheck -check-prefix=DST_NOT_INT %s --implicit-check-not="error:"
define float @bad(ptr %p) {
  %addr = ptrtoaddr ptr %p to float
  ; DST_NOT_INT: [[#@LINE-1]]:21: error: invalid cast opcode for cast from 'ptr' to 'float'
  ret float %addr
}

;--- dst_not_int_vec.ll
; RUN: not llvm-as %t/dst_not_int_vec.ll -o /dev/null 2>&1 | FileCheck -check-prefix=DST_NOT_INT_VEC %s --implicit-check-not="error:"
define <2 x float> @bad(<2 x ptr> %p) {
  %addr = ptrtoaddr <2 x ptr> %p to <2 x float>
  ; DST_NOT_INT_VEC: [[#@LINE-1]]:21: error: invalid cast opcode for cast from '<2 x ptr>' to '<2 x float>'
  ret <2 x float> %addr
}

;--- src_not_ptr.ll
; RUN: not llvm-as %t/src_not_ptr.ll -o /dev/null 2>&1 | FileCheck -check-prefix=SRC_NOT_PTR %s --implicit-check-not="error:"
define i64 @bad(i32 %p) {
  %addr = ptrtoaddr i32 %p to i64
  ; SRC_NOT_PTR: [[#@LINE-1]]:21: error: invalid cast opcode for cast from 'i32' to 'i64'
  ret i64 %addr
}

;--- src_not_ptr_vec.ll
; RUN: not llvm-as %t/src_not_ptr_vec.ll -o /dev/null 2>&1 | FileCheck -check-prefix=SRC_NOT_PTR_VEC %s --implicit-check-not="error:"
define <2 x i64> @bad(<2 x i32> %p) {
  %addr = ptrtoaddr <2 x i32> %p to <2 x i64>
  ; SRC_NOT_PTR_VEC: [[#@LINE-1]]:21: error: invalid cast opcode for cast from '<2 x i32>' to '<2 x i64>'
  ret <2 x i64> %addr
}

;--- vec_src_fewer_elems.ll
; RUN: not llvm-as %t/vec_src_fewer_elems.ll -o /dev/null 2>&1 | FileCheck -check-prefix=VEC_SRC_FEWER_ELEMS %s --implicit-check-not="error:"
define <4 x i64> @bad(<2 x ptr> %p) {
  %addr = ptrtoaddr <2 x ptr> %p to <4 x i64>
  ; VEC_SRC_FEWER_ELEMS: [[#@LINE-1]]:21: error: invalid cast opcode for cast from '<2 x ptr>' to '<4 x i64>'
  ret <4 x i64> %addr
}

;--- vec_dst_fewer_elems.ll
; RUN: not llvm-as %t/vec_dst_fewer_elems.ll -o /dev/null 2>&1 | FileCheck -check-prefix=VEC_DST_FEWER_ELEMS %s --implicit-check-not="error:"
define <2 x i64> @bad(<4 x ptr> %p) {
  %addr = ptrtoaddr <4 x ptr> %p to <2 x i64>
  ; VEC_DST_FEWER_ELEMS: [[#@LINE-1]]:21: error: invalid cast opcode for cast from '<4 x ptr>' to '<2 x i64>'
  ret <2 x i64> %addr
}

;--- dst_not_addr_size.ll
; The following invalid IR is caught by the verifier, not the parser:
; RUN: llvm-as %t/dst_not_addr_size.ll --disable-output --disable-verify
; RUN: not llvm-as %t/dst_not_addr_size.ll -o /dev/null 2>&1 | FileCheck -check-prefix=DST_NOT_ADDR_SIZE %s --implicit-check-not="error:"
; DST_NOT_ADDR_SIZE: assembly parsed, but does not verify as correct!
define i32 @bad(ptr %p) {
  %addr = ptrtoaddr ptr %p to i32
  ; DST_NOT_ADDR_SIZE: PtrToAddr result must be address width
  ret i32 %addr
}
define <4 x i32> @bad_vec(<4 x ptr> %p) {
  %addr = ptrtoaddr <4 x ptr> %p to <4 x i32>
  ; DST_NOT_ADDR_SIZE: PtrToAddr result must be address width
  ret <4 x i32> %addr
}
