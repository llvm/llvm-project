; RUN: not llvm-as -disable-output %s 2>&1 | FileCheck %s

target datalayout = "e-p:32:32:32-p1:16:16:16-p2:8:8:8-i1:8:32-i8:8:32-i16:16:32-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:32:64-v128:32:128-a0:0:32-n8:16:32"

; CHECK:  error: invalid cast opcode for cast from 'ptr addrspace(1)' to 'ptr addrspace(2)'

; Check that we can find inttoptr -> illegal bitcasts when hidden
; inside constantexpr pointer operands
define ptr addrspace(2) @illegal_bitcast_inttoptr_as_1_to_2_inside_gep() {
  %cast = getelementptr i32, ptr addrspace(2) bitcast (ptr addrspace(1) inttoptr (i32 1234 to ptr addrspace(1)) to ptr addrspace(2)), i32 3
  ret ptr addrspace(2) %cast
}

