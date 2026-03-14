; RUN: not llvm-as -disable-output %s 2>&1 | FileCheck %s

target datalayout = "e-p:32:32:32-p1:16:16:16-p2:32:32:32-i1:8:32-i8:8:32-i16:16:32-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:32:64-v128:32:128-a0:0:32-n8:16:32"

; CHECK: error: invalid cast opcode for cast from '<4 x ptr addrspace(1)>' to '<2 x ptr addrspace(2)>'

; The pointer in addrspace 1 of the size 16 while pointer in addrspace 2 of the size 32.
; Converting 4 element array of pointers from addrspace 2 to 2 element array in addrspace 2
; has the same total bit length but bitcast still does not allow conversion into
; different addrspace.
define <2 x ptr addrspace(2)> @vector_illegal_bitcast_as_1_to_2(<4 x ptr addrspace(1)> %p) {
   %cast = bitcast <4 x ptr addrspace(1)> %p to <2 x ptr addrspace(2)>
   ret <2 x ptr addrspace(2)> %cast
}

