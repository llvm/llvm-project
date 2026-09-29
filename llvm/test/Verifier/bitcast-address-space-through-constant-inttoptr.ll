; RUN: not llvm-as -disable-output %s 2>&1 | FileCheck %s

target datalayout = "e-p:32:32:32-p1:16:16:16-p2:8:8:8-i1:8:32-i8:8:32-i16:16:32-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:32:64-v128:32:128-a0:0:32-n8:16:32"


%struct.Foo = type { ptr addrspace(1) }

; CHECK: error: invalid cast opcode for cast from 'ptr addrspace(2)' to 'ptr addrspace(1)'

; Make sure we still reject the bitcast when the source is a inttoptr (constant int) in a global initializer
@bitcast_after_constant_inttoptr_initializer = global %struct.Foo { ptr addrspace(1) bitcast (ptr addrspace(2) inttoptr (i8 7 to ptr addrspace(2)) to ptr addrspace(1)) }


