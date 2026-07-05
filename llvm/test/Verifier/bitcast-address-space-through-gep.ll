; RUN: not llvm-as -disable-output %s 2>&1 | FileCheck %s

target datalayout = "e-p:32:32:32-p1:16:16:16-p2:8:8:8-i1:8:32-i8:8:32-i16:16:32-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:32:64-v128:32:128-a0:0:32-n8:16:32"

%struct.Foo = type { ptr addrspace(1) }


@as2_array = addrspace(2) global [32 x i32] zeroinitializer

; CHECK: error: invalid cast opcode for cast from 'ptr addrspace(2)' to 'ptr addrspace(1)'

; Make sure we still reject the bitcast after the value is accessed through a GEP
@bitcast_after_gep = global %struct.Foo { ptr bitcast (ptr addrspace(2) getelementptr ([32 x i32], ptr addrspace(2) @as2_array, i32 0, i32 8) to ptr addrspace(1)) }


