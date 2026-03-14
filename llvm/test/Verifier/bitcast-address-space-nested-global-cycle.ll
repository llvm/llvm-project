; RUN: not llvm-as -disable-output %s 2>&1 | FileCheck %s

; CHECK: error: invalid cast opcode for cast from 'ptr' to 'ptr addrspace(1)'

target datalayout = "e-p:32:32:32-p1:16:16:16-p2:8:8:8-i1:8:32-i8:8:32-i16:16:32-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:32:64-v128:32:128-a0:0:32-n8:16:32"

%struct.Self1 = type { ptr addrspace(1) }

@cycle1 = addrspace(1) constant %struct.Self1 { ptr addrspace(1) bitcast (ptr @cycle0 to ptr addrspace(1)) }
@cycle0 = addrspace(0) constant %struct.Self1 { ptr addrspace(1) @cycle1 }
