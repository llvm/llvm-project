; RUN: not llvm-as -disable-output %s 2>&1 | FileCheck %s

; CHECK: error: invalid cast opcode for cast from 'ptr' to 'ptr addrspace(1)'

target datalayout = "e-p:32:32:32-p1:16:16:16-p2:8:8:8-i1:8:32-i8:8:32-i16:16:32-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:32:64-v128:32:128-a0:0:32-n8:16:32"


%struct.Self1 = type { ptr addrspace(1) }

@nestedD = constant %struct.Self1 { ptr addrspace(1) bitcast (ptr @nestedC to ptr addrspace(1)) }
@nestedC = constant %struct.Self1 { ptr addrspace(1) bitcast (ptr @nestedB to ptr addrspace(1)) }
@nestedB = constant %struct.Self1 { ptr addrspace(1) bitcast (ptr @nestedA to ptr addrspace(1)) }
@nestedA = constant %struct.Self1 { ptr addrspace(1) null }
