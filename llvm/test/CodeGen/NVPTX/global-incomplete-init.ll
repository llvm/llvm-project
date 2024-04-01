; RUN: llc < %s -march=nvptx64 -mcpu=sm_50 | FileCheck %s
; RUN: %if ptxas %{ llc < %s -march=nvptx64 -mcpu=sm_50 | %ptxas-verify %}

; Make sure the globals constants have trailing zeros properly trimmed

; basic case
; CHECK-DAG: .b8 A[8] = {3, 4, 0, 0, 5};
@A = global [8 x i8] [i8 3, i8 4, i8 0, i8 0, i8 5, i8 0, i8 0, i8 0]

; all-zeros
; CHECK-DAG: .b8 B[2];
@B = global [2 x i8] [i8 0, i8 0]

; all-non-zeros
; CHECK-DAG: .b8 C[4] = {1, 2, 3, 4};
@C = global [4 x i8] [i8 1, i8 2, i8 3, i8 4]

; initializer with a symbol, the last 0 could be default initialized
; CHECK-DAG: .u8 e = 1;
; CHECK-DAG: .u64 D[4] = {e, 0, e, 0};
@e = addrspace(1) global i8 1
@D = addrspace(1) global [4 x ptr addrspace(1)] [ptr addrspace(1) @e, ptr addrspace(1) null, ptr addrspace(1) @e, ptr addrspace(1) null]
