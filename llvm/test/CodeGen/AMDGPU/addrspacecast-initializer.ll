; RUN: llc -march=amdgcn -verify-machineinstrs < %s | FileCheck %s

; CHECK: global.arr:
; CHECK: .zero 1024
; CHECK: .size global.arr, 1024

; CHECK: gv_flatptr_from_global:
; CHECK: .quad global.arr+32
; CHECK: .size gv_flatptr_from_global, 8

; CHECK: gv_global_ptr:
; CHECK: .quad global.arr+32
; CHECK: .size gv_global_ptr, 8

; CHECK: gv_flatptr_from_constant:
; CHECK: .quad constant.arr+32
; CHECK: .size	gv_flatptr_from_constant, 8

@global.arr = unnamed_addr addrspace(1) global [256 x i32] undef, align 4
@constant.arr = external unnamed_addr addrspace(4) global [256 x i32], align 4

@gv_flatptr_from_global = unnamed_addr addrspace(4) global i32 addrspace(0)* getelementptr ([256 x i32], [256 x i32] addrspace(0)* addrspacecast ([256 x i32] addrspace(1)* @global.arr to [256 x i32] addrspace(0)*), i64 0, i64 8), align 4


@gv_global_ptr = unnamed_addr addrspace(4) global i32 addrspace(1)* getelementptr ([256 x i32], [256 x i32] addrspace(1)* @global.arr, i64 0, i64 8), align 4

@gv_flatptr_from_constant = unnamed_addr addrspace(4) global i32 addrspace(0)* getelementptr ([256 x i32], [256 x i32] addrspace(0)* addrspacecast ([256 x i32] addrspace(4)* @constant.arr to [256 x i32] addrspace(0)*), i64 0, i64 8), align 4
