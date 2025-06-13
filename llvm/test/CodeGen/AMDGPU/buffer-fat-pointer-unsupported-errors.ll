; RUN: split-file %s %t
; RUN: not opt -mtriple=amdgcn-amd-amdhsa -disable-output -passes=amdgpu-lower-buffer-fat-pointers %t/contains-null-init.ll 2>&1 | FileCheck -check-prefix=ERR0 %s
; RUN: not opt -mtriple=amdgcn-amd-amdhsa -disable-output -passes=amdgpu-lower-buffer-fat-pointers %t/contains-poison-init.ll 2>&1 | FileCheck -check-prefix=ERR1 %s
; RUN: not opt -mtriple=amdgcn-amd-amdhsa -disable-output -passes=amdgpu-lower-buffer-fat-pointers %t/defined-gv-type.ll 2>&1 | FileCheck -check-prefix=ERR2 %s
; RUN: not opt -mtriple=amdgcn-amd-amdhsa -disable-output -passes=amdgpu-lower-buffer-fat-pointers %t/declared-gv-type.ll 2>&1 | FileCheck -check-prefix=ERR3 %s

;--- contains-null-init.ll
; ERR0: error: global variables that contain buffer fat pointers (address space 7 pointers) are unsupported. Use buffer resource pointers (address space 8) instead
@init_null = global ptr addrspace(7) null

;--- contains-poison-init.ll
; ERR1: error: global variables that contain buffer fat pointers (address space 7 pointers) are unsupported. Use buffer resource pointers (address space 8) instead
@init_poison = global ptr addrspace(7) poison

;--- defined-gv-type.ll
; ERR2: error: global variables with a buffer fat pointer address space (7) are not supported
@gv_is_addrspace_7 = addrspace(7) global i32 0

;--- declared-gv-type.ll
; ERR3: error: global variables with a buffer fat pointer address space (7) are not supported
@extern_gv_is_addrspace_7 = external addrspace(7) global i32

