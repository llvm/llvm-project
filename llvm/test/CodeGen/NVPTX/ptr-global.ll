; RUN: llc < %s -mtriple=nvptx -mcpu=sm_20 | FileCheck %s --check-prefix=PTX32
; RUN: llc < %s -mtriple=nvptx64 -mcpu=sm_20 | FileCheck %s --check-prefix=PTX64
; RUN: %if ptxas-ptr32 %{ llc < %s -mtriple=nvptx -mcpu=sm_20 | %ptxas-verify %}
; RUN: %if ptxas %{ llc < %s -mtriple=nvptx64 -mcpu=sm_20 | %ptxas-verify %}

; Pointer globals use the same unsigned integer type in declarations and
; definitions, with or without an initializer.
; PTX32-DAG: .extern .global .align 4 .u32 ptr_decl;
; PTX64-DAG: .extern .global .align 8 .u64 ptr_decl;
@ptr_decl = external addrspace(1) global ptr

; PTX32-DAG: .visible .global .align 4 .u32 ptr_null;
; PTX64-DAG: .visible .global .align 8 .u64 ptr_null;
@ptr_null = addrspace(1) global ptr null

; PTX32-DAG: .visible .global .align 4 .u32 value;
; PTX64-DAG: .visible .global .align 4 .u32 value;
@value = addrspace(1) global i32 0

; PTX32-DAG: .visible .global .align 4 .u32 ptr_init = value;
; PTX64-DAG: .visible .global .align 8 .u64 ptr_init = value;
@ptr_init = addrspace(1) global ptr addrspace(1) @value
