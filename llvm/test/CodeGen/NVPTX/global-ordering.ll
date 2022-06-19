; RUN: llc < %s -march=nvptx -mcpu=sm_20 | FileCheck %s --check-prefix=PTX32
; RUN: llc < %s -march=nvptx64 -mcpu=sm_20 | FileCheck %s --check-prefix=PTX64
; RUN: %if ptxas %{ llc < %s -march=nvptx -mcpu=sm_20 | %ptxas-verify %}
; RUN: %if ptxas %{ llc < %s -march=nvptx64 -mcpu=sm_20 | %ptxas-verify %}

; Make sure we emit these globals in def-use order


; PTX32:      .visible .global .align 1 .u8 a = 2;
; PTX32-NEXT: .visible .global .align 4 .u32 a2 = a;
; PTX64:      .visible .global .align 1 .u8 a = 2;
; PTX64-NEXT: .visible .global .align 8 .u64 a2 = a;
@a2 = addrspace(1) global i8 addrspace(1)* @a
@a = addrspace(1) global i8 2


; PTX32:      .visible .global .align 1 .u8 b = 1;
; PTX32-NEXT: .visible .global .align 4 .u32 b2[2] = {b, b};
; PTX64:      .visible .global .align 1 .u8 b = 1;
; PTX64-NEXT: .visible .global .align 8 .u64 b2[2] = {b, b};
@b2 = addrspace(1) global [2 x i8 addrspace(1)*] [i8 addrspace(1)* @b, i8 addrspace(1)* @b]
@b = addrspace(1) global i8 1
