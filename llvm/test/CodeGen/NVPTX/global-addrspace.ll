; RUN: llc < %s -march=nvptx -mcpu=sm_20 | FileCheck %s --check-prefix=PTX32
; RUN: llc < %s -march=nvptx64 -mcpu=sm_20 | FileCheck %s --check-prefix=PTX64
; RUN: %if ptxas %{ llc < %s -march=nvptx -mcpu=sm_20 | %ptxas-verify %}
; RUN: %if ptxas %{ llc < %s -march=nvptx64 -mcpu=sm_20 | %ptxas-verify %}

; PTX32: .visible .global .align 4 .u32 i;
; PTX32: .visible .const .align 4 .u32 j;
; PTX32: .visible .shared .align 4 .u32 k;
; PTX64: .visible .global .align 4 .u32 i;
; PTX64: .visible .const .align 4 .u32 j;
; PTX64: .visible .shared .align 4 .u32 k;
@i = addrspace(1) externally_initialized global i32 0, align 4
@j = addrspace(4) externally_initialized global i32 0, align 4
@k = addrspace(3) global i32 undef, align 4
