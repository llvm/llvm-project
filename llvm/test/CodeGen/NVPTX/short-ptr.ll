; RUN: llc < %s -march=nvptx64 -mcpu=sm_20 | FileCheck %s --check-prefix CHECK-DEFAULT
; RUN: llc < %s -march=nvptx -mcpu=sm_20 | FileCheck %s --check-prefix CHECK-DEFAULT-32
; RUN: llc < %s -march=nvptx64 -mcpu=sm_20 -nvptx-short-ptr | FileCheck %s --check-prefixes CHECK-SHORT-SHARED,CHECK-SHORT-CONST

; RUN: %if ptxas %{ llc < %s -march=nvptx -mcpu=sm_20 | %ptxas-verify %}
; RUN: %if ptxas %{ llc < %s -march=nvptx64 -mcpu=sm_20 | %ptxas-verify %}
; RUN: %if ptxas %{ llc < %s -march=nvptx64 -mcpu=sm_20 -nvptx-short-ptr | %ptxas-verify %}

; CHECK-DEFAULT: .visible .shared .align 8 .u64 s
; CHECK-DEFAULT-32: .visible .shared .align 8 .u32 s
; CHECK-SHORT-SHARED: .visible .shared .align 8 .u32 s
@s = local_unnamed_addr addrspace(3) global i32 addrspace(3)* null, align 8

; CHECK-DEFAULT: .visible .const .align 8 .u64 c
; CHECK-DEFAULT-32: .visible .const .align 8 .u32 c
; CHECK-SHORT-CONST: .visible .const .align 8 .u32 c
@c = local_unnamed_addr addrspace(4) global i32 addrspace(4)* null, align 8
