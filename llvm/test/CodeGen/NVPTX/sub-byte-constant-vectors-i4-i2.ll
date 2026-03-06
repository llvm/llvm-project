; Test to verify that NVPTX backend correctly handles constant global vectors
; containing sub-byte sized elements.

; RUN: llc < %s -mtriple=nvptx64 | FileCheck %s
; RUN: %if ptxas %{ llc < %s -mtriple=nvptx64 | %ptxas-verify %}

target triple = "nvptx-nvidia-cuda"

; CHECK: .visible .global .align 1 .b8 test0[1] = {33};
@test0 = local_unnamed_addr addrspace(1) constant <2 x i4> <i4 1, i4 2>, align 1

; CHECK: .visible .global .align 1 .b8 test1[2] = {33, 3};
@test1 = local_unnamed_addr addrspace(1) constant <3 x i4> <i4 1, i4 2, i4 3>, align 1

; CHECK: .visible .global .align 1 .b8 test2[1] = {228};
@test2 = local_unnamed_addr addrspace(1) constant <4 x i2> <i2 0, i2 1, i2 2, i2 3>, align 1
