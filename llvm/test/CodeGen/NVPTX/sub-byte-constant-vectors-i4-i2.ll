; Test to verify that NVPTX backend correctly handles constant global vectors
; containing sub-byte sized elements.

; RUN: llc < %s -mtriple=nvptx64 | FileCheck %s
; RUN: %if ptxas %{ llc < %s -mtriple=nvptx64 | %ptxas-verify %}

target triple = "nvptx-nvidia-cuda"

; Tests for vectors forming one complete byte.

; CHECK: .visible .global .align 1 .b8 test0[1] = {33};
@test0 = local_unnamed_addr addrspace(1) constant <2 x i4> <i4 1, i4 2>, align 1

; CHECK: .visible .global .align 1 .b8 test1[1] = {228};
@test1 = local_unnamed_addr addrspace(1) constant <4 x i2> <i2 0, i2 1, i2 2, i2 3>, align 1

; Tests for vectors forming multiple complete bytes.

; CHECK: .visible .global .align 1 .b8 test2[2] = {33, 67};
@test2 = local_unnamed_addr addrspace(1) constant <4 x i4> <i4 1, i4 2, i4 3, i4 4>, align 1

; CHECK: .visible .global .align 1 .b8 test3[2] = {228, 228};
@test3 = local_unnamed_addr addrspace(1) constant <8 x i2> <i2 0, i2 1, i2 2, i2 3, i2 0, i2 1, i2 2, i2 3>, align 1

; Tests for unevenly sized vectors which requires tail padding.

; CHECK: .visible .global .align 1 .b8 test4[1] = {1};
@test4 = local_unnamed_addr addrspace(1) constant <1 x i4> <i4 1>, align 1

; CHECK: .visible .global .align 1 .b8 test5[2] = {33, 3};
@test5 = local_unnamed_addr addrspace(1) constant <3 x i4> <i4 1, i4 2, i4 3>, align 1

; CHECK: .visible .global .align 1 .b8 test6[3] = {33, 33, 1};
@test6 = local_unnamed_addr addrspace(1) constant <5 x i4> <i4 1, i4 2, i4 1, i4 2, i4 1>, align 1

; CHECK: .visible .global .align 1 .b8 test7[2] = {228, 11};
@test7 = local_unnamed_addr addrspace(1) constant <6 x i2> <i2 0, i2 1, i2 2, i2 3, i2 3, i2 2>, align 1

; CHECK: .visible .global .align 1 .b8 test8[3] = {228, 228, 11};
@test8 = local_unnamed_addr addrspace(1) constant <10 x i2> <i2 0, i2 1, i2 2, i2 3, i2 0, i2 1, i2 2, i2 3, i2 3, i2 2>, align 1
