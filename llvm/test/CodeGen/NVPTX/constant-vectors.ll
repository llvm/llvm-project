; RUN: llc < %s -march=nvptx64 -mcpu=sm_20 | FileCheck %s
; RUN: %if ptxas %{ llc < %s -march=nvptx64 -mcpu=sm_20 | %ptxas-verify %}

target triple = "nvptx-nvidia-cuda"

; CHECK: .visible .global .align 16 .b8 testArray[8] = {0, 1, 2, 3, 4, 5, 6, 7};
@testArray = constant [2 x <4 x i8>] [<4 x i8> <i8 0, i8 1, i8 2, i8 3>, <4 x i8> <i8 4, i8 5, i8 6, i8 7>], align 16
