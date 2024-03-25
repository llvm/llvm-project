; RUN: llc < %s -O0 -march=nvptx64 -mcpu=sm_20 | FileCheck %s
; RUN: %if ptxas %{ llc < %s -O0 -march=nvptx64 -mcpu=sm_20 | %ptxas-verify %}

; CHECK: .visible .global .align 16 .b8 G1[16] = {1};
@G1 = global i128 1

; CHECK: .visible .global .align 16 .b8 G2[16];
@G2 = global i128 0
