; RUN: llc < %s -mtriple=nvptx64 -mcpu=sm_20 | FileCheck %s
; RUN: %if ptxas %{ llc < %s -mtriple=nvptx64 -mcpu=sm_20 | %ptxas-verify %}

target datalayout = "e-i64:64-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

@g1 = external global <4 x i32> ; external global variable
; CHECK: .extern .global .align 16 .b8 g1[16];
@g2 = global <4 x i32> zeroinitializer ; module-level global variable
; CHECK: .visible .global .align 16 .b8 g2[16];
