; RUN: llc < %s -march=nvptx -mcpu=sm_20 | FileCheck %s
; RUN: llc < %s -march=nvptx64 -mcpu=sm_20 | FileCheck %s

; Test that we don't crash if we're compiling a module with function references,
; but without any functions in it.

target datalayout = "e-i64:64-i128:128-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

@Funcs = local_unnamed_addr addrspace(1) externally_initialized
         global [1 x void (i8*)*] [void (i8*)* @func], align 8

declare void @func(i8*)

; CHECK: Funcs[1] = {func}
