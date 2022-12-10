; RUN: opt -S -passes=globalopt < %s | FileCheck %s

; Gracefully handle undef global_ctors/global_dtors

; CHECK: @llvm.global_ctors = appending global [0 x { i32, void ()*, i8* }] undef
; CHECK: @llvm.global_dtors = appending global [0 x { i32, void ()*, i8* }] undef

@llvm.global_ctors = appending global [0 x { i32, void ()*, i8* }] undef
@llvm.global_dtors = appending global [0 x { i32, void ()*, i8* }] undef
