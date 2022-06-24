; RUN: llvm-as --opaque-pointers=0 < %s | llvm-dis --opaque-pointers=0 | FileCheck %s
; RUN: verify-uselistorder --opaque-pointers=0 %s
; PR1358

; CHECK: icmp ne (i32 (...)* @test_weak, i32 (...)* null)
@G = global i1 icmp ne (i32 (...)* @test_weak, i32 (...)* null)

declare extern_weak i32 @test_weak(...)

