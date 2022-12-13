; RUN: llvm-as < %s | llvm-dis | FileCheck %s
; RUN: verify-uselistorder %s
; PR1358

; CHECK: icmp ne (ptr @test_weak, ptr null)
@G = global i1 icmp ne (ptr @test_weak, ptr null)

declare extern_weak i32 @test_weak(...)

