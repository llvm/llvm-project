; RUN: llc < %s -asm-verbose=false -wasm-keep-registers | FileCheck %s

target triple = "wasm32-unknown-unknown"

@.str = private unnamed_addr constant [8 x i8] c"custom0\00", section "llvm.metadata"
@.str.1 = private unnamed_addr constant [7 x i8] c"main.c\00", section "llvm.metadata"
@.str.2 = private unnamed_addr constant [8 x i8] c"custom1\00", section "llvm.metadata"
@.str.3 = private unnamed_addr constant [8 x i8] c"custom2\00", section "llvm.metadata"
@llvm.global.annotations = appending global [3 x { ptr, ptr, ptr, i32, ptr }] [{ ptr, ptr, ptr, i32, ptr } { ptr @test0, ptr @.str, ptr @.str.1, i32 4, ptr null }, { ptr, ptr, ptr, i32, ptr } { ptr @test1, ptr @.str, ptr @.str.1, i32 5, ptr null }, { ptr, ptr, ptr, i32, ptr } { ptr @test2, ptr @.str.2, ptr @.str.1, i32 6, ptr null }], section "llvm.metadata"

define void @test0() {
  ret void
}

define void @test1() {
  ret void
}

define void @test2() {
  ret void
}

define void @test3() {
  ret void
}

; CHECK:      .section        .custom_section.llvm.func_attr.annotate.custom0,"",@
; CHECK-NEXT: .int32  test0@FUNCINDEX
; CHECK-NEXT: .int32  test1@FUNCINDEX
; CHECK:      .section        .custom_section.llvm.func_attr.annotate.custom1,"",@
; CHECK-NEXT: .int32  test2@FUNCINDEX
