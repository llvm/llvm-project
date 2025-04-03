; RUN: llc -mtriple=powerpc64le-unknown-linux-gnu < %s | FileCheck %s

@index = global i32 0, align 4
@.str = private unnamed_addr constant [1 x i8] zeroinitializer, section "llvm.metadata"
@.str.1 = private unnamed_addr constant [7 x i8] c"test.c\00", section "llvm.metadata" 
@llvm.global.annotations = appending global [1 x { ptr, ptr, ptr, i32, ptr }] [{ ptr, ptr, ptr, i32, ptr } { ptr @index, ptr @.str, ptr @.str.1, i32 1, ptr null }], section "llvm.metadata"

; CHECK-NOT: .set
; CHECK-NOT: _MergedGlobals
