; RUN: opt -passes=attributor --attributor-disable=false -attributor-annotate-decl-cs -S < %s | FileCheck %s

declare noalias i8* @malloc(i64)

declare void @nocapture_func_frees_pointer(i8* nocapture)

declare void @func_throws(...)

declare void @sync_func(i8* %p)

declare void @sync_will_return(i8* %p) willreturn nounwind

declare void @no_sync_func(i8* nocapture %p) nofree nosync willreturn

declare void @nofree_func(i8* nocapture %p) nofree  nosync willreturn

declare void @foo(i32* %p)

declare void @foo_nounw(i32* %p) nounwind nofree

declare i32 @no_return_call() noreturn

declare void @free(i8* nocapture)

declare void @llvm.lifetime.start.p0i8(i64, i8* nocapture) nounwind

; CHECK: @nofree_arg_only(i8* nocapture nofree %p1, i8* nocapture %p2)
define void @nofree_arg_only(i8* %p1, i8* %p2) {
  tail call void @free(i8* %p2)
  tail call void @nofree_func(i8* %p1)
  ret void
}

; TEST 1 - negative, pointer freed in another function.

define void @test1() {
  %1 = tail call noalias i8* @malloc(i64 4)
  ; CHECK: @malloc(i64 4)
  ; CHECK-NEXT: @nocapture_func_frees_pointer(i8* noalias nocapture %1)
  tail call void @nocapture_func_frees_pointer(i8* %1)
  tail call void (...) @func_throws()
  tail call void @free(i8* %1)
  ret void
}

; TEST 2 - negative, call to a sync function.

define void @test2() {
  %1 = tail call noalias i8* @malloc(i64 4)
  ; CHECK: @malloc(i64 4)
  ; CHECK-NEXT: @sync_func(i8* %1)
  tail call void @sync_func(i8* %1)
  tail call void @free(i8* %1)
  ret void
}

; TEST 3 - 1 malloc, 1 free

define void @test3() {
  %1 = tail call noalias i8* @malloc(i64 4)
  ; CHECK: %1 = alloca i8, i64 4
  ; CHECK-NEXT: @no_sync_func(i8* noalias nocapture nofree %1)
  tail call void @no_sync_func(i8* %1)
  ; CHECK-NOT: @free(i8* %1)
  tail call void @free(i8* %1)
  ret void
}

define void @test3a(i8* %p) {
  %1 = tail call noalias i8* @malloc(i64 4)
  ; CHECK: %1 = alloca i8, i64 4
  ; CHECK-NEXT: tail call void @nofree_arg_only
  tail call void @nofree_arg_only(i8* %1, i8* %p)
  ; CHECK-NOT: @free(i8* %1)
  tail call void @free(i8* %1)
  ret void
}

declare noalias i8* @calloc(i64, i64)

define void @test0() {
  %1 = tail call noalias i8* @calloc(i64 2, i64 4)
  ; CHECK: %1 = alloca i8, i64 8
  ; CHECK-NEXT: %calloc_bc = bitcast i8* %1 to i8*
  ; CHECK-NEXT: call void @llvm.memset.p0i8.i64(i8* %calloc_bc, i8 0, i64 8, i1 false)
  ; CHECK-NEXT: @no_sync_func(i8* noalias nocapture nofree %1)
  tail call void @no_sync_func(i8* %1)
  ; CHECK-NOT: @free(i8* %1)
  tail call void @free(i8* %1)
  ret void
}

; TEST 4 
define void @test4() {
  %1 = tail call noalias i8* @malloc(i64 4)
  ; CHECK: %1 = alloca i8, i64 4
  ; CHECK-NEXT: @nofree_func(i8* noalias nocapture nofree %1)
  tail call void @nofree_func(i8* %1)
  ret void
}

; TEST 5 - not all exit paths have a call to free, but all uses of malloc
; are in nofree functions and are not captured

define void @test5(i32, i8* %p) {
  %2 = tail call noalias i8* @malloc(i64 4)
  ; CHECK: %2 = alloca i8, i64 4
  ; CHECK-NEXT: icmp eq i32 %0, 0
  %3 = icmp eq i32 %0, 0
  br i1 %3, label %5, label %4

4:                                                ; preds = %1
  tail call void @nofree_func(i8* %2)
  br label %6

5:                                                ; preds = %1
  tail call void @nofree_arg_only(i8* %2, i8* %p)
  tail call void @free(i8* %2)
  ; CHECK-NOT: @free(i8* %2)
  br label %6

6:                                                ; preds = %5, %4
  ret void
}

; TEST 6 - all exit paths have a call to free

define void @test6(i32) {
  %2 = tail call noalias i8* @malloc(i64 4)
  ; CHECK: %2 = alloca i8, i64 4
  ; CHECK-NEXT: icmp eq i32 %0, 0
  %3 = icmp eq i32 %0, 0
  br i1 %3, label %5, label %4

4:                                                ; preds = %1
  tail call void @nofree_func(i8* %2)
  tail call void @free(i8* %2)
  ; CHECK-NOT: @free(i8* %2)
  br label %6

5:                                                ; preds = %1
  tail call void @free(i8* %2)
  ; CHECK-NOT: @free(i8* %2)
  br label %6

6:                                                ; preds = %5, %4
  ret void
}

; TEST 7 - free is dead.

define void @test7() {
  %1 = tail call noalias i8* @malloc(i64 4)
  ; CHECK: alloca i8, i64 4
  ; CHECK-NEXT: tail call i32 @no_return_call()
  tail call i32 @no_return_call()
  ; CHECK-NOT: @free(i8* %1)
  tail call void @free(i8* %1)
  ret void
}

; TEST 8 - Negative: bitcast pointer used in capture function

define void @test8() {
  %1 = tail call noalias i8* @malloc(i64 4)
  ; CHECK: %1 = tail call noalias i8* @malloc(i64 4)
  ; CHECK-NEXT: @no_sync_func(i8* nocapture nofree %1)
  tail call void @no_sync_func(i8* %1)
  %2 = bitcast i8* %1 to i32*
  store i32 10, i32* %2
  %3 = load i32, i32* %2
  tail call void @foo(i32* %2)
  ; CHECK: @free(i8* %1)
  tail call void @free(i8* %1)
  ret void
}

; TEST 9 - FIXME: malloc should be converted.
define void @test9() {
  %1 = tail call noalias i8* @malloc(i64 4)
  ; CHECK: %1 = tail call noalias i8* @malloc(i64 4)
  ; CHECK-NEXT: @no_sync_func(i8* nocapture nofree %1)
  tail call void @no_sync_func(i8* %1)
  %2 = bitcast i8* %1 to i32*
  store i32 10, i32* %2
  %3 = load i32, i32* %2
  tail call void @foo_nounw(i32* %2)
  ; CHECK: @free(i8* %1)
  tail call void @free(i8* %1)
  ret void
}

; TEST 10 - 1 malloc, 1 free

define i32 @test10() {
  %1 = tail call noalias i8* @malloc(i64 4)
  ; CHECK: %1 = alloca i8, i64 4
  ; CHECK-NEXT: @no_sync_func(i8* noalias nocapture nofree %1)
  tail call void @no_sync_func(i8* %1)
  %2 = bitcast i8* %1 to i32*
  store i32 10, i32* %2
  %3 = load i32, i32* %2
  ; CHECK-NOT: @free(i8* %1)
  tail call void @free(i8* %1)
  ret i32 %3
}

define i32 @test_lifetime() {
  %1 = tail call noalias i8* @malloc(i64 4)
  ; CHECK: %1 = alloca i8, i64 4
  ; CHECK-NEXT: @no_sync_func(i8* noalias nocapture nofree %1)
  tail call void @no_sync_func(i8* %1)
  call void @llvm.lifetime.start.p0i8(i64 4, i8* %1)
  %2 = bitcast i8* %1 to i32*
  store i32 10, i32* %2
  %3 = load i32, i32* %2
  ; CHECK-NOT: @free(i8* %1)
  tail call void @free(i8* %1)
  ret i32 %3
}

; TEST 11 

define void @test11() {
  %1 = tail call noalias i8* @malloc(i64 4)
  ; CHECK: test11
  ; CHECK-NEXT: alloc
  ; CHECK-NEXT: @sync_will_return(i8* %1)
  tail call void @sync_will_return(i8* %1)
  tail call void @free(i8* %1)
  ret void
}

; TEST 12
define i32 @irreducible_cfg(i32 %0) {
  ; CHECK: alloca i8, i64 4
  ; CHECK-NEXT: %3 = bitcast
  %2 = call noalias i8* @malloc(i64 4)
  %3 = bitcast i8* %2 to i32*
  store i32 10, i32* %3, align 4
  %4 = icmp eq i32 %0, 1
  br i1 %4, label %5, label %7

5:                                                ; preds = %1
  %6 = add nsw i32 %0, 5
  br label %13

7:                                                ; preds = %1
  br label %8

8:                                                ; preds = %13, %7
  %.0 = phi i32 [ %14, %13 ], [ 1, %7 ]
  %9 = load i32, i32* %3, align 4
  %10 = add nsw i32 %9, -1
  store i32 %10, i32* %3, align 4
  %11 = icmp ne i32 %9, 0
  br i1 %11, label %12, label %15

12:                                               ; preds = %8
  br label %13

13:                                               ; preds = %12, %5
  %.1 = phi i32 [ %6, %5 ], [ %.0, %12 ]
  %14 = add nsw i32 %.1, 1
  br label %8

15:                                               ; preds = %8
  %16 = load i32, i32* %3, align 4
  %17 = bitcast i32* %3 to i8*
  call void @free(i8* %17)
  %18 = load i32, i32* %3, align 4
  ret i32 %18
}


define i32 @malloc_in_loop(i32 %0) {
  %2 = alloca i32, align 4
  %3 = alloca i32*, align 8
  store i32 %0, i32* %2, align 4
  br label %4

4:                                                ; preds = %8, %1
  %5 = load i32, i32* %2, align 4
  %6 = add nsw i32 %5, -1
  store i32 %6, i32* %2, align 4
  %7 = icmp sgt i32 %6, 0
  br i1 %7, label %8, label %11

8:                                                ; preds = %4
  %9 = call noalias i8* @malloc(i64 4)
  ; CHECK: alloca i8, i64 4
  %10 = bitcast i8* %9 to i32*
  store i32 1, i32* %10, align 8
  br label %4

11:                                               ; preds = %4
  ret i32 5
}

; Malloc/Calloc too large
define i32 @test13() {
  %1 = tail call noalias i8* @malloc(i64 256)
  ; CHECK: %1 = tail call noalias i8* @malloc(i64 256)
  ; CHECK-NEXT: @no_sync_func(i8* noalias nofree %1)
  tail call void @no_sync_func(i8* %1)
  %2 = bitcast i8* %1 to i32*
  store i32 10, i32* %2
  %3 = load i32, i32* %2
  tail call void @free(i8* %1)
  ; CHECK: tail call void @free(i8* noalias %1)
  ret i32 %3
}

define i32 @test_sle() {
  %1 = tail call noalias i8* @malloc(i64 -1)
  ; CHECK: %1 = tail call noalias i8* @malloc(i64 -1)
  ; CHECK-NEXT: @no_sync_func(i8* noalias nofree %1)
  tail call void @no_sync_func(i8* %1)
  %2 = bitcast i8* %1 to i32*
  store i32 10, i32* %2
  %3 = load i32, i32* %2
  tail call void @free(i8* %1)
  ; CHECK: tail call void @free(i8* noalias %1)
  ret i32 %3
}

define i32 @test_overflow() {
  %1 = tail call noalias i8* @calloc(i64 65537, i64 65537)
  ; CHECK: %1 = tail call noalias i8* @calloc(i64 65537, i64 65537)
  ; CHECK-NEXT: @no_sync_func(i8* noalias nofree %1)
  tail call void @no_sync_func(i8* %1)
  %2 = bitcast i8* %1 to i32*
  store i32 10, i32* %2
  %3 = load i32, i32* %2
  tail call void @free(i8* %1)
  ; CHECK: tail call void @free(i8* noalias %1)
  ret i32 %3
}

define void @test14() {
  %1 = tail call noalias i8* @calloc(i64 64, i64 4)
  ; CHECK: %1 = tail call noalias i8* @calloc(i64 64, i64 4)
  ; CHECK-NEXT: @no_sync_func(i8* noalias nofree %1)
  tail call void @no_sync_func(i8* %1)
  tail call void @free(i8* %1)
  ; CHECK: tail call void @free(i8* noalias %1)
  ret void
}

define void @test15(i64 %S) {
  ; CHECK: %1 = tail call noalias i8* @malloc(i64 %S)
  %1 = tail call noalias i8* @malloc(i64 %S)
  ; CHECK-NEXT: @no_sync_func(i8* noalias nofree %1)
  tail call void @no_sync_func(i8* %1)
  ; CHECK-NEXT: @free(i8* noalias %1)
  tail call void @free(i8* %1)
  ret void
}

define void @test16a(i8 %v, i8** %P) {
  ; CHECK: %1 = alloca
  %1 = tail call noalias i8* @malloc(i64 4)
  ; CHECK-NEXT: store i8 %v, i8* %1
  store i8 %v, i8* %1
  ; CHECK-NEXT: @no_sync_func(i8* noalias nocapture nofree %1)
  tail call void @no_sync_func(i8* %1)
  ; CHECK-NOT: @free(i8* %1)
  tail call void @free(i8* %1)
  ret void
}

define void @test16b(i8 %v, i8** %P) {
  ; CHECK: %1 = tail call noalias i8* @malloc(i64 4)
  %1 = tail call noalias i8* @malloc(i64 4)
  ; CHECK-NEXT: store i8* %1, i8** %P
  store i8* %1, i8** %P
  ; CHECK-NEXT: @no_sync_func(i8* nocapture nofree %1)
  tail call void @no_sync_func(i8* %1)
  ; CHECK-NEXT: @free(i8* %1)
  tail call void @free(i8* %1)
  ret void
}

define void @test16c(i8 %v, i8** %P) {
  ; CHECK: %1 = alloca
  %1 = tail call noalias i8* @malloc(i64 4)
  ; CHECK-NEXT: store i8* %1, i8** %P
  store i8* %1, i8** %P
  ; CHECK-NEXT: @no_sync_func(i8* nocapture nofree %1)
  tail call void @no_sync_func(i8* %1) nounwind
  ; CHECK-NOT: @free
  tail call void @free(i8* %1)
  ret void
}

define void @test16d(i8 %v, i8** %P) {
  ; CHECK: %1 = tail call noalias i8* @malloc(i64 4)
  %1 = tail call noalias i8* @malloc(i64 4)
  ; CHECK-NEXT: store i8* %1, i8** %P
  store i8* %1, i8** %P
  ret void
}
