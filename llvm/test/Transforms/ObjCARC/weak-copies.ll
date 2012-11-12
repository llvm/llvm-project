; RUN: opt -S -basicaa -objc-arc < %s | FileCheck %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-apple-darwin11.0.0"

%0 = type { i64, i64, i8*, i8*, i8*, i8* }
%1 = type <{ i8*, i32, i32, i8*, %struct.__block_descriptor*, i8* }>
%struct.__block_descriptor = type { i64, i64 }

@_NSConcreteStackBlock = external global i8*
@.str = private unnamed_addr constant [6 x i8] c"v8@?0\00"
@"\01L_OBJC_CLASS_NAME_" = internal global [3 x i8] c"\01@\00", section "__TEXT,__objc_classname,cstring_literals", align 1
@__block_descriptor_tmp = internal constant %0 { i64 0, i64 40, i8* bitcast (void (i8*, i8*)* @__copy_helper_block_ to i8*), i8* bitcast (void (i8*)* @__destroy_helper_block_ to i8*), i8* getelementptr inbounds ([6 x i8]* @.str, i32 0, i32 0), i8* getelementptr inbounds ([3 x i8]* @"\01L_OBJC_CLASS_NAME_", i32 0, i32 0) }
@"\01L_OBJC_IMAGE_INFO" = internal constant [2 x i32] [i32 0, i32 16], section "__DATA, __objc_imageinfo, regular, no_dead_strip"
@llvm.used = appending global [2 x i8*] [i8* getelementptr inbounds ([3 x i8]* @"\01L_OBJC_CLASS_NAME_", i32 0, i32 0), i8* bitcast ([2 x i32]* @"\01L_OBJC_IMAGE_INFO" to i8*)], section "llvm.metadata"

; Eliminate unnecessary weak pointer copies.

; CHECK:      define void @foo() {
; CHECK-NEXT: entry:
; CHECK-NEXT:   %call = call i8* @bar()
; CHECK-NEXT:   call void @use(i8* %call) nounwind
; CHECK-NEXT:   ret void
; CHECK-NEXT: }
define void @foo() {
entry:
  %w = alloca i8*, align 8
  %x = alloca i8*, align 8
  %call = call i8* @bar()
  %0 = call i8* @objc_initWeak(i8** %w, i8* %call) nounwind
  %1 = call i8* @objc_loadWeak(i8** %w) nounwind
  %2 = call i8* @objc_initWeak(i8** %x, i8* %1) nounwind
  %3 = call i8* @objc_loadWeak(i8** %x) nounwind
  call void @use(i8* %3) nounwind
  call void @objc_destroyWeak(i8** %x) nounwind
  call void @objc_destroyWeak(i8** %w) nounwind
  ret void
}

; Eliminate unnecessary weak pointer copies in a block initialization.

; CHECK:      define void @qux(i8* %me) nounwind {
; CHECK-NEXT: entry:
; CHECK-NEXT:   %block = alloca %1, align 8
; CHECK-NOT:    alloca
; CHECK:      }
define void @qux(i8* %me) nounwind {
entry:
  %w = alloca i8*, align 8
  %block = alloca %1, align 8
  %0 = call i8* @objc_retain(i8* %me) nounwind
  %1 = call i8* @objc_initWeak(i8** %w, i8* %0) nounwind
  %block.isa = getelementptr inbounds %1* %block, i64 0, i32 0
  store i8* bitcast (i8** @_NSConcreteStackBlock to i8*), i8** %block.isa, align 8
  %block.flags = getelementptr inbounds %1* %block, i64 0, i32 1
  store i32 1107296256, i32* %block.flags, align 8
  %block.reserved = getelementptr inbounds %1* %block, i64 0, i32 2
  store i32 0, i32* %block.reserved, align 4
  %block.invoke = getelementptr inbounds %1* %block, i64 0, i32 3
  store i8* bitcast (void (i8*)* @__qux_block_invoke_0 to i8*), i8** %block.invoke, align 8
  %block.descriptor = getelementptr inbounds %1* %block, i64 0, i32 4
  store %struct.__block_descriptor* bitcast (%0* @__block_descriptor_tmp to %struct.__block_descriptor*), %struct.__block_descriptor** %block.descriptor, align 8
  %block.captured = getelementptr inbounds %1* %block, i64 0, i32 5
  %2 = call i8* @objc_loadWeak(i8** %w) nounwind
  %3 = call i8* @objc_initWeak(i8** %block.captured, i8* %2) nounwind
  %4 = bitcast %1* %block to void ()*
  call void @use_block(void ()* %4) nounwind
  call void @objc_destroyWeak(i8** %block.captured) nounwind
  call void @objc_destroyWeak(i8** %w) nounwind
  call void @objc_release(i8* %0) nounwind, !clang.imprecise_release !0
  ret void
}

declare i8* @objc_retain(i8*)
declare void @use_block(void ()*) nounwind
declare void @__qux_block_invoke_0(i8* %.block_descriptor) nounwind
declare void @__copy_helper_block_(i8*, i8*) nounwind
declare void @objc_copyWeak(i8**, i8**)
declare void @__destroy_helper_block_(i8*) nounwind
declare void @objc_release(i8*)
declare i8* @bar()
declare i8* @objc_initWeak(i8**, i8*)
declare i8* @objc_loadWeak(i8**)
declare void @use(i8*) nounwind
declare void @objc_destroyWeak(i8**)

!0 = metadata !{}
