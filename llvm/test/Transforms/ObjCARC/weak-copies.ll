; RUN: opt -S -passes=objc-arc < %s | FileCheck %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-apple-darwin11.0.0"

%0 = type { i64, i64, ptr, ptr, ptr, ptr }
%1 = type <{ ptr, i32, i32, ptr, ptr, ptr }>
%struct.__block_descriptor = type { i64, i64 }

@_NSConcreteStackBlock = external global ptr
@.str = private unnamed_addr constant [6 x i8] c"v8@?0\00"
@"\01L_OBJC_CLASS_NAME_" = internal global [3 x i8] c"\01@\00", section "__TEXT,__objc_classname,cstring_literals", align 1
@__block_descriptor_tmp = internal constant %0 { i64 0, i64 40, ptr @__copy_helper_block_, ptr @__destroy_helper_block_, ptr @.str, ptr @"\01L_OBJC_CLASS_NAME_" }
@"\01L_OBJC_IMAGE_INFO" = internal constant [2 x i32] [i32 0, i32 16], section "__DATA, __objc_imageinfo, regular, no_dead_strip"
@llvm.used = appending global [2 x ptr] [ptr @"\01L_OBJC_CLASS_NAME_", ptr @"\01L_OBJC_IMAGE_INFO"], section "llvm.metadata"

; Eliminate unnecessary weak pointer copies.

; CHECK:      define void @foo() {
; CHECK-NEXT: entry:
; CHECK-NEXT:   %call = call ptr @bar()
; CHECK-NEXT:   call void @use(ptr %call) [[NUW:#[0-9]+]]
; CHECK-NEXT:   ret void
; CHECK-NEXT: }
define void @foo() {
entry:
  %w = alloca ptr, align 8
  %x = alloca ptr, align 8
  %call = call ptr @bar()
  %0 = call ptr @llvm.objc.initWeak(ptr %w, ptr %call) nounwind
  %1 = call ptr @llvm.objc.loadWeak(ptr %w) nounwind
  %2 = call ptr @llvm.objc.initWeak(ptr %x, ptr %1) nounwind
  %3 = call ptr @llvm.objc.loadWeak(ptr %x) nounwind
  call void @use(ptr %3) nounwind
  call void @llvm.objc.destroyWeak(ptr %x) nounwind
  call void @llvm.objc.destroyWeak(ptr %w) nounwind
  ret void
}

; Eliminate unnecessary weak pointer copies in a block initialization.

; CHECK:      define void @qux(ptr %me) #0 {
; CHECK-NEXT: entry:
; CHECK-NEXT:   %block = alloca %1, align 8
; CHECK-NOT:    alloca
; CHECK:      }
define void @qux(ptr %me) nounwind {
entry:
  %w = alloca ptr, align 8
  %block = alloca %1, align 8
  %0 = call ptr @llvm.objc.retain(ptr %me) nounwind
  %1 = call ptr @llvm.objc.initWeak(ptr %w, ptr %0) nounwind
  store ptr @_NSConcreteStackBlock, ptr %block, align 8
  %block.flags = getelementptr inbounds %1, ptr %block, i64 0, i32 1
  store i32 1107296256, ptr %block.flags, align 8
  %block.reserved = getelementptr inbounds %1, ptr %block, i64 0, i32 2
  store i32 0, ptr %block.reserved, align 4
  %block.invoke = getelementptr inbounds %1, ptr %block, i64 0, i32 3
  store ptr @__qux_block_invoke_0, ptr %block.invoke, align 8
  %block.descriptor = getelementptr inbounds %1, ptr %block, i64 0, i32 4
  store ptr @__block_descriptor_tmp, ptr %block.descriptor, align 8
  %block.captured = getelementptr inbounds %1, ptr %block, i64 0, i32 5
  %2 = call ptr @llvm.objc.loadWeak(ptr %w) nounwind
  %3 = call ptr @llvm.objc.initWeak(ptr %block.captured, ptr %2) nounwind
  call void @use_block(ptr %block) nounwind
  call void @llvm.objc.destroyWeak(ptr %block.captured) nounwind
  call void @llvm.objc.destroyWeak(ptr %w) nounwind
  call void @llvm.objc.release(ptr %0) nounwind, !clang.imprecise_release !0
  ret void
}

declare ptr @llvm.objc.retain(ptr)
declare void @use_block(ptr) nounwind
declare void @__qux_block_invoke_0(ptr %.block_descriptor) nounwind
declare void @__copy_helper_block_(ptr, ptr) nounwind
declare void @llvm.objc.copyWeak(ptr, ptr)
declare void @__destroy_helper_block_(ptr) nounwind
declare void @llvm.objc.release(ptr)
declare ptr @bar()
declare ptr @llvm.objc.initWeak(ptr, ptr)
declare ptr @llvm.objc.loadWeak(ptr)
declare void @use(ptr) nounwind
declare void @llvm.objc.destroyWeak(ptr)

; CHECK: attributes [[NUW]] = { nounwind }

!0 = !{}
