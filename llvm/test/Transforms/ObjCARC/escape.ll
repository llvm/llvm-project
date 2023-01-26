; RUN: opt -passes=objc-arc -S < %s | FileCheck %s
; rdar://11229925

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"

%struct.__block_byref_weakLogNTimes = type { ptr, ptr, i32, i32, ptr, ptr, ptr }
%struct.__block_descriptor = type { i64, i64 }

; Don't optimize away the retainBlock, because the object's address "escapes"
; with the objc_storeWeak call.

; CHECK-LABEL: define void @test0(
; CHECK: %tmp7 = call ptr @llvm.objc.retainBlock(ptr %block) [[NUW:#[0-9]+]], !clang.arc.copy_on_escape !0
; CHECK: call void @llvm.objc.release(ptr %tmp7) [[NUW]], !clang.imprecise_release !0
; CHECK: }
define void @test0() nounwind {
entry:
  %weakLogNTimes = alloca %struct.__block_byref_weakLogNTimes, align 8
  %block = alloca <{ ptr, i32, i32, ptr, ptr, ptr }>, align 8
  store ptr null, ptr %weakLogNTimes, align 8
  %byref.forwarding = getelementptr inbounds %struct.__block_byref_weakLogNTimes, ptr %weakLogNTimes, i64 0, i32 1
  store ptr %weakLogNTimes, ptr %byref.forwarding, align 8
  %byref.flags = getelementptr inbounds %struct.__block_byref_weakLogNTimes, ptr %weakLogNTimes, i64 0, i32 2
  store i32 33554432, ptr %byref.flags, align 8
  %byref.size = getelementptr inbounds %struct.__block_byref_weakLogNTimes, ptr %weakLogNTimes, i64 0, i32 3
  store i32 48, ptr %byref.size, align 4
  %tmp1 = getelementptr inbounds %struct.__block_byref_weakLogNTimes, ptr %weakLogNTimes, i64 0, i32 4
  store ptr @__Block_byref_object_copy_, ptr %tmp1, align 8
  %tmp2 = getelementptr inbounds %struct.__block_byref_weakLogNTimes, ptr %weakLogNTimes, i64 0, i32 5
  store ptr @__Block_byref_object_dispose_, ptr %tmp2, align 8
  %weakLogNTimes1 = getelementptr inbounds %struct.__block_byref_weakLogNTimes, ptr %weakLogNTimes, i64 0, i32 6
  %tmp4 = call ptr @llvm.objc.initWeak(ptr %weakLogNTimes1, ptr null) nounwind
  %block.isa = getelementptr inbounds <{ ptr, i32, i32, ptr, ptr, ptr }>, ptr %block, i64 0, i32 0
  store ptr null, ptr %block.isa, align 8
  %block.flags = getelementptr inbounds <{ ptr, i32, i32, ptr, ptr, ptr }>, ptr %block, i64 0, i32 1
  store i32 1107296256, ptr %block.flags, align 8
  %block.reserved = getelementptr inbounds <{ ptr, i32, i32, ptr, ptr, ptr }>, ptr %block, i64 0, i32 2
  store i32 0, ptr %block.reserved, align 4
  %block.invoke = getelementptr inbounds <{ ptr, i32, i32, ptr, ptr, ptr }>, ptr %block, i64 0, i32 3
  store ptr @__main_block_invoke_0, ptr %block.invoke, align 8
  %block.descriptor = getelementptr inbounds <{ ptr, i32, i32, ptr, ptr, ptr }>, ptr %block, i64 0, i32 4
  store ptr null, ptr %block.descriptor, align 8
  %block.captured = getelementptr inbounds <{ ptr, i32, i32, ptr, ptr, ptr }>, ptr %block, i64 0, i32 5
  store ptr %weakLogNTimes, ptr %block.captured, align 8
  %tmp7 = call ptr @llvm.objc.retainBlock(ptr %block) nounwind, !clang.arc.copy_on_escape !0
  %tmp8 = load ptr, ptr %byref.forwarding, align 8
  %weakLogNTimes3 = getelementptr inbounds %struct.__block_byref_weakLogNTimes, ptr %tmp8, i64 0, i32 6
  %tmp10 = call ptr @llvm.objc.storeWeak(ptr %weakLogNTimes3, ptr %tmp7) nounwind
  %tmp11 = getelementptr inbounds i8, ptr %tmp7, i64 16
  %tmp13 = load ptr, ptr %tmp11, align 8
  call void %tmp13(ptr %tmp7, i32 10) nounwind, !clang.arc.no_objc_arc_exceptions !0
  call void @llvm.objc.release(ptr %tmp7) nounwind, !clang.imprecise_release !0
  call void @_Block_object_dispose(ptr %weakLogNTimes, i32 8) nounwind
  call void @llvm.objc.destroyWeak(ptr %weakLogNTimes1) nounwind
  ret void
}

; Like test0, but it makes a regular call instead of a storeWeak call,
; so the optimization is valid.

; CHECK-LABEL: define void @test1(
; CHECK-NOT: @llvm.objc.retainBlock
; CHECK: }
define void @test1() nounwind {
entry:
  %weakLogNTimes = alloca %struct.__block_byref_weakLogNTimes, align 8
  %block = alloca <{ ptr, i32, i32, ptr, ptr, ptr }>, align 8
  store ptr null, ptr %weakLogNTimes, align 8
  %byref.forwarding = getelementptr inbounds %struct.__block_byref_weakLogNTimes, ptr %weakLogNTimes, i64 0, i32 1
  store ptr %weakLogNTimes, ptr %byref.forwarding, align 8
  %byref.flags = getelementptr inbounds %struct.__block_byref_weakLogNTimes, ptr %weakLogNTimes, i64 0, i32 2
  store i32 33554432, ptr %byref.flags, align 8
  %byref.size = getelementptr inbounds %struct.__block_byref_weakLogNTimes, ptr %weakLogNTimes, i64 0, i32 3
  store i32 48, ptr %byref.size, align 4
  %tmp1 = getelementptr inbounds %struct.__block_byref_weakLogNTimes, ptr %weakLogNTimes, i64 0, i32 4
  store ptr @__Block_byref_object_copy_, ptr %tmp1, align 8
  %tmp2 = getelementptr inbounds %struct.__block_byref_weakLogNTimes, ptr %weakLogNTimes, i64 0, i32 5
  store ptr @__Block_byref_object_dispose_, ptr %tmp2, align 8
  %weakLogNTimes1 = getelementptr inbounds %struct.__block_byref_weakLogNTimes, ptr %weakLogNTimes, i64 0, i32 6
  %tmp4 = call ptr @llvm.objc.initWeak(ptr %weakLogNTimes1, ptr null) nounwind
  %block.isa = getelementptr inbounds <{ ptr, i32, i32, ptr, ptr, ptr }>, ptr %block, i64 0, i32 0
  store ptr null, ptr %block.isa, align 8
  %block.flags = getelementptr inbounds <{ ptr, i32, i32, ptr, ptr, ptr }>, ptr %block, i64 0, i32 1
  store i32 1107296256, ptr %block.flags, align 8
  %block.reserved = getelementptr inbounds <{ ptr, i32, i32, ptr, ptr, ptr }>, ptr %block, i64 0, i32 2
  store i32 0, ptr %block.reserved, align 4
  %block.invoke = getelementptr inbounds <{ ptr, i32, i32, ptr, ptr, ptr }>, ptr %block, i64 0, i32 3
  store ptr @__main_block_invoke_0, ptr %block.invoke, align 8
  %block.descriptor = getelementptr inbounds <{ ptr, i32, i32, ptr, ptr, ptr }>, ptr %block, i64 0, i32 4
  store ptr null, ptr %block.descriptor, align 8
  %block.captured = getelementptr inbounds <{ ptr, i32, i32, ptr, ptr, ptr }>, ptr %block, i64 0, i32 5
  store ptr %weakLogNTimes, ptr %block.captured, align 8
  %tmp7 = call ptr @llvm.objc.retainBlock(ptr %block) nounwind, !clang.arc.copy_on_escape !0
  %tmp8 = load ptr, ptr %byref.forwarding, align 8
  %weakLogNTimes3 = getelementptr inbounds %struct.__block_byref_weakLogNTimes, ptr %tmp8, i64 0, i32 6
  %tmp10 = call ptr @not_really_objc_storeWeak(ptr %weakLogNTimes3, ptr %tmp7) nounwind
  %tmp11 = getelementptr inbounds i8, ptr %tmp7, i64 16
  %tmp13 = load ptr, ptr %tmp11, align 8
  call void %tmp13(ptr %tmp7, i32 10) nounwind, !clang.arc.no_objc_arc_exceptions !0
  call void @llvm.objc.release(ptr %tmp7) nounwind, !clang.imprecise_release !0
  call void @_Block_object_dispose(ptr %weakLogNTimes, i32 8) nounwind
  call void @llvm.objc.destroyWeak(ptr %weakLogNTimes1) nounwind
  ret void
}

declare void @__Block_byref_object_copy_(ptr, ptr) nounwind
declare void @__Block_byref_object_dispose_(ptr) nounwind
declare void @llvm.objc.destroyWeak(ptr)
declare ptr @llvm.objc.initWeak(ptr, ptr)
declare void @__main_block_invoke_0(ptr nocapture, i32) nounwind ssp
declare void @_Block_object_dispose(ptr, i32)
declare ptr @llvm.objc.retainBlock(ptr)
declare ptr @llvm.objc.storeWeak(ptr, ptr)
declare ptr @not_really_objc_storeWeak(ptr, ptr)
declare void @llvm.objc.release(ptr)

!0 = !{}

; CHECK: attributes [[NUW]] = { nounwind }
; CHECK: attributes #1 = { nounwind ssp }
