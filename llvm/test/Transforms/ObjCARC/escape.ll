; RUN: opt -objc-arc -S < %s | FileCheck %s
; rdar://11229925

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"

%struct.__block_byref_weakLogNTimes = type { i8*, %struct.__block_byref_weakLogNTimes*, i32, i32, i8*, i8*, void (...)* }
%struct.__block_descriptor = type { i64, i64 }

; Don't optimize away the retainBlock, because the object's address "escapes"
; with the objc_storeWeak call.

; CHECK: define void @test0(
; CHECK: %tmp7 = call i8* @objc_retainBlock(i8* %tmp6) nounwind, !clang.arc.copy_on_escape !0
; CHECK: call void @objc_release(i8* %tmp7) nounwind, !clang.imprecise_release !0
; CHECK: }
define void @test0() nounwind {
entry:
  %weakLogNTimes = alloca %struct.__block_byref_weakLogNTimes, align 8
  %block = alloca <{ i8*, i32, i32, i8*, %struct.__block_descriptor*, i8* }>, align 8
  %byref.isa = getelementptr inbounds %struct.__block_byref_weakLogNTimes* %weakLogNTimes, i64 0, i32 0
  store i8* null, i8** %byref.isa, align 8
  %byref.forwarding = getelementptr inbounds %struct.__block_byref_weakLogNTimes* %weakLogNTimes, i64 0, i32 1
  store %struct.__block_byref_weakLogNTimes* %weakLogNTimes, %struct.__block_byref_weakLogNTimes** %byref.forwarding, align 8
  %byref.flags = getelementptr inbounds %struct.__block_byref_weakLogNTimes* %weakLogNTimes, i64 0, i32 2
  store i32 33554432, i32* %byref.flags, align 8
  %byref.size = getelementptr inbounds %struct.__block_byref_weakLogNTimes* %weakLogNTimes, i64 0, i32 3
  store i32 48, i32* %byref.size, align 4
  %tmp1 = getelementptr inbounds %struct.__block_byref_weakLogNTimes* %weakLogNTimes, i64 0, i32 4
  store i8* bitcast (void (i8*, i8*)* @__Block_byref_object_copy_ to i8*), i8** %tmp1, align 8
  %tmp2 = getelementptr inbounds %struct.__block_byref_weakLogNTimes* %weakLogNTimes, i64 0, i32 5
  store i8* bitcast (void (i8*)* @__Block_byref_object_dispose_ to i8*), i8** %tmp2, align 8
  %weakLogNTimes1 = getelementptr inbounds %struct.__block_byref_weakLogNTimes* %weakLogNTimes, i64 0, i32 6
  %tmp3 = bitcast void (...)** %weakLogNTimes1 to i8**
  %tmp4 = call i8* @objc_initWeak(i8** %tmp3, i8* null) nounwind
  %block.isa = getelementptr inbounds <{ i8*, i32, i32, i8*, %struct.__block_descriptor*, i8* }>* %block, i64 0, i32 0
  store i8* null, i8** %block.isa, align 8
  %block.flags = getelementptr inbounds <{ i8*, i32, i32, i8*, %struct.__block_descriptor*, i8* }>* %block, i64 0, i32 1
  store i32 1107296256, i32* %block.flags, align 8
  %block.reserved = getelementptr inbounds <{ i8*, i32, i32, i8*, %struct.__block_descriptor*, i8* }>* %block, i64 0, i32 2
  store i32 0, i32* %block.reserved, align 4
  %block.invoke = getelementptr inbounds <{ i8*, i32, i32, i8*, %struct.__block_descriptor*, i8* }>* %block, i64 0, i32 3
  store i8* bitcast (void (i8*, i32)* @__main_block_invoke_0 to i8*), i8** %block.invoke, align 8
  %block.descriptor = getelementptr inbounds <{ i8*, i32, i32, i8*, %struct.__block_descriptor*, i8* }>* %block, i64 0, i32 4
  store %struct.__block_descriptor* null, %struct.__block_descriptor** %block.descriptor, align 8
  %block.captured = getelementptr inbounds <{ i8*, i32, i32, i8*, %struct.__block_descriptor*, i8* }>* %block, i64 0, i32 5
  %tmp5 = bitcast %struct.__block_byref_weakLogNTimes* %weakLogNTimes to i8*
  store i8* %tmp5, i8** %block.captured, align 8
  %tmp6 = bitcast <{ i8*, i32, i32, i8*, %struct.__block_descriptor*, i8* }>* %block to i8*
  %tmp7 = call i8* @objc_retainBlock(i8* %tmp6) nounwind, !clang.arc.copy_on_escape !0
  %tmp8 = load %struct.__block_byref_weakLogNTimes** %byref.forwarding, align 8
  %weakLogNTimes3 = getelementptr inbounds %struct.__block_byref_weakLogNTimes* %tmp8, i64 0, i32 6
  %tmp9 = bitcast void (...)** %weakLogNTimes3 to i8**
  %tmp10 = call i8* @objc_storeWeak(i8** %tmp9, i8* %tmp7) nounwind
  %tmp11 = getelementptr inbounds i8* %tmp7, i64 16
  %tmp12 = bitcast i8* %tmp11 to i8**
  %tmp13 = load i8** %tmp12, align 8
  %tmp14 = bitcast i8* %tmp13 to void (i8*, i32)*
  call void %tmp14(i8* %tmp7, i32 10) nounwind, !clang.arc.no_objc_arc_exceptions !0
  call void @objc_release(i8* %tmp7) nounwind, !clang.imprecise_release !0
  call void @_Block_object_dispose(i8* %tmp5, i32 8) nounwind
  call void @objc_destroyWeak(i8** %tmp3) nounwind
  ret void
}

; Like test0, but it makes a regular call instead of a storeWeak call,
; so the optimization is valid.

; CHECK: define void @test1(
; CHECK-NOT: @objc_retainBlock
; CHECK: }
define void @test1() nounwind {
entry:
  %weakLogNTimes = alloca %struct.__block_byref_weakLogNTimes, align 8
  %block = alloca <{ i8*, i32, i32, i8*, %struct.__block_descriptor*, i8* }>, align 8
  %byref.isa = getelementptr inbounds %struct.__block_byref_weakLogNTimes* %weakLogNTimes, i64 0, i32 0
  store i8* null, i8** %byref.isa, align 8
  %byref.forwarding = getelementptr inbounds %struct.__block_byref_weakLogNTimes* %weakLogNTimes, i64 0, i32 1
  store %struct.__block_byref_weakLogNTimes* %weakLogNTimes, %struct.__block_byref_weakLogNTimes** %byref.forwarding, align 8
  %byref.flags = getelementptr inbounds %struct.__block_byref_weakLogNTimes* %weakLogNTimes, i64 0, i32 2
  store i32 33554432, i32* %byref.flags, align 8
  %byref.size = getelementptr inbounds %struct.__block_byref_weakLogNTimes* %weakLogNTimes, i64 0, i32 3
  store i32 48, i32* %byref.size, align 4
  %tmp1 = getelementptr inbounds %struct.__block_byref_weakLogNTimes* %weakLogNTimes, i64 0, i32 4
  store i8* bitcast (void (i8*, i8*)* @__Block_byref_object_copy_ to i8*), i8** %tmp1, align 8
  %tmp2 = getelementptr inbounds %struct.__block_byref_weakLogNTimes* %weakLogNTimes, i64 0, i32 5
  store i8* bitcast (void (i8*)* @__Block_byref_object_dispose_ to i8*), i8** %tmp2, align 8
  %weakLogNTimes1 = getelementptr inbounds %struct.__block_byref_weakLogNTimes* %weakLogNTimes, i64 0, i32 6
  %tmp3 = bitcast void (...)** %weakLogNTimes1 to i8**
  %tmp4 = call i8* @objc_initWeak(i8** %tmp3, i8* null) nounwind
  %block.isa = getelementptr inbounds <{ i8*, i32, i32, i8*, %struct.__block_descriptor*, i8* }>* %block, i64 0, i32 0
  store i8* null, i8** %block.isa, align 8
  %block.flags = getelementptr inbounds <{ i8*, i32, i32, i8*, %struct.__block_descriptor*, i8* }>* %block, i64 0, i32 1
  store i32 1107296256, i32* %block.flags, align 8
  %block.reserved = getelementptr inbounds <{ i8*, i32, i32, i8*, %struct.__block_descriptor*, i8* }>* %block, i64 0, i32 2
  store i32 0, i32* %block.reserved, align 4
  %block.invoke = getelementptr inbounds <{ i8*, i32, i32, i8*, %struct.__block_descriptor*, i8* }>* %block, i64 0, i32 3
  store i8* bitcast (void (i8*, i32)* @__main_block_invoke_0 to i8*), i8** %block.invoke, align 8
  %block.descriptor = getelementptr inbounds <{ i8*, i32, i32, i8*, %struct.__block_descriptor*, i8* }>* %block, i64 0, i32 4
  store %struct.__block_descriptor* null, %struct.__block_descriptor** %block.descriptor, align 8
  %block.captured = getelementptr inbounds <{ i8*, i32, i32, i8*, %struct.__block_descriptor*, i8* }>* %block, i64 0, i32 5
  %tmp5 = bitcast %struct.__block_byref_weakLogNTimes* %weakLogNTimes to i8*
  store i8* %tmp5, i8** %block.captured, align 8
  %tmp6 = bitcast <{ i8*, i32, i32, i8*, %struct.__block_descriptor*, i8* }>* %block to i8*
  %tmp7 = call i8* @objc_retainBlock(i8* %tmp6) nounwind, !clang.arc.copy_on_escape !0
  %tmp8 = load %struct.__block_byref_weakLogNTimes** %byref.forwarding, align 8
  %weakLogNTimes3 = getelementptr inbounds %struct.__block_byref_weakLogNTimes* %tmp8, i64 0, i32 6
  %tmp9 = bitcast void (...)** %weakLogNTimes3 to i8**
  %tmp10 = call i8* @not_really_objc_storeWeak(i8** %tmp9, i8* %tmp7) nounwind
  %tmp11 = getelementptr inbounds i8* %tmp7, i64 16
  %tmp12 = bitcast i8* %tmp11 to i8**
  %tmp13 = load i8** %tmp12, align 8
  %tmp14 = bitcast i8* %tmp13 to void (i8*, i32)*
  call void %tmp14(i8* %tmp7, i32 10) nounwind, !clang.arc.no_objc_arc_exceptions !0
  call void @objc_release(i8* %tmp7) nounwind, !clang.imprecise_release !0
  call void @_Block_object_dispose(i8* %tmp5, i32 8) nounwind
  call void @objc_destroyWeak(i8** %tmp3) nounwind
  ret void
}

declare void @__Block_byref_object_copy_(i8*, i8*) nounwind
declare void @__Block_byref_object_dispose_(i8*) nounwind
declare void @objc_destroyWeak(i8**)
declare i8* @objc_initWeak(i8**, i8*)
declare void @__main_block_invoke_0(i8* nocapture, i32) nounwind ssp
declare void @_Block_object_dispose(i8*, i32)
declare i8* @objc_retainBlock(i8*)
declare i8* @objc_storeWeak(i8**, i8*)
declare i8* @not_really_objc_storeWeak(i8**, i8*)
declare void @objc_release(i8*)

!0 = metadata !{}
