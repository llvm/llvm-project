; RUN: opt -S -objc-arc < %s | FileCheck %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
%struct.__block_byref_x = type { i8*, %struct.__block_byref_x*, i32, i32, i32 }
%struct.__block_descriptor = type { i64, i64 }
@_NSConcreteStackBlock = external global i8*
@__block_descriptor_tmp = external hidden constant { i64, i64, i8*, i8*, i8*, i8* }

; The optimizer should make use of the !clang.arc.no_objc_arc_exceptions
; metadata and eliminate the retainBlock+release pair here.
; rdar://10803830.

; CHECK: define void @test0(
; CHECK-NOT: @objc
; CHECK: }
define void @test0() {
entry:
  %x = alloca %struct.__block_byref_x, align 8
  %block = alloca <{ i8*, i32, i32, i8*, %struct.__block_descriptor*, i8* }>, align 8
  %byref.isa = getelementptr inbounds %struct.__block_byref_x* %x, i64 0, i32 0
  store i8* null, i8** %byref.isa, align 8
  %byref.forwarding = getelementptr inbounds %struct.__block_byref_x* %x, i64 0, i32 1
  store %struct.__block_byref_x* %x, %struct.__block_byref_x** %byref.forwarding, align 8
  %byref.flags = getelementptr inbounds %struct.__block_byref_x* %x, i64 0, i32 2
  store i32 0, i32* %byref.flags, align 8
  %byref.size = getelementptr inbounds %struct.__block_byref_x* %x, i64 0, i32 3
  store i32 32, i32* %byref.size, align 4
  %block.isa = getelementptr inbounds <{ i8*, i32, i32, i8*, %struct.__block_descriptor*, i8* }>* %block, i64 0, i32 0
  store i8* bitcast (i8** @_NSConcreteStackBlock to i8*), i8** %block.isa, align 8
  %block.flags = getelementptr inbounds <{ i8*, i32, i32, i8*, %struct.__block_descriptor*, i8* }>* %block, i64 0, i32 1
  store i32 1107296256, i32* %block.flags, align 8
  %block.reserved = getelementptr inbounds <{ i8*, i32, i32, i8*, %struct.__block_descriptor*, i8* }>* %block, i64 0, i32 2
  store i32 0, i32* %block.reserved, align 4
  %block.invoke = getelementptr inbounds <{ i8*, i32, i32, i8*, %struct.__block_descriptor*, i8* }>* %block, i64 0, i32 3
  store i8* bitcast (void (i8*)* @__foo_block_invoke_0 to i8*), i8** %block.invoke, align 8
  %block.descriptor = getelementptr inbounds <{ i8*, i32, i32, i8*, %struct.__block_descriptor*, i8* }>* %block, i64 0, i32 4
  store %struct.__block_descriptor* bitcast ({ i64, i64, i8*, i8*, i8*, i8* }* @__block_descriptor_tmp to %struct.__block_descriptor*), %struct.__block_descriptor** %block.descriptor, align 8
  %block.captured = getelementptr inbounds <{ i8*, i32, i32, i8*, %struct.__block_descriptor*, i8* }>* %block, i64 0, i32 5
  %t1 = bitcast %struct.__block_byref_x* %x to i8*
  store i8* %t1, i8** %block.captured, align 8
  %t2 = bitcast <{ i8*, i32, i32, i8*, %struct.__block_descriptor*, i8* }>* %block to i8*
  %t3 = call i8* @objc_retainBlock(i8* %t2) nounwind, !clang.arc.copy_on_escape !4
  %t4 = getelementptr inbounds i8* %t3, i64 16
  %t5 = bitcast i8* %t4 to i8**
  %t6 = load i8** %t5, align 8
  %t7 = bitcast i8* %t6 to void (i8*)*
  invoke void %t7(i8* %t3)
          to label %invoke.cont unwind label %lpad, !clang.arc.no_objc_arc_exceptions !4

invoke.cont:                                      ; preds = %entry
  call void @objc_release(i8* %t3) nounwind, !clang.imprecise_release !4
  call void @_Block_object_dispose(i8* %t1, i32 8)
  ret void

lpad:                                             ; preds = %entry
  %t8 = landingpad { i8*, i32 } personality i8* bitcast (i32 (...)* @__objc_personality_v0 to i8*)
          cleanup
  call void @_Block_object_dispose(i8* %t1, i32 8)
  resume { i8*, i32 } %t8
}

; There is no !clang.arc.no_objc_arc_exceptions
; metadata here, so the optimizer shouldn't eliminate anything.

; CHECK: define void @test0_no_metadata(
; CHECK: call i8* @objc_retainBlock(
; CHECK: invoke
; CHECK: call void @objc_release(
; CHECK: }
define void @test0_no_metadata() {
entry:
  %x = alloca %struct.__block_byref_x, align 8
  %block = alloca <{ i8*, i32, i32, i8*, %struct.__block_descriptor*, i8* }>, align 8
  %byref.isa = getelementptr inbounds %struct.__block_byref_x* %x, i64 0, i32 0
  store i8* null, i8** %byref.isa, align 8
  %byref.forwarding = getelementptr inbounds %struct.__block_byref_x* %x, i64 0, i32 1
  store %struct.__block_byref_x* %x, %struct.__block_byref_x** %byref.forwarding, align 8
  %byref.flags = getelementptr inbounds %struct.__block_byref_x* %x, i64 0, i32 2
  store i32 0, i32* %byref.flags, align 8
  %byref.size = getelementptr inbounds %struct.__block_byref_x* %x, i64 0, i32 3
  store i32 32, i32* %byref.size, align 4
  %block.isa = getelementptr inbounds <{ i8*, i32, i32, i8*, %struct.__block_descriptor*, i8* }>* %block, i64 0, i32 0
  store i8* bitcast (i8** @_NSConcreteStackBlock to i8*), i8** %block.isa, align 8
  %block.flags = getelementptr inbounds <{ i8*, i32, i32, i8*, %struct.__block_descriptor*, i8* }>* %block, i64 0, i32 1
  store i32 1107296256, i32* %block.flags, align 8
  %block.reserved = getelementptr inbounds <{ i8*, i32, i32, i8*, %struct.__block_descriptor*, i8* }>* %block, i64 0, i32 2
  store i32 0, i32* %block.reserved, align 4
  %block.invoke = getelementptr inbounds <{ i8*, i32, i32, i8*, %struct.__block_descriptor*, i8* }>* %block, i64 0, i32 3
  store i8* bitcast (void (i8*)* @__foo_block_invoke_0 to i8*), i8** %block.invoke, align 8
  %block.descriptor = getelementptr inbounds <{ i8*, i32, i32, i8*, %struct.__block_descriptor*, i8* }>* %block, i64 0, i32 4
  store %struct.__block_descriptor* bitcast ({ i64, i64, i8*, i8*, i8*, i8* }* @__block_descriptor_tmp to %struct.__block_descriptor*), %struct.__block_descriptor** %block.descriptor, align 8
  %block.captured = getelementptr inbounds <{ i8*, i32, i32, i8*, %struct.__block_descriptor*, i8* }>* %block, i64 0, i32 5
  %t1 = bitcast %struct.__block_byref_x* %x to i8*
  store i8* %t1, i8** %block.captured, align 8
  %t2 = bitcast <{ i8*, i32, i32, i8*, %struct.__block_descriptor*, i8* }>* %block to i8*
  %t3 = call i8* @objc_retainBlock(i8* %t2) nounwind, !clang.arc.copy_on_escape !4
  %t4 = getelementptr inbounds i8* %t3, i64 16
  %t5 = bitcast i8* %t4 to i8**
  %t6 = load i8** %t5, align 8
  %t7 = bitcast i8* %t6 to void (i8*)*
  invoke void %t7(i8* %t3)
          to label %invoke.cont unwind label %lpad

invoke.cont:                                      ; preds = %entry
  call void @objc_release(i8* %t3) nounwind, !clang.imprecise_release !4
  call void @_Block_object_dispose(i8* %t1, i32 8)
  ret void

lpad:                                             ; preds = %entry
  %t8 = landingpad { i8*, i32 } personality i8* bitcast (i32 (...)* @__objc_personality_v0 to i8*)
          cleanup
  call void @_Block_object_dispose(i8* %t1, i32 8)
  resume { i8*, i32 } %t8
}

declare i8* @objc_retainBlock(i8*)
declare void @objc_release(i8*)
declare void @_Block_object_dispose(i8*, i32)
declare i32 @__objc_personality_v0(...)
declare void @__foo_block_invoke_0(i8* nocapture) uwtable ssp

!4 = metadata !{}
