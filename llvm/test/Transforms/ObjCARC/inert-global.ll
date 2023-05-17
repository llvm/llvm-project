; RUN: opt -passes=objc-arc -S < %s | FileCheck %s

%0 = type opaque
%struct.__NSConstantString_tag = type { ptr, i32, ptr, i64 }
%struct.__block_descriptor = type { i64, i64 }

@__CFConstantStringClassReference = external global [0 x i32]
@.str = private unnamed_addr constant [4 x i8] c"abc\00", section "__TEXT,__cstring,cstring_literals", align 1
@.str1 = private unnamed_addr constant [4 x i8] c"def\00", section "__TEXT,__cstring,cstring_literals", align 1
@_unnamed_cfstring_ = private global %struct.__NSConstantString_tag { ptr @__CFConstantStringClassReference, i32 1992, ptr @.str, i64 3 }, section "__DATA,__cfstring", align 8 #0
@_unnamed_cfstring_.1 = private global %struct.__NSConstantString_tag { ptr @__CFConstantStringClassReference, i32 1992, ptr @.str1, i64 3 }, section "__DATA,__cfstring", align 8 #0
@_unnamed_cfstring_wo_attr = private global %struct.__NSConstantString_tag { ptr @__CFConstantStringClassReference, i32 1992, ptr @.str1, i64 3 }, section "__DATA,__cfstring", align 8
@_NSConcreteGlobalBlock = external global ptr
@.str.1 = private unnamed_addr constant [6 x i8] c"v8@?0\00", align 1
@"__block_descriptor_32_e5_v8@?0l" = linkonce_odr hidden unnamed_addr constant { i64, i64, ptr, ptr } { i64 0, i64 32, ptr @.str.1, ptr null }, align 8
@__block_literal_global = internal constant { ptr, i32, i32, ptr, ptr } { ptr @_NSConcreteGlobalBlock, i32 1342177280, i32 0, ptr @__globalBlock_block_invoke, ptr @"__block_descriptor_32_e5_v8@?0l" }, align 8 #0

; CHECK-LABEL: define ptr @stringLiteral()
; CHECK-NOT: call
; CHECK: ret ptr @_unnamed_cfstring_

define ptr @stringLiteral() {
  %1 = tail call ptr @llvm.objc.retain(ptr @_unnamed_cfstring_)
  %2 = call ptr @llvm.objc.autorelease(ptr @_unnamed_cfstring_)
  ret ptr @_unnamed_cfstring_
}

; CHECK-LABEL: define ptr @stringLiteral1()
; CHECK-NEXT: call ptr @llvm.objc.retain(
; CHECK-NEXT: call ptr @llvm.objc.autorelease(
; CHECK-NEXT: ret ptr

define ptr @stringLiteral1() {
  %1 = tail call ptr @llvm.objc.retain(ptr @_unnamed_cfstring_wo_attr)
  %2 = call ptr @llvm.objc.autorelease(ptr @_unnamed_cfstring_wo_attr)
  ret ptr @_unnamed_cfstring_wo_attr
}

; CHECK-LABEL: define ptr @globalBlock()
; CHECK-NOT: call
; CHECK-NEXT: ret ptr @__block_literal_global

define ptr @globalBlock() {
  %1 = tail call ptr @llvm.objc.retainBlock(ptr @__block_literal_global)
  %2 = tail call ptr @llvm.objc.retainBlock(ptr %1)
  tail call void @llvm.objc.release(ptr %1)
  %3 = tail call ptr @llvm.objc.autoreleaseReturnValue(ptr %2)
  ret ptr %2
}

define internal void @__globalBlock_block_invoke(ptr nocapture readnone) {
  tail call void @foo()
  ret void
}

; CHECK: define ptr @test_conditional0(
; CHECK: %[[PHI0:.*]] = phi ptr [ @_unnamed_cfstring_, %{{.*}} ], [ null, %{{.*}} ]

; CHECK: %[[PHI1:.*]] = phi ptr [ @_unnamed_cfstring_, %{{.*}} ], [ %[[PHI0]], %{{.*}} ]
; CHECK-NEXT: %[[PHI2:.*]] = phi ptr [ @_unnamed_cfstring_, %{{.*}} ], [ %{{.*}}, %{{.*}} ]
; CHECK-NEXT: %[[V5:.*]] = tail call ptr @llvm.objc.autoreleaseReturnValue(ptr %[[PHI2]])
; CHECK-NEXT: ret ptr %[[PHI2]]

define ptr @test_conditional0(i32 %i, ptr %b) {
entry:
  %v0 = icmp eq i32 %i, 1
  br i1 %v0, label %bb2, label %bb1

bb1:
  %v1 = icmp eq i32 %i, 2
  br i1 %v1, label %bb2, label %return

bb2:
  %phi0 = phi ptr [ @_unnamed_cfstring_, %entry ], [ null, %bb1 ]
  br label %return

return:
  %phi1 = phi ptr [ @_unnamed_cfstring_, %bb1 ], [ %phi0, %bb2 ]
  %phi2 = phi ptr [ @_unnamed_cfstring_, %bb1 ], [ %b, %bb2 ]
  %v3 = tail call ptr @llvm.objc.autoreleaseReturnValue(ptr %phi1)
  %v5 = tail call ptr @llvm.objc.autoreleaseReturnValue(ptr %phi2)
  ret ptr %phi2
}

; CHECK-LABEL: define void @test_conditional1(
; CHECK-NOT: @llvm.objc
; CHECK: ret void

define void @test_conditional1(i32 %i) {
entry:
  %v0 = add nsw i32 %i, -1
  %c1 = icmp eq i32 %v0, 0
  br i1 %c1, label %while.end, label %while.body

while.body:
  %v1 = phi i32 [ %v5, %if.end ], [ %v0, %entry ]
  %v2 = phi ptr [ %v4, %if.end ], [ @_unnamed_cfstring_.1, %entry ]
  %v3 = tail call ptr @llvm.objc.retain(ptr %v2)
  %cmp = icmp eq i32 %v1, 2
  br i1 %cmp, label %if.then, label %if.end

if.then:
  call void @llvm.objc.release(ptr %v2)
  br label %if.end

if.end:
  %v4 = phi ptr [ @_unnamed_cfstring_, %if.then ], [ %v2, %while.body ]
  call void @llvm.objc.release(ptr %v2)
  %v5 = add nsw i32 %v1, -1
  %tobool = icmp eq i32 %v5, 0
  br i1 %tobool, label %while.end, label %while.body

while.end:
  %v6 = phi ptr [ null, %entry ], [ %v4, %if.end ]
  call void @llvm.objc.release(ptr %v6)
  ret void
}

declare void @foo()

declare ptr @llvm.objc.retain(ptr) local_unnamed_addr
declare ptr @llvm.objc.autoreleaseReturnValue(ptr) local_unnamed_addr
declare ptr @llvm.objc.retainBlock(ptr) local_unnamed_addr
declare void @llvm.objc.release(ptr) local_unnamed_addr
declare ptr @llvm.objc.autorelease(ptr) local_unnamed_addr

attributes #0 = { "objc_arc_inert" }
