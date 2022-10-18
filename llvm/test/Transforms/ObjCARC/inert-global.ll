; RUN: opt -passes=objc-arc -S < %s | FileCheck %s

%0 = type opaque
%struct.__NSConstantString_tag = type { i32*, i32, i8*, i64 }
%struct.__block_descriptor = type { i64, i64 }

@__CFConstantStringClassReference = external global [0 x i32]
@.str = private unnamed_addr constant [4 x i8] c"abc\00", section "__TEXT,__cstring,cstring_literals", align 1
@.str1 = private unnamed_addr constant [4 x i8] c"def\00", section "__TEXT,__cstring,cstring_literals", align 1
@_unnamed_cfstring_ = private global %struct.__NSConstantString_tag { i32* getelementptr inbounds ([0 x i32], [0 x i32]* @__CFConstantStringClassReference, i32 0, i32 0), i32 1992, i8* getelementptr inbounds ([4 x i8], [4 x i8]* @.str, i32 0, i32 0), i64 3 }, section "__DATA,__cfstring", align 8 #0
@_unnamed_cfstring_.1 = private global %struct.__NSConstantString_tag { i32* getelementptr inbounds ([0 x i32], [0 x i32]* @__CFConstantStringClassReference, i32 0, i32 0), i32 1992, i8* getelementptr inbounds ([4 x i8], [4 x i8]* @.str1, i32 0, i32 0), i64 3 }, section "__DATA,__cfstring", align 8 #0
@_unnamed_cfstring_wo_attr = private global %struct.__NSConstantString_tag { i32* getelementptr inbounds ([0 x i32], [0 x i32]* @__CFConstantStringClassReference, i32 0, i32 0), i32 1992, i8* getelementptr inbounds ([4 x i8], [4 x i8]* @.str1, i32 0, i32 0), i64 3 }, section "__DATA,__cfstring", align 8
@_NSConcreteGlobalBlock = external global i8*
@.str.1 = private unnamed_addr constant [6 x i8] c"v8@?0\00", align 1
@"__block_descriptor_32_e5_v8@?0l" = linkonce_odr hidden unnamed_addr constant { i64, i64, i8*, i8* } { i64 0, i64 32, i8* getelementptr inbounds ([6 x i8], [6 x i8]* @.str.1, i32 0, i32 0), i8* null }, align 8
@__block_literal_global = internal constant { i8**, i32, i32, i8*, %struct.__block_descriptor* } { i8** @_NSConcreteGlobalBlock, i32 1342177280, i32 0, i8* bitcast (void (i8*)* @__globalBlock_block_invoke to i8*), %struct.__block_descriptor* bitcast ({ i64, i64, i8*, i8* }* @"__block_descriptor_32_e5_v8@?0l" to %struct.__block_descriptor*) }, align 8 #0

; CHECK-LABEL: define %0* @stringLiteral()
; CHECK-NOT: call
; CHECK: ret %0* bitcast (%struct.__NSConstantString_tag* @_unnamed_cfstring_ to %0*)

define %0* @stringLiteral() {
  %1 = tail call i8* @llvm.objc.retain(i8* bitcast (%struct.__NSConstantString_tag* @_unnamed_cfstring_ to i8*))
  %2 = call i8* @llvm.objc.autorelease(i8* bitcast (%struct.__NSConstantString_tag* @_unnamed_cfstring_ to i8*))
  ret %0* bitcast (%struct.__NSConstantString_tag* @_unnamed_cfstring_ to %0*)
}

; CHECK-LABEL: define %0* @stringLiteral1()
; CHECK-NEXT: call i8* @llvm.objc.retain(
; CHECK-NEXT: call i8* @llvm.objc.autorelease(
; CHECK-NEXT: ret %0*

define %0* @stringLiteral1() {
  %1 = tail call i8* @llvm.objc.retain(i8* bitcast (%struct.__NSConstantString_tag* @_unnamed_cfstring_wo_attr to i8*))
  %2 = call i8* @llvm.objc.autorelease(i8* bitcast (%struct.__NSConstantString_tag* @_unnamed_cfstring_wo_attr to i8*))
  ret %0* bitcast (%struct.__NSConstantString_tag* @_unnamed_cfstring_wo_attr to %0*)
}

; CHECK-LABEL: define void (...)* @globalBlock()
; CHECK-NOT: call
; CHECK: %[[V1:.*]] = bitcast i8* bitcast ({ i8**, i32, i32, i8*, %struct.__block_descriptor* }* @__block_literal_global to i8*) to void (...)*
; CHECK-NEXT: ret void (...)* %[[V1]]

define void (...)* @globalBlock() {
  %1 = tail call i8* @llvm.objc.retainBlock(i8* bitcast ({ i8**, i32, i32, i8*, %struct.__block_descriptor* }* @__block_literal_global to i8*))
  %2 = tail call i8* @llvm.objc.retainBlock(i8* %1)
  %3 = bitcast i8* %2 to void (...)*
  tail call void @llvm.objc.release(i8* %1)
  %4 = tail call i8* @llvm.objc.autoreleaseReturnValue(i8* %2)
  ret void (...)* %3
}

define internal void @__globalBlock_block_invoke(i8* nocapture readnone) {
  tail call void @foo()
  ret void
}

; CHECK: define %[[V0:.*]]* @test_conditional0(
; CHECK: %[[PHI0:.*]] = phi %[[V0]]* [ bitcast (%struct.__NSConstantString_tag* @_unnamed_cfstring_ to %[[V0]]*), %{{.*}} ], [ null, %{{.*}} ]

; CHECK: %[[PHI1:.*]] = phi %[[V0]]* [ bitcast (%struct.__NSConstantString_tag* @_unnamed_cfstring_ to %[[V0]]*), %{{.*}} ], [ %[[PHI0]], %{{.*}} ]
; CHECK-NEXT: %[[PHI2:.*]] = phi %[[V0]]* [ bitcast (%struct.__NSConstantString_tag* @_unnamed_cfstring_ to %[[V0]]*), %{{.*}} ], [ %{{.*}}, %{{.*}} ]
; CHECK-NEXT: %[[V2:.*]] = bitcast %[[V0]]* %[[PHI1]] to i8*
; CHECK-NEXT: %[[V4:.*]] = bitcast %[[V0]]* %[[PHI2]] to i8*
; CHECK-NEXT: %[[V5:.*]] = tail call i8* @llvm.objc.autoreleaseReturnValue(i8* %[[V4]])
; CHECK-NEXT: ret %[[V0]]* %[[PHI2]]

define %0* @test_conditional0(i32 %i, %0* %b) {
entry:
  %v0 = icmp eq i32 %i, 1
  br i1 %v0, label %bb2, label %bb1

bb1:
  %v1 = icmp eq i32 %i, 2
  br i1 %v1, label %bb2, label %return

bb2:
  %phi0 = phi %0* [ bitcast (%struct.__NSConstantString_tag* @_unnamed_cfstring_ to %0*), %entry ], [ null, %bb1 ]
  br label %return

return:
  %phi1 = phi %0* [ bitcast (%struct.__NSConstantString_tag* @_unnamed_cfstring_ to %0*), %bb1 ], [ %phi0, %bb2 ]
  %phi2 = phi %0* [ bitcast (%struct.__NSConstantString_tag* @_unnamed_cfstring_ to %0*), %bb1 ], [ %b, %bb2 ]
  %v2 = bitcast %0* %phi1 to i8*
  %v3 = tail call i8* @llvm.objc.autoreleaseReturnValue(i8* %v2)
  %v4 = bitcast %0* %phi2 to i8*
  %v5 = tail call i8* @llvm.objc.autoreleaseReturnValue(i8* %v4)
  ret %0* %phi2
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
  %v2 = phi i8* [ %v4, %if.end ], [ bitcast (%struct.__NSConstantString_tag* @_unnamed_cfstring_.1 to i8*), %entry ]
  %v3 = tail call i8* @llvm.objc.retain(i8* %v2)
  %cmp = icmp eq i32 %v1, 2
  br i1 %cmp, label %if.then, label %if.end

if.then:
  call void @llvm.objc.release(i8* %v2)
  br label %if.end

if.end:
  %v4 = phi i8* [ bitcast (%struct.__NSConstantString_tag* @_unnamed_cfstring_ to i8*), %if.then ], [ %v2, %while.body ]
  call void @llvm.objc.release(i8* %v2)
  %v5 = add nsw i32 %v1, -1
  %tobool = icmp eq i32 %v5, 0
  br i1 %tobool, label %while.end, label %while.body

while.end:
  %v6 = phi i8* [ null, %entry ], [ %v4, %if.end ]
  call void @llvm.objc.release(i8* %v6)
  ret void
}

declare void @foo()

declare i8* @llvm.objc.retain(i8*) local_unnamed_addr
declare i8* @llvm.objc.autoreleaseReturnValue(i8*) local_unnamed_addr
declare i8* @llvm.objc.retainBlock(i8*) local_unnamed_addr
declare void @llvm.objc.release(i8*) local_unnamed_addr
declare i8* @llvm.objc.autorelease(i8*) local_unnamed_addr

attributes #0 = { "objc_arc_inert" }
