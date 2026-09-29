; RUN: llc -mtriple=hexagon -verify-machineinstrs < %s | FileCheck %s
; Check that this testcase compiles successfully.
; CHECK-LABEL: fred:
; CHECK: call foo

target triple = "hexagon"

%struct.0 = type { i32, i16, ptr }

declare void @llvm.lifetime.start.p0(i64, ptr nocapture) #1
declare void @llvm.lifetime.end.p0(i64, ptr nocapture) #1

define i32 @fred(ptr readonly %p0, ptr %p1) local_unnamed_addr #0 {
entry:
  %v0 = alloca i16, align 2
  %v1 = icmp eq ptr %p0, null
  br i1 %v1, label %if.then, label %lor.lhs.false

lor.lhs.false:                                    ; preds = %entry
  %v3 = load ptr, ptr %p0, align 4
  %v4 = icmp eq ptr %v3, null
  br i1 %v4, label %if.then, label %if.else

if.then:                                          ; preds = %lor.lhs.false, %ent
  %v5 = icmp eq ptr %p1, null
  br i1 %v5, label %cleanup, label %if.then3

if.then3:                                         ; preds = %if.then
  store i32 0, ptr %p1, align 4
  br label %cleanup

if.else:                                          ; preds = %lor.lhs.false
  call void @llvm.lifetime.start.p0(i64 2, ptr nonnull %v0) #0
  store i16 0, ptr %v0, align 2
  %v7 = call i32 @foo(ptr nonnull %v3, ptr nonnull %v0) #0
  %v8 = icmp eq ptr %p1, null
  br i1 %v8, label %if.end7, label %if.then6

if.then6:                                         ; preds = %if.else
  %v9 = load i16, ptr %v0, align 2
  %v10 = zext i16 %v9 to i32
  store i32 %v10, ptr %p1, align 4
  br label %if.end7

if.end7:                                          ; preds = %if.else, %if.then6
  call void @llvm.lifetime.end.p0(i64 2, ptr nonnull %v0) #0
  br label %cleanup

cleanup:                                          ; preds = %if.then3, %if.then,
  %v11 = phi i32 [ %v7, %if.end7 ], [ -2147024809, %if.then ], [ -2147024809, %if.then3 ]
  ret i32 %v11
}

declare i32 @foo(ptr, ptr) local_unnamed_addr #0

attributes #0 = { nounwind }
attributes #1 = { argmemonly nounwind }

