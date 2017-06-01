; RUN: llc -march=hexagon -verify-machineinstrs < %s | FileCheck %s
; Check that this testcase compiles successfully.
; CHECK-LABEL: fred:
; CHECK: call foo

target triple = "hexagon"

%struct.0 = type { i32, i16, i8* }

declare void @llvm.lifetime.start(i64, i8* nocapture) #1
declare void @llvm.lifetime.end(i64, i8* nocapture) #1

define i32 @fred(i8* readonly %p0, i32* %p1) local_unnamed_addr #0 {
entry:
  %v0 = alloca i16, align 2
  %v1 = icmp eq i8* %p0, null
  br i1 %v1, label %if.then, label %lor.lhs.false

lor.lhs.false:                                    ; preds = %entry
  %v2 = bitcast i8* %p0 to %struct.0**
  %v3 = load %struct.0*, %struct.0** %v2, align 4
  %v4 = icmp eq %struct.0* %v3, null
  br i1 %v4, label %if.then, label %if.else

if.then:                                          ; preds = %lor.lhs.false, %ent
  %v5 = icmp eq i32* %p1, null
  br i1 %v5, label %cleanup, label %if.then3

if.then3:                                         ; preds = %if.then
  store i32 0, i32* %p1, align 4
  br label %cleanup

if.else:                                          ; preds = %lor.lhs.false
  %v6 = bitcast i16* %v0 to i8*
  call void @llvm.lifetime.start(i64 2, i8* nonnull %v6) #0
  store i16 0, i16* %v0, align 2
  %v7 = call i32 @foo(%struct.0* nonnull %v3, i16* nonnull %v0) #0
  %v8 = icmp eq i32* %p1, null
  br i1 %v8, label %if.end7, label %if.then6

if.then6:                                         ; preds = %if.else
  %v9 = load i16, i16* %v0, align 2
  %v10 = zext i16 %v9 to i32
  store i32 %v10, i32* %p1, align 4
  br label %if.end7

if.end7:                                          ; preds = %if.else, %if.then6
  call void @llvm.lifetime.end(i64 2, i8* nonnull %v6) #0
  br label %cleanup

cleanup:                                          ; preds = %if.then3, %if.then,
  %v11 = phi i32 [ %v7, %if.end7 ], [ -2147024809, %if.then ], [ -2147024809, %if.then3 ]
  ret i32 %v11
}

declare i32 @foo(%struct.0*, i16*) local_unnamed_addr #0

attributes #0 = { nounwind }
attributes #1 = { argmemonly nounwind }

