; RUN: llc < %s -mtriple=thumbv6-apple-darwin -relocation-model=pic -frame-pointer=all -mattr=+v6 -verify-machineinstrs | FileCheck %s
; rdar://7157006

%struct.FILE = type { ptr, i32, i32, i16, i16, %struct.__sbuf, i32, ptr, ptr, ptr, ptr, ptr, %struct.__sbuf, ptr, i32, [3 x i8], [1 x i8], %struct.__sbuf, i32, i64 }
%struct.__sFILEX = type opaque
%struct.__sbuf = type { ptr, i32 }
%struct.asl_file_t = type { i32, i32, i32, ptr, i64, i64, i64, i64, i64, i64, i32, ptr, ptr, ptr }
%struct.file_string_t = type { i64, i32, ptr, [0 x i8] }

@llvm.used = appending global [1 x ptr] [ptr @t], section "llvm.metadata" ; <ptr> [#uses=0]

define i32 @t(ptr %s, i64 %off, ptr %out) nounwind optsize {
; CHECK-LABEL: t:
; CHECK: adds {{r[0-7]}}, #8
entry:
  %val = alloca i64, align 4                      ; <ptr> [#uses=3]
  %0 = icmp eq ptr %s, null       ; <i1> [#uses=1]
  br i1 %0, label %bb13, label %bb1

bb1:                                              ; preds = %entry
  %1 = getelementptr inbounds %struct.asl_file_t, ptr %s, i32 0, i32 11 ; <ptr> [#uses=2]
  %2 = load ptr, ptr %1, align 4            ; <ptr> [#uses=2]
  %3 = icmp eq ptr %2, null             ; <i1> [#uses=1]
  br i1 %3, label %bb13, label %bb3

bb3:                                              ; preds = %bb1
  %4 = add nsw i64 %off, 8                        ; <i64> [#uses=1]
  %5 = getelementptr inbounds %struct.asl_file_t, ptr %s, i32 0, i32 10 ; <ptr> [#uses=1]
  %6 = load i32, ptr %5, align 4                      ; <i32> [#uses=1]
  %7 = zext i32 %6 to i64                         ; <i64> [#uses=1]
  %8 = icmp sgt i64 %4, %7                        ; <i1> [#uses=1]
  br i1 %8, label %bb13, label %bb5

bb5:                                              ; preds = %bb3
  %9 = call  i32 @fseeko(ptr %2, i64 %off, i32 0) nounwind ; <i32> [#uses=1]
  %10 = icmp eq i32 %9, 0                         ; <i1> [#uses=1]
  br i1 %10, label %bb7, label %bb13

bb7:                                              ; preds = %bb5
  store i64 0, ptr %val, align 4
  %11 = load ptr, ptr %1, align 4           ; <ptr> [#uses=1]
  %12 = call  i32 @fread(ptr noalias %val, i32 8, i32 1, ptr noalias %11) nounwind ; <i32> [#uses=1]
  %13 = icmp eq i32 %12, 1                        ; <i1> [#uses=1]
  br i1 %13, label %bb10, label %bb13

bb10:                                             ; preds = %bb7
  %14 = icmp eq ptr %out, null                   ; <i1> [#uses=1]
  br i1 %14, label %bb13, label %bb11

bb11:                                             ; preds = %bb10
  %15 = load i64, ptr %val, align 4                   ; <i64> [#uses=1]
  %16 = call  i64 @asl_core_ntohq(i64 %15) nounwind ; <i64> [#uses=1]
  store i64 %16, ptr %out, align 4
  ret i32 0

bb13:                                             ; preds = %bb10, %bb7, %bb5, %bb3, %bb1, %entry
  %.0 = phi i32 [ 2, %entry ], [ 2, %bb1 ], [ 7, %bb3 ], [ 7, %bb5 ], [ 7, %bb7 ], [ 0, %bb10 ] ; <i32> [#uses=1]
  ret i32 %.0
}

declare i32 @fseeko(ptr nocapture, i64, i32) nounwind

declare i32 @fread(ptr noalias nocapture, i32, i32, ptr noalias nocapture) nounwind

declare i64 @asl_core_ntohq(i64)
