; RUN: llc < %s -mtriple=thumbv6-apple-darwin -relocation-model=pic -frame-pointer=all -mattr=+v6 -verify-machineinstrs | FileCheck %s
; rdar://7157006

%struct.FILE = type { i8*, i32, i32, i16, i16, %struct.__sbuf, i32, i8*, i32 (i8*)*, i32 (i8*, i8*, i32)*, i64 (i8*, i64, i32)*, i32 (i8*, i8*, i32)*, %struct.__sbuf, %struct.__sFILEX*, i32, [3 x i8], [1 x i8], %struct.__sbuf, i32, i64 }
%struct.__sFILEX = type opaque
%struct.__sbuf = type { i8*, i32 }
%struct.asl_file_t = type { i32, i32, i32, %struct.file_string_t*, i64, i64, i64, i64, i64, i64, i32, %struct.FILE*, i8*, i8* }
%struct.file_string_t = type { i64, i32, %struct.file_string_t*, [0 x i8] }

@llvm.used = appending global [1 x i8*] [i8* bitcast (i32 (%struct.asl_file_t*, i64, i64*)* @t to i8*)], section "llvm.metadata" ; <[1 x i8*]*> [#uses=0]

define i32 @t(%struct.asl_file_t* %s, i64 %off, i64* %out) nounwind optsize {
; CHECK-LABEL: t:
; CHECK: adds {{r[0-7]}}, #8
entry:
  %val = alloca i64, align 4                      ; <i64*> [#uses=3]
  %0 = icmp eq %struct.asl_file_t* %s, null       ; <i1> [#uses=1]
  br i1 %0, label %bb13, label %bb1

bb1:                                              ; preds = %entry
  %1 = getelementptr inbounds %struct.asl_file_t, %struct.asl_file_t* %s, i32 0, i32 11 ; <%struct.FILE**> [#uses=2]
  %2 = load %struct.FILE*, %struct.FILE** %1, align 4            ; <%struct.FILE*> [#uses=2]
  %3 = icmp eq %struct.FILE* %2, null             ; <i1> [#uses=1]
  br i1 %3, label %bb13, label %bb3

bb3:                                              ; preds = %bb1
  %4 = add nsw i64 %off, 8                        ; <i64> [#uses=1]
  %5 = getelementptr inbounds %struct.asl_file_t, %struct.asl_file_t* %s, i32 0, i32 10 ; <i32*> [#uses=1]
  %6 = load i32, i32* %5, align 4                      ; <i32> [#uses=1]
  %7 = zext i32 %6 to i64                         ; <i64> [#uses=1]
  %8 = icmp sgt i64 %4, %7                        ; <i1> [#uses=1]
  br i1 %8, label %bb13, label %bb5

bb5:                                              ; preds = %bb3
  %9 = call  i32 @fseeko(%struct.FILE* %2, i64 %off, i32 0) nounwind ; <i32> [#uses=1]
  %10 = icmp eq i32 %9, 0                         ; <i1> [#uses=1]
  br i1 %10, label %bb7, label %bb13

bb7:                                              ; preds = %bb5
  store i64 0, i64* %val, align 4
  %11 = load %struct.FILE*, %struct.FILE** %1, align 4           ; <%struct.FILE*> [#uses=1]
  %val8 = bitcast i64* %val to i8*                ; <i8*> [#uses=1]
  %12 = call  i32 @fread(i8* noalias %val8, i32 8, i32 1, %struct.FILE* noalias %11) nounwind ; <i32> [#uses=1]
  %13 = icmp eq i32 %12, 1                        ; <i1> [#uses=1]
  br i1 %13, label %bb10, label %bb13

bb10:                                             ; preds = %bb7
  %14 = icmp eq i64* %out, null                   ; <i1> [#uses=1]
  br i1 %14, label %bb13, label %bb11

bb11:                                             ; preds = %bb10
  %15 = load i64, i64* %val, align 4                   ; <i64> [#uses=1]
  %16 = call  i64 @asl_core_ntohq(i64 %15) nounwind ; <i64> [#uses=1]
  store i64 %16, i64* %out, align 4
  ret i32 0

bb13:                                             ; preds = %bb10, %bb7, %bb5, %bb3, %bb1, %entry
  %.0 = phi i32 [ 2, %entry ], [ 2, %bb1 ], [ 7, %bb3 ], [ 7, %bb5 ], [ 7, %bb7 ], [ 0, %bb10 ] ; <i32> [#uses=1]
  ret i32 %.0
}

declare i32 @fseeko(%struct.FILE* nocapture, i64, i32) nounwind

declare i32 @fread(i8* noalias nocapture, i32, i32, %struct.FILE* noalias nocapture) nounwind

declare i64 @asl_core_ntohq(i64)
