; RUN: llc -asm-verbose=false -O3 -relocation-model=pic -disable-fp-elim -mtriple=thumbv7-apple-darwin -mcpu=cortex-a8 < %s | FileCheck %s

target datalayout = "e-p:32:32:32-i1:8:32-i8:8:32-i16:16:32-i32:32:32-i64:32:32-f32:32:32-f64:32:32-v64:64:64-v128:128:128-a0:0:32"
target triple = "thumbv7-apple-darwin9"

@history = internal global [2 x [56 x i32]] [[56 x i32] [i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 0, i32 1, i32 2, i32 4, i32 2, i32 1, i32 0, i32 -1, i32 1, i32 3, i32 5, i32 7, i32 5, i32 3, i32 1, i32 -1, i32 2, i32 5, i32 8, i32 10, i32 8, i32 5, i32 2, i32 -1, i32 2, i32 5, i32 8, i32 10, i32 8, i32 5, i32 2, i32 -1, i32 1, i32 3, i32 5, i32 7, i32 5, i32 3, i32 1, i32 -1, i32 0, i32 1, i32 2, i32 4, i32 2, i32 1, i32 0], [56 x i32] [i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 0, i32 1, i32 2, i32 4, i32 2, i32 1, i32 0, i32 -1, i32 1, i32 3, i32 5, i32 7, i32 5, i32 3, i32 1, i32 -1, i32 2, i32 5, i32 8, i32 10, i32 8, i32 5, i32 2, i32 -1, i32 2, i32 5, i32 8, i32 10, i32 8, i32 5, i32 2, i32 -1, i32 1, i32 3, i32 5, i32 7, i32 5, i32 3, i32 1, i32 -1, i32 0, i32 1, i32 2, i32 4, i32 2, i32 1, i32 0]] ; <[2 x [56 x i32]]*> [#uses=3]
@nodes = internal global i64 0                    ; <i64*> [#uses=4]
@.str = private constant [9 x i8] c"##-<=>+#\00", align 1 ; <[9 x i8]*> [#uses=2]
@.str1 = private constant [6 x i8] c"%c%d\0A\00", align 1 ; <[6 x i8]*> [#uses=1]
@.str2 = private constant [16 x i8] c"Fhourstones 2.0\00", align 1 ; <[16 x i8]*> [#uses=1]
@.str3 = private constant [54 x i8] c"Using %d transposition table entries with %d probes.\0A\00", align 1 ; <[54 x i8]*> [#uses=1]
@.str4 = private constant [31 x i8] c"Solving %d-ply position after \00", align 1 ; <[31 x i8]*> [#uses=1]
@.str5 = private constant [7 x i8] c" . . .\00", align 1 ; <[7 x i8]*> [#uses=1]
@.str6 = private constant [28 x i8] c"score = %d (%c)  work = %d\0A\00", align 1 ; <[28 x i8]*> [#uses=1]
@.str7 = private constant [36 x i8] c"%lu pos / %lu msec = %.1f Kpos/sec\0A\00", align 1 ; <[36 x i8]*> [#uses=1]
@plycnt = internal global i32 0                   ; <i32*> [#uses=21]
@dias = internal global [19 x i32] zeroinitializer ; <[19 x i32]*> [#uses=43]
@columns = internal global [128 x i32] zeroinitializer ; <[128 x i32]*> [#uses=18]
@height = internal global [128 x i32] zeroinitializer ; <[128 x i32]*> [#uses=21]
@rows = internal global [8 x i32] zeroinitializer ; <[8 x i32]*> [#uses=20]
@colthr = internal global [128 x i32] zeroinitializer ; <[128 x i32]*> [#uses=5]
@moves = internal global [44 x i32] zeroinitializer ; <[44 x i32]*> [#uses=9]
@.str8 = private constant [3 x i8] c"%d\00", align 1 ; <[3 x i8]*> [#uses=1]
@he = internal global i8* null                    ; <i8**> [#uses=9]
@hits = internal global i64 0                     ; <i64*> [#uses=8]
@posed = internal global i64 0                    ; <i64*> [#uses=7]
@ht = internal global i32* null                   ; <i32**> [#uses=5]
@.str16 = private constant [19 x i8] c"store rate = %.3f\0A\00", align 1 ; <[19 x i8]*> [#uses=1]
@.str117 = private constant [45 x i8] c"- %5.3f  < %5.3f  = %5.3f  > %5.3f  + %5.3f\0A\00", align 1 ; <[45 x i8]*> [#uses=1]
@.str218 = private constant [6 x i8] c"%7d%c\00", align 1 ; <[6 x i8]*> [#uses=1]
@.str319 = private constant [30 x i8] c"Failed to allocate %u bytes.\0A\00", align 1 ; <[30 x i8]*> [#uses=1]

declare i32 @puts(i8* nocapture) nounwind

declare i32 @getchar() nounwind

define internal i32 @transpose() nounwind readonly {
; CHECK: push
entry:
  %0 = load i32* getelementptr inbounds ([128 x i32]* @columns, i32 0, i32 1), align 4 ; <i32> [#uses=1]
  %1 = shl i32 %0, 7                              ; <i32> [#uses=1]
  %2 = load i32* getelementptr inbounds ([128 x i32]* @columns, i32 0, i32 2), align 4 ; <i32> [#uses=1]
  %3 = or i32 %1, %2                              ; <i32> [#uses=1]
  %4 = shl i32 %3, 7                              ; <i32> [#uses=1]
  %5 = load i32* getelementptr inbounds ([128 x i32]* @columns, i32 0, i32 3), align 4 ; <i32> [#uses=1]
  %6 = or i32 %4, %5                              ; <i32> [#uses=3]
  %7 = load i32* getelementptr inbounds ([128 x i32]* @columns, i32 0, i32 7), align 4 ; <i32> [#uses=1]
  %8 = shl i32 %7, 7                              ; <i32> [#uses=1]
  %9 = load i32* getelementptr inbounds ([128 x i32]* @columns, i32 0, i32 6), align 4 ; <i32> [#uses=1]
  %10 = or i32 %8, %9                             ; <i32> [#uses=1]
  %11 = shl i32 %10, 7                            ; <i32> [#uses=1]
  %12 = load i32* getelementptr inbounds ([128 x i32]* @columns, i32 0, i32 5), align 4 ; <i32> [#uses=1]
  %13 = or i32 %11, %12                           ; <i32> [#uses=3]
  %14 = icmp ugt i32 %6, %13                      ; <i1> [#uses=2]
  %.pn2.in.i = select i1 %14, i32 %6, i32 %13     ; <i32> [#uses=1]
  %.pn1.in.i = select i1 %14, i32 %13, i32 %6     ; <i32> [#uses=1]
  %.pn2.i = shl i32 %.pn2.in.i, 7                 ; <i32> [#uses=1]
  %.pn3.i = load i32* getelementptr inbounds ([128 x i32]* @columns, i32 0, i32 4) ; <i32> [#uses=1]
  %.pn.in.in.i = or i32 %.pn2.i, %.pn3.i          ; <i32> [#uses=1]
  %.pn.in.i = zext i32 %.pn.in.in.i to i64        ; <i64> [#uses=1]
  %.pn.i = shl i64 %.pn.in.i, 21                  ; <i64> [#uses=1]
  %.pn1.i = zext i32 %.pn1.in.i to i64            ; <i64> [#uses=1]
  %iftmp.22.0.i = or i64 %.pn.i, %.pn1.i          ; <i64> [#uses=2]
  %15 = lshr i64 %iftmp.22.0.i, 17                ; <i64> [#uses=1]
  %16 = trunc i64 %15 to i32                      ; <i32> [#uses=2]
  %17 = urem i64 %iftmp.22.0.i, 1050011           ; <i64> [#uses=1]
  %18 = trunc i64 %17 to i32                      ; <i32> [#uses=1]
  %19 = urem i32 %16, 179                         ; <i32> [#uses=1]
  %20 = or i32 %19, 131072                        ; <i32> [#uses=1]
  %21 = load i32** @ht, align 4                   ; <i32*> [#uses=1]
  br label %bb5

bb:                                               ; preds = %bb5
  %22 = getelementptr inbounds i32* %21, i32 %x.0 ; <i32*> [#uses=1]
  %23 = load i32* %22, align 4                    ; <i32> [#uses=1]
  %24 = icmp eq i32 %23, %16                      ; <i1> [#uses=1]
  br i1 %24, label %bb1, label %bb2

bb1:                                              ; preds = %bb
  %25 = load i8** @he, align 4                    ; <i8*> [#uses=1]
  %26 = getelementptr inbounds i8* %25, i32 %x.0  ; <i8*> [#uses=1]
  %27 = load i8* %26, align 1                     ; <i8> [#uses=1]
  %28 = sext i8 %27 to i32                        ; <i32> [#uses=1]
  ret i32 %28

bb2:                                              ; preds = %bb
  %29 = add nsw i32 %20, %x.0                     ; <i32> [#uses=3]
  %30 = add i32 %29, -1050011                     ; <i32> [#uses=1]
  %31 = icmp sgt i32 %29, 1050010                 ; <i1> [#uses=1]
  %. = select i1 %31, i32 %30, i32 %29            ; <i32> [#uses=1]
  %32 = add i32 %33, 1                            ; <i32> [#uses=1]
  br label %bb5

bb5:                                              ; preds = %bb2, %entry
  %33 = phi i32 [ 0, %entry ], [ %32, %bb2 ]      ; <i32> [#uses=2]
  %x.0 = phi i32 [ %18, %entry ], [ %., %bb2 ]    ; <i32> [#uses=3]
  %34 = icmp sgt i32 %33, 7                       ; <i1> [#uses=1]
  br i1 %34, label %bb7, label %bb

bb7:                                              ; preds = %bb5
  ret i32 -128
}

declare noalias i8* @calloc(i32, i32) nounwind

declare void @llvm.memset.i64(i8* nocapture, i8, i64, i32) nounwind
