; RUN: llc < %s -mtriple=thumbv7-apple-darwin -O3 -relocation-model=pic -arm-atomic-cfg-tidy=0 | FileCheck %s
; rdar://8115404
; Tail merging must not split an IT block.

%struct.FILE = type { i8*, i32, i32, i16, i16, %struct.__sbuf, i32, i8*, i32 (i8*)*, i32 (i8*, i8*, i32)*, i64 (i8*, i64, i32)*, i32 (i8*, i8*, i32)*, %struct.__sbuf, %struct.__sFILEX*, i32, [3 x i8], [1 x i8], %struct.__sbuf, i32, i64 }
%struct._RuneCharClass = type { [14 x i8], i32 }
%struct._RuneEntry = type { i32, i32, i32, i32* }
%struct._RuneLocale = type { [8 x i8], [32 x i8], i32 (i8*, i32, i8**)*, i32 (i32, i8*, i32, i8**)*, i32, [256 x i32], [256 x i32], [256 x i32], %struct._RuneRange, %struct._RuneRange, %struct._RuneRange, i8*, i32, i32, %struct._RuneCharClass* }
%struct._RuneRange = type { i32, %struct._RuneEntry* }
%struct.__sFILEX = type opaque
%struct.__sbuf = type { i8*, i32 }

@finput = external global %struct.FILE*           ; <%struct.FILE**> [#uses=1]
@_DefaultRuneLocale = external global %struct._RuneLocale ; <%struct._RuneLocale*> [#uses=0]
@token_buffer = external global [1025 x i8], align 4 ; <[1025 x i8]*> [#uses=1]
@.str73 = external constant [6 x i8], align 4     ; <[6 x i8]*> [#uses=0]
@.str174 = external constant [5 x i8], align 4    ; <[5 x i8]*> [#uses=0]
@.str275 = external constant [6 x i8], align 4    ; <[6 x i8]*> [#uses=0]
@.str376 = external constant [5 x i8], align 4    ; <[5 x i8]*> [#uses=0]
@.str477 = external constant [6 x i8], align 4    ; <[6 x i8]*> [#uses=0]
@.str578 = external constant [6 x i8], align 4    ; <[6 x i8]*> [#uses=0]
@.str679 = external constant [7 x i8], align 4    ; <[7 x i8]*> [#uses=0]
@.str780 = external constant [6 x i8], align 4    ; <[6 x i8]*> [#uses=0]
@.str881 = external constant [5 x i8], align 4    ; <[5 x i8]*> [#uses=0]
@.str982 = external constant [6 x i8], align 4    ; <[6 x i8]*> [#uses=0]
@.str1083 = external constant [9 x i8], align 4   ; <[9 x i8]*> [#uses=0]
@.str1184 = external constant [7 x i8], align 4   ; <[7 x i8]*> [#uses=0]
@.str1285 = external constant [16 x i8], align 4  ; <[16 x i8]*> [#uses=0]
@.str1386 = external constant [12 x i8], align 4  ; <[12 x i8]*> [#uses=0]
@.str1487 = external constant [5 x i8], align 4   ; <[5 x i8]*> [#uses=0]
@llvm.used = external global [1 x i8*]            ; <[1 x i8*]*> [#uses=0]

define fastcc i32 @parse_percent_token() nounwind {
entry:
; CHECK: pop
; CHECK: pop
; CHECK: pop
; CHECK: pop
; CHECK: pop
; CHECK: pop
; CHECK: pop
; Do not convert into single stream code. BranchProbability Analysis assumes
; that branches which goes to "ret" instruction have lower probabilities.
  switch i32 undef, label %bb7 [
    i32 37, label %bb43
    i32 48, label %bb5
    i32 50, label %bb4
    i32 60, label %bb2
    i32 61, label %bb6
    i32 62, label %bb3
    i32 123, label %bb1
  ]

bb1:                                              ; preds = %entry
  ret i32 8

bb2:                                              ; preds = %entry
  ret i32 15

bb3:                                              ; preds = %entry
  ret i32 16

bb4:                                              ; preds = %entry
  ret i32 17

bb5:                                              ; preds = %entry
  ret i32 9

bb6:                                              ; preds = %entry
  ret i32 18

bb7:                                              ; preds = %entry
  br i1 undef, label %bb.i.i, label %bb1.i.i

bb.i.i:                                           ; preds = %bb7
  br i1 undef, label %bb43, label %bb12

bb1.i.i:                                          ; preds = %bb7
  unreachable

bb9:                                              ; preds = %bb.i.i2
  br i1 undef, label %bb10, label %bb11

bb10:                                             ; preds = %bb9
  br label %bb11

bb11:                                             ; preds = %bb10, %bb9
  %p.0 = phi i8* [ undef, %bb10 ], [ %p.1, %bb9 ] ; <i8*> [#uses=1]
  %0 = load %struct.FILE*, %struct.FILE** @finput, align 4       ; <%struct.FILE*> [#uses=1]
  %1 = tail call i32 @getc(%struct.FILE* %0) nounwind ; <i32> [#uses=0]
  br label %bb12

bb12:                                             ; preds = %bb11, %bb.i.i
  %p.1 = phi i8* [ %p.0, %bb11 ], [ getelementptr inbounds ([1025 x i8], [1025 x i8]* @token_buffer, i32 0, i32 0), %bb.i.i ] ; <i8*> [#uses=2]
  %2 = icmp ult i32 undef, 128                    ; <i1> [#uses=1]
  br i1 %2, label %bb.i.i2, label %bb1.i.i3

bb.i.i2:                                          ; preds = %bb12
  %3 = load i32, i32* null, align 4                    ; <i32> [#uses=1]
  %4 = lshr i32 %3, 8                             ; <i32> [#uses=1]
  %.lobit.i1 = and i32 %4, 1                      ; <i32> [#uses=1]
  %.not = icmp ne i32 %.lobit.i1, 0               ; <i1> [#uses=1]
  %or.cond = or i1 %.not, undef                   ; <i1> [#uses=1]
  br i1 %or.cond, label %bb9, label %bb14

bb1.i.i3:                                         ; preds = %bb12
  unreachable

bb14:                                             ; preds = %bb.i.i2
  store i8 0, i8* %p.1, align 1
  br i1 undef, label %bb43, label %bb15

bb15:                                             ; preds = %bb14
  unreachable

bb43:                                             ; preds = %bb14, %bb.i.i, %entry
  %.0 = phi i32 [ 7, %entry ], [ 24, %bb.i.i ], [ 9, %bb14 ] ; <i32> [#uses=1]
  ret i32 %.0
}

declare i32 @getc(%struct.FILE* nocapture) nounwind

declare i32 @strcmp(i8* nocapture, i8* nocapture) nounwind readonly

declare i32 @__maskrune(i32, i32)

declare i32 @ungetc(i32, %struct.FILE* nocapture) nounwind
