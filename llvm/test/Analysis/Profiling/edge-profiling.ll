; Test the edge profiling instrumentation.
; RUN: opt < %s -insert-edge-profiling -S | FileCheck %s

; ModuleID = '<stdin>'

@.str = private constant [12 x i8] c"hello world\00", align 1 ; <[12 x i8]*> [#uses=1]
@.str1 = private constant [6 x i8] c"franz\00", align 1 ; <[6 x i8]*> [#uses=1]
@.str2 = private constant [9 x i8] c"argc > 2\00", align 1 ; <[9 x i8]*> [#uses=1]
@.str3 = private constant [9 x i8] c"argc = 1\00", align 1 ; <[9 x i8]*> [#uses=1]
@.str4 = private constant [6 x i8] c"fritz\00", align 1 ; <[6 x i8]*> [#uses=1]
@.str5 = private constant [10 x i8] c"argc <= 1\00", align 1 ; <[10 x i8]*> [#uses=1]
; CHECK:@EdgeProfCounters
; CHECK:[19 x i32] 
; CHECK:zeroinitializer

define void @oneblock() nounwind {
entry:
; CHECK:entry:
; CHECK:%OldFuncCounter
; CHECK:load 
; CHECK:getelementptr
; CHECK:@EdgeProfCounters
; CHECK:i32 0
; CHECK:i32 0
; CHECK:%NewFuncCounter
; CHECK:add
; CHECK:%OldFuncCounter
; CHECK:store 
; CHECK:%NewFuncCounter
; CHECK:getelementptr
; CHECK:@EdgeProfCounters
  %0 = call i32 @puts(i8* getelementptr inbounds ([12 x i8]* @.str, i64 0, i64 0)) nounwind ; <i32> [#uses=0]
  ret void
}

declare i32 @puts(i8*)

define i32 @main(i32 %argc, i8** %argv) nounwind {
entry:
; CHECK:entry:
  %argc_addr = alloca i32                         ; <i32*> [#uses=4]
  %argv_addr = alloca i8**                        ; <i8***> [#uses=1]
  %retval = alloca i32                            ; <i32*> [#uses=2]
  %j = alloca i32                                 ; <i32*> [#uses=4]
  %i = alloca i32                                 ; <i32*> [#uses=4]
  %0 = alloca i32                                 ; <i32*> [#uses=2]
; CHECK:call 
; CHECK:@llvm_start_edge_profiling
; CHECK:@EdgeProfCounters
  %"alloca point" = bitcast i32 0 to i32          ; <i32> [#uses=0]
  store i32 %argc, i32* %argc_addr
  store i8** %argv, i8*** %argv_addr
  store i32 0, i32* %i, align 4
  br label %bb10

bb:                                               ; preds = %bb10
; CHECK:bb:
  %1 = load i32* %argc_addr, align 4              ; <i32> [#uses=1]
  %2 = icmp sgt i32 %1, 1                         ; <i1> [#uses=1]
  br i1 %2, label %bb1, label %bb8

bb1:                                              ; preds = %bb
; CHECK:bb1:
  store i32 0, i32* %j, align 4
  br label %bb6

bb2:                                              ; preds = %bb6
; CHECK:bb2:
  %3 = call i32 @puts(i8* getelementptr inbounds ([6 x i8]* @.str1, i64 0, i64 0)) nounwind ; <i32> [#uses=0]
  %4 = load i32* %argc_addr, align 4              ; <i32> [#uses=1]
  %5 = icmp sgt i32 %4, 2                         ; <i1> [#uses=1]
  br i1 %5, label %bb3, label %bb4

bb3:                                              ; preds = %bb2
; CHECK:bb3:
  %6 = call i32 @puts(i8* getelementptr inbounds ([9 x i8]* @.str2, i64 0, i64 0)) nounwind ; <i32> [#uses=0]
  br label %bb5

bb4:                                              ; preds = %bb2
; CHECK:bb4:
  %7 = call i32 @puts(i8* getelementptr inbounds ([9 x i8]* @.str3, i64 0, i64 0)) nounwind ; <i32> [#uses=0]
  br label %bb11

bb5:                                              ; preds = %bb3
; CHECK:bb5:
  %8 = call i32 @puts(i8* getelementptr inbounds ([6 x i8]* @.str4, i64 0, i64 0)) nounwind ; <i32> [#uses=0]
  %9 = load i32* %j, align 4                      ; <i32> [#uses=1]
  %10 = add nsw i32 %9, 1                         ; <i32> [#uses=1]
  store i32 %10, i32* %j, align 4
  br label %bb6

bb6:                                              ; preds = %bb5, %bb1
; CHECK:bb6:
  %11 = load i32* %j, align 4                     ; <i32> [#uses=1]
  %12 = load i32* %argc_addr, align 4             ; <i32> [#uses=1]
  %13 = icmp slt i32 %11, %12                     ; <i1> [#uses=1]
  br i1 %13, label %bb2, label %bb7

bb7:                                              ; preds = %bb6
; CHECK:bb7:
  br label %bb9

bb8:                                              ; preds = %bb
; CHECK:bb8:
  %14 = call i32 @puts(i8* getelementptr inbounds ([10 x i8]* @.str5, i64 0, i64 0)) nounwind ; <i32> [#uses=0]
  br label %bb9

bb9:                                              ; preds = %bb8, %bb7
; CHECK:bb9:
  %15 = load i32* %i, align 4                     ; <i32> [#uses=1]
  %16 = add nsw i32 %15, 1                        ; <i32> [#uses=1]
  store i32 %16, i32* %i, align 4
  br label %bb10

bb10:                                             ; preds = %bb9, %entry
; CHECK:bb10:
  %17 = load i32* %i, align 4                     ; <i32> [#uses=1]
  %18 = icmp ne i32 %17, 3                        ; <i1> [#uses=1]
  br i1 %18, label %bb, label %bb11
; CHECK:br
; CHECK:label %bb10.bb11_crit_edge

; CHECK:bb10.bb11_crit_edge:
; CHECK:br
; CHECK:label %bb11

bb11:                                             ; preds = %bb10, %bb4
; CHECK:bb11:
  call void @oneblock() nounwind
  store i32 0, i32* %0, align 4
  %19 = load i32* %0, align 4                     ; <i32> [#uses=1]
  store i32 %19, i32* %retval, align 4
  br label %return

return:                                           ; preds = %bb11
; CHECK:return:
  %retval12 = load i32* %retval                   ; <i32> [#uses=1]
  ret i32 %retval12
}
