; RUN: opt -licm -verify-memoryssa -S < %s | FileCheck %s
; REQUIRES: asserts

target datalayout = "E-m:e-i1:8:16-i8:8:16-i64:64-f128:64-v128:64-a:8:16-n32:64"
target triple = "s390x-ibm-linux"

@g_328 = external dso_local local_unnamed_addr global { i32, i16, i32, i8, i8, i32, i32 }, align 4

define dso_local void @func_45() local_unnamed_addr {
; CHECK-LABEL: @func_45()
bb:
  br label %bb7

bb7:                                              ; preds = %bb
  br label %bb8

bb8:                                              ; preds = %bb80, %bb7
  %tmp10 = load i32, ptr getelementptr inbounds ({ i32, i16, i32, i8, i8, i32, i32 }, ptr @g_328, i64 0, i32 5), align 4
  %0 = or i32 %tmp10, 9
  store i32 %0, ptr getelementptr inbounds ({ i32, i16, i32, i8, i8, i32, i32 }, ptr @g_328, i64 0, i32 5), align 4
  br label %bb41.preheader.preheader

bb41.preheader.preheader:                         ; preds = %bb80.thread, %bb8
  br label %bb68

bb84.thread.split.loop.exit67:                    ; preds = %bb71.1
  br label %bb84.thread

bb84.thread.split.loop.exit71:                    ; preds = %bb71.2
  br label %bb84.thread

bb84.thread.split.loop.exit91:                    ; preds = %bb71.1.2
  br label %bb84.thread

bb84.thread:                                      ; preds = %bb84.thread.split.loop.exit91, %bb84.thread.split.loop.exit71, %bb84.thread.split.loop.exit67
  unreachable

bb68:                                             ; preds = %bb41.preheader.preheader
  br i1 false, label %bb71, label %bb80

bb71:                                             ; preds = %bb68
  br label %bb71.1

bb80.thread:                                      ; preds = %bb71.1.2
  br label %bb41.preheader.preheader

bb80:                                             ; preds = %bb68
  br label %bb8

bb71.1:                                           ; preds = %bb71
  br i1 true, label %bb84.thread.split.loop.exit67, label %bb71.2

bb71.2:                                           ; preds = %bb71.1
  br i1 true, label %bb84.thread.split.loop.exit71, label %bb71.145

bb71.145:                                         ; preds = %bb71.2
  br label %bb71.1.2

bb71.1.2:                                         ; preds = %bb71.145
  br i1 true, label %bb84.thread.split.loop.exit91, label %bb80.thread
}

