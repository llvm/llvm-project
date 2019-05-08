; RUN: opt -S -licm -loop-unswitch -enable-mssa-loop-dependency -verify-memoryssa %s | FileCheck %s
; REQUIRES: asserts
target datalayout = "E-m:e-i1:8:16-i8:8:16-i64:64-f128:64-v128:64-a:8:16-n32:64"
target triple = "s390x-ibm-linux"

@g_92 = external dso_local local_unnamed_addr global i16, align 2
@g_993 = external dso_local local_unnamed_addr global i32, align 4

; CHECK-LABEL: @ff6
define dso_local fastcc void @ff6(i16 %arg1) unnamed_addr #0 {
bb:
  %tmp6.i = icmp sgt i16 %arg1, 0
  br label %bb10

bb10:                                             ; preds = %bb81.loopexit, %bb
  %tmp17 = load i16, i16* @g_92, align 2
  %tmp18 = add i16 %tmp17, 1
  store i16 %tmp18, i16* @g_92, align 2
  br label %bb19

bb19:                                             ; preds = %bb42, %bb10
  br label %bb24.preheader

bb24.preheader:                                   ; preds = %bb75, %bb19
  store i32 0, i32* @g_993, align 4
  br i1 %tmp6.i, label %bb24.preheader.split.us, label %bb24.preheader.split

bb24.preheader.split.us:                          ; preds = %bb24.preheader
  br label %bb61.us

bb67.us.loopexit:                                 ; preds = %bb65.us
  br label %bb75

bb61.us:                                          ; preds = %bb65.us, %bb24.preheader.split.us
  br i1 false, label %bb65.us, label %bb81.loopexit

bb65.us:                                          ; preds = %bb61.us
  br i1 false, label %bb61.us, label %bb67.us.loopexit

bb24.preheader.split:                             ; preds = %bb24.preheader
  br label %bb27

bb27:                                             ; preds = %bb24.preheader.split
  br i1 false, label %bb42, label %bb67

bb42:                                             ; preds = %bb27
  br label %bb19

bb67:                                             ; preds = %bb27
  br label %bb75

bb75:                                             ; preds = %bb67, %bb67.us.loopexit
  br i1 undef, label %bb24.preheader, label %bb84.loopexit

bb81.loopexit:                                    ; preds = %bb61.us
  br label %bb10

bb84.loopexit:                                    ; preds = %bb75
  ret void
}

attributes #0 = { "target-features"="+transactional-execution,+vector" }
