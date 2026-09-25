; RUN: opt %loadNPMPolly -passes="polly-custom<detect>" -polly-invariant-load-hoisting=true -polly-print-detect -disable-output %s | FileCheck %s --check-prefix=DETECT
; RUN: opt %loadNPMPolly -passes="polly-custom<detect;codegen>" -polly-invariant-load-hoisting=true -S %s | FileCheck %s --check-prefix=CODEGEN
;
; https://github.com/llvm/llvm-project/issues/192208
; If not already cached, getSCEV(%phi) would try to re-derive the
; SCEVAddRecExpr which will not work during the codegen phase where the SSA form
; has not been restored yet.
;
; DETECT: Valid Region for Scop: bb3 => bb5
; DETECT: Valid Region for Scop: bb10.peel => bb3
;
; CODEGEN: polly.start:
; CODEGEN: polly.start6:

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"


; Function Attrs: nofree norecurse nosync nounwind memory(argmem: readwrite)
define noundef i32 @wombat(ptr writeonly captures(none) initializes((0, 1)) %arg, ptr readonly captures(none) %arg1, i1 %arg2) local_unnamed_addr #0 {
bb:
  br label %bb3

bb3.loopexit:                                     ; preds = %bb10, %bb10.peel
  %add11.lcssa = phi i32 [ %add11.peel, %bb10.peel ], [ %add11, %bb10 ]
  br label %bb3

bb3:                                              ; preds = %bb3.loopexit, %bb
  %phi = phi i32 [ 0, %bb ], [ %add11.lcssa, %bb3.loopexit ]
  %load = load i8, ptr %arg1, align 4
  %icmp = icmp eq i8 %load, 0
  br i1 %icmp, label %bb4, label %bb5

bb4:                                              ; preds = %bb3
  store i32 0, ptr %arg, align 8
  br label %bb5

bb5:                                              ; preds = %bb4, %bb3
  store i8 0, ptr %arg, align 1
  br i1 %arg2, label %bb9, label %bb10.peel

bb10.peel:                                        ; preds = %bb5
  %zext = zext i8 %load to i32
  %add = add nuw nsw i32 %phi, %zext
  %add11.peel = add nuw nsw i32 %add, 1
  %icmp12.peel = icmp samesign ugt i32 %add, 1
  br i1 %icmp12.peel, label %bb10, label %bb3.loopexit

bb9:                                              ; preds = %bb5
  ret i32 0

bb10:                                             ; preds = %bb10, %bb10.peel
  %phi8 = phi i32 [ %add11, %bb10 ], [ %add11.peel, %bb10.peel ]
  %add11 = add i32 %phi8, 1
  %icmp12 = icmp ugt i32 %phi8, 1
  br i1 %icmp12, label %bb10, label %bb3.loopexit, !llvm.loop !0
}

attributes #0 = { nofree norecurse nosync nounwind memory(argmem: readwrite) }

!0 = distinct !{!0, !1}
!1 = !{!"llvm.loop.peeled.count", i32 1}
