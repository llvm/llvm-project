; RUN: llc -verify-machineinstrs -mcpu=i386 < %s
target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:32:32-n8:16:32"
target triple = "i386-pc-linux-gnu"

; The bb.i basic block gets split while emitting the schedule because
; -mcpu=i386 doesn't have CMOV.'
;
; That causes the PHI to be updated wrong because the jumptable data structure is remembering the original MBB.
;
; -cgp-critical-edge-splitting=0 prevents the edge to PHI from being split.

@.str146 = external constant [4 x i8], align 1
@.str706 = external constant [4 x i8], align 1
@.str1189 = external constant [5 x i8], align 1

declare i32 @memcmp(ptr nocapture, ptr nocapture, i32) nounwind readonly
declare i32 @strlen(ptr nocapture) nounwind readonly

define hidden zeroext i8 @f(ptr %this, ptr %Name.0, i32 %Name.1, ptr noalias %NameLoc, ptr %Operands) nounwind align 2 {
bb.i:
  %0 = icmp eq i8 undef, 0
  %iftmp.285.0 = select i1 %0, ptr @.str1189, ptr @.str706
  %1 = call i32 @strlen(ptr %iftmp.285.0) nounwind readonly
  switch i32 %Name.1, label %_ZNK4llvm12StringSwitchINS_9StringRefES1_E7DefaultERKS1_.exit [
    i32 3, label %bb1.i
    i32 4, label %bb1.i1237
    i32 5, label %bb1.i1266
    i32 6, label %bb1.i1275
    i32 2, label %bb1.i1434
    i32 8, label %bb1.i1523
    i32 7, label %bb1.i1537
  ]

bb1.i:                                            ; preds = %bb.i
  unreachable

bb1.i1237:                                        ; preds = %bb.i
  br i1 undef, label %bb.i1820, label %bb1.i1241

bb1.i1241:                                        ; preds = %bb1.i1237
  unreachable

bb1.i1266:                                        ; preds = %bb.i
  unreachable

bb1.i1275:                                        ; preds = %bb.i
  unreachable

bb1.i1434:                                        ; preds = %bb.i
  unreachable

bb1.i1523:                                        ; preds = %bb.i
  unreachable

bb1.i1537:                                        ; preds = %bb.i
  unreachable

bb.i1820:                                         ; preds = %bb1.i1237
  br label %_ZNK4llvm12StringSwitchINS_9StringRefES1_E7DefaultERKS1_.exit

_ZNK4llvm12StringSwitchINS_9StringRefES1_E7DefaultERKS1_.exit: ; preds = %bb.i1820, %bb.i
  %PatchedName.0.0 = phi ptr [ undef, %bb.i1820 ], [ %Name.0, %bb.i ]
  br i1 undef, label %bb141, label %_ZNK4llvm9StringRef10startswithES0_.exit

_ZNK4llvm9StringRef10startswithES0_.exit:         ; preds = %_ZNK4llvm12StringSwitchINS_9StringRefES1_E7DefaultERKS1_.exit
  %2 = call i32 @memcmp(ptr %PatchedName.0.0, ptr @.str146, i32 3) nounwind readonly
  unreachable

bb141:                                            ; preds = %_ZNK4llvm12StringSwitchINS_9StringRefES1_E7DefaultERKS1_.exit
  unreachable
}
