; RUN: opt %loadPolly -disable-basicaa -polly-scops -analyze < %s -stats 2>&1 | FileCheck %s --check-prefix=RTA
; RUN: opt %loadPolly -disable-basicaa -polly-scops -polly-use-runtime-alias-checks=false -analyze < %s -stats 2>&1 | FileCheck %s --check-prefix=NORTA
; REQUIRES: asserts

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"

define void @f(i32* noalias nocapture %a, i32* noalias nocapture %b) nounwind {
bb.nph:
  br label %bb

bb:                                               ; preds = %bb, %bb.nph
  %i.03 = phi i64 [ 0, %bb.nph ], [ %2, %bb ]     ; <i64> [#uses=3]
  %scevgep = getelementptr i32, i32* %b, i64 %i.03     ; <i32*> [#uses=1]
  %scevgep4 = getelementptr i32, i32* %a, i64 %i.03    ; <i32*> [#uses=1]
  %0 = load i32, i32* %scevgep, align 4                ; <i32> [#uses=1]
  %1 = add nsw i32 %0, 2                          ; <i32> [#uses=1]
  store i32 %1, i32* %scevgep4, align 4
  %2 = add nsw i64 %i.03, 1                       ; <i64> [#uses=2]
  %exitcond = icmp eq i64 %2, 128                 ; <i1> [#uses=1]
  br i1 %exitcond, label %return, label %bb

return:                                           ; preds = %bb
  ret void
}


; RTA:   1 polly-detect     - Number of regions that a valid part of Scop
; NORTA: 1 polly-detect     - Number of bad regions for Scop: Found base address alias
