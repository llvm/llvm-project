; REQUIRES: asserts
; RUN: opt -passes=slsr -enable-partial-strength-reduce=0 -stats -disable-output < %s 2>&1 | FileCheck %s --check-prefix=NO-PSR
; RUN: opt -passes=slsr -enable-partial-strength-reduce=1 -stats -disable-output < %s 2>&1 | FileCheck %s --check-prefix=PSR

; SLSR computes one candidate-basis difference. PSR also considers the two
; bases in %basis for %candidate. %other.basis has different symbolic leaves,
; so only %const.delta.basis requires another SCEV difference.
;
; NO-PSR: 1 slsr - Number of candidate-basis SCEV differences computed by SLSR
; PSR: 2 slsr - Number of candidate-basis SCEV differences computed by SLSR

target triple = "nvptx64-nvidia-cuda"

define ptr @different_unknowns(i1 %cond, i64 %idx, i64 %other) {
entry:
  %idx.minus.4 = sub i64 %idx, 4
  br i1 %cond, label %basis, label %join

basis:
  %const.delta.basis = getelementptr i8, ptr null, i64 %idx.minus.4
  %other.basis = getelementptr i8, ptr null, i64 %other
  br label %join

join:
  %candidate = getelementptr i8, ptr null, i64 %idx
  ret ptr %candidate
}
