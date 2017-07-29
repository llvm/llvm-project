; REQUIRES: asserts
; RUN: opt -regions -analyze < %s | FileCheck %s
; RUN: opt -regions -stats -disable-output < %s 2>&1 | FileCheck -check-prefix=STAT %s
; RUN: opt -regions -print-region-style=bb  -analyze < %s 2>&1 | FileCheck -check-prefix=BBIT %s
; RUN: opt -regions -print-region-style=rn  -analyze < %s 2>&1 | FileCheck -check-prefix=RNIT %s

; RUN: opt < %s -passes='print<regions>' 2>&1 | FileCheck %s

define void @MAIN__() nounwind {
entry:
  br label %__label_002001.outer

__label_002001.outer:                             ; preds = %bb236, %bb92
  br label %__label_002001

__label_002001:                                   ; preds = %bb229, %__label_002001.outer
  br i1  1, label %bb93, label %__label_000020

bb93:                                             ; preds = %__label_002001
  br i1  1, label %__label_000020, label %bb197

bb197:                                            ; preds = %bb193
  br i1  1, label %bb229, label %bb224

bb224:                                            ; preds = %bb223, %bb227
  br i1  1, label %bb229, label %bb224

bb229:                                            ; preds = %bb227, %bb223
  br i1  1, label %__label_002001, label %__label_002001.outer

__label_000020:                                   ; preds = %__label_002001, %bb194
  ret void
}

; CHECK-NOT: =>
; CHECK: [0] entry => <Function Return>
; CHECK-NEXT:  [1] __label_002001.outer => __label_000020
; CHECK-NEXT:      [2] bb197 => bb229
; CHECK-NEXT:            [3] bb224 => bb229

; STAT: 4 region - The # of regions
; STAT: 1 region - The # of simple regions

; BBIT: entry, __label_002001.outer, __label_002001, bb93, __label_000020, bb197, bb229, bb224,
; BBIT: __label_002001.outer, __label_002001, bb93, bb197, bb229, bb224,
; BBIT: bb197, bb224,
; BBIT: bb224,

; RNIT: entry, __label_002001.outer => __label_000020, __label_000020,
; RNIT: __label_002001.outer, __label_002001, bb93, bb197 => bb229, bb229,
; RNIT: bb197, bb224 => bb229,
; RNIT: bb224,
