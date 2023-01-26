; RUN: opt %s -S -passes='loop-unroll,loop-mssa(licm)' | FileCheck %s

; Check that we can deal with loops where a parent loop gets deleted before it
; is visited by LICM.
define void @test() {
; CHECK-LABEL: define void @test() {
; CHECK-LABEL: entry:
; CHECK-NEXT:    br label %for.body43

; CHECK-LABEL: for.body43:                                       ; preds = %entry
; CHECK-NEXT:    br label %if.else75

; CHECK-LABEL: if.else75:                                        ; preds = %for.body43
; CHECK-NEXT:    br label %for.body467

; CHECK-LABEL: for.body467:                                      ; preds = %for.body467.for.body467_crit_edge, %if.else75
; CHECK-NEXT:    br label %for.body467.for.body467_crit_edge

; CHECK-LABEL: for.body467.for.body467_crit_edge:                ; preds = %for.body467
; CHECK-NEXT:    br i1 false, label %for.end539, label %for.body467

; CHECK-LABEL: for.end539:                                       ; preds = %for.body467.for.body467_crit_edge
; CHECK-NEXT:    ret void
;

entry:
  br label %for.body43

for.body43:                                       ; preds = %for.end539, %entry
  br label %if.else75

if.else75:                                        ; preds = %for.body43
  br label %for.body467

for.body467:                                      ; preds = %for.body467.for.body467_crit_edge, %if.else75
  br label %for.body467.for.body467_crit_edge

for.body467.for.body467_crit_edge:                ; preds = %for.body467
  br i1 false, label %for.end539, label %for.body467

for.end539:                                       ; preds = %for.body467
  br i1 undef, label %for.body43, label %for.end547

for.end547:                                       ; preds = %for.body43
  ret void
}
