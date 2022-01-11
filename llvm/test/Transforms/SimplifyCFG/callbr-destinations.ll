; RUN: opt < %s -simplifycfg -simplifycfg-require-and-preserve-domtree=1 -disable-output
;
; Test that SimplifyCFG does not cause CallBr instructions to have duplicate
; destinations, which will cause the verifier to assert.

define void @fun0() {
entry:
  callbr void asm sideeffect "", "i"(i8* blockaddress(@fun0, %bb1))
          to label %bb2 [label %bb1]

bb1:                                              ; preds = %bb
  ret void

bb2:                                             ; preds = %bb
  ret void
}

define void @fun1() {
entry:
  callbr void asm sideeffect "", "i"(i8* blockaddress(@fun1, %bb1))
          to label %bb2 [label %bb1]

bb2:                                             ; preds = %bb
  ret void

bb1:                                              ; preds = %bb
  ret void
}
