; RUN: opt < %s -S -indvars -verify-scev
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.8.0"

define void @test1() nounwind uwtable ssp {
entry:
  br i1 undef, label %for.end, label %for.body

for.body:                                         ; preds = %for.body, %entry
  br i1 false, label %for.end, label %for.body

for.end:                                          ; preds = %for.body, %entry
  br i1 undef, label %for.end11, label %for.body3

for.body3:                                        ; preds = %for.end
  unreachable

for.end11:                                        ; preds = %for.end
  br i1 undef, label %while.body, label %while.end

while.body:                                       ; preds = %for.end11
  unreachable

while.end:                                        ; preds = %for.end11
  br i1 undef, label %if.end115, label %for.cond109

for.cond109:                                      ; preds = %while.end
  unreachable

if.end115:                                        ; preds = %while.end
  br i1 undef, label %while.body119.lr.ph.lr.ph, label %for.cond612

while.body119.lr.ph.lr.ph:                        ; preds = %if.end115
  br i1 undef, label %for.cond612, label %if.end123.us

if.end123.us:                                     ; preds = %while.body119.lr.ph.lr.ph
  br label %for.cond132.us

for.cond132.us:                                   ; preds = %for.cond132.us, %if.end123.us
  br i1 undef, label %if.then136.us, label %for.cond132.us

if.then136.us:                                    ; preds = %for.cond132.us
  br i1 undef, label %while.end220, label %while.body211

while.body211:                                    ; preds = %while.body211, %if.then136.us
  br i1 undef, label %while.end220, label %while.body211

while.end220:                                     ; preds = %while.body211, %if.then136.us
  br label %for.cond246.outer

for.cond246.outer:                                ; preds = %for.inc558, %for.cond394.preheader, %if.then274, %for.cond404.preheader, %while.end220
  br label %for.cond246

for.cond246:                                      ; preds = %for.cond372.loopexit, %for.cond246.outer
  br i1 undef, label %for.end562, label %if.end250

if.end250:                                        ; preds = %for.cond246
  br i1 undef, label %if.end256, label %for.end562

if.end256:                                        ; preds = %if.end250
  %cmp272 = icmp eq i32 undef, undef
  br i1 %cmp272, label %if.then274, label %for.cond404.preheader

for.cond404.preheader:                            ; preds = %if.end256
  br i1 undef, label %for.cond246.outer, label %for.body409.lr.ph

for.body409.lr.ph:                                ; preds = %for.cond404.preheader
  br label %for.body409

if.then274:                                       ; preds = %if.end256
  br i1 undef, label %for.cond246.outer, label %if.end309

if.end309:                                        ; preds = %if.then274
  br i1 undef, label %for.cond372.loopexit, label %for.body361

for.body361:                                      ; preds = %for.body361, %if.end309
  br i1 undef, label %for.cond372.loopexit, label %for.body361

for.cond372.loopexit:                             ; preds = %for.body361, %if.end309
  br i1 undef, label %for.cond394.preheader, label %for.cond246

for.cond394.preheader:                            ; preds = %for.cond372.loopexit
  br i1 undef, label %for.cond246.outer, label %for.body397

for.body397:                                      ; preds = %for.cond394.preheader
  unreachable

for.body409:                                      ; preds = %for.inc558, %for.body409.lr.ph
  %k.029 = phi i32 [ 1, %for.body409.lr.ph ], [ %inc559, %for.inc558 ]
  br i1 undef, label %if.then412, label %if.else433

if.then412:                                       ; preds = %for.body409
  br label %if.end440

if.else433:                                       ; preds = %for.body409
  br label %if.end440

if.end440:                                        ; preds = %if.else433, %if.then412
  br i1 undef, label %for.inc558, label %if.end461

if.end461:                                        ; preds = %if.end440
  br i1 undef, label %for.cond528.loopexit, label %for.body517

for.body517:                                      ; preds = %for.body517, %if.end461
  br i1 undef, label %for.cond528.loopexit, label %for.body517

for.cond528.loopexit:                             ; preds = %for.body517, %if.end461
  br label %for.inc558

for.inc558:                                       ; preds = %for.cond528.loopexit, %if.end440
  %inc559 = add nsw i32 %k.029, 1
  %cmp407 = icmp sgt i32 %inc559, undef
  br i1 %cmp407, label %for.cond246.outer, label %for.body409

for.end562:                                       ; preds = %if.end250, %for.cond246
  unreachable

for.cond612:                                      ; preds = %while.body119.lr.ph.lr.ph, %if.end115
  unreachable
}

define void @test2() nounwind uwtable ssp {
entry:
  br i1 undef, label %for.end, label %for.body

for.body:                                         ; preds = %for.body, %entry
  br i1 undef, label %for.end, label %for.body

for.end:                                          ; preds = %for.body, %entry
  br i1 undef, label %for.end11, label %for.body3

for.body3:                                        ; preds = %for.end
  unreachable

for.end11:                                        ; preds = %for.end
  br i1 undef, label %while.body, label %while.end

while.body:                                       ; preds = %for.end11
  unreachable

while.end:                                        ; preds = %for.end11
  br i1 undef, label %if.end115, label %for.cond109

for.cond109:                                      ; preds = %while.end
  unreachable

if.end115:                                        ; preds = %while.end
  br i1 undef, label %while.body119.lr.ph.lr.ph, label %for.cond612

while.body119.lr.ph.lr.ph:                        ; preds = %if.end115
  br i1 undef, label %for.cond612, label %if.end123.us

if.end123.us:                                     ; preds = %while.body119.lr.ph.lr.ph
  br label %for.cond132.us

for.cond132.us:                                   ; preds = %for.cond132.us, %if.end123.us
  br i1 undef, label %if.then136.us, label %for.cond132.us

if.then136.us:                                    ; preds = %for.cond132.us
  br i1 undef, label %while.end220, label %while.body211

while.body211:                                    ; preds = %while.body211, %if.then136.us
  br i1 undef, label %while.end220, label %while.body211

while.end220:                                     ; preds = %while.body211, %if.then136.us
  br label %for.cond246.outer

for.cond246.outer:                                ; preds = %for.inc558, %for.cond394.preheader, %if.then274, %for.cond404.preheader, %while.end220
  br label %for.cond246

for.cond246:                                      ; preds = %for.cond372.loopexit, %for.cond246.outer
  br i1 undef, label %for.end562, label %if.end250

if.end250:                                        ; preds = %for.cond246
  br i1 undef, label %if.end256, label %for.end562

if.end256:                                        ; preds = %if.end250
  %0 = load i32* undef, align 4
  br i1 undef, label %if.then274, label %for.cond404.preheader

for.cond404.preheader:                            ; preds = %if.end256
  %add406 = add i32 0, %0
  br i1 undef, label %for.cond246.outer, label %for.body409.lr.ph

for.body409.lr.ph:                                ; preds = %for.cond404.preheader
  br label %for.body409

if.then274:                                       ; preds = %if.end256
  br i1 undef, label %for.cond246.outer, label %if.end309

if.end309:                                        ; preds = %if.then274
  br i1 undef, label %for.cond372.loopexit, label %for.body361

for.body361:                                      ; preds = %for.body361, %if.end309
  br i1 undef, label %for.cond372.loopexit, label %for.body361

for.cond372.loopexit:                             ; preds = %for.body361, %if.end309
  br i1 undef, label %for.cond394.preheader, label %for.cond246

for.cond394.preheader:                            ; preds = %for.cond372.loopexit
  br i1 undef, label %for.cond246.outer, label %for.body397

for.body397:                                      ; preds = %for.cond394.preheader
  unreachable

for.body409:                                      ; preds = %for.inc558, %for.body409.lr.ph
  %k.029 = phi i32 [ 1, %for.body409.lr.ph ], [ %inc559, %for.inc558 ]
  br i1 undef, label %if.then412, label %if.else433

if.then412:                                       ; preds = %for.body409
  br label %if.end440

if.else433:                                       ; preds = %for.body409
  br label %if.end440

if.end440:                                        ; preds = %if.else433, %if.then412
  br i1 undef, label %for.inc558, label %if.end461

if.end461:                                        ; preds = %if.end440
  br i1 undef, label %for.cond528.loopexit, label %for.body517

for.body517:                                      ; preds = %for.body517, %if.end461
  br i1 undef, label %for.cond528.loopexit, label %for.body517

for.cond528.loopexit:                             ; preds = %for.body517, %if.end461
  br label %for.inc558

for.inc558:                                       ; preds = %for.cond528.loopexit, %if.end440
  %inc559 = add nsw i32 %k.029, 1
  %cmp407 = icmp sgt i32 %inc559, %add406
  br i1 %cmp407, label %for.cond246.outer, label %for.body409

for.end562:                                       ; preds = %if.end250, %for.cond246
  unreachable

for.cond612:                                      ; preds = %while.body119.lr.ph.lr.ph, %if.end115
  unreachable
}

define void @test3() nounwind uwtable ssp {
entry:
  br i1 undef, label %for.end, label %for.body

for.body:                                         ; preds = %for.body, %entry
  br i1 undef, label %for.end, label %for.body

for.end:                                          ; preds = %for.body, %entry
  br i1 undef, label %for.end11, label %for.body3

for.body3:                                        ; preds = %for.end
  unreachable

for.end11:                                        ; preds = %for.end
  br i1 undef, label %while.body, label %while.end

while.body:                                       ; preds = %for.end11
  unreachable

while.end:                                        ; preds = %for.end11
  br i1 undef, label %if.end115, label %for.cond109

for.cond109:                                      ; preds = %while.end
  unreachable

if.end115:                                        ; preds = %while.end
  br i1 undef, label %while.body119.lr.ph.lr.ph, label %for.cond612

while.body119.lr.ph.lr.ph:                        ; preds = %if.end115
  br i1 undef, label %for.cond612, label %if.end123.us

if.end123.us:                                     ; preds = %while.body119.lr.ph.lr.ph
  br label %for.cond132.us

for.cond132.us:                                   ; preds = %for.cond132.us, %if.end123.us
  br i1 undef, label %if.then136.us, label %for.cond132.us

if.then136.us:                                    ; preds = %for.cond132.us
  br i1 undef, label %while.end220, label %while.body211

while.body211:                                    ; preds = %while.body211, %if.then136.us
  br i1 undef, label %while.end220, label %while.body211

while.end220:                                     ; preds = %while.body211, %if.then136.us
  br label %for.cond246.outer

for.cond246.outer:                                ; preds = %for.inc558, %for.cond394.preheader, %if.then274, %for.cond404.preheader, %while.end220
  br label %for.cond246

for.cond246:                                      ; preds = %for.cond372.loopexit, %for.cond246.outer
  br i1 undef, label %for.end562, label %if.end250

if.end250:                                        ; preds = %for.cond246
  br i1 undef, label %if.end256, label %for.end562

if.end256:                                        ; preds = %if.end250
  br i1 undef, label %if.then274, label %for.cond404.preheader

for.cond404.preheader:                            ; preds = %if.end256
  br i1 undef, label %for.cond246.outer, label %for.body409.lr.ph

for.body409.lr.ph:                                ; preds = %for.cond404.preheader
  br label %for.body409

if.then274:                                       ; preds = %if.end256
  br i1 undef, label %for.cond246.outer, label %if.end309

if.end309:                                        ; preds = %if.then274
  br i1 undef, label %for.cond372.loopexit, label %for.body361

for.body361:                                      ; preds = %for.body361, %if.end309
  br i1 undef, label %for.cond372.loopexit, label %for.body361

for.cond372.loopexit:                             ; preds = %for.body361, %if.end309
  br i1 undef, label %for.cond394.preheader, label %for.cond246

for.cond394.preheader:                            ; preds = %for.cond372.loopexit
  br i1 undef, label %for.cond246.outer, label %for.body397

for.body397:                                      ; preds = %for.cond394.preheader
  unreachable

for.body409:                                      ; preds = %for.inc558, %for.body409.lr.ph
  br i1 undef, label %if.then412, label %if.else433

if.then412:                                       ; preds = %for.body409
  br label %if.end440

if.else433:                                       ; preds = %for.body409
  br label %if.end440

if.end440:                                        ; preds = %if.else433, %if.then412
  br i1 undef, label %for.inc558, label %if.end461

if.end461:                                        ; preds = %if.end440
  br i1 undef, label %for.cond528.loopexit, label %for.body517

for.body517:                                      ; preds = %for.body517, %if.end461
  br i1 undef, label %for.cond528.loopexit, label %for.body517

for.cond528.loopexit:                             ; preds = %for.body517, %if.end461
  br label %for.inc558

for.inc558:                                       ; preds = %for.cond528.loopexit, %if.end440
  br i1 undef, label %for.cond246.outer, label %for.body409

for.end562:                                       ; preds = %if.end250, %for.cond246
  unreachable

for.cond612:                                      ; preds = %while.body119.lr.ph.lr.ph, %if.end115
  unreachable
}

define void @test4() nounwind uwtable ssp {
entry:
  br i1 undef, label %if.end8, label %if.else

if.else:                                          ; preds = %entry
  br label %if.end8

if.end8:                                          ; preds = %if.else, %entry
  br i1 undef, label %if.end26, label %if.else22

if.else22:                                        ; preds = %if.end8
  br label %if.end26

if.end26:                                         ; preds = %if.else22, %if.end8
  br i1 undef, label %if.end35, label %if.else31

if.else31:                                        ; preds = %if.end26
  br label %if.end35

if.end35:                                         ; preds = %if.else31, %if.end26
  br i1 undef, label %for.end226, label %for.body.lr.ph

for.body.lr.ph:                                   ; preds = %if.end35
  br label %for.body48

for.body48:                                       ; preds = %for.inc221, %for.body.lr.ph
  br i1 undef, label %for.inc221, label %for.body65.lr.ph

for.body65.lr.ph:                                 ; preds = %for.body48
  %0 = load i32* undef, align 4
  br label %for.body65.us

for.body65.us:                                    ; preds = %for.inc219.us, %for.body65.lr.ph
  %k.09.us = phi i32 [ %inc.us, %for.inc219.us ], [ 1, %for.body65.lr.ph ]
  %idxprom66.us = sext i32 %k.09.us to i64
  br i1 undef, label %for.inc219.us, label %if.end72.us

if.end72.us:                                      ; preds = %for.body65.us
  br i1 undef, label %if.end93.us, label %if.then76.us

if.then76.us:                                     ; preds = %if.end72.us
  br label %if.end93.us

if.end93.us:                                      ; preds = %if.then76.us, %if.end72.us
  br i1 undef, label %if.end110.us, label %for.inc219.us

if.end110.us:                                     ; preds = %if.end93.us
  br i1 undef, label %for.inc219.us, label %for.body142.us

for.body142.us:                                   ; preds = %for.cond139.loopexit.us, %if.end110.us
  br label %for.cond152.us

for.cond152.us:                                   ; preds = %for.cond152.us, %for.body142.us
  br i1 undef, label %for.cond139.loopexit.us, label %for.cond152.us

for.inc219.us:                                    ; preds = %for.cond139.loopexit.us, %if.end110.us, %if.end93.us, %for.body65.us
  %inc.us = add nsw i32 %k.09.us, 1
  %cmp64.us = icmp sgt i32 %inc.us, %0
  br i1 %cmp64.us, label %for.inc221, label %for.body65.us

for.cond139.loopexit.us:                          ; preds = %for.cond152.us
  br i1 undef, label %for.inc219.us, label %for.body142.us

for.inc221:                                       ; preds = %for.inc219.us, %for.body48
  br label %for.body48

for.end226:                                       ; preds = %if.end35
  ret void
}
