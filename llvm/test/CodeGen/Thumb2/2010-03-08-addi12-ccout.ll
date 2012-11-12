; RUN: llc < %s -mtriple=thumbv7-apple-darwin

@.str41196 = external constant [2 x i8], align 4  ; <[2 x i8]*> [#uses=1]

declare void @syStopraw(i32) nounwind

declare i32 @SyFopen(i8*, i8*) nounwind

declare i8* @SyFgets(i8*, i32) nounwind

define void @SyHelp(i8* nocapture %topic, i32 %fin) nounwind {
entry:
  %line = alloca [256 x i8], align 4              ; <[256 x i8]*> [#uses=1]
  %secname = alloca [1024 x i8], align 4          ; <[1024 x i8]*> [#uses=0]
  %last = alloca [256 x i8], align 4              ; <[256 x i8]*> [#uses=1]
  %last2 = alloca [256 x i8], align 4             ; <[256 x i8]*> [#uses=1]
  br i1 undef, label %bb, label %bb2

bb:                                               ; preds = %entry
  br i1 undef, label %bb2, label %bb3

bb2:                                              ; preds = %bb, %entry
  br label %bb3

bb3:                                              ; preds = %bb2, %bb
  %storemerge = phi i32 [ 0, %bb2 ], [ 1, %bb ]   ; <i32> [#uses=1]
  br i1 undef, label %bb19, label %bb20

bb19:                                             ; preds = %bb3
  br label %bb20

bb20:                                             ; preds = %bb19, %bb3
  br i1 undef, label %bb25, label %bb26

bb25:                                             ; preds = %bb20
  br label %bb26

bb26:                                             ; preds = %bb25, %bb20
  %offset.2 = phi i32 [ -2, %bb25 ], [ 0, %bb20 ] ; <i32> [#uses=1]
  br i1 undef, label %bb.nph508, label %bb49

bb.nph508:                                        ; preds = %bb26
  unreachable

bb49:                                             ; preds = %bb26
  br i1 undef, label %bb51, label %bb50

bb50:                                             ; preds = %bb49
  br i1 undef, label %bb51, label %bb104

bb51:                                             ; preds = %bb50, %bb49
  unreachable

bb104:                                            ; preds = %bb50
  br i1 undef, label %bb106, label %bb105

bb105:                                            ; preds = %bb104
  br i1 undef, label %bb106, label %bb161

bb106:                                            ; preds = %bb105, %bb104
  unreachable

bb161:                                            ; preds = %bb105
  br i1 false, label %bb163, label %bb162

bb162:                                            ; preds = %bb161
  br i1 undef, label %bb163, label %bb224

bb163:                                            ; preds = %bb162, %bb161
  unreachable

bb224:                                            ; preds = %bb162
  %0 = call  i32 @SyFopen(i8* undef, i8* getelementptr inbounds ([2 x i8]* @.str41196, i32 0, i32 0)) nounwind ; <i32> [#uses=2]
  br i1 false, label %bb297, label %bb300

bb297:                                            ; preds = %bb224
  unreachable

bb300:                                            ; preds = %bb224
  %1 = icmp eq i32 %offset.2, -1                  ; <i1> [#uses=1]
  br label %bb440

bb307:                                            ; preds = %isdigit1498.exit67
  br label %bb308

bb308:                                            ; preds = %bb440, %bb307
  br i1 undef, label %bb309, label %isdigit1498.exit67

isdigit1498.exit67:                               ; preds = %bb308
  br i1 undef, label %bb309, label %bb307

bb309:                                            ; preds = %isdigit1498.exit67, %bb308
  br i1 undef, label %bb310, label %bb313

bb310:                                            ; preds = %bb309
  br label %bb313

bb313:                                            ; preds = %bb310, %bb309
  br i1 false, label %bb318, label %bb317

bb317:                                            ; preds = %bb313
  %2 = icmp sgt i8 undef, -1                      ; <i1> [#uses=1]
  br i1 %2, label %bb.i.i73, label %bb1.i.i74

bb.i.i73:                                         ; preds = %bb317
  br i1 false, label %bb318, label %bb329.outer

bb1.i.i74:                                        ; preds = %bb317
  unreachable

bb318:                                            ; preds = %bb.i.i73, %bb313
  ret void

bb329.outer:                                      ; preds = %bb.i.i73
  br i1 undef, label %bb333, label %bb329.us.us

bb329.us.us:                                      ; preds = %bb329.us.us, %bb329.outer
  br i1 undef, label %bb333, label %bb329.us.us

bb333:                                            ; preds = %bb329.us.us, %bb329.outer
  %match.0.lcssa = phi i32 [ undef, %bb329.us.us ], [ 2, %bb329.outer ] ; <i32> [#uses=2]
  br i1 undef, label %bb335, label %bb388

bb335:                                            ; preds = %bb333
  %3 = and i1 undef, %1                           ; <i1> [#uses=1]
  br i1 %3, label %bb339, label %bb348

bb339:                                            ; preds = %bb335
  br i1 false, label %bb340, label %bb345

bb340:                                            ; preds = %bb339
  br i1 undef, label %return, label %bb341

bb341:                                            ; preds = %bb340
  ret void

bb345:                                            ; preds = %bb345, %bb339
  %4 = phi i8 [ %5, %bb345 ], [ undef, %bb339 ]   ; <i8> [#uses=0]
  %indvar670 = phi i32 [ %tmp673, %bb345 ], [ 0, %bb339 ] ; <i32> [#uses=1]
  %tmp673 = add i32 %indvar670, 1                 ; <i32> [#uses=2]
  %scevgep674 = getelementptr [256 x i8]* %last, i32 0, i32 %tmp673 ; <i8*> [#uses=1]
  %5 = load i8* %scevgep674, align 1              ; <i8> [#uses=1]
  br i1 undef, label %bb347, label %bb345

bb347:                                            ; preds = %bb345
  br label %bb348

bb348:                                            ; preds = %bb347, %bb335
  br i1 false, label %bb352, label %bb356

bb352:                                            ; preds = %bb348
  unreachable

bb356:                                            ; preds = %bb348
  br i1 undef, label %bb360, label %bb369

bb360:                                            ; preds = %bb356
  br i1 false, label %bb361, label %bb366

bb361:                                            ; preds = %bb360
  br i1 undef, label %return, label %bb362

bb362:                                            ; preds = %bb361
  ret void

bb366:                                            ; preds = %bb366, %bb360
  %indvar662 = phi i32 [ %tmp665, %bb366 ], [ 0, %bb360 ] ; <i32> [#uses=1]
  %tmp665 = add i32 %indvar662, 1                 ; <i32> [#uses=2]
  %scevgep666 = getelementptr [256 x i8]* %last2, i32 0, i32 %tmp665 ; <i8*> [#uses=1]
  %6 = load i8* %scevgep666, align 1              ; <i8> [#uses=0]
  br i1 false, label %bb368, label %bb366

bb368:                                            ; preds = %bb366
  br label %bb369

bb369:                                            ; preds = %bb368, %bb356
  br i1 undef, label %bb373, label %bb388

bb373:                                            ; preds = %bb383, %bb369
  %7 = call  i8* @SyFgets(i8* undef, i32 %0) nounwind ; <i8*> [#uses=1]
  %8 = icmp eq i8* %7, null                       ; <i1> [#uses=1]
  br i1 %8, label %bb375, label %bb383

bb375:                                            ; preds = %bb373
  %9 = icmp eq i32 %storemerge, 0                 ; <i1> [#uses=1]
  br i1 %9, label %return, label %bb376

bb376:                                            ; preds = %bb375
  ret void

bb383:                                            ; preds = %bb373
  %10 = load i8* undef, align 1                   ; <i8> [#uses=1]
  %cond1 = icmp eq i8 %10, 46                     ; <i1> [#uses=1]
  br i1 %cond1, label %bb373, label %bb388

bb388:                                            ; preds = %bb383, %bb369, %bb333
  %match.1140 = phi i32 [ %match.0.lcssa, %bb369 ], [ 0, %bb333 ], [ %match.0.lcssa, %bb383 ] ; <i32> [#uses=1]
  br label %bb391

bb390:                                            ; preds = %isdigit1498.exit83, %bb392
  %indvar.next725 = add i32 %indvar724, 1         ; <i32> [#uses=1]
  br label %bb391

bb391:                                            ; preds = %bb390, %bb388
  %indvar724 = phi i32 [ %indvar.next725, %bb390 ], [ 0, %bb388 ] ; <i32> [#uses=2]
  %11 = load i8* undef, align 1                   ; <i8> [#uses=0]
  br i1 false, label %bb395, label %bb392

bb392:                                            ; preds = %bb391
  br i1 undef, label %bb390, label %isdigit1498.exit83

isdigit1498.exit83:                               ; preds = %bb392
  br i1 undef, label %bb390, label %bb395

bb394:                                            ; preds = %isdigit1498.exit87
  br label %bb395

bb395:                                            ; preds = %bb394, %isdigit1498.exit83, %bb391
  %storemerge14.sum = add i32 %indvar724, undef   ; <i32> [#uses=1]
  %p.26 = getelementptr [256 x i8]* %line, i32 0, i32 %storemerge14.sum ; <i8*> [#uses=1]
  br i1 undef, label %bb400, label %isdigit1498.exit87

isdigit1498.exit87:                               ; preds = %bb395
  br i1 false, label %bb400, label %bb394

bb400:                                            ; preds = %isdigit1498.exit87, %bb395
  br i1 undef, label %bb402, label %bb403

bb402:                                            ; preds = %bb400
  %12 = getelementptr inbounds i8* %p.26, i32 undef ; <i8*> [#uses=1]
  br label %bb403

bb403:                                            ; preds = %bb402, %bb400
  %p.29 = phi i8* [ %12, %bb402 ], [ undef, %bb400 ] ; <i8*> [#uses=0]
  br i1 undef, label %bb405, label %bb404

bb404:                                            ; preds = %bb403
  br i1 undef, label %bb405, label %bb407

bb405:                                            ; preds = %bb404, %bb403
  br i1 undef, label %return, label %bb406

bb406:                                            ; preds = %bb405
  call  void @syStopraw(i32 %fin) nounwind
  ret void

bb407:                                            ; preds = %bb404
  %cond = icmp eq i32 %match.1140, 2              ; <i1> [#uses=1]
  br i1 %cond, label %bb408, label %bb428

bb408:                                            ; preds = %bb407
  unreachable

bb428:                                            ; preds = %bb407
  br label %bb440

bb440:                                            ; preds = %bb428, %bb300
  %13 = call  i8* @SyFgets(i8* undef, i32 %0) nounwind ; <i8*> [#uses=0]
  br i1 false, label %bb442, label %bb308

bb442:                                            ; preds = %bb440
  unreachable

return:                                           ; preds = %bb405, %bb375, %bb361, %bb340
  ret void
}
