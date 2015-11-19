;RUN: llc < %s -mtriple=aarch64--linux-android -regalloc=greedy -enable-deferred-spilling=true -mcpu=cortex-a57 | FileCheck %s --check-prefix=CHECK --check-prefix=DEFERRED
;RUN: llc < %s -mtriple=aarch64--linux-android -regalloc=greedy -enable-deferred-spilling=false -mcpu=cortex-a57 | FileCheck %s --check-prefix=CHECK --check-prefix=REGULAR

; Check that we do not end up with useless spill code.
;
; Move to the basic block we are interested in.
;
; CHECK: // %if.then.120
;
; REGULAR: str w21, [sp, #[[OFFSET:[0-9]+]]] // 4-byte Folded Spill
; Check that w21 wouldn't need to be spilled since it is never reused.
; REGULAR-NOT: {{[wx]}}21{{,?}}
;
; Check that w22 is used to carry a value through the call.
; DEFERRED-NOT: str {{[wx]}}22,
; DEFERRED: mov {{[wx]}}22,
; DEFERRED-NOT: str {{[wx]}}22,
;
; CHECK:        bl      fprintf
;
; DEFERRED-NOT: ldr {{[wx]}}22,
; DEFERRED: mov {{[wx][0-9]+}}, {{[wx]}}22
; DEFERRED-NOT: ldr {{[wx]}}22,
;
; REGULAR-NOT: {{[wx]}}21{{,?}}
; REGULAR: ldr w21, [sp, #[[OFFSET]]] // 4-byte Folded Reload
;
; End of the basic block we are interested in.
; CHECK:        b
; CHECK: {{[^:]+}}: // %sw.bb.123

%struct.__sFILE = type { i8*, i32, i32, i32, i32, %struct.__sbuf, i32, i8*, i32 (i8*)*, i32 (i8*, i8*, i32)*, i64 (i8*, i64, i32)*, i32 (i8*, i8*, i32)*, %struct.__sbuf, i8*, i32, [3 x i8], [1 x i8], %struct.__sbuf, i32, i64 }
%struct.__sbuf = type { i8*, i64 }
%struct.DState = type { %struct.bz_stream*, i32, i8, i32, i8, i32, i32, i32, i32, i32, i8, i32, i32, i32, i32, i32, [256 x i32], i32, [257 x i32], [257 x i32], i32*, i16*, i8*, i32, i32, i32, i32, i32, [256 x i8], [16 x i8], [256 x i8], [4096 x i8], [16 x i32], [18002 x i8], [18002 x i8], [6 x [258 x i8]], [6 x [258 x i32]], [6 x [258 x i32]], [6 x [258 x i32]], [6 x i32], i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32*, i32*, i32* }
%struct.bz_stream = type { i8*, i32, i32, i32, i8*, i32, i32, i32, i8*, i8* (i8*, i32, i32)*, void (i8*, i8*)*, i8* }

@__sF = external global [0 x %struct.__sFILE], align 8
@.str = private unnamed_addr constant [20 x i8] c"\0A    [%d: stuff+mf \00", align 1

declare i32 @fprintf(%struct.__sFILE* nocapture, i8* nocapture readonly, ...)

declare void @bar(i32)

declare void @llvm.memset.p0i8.i64(i8* nocapture, i8, i64, i1)

define i32 @foo(%struct.DState* %s) {
entry:
  %state = getelementptr inbounds %struct.DState, %struct.DState* %s, i64 0, i32 1
  %tmp = load i32, i32* %state, align 4
  %cmp = icmp eq i32 %tmp, 10
  %save_i = getelementptr inbounds %struct.DState, %struct.DState* %s, i64 0, i32 40
  br i1 %cmp, label %if.end.thread, label %if.end

if.end.thread:                                    ; preds = %entry
  %save_j = getelementptr inbounds %struct.DState, %struct.DState* %s, i64 0, i32 41
  %save_t = getelementptr inbounds %struct.DState, %struct.DState* %s, i64 0, i32 42
  %save_alphaSize = getelementptr inbounds %struct.DState, %struct.DState* %s, i64 0, i32 43
  %save_nGroups = getelementptr inbounds %struct.DState, %struct.DState* %s, i64 0, i32 44
  %save_nSelectors = getelementptr inbounds %struct.DState, %struct.DState* %s, i64 0, i32 45
  %save_EOB = getelementptr inbounds %struct.DState, %struct.DState* %s, i64 0, i32 46
  %save_groupNo = getelementptr inbounds %struct.DState, %struct.DState* %s, i64 0, i32 47
  %save_groupPos = getelementptr inbounds %struct.DState, %struct.DState* %s, i64 0, i32 48
  %save_nextSym = getelementptr inbounds %struct.DState, %struct.DState* %s, i64 0, i32 49
  %save_nblockMAX = getelementptr inbounds %struct.DState, %struct.DState* %s, i64 0, i32 50
  %save_nblock = getelementptr inbounds %struct.DState, %struct.DState* %s, i64 0, i32 51
  %save_es = getelementptr inbounds %struct.DState, %struct.DState* %s, i64 0, i32 52
  %save_N = getelementptr inbounds %struct.DState, %struct.DState* %s, i64 0, i32 53
  %save_curr = getelementptr inbounds %struct.DState, %struct.DState* %s, i64 0, i32 54
  %save_zt = getelementptr inbounds %struct.DState, %struct.DState* %s, i64 0, i32 55
  %save_zn = getelementptr inbounds %struct.DState, %struct.DState* %s, i64 0, i32 56
  %save_zvec = getelementptr inbounds %struct.DState, %struct.DState* %s, i64 0, i32 57
  %save_zj = getelementptr inbounds %struct.DState, %struct.DState* %s, i64 0, i32 58
  %tmp1 = bitcast i32* %save_i to i8*
  call void @llvm.memset.p0i8.i64(i8* %tmp1, i8 0, i64 108, i1 false)
  br label %sw.default

if.end:                                           ; preds = %entry
  %.pre = load i32, i32* %save_i, align 4
  %save_j3.phi.trans.insert = getelementptr inbounds %struct.DState, %struct.DState* %s, i64 0, i32 41
  %.pre406 = load i32, i32* %save_j3.phi.trans.insert, align 4
  %save_t4.phi.trans.insert = getelementptr inbounds %struct.DState, %struct.DState* %s, i64 0, i32 42
  %.pre407 = load i32, i32* %save_t4.phi.trans.insert, align 4
  %save_alphaSize5.phi.trans.insert = getelementptr inbounds %struct.DState, %struct.DState* %s, i64 0, i32 43
  %.pre408 = load i32, i32* %save_alphaSize5.phi.trans.insert, align 4
  %save_nGroups6.phi.trans.insert = getelementptr inbounds %struct.DState, %struct.DState* %s, i64 0, i32 44
  %.pre409 = load i32, i32* %save_nGroups6.phi.trans.insert, align 4
  %save_nSelectors7.phi.trans.insert = getelementptr inbounds %struct.DState, %struct.DState* %s, i64 0, i32 45
  %.pre410 = load i32, i32* %save_nSelectors7.phi.trans.insert, align 4
  %save_EOB8.phi.trans.insert = getelementptr inbounds %struct.DState, %struct.DState* %s, i64 0, i32 46
  %.pre411 = load i32, i32* %save_EOB8.phi.trans.insert, align 4
  %save_groupNo9.phi.trans.insert = getelementptr inbounds %struct.DState, %struct.DState* %s, i64 0, i32 47
  %.pre412 = load i32, i32* %save_groupNo9.phi.trans.insert, align 4
  %save_groupPos10.phi.trans.insert = getelementptr inbounds %struct.DState, %struct.DState* %s, i64 0, i32 48
  %.pre413 = load i32, i32* %save_groupPos10.phi.trans.insert, align 4
  %save_nextSym11.phi.trans.insert = getelementptr inbounds %struct.DState, %struct.DState* %s, i64 0, i32 49
  %.pre414 = load i32, i32* %save_nextSym11.phi.trans.insert, align 4
  %save_nblockMAX12.phi.trans.insert = getelementptr inbounds %struct.DState, %struct.DState* %s, i64 0, i32 50
  %.pre415 = load i32, i32* %save_nblockMAX12.phi.trans.insert, align 4
  %save_nblock13.phi.trans.insert = getelementptr inbounds %struct.DState, %struct.DState* %s, i64 0, i32 51
  %.pre416 = load i32, i32* %save_nblock13.phi.trans.insert, align 4
  %save_es14.phi.trans.insert = getelementptr inbounds %struct.DState, %struct.DState* %s, i64 0, i32 52
  %.pre417 = load i32, i32* %save_es14.phi.trans.insert, align 4
  %save_N15.phi.trans.insert = getelementptr inbounds %struct.DState, %struct.DState* %s, i64 0, i32 53
  %.pre418 = load i32, i32* %save_N15.phi.trans.insert, align 4
  %save_curr16.phi.trans.insert = getelementptr inbounds %struct.DState, %struct.DState* %s, i64 0, i32 54
  %.pre419 = load i32, i32* %save_curr16.phi.trans.insert, align 4
  %save_zt17.phi.trans.insert = getelementptr inbounds %struct.DState, %struct.DState* %s, i64 0, i32 55
  %.pre420 = load i32, i32* %save_zt17.phi.trans.insert, align 4
  %save_zn18.phi.trans.insert = getelementptr inbounds %struct.DState, %struct.DState* %s, i64 0, i32 56
  %.pre421 = load i32, i32* %save_zn18.phi.trans.insert, align 4
  %save_zvec19.phi.trans.insert = getelementptr inbounds %struct.DState, %struct.DState* %s, i64 0, i32 57
  %.pre422 = load i32, i32* %save_zvec19.phi.trans.insert, align 4
  %save_zj20.phi.trans.insert = getelementptr inbounds %struct.DState, %struct.DState* %s, i64 0, i32 58
  %.pre423 = load i32, i32* %save_zj20.phi.trans.insert, align 4
  switch i32 %tmp, label %sw.default [
    i32 13, label %sw.bb
    i32 14, label %if.end.sw.bb.65_crit_edge
    i32 25, label %if.end.sw.bb.123_crit_edge
  ]

if.end.sw.bb.123_crit_edge:                       ; preds = %if.end
  %.pre433 = getelementptr inbounds %struct.DState, %struct.DState* %s, i64 0, i32 8
  br label %sw.bb.123

if.end.sw.bb.65_crit_edge:                        ; preds = %if.end
  %bsLive69.phi.trans.insert = getelementptr inbounds %struct.DState, %struct.DState* %s, i64 0, i32 8
  %.pre426 = load i32, i32* %bsLive69.phi.trans.insert, align 4
  br label %sw.bb.65

sw.bb:                                            ; preds = %if.end
  %sunkaddr = ptrtoint %struct.DState* %s to i64
  %sunkaddr485 = add i64 %sunkaddr, 8
  %sunkaddr486 = inttoptr i64 %sunkaddr485 to i32*
  store i32 13, i32* %sunkaddr486, align 4
  %bsLive = getelementptr inbounds %struct.DState, %struct.DState* %s, i64 0, i32 8
  %tmp2 = load i32, i32* %bsLive, align 4
  %cmp28.400 = icmp sgt i32 %tmp2, 7
  br i1 %cmp28.400, label %sw.bb.if.then.29_crit_edge, label %if.end.33.lr.ph

sw.bb.if.then.29_crit_edge:                       ; preds = %sw.bb
  %sunkaddr487 = ptrtoint %struct.DState* %s to i64
  %sunkaddr488 = add i64 %sunkaddr487, 32
  %sunkaddr489 = inttoptr i64 %sunkaddr488 to i32*
  %.pre425 = load i32, i32* %sunkaddr489, align 4
  br label %if.then.29

if.end.33.lr.ph:                                  ; preds = %sw.bb
  %tmp3 = bitcast %struct.DState* %s to %struct.bz_stream**
  %.pre424 = load %struct.bz_stream*, %struct.bz_stream** %tmp3, align 8
  %avail_in.phi.trans.insert = getelementptr inbounds %struct.bz_stream, %struct.bz_stream* %.pre424, i64 0, i32 1
  %.pre430 = load i32, i32* %avail_in.phi.trans.insert, align 4
  %tmp4 = add i32 %.pre430, -1
  br label %if.end.33

if.then.29:                                       ; preds = %while.body.backedge, %sw.bb.if.then.29_crit_edge
  %tmp5 = phi i32 [ %.pre425, %sw.bb.if.then.29_crit_edge ], [ %or, %while.body.backedge ]
  %.lcssa393 = phi i32 [ %tmp2, %sw.bb.if.then.29_crit_edge ], [ %add, %while.body.backedge ]
  %sub = add nsw i32 %.lcssa393, -8
  %shr = lshr i32 %tmp5, %sub
  %and = and i32 %shr, 255
  %sunkaddr491 = ptrtoint %struct.DState* %s to i64
  %sunkaddr492 = add i64 %sunkaddr491, 36
  %sunkaddr493 = inttoptr i64 %sunkaddr492 to i32*
  store i32 %sub, i32* %sunkaddr493, align 4
  %blockSize100k = getelementptr inbounds %struct.DState, %struct.DState* %s, i64 0, i32 9
  store i32 %and, i32* %blockSize100k, align 4
  %and.off = add nsw i32 %and, -49
  %tmp6 = icmp ugt i32 %and.off, 8
  br i1 %tmp6, label %save_state_and_return, label %if.end.62

if.end.33:                                        ; preds = %while.body.backedge, %if.end.33.lr.ph
  %lsr.iv482 = phi i32 [ %tmp4, %if.end.33.lr.ph ], [ %lsr.iv.next483, %while.body.backedge ]
  %tmp7 = phi i32 [ %tmp2, %if.end.33.lr.ph ], [ %add, %while.body.backedge ]
  %cmp35 = icmp eq i32 %lsr.iv482, -1
  br i1 %cmp35, label %save_state_and_return, label %if.end.37

if.end.37:                                        ; preds = %if.end.33
  %tmp8 = bitcast %struct.bz_stream* %.pre424 to i8**
  %sunkaddr494 = ptrtoint %struct.DState* %s to i64
  %sunkaddr495 = add i64 %sunkaddr494, 32
  %sunkaddr496 = inttoptr i64 %sunkaddr495 to i32*
  %tmp9 = load i32, i32* %sunkaddr496, align 4
  %shl = shl i32 %tmp9, 8
  %tmp10 = load i8*, i8** %tmp8, align 8
  %tmp11 = load i8, i8* %tmp10, align 1
  %conv = zext i8 %tmp11 to i32
  %or = or i32 %conv, %shl
  store i32 %or, i32* %sunkaddr496, align 4
  %add = add nsw i32 %tmp7, 8
  %sunkaddr497 = ptrtoint %struct.DState* %s to i64
  %sunkaddr498 = add i64 %sunkaddr497, 36
  %sunkaddr499 = inttoptr i64 %sunkaddr498 to i32*
  store i32 %add, i32* %sunkaddr499, align 4
  %incdec.ptr = getelementptr inbounds i8, i8* %tmp10, i64 1
  store i8* %incdec.ptr, i8** %tmp8, align 8
  %sunkaddr500 = ptrtoint %struct.bz_stream* %.pre424 to i64
  %sunkaddr501 = add i64 %sunkaddr500, 8
  %sunkaddr502 = inttoptr i64 %sunkaddr501 to i32*
  store i32 %lsr.iv482, i32* %sunkaddr502, align 4
  %sunkaddr503 = ptrtoint %struct.bz_stream* %.pre424 to i64
  %sunkaddr504 = add i64 %sunkaddr503, 12
  %sunkaddr505 = inttoptr i64 %sunkaddr504 to i32*
  %tmp12 = load i32, i32* %sunkaddr505, align 4
  %inc = add i32 %tmp12, 1
  store i32 %inc, i32* %sunkaddr505, align 4
  %cmp49 = icmp eq i32 %inc, 0
  br i1 %cmp49, label %if.then.51, label %while.body.backedge

if.then.51:                                       ; preds = %if.end.37
  %sunkaddr506 = ptrtoint %struct.bz_stream* %.pre424 to i64
  %sunkaddr507 = add i64 %sunkaddr506, 16
  %sunkaddr508 = inttoptr i64 %sunkaddr507 to i32*
  %tmp13 = load i32, i32* %sunkaddr508, align 4
  %inc53 = add i32 %tmp13, 1
  store i32 %inc53, i32* %sunkaddr508, align 4
  br label %while.body.backedge

while.body.backedge:                              ; preds = %if.then.51, %if.end.37
  %lsr.iv.next483 = add i32 %lsr.iv482, -1
  %cmp28 = icmp sgt i32 %add, 7
  br i1 %cmp28, label %if.then.29, label %if.end.33

if.end.62:                                        ; preds = %if.then.29
  %sub64 = add nsw i32 %and, -48
  %sunkaddr509 = ptrtoint %struct.DState* %s to i64
  %sunkaddr510 = add i64 %sunkaddr509, 40
  %sunkaddr511 = inttoptr i64 %sunkaddr510 to i32*
  store i32 %sub64, i32* %sunkaddr511, align 4
  br label %sw.bb.65

sw.bb.65:                                         ; preds = %if.end.62, %if.end.sw.bb.65_crit_edge
  %bsLive69.pre-phi = phi i32* [ %bsLive69.phi.trans.insert, %if.end.sw.bb.65_crit_edge ], [ %bsLive, %if.end.62 ]
  %tmp14 = phi i32 [ %.pre426, %if.end.sw.bb.65_crit_edge ], [ %sub, %if.end.62 ]
  %sunkaddr512 = ptrtoint %struct.DState* %s to i64
  %sunkaddr513 = add i64 %sunkaddr512, 8
  %sunkaddr514 = inttoptr i64 %sunkaddr513 to i32*
  store i32 14, i32* %sunkaddr514, align 4
  %cmp70.397 = icmp sgt i32 %tmp14, 7
  br i1 %cmp70.397, label %if.then.72, label %if.end.82.lr.ph

if.end.82.lr.ph:                                  ; preds = %sw.bb.65
  %tmp15 = bitcast %struct.DState* %s to %struct.bz_stream**
  %.pre427 = load %struct.bz_stream*, %struct.bz_stream** %tmp15, align 8
  %avail_in84.phi.trans.insert = getelementptr inbounds %struct.bz_stream, %struct.bz_stream* %.pre427, i64 0, i32 1
  %.pre431 = load i32, i32* %avail_in84.phi.trans.insert, align 4
  %tmp16 = add i32 %.pre431, -1
  br label %if.end.82

if.then.72:                                       ; preds = %while.body.68.backedge, %sw.bb.65
  %.lcssa390 = phi i32 [ %tmp14, %sw.bb.65 ], [ %add97, %while.body.68.backedge ]
  %sub76 = add nsw i32 %.lcssa390, -8
  %sunkaddr516 = ptrtoint %struct.DState* %s to i64
  %sunkaddr517 = add i64 %sunkaddr516, 36
  %sunkaddr518 = inttoptr i64 %sunkaddr517 to i32*
  store i32 %sub76, i32* %sunkaddr518, align 4
  %currBlockNo = getelementptr inbounds %struct.DState, %struct.DState* %s, i64 0, i32 11
  %tmp17 = load i32, i32* %currBlockNo, align 4
  %inc117 = add nsw i32 %tmp17, 1
  store i32 %inc117, i32* %currBlockNo, align 4
  %verbosity = getelementptr inbounds %struct.DState, %struct.DState* %s, i64 0, i32 12
  %tmp18 = load i32, i32* %verbosity, align 4
  %cmp118 = icmp sgt i32 %tmp18, 1
  br i1 %cmp118, label %if.then.120, label %sw.bb.123, !prof !0

if.end.82:                                        ; preds = %while.body.68.backedge, %if.end.82.lr.ph
  %lsr.iv480 = phi i32 [ %tmp16, %if.end.82.lr.ph ], [ %lsr.iv.next481, %while.body.68.backedge ]
  %tmp19 = phi i32 [ %tmp14, %if.end.82.lr.ph ], [ %add97, %while.body.68.backedge ]
  %cmp85 = icmp eq i32 %lsr.iv480, -1
  br i1 %cmp85, label %save_state_and_return, label %if.end.88

if.end.88:                                        ; preds = %if.end.82
  %tmp20 = bitcast %struct.bz_stream* %.pre427 to i8**
  %sunkaddr519 = ptrtoint %struct.DState* %s to i64
  %sunkaddr520 = add i64 %sunkaddr519, 32
  %sunkaddr521 = inttoptr i64 %sunkaddr520 to i32*
  %tmp21 = load i32, i32* %sunkaddr521, align 4
  %shl90 = shl i32 %tmp21, 8
  %tmp22 = load i8*, i8** %tmp20, align 8
  %tmp23 = load i8, i8* %tmp22, align 1
  %conv93 = zext i8 %tmp23 to i32
  %or94 = or i32 %conv93, %shl90
  store i32 %or94, i32* %sunkaddr521, align 4
  %add97 = add nsw i32 %tmp19, 8
  %sunkaddr522 = ptrtoint %struct.DState* %s to i64
  %sunkaddr523 = add i64 %sunkaddr522, 36
  %sunkaddr524 = inttoptr i64 %sunkaddr523 to i32*
  store i32 %add97, i32* %sunkaddr524, align 4
  %incdec.ptr100 = getelementptr inbounds i8, i8* %tmp22, i64 1
  store i8* %incdec.ptr100, i8** %tmp20, align 8
  %sunkaddr525 = ptrtoint %struct.bz_stream* %.pre427 to i64
  %sunkaddr526 = add i64 %sunkaddr525, 8
  %sunkaddr527 = inttoptr i64 %sunkaddr526 to i32*
  store i32 %lsr.iv480, i32* %sunkaddr527, align 4
  %sunkaddr528 = ptrtoint %struct.bz_stream* %.pre427 to i64
  %sunkaddr529 = add i64 %sunkaddr528, 12
  %sunkaddr530 = inttoptr i64 %sunkaddr529 to i32*
  %tmp24 = load i32, i32* %sunkaddr530, align 4
  %inc106 = add i32 %tmp24, 1
  store i32 %inc106, i32* %sunkaddr530, align 4
  %cmp109 = icmp eq i32 %inc106, 0
  br i1 %cmp109, label %if.then.111, label %while.body.68.backedge

if.then.111:                                      ; preds = %if.end.88
  %sunkaddr531 = ptrtoint %struct.bz_stream* %.pre427 to i64
  %sunkaddr532 = add i64 %sunkaddr531, 16
  %sunkaddr533 = inttoptr i64 %sunkaddr532 to i32*
  %tmp25 = load i32, i32* %sunkaddr533, align 4
  %inc114 = add i32 %tmp25, 1
  store i32 %inc114, i32* %sunkaddr533, align 4
  br label %while.body.68.backedge

while.body.68.backedge:                           ; preds = %if.then.111, %if.end.88
  %lsr.iv.next481 = add i32 %lsr.iv480, -1
  %cmp70 = icmp sgt i32 %add97, 7
  br i1 %cmp70, label %if.then.72, label %if.end.82

if.then.120:                                      ; preds = %if.then.72
  %call = tail call i32 (%struct.__sFILE*, i8*, ...) @fprintf(%struct.__sFILE* getelementptr inbounds ([0 x %struct.__sFILE], [0 x %struct.__sFILE]* @__sF, i64 0, i64 2), i8* getelementptr inbounds ([20 x i8], [20 x i8]* @.str, i64 0, i64 0), i32 %inc117)
  br label %sw.bb.123

sw.bb.123:                                        ; preds = %if.then.120, %if.then.72, %if.end.sw.bb.123_crit_edge
  %bsLive127.pre-phi = phi i32* [ %.pre433, %if.end.sw.bb.123_crit_edge ], [ %bsLive69.pre-phi, %if.then.72 ], [ %bsLive69.pre-phi, %if.then.120 ]
  %sunkaddr534 = ptrtoint %struct.DState* %s to i64
  %sunkaddr535 = add i64 %sunkaddr534, 8
  %sunkaddr536 = inttoptr i64 %sunkaddr535 to i32*
  store i32 25, i32* %sunkaddr536, align 4
  %tmp26 = load i32, i32* %bsLive127.pre-phi, align 4
  %cmp128.395 = icmp sgt i32 %tmp26, 7
  br i1 %cmp128.395, label %sw.bb.123.if.then.130_crit_edge, label %if.end.140.lr.ph

sw.bb.123.if.then.130_crit_edge:                  ; preds = %sw.bb.123
  %sunkaddr537 = ptrtoint %struct.DState* %s to i64
  %sunkaddr538 = add i64 %sunkaddr537, 32
  %sunkaddr539 = inttoptr i64 %sunkaddr538 to i32*
  %.pre429 = load i32, i32* %sunkaddr539, align 4
  br label %if.then.130

if.end.140.lr.ph:                                 ; preds = %sw.bb.123
  %tmp27 = bitcast %struct.DState* %s to %struct.bz_stream**
  %.pre428 = load %struct.bz_stream*, %struct.bz_stream** %tmp27, align 8
  %avail_in142.phi.trans.insert = getelementptr inbounds %struct.bz_stream, %struct.bz_stream* %.pre428, i64 0, i32 1
  %.pre432 = load i32, i32* %avail_in142.phi.trans.insert, align 4
  %tmp28 = add i32 %.pre432, -1
  br label %if.end.140

if.then.130:                                      ; preds = %while.body.126.backedge, %sw.bb.123.if.then.130_crit_edge
  %tmp29 = phi i32 [ %.pre429, %sw.bb.123.if.then.130_crit_edge ], [ %or152, %while.body.126.backedge ]
  %.lcssa = phi i32 [ %tmp26, %sw.bb.123.if.then.130_crit_edge ], [ %add155, %while.body.126.backedge ]
  %sub134 = add nsw i32 %.lcssa, -8
  %shr135 = lshr i32 %tmp29, %sub134
  store i32 %sub134, i32* %bsLive127.pre-phi, align 4
  %origPtr = getelementptr inbounds %struct.DState, %struct.DState* %s, i64 0, i32 13
  %tmp30 = load i32, i32* %origPtr, align 4
  %shl175 = shl i32 %tmp30, 8
  %conv176 = and i32 %shr135, 255
  %or177 = or i32 %shl175, %conv176
  store i32 %or177, i32* %origPtr, align 4
  %nInUse = getelementptr inbounds %struct.DState, %struct.DState* %s, i64 0, i32 27
  %tmp31 = load i32, i32* %nInUse, align 4
  %add179 = add nsw i32 %tmp31, 2
  br label %save_state_and_return

if.end.140:                                       ; preds = %while.body.126.backedge, %if.end.140.lr.ph
  %lsr.iv = phi i32 [ %tmp28, %if.end.140.lr.ph ], [ %lsr.iv.next, %while.body.126.backedge ]
  %tmp32 = phi i32 [ %tmp26, %if.end.140.lr.ph ], [ %add155, %while.body.126.backedge ]
  %cmp143 = icmp eq i32 %lsr.iv, -1
  br i1 %cmp143, label %save_state_and_return, label %if.end.146

if.end.146:                                       ; preds = %if.end.140
  %tmp33 = bitcast %struct.bz_stream* %.pre428 to i8**
  %sunkaddr541 = ptrtoint %struct.DState* %s to i64
  %sunkaddr542 = add i64 %sunkaddr541, 32
  %sunkaddr543 = inttoptr i64 %sunkaddr542 to i32*
  %tmp34 = load i32, i32* %sunkaddr543, align 4
  %shl148 = shl i32 %tmp34, 8
  %tmp35 = load i8*, i8** %tmp33, align 8
  %tmp36 = load i8, i8* %tmp35, align 1
  %conv151 = zext i8 %tmp36 to i32
  %or152 = or i32 %conv151, %shl148
  store i32 %or152, i32* %sunkaddr543, align 4
  %add155 = add nsw i32 %tmp32, 8
  store i32 %add155, i32* %bsLive127.pre-phi, align 4
  %incdec.ptr158 = getelementptr inbounds i8, i8* %tmp35, i64 1
  store i8* %incdec.ptr158, i8** %tmp33, align 8
  %sunkaddr544 = ptrtoint %struct.bz_stream* %.pre428 to i64
  %sunkaddr545 = add i64 %sunkaddr544, 8
  %sunkaddr546 = inttoptr i64 %sunkaddr545 to i32*
  store i32 %lsr.iv, i32* %sunkaddr546, align 4
  %sunkaddr547 = ptrtoint %struct.bz_stream* %.pre428 to i64
  %sunkaddr548 = add i64 %sunkaddr547, 12
  %sunkaddr549 = inttoptr i64 %sunkaddr548 to i32*
  %tmp37 = load i32, i32* %sunkaddr549, align 4
  %inc164 = add i32 %tmp37, 1
  store i32 %inc164, i32* %sunkaddr549, align 4
  %cmp167 = icmp eq i32 %inc164, 0
  br i1 %cmp167, label %if.then.169, label %while.body.126.backedge

if.then.169:                                      ; preds = %if.end.146
  %sunkaddr550 = ptrtoint %struct.bz_stream* %.pre428 to i64
  %sunkaddr551 = add i64 %sunkaddr550, 16
  %sunkaddr552 = inttoptr i64 %sunkaddr551 to i32*
  %tmp38 = load i32, i32* %sunkaddr552, align 4
  %inc172 = add i32 %tmp38, 1
  store i32 %inc172, i32* %sunkaddr552, align 4
  br label %while.body.126.backedge

while.body.126.backedge:                          ; preds = %if.then.169, %if.end.146
  %lsr.iv.next = add i32 %lsr.iv, -1
  %cmp128 = icmp sgt i32 %add155, 7
  br i1 %cmp128, label %if.then.130, label %if.end.140

sw.default:                                       ; preds = %if.end, %if.end.thread
  %tmp39 = phi i32 [ 0, %if.end.thread ], [ %.pre, %if.end ]
  %tmp40 = phi i32 [ 0, %if.end.thread ], [ %.pre406, %if.end ]
  %tmp41 = phi i32 [ 0, %if.end.thread ], [ %.pre407, %if.end ]
  %tmp42 = phi i32 [ 0, %if.end.thread ], [ %.pre408, %if.end ]
  %tmp43 = phi i32 [ 0, %if.end.thread ], [ %.pre409, %if.end ]
  %tmp44 = phi i32 [ 0, %if.end.thread ], [ %.pre410, %if.end ]
  %tmp45 = phi i32 [ 0, %if.end.thread ], [ %.pre411, %if.end ]
  %tmp46 = phi i32 [ 0, %if.end.thread ], [ %.pre412, %if.end ]
  %tmp47 = phi i32 [ 0, %if.end.thread ], [ %.pre413, %if.end ]
  %tmp48 = phi i32 [ 0, %if.end.thread ], [ %.pre414, %if.end ]
  %tmp49 = phi i32 [ 0, %if.end.thread ], [ %.pre415, %if.end ]
  %tmp50 = phi i32 [ 0, %if.end.thread ], [ %.pre416, %if.end ]
  %tmp51 = phi i32 [ 0, %if.end.thread ], [ %.pre417, %if.end ]
  %tmp52 = phi i32 [ 0, %if.end.thread ], [ %.pre418, %if.end ]
  %tmp53 = phi i32 [ 0, %if.end.thread ], [ %.pre419, %if.end ]
  %tmp54 = phi i32 [ 0, %if.end.thread ], [ %.pre420, %if.end ]
  %tmp55 = phi i32 [ 0, %if.end.thread ], [ %.pre421, %if.end ]
  %tmp56 = phi i32 [ 0, %if.end.thread ], [ %.pre422, %if.end ]
  %tmp57 = phi i32 [ 0, %if.end.thread ], [ %.pre423, %if.end ]
  %save_j3.pre-phi469 = phi i32* [ %save_j, %if.end.thread ], [ %save_j3.phi.trans.insert, %if.end ]
  %save_t4.pre-phi467 = phi i32* [ %save_t, %if.end.thread ], [ %save_t4.phi.trans.insert, %if.end ]
  %save_alphaSize5.pre-phi465 = phi i32* [ %save_alphaSize, %if.end.thread ], [ %save_alphaSize5.phi.trans.insert, %if.end ]
  %save_nGroups6.pre-phi463 = phi i32* [ %save_nGroups, %if.end.thread ], [ %save_nGroups6.phi.trans.insert, %if.end ]
  %save_nSelectors7.pre-phi461 = phi i32* [ %save_nSelectors, %if.end.thread ], [ %save_nSelectors7.phi.trans.insert, %if.end ]
  %save_EOB8.pre-phi459 = phi i32* [ %save_EOB, %if.end.thread ], [ %save_EOB8.phi.trans.insert, %if.end ]
  %save_groupNo9.pre-phi457 = phi i32* [ %save_groupNo, %if.end.thread ], [ %save_groupNo9.phi.trans.insert, %if.end ]
  %save_groupPos10.pre-phi455 = phi i32* [ %save_groupPos, %if.end.thread ], [ %save_groupPos10.phi.trans.insert, %if.end ]
  %save_nextSym11.pre-phi453 = phi i32* [ %save_nextSym, %if.end.thread ], [ %save_nextSym11.phi.trans.insert, %if.end ]
  %save_nblockMAX12.pre-phi451 = phi i32* [ %save_nblockMAX, %if.end.thread ], [ %save_nblockMAX12.phi.trans.insert, %if.end ]
  %save_nblock13.pre-phi449 = phi i32* [ %save_nblock, %if.end.thread ], [ %save_nblock13.phi.trans.insert, %if.end ]
  %save_es14.pre-phi447 = phi i32* [ %save_es, %if.end.thread ], [ %save_es14.phi.trans.insert, %if.end ]
  %save_N15.pre-phi445 = phi i32* [ %save_N, %if.end.thread ], [ %save_N15.phi.trans.insert, %if.end ]
  %save_curr16.pre-phi443 = phi i32* [ %save_curr, %if.end.thread ], [ %save_curr16.phi.trans.insert, %if.end ]
  %save_zt17.pre-phi441 = phi i32* [ %save_zt, %if.end.thread ], [ %save_zt17.phi.trans.insert, %if.end ]
  %save_zn18.pre-phi439 = phi i32* [ %save_zn, %if.end.thread ], [ %save_zn18.phi.trans.insert, %if.end ]
  %save_zvec19.pre-phi437 = phi i32* [ %save_zvec, %if.end.thread ], [ %save_zvec19.phi.trans.insert, %if.end ]
  %save_zj20.pre-phi435 = phi i32* [ %save_zj, %if.end.thread ], [ %save_zj20.phi.trans.insert, %if.end ]
  tail call void @bar(i32 4001)
  br label %save_state_and_return

save_state_and_return:                            ; preds = %sw.default, %if.end.140, %if.then.130, %if.end.82, %if.end.33, %if.then.29
  %tmp58 = phi i32 [ %tmp39, %sw.default ], [ %.pre, %if.then.29 ], [ %.pre, %if.then.130 ], [ %.pre, %if.end.140 ], [ %.pre, %if.end.82 ], [ %.pre, %if.end.33 ]
  %tmp59 = phi i32 [ %tmp40, %sw.default ], [ %.pre406, %if.then.29 ], [ %.pre406, %if.then.130 ], [ %.pre406, %if.end.140 ], [ %.pre406, %if.end.82 ], [ %.pre406, %if.end.33 ]
  %tmp60 = phi i32 [ %tmp41, %sw.default ], [ %.pre407, %if.then.29 ], [ %.pre407, %if.then.130 ], [ %.pre407, %if.end.140 ], [ %.pre407, %if.end.82 ], [ %.pre407, %if.end.33 ]
  %tmp61 = phi i32 [ %tmp43, %sw.default ], [ %.pre409, %if.then.29 ], [ %.pre409, %if.then.130 ], [ %.pre409, %if.end.140 ], [ %.pre409, %if.end.82 ], [ %.pre409, %if.end.33 ]
  %tmp62 = phi i32 [ %tmp44, %sw.default ], [ %.pre410, %if.then.29 ], [ %.pre410, %if.then.130 ], [ %.pre410, %if.end.140 ], [ %.pre410, %if.end.82 ], [ %.pre410, %if.end.33 ]
  %tmp63 = phi i32 [ %tmp45, %sw.default ], [ %.pre411, %if.then.29 ], [ %.pre411, %if.then.130 ], [ %.pre411, %if.end.140 ], [ %.pre411, %if.end.82 ], [ %.pre411, %if.end.33 ]
  %tmp64 = phi i32 [ %tmp46, %sw.default ], [ %.pre412, %if.then.29 ], [ %.pre412, %if.then.130 ], [ %.pre412, %if.end.140 ], [ %.pre412, %if.end.82 ], [ %.pre412, %if.end.33 ]
  %tmp65 = phi i32 [ %tmp47, %sw.default ], [ %.pre413, %if.then.29 ], [ %.pre413, %if.then.130 ], [ %.pre413, %if.end.140 ], [ %.pre413, %if.end.82 ], [ %.pre413, %if.end.33 ]
  %tmp66 = phi i32 [ %tmp48, %sw.default ], [ %.pre414, %if.then.29 ], [ %.pre414, %if.then.130 ], [ %.pre414, %if.end.140 ], [ %.pre414, %if.end.82 ], [ %.pre414, %if.end.33 ]
  %tmp67 = phi i32 [ %tmp49, %sw.default ], [ %.pre415, %if.then.29 ], [ %.pre415, %if.then.130 ], [ %.pre415, %if.end.140 ], [ %.pre415, %if.end.82 ], [ %.pre415, %if.end.33 ]
  %tmp68 = phi i32 [ %tmp51, %sw.default ], [ %.pre417, %if.then.29 ], [ %.pre417, %if.then.130 ], [ %.pre417, %if.end.140 ], [ %.pre417, %if.end.82 ], [ %.pre417, %if.end.33 ]
  %tmp69 = phi i32 [ %tmp52, %sw.default ], [ %.pre418, %if.then.29 ], [ %.pre418, %if.then.130 ], [ %.pre418, %if.end.140 ], [ %.pre418, %if.end.82 ], [ %.pre418, %if.end.33 ]
  %tmp70 = phi i32 [ %tmp53, %sw.default ], [ %.pre419, %if.then.29 ], [ %.pre419, %if.then.130 ], [ %.pre419, %if.end.140 ], [ %.pre419, %if.end.82 ], [ %.pre419, %if.end.33 ]
  %tmp71 = phi i32 [ %tmp54, %sw.default ], [ %.pre420, %if.then.29 ], [ %.pre420, %if.then.130 ], [ %.pre420, %if.end.140 ], [ %.pre420, %if.end.82 ], [ %.pre420, %if.end.33 ]
  %tmp72 = phi i32 [ %tmp55, %sw.default ], [ %.pre421, %if.then.29 ], [ %.pre421, %if.then.130 ], [ %.pre421, %if.end.140 ], [ %.pre421, %if.end.82 ], [ %.pre421, %if.end.33 ]
  %tmp73 = phi i32 [ %tmp56, %sw.default ], [ %.pre422, %if.then.29 ], [ %.pre422, %if.then.130 ], [ %.pre422, %if.end.140 ], [ %.pre422, %if.end.82 ], [ %.pre422, %if.end.33 ]
  %tmp74 = phi i32 [ %tmp57, %sw.default ], [ %.pre423, %if.then.29 ], [ %.pre423, %if.then.130 ], [ %.pre423, %if.end.140 ], [ %.pre423, %if.end.82 ], [ %.pre423, %if.end.33 ]
  %save_j3.pre-phi468 = phi i32* [ %save_j3.pre-phi469, %sw.default ], [ %save_j3.phi.trans.insert, %if.then.29 ], [ %save_j3.phi.trans.insert, %if.then.130 ], [ %save_j3.phi.trans.insert, %if.end.140 ], [ %save_j3.phi.trans.insert, %if.end.82 ], [ %save_j3.phi.trans.insert, %if.end.33 ]
  %save_t4.pre-phi466 = phi i32* [ %save_t4.pre-phi467, %sw.default ], [ %save_t4.phi.trans.insert, %if.then.29 ], [ %save_t4.phi.trans.insert, %if.then.130 ], [ %save_t4.phi.trans.insert, %if.end.140 ], [ %save_t4.phi.trans.insert, %if.end.82 ], [ %save_t4.phi.trans.insert, %if.end.33 ]
  %save_alphaSize5.pre-phi464 = phi i32* [ %save_alphaSize5.pre-phi465, %sw.default ], [ %save_alphaSize5.phi.trans.insert, %if.then.29 ], [ %save_alphaSize5.phi.trans.insert, %if.then.130 ], [ %save_alphaSize5.phi.trans.insert, %if.end.140 ], [ %save_alphaSize5.phi.trans.insert, %if.end.82 ], [ %save_alphaSize5.phi.trans.insert, %if.end.33 ]
  %save_nGroups6.pre-phi462 = phi i32* [ %save_nGroups6.pre-phi463, %sw.default ], [ %save_nGroups6.phi.trans.insert, %if.then.29 ], [ %save_nGroups6.phi.trans.insert, %if.then.130 ], [ %save_nGroups6.phi.trans.insert, %if.end.140 ], [ %save_nGroups6.phi.trans.insert, %if.end.82 ], [ %save_nGroups6.phi.trans.insert, %if.end.33 ]
  %save_nSelectors7.pre-phi460 = phi i32* [ %save_nSelectors7.pre-phi461, %sw.default ], [ %save_nSelectors7.phi.trans.insert, %if.then.29 ], [ %save_nSelectors7.phi.trans.insert, %if.then.130 ], [ %save_nSelectors7.phi.trans.insert, %if.end.140 ], [ %save_nSelectors7.phi.trans.insert, %if.end.82 ], [ %save_nSelectors7.phi.trans.insert, %if.end.33 ]
  %save_EOB8.pre-phi458 = phi i32* [ %save_EOB8.pre-phi459, %sw.default ], [ %save_EOB8.phi.trans.insert, %if.then.29 ], [ %save_EOB8.phi.trans.insert, %if.then.130 ], [ %save_EOB8.phi.trans.insert, %if.end.140 ], [ %save_EOB8.phi.trans.insert, %if.end.82 ], [ %save_EOB8.phi.trans.insert, %if.end.33 ]
  %save_groupNo9.pre-phi456 = phi i32* [ %save_groupNo9.pre-phi457, %sw.default ], [ %save_groupNo9.phi.trans.insert, %if.then.29 ], [ %save_groupNo9.phi.trans.insert, %if.then.130 ], [ %save_groupNo9.phi.trans.insert, %if.end.140 ], [ %save_groupNo9.phi.trans.insert, %if.end.82 ], [ %save_groupNo9.phi.trans.insert, %if.end.33 ]
  %save_groupPos10.pre-phi454 = phi i32* [ %save_groupPos10.pre-phi455, %sw.default ], [ %save_groupPos10.phi.trans.insert, %if.then.29 ], [ %save_groupPos10.phi.trans.insert, %if.then.130 ], [ %save_groupPos10.phi.trans.insert, %if.end.140 ], [ %save_groupPos10.phi.trans.insert, %if.end.82 ], [ %save_groupPos10.phi.trans.insert, %if.end.33 ]
  %save_nextSym11.pre-phi452 = phi i32* [ %save_nextSym11.pre-phi453, %sw.default ], [ %save_nextSym11.phi.trans.insert, %if.then.29 ], [ %save_nextSym11.phi.trans.insert, %if.then.130 ], [ %save_nextSym11.phi.trans.insert, %if.end.140 ], [ %save_nextSym11.phi.trans.insert, %if.end.82 ], [ %save_nextSym11.phi.trans.insert, %if.end.33 ]
  %save_nblockMAX12.pre-phi450 = phi i32* [ %save_nblockMAX12.pre-phi451, %sw.default ], [ %save_nblockMAX12.phi.trans.insert, %if.then.29 ], [ %save_nblockMAX12.phi.trans.insert, %if.then.130 ], [ %save_nblockMAX12.phi.trans.insert, %if.end.140 ], [ %save_nblockMAX12.phi.trans.insert, %if.end.82 ], [ %save_nblockMAX12.phi.trans.insert, %if.end.33 ]
  %save_nblock13.pre-phi448 = phi i32* [ %save_nblock13.pre-phi449, %sw.default ], [ %save_nblock13.phi.trans.insert, %if.then.29 ], [ %save_nblock13.phi.trans.insert, %if.then.130 ], [ %save_nblock13.phi.trans.insert, %if.end.140 ], [ %save_nblock13.phi.trans.insert, %if.end.82 ], [ %save_nblock13.phi.trans.insert, %if.end.33 ]
  %save_es14.pre-phi446 = phi i32* [ %save_es14.pre-phi447, %sw.default ], [ %save_es14.phi.trans.insert, %if.then.29 ], [ %save_es14.phi.trans.insert, %if.then.130 ], [ %save_es14.phi.trans.insert, %if.end.140 ], [ %save_es14.phi.trans.insert, %if.end.82 ], [ %save_es14.phi.trans.insert, %if.end.33 ]
  %save_N15.pre-phi444 = phi i32* [ %save_N15.pre-phi445, %sw.default ], [ %save_N15.phi.trans.insert, %if.then.29 ], [ %save_N15.phi.trans.insert, %if.then.130 ], [ %save_N15.phi.trans.insert, %if.end.140 ], [ %save_N15.phi.trans.insert, %if.end.82 ], [ %save_N15.phi.trans.insert, %if.end.33 ]
  %save_curr16.pre-phi442 = phi i32* [ %save_curr16.pre-phi443, %sw.default ], [ %save_curr16.phi.trans.insert, %if.then.29 ], [ %save_curr16.phi.trans.insert, %if.then.130 ], [ %save_curr16.phi.trans.insert, %if.end.140 ], [ %save_curr16.phi.trans.insert, %if.end.82 ], [ %save_curr16.phi.trans.insert, %if.end.33 ]
  %save_zt17.pre-phi440 = phi i32* [ %save_zt17.pre-phi441, %sw.default ], [ %save_zt17.phi.trans.insert, %if.then.29 ], [ %save_zt17.phi.trans.insert, %if.then.130 ], [ %save_zt17.phi.trans.insert, %if.end.140 ], [ %save_zt17.phi.trans.insert, %if.end.82 ], [ %save_zt17.phi.trans.insert, %if.end.33 ]
  %save_zn18.pre-phi438 = phi i32* [ %save_zn18.pre-phi439, %sw.default ], [ %save_zn18.phi.trans.insert, %if.then.29 ], [ %save_zn18.phi.trans.insert, %if.then.130 ], [ %save_zn18.phi.trans.insert, %if.end.140 ], [ %save_zn18.phi.trans.insert, %if.end.82 ], [ %save_zn18.phi.trans.insert, %if.end.33 ]
  %save_zvec19.pre-phi436 = phi i32* [ %save_zvec19.pre-phi437, %sw.default ], [ %save_zvec19.phi.trans.insert, %if.then.29 ], [ %save_zvec19.phi.trans.insert, %if.then.130 ], [ %save_zvec19.phi.trans.insert, %if.end.140 ], [ %save_zvec19.phi.trans.insert, %if.end.82 ], [ %save_zvec19.phi.trans.insert, %if.end.33 ]
  %save_zj20.pre-phi434 = phi i32* [ %save_zj20.pre-phi435, %sw.default ], [ %save_zj20.phi.trans.insert, %if.then.29 ], [ %save_zj20.phi.trans.insert, %if.then.130 ], [ %save_zj20.phi.trans.insert, %if.end.140 ], [ %save_zj20.phi.trans.insert, %if.end.82 ], [ %save_zj20.phi.trans.insert, %if.end.33 ]
  %nblock.1 = phi i32 [ %tmp50, %sw.default ], [ %.pre416, %if.then.29 ], [ 0, %if.then.130 ], [ %.pre416, %if.end.140 ], [ %.pre416, %if.end.82 ], [ %.pre416, %if.end.33 ]
  %alphaSize.1 = phi i32 [ %tmp42, %sw.default ], [ %.pre408, %if.then.29 ], [ %add179, %if.then.130 ], [ %.pre408, %if.end.140 ], [ %.pre408, %if.end.82 ], [ %.pre408, %if.end.33 ]
  %retVal.0 = phi i32 [ 0, %sw.default ], [ -5, %if.then.29 ], [ -4, %if.then.130 ], [ 0, %if.end.140 ], [ 0, %if.end.82 ], [ 0, %if.end.33 ]
  store i32 %tmp58, i32* %save_i, align 4
  store i32 %tmp59, i32* %save_j3.pre-phi468, align 4
  store i32 %tmp60, i32* %save_t4.pre-phi466, align 4
  store i32 %alphaSize.1, i32* %save_alphaSize5.pre-phi464, align 4
  store i32 %tmp61, i32* %save_nGroups6.pre-phi462, align 4
  store i32 %tmp62, i32* %save_nSelectors7.pre-phi460, align 4
  store i32 %tmp63, i32* %save_EOB8.pre-phi458, align 4
  store i32 %tmp64, i32* %save_groupNo9.pre-phi456, align 4
  store i32 %tmp65, i32* %save_groupPos10.pre-phi454, align 4
  store i32 %tmp66, i32* %save_nextSym11.pre-phi452, align 4
  store i32 %tmp67, i32* %save_nblockMAX12.pre-phi450, align 4
  store i32 %nblock.1, i32* %save_nblock13.pre-phi448, align 4
  store i32 %tmp68, i32* %save_es14.pre-phi446, align 4
  store i32 %tmp69, i32* %save_N15.pre-phi444, align 4
  store i32 %tmp70, i32* %save_curr16.pre-phi442, align 4
  store i32 %tmp71, i32* %save_zt17.pre-phi440, align 4
  store i32 %tmp72, i32* %save_zn18.pre-phi438, align 4
  store i32 %tmp73, i32* %save_zvec19.pre-phi436, align 4
  store i32 %tmp74, i32* %save_zj20.pre-phi434, align 4
  ret i32 %retVal.0
}

!0 = !{!"branch_weights", i32 10, i32 1}
