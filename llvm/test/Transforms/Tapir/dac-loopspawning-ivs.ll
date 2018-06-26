; Test correct IV updating in loop spawning pass.
;
; Credit to Guy Blelloch for providing the original source code for
; this test.

; RUN: opt < %s -loop-spawning -S | FileCheck %s

%struct._IO_FILE = type { i32, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, %struct._IO_marker*, %struct._IO_FILE*, i32, i32, i64, i16, i8, [1 x i8], i8*, i64, i8*, i8*, i8*, i8*, i64, i32, [20 x i8] }
%struct._IO_marker = type { %struct._IO_marker*, %struct._IO_FILE*, i32 }
%struct.sequence.46 = type <{ i64*, i64*, i8, [7 x i8] }>
%"struct.std::less" = type { i8 }

$_ZN4pbbs12sample_sort_Im8sequenceIlESt4lessIlEEES1_INT0_1TEES5_RKT1_b = comdat any

@.str.47 = private unnamed_addr constant [22 x i8] c"Cannot allocate space\00", align 1
@stderr = external local_unnamed_addr global %struct._IO_FILE*, align 8

declare i64* @_ZN4pbbs17transpose_bucketsIlmEEPmPT_S3_PT0_mmmm(i64* %From, i64* %To, i64* %counts, i64 %n, i64 %block_size, i64 %num_blocks, i64 %num_buckets) #6

declare void @_ZN4pbbs9quicksortIlSt4lessIlEEEvPT_mRKT0_(i64* %A, i64 %n, %"struct.std::less"* dereferenceable(1) %f) #6

declare i32 @__gxx_personality_v0(...)

; Function Attrs: argmemonly nounwind
declare void @llvm.lifetime.start.p0i8(i64, i8* nocapture) #3

; Function Attrs: argmemonly nounwind
declare void @llvm.lifetime.end.p0i8(i64, i8* nocapture) #3

; Function Attrs: argmemonly nounwind
declare token @llvm.syncregion.start() #3

; Function Attrs: nounwind
declare noalias i8* @aligned_alloc(i64, i64) local_unnamed_addr #1

; Function Attrs: nounwind
declare void @free(i8* nocapture) local_unnamed_addr #1

; Function Attrs: noreturn nounwind
declare void @exit(i32) local_unnamed_addr #9

; Function Attrs: nounwind
declare i64 @fwrite(i8* nocapture, i64, i64, %struct._IO_FILE* nocapture) local_unnamed_addr #5

; Function Attrs: nounwind readnone speculatable
declare double @llvm.ceil.f64(double) #14

; Function Attrs: argmemonly
declare void @llvm.detached.rethrow.sl_p0i8i32s(token, { i8*, i32 }) #13

; Function Attrs: nounwind readnone speculatable
declare double @llvm.fabs.f64(double) #14

; Function Attrs: argmemonly nounwind
declare void @llvm.memcpy.p0i8.p0i8.i64(i8* nocapture writeonly, i8* nocapture readonly, i64, i32, i1) #3

; Function Attrs: argmemonly nounwind
declare void @llvm.memset.p0i8.i64(i8* nocapture writeonly, i8, i64, i32, i1) #3

declare double @sqrt(double) local_unnamed_addr

; Function Attrs: uwtable
define linkonce_odr void @_ZN4pbbs12sample_sort_Im8sequenceIlESt4lessIlEEES1_INT0_1TEES5_RKT1_b(%struct.sequence.46* noalias sret %agg.result, %struct.sequence.46* %A, %"struct.std::less"* dereferenceable(1) %f, i1 zeroext %inplace) local_unnamed_addr #6 comdat personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*) {
entry:
  %syncreg.i599 = tail call token @llvm.syncregion.start()
  %syncreg.i572 = tail call token @llvm.syncregion.start()
  %syncreg.i538 = tail call token @llvm.syncregion.start()
  %syncreg.i = tail call token @llvm.syncregion.start()
  %syncreg = tail call token @llvm.syncregion.start()
  %syncreg58 = tail call token @llvm.syncregion.start()
  %syncreg111 = tail call token @llvm.syncregion.start()
  %syncreg181 = tail call token @llvm.syncregion.start()
  %e.i = getelementptr inbounds %struct.sequence.46, %struct.sequence.46* %A, i64 0, i32 1
  %0 = bitcast i64** %e.i to i64*
  %1 = load i64, i64* %0, align 8, !tbaa !4932
  %2 = bitcast %struct.sequence.46* %A to i64*
  %3 = load i64, i64* %2, align 8, !tbaa !4929
  %sub.ptr.sub.i = sub i64 %1, %3
  %sub.ptr.div.i = ashr exact i64 %sub.ptr.sub.i, 3
  %cmp = icmp ult i64 %sub.ptr.div.i, 16384
  br i1 %cmp, label %if.then, label %if.else19

if.then:                                          ; preds = %entry
  %allocated.i = getelementptr inbounds %struct.sequence.46, %struct.sequence.46* %agg.result, i64 0, i32 2
  store i8 0, i8* %allocated.i, align 8, !tbaa !4934
  br i1 %inplace, label %_ZN8sequenceIlED2Ev.exit466, label %if.else

_ZN8sequenceIlED2Ev.exit466:                      ; preds = %if.then
  %4 = bitcast %struct.sequence.46* %agg.result to i64*
  store i64 %3, i64* %4, align 8, !tbaa !4929
  %e3.i461 = getelementptr inbounds %struct.sequence.46, %struct.sequence.46* %agg.result, i64 0, i32 1
  %5 = bitcast i64** %e3.i461 to i64*
  store i64 %1, i64* %5, align 8, !tbaa !4932
  store i8 0, i8* %allocated.i, align 8, !tbaa !4934
  %6 = inttoptr i64 %3 to i64*
  br label %if.end

if.else:                                          ; preds = %if.then
  %7 = lshr i64 %sub.ptr.div.i, 3
  %add.i.i = shl i64 %7, 6
  %mul1.i.i = add i64 %add.i.i, 64
  %call.i.i = tail call noalias i8* @aligned_alloc(i64 64, i64 %mul1.i.i) #5
  %8 = bitcast i8* %call.i.i to i64*
  %cmp.i.i = icmp eq i8* %call.i.i, null
  br i1 %cmp.i.i, label %if.then.i.i471, label %_ZN4pbbs17new_array_no_initIlEEPT_mb.exit.i

if.then.i.i471:                                   ; preds = %if.else
  %9 = load %struct._IO_FILE*, %struct._IO_FILE** @stderr, align 8, !tbaa !213
  %10 = tail call i64 @fwrite(i8* getelementptr inbounds ([22 x i8], [22 x i8]* @.str.47, i64 0, i64 0), i64 21, i64 1, %struct._IO_FILE* %9) #20
  tail call void @exit(i32 1) #21
  unreachable

_ZN4pbbs17new_array_no_initIlEEPT_mb.exit.i:      ; preds = %if.else
  %add.ptr.i473 = getelementptr inbounds i64, i64* %8, i64 %sub.ptr.div.i
  %cmp31.i = icmp eq i64 %sub.ptr.sub.i, 0
  %11 = ptrtoint i8* %call.i.i to i64
  %12 = ptrtoint i64* %add.ptr.i473 to i64
  br i1 %cmp31.i, label %pfor.cond.cleanup.i, label %pfor.detach.lr.ph.i

pfor.detach.lr.ph.i:                              ; preds = %_ZN4pbbs17new_array_no_initIlEEPT_mb.exit.i
  %s.i.i.i = getelementptr inbounds %struct.sequence.46, %struct.sequence.46* %A, i64 0, i32 0
  br label %pfor.detach.i

pfor.cond.cleanup.i:                              ; preds = %pfor.inc.i, %_ZN4pbbs17new_array_no_initIlEEPT_mb.exit.i
  sync within %syncreg.i, label %_ZN8sequenceIlED2Ev.exit489

pfor.detach.i:                                    ; preds = %pfor.inc.i, %pfor.detach.lr.ph.i
  %__begin.032.i = phi i64 [ 0, %pfor.detach.lr.ph.i ], [ %inc.i, %pfor.inc.i ]
  detach within %syncreg.i, label %pfor.body.i, label %pfor.inc.i unwind label %lpad8.i

pfor.body.i:                                      ; preds = %pfor.detach.i
  %13 = load i64*, i64** %s.i.i.i, align 8, !tbaa !4929
  %arrayidx.i.i.i = getelementptr inbounds i64, i64* %13, i64 %__begin.032.i
  %14 = load i64, i64* %arrayidx.i.i.i, align 8, !tbaa !100
  %add.ptr7.i = getelementptr inbounds i64, i64* %8, i64 %__begin.032.i
  store i64 %14, i64* %add.ptr7.i, align 8, !tbaa !100
  reattach within %syncreg.i, label %pfor.inc.i

pfor.inc.i:                                       ; preds = %pfor.body.i, %pfor.detach.i
  %inc.i = add nuw i64 %__begin.032.i, 1
  %cmp.i475 = icmp ult i64 %inc.i, %sub.ptr.div.i
  br i1 %cmp.i475, label %pfor.detach.i, label %pfor.cond.cleanup.i, !llvm.loop !31381

lpad8.i:                                          ; preds = %pfor.detach.i
  %15 = landingpad { i8*, i32 }
          cleanup
  sync within %syncreg.i, label %ehcleanup17.thread

_ZN8sequenceIlED2Ev.exit489:                      ; preds = %pfor.cond.cleanup.i
  %16 = bitcast %struct.sequence.46* %agg.result to i64*
  store i64 %11, i64* %16, align 8, !tbaa !4929
  %e3.i482 = getelementptr inbounds %struct.sequence.46, %struct.sequence.46* %agg.result, i64 0, i32 1
  %17 = bitcast i64** %e3.i482 to i64*
  store i64 %12, i64* %17, align 8, !tbaa !4932
  store i8 1, i8* %allocated.i, align 8, !tbaa !4934
  %18 = bitcast i8* %call.i.i to i64*
  br label %if.end

ehcleanup17.thread:                               ; preds = %lpad8.i
  %19 = extractvalue { i8*, i32 } %15, 0
  %20 = extractvalue { i8*, i32 } %15, 1
  br label %_ZN8sequenceIlED2Ev.exit497

if.end:                                           ; preds = %_ZN8sequenceIlED2Ev.exit489, %_ZN8sequenceIlED2Ev.exit466
  %21 = phi i8 [ 1, %_ZN8sequenceIlED2Ev.exit489 ], [ 0, %_ZN8sequenceIlED2Ev.exit466 ]
  %22 = phi i64* [ %18, %_ZN8sequenceIlED2Ev.exit489 ], [ %6, %_ZN8sequenceIlED2Ev.exit466 ]
  invoke void @_ZN4pbbs9quicksortIlSt4lessIlEEEvPT_mRKT0_(i64* %22, i64 %sub.ptr.div.i, %"struct.std::less"* nonnull dereferenceable(1) %f)
          to label %cleanup283 unwind label %ehcleanup17

ehcleanup17:                                      ; preds = %if.end
  %23 = landingpad { i8*, i32 }
          cleanup
  %24 = extractvalue { i8*, i32 } %23, 0
  %25 = extractvalue { i8*, i32 } %23, 1
  %tobool.i.i495 = icmp eq i8 %21, 0
  br i1 %tobool.i.i495, label %_ZN8sequenceIlED2Ev.exit497, label %if.then.i.i496

if.then.i.i496:                                   ; preds = %ehcleanup17
  %26 = bitcast %struct.sequence.46* %agg.result to i8**
  %27 = load i8*, i8** %26, align 8, !tbaa !4929
  tail call void @free(i8* %27) #5
  br label %_ZN8sequenceIlED2Ev.exit497

_ZN8sequenceIlED2Ev.exit497:                      ; preds = %ehcleanup17.thread, %ehcleanup17, %if.then.i.i496
  %exn.slot.2676 = phi i8* [ %19, %ehcleanup17.thread ], [ %24, %ehcleanup17 ], [ %24, %if.then.i.i496 ]
  %ehselector.slot.2675 = phi i32 [ %20, %ehcleanup17.thread ], [ %25, %ehcleanup17 ], [ %25, %if.then.i.i496 ]
  %28 = bitcast %struct.sequence.46* %agg.result to i8*
  tail call void @llvm.memset.p0i8.i64(i8* %28, i8 0, i64 16, i32 8, i1 false) #5
  br label %ehcleanup284

if.else19:                                        ; preds = %entry
  %s.i498 = getelementptr inbounds %struct.sequence.46, %struct.sequence.46* %A, i64 0, i32 0
  %conv.i = uitofp i64 %sub.ptr.div.i to double
  %sqrt = tail call double @sqrt(double %conv.i) #1
  %29 = tail call double @llvm.fabs.f64(double %sqrt)
  %30 = tail call double @llvm.ceil.f64(double %29)
  %conv = fptoui double %30 to i64
  %div = udiv i64 %conv, 5
  %31 = icmp ult i64 %conv, 5
  br i1 %31, label %_ZN4pbbsL7log2_upImEEiT_.exit, label %while.body.i.preheader

while.body.i.preheader:                           ; preds = %if.else19
  br label %while.body.i

while.body.i:                                     ; preds = %while.body.i.preheader, %while.body.i
  %b.07.i = phi i64 [ %shr.i, %while.body.i ], [ %div, %while.body.i.preheader ]
  %a.06.i = phi i32 [ %inc.i499, %while.body.i ], [ 0, %while.body.i.preheader ]
  %shr.i = lshr i64 %b.07.i, 1
  %inc.i499 = add nuw nsw i32 %a.06.i, 1
  %cmp.i500 = icmp eq i64 %shr.i, 0
  br i1 %cmp.i500, label %_ZN4pbbsL7log2_upImEEiT_.exit, label %while.body.i, !llvm.loop !559

_ZN4pbbsL7log2_upImEEiT_.exit:                    ; preds = %while.body.i, %if.else19
  %a.0.lcssa.i = phi i32 [ 0, %if.else19 ], [ %inc.i499, %while.body.i ]
  %shl = shl i32 1, %a.0.lcssa.i
  %conv27 = sext i32 %shl to i64
  %sub = add nsw i64 %sub.ptr.div.i, -1
  %div28 = udiv i64 %sub, %conv27
  %add29 = add i64 %div28, 1
  %add31 = add nuw nsw i64 %div, 1
  %mul = shl i64 %add31, 3
  %mul32 = mul i64 %add31, %conv27
  %add.i.i502 = shl i64 %add31, 6
  %mul1.i.i503 = add i64 %add.i.i502, 64
  %call.i.i504 = tail call noalias i8* @aligned_alloc(i64 64, i64 %mul1.i.i503) #5
  %cmp.i.i505 = icmp eq i8* %call.i.i504, null
  br i1 %cmp.i.i505, label %if.then.i.i506, label %_ZN4pbbs9new_arrayIlEEPT_m.exit

if.then.i.i506:                                   ; preds = %_ZN4pbbsL7log2_upImEEiT_.exit
  %32 = load %struct._IO_FILE*, %struct._IO_FILE** @stderr, align 8, !tbaa !213
  %33 = tail call i64 @fwrite(i8* getelementptr inbounds ([22 x i8], [22 x i8]* @.str.47, i64 0, i64 0), i64 21, i64 1, %struct._IO_FILE* %32) #20
  tail call void @exit(i32 1) #21
  unreachable

_ZN4pbbs9new_arrayIlEEPT_m.exit:                  ; preds = %_ZN4pbbsL7log2_upImEEiT_.exit
  %34 = bitcast i8* %call.i.i504 to i64*
  %cmp38667 = icmp eq i64 %mul, 0
  br i1 %cmp38667, label %pfor.cond.cleanup, label %pfor.detach.preheader

pfor.detach.preheader:                            ; preds = %_ZN4pbbs9new_arrayIlEEPT_m.exit
  br label %pfor.detach

pfor.cond.cleanup:                                ; preds = %pfor.inc, %_ZN4pbbs9new_arrayIlEEPT_m.exit
  sync within %syncreg, label %sync.continue

pfor.detach:                                      ; preds = %pfor.detach.preheader, %pfor.inc
  %__begin.0668 = phi i64 [ %inc, %pfor.inc ], [ 0, %pfor.detach.preheader ]
  detach within %syncreg, label %pfor.body, label %pfor.inc unwind label %lpad49

pfor.body:                                        ; preds = %pfor.detach
  %mul.i = mul i64 %__begin.0668, 3935559000370003845
  %add.i = add i64 %mul.i, 2691343689449507681
  %shr.i508 = lshr i64 %add.i, 21
  %xor.i = xor i64 %shr.i508, %add.i
  %shl.i = shl i64 %xor.i, 37
  %xor1.i = xor i64 %shl.i, %xor.i
  %shr2.i = lshr i64 %xor1.i, 4
  %xor3.i = xor i64 %shr2.i, %xor1.i
  %mul4.i = mul i64 %xor3.i, 4768777513237032717
  %shl5.i = mul i64 %xor3.i, -7053316708176494592
  %xor6.i = xor i64 %mul4.i, %shl5.i
  %shr7.i = lshr i64 %xor6.i, 41
  %xor8.i = xor i64 %shr7.i, %xor6.i
  %shl9.i = shl i64 %xor8.i, 5
  %xor10.i = xor i64 %shl9.i, %xor8.i
  %rem = urem i64 %xor10.i, %sub.ptr.div.i
  %35 = load i64*, i64** %s.i498, align 8, !tbaa !4929
  %arrayidx.i = getelementptr inbounds i64, i64* %35, i64 %rem
  %36 = load i64, i64* %arrayidx.i, align 8, !tbaa !100
  %arrayidx = getelementptr inbounds i64, i64* %34, i64 %__begin.0668
  store i64 %36, i64* %arrayidx, align 8, !tbaa !100
  reattach within %syncreg, label %pfor.inc

pfor.inc:                                         ; preds = %pfor.detach, %pfor.body
  %inc = add nuw i64 %__begin.0668, 1
  %cmp38 = icmp ult i64 %inc, %mul
  br i1 %cmp38, label %pfor.detach, label %pfor.cond.cleanup, !llvm.loop !31460

lpad49:                                           ; preds = %pfor.detach
  %37 = landingpad { i8*, i32 }
          cleanup
  %38 = extractvalue { i8*, i32 } %37, 0
  %39 = extractvalue { i8*, i32 } %37, 1
  sync within %syncreg, label %ehcleanup284

sync.continue:                                    ; preds = %pfor.cond.cleanup
  tail call void @_ZN4pbbs9quicksortIlSt4lessIlEEEvPT_mRKT0_(i64* %34, i64 %mul, %"struct.std::less"* nonnull dereferenceable(1) %f)
  %40 = lshr i64 %div, 3
  %add.i.i510 = shl i64 %40, 6
  %mul1.i.i511 = add i64 %add.i.i510, 64
  %call.i.i512 = tail call noalias i8* @aligned_alloc(i64 64, i64 %mul1.i.i511) #5
  %cmp.i.i513 = icmp eq i8* %call.i.i512, null
  br i1 %cmp.i.i513, label %if.then.i.i514, label %_ZN4pbbs9new_arrayIlEEPT_m.exit516

if.then.i.i514:                                   ; preds = %sync.continue
  %41 = load %struct._IO_FILE*, %struct._IO_FILE** @stderr, align 8, !tbaa !213
  %42 = tail call i64 @fwrite(i8* getelementptr inbounds ([22 x i8], [22 x i8]* @.str.47, i64 0, i64 0), i64 21, i64 1, %struct._IO_FILE* %41) #20
  tail call void @exit(i32 1) #21
  unreachable

_ZN4pbbs9new_arrayIlEEPT_m.exit516:               ; preds = %sync.continue
  %43 = bitcast i8* %call.i.i512 to i64*
  %44 = icmp ugt i64 %conv, 4
  br i1 %44, label %pfor.detach70.preheader, label %pfor.cond.cleanup69

pfor.detach70.preheader:                          ; preds = %_ZN4pbbs9new_arrayIlEEPT_m.exit516
  br label %pfor.detach70

pfor.cond.cleanup69:                              ; preds = %pfor.inc79, %_ZN4pbbs9new_arrayIlEEPT_m.exit516
  sync within %syncreg58, label %sync.continue81

pfor.detach70:                                    ; preds = %pfor.detach70.preheader, %pfor.inc79
  %__begin60.0666 = phi i64 [ %inc80, %pfor.inc79 ], [ 0, %pfor.detach70.preheader ]
  detach within %syncreg58, label %pfor.body74, label %pfor.inc79

pfor.body74:                                      ; preds = %pfor.detach70
  %mul75 = shl i64 %__begin60.0666, 3
  %arrayidx76 = getelementptr inbounds i64, i64* %34, i64 %mul75
  %45 = load i64, i64* %arrayidx76, align 8, !tbaa !100
  %arrayidx77 = getelementptr inbounds i64, i64* %43, i64 %__begin60.0666
  store i64 %45, i64* %arrayidx77, align 8, !tbaa !100
  reattach within %syncreg58, label %pfor.inc79

pfor.inc79:                                       ; preds = %pfor.body74, %pfor.detach70
  %inc80 = add nuw nsw i64 %__begin60.0666, 1
  %cmp68 = icmp ult i64 %inc80, %div
  br i1 %cmp68, label %pfor.detach70, label %pfor.cond.cleanup69, !llvm.loop !31479

sync.continue81:                                  ; preds = %pfor.cond.cleanup69
  tail call void @free(i8* %call.i.i504) #5
  br i1 %inplace, label %_ZN8sequenceIlED2Ev.exit533, label %if.else93

_ZN8sequenceIlED2Ev.exit533:                      ; preds = %sync.continue81
  %46 = load i64, i64* %2, align 8, !tbaa !4929, !noalias !31486
  %47 = load i64, i64* %0, align 8, !tbaa !4932, !noalias !31486
  br label %if.end104

if.else93:                                        ; preds = %sync.continue81
  %48 = lshr i64 %sub.ptr.div.i, 3
  %add.i539 = shl i64 %48, 6
  %mul1.i = add i64 %add.i539, 64
  %call.i540 = tail call noalias i8* @aligned_alloc(i64 64, i64 %mul1.i) #5
  %49 = bitcast i8* %call.i540 to i64*
  %cmp.i541 = icmp eq i8* %call.i540, null
  br i1 %cmp.i541, label %if.then.i542, label %if.end.i

if.then.i542:                                     ; preds = %if.else93
  %50 = load %struct._IO_FILE*, %struct._IO_FILE** @stderr, align 8, !tbaa !213
  %51 = tail call i64 @fwrite(i8* getelementptr inbounds ([22 x i8], [22 x i8]* @.str.47, i64 0, i64 0), i64 21, i64 1, %struct._IO_FILE* %50) #20
  tail call void @exit(i32 1) #21
  unreachable

if.end.i:                                         ; preds = %if.else93
  %cmp420.i = icmp eq i64 %mul1.i, 0
  br i1 %cmp420.i, label %pfor.cond.cleanup.i544, label %pfor.detach.i545.preheader

pfor.detach.i545.preheader:                       ; preds = %if.end.i
  br label %pfor.detach.i545

pfor.cond.cleanup.i544:                           ; preds = %pfor.inc.i548, %if.end.i
  sync within %syncreg.i538, label %_ZN8sequenceIlED2Ev.exit566

pfor.detach.i545:                                 ; preds = %pfor.detach.i545.preheader, %pfor.inc.i548
  %i.021.i = phi i64 [ %add5.i, %pfor.inc.i548 ], [ 0, %pfor.detach.i545.preheader ]
  detach within %syncreg.i538, label %pfor.body.i547, label %pfor.inc.i548

pfor.body.i547:                                   ; preds = %pfor.detach.i545
  %arrayidx.i546 = getelementptr inbounds i8, i8* %call.i540, i64 %i.021.i
  store i8 0, i8* %arrayidx.i546, align 1, !tbaa !218
  reattach within %syncreg.i538, label %pfor.inc.i548

pfor.inc.i548:                                    ; preds = %pfor.body.i547, %pfor.detach.i545
  %add5.i = add i64 %i.021.i, 2097152
  %cmp4.i = icmp ult i64 %add5.i, %mul1.i
  br i1 %cmp4.i, label %pfor.detach.i545, label %pfor.cond.cleanup.i544, !llvm.loop !4745

_ZN8sequenceIlED2Ev.exit566:                      ; preds = %pfor.cond.cleanup.i544
  %52 = ptrtoint i8* %call.i540 to i64
  %add.ptr.i551 = getelementptr inbounds i64, i64* %49, i64 %sub.ptr.div.i
  %53 = ptrtoint i64* %add.ptr.i551 to i64
  br label %if.end104

if.end104:                                        ; preds = %_ZN8sequenceIlED2Ev.exit566, %_ZN8sequenceIlED2Ev.exit533
  %Bs.sroa.0.0 = phi i64 [ %46, %_ZN8sequenceIlED2Ev.exit533 ], [ %52, %_ZN8sequenceIlED2Ev.exit566 ]
  %Bs.sroa.16.0 = phi i64 [ %47, %_ZN8sequenceIlED2Ev.exit533 ], [ %53, %_ZN8sequenceIlED2Ev.exit566 ]
  %54 = inttoptr i64 %Bs.sroa.0.0 to i64*
  %55 = inttoptr i64 %Bs.sroa.0.0 to i8*
  %56 = lshr i64 %mul32, 3
  %add.i573 = shl i64 %56, 6
  %mul1.i574 = add i64 %add.i573, 64
  %call.i575 = tail call noalias i8* @aligned_alloc(i64 64, i64 %mul1.i574) #5
  %57 = bitcast i8* %call.i575 to i64*
  %cmp.i576 = icmp eq i8* %call.i575, null
  br i1 %cmp.i576, label %if.then.i577, label %if.end.i578

if.then.i577:                                     ; preds = %if.end104
  %58 = load %struct._IO_FILE*, %struct._IO_FILE** @stderr, align 8, !tbaa !213
  %59 = tail call i64 @fwrite(i8* getelementptr inbounds ([22 x i8], [22 x i8]* @.str.47, i64 0, i64 0), i64 21, i64 1, %struct._IO_FILE* %58) #20
  tail call void @exit(i32 1) #21
  unreachable

if.end.i578:                                      ; preds = %if.end104
  %cmp420.i579 = icmp eq i64 %mul1.i574, 0
  br i1 %cmp420.i579, label %pfor.cond.cleanup.i581, label %pfor.detach.i583.preheader

pfor.detach.i583.preheader:                       ; preds = %if.end.i578
  br label %pfor.detach.i583

pfor.cond.cleanup.i581:                           ; preds = %pfor.inc.i588, %if.end.i578
  sync within %syncreg.i572, label %invoke.cont109

pfor.detach.i583:                                 ; preds = %pfor.detach.i583.preheader, %pfor.inc.i588
  %i.021.i582 = phi i64 [ %add5.i586, %pfor.inc.i588 ], [ 0, %pfor.detach.i583.preheader ]
  detach within %syncreg.i572, label %pfor.body.i585, label %pfor.inc.i588

pfor.body.i585:                                   ; preds = %pfor.detach.i583
  %arrayidx.i584 = getelementptr inbounds i8, i8* %call.i575, i64 %i.021.i582
  store i8 0, i8* %arrayidx.i584, align 1, !tbaa !218
  reattach within %syncreg.i572, label %pfor.inc.i588

pfor.inc.i588:                                    ; preds = %pfor.body.i585, %pfor.detach.i583
  %add5.i586 = add i64 %i.021.i582, 2097152
  %cmp4.i587 = icmp ult i64 %add5.i586, %mul1.i574
  br i1 %cmp4.i587, label %pfor.detach.i583, label %pfor.cond.cleanup.i581, !llvm.loop !221

invoke.cont109:                                   ; preds = %pfor.cond.cleanup.i581
  %sub116 = add nsw i64 %conv27, -1
  %add.ptr2.i = getelementptr inbounds i64, i64* %43, i64 %div
  %60 = shl i64 %div, 3
  %61 = add i64 %60, 8
  %62 = shl i64 %div28, 3
  %63 = add i64 %62, 8
  %inplace.not = xor i1 %inplace, true
  br label %pfor.detach122

pfor.cond.cleanup121:                             ; preds = %pfor.inc159
  sync within %syncreg111, label %sync.continue168

pfor.detach122:                                   ; preds = %invoke.cont109, %pfor.inc159
  %indvars.iv = phi i64 [ 0, %invoke.cont109 ], [ %indvars.iv.next, %pfor.inc159 ]
  %__begin113.0663 = phi i64 [ 0, %invoke.cont109 ], [ %inc160, %pfor.inc159 ]
  %64 = mul i64 %61, %__begin113.0663
  %scevgep = getelementptr i8, i8* %call.i575, i64 %64
  detach within %syncreg111, label %pfor.body126, label %pfor.inc159 unwind label %lpad161.loopexit
; CHECK: define internal fastcc void @_ZN4pbbs12sample_sort_Im8sequenceIlESt4lessIlEEES1_INT0_1TEES5_RKT1_b_pfor.detach122.ls(i64 %__begin113.0663.start.ls,
; CHECK: pfor.detach122.ls:
; CHECK-NOT: indvars.iv = phi
; CHECK: %__begin113.0663.ls = phi i64
; CHECK-NOT: phi
; CHECK: mul i64 %add29.ls, %__begin113.0663.ls
; CHECK: br label %pfor.body126.ls

pfor.body126:                                     ; preds = %pfor.detach122
  %mul127 = mul i64 %__begin113.0663, %add29
  %cmp129 = icmp ult i64 %__begin113.0663, %sub116
  %sub130 = sub i64 %sub.ptr.div.i, %mul127
  %cond = select i1 %cmp129, i64 %add29, i64 %sub130
  %add134 = add i64 %cond, %mul127
  %cmp135661 = icmp ult i64 %mul127, %add134
  %or.cond681 = and i1 %cmp135661, %inplace.not
  br i1 %or.cond681, label %for.body.lr.ph, label %if.end146

for.body.lr.ph:                                   ; preds = %pfor.body126
  %65 = load i64*, i64** %s.i498, align 8, !tbaa !4929
  %66 = add i64 %cond, %indvars.iv
  %min.iters.check = icmp ult i64 %cond, 16
  br i1 %min.iters.check, label %for.body.preheader, label %vector.memcheck

vector.memcheck:                                  ; preds = %for.body.lr.ph
  %67 = bitcast i64* %65 to i8*
  %68 = mul i64 %63, %__begin113.0663
  %uglygep = getelementptr i8, i8* %55, i64 %68
  %69 = shl i64 %cond, 3
  %scevgep686 = getelementptr i8, i8* %uglygep, i64 %69
  %uglygep687 = getelementptr i8, i8* %67, i64 %68
  %scevgep688 = getelementptr i64, i64* %65, i64 %cond
  %scevgep688689 = bitcast i64* %scevgep688 to i8*
  %bound0 = icmp ult i8* %55, %scevgep688689
  %bound1 = icmp ult i8* %uglygep687, %scevgep686
  %memcheck.conflict = and i1 %bound0, %bound1
  br i1 %memcheck.conflict, label %for.body.preheader, label %vector.ph

vector.ph:                                        ; preds = %vector.memcheck
  %n.vec = and i64 %cond, -16
  %ind.end = add i64 %mul127, %n.vec
  %70 = add i64 %n.vec, -16
  %71 = lshr exact i64 %70, 4
  %72 = add nuw nsw i64 %71, 1
  %xtraiter = and i64 %72, 3
  %73 = icmp ult i64 %70, 48
  br i1 %73, label %middle.block.unr-lcssa, label %vector.ph.new

vector.ph.new:                                    ; preds = %vector.ph
  %unroll_iter = sub nsw i64 %72, %xtraiter
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph.new
  %index = phi i64 [ 0, %vector.ph.new ], [ %index.next.3, %vector.body ]
  %niter = phi i64 [ %unroll_iter, %vector.ph.new ], [ %niter.nsub.3, %vector.body ]
  %74 = add i64 %mul127, %index
  %75 = getelementptr inbounds i64, i64* %54, i64 %74
  %76 = getelementptr inbounds i64, i64* %65, i64 %74
  %77 = bitcast i64* %76 to <4 x i64>*
  %wide.load = load <4 x i64>, <4 x i64>* %77, align 8, !tbaa !100, !alias.scope !31541
  %78 = getelementptr i64, i64* %76, i64 4
  %79 = bitcast i64* %78 to <4 x i64>*
  %wide.load694 = load <4 x i64>, <4 x i64>* %79, align 8, !tbaa !100, !alias.scope !31541
  %80 = getelementptr i64, i64* %76, i64 8
  %81 = bitcast i64* %80 to <4 x i64>*
  %wide.load695 = load <4 x i64>, <4 x i64>* %81, align 8, !tbaa !100, !alias.scope !31541
  %82 = getelementptr i64, i64* %76, i64 12
  %83 = bitcast i64* %82 to <4 x i64>*
  %wide.load696 = load <4 x i64>, <4 x i64>* %83, align 8, !tbaa !100, !alias.scope !31541
  %84 = bitcast i64* %75 to <4 x i64>*
  store <4 x i64> %wide.load, <4 x i64>* %84, align 8, !tbaa !100, !alias.scope !31545, !noalias !31541
  %85 = getelementptr i64, i64* %75, i64 4
  %86 = bitcast i64* %85 to <4 x i64>*
  store <4 x i64> %wide.load694, <4 x i64>* %86, align 8, !tbaa !100, !alias.scope !31545, !noalias !31541
  %87 = getelementptr i64, i64* %75, i64 8
  %88 = bitcast i64* %87 to <4 x i64>*
  store <4 x i64> %wide.load695, <4 x i64>* %88, align 8, !tbaa !100, !alias.scope !31545, !noalias !31541
  %89 = getelementptr i64, i64* %75, i64 12
  %90 = bitcast i64* %89 to <4 x i64>*
  store <4 x i64> %wide.load696, <4 x i64>* %90, align 8, !tbaa !100, !alias.scope !31545, !noalias !31541
  %index.next = or i64 %index, 16
  %91 = add i64 %mul127, %index.next
  %92 = getelementptr inbounds i64, i64* %54, i64 %91
  %93 = getelementptr inbounds i64, i64* %65, i64 %91
  %94 = bitcast i64* %93 to <4 x i64>*
  %wide.load.1 = load <4 x i64>, <4 x i64>* %94, align 8, !tbaa !100, !alias.scope !31541
  %95 = getelementptr i64, i64* %93, i64 4
  %96 = bitcast i64* %95 to <4 x i64>*
  %wide.load694.1 = load <4 x i64>, <4 x i64>* %96, align 8, !tbaa !100, !alias.scope !31541
  %97 = getelementptr i64, i64* %93, i64 8
  %98 = bitcast i64* %97 to <4 x i64>*
  %wide.load695.1 = load <4 x i64>, <4 x i64>* %98, align 8, !tbaa !100, !alias.scope !31541
  %99 = getelementptr i64, i64* %93, i64 12
  %100 = bitcast i64* %99 to <4 x i64>*
  %wide.load696.1 = load <4 x i64>, <4 x i64>* %100, align 8, !tbaa !100, !alias.scope !31541
  %101 = bitcast i64* %92 to <4 x i64>*
  store <4 x i64> %wide.load.1, <4 x i64>* %101, align 8, !tbaa !100, !alias.scope !31545, !noalias !31541
  %102 = getelementptr i64, i64* %92, i64 4
  %103 = bitcast i64* %102 to <4 x i64>*
  store <4 x i64> %wide.load694.1, <4 x i64>* %103, align 8, !tbaa !100, !alias.scope !31545, !noalias !31541
  %104 = getelementptr i64, i64* %92, i64 8
  %105 = bitcast i64* %104 to <4 x i64>*
  store <4 x i64> %wide.load695.1, <4 x i64>* %105, align 8, !tbaa !100, !alias.scope !31545, !noalias !31541
  %106 = getelementptr i64, i64* %92, i64 12
  %107 = bitcast i64* %106 to <4 x i64>*
  store <4 x i64> %wide.load696.1, <4 x i64>* %107, align 8, !tbaa !100, !alias.scope !31545, !noalias !31541
  %index.next.1 = or i64 %index, 32
  %108 = add i64 %mul127, %index.next.1
  %109 = getelementptr inbounds i64, i64* %54, i64 %108
  %110 = getelementptr inbounds i64, i64* %65, i64 %108
  %111 = bitcast i64* %110 to <4 x i64>*
  %wide.load.2 = load <4 x i64>, <4 x i64>* %111, align 8, !tbaa !100, !alias.scope !31541
  %112 = getelementptr i64, i64* %110, i64 4
  %113 = bitcast i64* %112 to <4 x i64>*
  %wide.load694.2 = load <4 x i64>, <4 x i64>* %113, align 8, !tbaa !100, !alias.scope !31541
  %114 = getelementptr i64, i64* %110, i64 8
  %115 = bitcast i64* %114 to <4 x i64>*
  %wide.load695.2 = load <4 x i64>, <4 x i64>* %115, align 8, !tbaa !100, !alias.scope !31541
  %116 = getelementptr i64, i64* %110, i64 12
  %117 = bitcast i64* %116 to <4 x i64>*
  %wide.load696.2 = load <4 x i64>, <4 x i64>* %117, align 8, !tbaa !100, !alias.scope !31541
  %118 = bitcast i64* %109 to <4 x i64>*
  store <4 x i64> %wide.load.2, <4 x i64>* %118, align 8, !tbaa !100, !alias.scope !31545, !noalias !31541
  %119 = getelementptr i64, i64* %109, i64 4
  %120 = bitcast i64* %119 to <4 x i64>*
  store <4 x i64> %wide.load694.2, <4 x i64>* %120, align 8, !tbaa !100, !alias.scope !31545, !noalias !31541
  %121 = getelementptr i64, i64* %109, i64 8
  %122 = bitcast i64* %121 to <4 x i64>*
  store <4 x i64> %wide.load695.2, <4 x i64>* %122, align 8, !tbaa !100, !alias.scope !31545, !noalias !31541
  %123 = getelementptr i64, i64* %109, i64 12
  %124 = bitcast i64* %123 to <4 x i64>*
  store <4 x i64> %wide.load696.2, <4 x i64>* %124, align 8, !tbaa !100, !alias.scope !31545, !noalias !31541
  %index.next.2 = or i64 %index, 48
  %125 = add i64 %mul127, %index.next.2
  %126 = getelementptr inbounds i64, i64* %54, i64 %125
  %127 = getelementptr inbounds i64, i64* %65, i64 %125
  %128 = bitcast i64* %127 to <4 x i64>*
  %wide.load.3 = load <4 x i64>, <4 x i64>* %128, align 8, !tbaa !100, !alias.scope !31541
  %129 = getelementptr i64, i64* %127, i64 4
  %130 = bitcast i64* %129 to <4 x i64>*
  %wide.load694.3 = load <4 x i64>, <4 x i64>* %130, align 8, !tbaa !100, !alias.scope !31541
  %131 = getelementptr i64, i64* %127, i64 8
  %132 = bitcast i64* %131 to <4 x i64>*
  %wide.load695.3 = load <4 x i64>, <4 x i64>* %132, align 8, !tbaa !100, !alias.scope !31541
  %133 = getelementptr i64, i64* %127, i64 12
  %134 = bitcast i64* %133 to <4 x i64>*
  %wide.load696.3 = load <4 x i64>, <4 x i64>* %134, align 8, !tbaa !100, !alias.scope !31541
  %135 = bitcast i64* %126 to <4 x i64>*
  store <4 x i64> %wide.load.3, <4 x i64>* %135, align 8, !tbaa !100, !alias.scope !31545, !noalias !31541
  %136 = getelementptr i64, i64* %126, i64 4
  %137 = bitcast i64* %136 to <4 x i64>*
  store <4 x i64> %wide.load694.3, <4 x i64>* %137, align 8, !tbaa !100, !alias.scope !31545, !noalias !31541
  %138 = getelementptr i64, i64* %126, i64 8
  %139 = bitcast i64* %138 to <4 x i64>*
  store <4 x i64> %wide.load695.3, <4 x i64>* %139, align 8, !tbaa !100, !alias.scope !31545, !noalias !31541
  %140 = getelementptr i64, i64* %126, i64 12
  %141 = bitcast i64* %140 to <4 x i64>*
  store <4 x i64> %wide.load696.3, <4 x i64>* %141, align 8, !tbaa !100, !alias.scope !31545, !noalias !31541
  %index.next.3 = add i64 %index, 64
  %niter.nsub.3 = add i64 %niter, -4
  %niter.ncmp.3 = icmp eq i64 %niter.nsub.3, 0
  br i1 %niter.ncmp.3, label %middle.block.unr-lcssa, label %vector.body, !llvm.loop !31547

middle.block.unr-lcssa:                           ; preds = %vector.body, %vector.ph
  %index.unr = phi i64 [ 0, %vector.ph ], [ %index.next.3, %vector.body ]
  %lcmp.mod = icmp eq i64 %xtraiter, 0
  br i1 %lcmp.mod, label %middle.block, label %vector.body.epil.preheader

vector.body.epil.preheader:                       ; preds = %middle.block.unr-lcssa
  br label %vector.body.epil

vector.body.epil:                                 ; preds = %vector.body.epil, %vector.body.epil.preheader
  %index.epil = phi i64 [ %index.unr, %vector.body.epil.preheader ], [ %index.next.epil, %vector.body.epil ]
  %epil.iter = phi i64 [ %xtraiter, %vector.body.epil.preheader ], [ %epil.iter.sub, %vector.body.epil ]
  %142 = add i64 %mul127, %index.epil
  %143 = getelementptr inbounds i64, i64* %54, i64 %142
  %144 = getelementptr inbounds i64, i64* %65, i64 %142
  %145 = bitcast i64* %144 to <4 x i64>*
  %wide.load.epil = load <4 x i64>, <4 x i64>* %145, align 8, !tbaa !100, !alias.scope !31541
  %146 = getelementptr i64, i64* %144, i64 4
  %147 = bitcast i64* %146 to <4 x i64>*
  %wide.load694.epil = load <4 x i64>, <4 x i64>* %147, align 8, !tbaa !100, !alias.scope !31541
  %148 = getelementptr i64, i64* %144, i64 8
  %149 = bitcast i64* %148 to <4 x i64>*
  %wide.load695.epil = load <4 x i64>, <4 x i64>* %149, align 8, !tbaa !100, !alias.scope !31541
  %150 = getelementptr i64, i64* %144, i64 12
  %151 = bitcast i64* %150 to <4 x i64>*
  %wide.load696.epil = load <4 x i64>, <4 x i64>* %151, align 8, !tbaa !100, !alias.scope !31541
  %152 = bitcast i64* %143 to <4 x i64>*
  store <4 x i64> %wide.load.epil, <4 x i64>* %152, align 8, !tbaa !100, !alias.scope !31545, !noalias !31541
  %153 = getelementptr i64, i64* %143, i64 4
  %154 = bitcast i64* %153 to <4 x i64>*
  store <4 x i64> %wide.load694.epil, <4 x i64>* %154, align 8, !tbaa !100, !alias.scope !31545, !noalias !31541
  %155 = getelementptr i64, i64* %143, i64 8
  %156 = bitcast i64* %155 to <4 x i64>*
  store <4 x i64> %wide.load695.epil, <4 x i64>* %156, align 8, !tbaa !100, !alias.scope !31545, !noalias !31541
  %157 = getelementptr i64, i64* %143, i64 12
  %158 = bitcast i64* %157 to <4 x i64>*
  store <4 x i64> %wide.load696.epil, <4 x i64>* %158, align 8, !tbaa !100, !alias.scope !31545, !noalias !31541
  %index.next.epil = add i64 %index.epil, 16
  %epil.iter.sub = add i64 %epil.iter, -1
  %epil.iter.cmp = icmp eq i64 %epil.iter.sub, 0
  br i1 %epil.iter.cmp, label %middle.block, label %vector.body.epil, !llvm.loop !31549

middle.block:                                     ; preds = %vector.body.epil, %middle.block.unr-lcssa
  %cmp.n = icmp eq i64 %cond, %n.vec
  br i1 %cmp.n, label %if.end146, label %for.body.preheader

for.body.preheader:                               ; preds = %middle.block, %vector.memcheck, %for.body.lr.ph
  %j133.0662.ph = phi i64 [ %mul127, %vector.memcheck ], [ %mul127, %for.body.lr.ph ], [ %ind.end, %middle.block ]
  %159 = add i64 %cond, %mul127
  %160 = sub i64 %159, %j133.0662.ph
  %161 = add i64 %mul127, -1
  %162 = add i64 %cond, %161
  %163 = sub i64 %162, %j133.0662.ph
  %xtraiter701 = and i64 %160, 7
  %lcmp.mod702 = icmp eq i64 %xtraiter701, 0
  br i1 %lcmp.mod702, label %for.body.prol.loopexit, label %for.body.prol.preheader

for.body.prol.preheader:                          ; preds = %for.body.preheader
  br label %for.body.prol

for.body.prol:                                    ; preds = %for.body.prol, %for.body.prol.preheader
  %j133.0662.prol = phi i64 [ %inc144.prol, %for.body.prol ], [ %j133.0662.ph, %for.body.prol.preheader ]
  %prol.iter = phi i64 [ %prol.iter.sub, %for.body.prol ], [ %xtraiter701, %for.body.prol.preheader ]
  %arrayidx137.prol = getelementptr inbounds i64, i64* %54, i64 %j133.0662.prol
  %arrayidx.i590.prol = getelementptr inbounds i64, i64* %65, i64 %j133.0662.prol
  %164 = load i64, i64* %arrayidx.i590.prol, align 8, !tbaa !100
  store i64 %164, i64* %arrayidx137.prol, align 8, !tbaa !100
  %inc144.prol = add i64 %j133.0662.prol, 1
  %prol.iter.sub = add i64 %prol.iter, -1
  %prol.iter.cmp = icmp eq i64 %prol.iter.sub, 0
  br i1 %prol.iter.cmp, label %for.body.prol.loopexit, label %for.body.prol, !llvm.loop !31551

for.body.prol.loopexit:                           ; preds = %for.body.prol, %for.body.preheader
  %j133.0662.unr = phi i64 [ %j133.0662.ph, %for.body.preheader ], [ %inc144.prol, %for.body.prol ]
  %165 = icmp ult i64 %163, 7
  br i1 %165, label %if.end146, label %for.body.preheader.new

for.body.preheader.new:                           ; preds = %for.body.prol.loopexit
  br label %for.body

for.body:                                         ; preds = %for.body, %for.body.preheader.new
  %j133.0662 = phi i64 [ %j133.0662.unr, %for.body.preheader.new ], [ %inc144.7, %for.body ]
  %arrayidx137 = getelementptr inbounds i64, i64* %54, i64 %j133.0662
  %arrayidx.i590 = getelementptr inbounds i64, i64* %65, i64 %j133.0662
  %166 = load i64, i64* %arrayidx.i590, align 8, !tbaa !100
  store i64 %166, i64* %arrayidx137, align 8, !tbaa !100
  %inc144 = add i64 %j133.0662, 1
  %arrayidx137.1 = getelementptr inbounds i64, i64* %54, i64 %inc144
  %arrayidx.i590.1 = getelementptr inbounds i64, i64* %65, i64 %inc144
  %167 = load i64, i64* %arrayidx.i590.1, align 8, !tbaa !100
  store i64 %167, i64* %arrayidx137.1, align 8, !tbaa !100
  %inc144.1 = add i64 %j133.0662, 2
  %arrayidx137.2 = getelementptr inbounds i64, i64* %54, i64 %inc144.1
  %arrayidx.i590.2 = getelementptr inbounds i64, i64* %65, i64 %inc144.1
  %168 = load i64, i64* %arrayidx.i590.2, align 8, !tbaa !100
  store i64 %168, i64* %arrayidx137.2, align 8, !tbaa !100
  %inc144.2 = add i64 %j133.0662, 3
  %arrayidx137.3 = getelementptr inbounds i64, i64* %54, i64 %inc144.2
  %arrayidx.i590.3 = getelementptr inbounds i64, i64* %65, i64 %inc144.2
  %169 = load i64, i64* %arrayidx.i590.3, align 8, !tbaa !100
  store i64 %169, i64* %arrayidx137.3, align 8, !tbaa !100
  %inc144.3 = add i64 %j133.0662, 4
  %arrayidx137.4 = getelementptr inbounds i64, i64* %54, i64 %inc144.3
  %arrayidx.i590.4 = getelementptr inbounds i64, i64* %65, i64 %inc144.3
  %170 = load i64, i64* %arrayidx.i590.4, align 8, !tbaa !100
  store i64 %170, i64* %arrayidx137.4, align 8, !tbaa !100
  %inc144.4 = add i64 %j133.0662, 5
  %arrayidx137.5 = getelementptr inbounds i64, i64* %54, i64 %inc144.4
  %arrayidx.i590.5 = getelementptr inbounds i64, i64* %65, i64 %inc144.4
  %171 = load i64, i64* %arrayidx.i590.5, align 8, !tbaa !100
  store i64 %171, i64* %arrayidx137.5, align 8, !tbaa !100
  %inc144.5 = add i64 %j133.0662, 6
  %arrayidx137.6 = getelementptr inbounds i64, i64* %54, i64 %inc144.5
  %arrayidx.i590.6 = getelementptr inbounds i64, i64* %65, i64 %inc144.5
  %172 = load i64, i64* %arrayidx.i590.6, align 8, !tbaa !100
  store i64 %172, i64* %arrayidx137.6, align 8, !tbaa !100
  %inc144.6 = add i64 %j133.0662, 7
  %arrayidx137.7 = getelementptr inbounds i64, i64* %54, i64 %inc144.6
  %arrayidx.i590.7 = getelementptr inbounds i64, i64* %65, i64 %inc144.6
  %173 = load i64, i64* %arrayidx.i590.7, align 8, !tbaa !100
  store i64 %173, i64* %arrayidx137.7, align 8, !tbaa !100
  %inc144.7 = add i64 %j133.0662, 8
  %exitcond.7 = icmp eq i64 %inc144.7, %66
  br i1 %exitcond.7, label %if.end146, label %for.body, !llvm.loop !31552

if.end146:                                        ; preds = %for.body.prol.loopexit, %for.body, %middle.block, %pfor.body126
  %add.ptr = getelementptr inbounds i64, i64* %54, i64 %mul127
  invoke void @_ZN4pbbs9quicksortIlSt4lessIlEEEvPT_mRKT0_(i64* %add.ptr, i64 %cond, %"struct.std::less"* nonnull dereferenceable(1) %f)
          to label %invoke.cont148 unwind label %lpad147

invoke.cont148:                                   ; preds = %if.end146
  %cmp.i591 = icmp eq i64 %cond, 0
  %or.cond.i = or i1 %31, %cmp.i591
  br i1 %or.cond.i, label %_ZN4pbbs9merge_seqIlSt4lessIlEmEEvPT_S4_PT1_mmT0_.exit, label %if.end.i593

if.end.i593:                                      ; preds = %invoke.cont148
  %mul150 = mul i64 %__begin113.0663, %add31
  %add.ptr151 = getelementptr inbounds i64, i64* %57, i64 %mul150
  call void @llvm.memset.p0i8.i64(i8* %scevgep, i8 0, i64 %61, i32 8, i1 false)
  %add.ptr.i592 = getelementptr inbounds i64, i64* %add.ptr, i64 %cond
  %.pre.i = load i64, i64* %add.ptr, align 8, !tbaa !100
  br label %while.cond.i

while.cond.i:                                     ; preds = %while.cond.i.backedge, %if.end.i593
  %174 = phi i64 [ %.pre.i, %if.end.i593 ], [ %.be, %while.cond.i.backedge ]
  %sC.addr.0.i = phi i64* [ %add.ptr151, %if.end.i593 ], [ %sC.addr.0.i.be, %while.cond.i.backedge ]
  %sB.addr.0.i = phi i64* [ %43, %if.end.i593 ], [ %sB.addr.0.i.be, %while.cond.i.backedge ]
  %sA.addr.0.i = phi i64* [ %add.ptr, %if.end.i593 ], [ %sA.addr.0.i.be, %while.cond.i.backedge ]
  %175 = load i64, i64* %sB.addr.0.i, align 8, !tbaa !100
  %cmp.i71.i = icmp slt i64 %174, %175
  br i1 %cmp.i71.i, label %while.body5.lr.ph.i, label %while.end.i

while.body5.lr.ph.i:                              ; preds = %while.cond.i
  %.pre82.i = load i64, i64* %sC.addr.0.i, align 8, !tbaa !100
  br label %while.body5.i

while.cond4.i:                                    ; preds = %while.body5.i
  %176 = load i64, i64* %incdec.ptr.i, align 8, !tbaa !100
  %cmp.i.i596 = icmp slt i64 %176, %175
  br i1 %cmp.i.i596, label %while.body5.i, label %while.end.i, !llvm.loop !31570

while.body5.i:                                    ; preds = %while.cond4.i, %while.body5.lr.ph.i
  %177 = phi i64 [ %.pre82.i, %while.body5.lr.ph.i ], [ %inc6.i, %while.cond4.i ]
  %sA.addr.172.i = phi i64* [ %sA.addr.0.i, %while.body5.lr.ph.i ], [ %incdec.ptr.i, %while.cond4.i ]
  %inc6.i = add i64 %177, 1
  store i64 %inc6.i, i64* %sC.addr.0.i, align 8, !tbaa !100
  %incdec.ptr.i = getelementptr inbounds i64, i64* %sA.addr.172.i, i64 1
  %cmp7.i = icmp eq i64* %incdec.ptr.i, %add.ptr.i592
  br i1 %cmp7.i, label %_ZN4pbbs9merge_seqIlSt4lessIlEmEEvPT_S4_PT1_mmT0_.exit, label %while.cond4.i

while.end.i:                                      ; preds = %while.cond4.i, %while.cond.i
  %178 = phi i64 [ %174, %while.cond.i ], [ %176, %while.cond4.i ]
  %sA.addr.1.lcssa.i = phi i64* [ %sA.addr.0.i, %while.cond.i ], [ %incdec.ptr.i, %while.cond4.i ]
  %incdec.ptr10.i = getelementptr inbounds i64, i64* %sB.addr.0.i, i64 1
  %incdec.ptr11.i = getelementptr inbounds i64, i64* %sC.addr.0.i, i64 1
  %cmp12.i = icmp eq i64* %incdec.ptr10.i, %add.ptr2.i
  br i1 %cmp12.i, label %while.end33.i, label %if.end14.i

if.end14.i:                                       ; preds = %while.end.i
  %179 = load i64, i64* %incdec.ptr10.i, align 8, !tbaa !100
  %cmp.i66.i = icmp slt i64 %175, %179
  br i1 %cmp.i66.i, label %while.cond.i.backedge, label %if.then17.i

if.then17.i:                                      ; preds = %if.end14.i
  %cmp.i6574.i = icmp slt i64 %179, %178
  br i1 %cmp.i6574.i, label %while.end26.i, label %while.body20.lr.ph.i

while.body20.lr.ph.i:                             ; preds = %if.then17.i
  %.pre83.i = load i64, i64* %incdec.ptr11.i, align 8, !tbaa !100
  br label %while.body20.i

while.cond18.i:                                   ; preds = %while.body20.i
  %180 = load i64, i64* %incdec.ptr22.i, align 8, !tbaa !100
  %cmp.i65.i = icmp slt i64 %179, %180
  br i1 %cmp.i65.i, label %while.end26.i, label %while.body20.i, !llvm.loop !31588

while.body20.i:                                   ; preds = %while.cond18.i, %while.body20.lr.ph.i
  %181 = phi i64 [ %.pre83.i, %while.body20.lr.ph.i ], [ %inc21.i, %while.cond18.i ]
  %sA.addr.275.i = phi i64* [ %sA.addr.1.lcssa.i, %while.body20.lr.ph.i ], [ %incdec.ptr22.i, %while.cond18.i ]
  %inc21.i = add i64 %181, 1
  store i64 %inc21.i, i64* %incdec.ptr11.i, align 8, !tbaa !100
  %incdec.ptr22.i = getelementptr inbounds i64, i64* %sA.addr.275.i, i64 1
  %cmp23.i = icmp eq i64* %incdec.ptr22.i, %add.ptr.i592
  br i1 %cmp23.i, label %_ZN4pbbs9merge_seqIlSt4lessIlEmEEvPT_S4_PT1_mmT0_.exit, label %while.cond18.i

while.end26.i:                                    ; preds = %while.cond18.i, %if.then17.i
  %182 = phi i64 [ %178, %if.then17.i ], [ %180, %while.cond18.i ]
  %sA.addr.2.lcssa.i = phi i64* [ %sA.addr.1.lcssa.i, %if.then17.i ], [ %incdec.ptr22.i, %while.cond18.i ]
  %incdec.ptr27.i = getelementptr inbounds i64, i64* %sB.addr.0.i, i64 2
  %incdec.ptr28.i = getelementptr inbounds i64, i64* %sC.addr.0.i, i64 2
  %cmp29.i = icmp eq i64* %incdec.ptr27.i, %add.ptr2.i
  br i1 %cmp29.i, label %while.end33.i, label %while.cond.i.backedge

while.cond.i.backedge:                            ; preds = %while.end26.i, %if.end14.i
  %.be = phi i64 [ %178, %if.end14.i ], [ %182, %while.end26.i ]
  %sC.addr.0.i.be = phi i64* [ %incdec.ptr11.i, %if.end14.i ], [ %incdec.ptr28.i, %while.end26.i ]
  %sB.addr.0.i.be = phi i64* [ %incdec.ptr10.i, %if.end14.i ], [ %incdec.ptr27.i, %while.end26.i ]
  %sA.addr.0.i.be = phi i64* [ %sA.addr.1.lcssa.i, %if.end14.i ], [ %sA.addr.2.lcssa.i, %while.end26.i ]
  br label %while.cond.i, !llvm.loop !31597

while.end33.i:                                    ; preds = %while.end26.i, %while.end.i
  %sC.addr.2.i = phi i64* [ %incdec.ptr11.i, %while.end.i ], [ %incdec.ptr28.i, %while.end26.i ]
  %sA.addr.4.i = phi i64* [ %sA.addr.1.lcssa.i, %while.end.i ], [ %sA.addr.2.lcssa.i, %while.end26.i ]
  %sub.ptr.lhs.cast.i = ptrtoint i64* %add.ptr.i592 to i64
  %sub.ptr.rhs.cast.i = ptrtoint i64* %sA.addr.4.i to i64
  %sub.ptr.sub.i597 = sub i64 %sub.ptr.lhs.cast.i, %sub.ptr.rhs.cast.i
  %sub.ptr.div.i598 = ashr exact i64 %sub.ptr.sub.i597, 3
  store i64 %sub.ptr.div.i598, i64* %sC.addr.2.i, align 8, !tbaa !100
  br label %_ZN4pbbs9merge_seqIlSt4lessIlEmEEvPT_S4_PT1_mmT0_.exit

_ZN4pbbs9merge_seqIlSt4lessIlEmEEvPT_S4_PT1_mmT0_.exit: ; preds = %while.body5.i, %while.body20.i, %invoke.cont148, %while.end33.i
  reattach within %syncreg111, label %pfor.inc159

pfor.inc159:                                      ; preds = %pfor.detach122, %_ZN4pbbs9merge_seqIlSt4lessIlEmEEvPT_S4_PT1_mmT0_.exit
  %inc160 = add nuw i64 %__begin113.0663, 1
  %cmp120 = icmp ult i64 %inc160, %conv27
  %indvars.iv.next = add i64 %indvars.iv, %add29
  br i1 %cmp120, label %pfor.detach122, label %pfor.cond.cleanup121, !llvm.loop !31604

lpad147:                                          ; preds = %if.end146
  %183 = landingpad { i8*, i32 }
          catch i8* null
  invoke void @llvm.detached.rethrow.sl_p0i8i32s(token %syncreg111, { i8*, i32 } %183)
          to label %det.rethrow.unreachable167 unwind label %lpad161.loopexit.split-lp

det.rethrow.unreachable167:                       ; preds = %lpad147
  unreachable

lpad161.loopexit:                                 ; preds = %pfor.detach122
  %lpad.loopexit654 = landingpad { i8*, i32 }
          cleanup
  br label %lpad161

lpad161.loopexit.split-lp:                        ; preds = %lpad147
  %lpad.loopexit.split-lp655 = landingpad { i8*, i32 }
          cleanup
  br label %lpad161

lpad161:                                          ; preds = %lpad161.loopexit.split-lp, %lpad161.loopexit
  %lpad.phi656 = phi { i8*, i32 } [ %lpad.loopexit654, %lpad161.loopexit ], [ %lpad.loopexit.split-lp655, %lpad161.loopexit.split-lp ]
  %184 = extractvalue { i8*, i32 } %lpad.phi656, 0
  %185 = extractvalue { i8*, i32 } %lpad.phi656, 1
  sync within %syncreg111, label %ehcleanup284

sync.continue168:                                 ; preds = %pfor.cond.cleanup121
  %186 = lshr i64 %sub.ptr.div.i, 3
  %add.i600 = shl i64 %186, 6
  %mul1.i601 = add i64 %add.i600, 64
  %call.i602 = tail call noalias i8* @aligned_alloc(i64 64, i64 %mul1.i601) #5
  %187 = bitcast i8* %call.i602 to i64*
  %cmp.i603 = icmp eq i8* %call.i602, null
  br i1 %cmp.i603, label %if.then.i604, label %if.end.i605

if.then.i604:                                     ; preds = %sync.continue168
  %188 = load %struct._IO_FILE*, %struct._IO_FILE** @stderr, align 8, !tbaa !213
  %189 = tail call i64 @fwrite(i8* getelementptr inbounds ([22 x i8], [22 x i8]* @.str.47, i64 0, i64 0), i64 21, i64 1, %struct._IO_FILE* %188) #20
  tail call void @exit(i32 1) #21
  unreachable

if.end.i605:                                      ; preds = %sync.continue168
  %cmp420.i606 = icmp eq i64 %mul1.i601, 0
  br i1 %cmp420.i606, label %pfor.cond.cleanup.i608, label %pfor.detach.i610.preheader

pfor.detach.i610.preheader:                       ; preds = %if.end.i605
  br label %pfor.detach.i610

pfor.cond.cleanup.i608:                           ; preds = %pfor.inc.i615, %if.end.i605
  sync within %syncreg.i599, label %invoke.cont176

pfor.detach.i610:                                 ; preds = %pfor.detach.i610.preheader, %pfor.inc.i615
  %i.021.i609 = phi i64 [ %add5.i613, %pfor.inc.i615 ], [ 0, %pfor.detach.i610.preheader ]
  detach within %syncreg.i599, label %pfor.body.i612, label %pfor.inc.i615

pfor.body.i612:                                   ; preds = %pfor.detach.i610
  %arrayidx.i611 = getelementptr inbounds i8, i8* %call.i602, i64 %i.021.i609
  store i8 0, i8* %arrayidx.i611, align 1, !tbaa !218
  reattach within %syncreg.i599, label %pfor.inc.i615

pfor.inc.i615:                                    ; preds = %pfor.body.i612, %pfor.detach.i610
  %add5.i613 = add i64 %i.021.i609, 2097152
  %cmp4.i614 = icmp ult i64 %add5.i613, %mul1.i601
  br i1 %cmp4.i614, label %pfor.detach.i610, label %pfor.cond.cleanup.i608, !llvm.loop !4745

invoke.cont176:                                   ; preds = %pfor.cond.cleanup.i608
  %call180 = invoke i64* @_ZN4pbbs17transpose_bucketsIlmEEPmPT_S3_PT0_mmmm(i64* %54, i64* %187, i64* %57, i64 %sub.ptr.div.i, i64 %add29, i64 %conv27, i64 %add31)
          to label %invoke.cont179 unwind label %lpad178

invoke.cont179:                                   ; preds = %invoke.cont176
  tail call void @free(i8* %call.i575) #5
  br label %pfor.detach192

pfor.cond.cleanup191:                             ; preds = %pfor.inc229
  sync within %syncreg181, label %sync.continue238

lpad178:                                          ; preds = %invoke.cont176
  %190 = landingpad { i8*, i32 }
          cleanup
  %191 = extractvalue { i8*, i32 } %190, 0
  %192 = extractvalue { i8*, i32 } %190, 1
  br label %ehcleanup284

pfor.detach192:                                   ; preds = %pfor.inc229, %invoke.cont179
  %__begin183.0660 = phi i64 [ 0, %invoke.cont179 ], [ %inc230, %pfor.inc229 ]
  detach within %syncreg181, label %pfor.body197, label %pfor.inc229 unwind label %lpad231.loopexit

pfor.body197:                                     ; preds = %pfor.detach192
  %arrayidx198 = getelementptr inbounds i64, i64* %call180, i64 %__begin183.0660
  %193 = load i64, i64* %arrayidx198, align 8, !tbaa !100
  %add199 = add nuw nsw i64 %__begin183.0660, 1
  %arrayidx200 = getelementptr inbounds i64, i64* %call180, i64 %add199
  %194 = load i64, i64* %arrayidx200, align 8, !tbaa !100
  %cmp201 = icmp eq i64 %__begin183.0660, 0
  %cmp203 = icmp eq i64 %__begin183.0660, %div
  %or.cond = or i1 %cmp201, %cmp203
  br i1 %or.cond, label %if.then213, label %lor.lhs.false204

lor.lhs.false204:                                 ; preds = %pfor.body197
  %sub205 = add nsw i64 %__begin183.0660, -1
  %arrayidx206 = getelementptr inbounds i64, i64* %43, i64 %sub205
  %arrayidx207 = getelementptr inbounds i64, i64* %43, i64 %__begin183.0660
  %195 = load i64, i64* %arrayidx206, align 8, !tbaa !100
  %196 = load i64, i64* %arrayidx207, align 8, !tbaa !100
  %cmp.i501 = icmp slt i64 %195, %196
  br i1 %cmp.i501, label %if.then213, label %if.end217

if.then213:                                       ; preds = %lor.lhs.false204, %pfor.body197
  %add.ptr214 = getelementptr inbounds i64, i64* %187, i64 %193
  %sub215 = sub i64 %194, %193
  invoke void @_ZN4pbbs9quicksortIlSt4lessIlEEEvPT_mRKT0_(i64* %add.ptr214, i64 %sub215, %"struct.std::less"* nonnull dereferenceable(1) %f)
          to label %if.end217 unwind label %lpad208

lpad208:                                          ; preds = %if.then213
  %197 = landingpad { i8*, i32 }
          catch i8* null
  invoke void @llvm.detached.rethrow.sl_p0i8i32s(token %syncreg181, { i8*, i32 } %197)
          to label %det.rethrow.unreachable237 unwind label %lpad231.loopexit.split-lp

det.rethrow.unreachable237:                       ; preds = %lpad208
  unreachable

if.end217:                                        ; preds = %if.then213, %lor.lhs.false204
  br i1 %inplace, label %if.then219, label %if.end224

if.then219:                                       ; preds = %if.end217
  %add.ptr220 = getelementptr inbounds i64, i64* %54, i64 %193
  %198 = bitcast i64* %add.ptr220 to i8*
  %add.ptr221 = getelementptr inbounds i64, i64* %187, i64 %193
  %199 = bitcast i64* %add.ptr221 to i8*
  %sub222 = sub i64 %194, %193
  %mul223 = shl i64 %sub222, 3
  tail call void @llvm.memcpy.p0i8.p0i8.i64(i8* %198, i8* %199, i64 %mul223, i32 1, i1 false)
  br label %if.end224

if.end224:                                        ; preds = %if.then219, %if.end217
  reattach within %syncreg181, label %pfor.inc229

pfor.inc229:                                      ; preds = %pfor.detach192, %if.end224
  %inc230 = add nuw nsw i64 %__begin183.0660, 1
  %cmp190 = icmp ult i64 %__begin183.0660, %div
  br i1 %cmp190, label %pfor.detach192, label %pfor.cond.cleanup191, !llvm.loop !31647

lpad231.loopexit:                                 ; preds = %pfor.detach192
  %lpad.loopexit = landingpad { i8*, i32 }
          cleanup
  br label %lpad231

lpad231.loopexit.split-lp:                        ; preds = %lpad208
  %lpad.loopexit.split-lp = landingpad { i8*, i32 }
          cleanup
  br label %lpad231

lpad231:                                          ; preds = %lpad231.loopexit.split-lp, %lpad231.loopexit
  %lpad.phi = phi { i8*, i32 } [ %lpad.loopexit, %lpad231.loopexit ], [ %lpad.loopexit.split-lp, %lpad231.loopexit.split-lp ]
  %200 = extractvalue { i8*, i32 } %lpad.phi, 0
  %201 = extractvalue { i8*, i32 } %lpad.phi, 1
  sync within %syncreg181, label %ehcleanup284

sync.continue238:                                 ; preds = %pfor.cond.cleanup191
  tail call void @free(i8* %call.i.i512) #5
  %202 = bitcast i64* %call180 to i8*
  tail call void @free(i8* %202) #5
  br i1 %inplace, label %if.then248, label %if.else250

if.then248:                                       ; preds = %sync.continue238
  tail call void @free(i8* %call.i602) #5
  %203 = bitcast %struct.sequence.46* %agg.result to i64*
  store i64 %Bs.sroa.0.0, i64* %203, align 8, !tbaa !4929
  %e.i455 = getelementptr inbounds %struct.sequence.46, %struct.sequence.46* %agg.result, i64 0, i32 1
  %204 = bitcast i64** %e.i455 to i64*
  store i64 %Bs.sroa.16.0, i64* %204, align 8, !tbaa !4932
  %allocated.i456 = getelementptr inbounds %struct.sequence.46, %struct.sequence.46* %agg.result, i64 0, i32 2
  store i8 0, i8* %allocated.i456, align 8, !tbaa !4934
  br label %cleanup283

if.else250:                                       ; preds = %sync.continue238
  %205 = inttoptr i64 %Bs.sroa.0.0 to i8*
  tail call void @free(i8* %205) #5
  %206 = bitcast %struct.sequence.46* %agg.result to i8**
  store i8* %call.i602, i8** %206, align 8, !tbaa !4929
  %e.i454 = getelementptr inbounds %struct.sequence.46, %struct.sequence.46* %agg.result, i64 0, i32 1
  %add.ptr.i = getelementptr inbounds i64, i64* %187, i64 %sub.ptr.div.i
  store i64* %add.ptr.i, i64** %e.i454, align 8, !tbaa !4932
  %allocated3.i = getelementptr inbounds %struct.sequence.46, %struct.sequence.46* %agg.result, i64 0, i32 2
  store i8 1, i8* %allocated3.i, align 8, !tbaa !4934
  br label %cleanup283

cleanup283:                                       ; preds = %if.else250, %if.then248, %if.end
  ret void

ehcleanup284:                                     ; preds = %lpad161, %lpad178, %lpad231, %lpad49, %_ZN8sequenceIlED2Ev.exit497
  %ehselector.slot.10 = phi i32 [ %ehselector.slot.2675, %_ZN8sequenceIlED2Ev.exit497 ], [ %39, %lpad49 ], [ %185, %lpad161 ], [ %192, %lpad178 ], [ %201, %lpad231 ]
  %exn.slot.10 = phi i8* [ %exn.slot.2676, %_ZN8sequenceIlED2Ev.exit497 ], [ %38, %lpad49 ], [ %184, %lpad161 ], [ %191, %lpad178 ], [ %200, %lpad231 ]
  %lpad.val288 = insertvalue { i8*, i32 } undef, i8* %exn.slot.10, 0
  %lpad.val289 = insertvalue { i8*, i32 } %lpad.val288, i32 %ehselector.slot.10, 1
  resume { i8*, i32 } %lpad.val289
}

attributes #0 = { nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="haswell" "target-features"="+aes,+avx,+avx2,+bmi,+bmi2,+cmov,+cx16,+f16c,+fma,+fsgsbase,+fxsr,+lzcnt,+mmx,+movbe,+pclmul,+popcnt,+rdrnd,+sse,+sse2,+sse3,+sse4.1,+sse4.2,+ssse3,+x87,+xsave,+xsaveopt,-adx,-avx512bitalg,-avx512bw,-avx512cd,-avx512dq,-avx512er,-avx512f,-avx512ifma,-avx512pf,-avx512vbmi,-avx512vbmi2,-avx512vl,-avx512vnni,-avx512vpopcntdq,-clflushopt,-clwb,-clzero,-fma4,-gfni,-ibt,-lwp,-mwaitx,-pku,-prefetchwt1,-prfchw,-rdseed,-rtm,-sgx,-sha,-shstk,-sse4a,-tbm,-vaes,-vpclmulqdq,-xop,-xsavec,-xsaves" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="haswell" "target-features"="+aes,+avx,+avx2,+bmi,+bmi2,+cmov,+cx16,+f16c,+fma,+fsgsbase,+fxsr,+lzcnt,+mmx,+movbe,+pclmul,+popcnt,+rdrnd,+sse,+sse2,+sse3,+sse4.1,+sse4.2,+ssse3,+x87,+xsave,+xsaveopt,-adx,-avx512bitalg,-avx512bw,-avx512cd,-avx512dq,-avx512er,-avx512f,-avx512ifma,-avx512pf,-avx512vbmi,-avx512vbmi2,-avx512vl,-avx512vnni,-avx512vpopcntdq,-clflushopt,-clwb,-clzero,-fma4,-gfni,-ibt,-lwp,-mwaitx,-pku,-prefetchwt1,-prfchw,-rdseed,-rtm,-sgx,-sha,-shstk,-sse4a,-tbm,-vaes,-vpclmulqdq,-xop,-xsavec,-xsaves" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { norecurse nounwind readnone uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="haswell" "target-features"="+aes,+avx,+avx2,+bmi,+bmi2,+cmov,+cx16,+f16c,+fma,+fsgsbase,+fxsr,+lzcnt,+mmx,+movbe,+pclmul,+popcnt,+rdrnd,+sse,+sse2,+sse3,+sse4.1,+sse4.2,+ssse3,+x87,+xsave,+xsaveopt,-adx,-avx512bitalg,-avx512bw,-avx512cd,-avx512dq,-avx512er,-avx512f,-avx512ifma,-avx512pf,-avx512vbmi,-avx512vbmi2,-avx512vl,-avx512vnni,-avx512vpopcntdq,-clflushopt,-clwb,-clzero,-fma4,-gfni,-ibt,-lwp,-mwaitx,-pku,-prefetchwt1,-prfchw,-rdseed,-rtm,-sgx,-sha,-shstk,-sse4a,-tbm,-vaes,-vpclmulqdq,-xop,-xsavec,-xsaves" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #3 = { argmemonly nounwind }
attributes #4 = { "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="haswell" "target-features"="+aes,+avx,+avx2,+bmi,+bmi2,+cmov,+cx16,+f16c,+fma,+fsgsbase,+fxsr,+lzcnt,+mmx,+movbe,+pclmul,+popcnt,+rdrnd,+sse,+sse2,+sse3,+sse4.1,+sse4.2,+ssse3,+x87,+xsave,+xsaveopt,-adx,-avx512bitalg,-avx512bw,-avx512cd,-avx512dq,-avx512er,-avx512f,-avx512ifma,-avx512pf,-avx512vbmi,-avx512vbmi2,-avx512vl,-avx512vnni,-avx512vpopcntdq,-clflushopt,-clwb,-clzero,-fma4,-gfni,-ibt,-lwp,-mwaitx,-pku,-prefetchwt1,-prfchw,-rdseed,-rtm,-sgx,-sha,-shstk,-sse4a,-tbm,-vaes,-vpclmulqdq,-xop,-xsavec,-xsaves" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #5 = { nounwind }
attributes #6 = { uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="haswell" "target-features"="+aes,+avx,+avx2,+bmi,+bmi2,+cmov,+cx16,+f16c,+fma,+fsgsbase,+fxsr,+lzcnt,+mmx,+movbe,+pclmul,+popcnt,+rdrnd,+sse,+sse2,+sse3,+sse4.1,+sse4.2,+ssse3,+x87,+xsave,+xsaveopt,-adx,-avx512bitalg,-avx512bw,-avx512cd,-avx512dq,-avx512er,-avx512f,-avx512ifma,-avx512pf,-avx512vbmi,-avx512vbmi2,-avx512vl,-avx512vnni,-avx512vpopcntdq,-clflushopt,-clwb,-clzero,-fma4,-gfni,-ibt,-lwp,-mwaitx,-pku,-prefetchwt1,-prfchw,-rdseed,-rtm,-sgx,-sha,-shstk,-sse4a,-tbm,-vaes,-vpclmulqdq,-xop,-xsavec,-xsaves" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #7 = { inlinehint nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="haswell" "target-features"="+aes,+avx,+avx2,+bmi,+bmi2,+cmov,+cx16,+f16c,+fma,+fsgsbase,+fxsr,+lzcnt,+mmx,+movbe,+pclmul,+popcnt,+rdrnd,+sse,+sse2,+sse3,+sse4.1,+sse4.2,+ssse3,+x87,+xsave,+xsaveopt,-adx,-avx512bitalg,-avx512bw,-avx512cd,-avx512dq,-avx512er,-avx512f,-avx512ifma,-avx512pf,-avx512vbmi,-avx512vbmi2,-avx512vl,-avx512vnni,-avx512vpopcntdq,-clflushopt,-clwb,-clzero,-fma4,-gfni,-ibt,-lwp,-mwaitx,-pku,-prefetchwt1,-prfchw,-rdseed,-rtm,-sgx,-sha,-shstk,-sse4a,-tbm,-vaes,-vpclmulqdq,-xop,-xsavec,-xsaves" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #8 = { norecurse nounwind readonly uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="haswell" "target-features"="+aes,+avx,+avx2,+bmi,+bmi2,+cmov,+cx16,+f16c,+fma,+fsgsbase,+fxsr,+lzcnt,+mmx,+movbe,+pclmul,+popcnt,+rdrnd,+sse,+sse2,+sse3,+sse4.1,+sse4.2,+ssse3,+x87,+xsave,+xsaveopt,-adx,-avx512bitalg,-avx512bw,-avx512cd,-avx512dq,-avx512er,-avx512f,-avx512ifma,-avx512pf,-avx512vbmi,-avx512vbmi2,-avx512vl,-avx512vnni,-avx512vpopcntdq,-clflushopt,-clwb,-clzero,-fma4,-gfni,-ibt,-lwp,-mwaitx,-pku,-prefetchwt1,-prfchw,-rdseed,-rtm,-sgx,-sha,-shstk,-sse4a,-tbm,-vaes,-vpclmulqdq,-xop,-xsavec,-xsaves" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #9 = { noreturn nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="haswell" "target-features"="+aes,+avx,+avx2,+bmi,+bmi2,+cmov,+cx16,+f16c,+fma,+fsgsbase,+fxsr,+lzcnt,+mmx,+movbe,+pclmul,+popcnt,+rdrnd,+sse,+sse2,+sse3,+sse4.1,+sse4.2,+ssse3,+x87,+xsave,+xsaveopt,-adx,-avx512bitalg,-avx512bw,-avx512cd,-avx512dq,-avx512er,-avx512f,-avx512ifma,-avx512pf,-avx512vbmi,-avx512vbmi2,-avx512vl,-avx512vnni,-avx512vpopcntdq,-clflushopt,-clwb,-clzero,-fma4,-gfni,-ibt,-lwp,-mwaitx,-pku,-prefetchwt1,-prfchw,-rdseed,-rtm,-sgx,-sha,-shstk,-sse4a,-tbm,-vaes,-vpclmulqdq,-xop,-xsavec,-xsaves" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #10 = { norecurse uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="haswell" "target-features"="+aes,+avx,+avx2,+bmi,+bmi2,+cmov,+cx16,+f16c,+fma,+fsgsbase,+fxsr,+lzcnt,+mmx,+movbe,+pclmul,+popcnt,+rdrnd,+sse,+sse2,+sse3,+sse4.1,+sse4.2,+ssse3,+x87,+xsave,+xsaveopt,-adx,-avx512bitalg,-avx512bw,-avx512cd,-avx512dq,-avx512er,-avx512f,-avx512ifma,-avx512pf,-avx512vbmi,-avx512vbmi2,-avx512vl,-avx512vnni,-avx512vpopcntdq,-clflushopt,-clwb,-clzero,-fma4,-gfni,-ibt,-lwp,-mwaitx,-pku,-prefetchwt1,-prfchw,-rdseed,-rtm,-sgx,-sha,-shstk,-sse4a,-tbm,-vaes,-vpclmulqdq,-xop,-xsavec,-xsaves" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #11 = { noreturn "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="haswell" "target-features"="+aes,+avx,+avx2,+bmi,+bmi2,+cmov,+cx16,+f16c,+fma,+fsgsbase,+fxsr,+lzcnt,+mmx,+movbe,+pclmul,+popcnt,+rdrnd,+sse,+sse2,+sse3,+sse4.1,+sse4.2,+ssse3,+x87,+xsave,+xsaveopt,-adx,-avx512bitalg,-avx512bw,-avx512cd,-avx512dq,-avx512er,-avx512f,-avx512ifma,-avx512pf,-avx512vbmi,-avx512vbmi2,-avx512vl,-avx512vnni,-avx512vpopcntdq,-clflushopt,-clwb,-clzero,-fma4,-gfni,-ibt,-lwp,-mwaitx,-pku,-prefetchwt1,-prfchw,-rdseed,-rtm,-sgx,-sha,-shstk,-sse4a,-tbm,-vaes,-vpclmulqdq,-xop,-xsavec,-xsaves" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #12 = { nobuiltin nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="haswell" "target-features"="+aes,+avx,+avx2,+bmi,+bmi2,+cmov,+cx16,+f16c,+fma,+fsgsbase,+fxsr,+lzcnt,+mmx,+movbe,+pclmul,+popcnt,+rdrnd,+sse,+sse2,+sse3,+sse4.1,+sse4.2,+ssse3,+x87,+xsave,+xsaveopt,-adx,-avx512bitalg,-avx512bw,-avx512cd,-avx512dq,-avx512er,-avx512f,-avx512ifma,-avx512pf,-avx512vbmi,-avx512vbmi2,-avx512vl,-avx512vnni,-avx512vpopcntdq,-clflushopt,-clwb,-clzero,-fma4,-gfni,-ibt,-lwp,-mwaitx,-pku,-prefetchwt1,-prfchw,-rdseed,-rtm,-sgx,-sha,-shstk,-sse4a,-tbm,-vaes,-vpclmulqdq,-xop,-xsavec,-xsaves" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #13 = { argmemonly }
attributes #14 = { nounwind readnone speculatable }
attributes #15 = { nobuiltin "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="haswell" "target-features"="+aes,+avx,+avx2,+bmi,+bmi2,+cmov,+cx16,+f16c,+fma,+fsgsbase,+fxsr,+lzcnt,+mmx,+movbe,+pclmul,+popcnt,+rdrnd,+sse,+sse2,+sse3,+sse4.1,+sse4.2,+ssse3,+x87,+xsave,+xsaveopt,-adx,-avx512bitalg,-avx512bw,-avx512cd,-avx512dq,-avx512er,-avx512f,-avx512ifma,-avx512pf,-avx512vbmi,-avx512vbmi2,-avx512vl,-avx512vnni,-avx512vpopcntdq,-clflushopt,-clwb,-clzero,-fma4,-gfni,-ibt,-lwp,-mwaitx,-pku,-prefetchwt1,-prfchw,-rdseed,-rtm,-sgx,-sha,-shstk,-sse4a,-tbm,-vaes,-vpclmulqdq,-xop,-xsavec,-xsaves" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #16 = { inlinehint norecurse uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="haswell" "target-features"="+aes,+avx,+avx2,+bmi,+bmi2,+cmov,+cx16,+f16c,+fma,+fsgsbase,+fxsr,+lzcnt,+mmx,+movbe,+pclmul,+popcnt,+rdrnd,+sse,+sse2,+sse3,+sse4.1,+sse4.2,+ssse3,+x87,+xsave,+xsaveopt,-adx,-avx512bitalg,-avx512bw,-avx512cd,-avx512dq,-avx512er,-avx512f,-avx512ifma,-avx512pf,-avx512vbmi,-avx512vbmi2,-avx512vl,-avx512vnni,-avx512vpopcntdq,-clflushopt,-clwb,-clzero,-fma4,-gfni,-ibt,-lwp,-mwaitx,-pku,-prefetchwt1,-prfchw,-rdseed,-rtm,-sgx,-sha,-shstk,-sse4a,-tbm,-vaes,-vpclmulqdq,-xop,-xsavec,-xsaves" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #17 = { norecurse nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="haswell" "target-features"="+aes,+avx,+avx2,+bmi,+bmi2,+cmov,+cx16,+f16c,+fma,+fsgsbase,+fxsr,+lzcnt,+mmx,+movbe,+pclmul,+popcnt,+rdrnd,+sse,+sse2,+sse3,+sse4.1,+sse4.2,+ssse3,+x87,+xsave,+xsaveopt,-adx,-avx512bitalg,-avx512bw,-avx512cd,-avx512dq,-avx512er,-avx512f,-avx512ifma,-avx512pf,-avx512vbmi,-avx512vbmi2,-avx512vl,-avx512vnni,-avx512vpopcntdq,-clflushopt,-clwb,-clzero,-fma4,-gfni,-ibt,-lwp,-mwaitx,-pku,-prefetchwt1,-prfchw,-rdseed,-rtm,-sgx,-sha,-shstk,-sse4a,-tbm,-vaes,-vpclmulqdq,-xop,-xsavec,-xsaves" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #18 = { inlinehint uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="haswell" "target-features"="+aes,+avx,+avx2,+bmi,+bmi2,+cmov,+cx16,+f16c,+fma,+fsgsbase,+fxsr,+lzcnt,+mmx,+movbe,+pclmul,+popcnt,+rdrnd,+sse,+sse2,+sse3,+sse4.1,+sse4.2,+ssse3,+x87,+xsave,+xsaveopt,-adx,-avx512bitalg,-avx512bw,-avx512cd,-avx512dq,-avx512er,-avx512f,-avx512ifma,-avx512pf,-avx512vbmi,-avx512vbmi2,-avx512vl,-avx512vnni,-avx512vpopcntdq,-clflushopt,-clwb,-clzero,-fma4,-gfni,-ibt,-lwp,-mwaitx,-pku,-prefetchwt1,-prfchw,-rdseed,-rtm,-sgx,-sha,-shstk,-sse4a,-tbm,-vaes,-vpclmulqdq,-xop,-xsavec,-xsaves" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #19 = { noreturn }
attributes #20 = { cold }
attributes #21 = { noreturn nounwind }
attributes #22 = { builtin }
attributes #23 = { builtin nounwind }

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "clang version 6.0.0 (git@github.com:wsmoses/Tapir-Clang.git 96fbe7006d96197be05a8c45720a2b1d281e1678) (git@github.com:wsmoses/Tapir-LLVM.git 3eb305d6fa3e479ea0faff1fae6c95a5991a0ed4)", isOptimized: true, runtimeVersion: 0, emissionKind: NoDebug, enums: !2)
!1 = !DIFile(filename: "time_tests.cpp", directory: "/home/neboat/tmp/pbbslib/lib")
!2 = !{}
!7 = !DIFile(filename: "./utilities.h", directory: "/home/neboat/tmp/pbbslib/lib")
!8 = !DISubroutineType(types: !2)
!54 = !{!"omnipotent char", !55, i64 0}
!55 = !{!"Simple C++ TBAA"}
!56 = !{!"bool", !54, i64 0}
!57 = !{!"_ZTSNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE", !58, i64 0, !60, i64 8, !54, i64 16}
!58 = !{!"_ZTSNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE12_Alloc_hiderE", !59, i64 0}
!59 = !{!"any pointer", !54, i64 0}
!60 = !{!"long", !54, i64 0}
!61 = !{!"_ZTS8timezone", !62, i64 0, !62, i64 4}
!62 = !{!"int", !54, i64 0}
!100 = !{!60, !60, i64 0}
!202 = distinct !DISubprogram(name: "new_array_no_init<unsigned long>", scope: !7, file: !7, line: 108, type: !8, isLocal: false, isDefinition: true, scopeLine: 108, flags: DIFlagPrototyped, isOptimized: true, unit: !0, variables: !2)
!205 = !DIFile(filename: "./seq.h", directory: "/home/neboat/tmp/pbbslib/lib")
!213 = !{!59, !59, i64 0}
!218 = !{!56, !56, i64 0}
!221 = distinct !{!221, !222, !223, !224}
!222 = !DILocation(line: 116, column: 7, scope: !202)
!223 = !DILocation(line: 117, column: 19, scope: !202)
!224 = !{!"tapir.loop.spawn.strategy", i32 1}
!312 = !{!"llvm.loop.unroll.disable"}
!553 = distinct !DISubprogram(name: "log2_up<unsigned long>", scope: !7, file: !7, line: 177, type: !8, isLocal: true, isDefinition: true, scopeLine: 177, flags: DIFlagPrototyped, isOptimized: true, unit: !0, variables: !2)
!559 = distinct !{!559, !560, !561}
!560 = !DILocation(line: 180, column: 5, scope: !553)
!561 = !DILocation(line: 180, column: 36, scope: !553)
!762 = !{!"llvm.loop.isvectorized", i32 1}
!4728 = distinct !DISubprogram(name: "new_array_no_init<long>", scope: !7, file: !7, line: 108, type: !8, isLocal: false, isDefinition: true, scopeLine: 108, flags: DIFlagPrototyped, isOptimized: true, unit: !0, variables: !2)
!4745 = distinct !{!4745, !4746, !4747, !224}
!4746 = !DILocation(line: 116, column: 7, scope: !4728)
!4747 = !DILocation(line: 117, column: 19, scope: !4728)
!4929 = !{!4930, !59, i64 0}
!4930 = !{!"_ZTS8sequenceIlE", !59, i64 0, !59, i64 8, !56, i64 16}
!4932 = !{!4930, !59, i64 8}
!4934 = !{!4930, !56, i64 16}
!30951 = !DIFile(filename: "./sample_sort.h", directory: "/home/neboat/tmp/pbbslib/lib")
!31340 = distinct !DISubprogram(name: "sample_sort_<unsigned long, sequence<long>, std::less<long> >", scope: !30951, file: !30951, line: 74, type: !8, isLocal: false, isDefinition: true, scopeLine: 75, flags: DIFlagPrototyped, isOptimized: true, unit: !0, variables: !2)
!31361 = distinct !DISubprogram(name: "sequence<(lambda at ./sample_sort.h:81:16)>", scope: !205, file: !205, line: 46, type: !8, isLocal: false, isDefinition: true, scopeLine: 47, flags: DIFlagPrototyped, isOptimized: true, unit: !0, variables: !2)
!31381 = distinct !{!31381, !31382, !31383, !224}
!31382 = !DILocation(line: 49, column: 5, scope: !31361)
!31383 = !DILocation(line: 52, column: 5, scope: !31361)
!31437 = !DILocation(line: 105, column: 7, scope: !31340)
!31460 = distinct !{!31460, !31437, !31461, !224}
!31461 = !DILocation(line: 106, column: 31, scope: !31340)
!31474 = !DILocation(line: 113, column: 7, scope: !31340)
!31479 = distinct !{!31479, !31474, !31480, !224}
!31480 = !DILocation(line: 114, column: 38, scope: !31340)
!31486 = !{!31487}
!31487 = distinct !{!31487, !31488, !"_ZN8sequenceIlE11as_sequenceEv: %agg.result"}
!31488 = distinct !{!31488, !"_ZN8sequenceIlE11as_sequenceEv"}
!31527 = !DILocation(line: 126, column: 7, scope: !31340)
!31535 = !DILocation(line: 130, column: 4, scope: !31340)
!31541 = !{!31542}
!31542 = distinct !{!31542, !31543}
!31543 = distinct !{!31543, !"LVerDomain"}
!31545 = !{!31546}
!31546 = distinct !{!31546, !31543}
!31547 = distinct !{!31547, !31535, !31548, !762}
!31548 = !DILocation(line: 131, column: 37, scope: !31340)
!31549 = distinct !{!31549, !312}
!31551 = distinct !{!31551, !312}
!31552 = distinct !{!31552, !31535, !31548, !762}
!31570 = distinct !{!31570, !31571, !31572}
!31556 = distinct !DISubprogram(name: "merge_seq<long, std::less<long>, unsigned long>", scope: !30951, file: !30951, line: 54, type: !8, isLocal: false, isDefinition: true, scopeLine: 55, flags: DIFlagPrototyped, isOptimized: true, unit: !0, variables: !2)
!31571 = !DILocation(line: 61, column: 7, scope: !31556)
!31572 = !DILocation(line: 61, column: 60, scope: !31556)
!31588 = distinct !{!31588, !31589, !31590}
!31589 = !DILocation(line: 65, column: 2, scope: !31556)
!31590 = !DILocation(line: 65, column: 56, scope: !31556)
!31597 = distinct !{!31597, !31598, !31599}
!31598 = !DILocation(line: 60, column: 5, scope: !31556)
!31599 = !DILocation(line: 69, column: 5, scope: !31556)
!31603 = !DILocation(line: 135, column: 7, scope: !31340)
!31604 = distinct !{!31604, !31527, !31603, !224}
!31622 = !DILocation(line: 146, column: 7, scope: !31340)
!31646 = !DILocation(line: 158, column: 7, scope: !31340)
!31647 = distinct !{!31647, !31622, !31646, !224}
