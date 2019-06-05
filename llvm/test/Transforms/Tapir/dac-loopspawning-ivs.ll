; Test correct IV updating in loop spawning pass.
;
; Credit to Guy Blelloch for providing the original source code for
; this test.

; RUN: opt < %s -loop-spawning-ti -S | FileCheck %s
; RUN: opt < %s -passes='loop-spawning' -S | FileCheck %s
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct._IO_FILE = type { i32, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, %struct._IO_marker*, %struct._IO_FILE*, i32, i32, i64, i16, i8, [1 x i8], i8*, i64, i8*, i8*, i8*, i8*, i64, i32, [20 x i8] }
%struct._IO_marker = type { %struct._IO_marker*, %struct._IO_FILE*, i32 }
%"struct.std::less" = type { i8 }
%struct.sequence.46 = type <{ i64*, i64*, i8, [7 x i8] }>

$_ZN4pbbs12sample_sort_Im8sequenceIlESt4lessIlEEES1_INT0_1TEES5_RKT1_b = comdat any

@.str.47 = private unnamed_addr constant [22 x i8] c"Cannot allocate space\00", align 1
@stderr = external local_unnamed_addr global %struct._IO_FILE*, align 8

; Function Attrs: uwtable
declare i64* @_ZN4pbbs17transpose_bucketsIlmEEPmPT_S3_PT0_mmmm(i64*, i64*, i64*, i64, i64, i64, i64) #0

; Function Attrs: uwtable
declare void @_ZN4pbbs9quicksortIlSt4lessIlEEEvPT_mRKT0_(i64*, i64, %"struct.std::less"* dereferenceable(1)) #0

declare i32 @__gxx_personality_v0(...)

; Function Attrs: argmemonly nounwind
declare void @llvm.lifetime.start.p0i8(i64, i8* nocapture) #1

; Function Attrs: argmemonly nounwind
declare void @llvm.lifetime.end.p0i8(i64, i8* nocapture) #1

; Function Attrs: argmemonly nounwind
declare token @llvm.syncregion.start() #1

; Function Attrs: nounwind
declare noalias i8* @aligned_alloc(i64, i64) local_unnamed_addr #2

; Function Attrs: nounwind
declare void @free(i8* nocapture) local_unnamed_addr #2

; Function Attrs: noreturn nounwind
declare void @exit(i32) local_unnamed_addr #3

; Function Attrs: nounwind
declare i64 @fwrite(i8* nocapture, i64, i64, %struct._IO_FILE* nocapture) local_unnamed_addr #4

; Function Attrs: nounwind readnone speculatable
declare double @llvm.ceil.f64(double) #5

; Function Attrs: argmemonly
declare void @llvm.detached.rethrow.sl_p0i8i32s(token, { i8*, i32 }) #6

; Function Attrs: nounwind readnone speculatable
declare double @llvm.fabs.f64(double) #5

; Function Attrs: argmemonly nounwind
declare void @llvm.memcpy.p0i8.p0i8.i64(i8* nocapture writeonly, i8* nocapture readonly, i64, i32, i1) #1

; Function Attrs: argmemonly nounwind
declare void @llvm.memset.p0i8.i64(i8* nocapture writeonly, i8, i64, i32, i1) #1

declare double @sqrt(double) local_unnamed_addr

; Function Attrs: uwtable
define linkonce_odr void @_ZN4pbbs12sample_sort_Im8sequenceIlESt4lessIlEEES1_INT0_1TEES5_RKT1_b(%struct.sequence.46* noalias sret %agg.result, %struct.sequence.46* %A, %"struct.std::less"* dereferenceable(1) %f, i1 zeroext %inplace) local_unnamed_addr #0 comdat personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*) {
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
  %1 = load i64, i64* %0, align 8, !tbaa !0
  %2 = bitcast %struct.sequence.46* %A to i64*
  %3 = load i64, i64* %2, align 8, !tbaa !6
  %sub.ptr.sub.i = sub i64 %1, %3
  %sub.ptr.div.i = ashr exact i64 %sub.ptr.sub.i, 3
  %cmp = icmp ult i64 %sub.ptr.div.i, 16384
  br i1 %cmp, label %if.then, label %if.else19

if.then:                                          ; preds = %entry
  %allocated.i = getelementptr inbounds %struct.sequence.46, %struct.sequence.46* %agg.result, i64 0, i32 2
  store i8 0, i8* %allocated.i, align 8, !tbaa !7
  br i1 %inplace, label %_ZN8sequenceIlED2Ev.exit466, label %if.else

_ZN8sequenceIlED2Ev.exit466:                      ; preds = %if.then
  %4 = bitcast %struct.sequence.46* %agg.result to i64*
  store i64 %3, i64* %4, align 8, !tbaa !6
  %e3.i461 = getelementptr inbounds %struct.sequence.46, %struct.sequence.46* %agg.result, i64 0, i32 1
  %5 = bitcast i64** %e3.i461 to i64*
  store i64 %1, i64* %5, align 8, !tbaa !0
  store i8 0, i8* %allocated.i, align 8, !tbaa !7
  %6 = inttoptr i64 %3 to i64*
  br label %if.end

if.else:                                          ; preds = %if.then
  %7 = lshr i64 %sub.ptr.div.i, 3
  %add.i.i = shl i64 %7, 6
  %mul1.i.i = add i64 %add.i.i, 64
  %call.i.i = tail call noalias i8* @aligned_alloc(i64 64, i64 %mul1.i.i) #4
  %8 = bitcast i8* %call.i.i to i64*
  %cmp.i.i = icmp eq i8* %call.i.i, null
  br i1 %cmp.i.i, label %if.then.i.i471, label %_ZN4pbbs17new_array_no_initIlEEPT_mb.exit.i

if.then.i.i471:                                   ; preds = %if.else
  %9 = load %struct._IO_FILE*, %struct._IO_FILE** @stderr, align 8, !tbaa !8
  %10 = tail call i64 @fwrite(i8* getelementptr inbounds ([22 x i8], [22 x i8]* @.str.47, i64 0, i64 0), i64 21, i64 1, %struct._IO_FILE* %9) #7
  tail call void @exit(i32 1) #8
  unreachable

_ZN4pbbs17new_array_no_initIlEEPT_mb.exit.i:      ; preds = %if.else
  %add.ptr.i473 = getelementptr inbounds i64, i64* %8, i64 %sub.ptr.div.i
  %cmp31.i = icmp eq i64 %sub.ptr.sub.i, 0
  %11 = ptrtoint i8* %call.i.i to i64
  %12 = ptrtoint i64* %add.ptr.i473 to i64
  br i1 %cmp31.i, label %pfor.cond.cleanup.i, label %pfor.detach.lr.ph.i

pfor.detach.lr.ph.i:                              ; preds = %_ZN4pbbs17new_array_no_initIlEEPT_mb.exit.i
  %s.i.i.i = getelementptr inbounds %struct.sequence.46, %struct.sequence.46* %A, i64 0, i32 0
  %13 = icmp ugt i64 %sub.ptr.div.i, 1
  %umax = select i1 %13, i64 %sub.ptr.div.i, i64 1
  br label %pfor.detach.i

pfor.cond.cleanup.i.loopexit:                     ; preds = %pfor.inc.i
  br label %pfor.cond.cleanup.i

pfor.cond.cleanup.i:                              ; preds = %pfor.cond.cleanup.i.loopexit, %_ZN4pbbs17new_array_no_initIlEEPT_mb.exit.i
  sync within %syncreg.i, label %_ZN8sequenceIlED2Ev.exit489

pfor.detach.i:                                    ; preds = %pfor.inc.i, %pfor.detach.lr.ph.i
  %__begin.032.i = phi i64 [ 0, %pfor.detach.lr.ph.i ], [ %inc.i, %pfor.inc.i ]
  detach within %syncreg.i, label %pfor.body.i, label %pfor.inc.i unwind label %lpad8.i

pfor.body.i:                                      ; preds = %pfor.detach.i
  %14 = load i64*, i64** %s.i.i.i, align 8, !tbaa !6
  %arrayidx.i.i.i = getelementptr inbounds i64, i64* %14, i64 %__begin.032.i
  %15 = load i64, i64* %arrayidx.i.i.i, align 8, !tbaa !9
  %add.ptr7.i = getelementptr inbounds i64, i64* %8, i64 %__begin.032.i
  store i64 %15, i64* %add.ptr7.i, align 8, !tbaa !9
  reattach within %syncreg.i, label %pfor.inc.i

pfor.inc.i:                                       ; preds = %pfor.body.i, %pfor.detach.i
  %inc.i = add nuw i64 %__begin.032.i, 1
  %exitcond = icmp ne i64 %inc.i, %umax
  br i1 %exitcond, label %pfor.detach.i, label %pfor.cond.cleanup.i.loopexit, !llvm.loop !11

lpad8.i:                                          ; preds = %pfor.detach.i
  %16 = landingpad { i8*, i32 }
          cleanup
  sync within %syncreg.i, label %ehcleanup17.thread

_ZN8sequenceIlED2Ev.exit489:                      ; preds = %pfor.cond.cleanup.i
  %17 = bitcast %struct.sequence.46* %agg.result to i64*
  store i64 %11, i64* %17, align 8, !tbaa !6
  %e3.i482 = getelementptr inbounds %struct.sequence.46, %struct.sequence.46* %agg.result, i64 0, i32 1
  %18 = bitcast i64** %e3.i482 to i64*
  store i64 %12, i64* %18, align 8, !tbaa !0
  store i8 1, i8* %allocated.i, align 8, !tbaa !7
  %19 = bitcast i8* %call.i.i to i64*
  br label %if.end

ehcleanup17.thread:                               ; preds = %lpad8.i
  %20 = extractvalue { i8*, i32 } %16, 0
  %21 = extractvalue { i8*, i32 } %16, 1
  br label %_ZN8sequenceIlED2Ev.exit497

if.end:                                           ; preds = %_ZN8sequenceIlED2Ev.exit489, %_ZN8sequenceIlED2Ev.exit466
  %22 = phi i8 [ 1, %_ZN8sequenceIlED2Ev.exit489 ], [ 0, %_ZN8sequenceIlED2Ev.exit466 ]
  %23 = phi i64* [ %19, %_ZN8sequenceIlED2Ev.exit489 ], [ %6, %_ZN8sequenceIlED2Ev.exit466 ]
  invoke void @_ZN4pbbs9quicksortIlSt4lessIlEEEvPT_mRKT0_(i64* %23, i64 %sub.ptr.div.i, %"struct.std::less"* nonnull dereferenceable(1) %f)
          to label %cleanup283 unwind label %ehcleanup17

ehcleanup17:                                      ; preds = %if.end
  %24 = landingpad { i8*, i32 }
          cleanup
  %25 = extractvalue { i8*, i32 } %24, 0
  %26 = extractvalue { i8*, i32 } %24, 1
  %tobool.i.i495 = icmp eq i8 %22, 0
  br i1 %tobool.i.i495, label %_ZN8sequenceIlED2Ev.exit497, label %if.then.i.i496

if.then.i.i496:                                   ; preds = %ehcleanup17
  %27 = bitcast %struct.sequence.46* %agg.result to i8**
  %28 = load i8*, i8** %27, align 8, !tbaa !6
  tail call void @free(i8* %28) #4
  br label %_ZN8sequenceIlED2Ev.exit497

_ZN8sequenceIlED2Ev.exit497:                      ; preds = %if.then.i.i496, %ehcleanup17, %ehcleanup17.thread
  %exn.slot.2676 = phi i8* [ %20, %ehcleanup17.thread ], [ %25, %ehcleanup17 ], [ %25, %if.then.i.i496 ]
  %ehselector.slot.2675 = phi i32 [ %21, %ehcleanup17.thread ], [ %26, %ehcleanup17 ], [ %26, %if.then.i.i496 ]
  %29 = bitcast %struct.sequence.46* %agg.result to i8*
  tail call void @llvm.memset.p0i8.i64(i8* %29, i8 0, i64 16, i32 8, i1 false) #4
  br label %ehcleanup284

if.else19:                                        ; preds = %entry
  %s.i498 = getelementptr inbounds %struct.sequence.46, %struct.sequence.46* %A, i64 0, i32 0
  %conv.i = uitofp i64 %sub.ptr.div.i to double
  %sqrt = tail call double @sqrt(double %conv.i) #2
  %30 = tail call double @llvm.fabs.f64(double %sqrt)
  %31 = tail call double @llvm.ceil.f64(double %30)
  %conv = fptoui double %31 to i64
  %div = udiv i64 %conv, 5
  %32 = icmp ult i64 %conv, 5
  br i1 %32, label %_ZN4pbbsL7log2_upImEEiT_.exit, label %while.body.i.preheader

while.body.i.preheader:                           ; preds = %if.else19
  br label %while.body.i

while.body.i:                                     ; preds = %while.body.i, %while.body.i.preheader
  %b.07.i = phi i64 [ %shr.i, %while.body.i ], [ %div, %while.body.i.preheader ]
  %a.06.i = phi i32 [ %inc.i499, %while.body.i ], [ 0, %while.body.i.preheader ]
  %shr.i = lshr i64 %b.07.i, 1
  %inc.i499 = add nuw nsw i32 %a.06.i, 1
  %cmp.i500 = icmp eq i64 %shr.i, 0
  br i1 %cmp.i500, label %_ZN4pbbsL7log2_upImEEiT_.exit.loopexit, label %while.body.i

_ZN4pbbsL7log2_upImEEiT_.exit.loopexit:           ; preds = %while.body.i
  %inc.i499.lcssa = phi i32 [ %inc.i499, %while.body.i ]
  br label %_ZN4pbbsL7log2_upImEEiT_.exit

_ZN4pbbsL7log2_upImEEiT_.exit:                    ; preds = %_ZN4pbbsL7log2_upImEEiT_.exit.loopexit, %if.else19
  %a.0.lcssa.i = phi i32 [ 0, %if.else19 ], [ %inc.i499.lcssa, %_ZN4pbbsL7log2_upImEEiT_.exit.loopexit ]
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
  %call.i.i504 = tail call noalias i8* @aligned_alloc(i64 64, i64 %mul1.i.i503) #4
  %cmp.i.i505 = icmp eq i8* %call.i.i504, null
  br i1 %cmp.i.i505, label %if.then.i.i506, label %_ZN4pbbs9new_arrayIlEEPT_m.exit

if.then.i.i506:                                   ; preds = %_ZN4pbbsL7log2_upImEEiT_.exit
  %33 = load %struct._IO_FILE*, %struct._IO_FILE** @stderr, align 8, !tbaa !8
  %34 = tail call i64 @fwrite(i8* getelementptr inbounds ([22 x i8], [22 x i8]* @.str.47, i64 0, i64 0), i64 21, i64 1, %struct._IO_FILE* %33) #7
  tail call void @exit(i32 1) #8
  unreachable

_ZN4pbbs9new_arrayIlEEPT_m.exit:                  ; preds = %_ZN4pbbsL7log2_upImEEiT_.exit
  %35 = bitcast i8* %call.i.i504 to i64*
  %cmp38667 = icmp eq i64 %mul, 0
  br i1 %cmp38667, label %pfor.cond.cleanup, label %pfor.detach.preheader

pfor.detach.preheader:                            ; preds = %_ZN4pbbs9new_arrayIlEEPT_m.exit
  br label %pfor.detach

pfor.cond.cleanup.loopexit:                       ; preds = %pfor.inc
  br label %pfor.cond.cleanup

pfor.cond.cleanup:                                ; preds = %pfor.cond.cleanup.loopexit, %_ZN4pbbs9new_arrayIlEEPT_m.exit
  sync within %syncreg, label %sync.continue

pfor.detach:                                      ; preds = %pfor.inc, %pfor.detach.preheader
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
  %36 = load i64*, i64** %s.i498, align 8, !tbaa !6
  %arrayidx.i = getelementptr inbounds i64, i64* %36, i64 %rem
  %37 = load i64, i64* %arrayidx.i, align 8, !tbaa !9
  %arrayidx = getelementptr inbounds i64, i64* %35, i64 %__begin.0668
  store i64 %37, i64* %arrayidx, align 8, !tbaa !9
  reattach within %syncreg, label %pfor.inc

pfor.inc:                                         ; preds = %pfor.body, %pfor.detach
  %inc = add nuw i64 %__begin.0668, 1
  %exitcond10 = icmp ne i64 %inc, %mul
  br i1 %exitcond10, label %pfor.detach, label %pfor.cond.cleanup.loopexit, !llvm.loop !13

lpad49:                                           ; preds = %pfor.detach
  %38 = landingpad { i8*, i32 }
          cleanup
  %39 = extractvalue { i8*, i32 } %38, 0
  %40 = extractvalue { i8*, i32 } %38, 1
  sync within %syncreg, label %ehcleanup284

sync.continue:                                    ; preds = %pfor.cond.cleanup
  tail call void @_ZN4pbbs9quicksortIlSt4lessIlEEEvPT_mRKT0_(i64* %35, i64 %mul, %"struct.std::less"* nonnull dereferenceable(1) %f)
  %41 = lshr i64 %div, 3
  %add.i.i510 = shl i64 %41, 6
  %mul1.i.i511 = add i64 %add.i.i510, 64
  %call.i.i512 = tail call noalias i8* @aligned_alloc(i64 64, i64 %mul1.i.i511) #4
  %cmp.i.i513 = icmp eq i8* %call.i.i512, null
  br i1 %cmp.i.i513, label %if.then.i.i514, label %_ZN4pbbs9new_arrayIlEEPT_m.exit516

if.then.i.i514:                                   ; preds = %sync.continue
  %42 = load %struct._IO_FILE*, %struct._IO_FILE** @stderr, align 8, !tbaa !8
  %43 = tail call i64 @fwrite(i8* getelementptr inbounds ([22 x i8], [22 x i8]* @.str.47, i64 0, i64 0), i64 21, i64 1, %struct._IO_FILE* %42) #7
  tail call void @exit(i32 1) #8
  unreachable

_ZN4pbbs9new_arrayIlEEPT_m.exit516:               ; preds = %sync.continue
  %44 = bitcast i8* %call.i.i512 to i64*
  %45 = icmp ugt i64 %conv, 4
  br i1 %45, label %pfor.detach70.preheader, label %pfor.cond.cleanup69

pfor.detach70.preheader:                          ; preds = %_ZN4pbbs9new_arrayIlEEPT_m.exit516
  %46 = icmp ugt i64 %div, 1
  %umax8 = select i1 %46, i64 %div, i64 1
  br label %pfor.detach70

pfor.cond.cleanup69.loopexit:                     ; preds = %pfor.inc79
  br label %pfor.cond.cleanup69

pfor.cond.cleanup69:                              ; preds = %pfor.cond.cleanup69.loopexit, %_ZN4pbbs9new_arrayIlEEPT_m.exit516
  sync within %syncreg58, label %sync.continue81

pfor.detach70:                                    ; preds = %pfor.inc79, %pfor.detach70.preheader
  %__begin60.0666 = phi i64 [ %inc80, %pfor.inc79 ], [ 0, %pfor.detach70.preheader ]
  detach within %syncreg58, label %pfor.body74, label %pfor.inc79

pfor.body74:                                      ; preds = %pfor.detach70
  %mul75 = shl i64 %__begin60.0666, 3
  %arrayidx76 = getelementptr inbounds i64, i64* %35, i64 %mul75
  %47 = load i64, i64* %arrayidx76, align 8, !tbaa !9
  %arrayidx77 = getelementptr inbounds i64, i64* %44, i64 %__begin60.0666
  store i64 %47, i64* %arrayidx77, align 8, !tbaa !9
  reattach within %syncreg58, label %pfor.inc79

pfor.inc79:                                       ; preds = %pfor.body74, %pfor.detach70
  %inc80 = add nuw nsw i64 %__begin60.0666, 1
  %exitcond9 = icmp ne i64 %inc80, %umax8
  br i1 %exitcond9, label %pfor.detach70, label %pfor.cond.cleanup69.loopexit, !llvm.loop !14

sync.continue81:                                  ; preds = %pfor.cond.cleanup69
  tail call void @free(i8* %call.i.i504) #4
  br i1 %inplace, label %_ZN8sequenceIlED2Ev.exit533, label %if.else93

_ZN8sequenceIlED2Ev.exit533:                      ; preds = %sync.continue81
  %48 = load i64, i64* %2, align 8, !tbaa !6, !noalias !15
  %49 = load i64, i64* %0, align 8, !tbaa !0, !noalias !15
  br label %if.end104

if.else93:                                        ; preds = %sync.continue81
  %50 = lshr i64 %sub.ptr.div.i, 3
  %add.i539 = shl i64 %50, 6
  %mul1.i = add i64 %add.i539, 64
  %call.i540 = tail call noalias i8* @aligned_alloc(i64 64, i64 %mul1.i) #4
  %51 = bitcast i8* %call.i540 to i64*
  %cmp.i541 = icmp eq i8* %call.i540, null
  br i1 %cmp.i541, label %if.then.i542, label %if.end.i

if.then.i542:                                     ; preds = %if.else93
  %52 = load %struct._IO_FILE*, %struct._IO_FILE** @stderr, align 8, !tbaa !8
  %53 = tail call i64 @fwrite(i8* getelementptr inbounds ([22 x i8], [22 x i8]* @.str.47, i64 0, i64 0), i64 21, i64 1, %struct._IO_FILE* %52) #7
  tail call void @exit(i32 1) #8
  unreachable

if.end.i:                                         ; preds = %if.else93
  %cmp420.i = icmp eq i64 %mul1.i, 0
  br i1 %cmp420.i, label %pfor.cond.cleanup.i544, label %pfor.detach.i545.preheader

pfor.detach.i545.preheader:                       ; preds = %if.end.i
  br label %pfor.detach.i545

pfor.cond.cleanup.i544.loopexit:                  ; preds = %pfor.inc.i548
  br label %pfor.cond.cleanup.i544

pfor.cond.cleanup.i544:                           ; preds = %pfor.cond.cleanup.i544.loopexit, %if.end.i
  sync within %syncreg.i538, label %_ZN8sequenceIlED2Ev.exit566

pfor.detach.i545:                                 ; preds = %pfor.inc.i548, %pfor.detach.i545.preheader
  %i.021.i = phi i64 [ %add5.i, %pfor.inc.i548 ], [ 0, %pfor.detach.i545.preheader ]
  detach within %syncreg.i538, label %pfor.body.i547, label %pfor.inc.i548

pfor.body.i547:                                   ; preds = %pfor.detach.i545
  %arrayidx.i546 = getelementptr inbounds i8, i8* %call.i540, i64 %i.021.i
  store i8 0, i8* %arrayidx.i546, align 1, !tbaa !18
  reattach within %syncreg.i538, label %pfor.inc.i548

pfor.inc.i548:                                    ; preds = %pfor.body.i547, %pfor.detach.i545
  %add5.i = add i64 %i.021.i, 2097152
  %cmp4.i = icmp ult i64 %add5.i, %mul1.i
  br i1 %cmp4.i, label %pfor.detach.i545, label %pfor.cond.cleanup.i544.loopexit, !llvm.loop !19

_ZN8sequenceIlED2Ev.exit566:                      ; preds = %pfor.cond.cleanup.i544
  %54 = ptrtoint i8* %call.i540 to i64
  %add.ptr.i551 = getelementptr inbounds i64, i64* %51, i64 %sub.ptr.div.i
  %55 = ptrtoint i64* %add.ptr.i551 to i64
  br label %if.end104

if.end104:                                        ; preds = %_ZN8sequenceIlED2Ev.exit566, %_ZN8sequenceIlED2Ev.exit533
  %Bs.sroa.0.0 = phi i64 [ %48, %_ZN8sequenceIlED2Ev.exit533 ], [ %54, %_ZN8sequenceIlED2Ev.exit566 ]
  %Bs.sroa.16.0 = phi i64 [ %49, %_ZN8sequenceIlED2Ev.exit533 ], [ %55, %_ZN8sequenceIlED2Ev.exit566 ]
  %56 = inttoptr i64 %Bs.sroa.0.0 to i64*
  %57 = inttoptr i64 %Bs.sroa.0.0 to i8*
  %58 = lshr i64 %mul32, 3
  %add.i573 = shl i64 %58, 6
  %mul1.i574 = add i64 %add.i573, 64
  %call.i575 = tail call noalias i8* @aligned_alloc(i64 64, i64 %mul1.i574) #4
  %59 = bitcast i8* %call.i575 to i64*
  %cmp.i576 = icmp eq i8* %call.i575, null
  br i1 %cmp.i576, label %if.then.i577, label %if.end.i578

if.then.i577:                                     ; preds = %if.end104
  %60 = load %struct._IO_FILE*, %struct._IO_FILE** @stderr, align 8, !tbaa !8
  %61 = tail call i64 @fwrite(i8* getelementptr inbounds ([22 x i8], [22 x i8]* @.str.47, i64 0, i64 0), i64 21, i64 1, %struct._IO_FILE* %60) #7
  tail call void @exit(i32 1) #8
  unreachable

if.end.i578:                                      ; preds = %if.end104
  %cmp420.i579 = icmp eq i64 %mul1.i574, 0
  br i1 %cmp420.i579, label %pfor.cond.cleanup.i581, label %pfor.detach.i583.preheader

pfor.detach.i583.preheader:                       ; preds = %if.end.i578
  br label %pfor.detach.i583

pfor.cond.cleanup.i581.loopexit:                  ; preds = %pfor.inc.i588
  br label %pfor.cond.cleanup.i581

pfor.cond.cleanup.i581:                           ; preds = %pfor.cond.cleanup.i581.loopexit, %if.end.i578
  sync within %syncreg.i572, label %invoke.cont109

pfor.detach.i583:                                 ; preds = %pfor.inc.i588, %pfor.detach.i583.preheader
  %i.021.i582 = phi i64 [ %add5.i586, %pfor.inc.i588 ], [ 0, %pfor.detach.i583.preheader ]
  detach within %syncreg.i572, label %pfor.body.i585, label %pfor.inc.i588

pfor.body.i585:                                   ; preds = %pfor.detach.i583
  %arrayidx.i584 = getelementptr inbounds i8, i8* %call.i575, i64 %i.021.i582
  store i8 0, i8* %arrayidx.i584, align 1, !tbaa !18
  reattach within %syncreg.i572, label %pfor.inc.i588

pfor.inc.i588:                                    ; preds = %pfor.body.i585, %pfor.detach.i583
  %add5.i586 = add i64 %i.021.i582, 2097152
  %cmp4.i587 = icmp ult i64 %add5.i586, %mul1.i574
  br i1 %cmp4.i587, label %pfor.detach.i583, label %pfor.cond.cleanup.i581.loopexit, !llvm.loop !20

invoke.cont109:                                   ; preds = %pfor.cond.cleanup.i581
  %sub116 = add nsw i64 %conv27, -1
  %add.ptr2.i = getelementptr inbounds i64, i64* %44, i64 %div
  %62 = shl i64 %div, 3
  %63 = add i64 %62, 8
  %64 = shl i64 %div28, 3
  %65 = add i64 %64, 8
  %inplace.not = xor i1 %inplace, true
  %66 = icmp ugt i64 %conv27, 1
  %umax6 = select i1 %66, i64 %conv27, i64 1
  br label %pfor.detach122

pfor.cond.cleanup121:                             ; preds = %pfor.inc159
  sync within %syncreg111, label %sync.continue168

pfor.detach122:                                   ; preds = %pfor.inc159, %invoke.cont109
  %indvars.iv = phi i64 [ 0, %invoke.cont109 ], [ %indvars.iv.next, %pfor.inc159 ]
  %__begin113.0663 = phi i64 [ 0, %invoke.cont109 ], [ %inc160, %pfor.inc159 ]
  %67 = mul i64 %63, %__begin113.0663
  %scevgep = getelementptr i8, i8* %call.i575, i64 %67
  detach within %syncreg111, label %pfor.body126, label %pfor.inc159 unwind label %lpad161.loopexit
; CHECK: define private fastcc void @_ZN4pbbs12sample_sort_Im8sequenceIlESt4lessIlEEES1_INT0_1TEES5_RKT1_b.outline_pfor.detach122.ls1(i64 %__begin113.0663.start.ls1,
; CHECK: pfor.detach122.ls1:
; CHECK-NOT: indvars.iv = phi
; CHECK: %__begin113.0663.ls1 = phi i64
; CHECK-NOT: phi
; CHECK: mul i64 {{%add29.ls1, %__begin113.0663.ls1|%__begin113.0663.ls1, %add29.ls1}}
; CHECK: br label %pfor.body126.ls1

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
  %68 = load i64*, i64** %s.i498, align 8, !tbaa !6
  %69 = add i64 %cond, %indvars.iv
  %min.iters.check = icmp ult i64 %cond, 16
  br i1 %min.iters.check, label %for.body.preheader, label %vector.memcheck

vector.memcheck:                                  ; preds = %for.body.lr.ph
  %70 = bitcast i64* %68 to i8*
  %71 = mul i64 %65, %__begin113.0663
  %uglygep = getelementptr i8, i8* %57, i64 %71
  %72 = shl i64 %cond, 3
  %scevgep686 = getelementptr i8, i8* %uglygep, i64 %72
  %uglygep687 = getelementptr i8, i8* %70, i64 %71
  %scevgep688 = getelementptr i64, i64* %68, i64 %cond
  %scevgep688689 = bitcast i64* %scevgep688 to i8*
  %bound0 = icmp ult i8* %57, %scevgep688689
  %bound1 = icmp ult i8* %uglygep687, %scevgep686
  %memcheck.conflict = and i1 %bound0, %bound1
  br i1 %memcheck.conflict, label %for.body.preheader, label %vector.ph

vector.ph:                                        ; preds = %vector.memcheck
  %n.vec = and i64 %cond, -16
  %ind.end = add i64 %mul127, %n.vec
  %73 = add i64 %n.vec, -16
  %74 = lshr exact i64 %73, 4
  %75 = add nuw nsw i64 %74, 1
  %xtraiter = and i64 %75, 3
  %76 = icmp ult i64 %73, 48
  br i1 %76, label %middle.block.unr-lcssa, label %vector.ph.new

vector.ph.new:                                    ; preds = %vector.ph
  %unroll_iter = sub nsw i64 %75, %xtraiter
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph.new
  %index = phi i64 [ 0, %vector.ph.new ], [ %index.next.3, %vector.body ]
  %niter = phi i64 [ %unroll_iter, %vector.ph.new ], [ %niter.nsub.3, %vector.body ]
  %77 = add i64 %mul127, %index
  %78 = getelementptr inbounds i64, i64* %56, i64 %77
  %79 = getelementptr inbounds i64, i64* %68, i64 %77
  %80 = bitcast i64* %79 to <4 x i64>*
  %wide.load = load <4 x i64>, <4 x i64>* %80, align 8, !tbaa !9, !alias.scope !21
  %81 = getelementptr i64, i64* %79, i64 4
  %82 = bitcast i64* %81 to <4 x i64>*
  %wide.load694 = load <4 x i64>, <4 x i64>* %82, align 8, !tbaa !9, !alias.scope !21
  %83 = getelementptr i64, i64* %79, i64 8
  %84 = bitcast i64* %83 to <4 x i64>*
  %wide.load695 = load <4 x i64>, <4 x i64>* %84, align 8, !tbaa !9, !alias.scope !21
  %85 = getelementptr i64, i64* %79, i64 12
  %86 = bitcast i64* %85 to <4 x i64>*
  %wide.load696 = load <4 x i64>, <4 x i64>* %86, align 8, !tbaa !9, !alias.scope !21
  %87 = bitcast i64* %78 to <4 x i64>*
  store <4 x i64> %wide.load, <4 x i64>* %87, align 8, !tbaa !9, !alias.scope !24, !noalias !21
  %88 = getelementptr i64, i64* %78, i64 4
  %89 = bitcast i64* %88 to <4 x i64>*
  store <4 x i64> %wide.load694, <4 x i64>* %89, align 8, !tbaa !9, !alias.scope !24, !noalias !21
  %90 = getelementptr i64, i64* %78, i64 8
  %91 = bitcast i64* %90 to <4 x i64>*
  store <4 x i64> %wide.load695, <4 x i64>* %91, align 8, !tbaa !9, !alias.scope !24, !noalias !21
  %92 = getelementptr i64, i64* %78, i64 12
  %93 = bitcast i64* %92 to <4 x i64>*
  store <4 x i64> %wide.load696, <4 x i64>* %93, align 8, !tbaa !9, !alias.scope !24, !noalias !21
  %index.next = or i64 %index, 16
  %94 = add i64 %mul127, %index.next
  %95 = getelementptr inbounds i64, i64* %56, i64 %94
  %96 = getelementptr inbounds i64, i64* %68, i64 %94
  %97 = bitcast i64* %96 to <4 x i64>*
  %wide.load.1 = load <4 x i64>, <4 x i64>* %97, align 8, !tbaa !9, !alias.scope !21
  %98 = getelementptr i64, i64* %96, i64 4
  %99 = bitcast i64* %98 to <4 x i64>*
  %wide.load694.1 = load <4 x i64>, <4 x i64>* %99, align 8, !tbaa !9, !alias.scope !21
  %100 = getelementptr i64, i64* %96, i64 8
  %101 = bitcast i64* %100 to <4 x i64>*
  %wide.load695.1 = load <4 x i64>, <4 x i64>* %101, align 8, !tbaa !9, !alias.scope !21
  %102 = getelementptr i64, i64* %96, i64 12
  %103 = bitcast i64* %102 to <4 x i64>*
  %wide.load696.1 = load <4 x i64>, <4 x i64>* %103, align 8, !tbaa !9, !alias.scope !21
  %104 = bitcast i64* %95 to <4 x i64>*
  store <4 x i64> %wide.load.1, <4 x i64>* %104, align 8, !tbaa !9, !alias.scope !24, !noalias !21
  %105 = getelementptr i64, i64* %95, i64 4
  %106 = bitcast i64* %105 to <4 x i64>*
  store <4 x i64> %wide.load694.1, <4 x i64>* %106, align 8, !tbaa !9, !alias.scope !24, !noalias !21
  %107 = getelementptr i64, i64* %95, i64 8
  %108 = bitcast i64* %107 to <4 x i64>*
  store <4 x i64> %wide.load695.1, <4 x i64>* %108, align 8, !tbaa !9, !alias.scope !24, !noalias !21
  %109 = getelementptr i64, i64* %95, i64 12
  %110 = bitcast i64* %109 to <4 x i64>*
  store <4 x i64> %wide.load696.1, <4 x i64>* %110, align 8, !tbaa !9, !alias.scope !24, !noalias !21
  %index.next.1 = or i64 %index, 32
  %111 = add i64 %mul127, %index.next.1
  %112 = getelementptr inbounds i64, i64* %56, i64 %111
  %113 = getelementptr inbounds i64, i64* %68, i64 %111
  %114 = bitcast i64* %113 to <4 x i64>*
  %wide.load.2 = load <4 x i64>, <4 x i64>* %114, align 8, !tbaa !9, !alias.scope !21
  %115 = getelementptr i64, i64* %113, i64 4
  %116 = bitcast i64* %115 to <4 x i64>*
  %wide.load694.2 = load <4 x i64>, <4 x i64>* %116, align 8, !tbaa !9, !alias.scope !21
  %117 = getelementptr i64, i64* %113, i64 8
  %118 = bitcast i64* %117 to <4 x i64>*
  %wide.load695.2 = load <4 x i64>, <4 x i64>* %118, align 8, !tbaa !9, !alias.scope !21
  %119 = getelementptr i64, i64* %113, i64 12
  %120 = bitcast i64* %119 to <4 x i64>*
  %wide.load696.2 = load <4 x i64>, <4 x i64>* %120, align 8, !tbaa !9, !alias.scope !21
  %121 = bitcast i64* %112 to <4 x i64>*
  store <4 x i64> %wide.load.2, <4 x i64>* %121, align 8, !tbaa !9, !alias.scope !24, !noalias !21
  %122 = getelementptr i64, i64* %112, i64 4
  %123 = bitcast i64* %122 to <4 x i64>*
  store <4 x i64> %wide.load694.2, <4 x i64>* %123, align 8, !tbaa !9, !alias.scope !24, !noalias !21
  %124 = getelementptr i64, i64* %112, i64 8
  %125 = bitcast i64* %124 to <4 x i64>*
  store <4 x i64> %wide.load695.2, <4 x i64>* %125, align 8, !tbaa !9, !alias.scope !24, !noalias !21
  %126 = getelementptr i64, i64* %112, i64 12
  %127 = bitcast i64* %126 to <4 x i64>*
  store <4 x i64> %wide.load696.2, <4 x i64>* %127, align 8, !tbaa !9, !alias.scope !24, !noalias !21
  %index.next.2 = or i64 %index, 48
  %128 = add i64 %mul127, %index.next.2
  %129 = getelementptr inbounds i64, i64* %56, i64 %128
  %130 = getelementptr inbounds i64, i64* %68, i64 %128
  %131 = bitcast i64* %130 to <4 x i64>*
  %wide.load.3 = load <4 x i64>, <4 x i64>* %131, align 8, !tbaa !9, !alias.scope !21
  %132 = getelementptr i64, i64* %130, i64 4
  %133 = bitcast i64* %132 to <4 x i64>*
  %wide.load694.3 = load <4 x i64>, <4 x i64>* %133, align 8, !tbaa !9, !alias.scope !21
  %134 = getelementptr i64, i64* %130, i64 8
  %135 = bitcast i64* %134 to <4 x i64>*
  %wide.load695.3 = load <4 x i64>, <4 x i64>* %135, align 8, !tbaa !9, !alias.scope !21
  %136 = getelementptr i64, i64* %130, i64 12
  %137 = bitcast i64* %136 to <4 x i64>*
  %wide.load696.3 = load <4 x i64>, <4 x i64>* %137, align 8, !tbaa !9, !alias.scope !21
  %138 = bitcast i64* %129 to <4 x i64>*
  store <4 x i64> %wide.load.3, <4 x i64>* %138, align 8, !tbaa !9, !alias.scope !24, !noalias !21
  %139 = getelementptr i64, i64* %129, i64 4
  %140 = bitcast i64* %139 to <4 x i64>*
  store <4 x i64> %wide.load694.3, <4 x i64>* %140, align 8, !tbaa !9, !alias.scope !24, !noalias !21
  %141 = getelementptr i64, i64* %129, i64 8
  %142 = bitcast i64* %141 to <4 x i64>*
  store <4 x i64> %wide.load695.3, <4 x i64>* %142, align 8, !tbaa !9, !alias.scope !24, !noalias !21
  %143 = getelementptr i64, i64* %129, i64 12
  %144 = bitcast i64* %143 to <4 x i64>*
  store <4 x i64> %wide.load696.3, <4 x i64>* %144, align 8, !tbaa !9, !alias.scope !24, !noalias !21
  %index.next.3 = add i64 %index, 64
  %niter.nsub.3 = add i64 %niter, -4
  %niter.ncmp.3 = icmp eq i64 %niter.nsub.3, 0
  br i1 %niter.ncmp.3, label %middle.block.unr-lcssa.loopexit, label %vector.body, !llvm.loop !26

middle.block.unr-lcssa.loopexit:                  ; preds = %vector.body
  %index.next.3.lcssa = phi i64 [ %index.next.3, %vector.body ]
  br label %middle.block.unr-lcssa

middle.block.unr-lcssa:                           ; preds = %middle.block.unr-lcssa.loopexit, %vector.ph
  %index.unr = phi i64 [ 0, %vector.ph ], [ %index.next.3.lcssa, %middle.block.unr-lcssa.loopexit ]
  %lcmp.mod = icmp eq i64 %xtraiter, 0
  br i1 %lcmp.mod, label %middle.block, label %vector.body.epil.preheader

vector.body.epil.preheader:                       ; preds = %middle.block.unr-lcssa
  br label %vector.body.epil

vector.body.epil:                                 ; preds = %vector.body.epil, %vector.body.epil.preheader
  %index.epil = phi i64 [ %index.unr, %vector.body.epil.preheader ], [ %index.next.epil, %vector.body.epil ]
  %epil.iter = phi i64 [ %xtraiter, %vector.body.epil.preheader ], [ %epil.iter.sub, %vector.body.epil ]
  %145 = add i64 %mul127, %index.epil
  %146 = getelementptr inbounds i64, i64* %56, i64 %145
  %147 = getelementptr inbounds i64, i64* %68, i64 %145
  %148 = bitcast i64* %147 to <4 x i64>*
  %wide.load.epil = load <4 x i64>, <4 x i64>* %148, align 8, !tbaa !9, !alias.scope !21
  %149 = getelementptr i64, i64* %147, i64 4
  %150 = bitcast i64* %149 to <4 x i64>*
  %wide.load694.epil = load <4 x i64>, <4 x i64>* %150, align 8, !tbaa !9, !alias.scope !21
  %151 = getelementptr i64, i64* %147, i64 8
  %152 = bitcast i64* %151 to <4 x i64>*
  %wide.load695.epil = load <4 x i64>, <4 x i64>* %152, align 8, !tbaa !9, !alias.scope !21
  %153 = getelementptr i64, i64* %147, i64 12
  %154 = bitcast i64* %153 to <4 x i64>*
  %wide.load696.epil = load <4 x i64>, <4 x i64>* %154, align 8, !tbaa !9, !alias.scope !21
  %155 = bitcast i64* %146 to <4 x i64>*
  store <4 x i64> %wide.load.epil, <4 x i64>* %155, align 8, !tbaa !9, !alias.scope !24, !noalias !21
  %156 = getelementptr i64, i64* %146, i64 4
  %157 = bitcast i64* %156 to <4 x i64>*
  store <4 x i64> %wide.load694.epil, <4 x i64>* %157, align 8, !tbaa !9, !alias.scope !24, !noalias !21
  %158 = getelementptr i64, i64* %146, i64 8
  %159 = bitcast i64* %158 to <4 x i64>*
  store <4 x i64> %wide.load695.epil, <4 x i64>* %159, align 8, !tbaa !9, !alias.scope !24, !noalias !21
  %160 = getelementptr i64, i64* %146, i64 12
  %161 = bitcast i64* %160 to <4 x i64>*
  store <4 x i64> %wide.load696.epil, <4 x i64>* %161, align 8, !tbaa !9, !alias.scope !24, !noalias !21
  %index.next.epil = add i64 %index.epil, 16
  %epil.iter.sub = add nsw i64 %epil.iter, -1
  %epil.iter.cmp = icmp eq i64 %epil.iter.sub, 0
  br i1 %epil.iter.cmp, label %middle.block.loopexit, label %vector.body.epil, !llvm.loop !28

middle.block.loopexit:                            ; preds = %vector.body.epil
  br label %middle.block

middle.block:                                     ; preds = %middle.block.loopexit, %middle.block.unr-lcssa
  %cmp.n = icmp eq i64 %cond, %n.vec
  br i1 %cmp.n, label %if.end146, label %for.body.preheader

for.body.preheader:                               ; preds = %middle.block, %vector.memcheck, %for.body.lr.ph
  %j133.0662.ph = phi i64 [ %mul127, %vector.memcheck ], [ %mul127, %for.body.lr.ph ], [ %ind.end, %middle.block ]
  %162 = add i64 %cond, %mul127
  %163 = sub i64 %162, %j133.0662.ph
  %164 = add i64 %mul127, -1
  %165 = add i64 %cond, %164
  %166 = sub i64 %165, %j133.0662.ph
  %xtraiter701 = and i64 %163, 7
  %lcmp.mod702 = icmp eq i64 %xtraiter701, 0
  br i1 %lcmp.mod702, label %for.body.prol.loopexit, label %for.body.prol.preheader

for.body.prol.preheader:                          ; preds = %for.body.preheader
  br label %for.body.prol

for.body.prol:                                    ; preds = %for.body.prol, %for.body.prol.preheader
  %j133.0662.prol = phi i64 [ %inc144.prol, %for.body.prol ], [ %j133.0662.ph, %for.body.prol.preheader ]
  %prol.iter = phi i64 [ %prol.iter.sub, %for.body.prol ], [ %xtraiter701, %for.body.prol.preheader ]
  %arrayidx137.prol = getelementptr inbounds i64, i64* %56, i64 %j133.0662.prol
  %arrayidx.i590.prol = getelementptr inbounds i64, i64* %68, i64 %j133.0662.prol
  %167 = load i64, i64* %arrayidx.i590.prol, align 8, !tbaa !9
  store i64 %167, i64* %arrayidx137.prol, align 8, !tbaa !9
  %inc144.prol = add i64 %j133.0662.prol, 1
  %prol.iter.sub = add nsw i64 %prol.iter, -1
  %prol.iter.cmp = icmp eq i64 %prol.iter.sub, 0
  br i1 %prol.iter.cmp, label %for.body.prol.loopexit.loopexit, label %for.body.prol, !llvm.loop !30

for.body.prol.loopexit.loopexit:                  ; preds = %for.body.prol
  %inc144.prol.lcssa = phi i64 [ %inc144.prol, %for.body.prol ]
  br label %for.body.prol.loopexit

for.body.prol.loopexit:                           ; preds = %for.body.prol.loopexit.loopexit, %for.body.preheader
  %j133.0662.unr = phi i64 [ %j133.0662.ph, %for.body.preheader ], [ %inc144.prol.lcssa, %for.body.prol.loopexit.loopexit ]
  %168 = icmp ult i64 %166, 7
  br i1 %168, label %if.end146, label %for.body.preheader.new

for.body.preheader.new:                           ; preds = %for.body.prol.loopexit
  br label %for.body

for.body:                                         ; preds = %for.body, %for.body.preheader.new
  %j133.0662 = phi i64 [ %j133.0662.unr, %for.body.preheader.new ], [ %inc144.7, %for.body ]
  %arrayidx137 = getelementptr inbounds i64, i64* %56, i64 %j133.0662
  %arrayidx.i590 = getelementptr inbounds i64, i64* %68, i64 %j133.0662
  %169 = load i64, i64* %arrayidx.i590, align 8, !tbaa !9
  store i64 %169, i64* %arrayidx137, align 8, !tbaa !9
  %inc144 = add i64 %j133.0662, 1
  %arrayidx137.1 = getelementptr inbounds i64, i64* %56, i64 %inc144
  %arrayidx.i590.1 = getelementptr inbounds i64, i64* %68, i64 %inc144
  %170 = load i64, i64* %arrayidx.i590.1, align 8, !tbaa !9
  store i64 %170, i64* %arrayidx137.1, align 8, !tbaa !9
  %inc144.1 = add i64 %j133.0662, 2
  %arrayidx137.2 = getelementptr inbounds i64, i64* %56, i64 %inc144.1
  %arrayidx.i590.2 = getelementptr inbounds i64, i64* %68, i64 %inc144.1
  %171 = load i64, i64* %arrayidx.i590.2, align 8, !tbaa !9
  store i64 %171, i64* %arrayidx137.2, align 8, !tbaa !9
  %inc144.2 = add i64 %j133.0662, 3
  %arrayidx137.3 = getelementptr inbounds i64, i64* %56, i64 %inc144.2
  %arrayidx.i590.3 = getelementptr inbounds i64, i64* %68, i64 %inc144.2
  %172 = load i64, i64* %arrayidx.i590.3, align 8, !tbaa !9
  store i64 %172, i64* %arrayidx137.3, align 8, !tbaa !9
  %inc144.3 = add i64 %j133.0662, 4
  %arrayidx137.4 = getelementptr inbounds i64, i64* %56, i64 %inc144.3
  %arrayidx.i590.4 = getelementptr inbounds i64, i64* %68, i64 %inc144.3
  %173 = load i64, i64* %arrayidx.i590.4, align 8, !tbaa !9
  store i64 %173, i64* %arrayidx137.4, align 8, !tbaa !9
  %inc144.4 = add i64 %j133.0662, 5
  %arrayidx137.5 = getelementptr inbounds i64, i64* %56, i64 %inc144.4
  %arrayidx.i590.5 = getelementptr inbounds i64, i64* %68, i64 %inc144.4
  %174 = load i64, i64* %arrayidx.i590.5, align 8, !tbaa !9
  store i64 %174, i64* %arrayidx137.5, align 8, !tbaa !9
  %inc144.5 = add i64 %j133.0662, 6
  %arrayidx137.6 = getelementptr inbounds i64, i64* %56, i64 %inc144.5
  %arrayidx.i590.6 = getelementptr inbounds i64, i64* %68, i64 %inc144.5
  %175 = load i64, i64* %arrayidx.i590.6, align 8, !tbaa !9
  store i64 %175, i64* %arrayidx137.6, align 8, !tbaa !9
  %inc144.6 = add i64 %j133.0662, 7
  %arrayidx137.7 = getelementptr inbounds i64, i64* %56, i64 %inc144.6
  %arrayidx.i590.7 = getelementptr inbounds i64, i64* %68, i64 %inc144.6
  %176 = load i64, i64* %arrayidx.i590.7, align 8, !tbaa !9
  store i64 %176, i64* %arrayidx137.7, align 8, !tbaa !9
  %inc144.7 = add i64 %j133.0662, 8
  %exitcond.7 = icmp eq i64 %inc144.7, %69
  br i1 %exitcond.7, label %if.end146.loopexit, label %for.body, !llvm.loop !31

if.end146.loopexit:                               ; preds = %for.body
  br label %if.end146

if.end146:                                        ; preds = %if.end146.loopexit, %for.body.prol.loopexit, %middle.block, %pfor.body126
  %add.ptr = getelementptr inbounds i64, i64* %56, i64 %mul127
  invoke void @_ZN4pbbs9quicksortIlSt4lessIlEEEvPT_mRKT0_(i64* %add.ptr, i64 %cond, %"struct.std::less"* nonnull dereferenceable(1) %f)
          to label %invoke.cont148 unwind label %lpad147

invoke.cont148:                                   ; preds = %if.end146
  %cmp.i591 = icmp eq i64 %cond, 0
  %or.cond.i = or i1 %32, %cmp.i591
  br i1 %or.cond.i, label %_ZN4pbbs9merge_seqIlSt4lessIlEmEEvPT_S4_PT1_mmT0_.exit, label %if.end.i593

if.end.i593:                                      ; preds = %invoke.cont148
  %mul150 = mul i64 %__begin113.0663, %add31
  %add.ptr151 = getelementptr inbounds i64, i64* %59, i64 %mul150
  call void @llvm.memset.p0i8.i64(i8* %scevgep, i8 0, i64 %63, i32 8, i1 false)
  %add.ptr.i592 = getelementptr inbounds i64, i64* %add.ptr, i64 %cond
  %.pre.i = load i64, i64* %add.ptr, align 8, !tbaa !9
  br label %while.cond.i

while.cond.i:                                     ; preds = %while.cond.i.backedge, %if.end.i593
  %177 = phi i64 [ %.pre.i, %if.end.i593 ], [ %.be, %while.cond.i.backedge ]
  %sC.addr.0.i = phi i64* [ %add.ptr151, %if.end.i593 ], [ %sC.addr.0.i.be, %while.cond.i.backedge ]
  %sB.addr.0.i = phi i64* [ %44, %if.end.i593 ], [ %sB.addr.0.i.be, %while.cond.i.backedge ]
  %sA.addr.0.i = phi i64* [ %add.ptr, %if.end.i593 ], [ %sA.addr.0.i.be, %while.cond.i.backedge ]
  %178 = load i64, i64* %sB.addr.0.i, align 8, !tbaa !9
  %cmp.i71.i = icmp slt i64 %177, %178
  br i1 %cmp.i71.i, label %while.body5.lr.ph.i, label %while.end.i

while.body5.lr.ph.i:                              ; preds = %while.cond.i
  %.pre82.i = load i64, i64* %sC.addr.0.i, align 8, !tbaa !9
  br label %while.body5.i

while.cond4.i:                                    ; preds = %while.body5.i
  %179 = load i64, i64* %incdec.ptr.i, align 8, !tbaa !9
  %cmp.i.i596 = icmp slt i64 %179, %178
  br i1 %cmp.i.i596, label %while.body5.i, label %while.end.i.loopexit

while.body5.i:                                    ; preds = %while.cond4.i, %while.body5.lr.ph.i
  %180 = phi i64 [ %.pre82.i, %while.body5.lr.ph.i ], [ %inc6.i, %while.cond4.i ]
  %sA.addr.172.i = phi i64* [ %sA.addr.0.i, %while.body5.lr.ph.i ], [ %incdec.ptr.i, %while.cond4.i ]
  %inc6.i = add i64 %180, 1
  store i64 %inc6.i, i64* %sC.addr.0.i, align 8, !tbaa !9
  %incdec.ptr.i = getelementptr inbounds i64, i64* %sA.addr.172.i, i64 1
  %cmp7.i = icmp eq i64* %incdec.ptr.i, %add.ptr.i592
  br i1 %cmp7.i, label %_ZN4pbbs9merge_seqIlSt4lessIlEmEEvPT_S4_PT1_mmT0_.exit.loopexit1, label %while.cond4.i

while.end.i.loopexit:                             ; preds = %while.cond4.i
  %.lcssa = phi i64 [ %179, %while.cond4.i ]
  %incdec.ptr.i.lcssa2 = phi i64* [ %incdec.ptr.i, %while.cond4.i ]
  br label %while.end.i

while.end.i:                                      ; preds = %while.end.i.loopexit, %while.cond.i
  %181 = phi i64 [ %177, %while.cond.i ], [ %.lcssa, %while.end.i.loopexit ]
  %sA.addr.1.lcssa.i = phi i64* [ %sA.addr.0.i, %while.cond.i ], [ %incdec.ptr.i.lcssa2, %while.end.i.loopexit ]
  %incdec.ptr10.i = getelementptr inbounds i64, i64* %sB.addr.0.i, i64 1
  %incdec.ptr11.i = getelementptr inbounds i64, i64* %sC.addr.0.i, i64 1
  %cmp12.i = icmp eq i64* %incdec.ptr10.i, %add.ptr2.i
  br i1 %cmp12.i, label %while.end33.i, label %if.end14.i

if.end14.i:                                       ; preds = %while.end.i
  %182 = load i64, i64* %incdec.ptr10.i, align 8, !tbaa !9
  %cmp.i66.i = icmp slt i64 %178, %182
  br i1 %cmp.i66.i, label %while.cond.i.backedge, label %if.then17.i

if.then17.i:                                      ; preds = %if.end14.i
  %cmp.i6574.i = icmp slt i64 %182, %181
  br i1 %cmp.i6574.i, label %while.end26.i, label %while.body20.lr.ph.i

while.body20.lr.ph.i:                             ; preds = %if.then17.i
  %.pre83.i = load i64, i64* %incdec.ptr11.i, align 8, !tbaa !9
  br label %while.body20.i

while.cond18.i:                                   ; preds = %while.body20.i
  %183 = load i64, i64* %incdec.ptr22.i, align 8, !tbaa !9
  %cmp.i65.i = icmp slt i64 %182, %183
  br i1 %cmp.i65.i, label %while.end26.i.loopexit, label %while.body20.i

while.body20.i:                                   ; preds = %while.cond18.i, %while.body20.lr.ph.i
  %184 = phi i64 [ %.pre83.i, %while.body20.lr.ph.i ], [ %inc21.i, %while.cond18.i ]
  %sA.addr.275.i = phi i64* [ %sA.addr.1.lcssa.i, %while.body20.lr.ph.i ], [ %incdec.ptr22.i, %while.cond18.i ]
  %inc21.i = add i64 %184, 1
  store i64 %inc21.i, i64* %incdec.ptr11.i, align 8, !tbaa !9
  %incdec.ptr22.i = getelementptr inbounds i64, i64* %sA.addr.275.i, i64 1
  %cmp23.i = icmp eq i64* %incdec.ptr22.i, %add.ptr.i592
  br i1 %cmp23.i, label %_ZN4pbbs9merge_seqIlSt4lessIlEmEEvPT_S4_PT1_mmT0_.exit.loopexit, label %while.cond18.i

while.end26.i.loopexit:                           ; preds = %while.cond18.i
  %.lcssa4 = phi i64 [ %183, %while.cond18.i ]
  %incdec.ptr22.i.lcssa3 = phi i64* [ %incdec.ptr22.i, %while.cond18.i ]
  br label %while.end26.i

while.end26.i:                                    ; preds = %while.end26.i.loopexit, %if.then17.i
  %185 = phi i64 [ %181, %if.then17.i ], [ %.lcssa4, %while.end26.i.loopexit ]
  %sA.addr.2.lcssa.i = phi i64* [ %sA.addr.1.lcssa.i, %if.then17.i ], [ %incdec.ptr22.i.lcssa3, %while.end26.i.loopexit ]
  %incdec.ptr27.i = getelementptr inbounds i64, i64* %sB.addr.0.i, i64 2
  %incdec.ptr28.i = getelementptr inbounds i64, i64* %sC.addr.0.i, i64 2
  %cmp29.i = icmp eq i64* %incdec.ptr27.i, %add.ptr2.i
  br i1 %cmp29.i, label %while.end33.i, label %while.cond.i.backedge

while.cond.i.backedge:                            ; preds = %while.end26.i, %if.end14.i
  %.be = phi i64 [ %181, %if.end14.i ], [ %185, %while.end26.i ]
  %sC.addr.0.i.be = phi i64* [ %incdec.ptr11.i, %if.end14.i ], [ %incdec.ptr28.i, %while.end26.i ]
  %sB.addr.0.i.be = phi i64* [ %incdec.ptr10.i, %if.end14.i ], [ %incdec.ptr27.i, %while.end26.i ]
  %sA.addr.0.i.be = phi i64* [ %sA.addr.1.lcssa.i, %if.end14.i ], [ %sA.addr.2.lcssa.i, %while.end26.i ]
  br label %while.cond.i

while.end33.i:                                    ; preds = %while.end26.i, %while.end.i
  %sC.addr.2.i = phi i64* [ %incdec.ptr11.i, %while.end.i ], [ %incdec.ptr28.i, %while.end26.i ]
  %sA.addr.4.i = phi i64* [ %sA.addr.1.lcssa.i, %while.end.i ], [ %sA.addr.2.lcssa.i, %while.end26.i ]
  %sub.ptr.lhs.cast.i = ptrtoint i64* %add.ptr.i592 to i64
  %sub.ptr.rhs.cast.i = ptrtoint i64* %sA.addr.4.i to i64
  %sub.ptr.sub.i597 = sub i64 %sub.ptr.lhs.cast.i, %sub.ptr.rhs.cast.i
  %sub.ptr.div.i598 = ashr exact i64 %sub.ptr.sub.i597, 3
  store i64 %sub.ptr.div.i598, i64* %sC.addr.2.i, align 8, !tbaa !9
  br label %_ZN4pbbs9merge_seqIlSt4lessIlEmEEvPT_S4_PT1_mmT0_.exit

_ZN4pbbs9merge_seqIlSt4lessIlEmEEvPT_S4_PT1_mmT0_.exit.loopexit: ; preds = %while.body20.i
  br label %_ZN4pbbs9merge_seqIlSt4lessIlEmEEvPT_S4_PT1_mmT0_.exit

_ZN4pbbs9merge_seqIlSt4lessIlEmEEvPT_S4_PT1_mmT0_.exit.loopexit1: ; preds = %while.body5.i
  br label %_ZN4pbbs9merge_seqIlSt4lessIlEmEEvPT_S4_PT1_mmT0_.exit

_ZN4pbbs9merge_seqIlSt4lessIlEmEEvPT_S4_PT1_mmT0_.exit: ; preds = %_ZN4pbbs9merge_seqIlSt4lessIlEmEEvPT_S4_PT1_mmT0_.exit.loopexit1, %_ZN4pbbs9merge_seqIlSt4lessIlEmEEvPT_S4_PT1_mmT0_.exit.loopexit, %while.end33.i, %invoke.cont148
  reattach within %syncreg111, label %pfor.inc159

pfor.inc159:                                      ; preds = %_ZN4pbbs9merge_seqIlSt4lessIlEmEEvPT_S4_PT1_mmT0_.exit, %pfor.detach122
  %inc160 = add nuw i64 %__begin113.0663, 1
  %indvars.iv.next = add i64 %indvars.iv, %add29
  %exitcond7 = icmp ne i64 %inc160, %umax6
  br i1 %exitcond7, label %pfor.detach122, label %pfor.cond.cleanup121, !llvm.loop !32

lpad147:                                          ; preds = %if.end146
  %186 = landingpad { i8*, i32 }
          catch i8* null
  invoke void @llvm.detached.rethrow.sl_p0i8i32s(token %syncreg111, { i8*, i32 } %186)
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
  %187 = extractvalue { i8*, i32 } %lpad.phi656, 0
  %188 = extractvalue { i8*, i32 } %lpad.phi656, 1
  sync within %syncreg111, label %ehcleanup284

sync.continue168:                                 ; preds = %pfor.cond.cleanup121
  %189 = lshr i64 %sub.ptr.div.i, 3
  %add.i600 = shl i64 %189, 6
  %mul1.i601 = add i64 %add.i600, 64
  %call.i602 = tail call noalias i8* @aligned_alloc(i64 64, i64 %mul1.i601) #4
  %190 = bitcast i8* %call.i602 to i64*
  %cmp.i603 = icmp eq i8* %call.i602, null
  br i1 %cmp.i603, label %if.then.i604, label %if.end.i605

if.then.i604:                                     ; preds = %sync.continue168
  %191 = load %struct._IO_FILE*, %struct._IO_FILE** @stderr, align 8, !tbaa !8
  %192 = tail call i64 @fwrite(i8* getelementptr inbounds ([22 x i8], [22 x i8]* @.str.47, i64 0, i64 0), i64 21, i64 1, %struct._IO_FILE* %191) #7
  tail call void @exit(i32 1) #8
  unreachable

if.end.i605:                                      ; preds = %sync.continue168
  %cmp420.i606 = icmp eq i64 %mul1.i601, 0
  br i1 %cmp420.i606, label %pfor.cond.cleanup.i608, label %pfor.detach.i610.preheader

pfor.detach.i610.preheader:                       ; preds = %if.end.i605
  br label %pfor.detach.i610

pfor.cond.cleanup.i608.loopexit:                  ; preds = %pfor.inc.i615
  br label %pfor.cond.cleanup.i608

pfor.cond.cleanup.i608:                           ; preds = %pfor.cond.cleanup.i608.loopexit, %if.end.i605
  sync within %syncreg.i599, label %invoke.cont176

pfor.detach.i610:                                 ; preds = %pfor.inc.i615, %pfor.detach.i610.preheader
  %i.021.i609 = phi i64 [ %add5.i613, %pfor.inc.i615 ], [ 0, %pfor.detach.i610.preheader ]
  detach within %syncreg.i599, label %pfor.body.i612, label %pfor.inc.i615

pfor.body.i612:                                   ; preds = %pfor.detach.i610
  %arrayidx.i611 = getelementptr inbounds i8, i8* %call.i602, i64 %i.021.i609
  store i8 0, i8* %arrayidx.i611, align 1, !tbaa !18
  reattach within %syncreg.i599, label %pfor.inc.i615

pfor.inc.i615:                                    ; preds = %pfor.body.i612, %pfor.detach.i610
  %add5.i613 = add i64 %i.021.i609, 2097152
  %cmp4.i614 = icmp ult i64 %add5.i613, %mul1.i601
  br i1 %cmp4.i614, label %pfor.detach.i610, label %pfor.cond.cleanup.i608.loopexit, !llvm.loop !19

invoke.cont176:                                   ; preds = %pfor.cond.cleanup.i608
  %call180 = invoke i64* @_ZN4pbbs17transpose_bucketsIlmEEPmPT_S3_PT0_mmmm(i64* %56, i64* %190, i64* %59, i64 %sub.ptr.div.i, i64 %add29, i64 %conv27, i64 %add31)
          to label %invoke.cont179 unwind label %lpad178

invoke.cont179:                                   ; preds = %invoke.cont176
  tail call void @free(i8* %call.i575) #4
  %193 = add i64 %div, 1
  br label %pfor.detach192

pfor.cond.cleanup191:                             ; preds = %pfor.inc229
  sync within %syncreg181, label %sync.continue238

lpad178:                                          ; preds = %invoke.cont176
  %194 = landingpad { i8*, i32 }
          cleanup
  %195 = extractvalue { i8*, i32 } %194, 0
  %196 = extractvalue { i8*, i32 } %194, 1
  br label %ehcleanup284

pfor.detach192:                                   ; preds = %pfor.inc229, %invoke.cont179
  %__begin183.0660 = phi i64 [ 0, %invoke.cont179 ], [ %inc230, %pfor.inc229 ]
  detach within %syncreg181, label %pfor.body197, label %pfor.inc229 unwind label %lpad231.loopexit

pfor.body197:                                     ; preds = %pfor.detach192
  %arrayidx198 = getelementptr inbounds i64, i64* %call180, i64 %__begin183.0660
  %197 = load i64, i64* %arrayidx198, align 8, !tbaa !9
  %add199 = add nuw nsw i64 %__begin183.0660, 1
  %arrayidx200 = getelementptr inbounds i64, i64* %call180, i64 %add199
  %198 = load i64, i64* %arrayidx200, align 8, !tbaa !9
  %cmp201 = icmp eq i64 %__begin183.0660, 0
  %cmp203 = icmp eq i64 %__begin183.0660, %div
  %or.cond = or i1 %cmp201, %cmp203
  br i1 %or.cond, label %if.then213, label %lor.lhs.false204

lor.lhs.false204:                                 ; preds = %pfor.body197
  %sub205 = add nsw i64 %__begin183.0660, -1
  %arrayidx206 = getelementptr inbounds i64, i64* %44, i64 %sub205
  %arrayidx207 = getelementptr inbounds i64, i64* %44, i64 %__begin183.0660
  %199 = load i64, i64* %arrayidx206, align 8, !tbaa !9
  %200 = load i64, i64* %arrayidx207, align 8, !tbaa !9
  %cmp.i501 = icmp slt i64 %199, %200
  br i1 %cmp.i501, label %if.then213, label %if.end217

if.then213:                                       ; preds = %lor.lhs.false204, %pfor.body197
  %add.ptr214 = getelementptr inbounds i64, i64* %190, i64 %197
  %sub215 = sub i64 %198, %197
  invoke void @_ZN4pbbs9quicksortIlSt4lessIlEEEvPT_mRKT0_(i64* %add.ptr214, i64 %sub215, %"struct.std::less"* nonnull dereferenceable(1) %f)
          to label %if.end217 unwind label %lpad208

lpad208:                                          ; preds = %if.then213
  %201 = landingpad { i8*, i32 }
          catch i8* null
  invoke void @llvm.detached.rethrow.sl_p0i8i32s(token %syncreg181, { i8*, i32 } %201)
          to label %det.rethrow.unreachable237 unwind label %lpad231.loopexit.split-lp

det.rethrow.unreachable237:                       ; preds = %lpad208
  unreachable

if.end217:                                        ; preds = %if.then213, %lor.lhs.false204
  br i1 %inplace, label %if.then219, label %if.end224

if.then219:                                       ; preds = %if.end217
  %add.ptr220 = getelementptr inbounds i64, i64* %56, i64 %197
  %202 = bitcast i64* %add.ptr220 to i8*
  %add.ptr221 = getelementptr inbounds i64, i64* %190, i64 %197
  %203 = bitcast i64* %add.ptr221 to i8*
  %sub222 = sub i64 %198, %197
  %mul223 = shl i64 %sub222, 3
  tail call void @llvm.memcpy.p0i8.p0i8.i64(i8* %202, i8* %203, i64 %mul223, i32 1, i1 false)
  br label %if.end224

if.end224:                                        ; preds = %if.then219, %if.end217
  reattach within %syncreg181, label %pfor.inc229

pfor.inc229:                                      ; preds = %if.end224, %pfor.detach192
  %inc230 = add nuw nsw i64 %__begin183.0660, 1
  %exitcond5 = icmp ne i64 %inc230, %193
  br i1 %exitcond5, label %pfor.detach192, label %pfor.cond.cleanup191, !llvm.loop !33

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
  %204 = extractvalue { i8*, i32 } %lpad.phi, 0
  %205 = extractvalue { i8*, i32 } %lpad.phi, 1
  sync within %syncreg181, label %ehcleanup284

sync.continue238:                                 ; preds = %pfor.cond.cleanup191
  tail call void @free(i8* %call.i.i512) #4
  %206 = bitcast i64* %call180 to i8*
  tail call void @free(i8* %206) #4
  br i1 %inplace, label %if.then248, label %if.else250

if.then248:                                       ; preds = %sync.continue238
  tail call void @free(i8* %call.i602) #4
  %207 = bitcast %struct.sequence.46* %agg.result to i64*
  store i64 %Bs.sroa.0.0, i64* %207, align 8, !tbaa !6
  %e.i455 = getelementptr inbounds %struct.sequence.46, %struct.sequence.46* %agg.result, i64 0, i32 1
  %208 = bitcast i64** %e.i455 to i64*
  store i64 %Bs.sroa.16.0, i64* %208, align 8, !tbaa !0
  %allocated.i456 = getelementptr inbounds %struct.sequence.46, %struct.sequence.46* %agg.result, i64 0, i32 2
  store i8 0, i8* %allocated.i456, align 8, !tbaa !7
  br label %cleanup283

if.else250:                                       ; preds = %sync.continue238
  %209 = inttoptr i64 %Bs.sroa.0.0 to i8*
  tail call void @free(i8* %209) #4
  %210 = bitcast %struct.sequence.46* %agg.result to i8**
  store i8* %call.i602, i8** %210, align 8, !tbaa !6
  %e.i454 = getelementptr inbounds %struct.sequence.46, %struct.sequence.46* %agg.result, i64 0, i32 1
  %add.ptr.i = getelementptr inbounds i64, i64* %190, i64 %sub.ptr.div.i
  store i64* %add.ptr.i, i64** %e.i454, align 8, !tbaa !0
  %allocated3.i = getelementptr inbounds %struct.sequence.46, %struct.sequence.46* %agg.result, i64 0, i32 2
  store i8 1, i8* %allocated3.i, align 8, !tbaa !7
  br label %cleanup283

cleanup283:                                       ; preds = %if.else250, %if.then248, %if.end
  ret void

ehcleanup284:                                     ; preds = %lpad231, %lpad178, %lpad161, %lpad49, %_ZN8sequenceIlED2Ev.exit497
  %ehselector.slot.10 = phi i32 [ %ehselector.slot.2675, %_ZN8sequenceIlED2Ev.exit497 ], [ %40, %lpad49 ], [ %188, %lpad161 ], [ %196, %lpad178 ], [ %205, %lpad231 ]
  %exn.slot.10 = phi i8* [ %exn.slot.2676, %_ZN8sequenceIlED2Ev.exit497 ], [ %39, %lpad49 ], [ %187, %lpad161 ], [ %195, %lpad178 ], [ %204, %lpad231 ]
  %lpad.val288 = insertvalue { i8*, i32 } undef, i8* %exn.slot.10, 0
  %lpad.val289 = insertvalue { i8*, i32 } %lpad.val288, i32 %ehselector.slot.10, 1
  resume { i8*, i32 } %lpad.val289
}

attributes #0 = { uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="haswell" "target-features"="+aes,+avx,+avx2,+bmi,+bmi2,+cmov,+cx16,+f16c,+fma,+fsgsbase,+fxsr,+lzcnt,+mmx,+movbe,+pclmul,+popcnt,+rdrnd,+sse,+sse2,+sse3,+sse4.1,+sse4.2,+ssse3,+x87,+xsave,+xsaveopt,-adx,-avx512bitalg,-avx512bw,-avx512cd,-avx512dq,-avx512er,-avx512f,-avx512ifma,-avx512pf,-avx512vbmi,-avx512vbmi2,-avx512vl,-avx512vnni,-avx512vpopcntdq,-clflushopt,-clwb,-clzero,-fma4,-gfni,-ibt,-lwp,-mwaitx,-pku,-prefetchwt1,-prfchw,-rdseed,-rtm,-sgx,-sha,-shstk,-sse4a,-tbm,-vaes,-vpclmulqdq,-xop,-xsavec,-xsaves" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { argmemonly nounwind }
attributes #2 = { nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="haswell" "target-features"="+aes,+avx,+avx2,+bmi,+bmi2,+cmov,+cx16,+f16c,+fma,+fsgsbase,+fxsr,+lzcnt,+mmx,+movbe,+pclmul,+popcnt,+rdrnd,+sse,+sse2,+sse3,+sse4.1,+sse4.2,+ssse3,+x87,+xsave,+xsaveopt,-adx,-avx512bitalg,-avx512bw,-avx512cd,-avx512dq,-avx512er,-avx512f,-avx512ifma,-avx512pf,-avx512vbmi,-avx512vbmi2,-avx512vl,-avx512vnni,-avx512vpopcntdq,-clflushopt,-clwb,-clzero,-fma4,-gfni,-ibt,-lwp,-mwaitx,-pku,-prefetchwt1,-prfchw,-rdseed,-rtm,-sgx,-sha,-shstk,-sse4a,-tbm,-vaes,-vpclmulqdq,-xop,-xsavec,-xsaves" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #3 = { noreturn nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="haswell" "target-features"="+aes,+avx,+avx2,+bmi,+bmi2,+cmov,+cx16,+f16c,+fma,+fsgsbase,+fxsr,+lzcnt,+mmx,+movbe,+pclmul,+popcnt,+rdrnd,+sse,+sse2,+sse3,+sse4.1,+sse4.2,+ssse3,+x87,+xsave,+xsaveopt,-adx,-avx512bitalg,-avx512bw,-avx512cd,-avx512dq,-avx512er,-avx512f,-avx512ifma,-avx512pf,-avx512vbmi,-avx512vbmi2,-avx512vl,-avx512vnni,-avx512vpopcntdq,-clflushopt,-clwb,-clzero,-fma4,-gfni,-ibt,-lwp,-mwaitx,-pku,-prefetchwt1,-prfchw,-rdseed,-rtm,-sgx,-sha,-shstk,-sse4a,-tbm,-vaes,-vpclmulqdq,-xop,-xsavec,-xsaves" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #4 = { nounwind }
attributes #5 = { nounwind readnone speculatable }
attributes #6 = { argmemonly }
attributes #7 = { cold }
attributes #8 = { noreturn nounwind }

!0 = !{!1, !2, i64 8}
!1 = !{!"_ZTS8sequenceIlE", !2, i64 0, !2, i64 8, !5, i64 16}
!2 = !{!"any pointer", !3, i64 0}
!3 = !{!"omnipotent char", !4, i64 0}
!4 = !{!"Simple C++ TBAA"}
!5 = !{!"bool", !3, i64 0}
!6 = !{!1, !2, i64 0}
!7 = !{!1, !5, i64 16}
!8 = !{!2, !2, i64 0}
!9 = !{!10, !10, i64 0}
!10 = !{!"long", !3, i64 0}
!11 = distinct !{!11, !12}
!12 = !{!"tapir.loop.spawn.strategy", i32 1}
!13 = distinct !{!13, !12}
!14 = distinct !{!14, !12}
!15 = !{!16}
!16 = distinct !{!16, !17, !"_ZN8sequenceIlE11as_sequenceEv: %agg.result"}
!17 = distinct !{!17, !"_ZN8sequenceIlE11as_sequenceEv"}
!18 = !{!5, !5, i64 0}
!19 = distinct !{!19, !12}
!20 = distinct !{!20, !12}
!21 = !{!22}
!22 = distinct !{!22, !23}
!23 = distinct !{!23, !"LVerDomain"}
!24 = !{!25}
!25 = distinct !{!25, !23}
!26 = distinct !{!26, !27}
!27 = !{!"llvm.loop.isvectorized", i32 1}
!28 = distinct !{!28, !29}
!29 = !{!"llvm.loop.unroll.disable"}
!30 = distinct !{!30, !29}
!31 = distinct !{!31, !27}
!32 = distinct !{!32, !12}
!33 = distinct !{!33, !12}
