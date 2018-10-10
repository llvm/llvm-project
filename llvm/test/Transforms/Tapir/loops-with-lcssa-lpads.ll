; RUN: opt < %s -passes=loop-spawning -S | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct.transpose = type { i32*, i32* }
%struct.blockTrans = type { i32*, i32*, i32*, i32*, i32* }

$_Z10sampleSortIiSt4lessIiEiEvPT_T1_T0_ = comdat any

; Function Attrs: uwtable
define linkonce_odr void @_Z10sampleSortIiSt4lessIiEiEvPT_T1_T0_(i32* %A, i32 %n) local_unnamed_addr #4 comdat personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*) {
entry:
  %syncreg = tail call token @llvm.syncregion.start()
  %syncreg31 = tail call token @llvm.syncregion.start()
  %syncreg61 = tail call token @llvm.syncregion.start()
  %ref.tmp = alloca %struct.transpose, align 8
  %ref.tmp130 = alloca %struct.blockTrans, align 8
  %syncreg132 = tail call token @llvm.syncregion.start()
  %cmp = icmp slt i32 %n, 1000
  br i1 %cmp, label %if.then, label %if.else

if.then:                                          ; preds = %entry
  tail call void @_Z9quickSortIiSt4lessIiEiEvPT_T1_T0_(i32* %A, i32 %n)
  br label %if.end208

if.else:                                          ; preds = %entry
  %conv.i = sitofp i32 %n to double
  %0 = tail call fast double @llvm.sqrt.f64(double %conv.i)
  %1 = tail call fast double @llvm.ceil.f64(double %0)
  %conv = fptosi double %1 to i64
  %div = sdiv i64 %conv, 2
  %add = add nsw i64 %div, 1
  %sub = add nsw i32 %n, -1
  %conv1 = sext i32 %sub to i64
  %div2 = sdiv i64 %conv1, %add
  %add3 = add nsw i64 %div2, 1
  %conv6 = trunc i64 %add to i32
  %sext = mul i64 %add, 42949672960
  %conv7 = ashr exact i64 %sext, 32
  %mul8 = ashr exact i64 %sext, 30
  %call9 = tail call noalias i8* @malloc(i64 %mul8) #2
  %2 = bitcast i8* %call9 to i32*
  %cmp14386 = icmp sgt i64 %sext, 0
  br i1 %cmp14386, label %pfor.detach.lr.ph, label %pfor.cond.cleanup

pfor.detach.lr.ph:                                ; preds = %if.else
  %conv18 = sext i32 %n to i64
  %3 = icmp sgt i64 %conv7, 1
  %smax398 = select i1 %3, i64 %conv7, i64 1
  br label %pfor.detach

pfor.cond.cleanup:                                ; preds = %pfor.inc, %if.else
  sync within %syncreg, label %sync.continue

pfor.detach:                                      ; preds = %pfor.inc, %pfor.detach.lr.ph
  %__begin.0387 = phi i64 [ 0, %pfor.detach.lr.ph ], [ %inc, %pfor.inc ]
  detach within %syncreg, label %pfor.body, label %pfor.inc unwind label %lpad20

pfor.body:                                        ; preds = %pfor.detach
  %mul.i = mul nuw nsw i64 %__begin.0387, 982451653
  %add.i = add nuw nsw i64 %mul.i, 12345
  %rem = urem i64 %add.i, %conv18
  %arrayidx = getelementptr inbounds i32, i32* %A, i64 %rem
  %4 = load i32, i32* %arrayidx, align 4, !tbaa !42
  %arrayidx19 = getelementptr inbounds i32, i32* %2, i64 %__begin.0387
  store i32 %4, i32* %arrayidx19, align 4, !tbaa !42
  reattach within %syncreg, label %pfor.inc

pfor.inc:                                         ; preds = %pfor.detach, %pfor.body
  %inc = add nuw nsw i64 %__begin.0387, 1
  %exitcond399 = icmp eq i64 %inc, %smax398
  br i1 %exitcond399, label %pfor.cond.cleanup, label %pfor.detach, !llvm.loop !139

lpad20:                                           ; preds = %pfor.detach
  %5 = landingpad { i8*, i32 }
          cleanup
  %6 = extractvalue { i8*, i32 } %5, 0
  %7 = extractvalue { i8*, i32 } %5, 1
  sync within %syncreg, label %ehcleanup202

sync.continue:                                    ; preds = %pfor.cond.cleanup
  tail call void @_Z9quickSortIiSt4lessIiElEvPT_T1_T0_(i32* %2, i64 %conv7)
  %sub27 = shl i64 %add, 32
  %sext362 = add i64 %sub27, -4294967296
  %conv28 = ashr exact i64 %sext362, 32
  %mul29 = ashr exact i64 %sext362, 30
  %call30 = tail call noalias i8* @malloc(i64 %mul29) #2
  %8 = bitcast i8* %call30 to i32*
  %cmp42384 = icmp sgt i64 %sext362, 0
  br i1 %cmp42384, label %pfor.detach44.lr.ph, label %pfor.cond.cleanup43

pfor.detach44.lr.ph:                              ; preds = %sync.continue
  %9 = icmp sgt i64 %conv28, 1
  %smax396 = select i1 %9, i64 %conv28, i64 1
  br label %pfor.detach44

pfor.cond.cleanup43:                              ; preds = %pfor.inc53, %sync.continue
  sync within %syncreg31, label %sync.continue55

pfor.detach44:                                    ; preds = %pfor.inc53, %pfor.detach44.lr.ph
  %__begin33.0385 = phi i64 [ 0, %pfor.detach44.lr.ph ], [ %inc54, %pfor.inc53 ]
  detach within %syncreg31, label %pfor.body48, label %pfor.inc53

pfor.body48:                                      ; preds = %pfor.detach44
  %mul49 = mul nuw nsw i64 %__begin33.0385, 10
  %arrayidx50 = getelementptr inbounds i32, i32* %2, i64 %mul49
  %10 = load i32, i32* %arrayidx50, align 4, !tbaa !42
  %arrayidx51 = getelementptr inbounds i32, i32* %8, i64 %__begin33.0385
  store i32 %10, i32* %arrayidx51, align 4, !tbaa !42
  reattach within %syncreg31, label %pfor.inc53

pfor.inc53:                                       ; preds = %pfor.body48, %pfor.detach44
  %inc54 = add nuw nsw i64 %__begin33.0385, 1
  %exitcond397 = icmp eq i64 %inc54, %smax396
  br i1 %exitcond397, label %pfor.cond.cleanup43, label %pfor.detach44, !llvm.loop !140

sync.continue55:                                  ; preds = %pfor.cond.cleanup43
  tail call void @free(i8* %call9) #2
  %conv57 = ashr exact i64 %sub27, 32
  %mul58 = mul nsw i64 %conv57, %add
  %mul59 = shl i64 %mul58, 2
  %call60 = tail call noalias i8* @malloc(i64 %mul59) #2
  %11 = bitcast i8* %call60 to i32*
  %cmp70381 = icmp slt i64 %conv, -1
  br i1 %cmp70381, label %pfor.cond.cleanup71, label %pfor.detach73.lr.ph

pfor.detach73.lr.ph:                              ; preds = %sync.continue55
  %conv81 = sext i32 %n to i64
  %cmp1.i = icmp eq i64 %sext362, 0
  %add.ptr2.i = getelementptr inbounds i32, i32* %8, i64 %conv28
  %cmp377.i = icmp slt i64 %sext362, 0
  %12 = add nsw i64 %mul29, 4
  br i1 %cmp1.i, label %pfor.detach73.us.preheader, label %pfor.detach73.preheader

pfor.detach73.preheader:                          ; preds = %pfor.detach73.lr.ph
  br label %pfor.detach73

pfor.detach73.us.preheader:                       ; preds = %pfor.detach73.lr.ph
  br label %pfor.detach73.us

pfor.detach73.us:                                 ; preds = %pfor.detach73.us.preheader, %pfor.inc97.us
  %__begin63.0382.us = phi i64 [ %inc98.us, %pfor.inc97.us ], [ 0, %pfor.detach73.us.preheader ]
  detach within %syncreg61, label %pfor.body77.us, label %pfor.inc97.us unwind label %lpad99.loopexit.us-lcssa.us

pfor.body77.us:                                   ; preds = %pfor.detach73.us
  %mul78.us = mul nsw i64 %__begin63.0382.us, %add3
  %cmp80.us = icmp slt i64 %__begin63.0382.us, %div
  %sub82.us = sub nsw i64 %conv81, %mul78.us
  %cond.us = select i1 %cmp80.us, i64 %add3, i64 %sub82.us
  %add.ptr.us = getelementptr inbounds i32, i32* %A, i64 %mul78.us
  invoke void @_Z9quickSortIiSt4lessIiElEvPT_T1_T0_(i32* %add.ptr.us, i64 %cond.us)
          to label %invoke.cont87.us unwind label %lpad84.us-lcssa.us

invoke.cont87.us:                                 ; preds = %pfor.body77.us
  reattach within %syncreg61, label %pfor.inc97.us

pfor.inc97.us:                                    ; preds = %invoke.cont87.us, %pfor.detach73.us
  %inc98.us = add nuw nsw i64 %__begin63.0382.us, 1
  %exitcond394 = icmp eq i64 %inc98.us, %add
  br i1 %exitcond394, label %pfor.cond.cleanup71, label %pfor.detach73.us, !llvm.loop !141

lpad99.loopexit.us-lcssa.us:                      ; preds = %pfor.detach73.us
  %lpad.us-lcssa.us = landingpad { i8*, i32 }
          cleanup
  br label %lpad99

lpad84.us-lcssa.us:                               ; preds = %pfor.body77.us
  %lpad.us-lcssa383.us = landingpad { i8*, i32 }
          catch i8* null
  br label %lpad84

pfor.cond.cleanup71:                              ; preds = %pfor.inc97, %pfor.inc97.us, %sync.continue55
  sync within %syncreg61, label %sync.continue106

pfor.detach73:                                    ; preds = %pfor.detach73.preheader, %pfor.inc97
  %__begin63.0382 = phi i64 [ %inc98, %pfor.inc97 ], [ 0, %pfor.detach73.preheader ]
  detach within %syncreg61, label %pfor.body77, label %pfor.inc97 unwind label %lpad99.loopexit.us-lcssa

pfor.body77:                                      ; preds = %pfor.detach73
  %mul78 = mul nsw i64 %__begin63.0382, %add3
  %cmp80 = icmp slt i64 %__begin63.0382, %div
  %sub82 = sub nsw i64 %conv81, %mul78
  %cond = select i1 %cmp80, i64 %add3, i64 %sub82
  %add.ptr = getelementptr inbounds i32, i32* %A, i64 %mul78
  invoke void @_Z9quickSortIiSt4lessIiElEvPT_T1_T0_(i32* %add.ptr, i64 %cond)
          to label %invoke.cont87 unwind label %lpad84.us-lcssa

invoke.cont87:                                    ; preds = %pfor.body77
  %mul90 = mul nsw i64 %__begin63.0382, %conv57
  %add.ptr91 = getelementptr inbounds i32, i32* %11, i64 %mul90
  %sC83.i = bitcast i32* %add.ptr91 to i8*
  %cmp.i364 = icmp eq i64 %cond, 0
  br i1 %cmp.i364, label %invoke.cont95, label %if.end.i

if.end.i:                                         ; preds = %invoke.cont87
  %add.ptr.i = getelementptr inbounds i32, i32* %add.ptr, i64 %cond
  br i1 %cmp377.i, label %for.cond.cleanup.i, label %for.body.lr.ph.i

for.body.lr.ph.i:                                 ; preds = %if.end.i
  tail call void @llvm.memset.p0i8.i64(i8* %sC83.i, i8 0, i64 %12, i32 4, i1 false)
  br label %for.cond.cleanup.i

for.cond.cleanup.i:                               ; preds = %for.body.lr.ph.i, %if.end.i
  %.pre.i = load i32, i32* %add.ptr, align 4, !tbaa !42
  br label %while.cond.i

while.cond.i:                                     ; preds = %while.cond.i.backedge, %for.cond.cleanup.i
  %13 = phi i32 [ %.pre.i, %for.cond.cleanup.i ], [ %.be, %while.cond.i.backedge ]
  %sC.addr.0.i = phi i32* [ %add.ptr91, %for.cond.cleanup.i ], [ %sC.addr.0.i.be, %while.cond.i.backedge ]
  %sB.addr.0.i = phi i32* [ %8, %for.cond.cleanup.i ], [ %sB.addr.0.i.be, %while.cond.i.backedge ]
  %sA.addr.0.i = phi i32* [ %add.ptr, %for.cond.cleanup.i ], [ %sA.addr.0.i.be, %while.cond.i.backedge ]
  %14 = load i32, i32* %sB.addr.0.i, align 4, !tbaa !42
  %cmp.i71.i = icmp slt i32 %13, %14
  br i1 %cmp.i71.i, label %while.body5.lr.ph.i, label %while.end.i

while.body5.lr.ph.i:                              ; preds = %while.cond.i
  %.pre84.i = load i32, i32* %sC.addr.0.i, align 4, !tbaa !42
  br label %while.body5.i

while.cond4.i:                                    ; preds = %while.body5.i
  %15 = load i32, i32* %incdec.ptr.i, align 4, !tbaa !42
  %cmp.i.i = icmp slt i32 %15, %14
  br i1 %cmp.i.i, label %while.body5.i, label %while.end.i

while.body5.i:                                    ; preds = %while.cond4.i, %while.body5.lr.ph.i
  %16 = phi i32 [ %.pre84.i, %while.body5.lr.ph.i ], [ %inc6.i, %while.cond4.i ]
  %sA.addr.172.i = phi i32* [ %sA.addr.0.i, %while.body5.lr.ph.i ], [ %incdec.ptr.i, %while.cond4.i ]
  %inc6.i = add nsw i32 %16, 1
  store i32 %inc6.i, i32* %sC.addr.0.i, align 4, !tbaa !42
  %incdec.ptr.i = getelementptr inbounds i32, i32* %sA.addr.172.i, i64 1
  %cmp7.i = icmp eq i32* %incdec.ptr.i, %add.ptr.i
  br i1 %cmp7.i, label %invoke.cont95, label %while.cond4.i

while.end.i:                                      ; preds = %while.cond4.i, %while.cond.i
  %17 = phi i32 [ %13, %while.cond.i ], [ %15, %while.cond4.i ]
  %sA.addr.1.lcssa.i = phi i32* [ %sA.addr.0.i, %while.cond.i ], [ %incdec.ptr.i, %while.cond4.i ]
  %incdec.ptr10.i = getelementptr inbounds i32, i32* %sB.addr.0.i, i64 1
  %incdec.ptr11.i = getelementptr inbounds i32, i32* %sC.addr.0.i, i64 1
  %cmp12.i = icmp eq i32* %incdec.ptr10.i, %add.ptr2.i
  br i1 %cmp12.i, label %while.end33.i, label %if.end14.i

if.end14.i:                                       ; preds = %while.end.i
  %18 = load i32, i32* %incdec.ptr10.i, align 4, !tbaa !42
  %cmp.i66.i = icmp slt i32 %14, %18
  br i1 %cmp.i66.i, label %while.cond.i.backedge, label %if.then17.i

if.then17.i:                                      ; preds = %if.end14.i
  %cmp.i6574.i = icmp slt i32 %18, %17
  br i1 %cmp.i6574.i, label %while.end26.i, label %while.body20.lr.ph.i

while.body20.lr.ph.i:                             ; preds = %if.then17.i
  %.pre85.i = load i32, i32* %incdec.ptr11.i, align 4, !tbaa !42
  br label %while.body20.i

while.cond18.i:                                   ; preds = %while.body20.i
  %19 = load i32, i32* %incdec.ptr22.i, align 4, !tbaa !42
  %cmp.i65.i = icmp slt i32 %18, %19
  br i1 %cmp.i65.i, label %while.end26.i, label %while.body20.i

while.body20.i:                                   ; preds = %while.cond18.i, %while.body20.lr.ph.i
  %20 = phi i32 [ %.pre85.i, %while.body20.lr.ph.i ], [ %inc21.i, %while.cond18.i ]
  %sA.addr.275.i = phi i32* [ %sA.addr.1.lcssa.i, %while.body20.lr.ph.i ], [ %incdec.ptr22.i, %while.cond18.i ]
  %inc21.i = add nsw i32 %20, 1
  store i32 %inc21.i, i32* %incdec.ptr11.i, align 4, !tbaa !42
  %incdec.ptr22.i = getelementptr inbounds i32, i32* %sA.addr.275.i, i64 1
  %cmp23.i = icmp eq i32* %incdec.ptr22.i, %add.ptr.i
  br i1 %cmp23.i, label %invoke.cont95, label %while.cond18.i

while.end26.i:                                    ; preds = %while.cond18.i, %if.then17.i
  %21 = phi i32 [ %17, %if.then17.i ], [ %19, %while.cond18.i ]
  %sA.addr.2.lcssa.i = phi i32* [ %sA.addr.1.lcssa.i, %if.then17.i ], [ %incdec.ptr22.i, %while.cond18.i ]
  %incdec.ptr27.i = getelementptr inbounds i32, i32* %sB.addr.0.i, i64 2
  %incdec.ptr28.i = getelementptr inbounds i32, i32* %sC.addr.0.i, i64 2
  %cmp29.i = icmp eq i32* %incdec.ptr27.i, %add.ptr2.i
  br i1 %cmp29.i, label %while.end33.i, label %while.cond.i.backedge

while.cond.i.backedge:                            ; preds = %while.end26.i, %if.end14.i
  %.be = phi i32 [ %17, %if.end14.i ], [ %21, %while.end26.i ]
  %sC.addr.0.i.be = phi i32* [ %incdec.ptr11.i, %if.end14.i ], [ %incdec.ptr28.i, %while.end26.i ]
  %sB.addr.0.i.be = phi i32* [ %incdec.ptr10.i, %if.end14.i ], [ %incdec.ptr27.i, %while.end26.i ]
  %sA.addr.0.i.be = phi i32* [ %sA.addr.1.lcssa.i, %if.end14.i ], [ %sA.addr.2.lcssa.i, %while.end26.i ]
  br label %while.cond.i

while.end33.i:                                    ; preds = %while.end26.i, %while.end.i
  %sC.addr.2.i = phi i32* [ %incdec.ptr11.i, %while.end.i ], [ %incdec.ptr28.i, %while.end26.i ]
  %sA.addr.4.i = phi i32* [ %sA.addr.1.lcssa.i, %while.end.i ], [ %sA.addr.2.lcssa.i, %while.end26.i ]
  %sub.ptr.lhs.cast.i = ptrtoint i32* %add.ptr.i to i64
  %sub.ptr.rhs.cast.i = ptrtoint i32* %sA.addr.4.i to i64
  %sub.ptr.sub.i = sub i64 %sub.ptr.lhs.cast.i, %sub.ptr.rhs.cast.i
  %22 = lshr exact i64 %sub.ptr.sub.i, 2
  %conv.i365 = trunc i64 %22 to i32
  store i32 %conv.i365, i32* %sC.addr.2.i, align 4, !tbaa !42
  br label %invoke.cont95

invoke.cont95:                                    ; preds = %while.body5.i, %while.body20.i, %while.end33.i, %invoke.cont87
  reattach within %syncreg61, label %pfor.inc97

pfor.inc97:                                       ; preds = %pfor.detach73, %invoke.cont95
  %inc98 = add nuw nsw i64 %__begin63.0382, 1
  %exitcond395 = icmp eq i64 %inc98, %add
  br i1 %exitcond395, label %pfor.cond.cleanup71, label %pfor.detach73, !llvm.loop !141

lpad84.us-lcssa:                                  ; preds = %pfor.body77
  %lpad.us-lcssa383 = landingpad { i8*, i32 }
          catch i8* null
  br label %lpad84

lpad84:                                           ; preds = %lpad84.us-lcssa.us, %lpad84.us-lcssa
  %23 = phi { i8*, i32 } [ %lpad.us-lcssa383.us, %lpad84.us-lcssa.us ], [ %lpad.us-lcssa383, %lpad84.us-lcssa ]
  invoke void @llvm.detached.rethrow.sl_p0i8i32s(token %syncreg61, { i8*, i32 } %23)
          to label %det.rethrow.unreachable105 unwind label %lpad99.loopexit.split-lp

det.rethrow.unreachable105:                       ; preds = %lpad84
  unreachable

lpad99.loopexit.us-lcssa:                         ; preds = %pfor.detach73
  %lpad.us-lcssa = landingpad { i8*, i32 }
          cleanup
  br label %lpad99

lpad99.loopexit.split-lp:                         ; preds = %lpad84
  %lpad.loopexit.split-lp371 = landingpad { i8*, i32 }
          cleanup
  br label %lpad99

lpad99:                                           ; preds = %lpad99.loopexit.us-lcssa, %lpad99.loopexit.us-lcssa.us, %lpad99.loopexit.split-lp
  %lpad.phi372 = phi { i8*, i32 } [ %lpad.loopexit.split-lp371, %lpad99.loopexit.split-lp ], [ %lpad.us-lcssa.us, %lpad99.loopexit.us-lcssa.us ], [ %lpad.us-lcssa, %lpad99.loopexit.us-lcssa ]
  %24 = extractvalue { i8*, i32 } %lpad.phi372, 0
  %25 = extractvalue { i8*, i32 } %lpad.phi372, 1
  sync within %syncreg61, label %ehcleanup202

sync.continue106:                                 ; preds = %pfor.cond.cleanup71
  %mul110 = shl i64 %add, 2
  %mul111 = mul i64 %mul110, %add3
  %call112 = tail call noalias i8* @malloc(i64 %mul111) #2
  %26 = bitcast i8* %call112 to i32*
  %call116 = tail call noalias i8* @malloc(i64 %mul59) #2
  %27 = bitcast i8* %call116 to i32*
  %call120 = tail call noalias i8* @malloc(i64 %mul59) #2
  %28 = bitcast i8* %call120 to i32*
  %call.i366 = tail call i32 @_ZN8sequence4scanIilSt4plusIiENS_4getAIilEEEET_PS5_T0_S7_T1_T2_S5_bb(i32* %27, i64 0, i64 %mul58, i32* %11, i32 0, i1 zeroext false, i1 zeroext false)
  %29 = bitcast %struct.transpose* %ref.tmp to i8*
  call void @llvm.lifetime.start.p0i8(i64 16, i8* nonnull %29) #2
  %30 = bitcast %struct.transpose* %ref.tmp to i8**
  store i8* %call60, i8** %30, align 8, !tbaa !142
  %B.i368 = getelementptr inbounds %struct.transpose, %struct.transpose* %ref.tmp, i64 0, i32 1
  %31 = bitcast i32** %B.i368 to i8**
  store i8* %call120, i8** %31, align 8, !tbaa !144
  call void @_ZN9transposeIiiE6transREiiiiii(%struct.transpose* nonnull %ref.tmp, i32 0, i32 %conv6, i32 %conv6, i32 0, i32 %conv6, i32 %conv6)
  call void @llvm.lifetime.end.p0i8(i64 16, i8* nonnull %29) #2
  %call.i = call i32 @_ZN8sequence4scanIilSt4plusIiENS_4getAIilEEEET_PS5_T0_S7_T1_T2_S5_bb(i32* %28, i64 0, i64 %mul58, i32* %28, i32 0, i1 zeroext false, i1 zeroext false)
  %32 = bitcast %struct.blockTrans* %ref.tmp130 to i8*
  call void @llvm.lifetime.start.p0i8(i64 40, i8* nonnull %32) #2
  %A.i = getelementptr inbounds %struct.blockTrans, %struct.blockTrans* %ref.tmp130, i64 0, i32 0
  store i32* %A, i32** %A.i, align 8, !tbaa !145
  %B.i = getelementptr inbounds %struct.blockTrans, %struct.blockTrans* %ref.tmp130, i64 0, i32 1
  %33 = bitcast i32** %B.i to i8**
  store i8* %call112, i8** %33, align 8, !tbaa !147
  %OA.i = getelementptr inbounds %struct.blockTrans, %struct.blockTrans* %ref.tmp130, i64 0, i32 2
  %34 = bitcast i32** %OA.i to i8**
  store i8* %call116, i8** %34, align 8, !tbaa !148
  %OB.i = getelementptr inbounds %struct.blockTrans, %struct.blockTrans* %ref.tmp130, i64 0, i32 3
  %35 = bitcast i32** %OB.i to i8**
  store i8* %call120, i8** %35, align 8, !tbaa !149
  %L.i = getelementptr inbounds %struct.blockTrans, %struct.blockTrans* %ref.tmp130, i64 0, i32 4
  %36 = bitcast i32** %L.i to i8**
  store i8* %call60, i8** %36, align 8, !tbaa !150
  call void @_ZN10blockTransIiiE6transREiiiiii(%struct.blockTrans* nonnull %ref.tmp130, i32 0, i32 %conv6, i32 %conv6, i32 0, i32 %conv6, i32 %conv6)
  call void @llvm.lifetime.end.p0i8(i64 40, i8* nonnull %32) #2
  call void @free(i8* %call116) #2
  call void @free(i8* %call60) #2
  %cmp142378 = icmp sgt i64 %sub27, 0
  br i1 %cmp142378, label %pfor.detach144.lr.ph, label %pfor.cond.cleanup143

pfor.detach144.lr.ph:                             ; preds = %sync.continue106
  %37 = icmp sgt i64 %conv57, 1
  %smax = select i1 %37, i64 %conv57, i64 1
  br label %pfor.detach144

pfor.cond.cleanup143:                             ; preds = %pfor.inc189, %sync.continue106
  sync within %syncreg132, label %sync.continue198

pfor.detach144:                                   ; preds = %pfor.inc189, %pfor.detach144.lr.ph
  %__begin134.0379 = phi i64 [ 0, %pfor.detach144.lr.ph ], [ %inc190, %pfor.inc189 ]
  detach within %syncreg132, label %pfor.body149, label %pfor.inc189 unwind label %lpad191.loopexit

pfor.body149:                                     ; preds = %pfor.detach144
  %mul150 = mul nsw i64 %__begin134.0379, %add
  %arrayidx151 = getelementptr inbounds i32, i32* %28, i64 %mul150
  %38 = load i32, i32* %arrayidx151, align 4, !tbaa !42
  %conv152 = sext i32 %38 to i64
  %cmp155 = icmp slt i64 %__begin134.0379, %conv28
  br i1 %cmp155, label %cond.true156, label %cond.end161

cond.true156:                                     ; preds = %pfor.body149
  %add157 = add nuw nsw i64 %__begin134.0379, 1
  %mul158 = mul nsw i64 %add157, %add
  %arrayidx159 = getelementptr inbounds i32, i32* %28, i64 %mul158
  %39 = load i32, i32* %arrayidx159, align 4, !tbaa !42
  br label %cond.end161

cond.end161:                                      ; preds = %pfor.body149, %cond.true156
  %cond162 = phi i32 [ %39, %cond.true156 ], [ %n, %pfor.body149 ]
  %conv163 = sext i32 %cond162 to i64
  %cmp164 = icmp eq i64 %__begin134.0379, 0
  %cmp167 = icmp eq i64 %__begin134.0379, %conv28
  %or.cond = or i1 %cmp164, %cmp167
  br i1 %or.cond, label %if.then177, label %lor.lhs.false168

lor.lhs.false168:                                 ; preds = %cond.end161
  %sub169 = add nsw i64 %__begin134.0379, -1
  %arrayidx170 = getelementptr inbounds i32, i32* %8, i64 %sub169
  %arrayidx171 = getelementptr inbounds i32, i32* %8, i64 %__begin134.0379
  %40 = load i32, i32* %arrayidx170, align 4, !tbaa !42
  %41 = load i32, i32* %arrayidx171, align 4, !tbaa !42
  %cmp.i = icmp slt i32 %40, %41
  br i1 %cmp.i, label %if.then177, label %if.end

if.then177:                                       ; preds = %lor.lhs.false168, %cond.end161
  %add.ptr178 = getelementptr inbounds i32, i32* %26, i64 %conv152
  %sub179 = sub nsw i64 %conv163, %conv152
  invoke void @_Z9quickSortIiSt4lessIiElEvPT_T1_T0_(i32* %add.ptr178, i64 %sub179)
          to label %if.end unwind label %lpad172

lpad172:                                          ; preds = %if.then177
  %42 = landingpad { i8*, i32 }
          catch i8* null
  invoke void @llvm.detached.rethrow.sl_p0i8i32s(token %syncreg132, { i8*, i32 } %42)
          to label %det.rethrow.unreachable197 unwind label %lpad191.loopexit.split-lp

det.rethrow.unreachable197:                       ; preds = %lpad172
  unreachable

if.end:                                           ; preds = %if.then177, %lor.lhs.false168
  %cmp183376 = icmp sgt i32 %cond162, %38
  br i1 %cmp183376, label %for.body.lr.ph, label %for.cond.cleanup

for.body.lr.ph:                                   ; preds = %if.end
  %scevgep = getelementptr i32, i32* %A, i64 %conv152
  %scevgep400 = bitcast i32* %scevgep to i8*
  %43 = shl nsw i64 %conv152, 2
  %scevgep401 = getelementptr i8, i8* %call112, i64 %43
  %44 = shl nsw i64 %conv163, 2
  %45 = sub nsw i64 %44, %43
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %scevgep400, i8* %scevgep401, i64 %45, i32 4, i1 false)
  br label %for.cond.cleanup

for.cond.cleanup:                                 ; preds = %for.body.lr.ph, %if.end
  reattach within %syncreg132, label %pfor.inc189

pfor.inc189:                                      ; preds = %pfor.detach144, %for.cond.cleanup
  %inc190 = add nuw nsw i64 %__begin134.0379, 1
  %exitcond393 = icmp eq i64 %inc190, %smax
  br i1 %exitcond393, label %pfor.cond.cleanup143, label %pfor.detach144, !llvm.loop !151

lpad191.loopexit:                                 ; preds = %pfor.detach144
  %lpad.loopexit = landingpad { i8*, i32 }
          cleanup
  br label %lpad191

lpad191.loopexit.split-lp:                        ; preds = %lpad172
  %lpad.loopexit.split-lp = landingpad { i8*, i32 }
          cleanup
  br label %lpad191

lpad191:                                          ; preds = %lpad191.loopexit.split-lp, %lpad191.loopexit
  %lpad.phi = phi { i8*, i32 } [ %lpad.loopexit, %lpad191.loopexit ], [ %lpad.loopexit.split-lp, %lpad191.loopexit.split-lp ]
  %46 = extractvalue { i8*, i32 } %lpad.phi, 0
  %47 = extractvalue { i8*, i32 } %lpad.phi, 1
  sync within %syncreg132, label %ehcleanup202

sync.continue198:                                 ; preds = %pfor.cond.cleanup143
  call void @free(i8* %call30) #2
  call void @free(i8* %call120) #2
  call void @free(i8* %call112) #2
  br label %if.end208

ehcleanup202:                                     ; preds = %lpad99, %lpad191, %lpad20
  %ehselector.slot22.1 = phi i32 [ %7, %lpad20 ], [ %25, %lpad99 ], [ %47, %lpad191 ]
  %exn.slot21.1 = phi i8* [ %6, %lpad20 ], [ %24, %lpad99 ], [ %46, %lpad191 ]
  %lpad.val211 = insertvalue { i8*, i32 } undef, i8* %exn.slot21.1, 0
  %lpad.val212 = insertvalue { i8*, i32 } %lpad.val211, i32 %ehselector.slot22.1, 1
  resume { i8*, i32 } %lpad.val212

if.end208:                                        ; preds = %sync.continue198, %if.then
  ret void
}

; Function Attrs: uwtable
declare void @_Z9quickSortIiSt4lessIiElEvPT_T1_T0_(i32* %A, i64 %n) local_unnamed_addr #4

; Function Attrs: uwtable
declare void @_Z9quickSortIiSt4lessIiEiEvPT_T1_T0_(i32* %A, i32 %n) local_unnamed_addr #4

; Function Attrs: uwtable
declare void @_ZN10blockTransIiiE6transREiiiiii(%struct.blockTrans* %this, i32 %rStart, i32 %rCount, i32 %rLength, i32 %cStart, i32 %cCount, i32 %cLength) local_unnamed_addr #4

; Function Attrs: uwtable
declare i32 @_ZN8sequence4scanIilSt4plusIiENS_4getAIilEEEET_PS5_T0_S7_T1_T2_S5_bb(i32* %Out, i64 %s, i64 %e, i32* %g.coerce, i32 %zero, i1 zeroext %inclusive, i1 zeroext %back) local_unnamed_addr #4

; Function Attrs: uwtable
declare void @_ZN9transposeIiiE6transREiiiiii(%struct.transpose* %this, i32 %rStart, i32 %rCount, i32 %rLength, i32 %cStart, i32 %cCount, i32 %cLength) local_unnamed_addr #4

; Function Attrs: argmemonly nounwind
declare token @llvm.syncregion.start() #5

; Function Attrs: argmemonly nounwind
declare void @llvm.lifetime.start.p0i8(i64, i8* nocapture) #5

declare i32 @__gxx_personality_v0(...)

; Function Attrs: argmemonly nounwind
declare void @llvm.lifetime.end.p0i8(i64, i8* nocapture) #5

; Function Attrs: argmemonly
declare void @llvm.detached.rethrow.sl_p0i8i32s(token, { i8*, i32 }) #6

; Function Attrs: nounwind
declare noalias i8* @malloc(i64) local_unnamed_addr #1

; Function Attrs: nounwind
declare void @free(i8* nocapture) local_unnamed_addr #1

; Function Attrs: nounwind readnone speculatable
declare double @llvm.ceil.f64(double) #10

; Function Attrs: nounwind readnone speculatable
declare double @llvm.sqrt.f64(double) #10

; Function Attrs: argmemonly nounwind
declare void @llvm.memset.p0i8.i64(i8* nocapture writeonly, i8, i64, i32, i1) #5

; Function Attrs: argmemonly nounwind
declare void @llvm.memcpy.p0i8.p0i8.i64(i8* nocapture writeonly, i8* nocapture readonly, i64, i32, i1) #5

; CHECK-LABEL: define internal fastcc void @_Z10sampleSortIiSt4lessIiEiEvPT_T1_T0_.outline_pfor.detach73.ls1(i64 %__begin63.0382.start.ls1
; CHECK: lpad84.us-lcssa.ls1:
; CHECK: br label %lpad84.ls1
; CHECK: lpad84.ls1:
; CHECK-NOT: phi { i8*, i32 }{{.+}}[ {{%lpad.us-lcssa383.us|%lpad.us-lcssa383.us.ls1}}, {{%lpad84.us-lcssa.us|%lpad84.us-lcssa.us.ls1}} ]

; CHECK-LABEL: define internal fastcc void @_Z10sampleSortIiSt4lessIiEiEvPT_T1_T0_.outline_pfor.detach73.us.ls1(i64 %__begin63.0382.us.start.ls1
; CHECK: lpad84.us-lcssa.us.ls1:
; CHECK: br label %lpad84.ls1
; CHECK: lpad84.ls1:
; CHECK-NOT: phi { i8*, i32 }{{.+}}[ {{%lpad.us-lcssa383|%lpad.us-lcssa383.ls1}}, {{%lpad84.us-lcssa|%lpad84.us-lcssa.ls1}} ]


attributes #0 = { "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="haswell" "target-features"="+aes,+avx,+avx2,+bmi,+bmi2,+cmov,+cx16,+f16c,+fma,+fsgsbase,+fxsr,+lzcnt,+mmx,+movbe,+pclmul,+popcnt,+rdrnd,+sse,+sse2,+sse3,+sse4.1,+sse4.2,+ssse3,+x87,+xsave,+xsaveopt,-adx,-avx512bitalg,-avx512bw,-avx512cd,-avx512dq,-avx512er,-avx512f,-avx512ifma,-avx512pf,-avx512vbmi,-avx512vbmi2,-avx512vl,-avx512vnni,-avx512vpopcntdq,-clflushopt,-clwb,-clzero,-fma4,-gfni,-ibt,-lwp,-mwaitx,-pku,-prefetchwt1,-prfchw,-rdseed,-rtm,-sgx,-sha,-shstk,-sse4a,-tbm,-vaes,-vpclmulqdq,-xop,-xsavec,-xsaves" "unsafe-fp-math"="true" "use-soft-float"="false" }
attributes #1 = { nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="haswell" "target-features"="+aes,+avx,+avx2,+bmi,+bmi2,+cmov,+cx16,+f16c,+fma,+fsgsbase,+fxsr,+lzcnt,+mmx,+movbe,+pclmul,+popcnt,+rdrnd,+sse,+sse2,+sse3,+sse4.1,+sse4.2,+ssse3,+x87,+xsave,+xsaveopt,-adx,-avx512bitalg,-avx512bw,-avx512cd,-avx512dq,-avx512er,-avx512f,-avx512ifma,-avx512pf,-avx512vbmi,-avx512vbmi2,-avx512vl,-avx512vnni,-avx512vpopcntdq,-clflushopt,-clwb,-clzero,-fma4,-gfni,-ibt,-lwp,-mwaitx,-pku,-prefetchwt1,-prfchw,-rdseed,-rtm,-sgx,-sha,-shstk,-sse4a,-tbm,-vaes,-vpclmulqdq,-xop,-xsavec,-xsaves" "unsafe-fp-math"="true" "use-soft-float"="false" }
attributes #2 = { nounwind }
attributes #3 = { norecurse nounwind readnone uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="true" "no-jump-tables"="false" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="haswell" "target-features"="+aes,+avx,+avx2,+bmi,+bmi2,+cmov,+cx16,+f16c,+fma,+fsgsbase,+fxsr,+lzcnt,+mmx,+movbe,+pclmul,+popcnt,+rdrnd,+sse,+sse2,+sse3,+sse4.1,+sse4.2,+ssse3,+x87,+xsave,+xsaveopt,-adx,-avx512bitalg,-avx512bw,-avx512cd,-avx512dq,-avx512er,-avx512f,-avx512ifma,-avx512pf,-avx512vbmi,-avx512vbmi2,-avx512vl,-avx512vnni,-avx512vpopcntdq,-clflushopt,-clwb,-clzero,-fma4,-gfni,-ibt,-lwp,-mwaitx,-pku,-prefetchwt1,-prfchw,-rdseed,-rtm,-sgx,-sha,-shstk,-sse4a,-tbm,-vaes,-vpclmulqdq,-xop,-xsavec,-xsaves" "unsafe-fp-math"="true" "use-soft-float"="false" }
attributes #4 = { uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="true" "no-jump-tables"="false" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="haswell" "target-features"="+aes,+avx,+avx2,+bmi,+bmi2,+cmov,+cx16,+f16c,+fma,+fsgsbase,+fxsr,+lzcnt,+mmx,+movbe,+pclmul,+popcnt,+rdrnd,+sse,+sse2,+sse3,+sse4.1,+sse4.2,+ssse3,+x87,+xsave,+xsaveopt,-adx,-avx512bitalg,-avx512bw,-avx512cd,-avx512dq,-avx512er,-avx512f,-avx512ifma,-avx512pf,-avx512vbmi,-avx512vbmi2,-avx512vl,-avx512vnni,-avx512vpopcntdq,-clflushopt,-clwb,-clzero,-fma4,-gfni,-ibt,-lwp,-mwaitx,-pku,-prefetchwt1,-prfchw,-rdseed,-rtm,-sgx,-sha,-shstk,-sse4a,-tbm,-vaes,-vpclmulqdq,-xop,-xsavec,-xsaves" "unsafe-fp-math"="true" "use-soft-float"="false" }
attributes #5 = { argmemonly nounwind }
attributes #6 = { argmemonly }
attributes #7 = { inlinehint uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="true" "no-jump-tables"="false" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="haswell" "target-features"="+aes,+avx,+avx2,+bmi,+bmi2,+cmov,+cx16,+f16c,+fma,+fsgsbase,+fxsr,+lzcnt,+mmx,+movbe,+pclmul,+popcnt,+rdrnd,+sse,+sse2,+sse3,+sse4.1,+sse4.2,+ssse3,+x87,+xsave,+xsaveopt,-adx,-avx512bitalg,-avx512bw,-avx512cd,-avx512dq,-avx512er,-avx512f,-avx512ifma,-avx512pf,-avx512vbmi,-avx512vbmi2,-avx512vl,-avx512vnni,-avx512vpopcntdq,-clflushopt,-clwb,-clzero,-fma4,-gfni,-ibt,-lwp,-mwaitx,-pku,-prefetchwt1,-prfchw,-rdseed,-rtm,-sgx,-sha,-shstk,-sse4a,-tbm,-vaes,-vpclmulqdq,-xop,-xsavec,-xsaves" "unsafe-fp-math"="true" "use-soft-float"="false" }
attributes #8 = { nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="true" "no-jump-tables"="false" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="haswell" "target-features"="+aes,+avx,+avx2,+bmi,+bmi2,+cmov,+cx16,+f16c,+fma,+fsgsbase,+fxsr,+lzcnt,+mmx,+movbe,+pclmul,+popcnt,+rdrnd,+sse,+sse2,+sse3,+sse4.1,+sse4.2,+ssse3,+x87,+xsave,+xsaveopt,-adx,-avx512bitalg,-avx512bw,-avx512cd,-avx512dq,-avx512er,-avx512f,-avx512ifma,-avx512pf,-avx512vbmi,-avx512vbmi2,-avx512vl,-avx512vnni,-avx512vpopcntdq,-clflushopt,-clwb,-clzero,-fma4,-gfni,-ibt,-lwp,-mwaitx,-pku,-prefetchwt1,-prfchw,-rdseed,-rtm,-sgx,-sha,-shstk,-sse4a,-tbm,-vaes,-vpclmulqdq,-xop,-xsavec,-xsaves" "unsafe-fp-math"="true" "use-soft-float"="false" }
attributes #9 = { noreturn nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="haswell" "target-features"="+aes,+avx,+avx2,+bmi,+bmi2,+cmov,+cx16,+f16c,+fma,+fsgsbase,+fxsr,+lzcnt,+mmx,+movbe,+pclmul,+popcnt,+rdrnd,+sse,+sse2,+sse3,+sse4.1,+sse4.2,+ssse3,+x87,+xsave,+xsaveopt,-adx,-avx512bitalg,-avx512bw,-avx512cd,-avx512dq,-avx512er,-avx512f,-avx512ifma,-avx512pf,-avx512vbmi,-avx512vbmi2,-avx512vl,-avx512vnni,-avx512vpopcntdq,-clflushopt,-clwb,-clzero,-fma4,-gfni,-ibt,-lwp,-mwaitx,-pku,-prefetchwt1,-prfchw,-rdseed,-rtm,-sgx,-sha,-shstk,-sse4a,-tbm,-vaes,-vpclmulqdq,-xop,-xsavec,-xsaves" "unsafe-fp-math"="true" "use-soft-float"="false" }
attributes #10 = { nounwind readnone speculatable }
attributes #11 = { nobuiltin "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="haswell" "target-features"="+aes,+avx,+avx2,+bmi,+bmi2,+cmov,+cx16,+f16c,+fma,+fsgsbase,+fxsr,+lzcnt,+mmx,+movbe,+pclmul,+popcnt,+rdrnd,+sse,+sse2,+sse3,+sse4.1,+sse4.2,+ssse3,+x87,+xsave,+xsaveopt,-adx,-avx512bitalg,-avx512bw,-avx512cd,-avx512dq,-avx512er,-avx512f,-avx512ifma,-avx512pf,-avx512vbmi,-avx512vbmi2,-avx512vl,-avx512vnni,-avx512vpopcntdq,-clflushopt,-clwb,-clzero,-fma4,-gfni,-ibt,-lwp,-mwaitx,-pku,-prefetchwt1,-prfchw,-rdseed,-rtm,-sgx,-sha,-shstk,-sse4a,-tbm,-vaes,-vpclmulqdq,-xop,-xsavec,-xsaves" "unsafe-fp-math"="true" "use-soft-float"="false" }
attributes #12 = { nobuiltin nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="haswell" "target-features"="+aes,+avx,+avx2,+bmi,+bmi2,+cmov,+cx16,+f16c,+fma,+fsgsbase,+fxsr,+lzcnt,+mmx,+movbe,+pclmul,+popcnt,+rdrnd,+sse,+sse2,+sse3,+sse4.1,+sse4.2,+ssse3,+x87,+xsave,+xsaveopt,-adx,-avx512bitalg,-avx512bw,-avx512cd,-avx512dq,-avx512er,-avx512f,-avx512ifma,-avx512pf,-avx512vbmi,-avx512vbmi2,-avx512vl,-avx512vnni,-avx512vpopcntdq,-clflushopt,-clwb,-clzero,-fma4,-gfni,-ibt,-lwp,-mwaitx,-pku,-prefetchwt1,-prfchw,-rdseed,-rtm,-sgx,-sha,-shstk,-sse4a,-tbm,-vaes,-vpclmulqdq,-xop,-xsavec,-xsaves" "unsafe-fp-math"="true" "use-soft-float"="false" }
attributes #13 = { norecurse uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="true" "no-jump-tables"="false" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="haswell" "target-features"="+aes,+avx,+avx2,+bmi,+bmi2,+cmov,+cx16,+f16c,+fma,+fsgsbase,+fxsr,+lzcnt,+mmx,+movbe,+pclmul,+popcnt,+rdrnd,+sse,+sse2,+sse3,+sse4.1,+sse4.2,+ssse3,+x87,+xsave,+xsaveopt,-adx,-avx512bitalg,-avx512bw,-avx512cd,-avx512dq,-avx512er,-avx512f,-avx512ifma,-avx512pf,-avx512vbmi,-avx512vbmi2,-avx512vl,-avx512vnni,-avx512vpopcntdq,-clflushopt,-clwb,-clzero,-fma4,-gfni,-ibt,-lwp,-mwaitx,-pku,-prefetchwt1,-prfchw,-rdseed,-rtm,-sgx,-sha,-shstk,-sse4a,-tbm,-vaes,-vpclmulqdq,-xop,-xsavec,-xsaves" "unsafe-fp-math"="true" "use-soft-float"="false" }
attributes #14 = { noreturn "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="haswell" "target-features"="+aes,+avx,+avx2,+bmi,+bmi2,+cmov,+cx16,+f16c,+fma,+fsgsbase,+fxsr,+lzcnt,+mmx,+movbe,+pclmul,+popcnt,+rdrnd,+sse,+sse2,+sse3,+sse4.1,+sse4.2,+ssse3,+x87,+xsave,+xsaveopt,-adx,-avx512bitalg,-avx512bw,-avx512cd,-avx512dq,-avx512er,-avx512f,-avx512ifma,-avx512pf,-avx512vbmi,-avx512vbmi2,-avx512vl,-avx512vnni,-avx512vpopcntdq,-clflushopt,-clwb,-clzero,-fma4,-gfni,-ibt,-lwp,-mwaitx,-pku,-prefetchwt1,-prfchw,-rdseed,-rtm,-sgx,-sha,-shstk,-sse4a,-tbm,-vaes,-vpclmulqdq,-xop,-xsavec,-xsaves" "unsafe-fp-math"="true" "use-soft-float"="false" }
attributes #15 = { nounwind readonly "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="haswell" "target-features"="+aes,+avx,+avx2,+bmi,+bmi2,+cmov,+cx16,+f16c,+fma,+fsgsbase,+fxsr,+lzcnt,+mmx,+movbe,+pclmul,+popcnt,+rdrnd,+sse,+sse2,+sse3,+sse4.1,+sse4.2,+ssse3,+x87,+xsave,+xsaveopt,-adx,-avx512bitalg,-avx512bw,-avx512cd,-avx512dq,-avx512er,-avx512f,-avx512ifma,-avx512pf,-avx512vbmi,-avx512vbmi2,-avx512vl,-avx512vnni,-avx512vpopcntdq,-clflushopt,-clwb,-clzero,-fma4,-gfni,-ibt,-lwp,-mwaitx,-pku,-prefetchwt1,-prfchw,-rdseed,-rtm,-sgx,-sha,-shstk,-sse4a,-tbm,-vaes,-vpclmulqdq,-xop,-xsavec,-xsaves" "unsafe-fp-math"="true" "use-soft-float"="false" }
attributes #16 = { argmemonly nounwind readonly "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="haswell" "target-features"="+aes,+avx,+avx2,+bmi,+bmi2,+cmov,+cx16,+f16c,+fma,+fsgsbase,+fxsr,+lzcnt,+mmx,+movbe,+pclmul,+popcnt,+rdrnd,+sse,+sse2,+sse3,+sse4.1,+sse4.2,+ssse3,+x87,+xsave,+xsaveopt,-adx,-avx512bitalg,-avx512bw,-avx512cd,-avx512dq,-avx512er,-avx512f,-avx512ifma,-avx512pf,-avx512vbmi,-avx512vbmi2,-avx512vl,-avx512vnni,-avx512vpopcntdq,-clflushopt,-clwb,-clzero,-fma4,-gfni,-ibt,-lwp,-mwaitx,-pku,-prefetchwt1,-prfchw,-rdseed,-rtm,-sgx,-sha,-shstk,-sse4a,-tbm,-vaes,-vpclmulqdq,-xop,-xsavec,-xsaves" "unsafe-fp-math"="true" "use-soft-float"="false" }
attributes #17 = { nounwind readonly }
attributes #18 = { noreturn }
attributes #19 = { noreturn nounwind }
attributes #20 = { builtin }
attributes #21 = { builtin nounwind }

!2 = !{!3, !3, i64 0}
!3 = !{!"omnipotent char", !4, i64 0}
!4 = !{!"Simple C++ TBAA"}
!5 = distinct !{!5, !6}
!6 = !{!"tapir.loop.spawn.strategy", i32 1}
!7 = !{!8, !8, i64 0}
!8 = !{!"bool", !3, i64 0}
!9 = distinct !{!9, !6}
!10 = !{!11, !11, i64 0}
!11 = !{!"long", !3, i64 0}
!12 = !{!13, !13, i64 0}
!13 = !{!"any pointer", !3, i64 0}
!14 = distinct !{!14, !6}
!15 = !{!16, !11, i64 0}
!16 = !{!"_ZTSN7benchIO5wordsE", !11, i64 0, !13, i64 8, !11, i64 16, !13, i64 24}
!17 = !{!16, !13, i64 8}
!18 = !{!16, !11, i64 16}
!19 = !{!16, !13, i64 24}
!20 = !{!21, !21, i64 0}
!21 = !{!"vtable pointer", !4, i64 0}
!22 = !{!23, !25, i64 32}
!23 = !{!"_ZTSSt8ios_base", !11, i64 8, !11, i64 16, !24, i64 24, !25, i64 28, !25, i64 32, !13, i64 40, !26, i64 48, !3, i64 64, !27, i64 192, !13, i64 200, !28, i64 208}
!24 = !{!"_ZTSSt13_Ios_Fmtflags", !3, i64 0}
!25 = !{!"_ZTSSt12_Ios_Iostate", !3, i64 0}
!26 = !{!"_ZTSNSt8ios_base6_WordsE", !13, i64 0, !11, i64 8}
!27 = !{!"int", !3, i64 0}
!28 = !{!"_ZTSSt6locale", !13, i64 0}
!29 = !{!30, !13, i64 240}
!30 = !{!"_ZTSSt9basic_iosIcSt11char_traitsIcEE", !13, i64 216, !3, i64 224, !8, i64 225, !13, i64 232, !13, i64 240, !13, i64 248, !13, i64 256}
!31 = !{!32, !3, i64 56}
!32 = !{!"_ZTSSt5ctypeIcE", !13, i64 16, !8, i64 24, !13, i64 32, !13, i64 40, !13, i64 48, !3, i64 56, !3, i64 57, !3, i64 313, !3, i64 569}
!33 = distinct !{!33, !6}
!34 = !{!35, !13, i64 0}
!35 = !{!"_ZTSNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE12_Alloc_hiderE", !13, i64 0}
!36 = !{!37, !11, i64 8}
!37 = !{!"_ZTSNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE", !35, i64 0, !11, i64 8, !3, i64 16}
!38 = !{!37, !13, i64 0}
!39 = !{!40}
!40 = distinct !{!40, !41, !"_ZN7benchIO9seqHeaderB5cxx11ENS_11elementTypeE: %agg.result"}
!41 = distinct !{!41, !"_ZN7benchIO9seqHeaderB5cxx11ENS_11elementTypeE"}
!42 = !{!27, !27, i64 0}
!43 = distinct !{!43, !6}
!44 = !{!45, !13, i64 0}
!45 = !{!"_ZTSN7benchIO7seqDataE", !13, i64 0, !11, i64 8, !46, i64 16, !13, i64 24}
!46 = !{!"_ZTSN7benchIO11elementTypeE", !3, i64 0}
!47 = !{!45, !11, i64 8}
!48 = !{!45, !46, i64 16}
!49 = !{!45, !13, i64 24}
!50 = !{!51}
!51 = distinct !{!51, !52, !"_ZN7benchIO9seqHeaderB5cxx11ENS_11elementTypeE: %agg.result"}
!52 = distinct !{!52, !"_ZN7benchIO9seqHeaderB5cxx11ENS_11elementTypeE"}
!53 = !{!54, !54, i64 0}
!54 = !{!"double", !3, i64 0}
!55 = distinct !{!55, !6}
!56 = !{!57}
!57 = distinct !{!57, !58, !"_ZN7benchIO9seqHeaderB5cxx11ENS_11elementTypeE: %agg.result"}
!58 = distinct !{!58, !"_ZN7benchIO9seqHeaderB5cxx11ENS_11elementTypeE"}
!59 = distinct !{!59, !6}
!60 = !{!61}
!61 = distinct !{!61, !62, !"_ZN7benchIO9seqHeaderB5cxx11ENS_11elementTypeE: %agg.result"}
!62 = distinct !{!62, !"_ZN7benchIO9seqHeaderB5cxx11ENS_11elementTypeE"}
!63 = !{!64, !27, i64 0}
!64 = !{!"_ZTSSt4pairIiiE", !27, i64 0, !27, i64 4}
!65 = !{!64, !27, i64 4}
!66 = distinct !{!66, !6}
!67 = !{!68}
!68 = distinct !{!68, !69, !"_ZN7benchIO9seqHeaderB5cxx11ENS_11elementTypeE: %agg.result"}
!69 = distinct !{!69, !"_ZN7benchIO9seqHeaderB5cxx11ENS_11elementTypeE"}
!70 = !{!71, !13, i64 0}
!71 = !{!"_ZTSSt4pairIPciE", !13, i64 0, !27, i64 8}
!72 = !{!71, !27, i64 8}
!73 = distinct !{!73, !74}
!74 = !{!"llvm.loop.unroll.disable"}
!75 = distinct !{!75, !6}
!76 = !{!77, !27, i64 0}
!77 = !{!"_ZTS11commandLine", !27, i64 0, !13, i64 8, !37, i64 16}
!78 = !{!77, !13, i64 8}
!79 = distinct !{!79, !6}
!80 = !{!81, !8, i64 24}
!81 = !{!"_ZTS5timer", !54, i64 0, !54, i64 8, !54, i64 16, !8, i64 24, !82, i64 28}
!82 = !{!"_ZTS8timezone", !27, i64 0, !27, i64 4}
!83 = !{!84, !11, i64 0}
!84 = !{!"_ZTS7timeval", !11, i64 0, !11, i64 8}
!85 = !{!84, !11, i64 8}
!86 = !{!81, !54, i64 8}
!87 = !{i8 0, i8 2}
!88 = !{!81, !54, i64 0}
!89 = distinct !{!89, !6}
!90 = !{!91}
!91 = distinct !{!91, !92, !"_ZN7benchIO9seqHeaderB5cxx11ENS_11elementTypeE: %agg.result"}
!92 = distinct !{!92, !"_ZN7benchIO9seqHeaderB5cxx11ENS_11elementTypeE"}
!93 = distinct !{!93, !6}
!94 = distinct !{!94, !6}
!95 = !{!96}
!96 = distinct !{!96, !97, !"_ZN7benchIO9seqHeaderB5cxx11ENS_11elementTypeE: %agg.result"}
!97 = distinct !{!97, !"_ZN7benchIO9seqHeaderB5cxx11ENS_11elementTypeE"}
!98 = distinct !{!98, !6}
!99 = distinct !{!99, !6}
!100 = !{!101}
!101 = distinct !{!101, !102, !"_ZN7benchIO9seqHeaderB5cxx11ENS_11elementTypeE: %agg.result"}
!102 = distinct !{!102, !"_ZN7benchIO9seqHeaderB5cxx11ENS_11elementTypeE"}
!103 = distinct !{!103, !6}
!104 = distinct !{!104, !6}
!105 = !{!106}
!106 = distinct !{!106, !107, !"_ZN7benchIO9seqHeaderB5cxx11ENS_11elementTypeE: %agg.result"}
!107 = distinct !{!107, !"_ZN7benchIO9seqHeaderB5cxx11ENS_11elementTypeE"}
!108 = distinct !{!108, !109}
!109 = !{!"llvm.loop.isvectorized", i32 1}
!110 = distinct !{!110, !74}
!111 = distinct !{!111, !112, !109}
!112 = !{!"llvm.loop.unroll.runtime.disable"}
!113 = distinct !{!113, !109}
!114 = distinct !{!114, !74}
!115 = distinct !{!115, !112, !109}
!116 = distinct !{!116, !6}
!117 = distinct !{!117, !109}
!118 = distinct !{!118, !74}
!119 = distinct !{!119, !112, !109}
!120 = distinct !{!120, !74}
!121 = distinct !{!121, !6}
!122 = distinct !{!122, !74}
!123 = distinct !{!123, !74}
!124 = distinct !{!124, !74}
!125 = distinct !{!125, !74}
!126 = distinct !{!126, !74}
!127 = distinct !{!127, !109}
!128 = distinct !{!128, !74}
!129 = distinct !{!129, !112, !109}
!130 = distinct !{!130, !6}
!131 = distinct !{!131, !74}
!132 = distinct !{!132, !74}
!133 = distinct !{!133, !6}
!134 = distinct !{!134, !74}
!135 = distinct !{!135, !74}
!136 = distinct !{!136, !6}
!137 = distinct !{!137, !6}
!138 = distinct !{!138, !6}
!139 = distinct !{!139, !6}
!140 = distinct !{!140, !6}
!141 = distinct !{!141, !6}
!142 = !{!143, !13, i64 0}
!143 = !{!"_ZTS9transposeIiiE", !13, i64 0, !13, i64 8}
!144 = !{!143, !13, i64 8}
!145 = !{!146, !13, i64 0}
!146 = !{!"_ZTS10blockTransIiiE", !13, i64 0, !13, i64 8, !13, i64 16, !13, i64 24, !13, i64 32}
!147 = !{!146, !13, i64 8}
!148 = !{!146, !13, i64 16}
!149 = !{!146, !13, i64 24}
!150 = !{!146, !13, i64 32}
!151 = distinct !{!151, !6}
!152 = !{!23, !11, i64 8}
!153 = distinct !{!153, !109}
!154 = distinct !{!154, !74}
!155 = distinct !{!155, !112, !109}
!156 = distinct !{!156, !6}
!157 = distinct !{!157, !6}
!158 = distinct !{!158, !109}
!159 = distinct !{!159, !74}
!160 = distinct !{!160, !112, !109}
!161 = distinct !{!161, !74}
!162 = distinct !{!162, !74}
!163 = distinct !{!163, !74}
!164 = distinct !{!164, !109}
!165 = distinct !{!165, !112, !109}
!166 = distinct !{!166, !6}
!167 = distinct !{!167, !74}
!168 = distinct !{!168, !74}
!169 = distinct !{!169, !6}
!170 = distinct !{!170, !74}
!171 = distinct !{!171, !74}
!172 = distinct !{!172, !74}
!173 = distinct !{!173, !74}
!174 = distinct !{!174, !109}
!175 = distinct !{!175, !74}
!176 = distinct !{!176, !112, !109}
!177 = distinct !{!177, !6}
!178 = distinct !{!178, !74}
!179 = distinct !{!179, !74}
!180 = distinct !{!180, !6}
!181 = distinct !{!181, !74}
!182 = distinct !{!182, !74}
!183 = !{!184}
!184 = distinct !{!184, !185}
!185 = distinct !{!185, !"LVerDomain"}
!186 = !{!187}
!187 = distinct !{!187, !185}
!188 = distinct !{!188, !109}
!189 = distinct !{!189, !74}
!190 = distinct !{!190, !109}
!191 = distinct !{!191, !109}
!192 = distinct !{!192, !74}
!193 = !{!194}
!194 = distinct !{!194, !195}
!195 = distinct !{!195, !"LVerDomain"}
!196 = !{!197}
!197 = distinct !{!197, !195}
!198 = distinct !{!198, !109}
!199 = distinct !{!199, !6}
!200 = distinct !{!200, !6}
!201 = distinct !{!201, !6}
!202 = distinct !{!202, !6}
!203 = distinct !{!203, !74}
!204 = distinct !{!204, !109}
!205 = distinct !{!205, !74}
!206 = distinct !{!206, !112, !109}
!207 = distinct !{!207, !109}
!208 = distinct !{!208, !74}
!209 = distinct !{!209, !112, !109}
!210 = distinct !{!210, !6}
!211 = distinct !{!211, !109}
!212 = distinct !{!212, !74}
!213 = distinct !{!213, !112, !109}
!214 = distinct !{!214, !74}
!215 = distinct !{!215, !6}
!216 = distinct !{!216, !74}
!217 = distinct !{!217, !6}
!218 = distinct !{!218, !6}
!219 = distinct !{!219, !6}
!220 = distinct !{!220, !6}
!221 = distinct !{!221, !6}
!222 = distinct !{!222, !6}
!223 = !{!224, !13, i64 0}
!224 = !{!"_ZTS10blockTransISt4pairIiiEiE", !13, i64 0, !13, i64 8, !13, i64 16, !13, i64 24, !13, i64 32}
!225 = !{!224, !13, i64 8}
!226 = !{!224, !13, i64 16}
!227 = !{!224, !13, i64 24}
!228 = !{!224, !13, i64 32}
!229 = distinct !{!229, !74}
!230 = distinct !{!230, !6}
!231 = distinct !{!231, !74}
!232 = distinct !{!232, !6}
!233 = distinct !{!233, !6}
!234 = distinct !{!234, !6}
!235 = distinct !{!235, !74}
!236 = distinct !{!236, !6}
!237 = distinct !{!237, !6}
!238 = distinct !{!238, !6}
!239 = distinct !{!239, !6}
!240 = distinct !{!240, !6}
!241 = distinct !{!241, !6}
!242 = !{!243, !13, i64 0}
!243 = !{!"_ZTS10blockTransIdiE", !13, i64 0, !13, i64 8, !13, i64 16, !13, i64 24, !13, i64 32}
!244 = !{!243, !13, i64 8}
!245 = !{!243, !13, i64 16}
!246 = !{!243, !13, i64 24}
!247 = !{!243, !13, i64 32}
!248 = distinct !{!248, !6}
!249 = distinct !{!249, !109}
!250 = distinct !{!250, !74}
!251 = !{!252}
!252 = distinct !{!252, !253}
!253 = distinct !{!253, !"LVerDomain"}
!254 = !{!255}
!255 = distinct !{!255, !253}
!256 = distinct !{!256, !109}
!257 = distinct !{!257, !6}
!258 = distinct !{!258, !6}
!259 = distinct !{!259, !6}
!260 = distinct !{!260, !74}
!261 = distinct !{!261, !6}
!262 = distinct !{!262, !6}
!263 = distinct !{!263, !6}
!264 = distinct !{!264, !6}
!265 = distinct !{!265, !6}
!266 = distinct !{!266, !6}
!267 = !{!268, !13, i64 0}
!268 = !{!"_ZTS10blockTransIPciE", !13, i64 0, !13, i64 8, !13, i64 16, !13, i64 24, !13, i64 32}
!269 = !{!268, !13, i64 8}
!270 = !{!268, !13, i64 16}
!271 = !{!268, !13, i64 24}
!272 = !{!268, !13, i64 32}
!273 = distinct !{!273, !6}
!274 = distinct !{!274, !109}
!275 = distinct !{!275, !74}
!276 = !{!277}
!277 = distinct !{!277, !278}
!278 = distinct !{!278, !"LVerDomain"}
!279 = !{!280}
!280 = distinct !{!280, !278}
!281 = distinct !{!281, !109}
!282 = distinct !{!282, !6}
!283 = distinct !{!283, !6}
!284 = distinct !{!284, !6}
!285 = distinct !{!285, !74}
!286 = !{!81, !54, i64 16}
