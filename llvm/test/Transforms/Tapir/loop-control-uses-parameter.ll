; RUN: opt < %s -loop-spawning-ti -S 2>&1 | FileCheck %s
; RUN: opt < %s -passes=loop-spawning -S 2>&1 | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%"struct.pbbs::transpose" = type { i32*, i32* }

$_ZN4pbbs9transposeIjjE6transREmmmmmm = comdat any

; Function Attrs: uwtable
define linkonce_odr void @_ZN4pbbs9transposeIjjE6transREmmmmmm(%"struct.pbbs::transpose"* %this, i64 %rStart, i64 %rCount, i64 %rLength, i64 %cStart, i64 %cCount, i64 %cLength) local_unnamed_addr #6 comdat align 2 personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*) {
entry:
  %syncreg22 = tail call token @llvm.syncregion.start()
  %0 = or i64 %cCount, %rCount
  %1 = icmp ult i64 %0, 64
  br i1 %1, label %if.then, label %if.else

if.then:                                          ; preds = %entry
  %cmp5116 = icmp eq i64 %rCount, 0
  br i1 %cmp5116, label %pfor.cond.cleanup, label %pfor.detach.lr.ph

pfor.detach.lr.ph:                                ; preds = %if.then
  %add7 = add i64 %cCount, %cStart
  %cmp8114 = icmp ugt i64 %add7, %cStart
  %A = getelementptr inbounds %"struct.pbbs::transpose", %"struct.pbbs::transpose"* %this, i64 0, i32 0
  %B = getelementptr inbounds %"struct.pbbs::transpose", %"struct.pbbs::transpose"* %this, i64 0, i32 1
  %2 = add i64 %cStart, %rStart
  %3 = add i64 %add7, %rStart
  %4 = mul i64 %rLength, %rStart
  %5 = add i64 %4, %cStart
  %6 = add i64 %cCount, %cStart
  %7 = add i64 %6, %4
  %8 = add i64 %cCount, -32
  %9 = lshr i64 %8, 5
  %10 = add nuw nsw i64 %9, 1
  %11 = add i64 %cCount, %cStart
  %12 = add i64 %cCount, %cStart
  %13 = add i64 %12, -1
  %min.iters.check = icmp ugt i64 %cCount, 31
  %ident.check = icmp eq i64 %cLength, 1
  %or.cond = and i1 %min.iters.check, %ident.check
  %n.vec = and i64 %cCount, -32
  %ind.end = add i64 %n.vec, %cStart
  %xtraiter = and i64 %10, 1
  %14 = icmp eq i64 %9, 0
  %unroll_iter = sub nsw i64 %10, %xtraiter
  %lcmp.mod = icmp eq i64 %xtraiter, 0
  %cmp.n = icmp eq i64 %n.vec, %cCount
  br label %pfor.detach

; CHECK: pfor.detach.lr.ph:
; CHECK: call fastcc void @_ZN4pbbs9transposeIjjE6transREmmmmmm.outline_pfor.detach.ls1(
; CHECK: i64 %rStart
; CHECK: )

pfor.cond.cleanup:                                ; preds = %pfor.inc, %if.then
  sync within %syncreg22, label %sync.continue55

pfor.detach:                                      ; preds = %pfor.inc, %pfor.detach.lr.ph
  %__begin.0117 = phi i64 [ 0, %pfor.detach.lr.ph ], [ %inc15, %pfor.inc ]
  %15 = add i64 %2, %__begin.0117
  %16 = add i64 %3, %__begin.0117
  %17 = mul i64 %__begin.0117, %rLength
  %18 = add i64 %5, %17
  %19 = add i64 %7, %17
  %add6 = add i64 %__begin.0117, %rStart
  detach within %syncreg22, label %pfor.body, label %pfor.inc

pfor.body:                                        ; preds = %pfor.detach
  br i1 %cmp8114, label %for.body.lr.ph, label %for.cond.cleanup

for.body.lr.ph:                                   ; preds = %pfor.body
  %20 = load i32*, i32** %A, align 8, !tbaa !181
  %mul10 = mul i64 %add6, %rLength
  %21 = load i32*, i32** %B, align 8, !tbaa !183
  br i1 %or.cond, label %vector.memcheck, label %for.body.preheader

vector.memcheck:                                  ; preds = %for.body.lr.ph
  %scevgep = getelementptr i32, i32* %21, i64 %15
  %scevgep120 = getelementptr i32, i32* %21, i64 %16
  %scevgep122 = getelementptr i32, i32* %20, i64 %18
  %scevgep124 = getelementptr i32, i32* %20, i64 %19
  %bound0 = icmp ult i32* %scevgep, %scevgep124
  %bound1 = icmp ult i32* %scevgep122, %scevgep120
  %memcheck.conflict = and i1 %bound0, %bound1
  br i1 %memcheck.conflict, label %for.body.preheader, label %vector.ph

for.body.preheader:                               ; preds = %for.body.lr.ph, %middle.block, %vector.memcheck
  %j.0115.ph = phi i64 [ %cStart, %vector.memcheck ], [ %cStart, %for.body.lr.ph ], [ %ind.end, %middle.block ]
  %22 = sub i64 %11, %j.0115.ph
  %23 = sub i64 %13, %j.0115.ph
  %xtraiter132 = and i64 %22, 3
  %lcmp.mod133 = icmp eq i64 %xtraiter132, 0
  br i1 %lcmp.mod133, label %for.body.prol.loopexit, label %for.body.prol.preheader

for.body.prol.preheader:                          ; preds = %for.body.preheader
  br label %for.body.prol

for.body.prol:                                    ; preds = %for.body.prol, %for.body.prol.preheader
  %j.0115.prol = phi i64 [ %inc.prol, %for.body.prol ], [ %j.0115.ph, %for.body.prol.preheader ]
  %prol.iter = phi i64 [ %prol.iter.sub, %for.body.prol ], [ %xtraiter132, %for.body.prol.preheader ]
  %add11.prol = add i64 %j.0115.prol, %mul10
  %arrayidx.prol = getelementptr inbounds i32, i32* %20, i64 %add11.prol
  %24 = load i32, i32* %arrayidx.prol, align 4, !tbaa !57
  %mul12.prol = mul i64 %j.0115.prol, %cLength
  %add13.prol = add i64 %mul12.prol, %add6
  %arrayidx14.prol = getelementptr inbounds i32, i32* %21, i64 %add13.prol
  store i32 %24, i32* %arrayidx14.prol, align 4, !tbaa !57
  %inc.prol = add nuw i64 %j.0115.prol, 1
  %prol.iter.sub = add i64 %prol.iter, -1
  %prol.iter.cmp = icmp eq i64 %prol.iter.sub, 0
  br i1 %prol.iter.cmp, label %for.body.prol.loopexit, label %for.body.prol, !llvm.loop !237

for.body.prol.loopexit:                           ; preds = %for.body.prol, %for.body.preheader
  %j.0115.unr = phi i64 [ %j.0115.ph, %for.body.preheader ], [ %inc.prol, %for.body.prol ]
  %25 = icmp ult i64 %23, 3
  br i1 %25, label %for.cond.cleanup, label %for.body.preheader.new

for.body.preheader.new:                           ; preds = %for.body.prol.loopexit
  br label %for.body

vector.ph:                                        ; preds = %vector.memcheck
  br i1 %14, label %middle.block.unr-lcssa, label %vector.ph.new

vector.ph.new:                                    ; preds = %vector.ph
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph.new
  %index = phi i64 [ 0, %vector.ph.new ], [ %index.next.1, %vector.body ]
  %niter = phi i64 [ %unroll_iter, %vector.ph.new ], [ %niter.nsub.1, %vector.body ]
  %26 = add i64 %index, %cStart
  %27 = add i64 %26, %mul10
  %28 = getelementptr inbounds i32, i32* %20, i64 %27
  %29 = bitcast i32* %28 to <8 x i32>*
  %wide.load = load <8 x i32>, <8 x i32>* %29, align 4, !tbaa !57, !alias.scope !238
  %30 = getelementptr i32, i32* %28, i64 8
  %31 = bitcast i32* %30 to <8 x i32>*
  %wide.load129 = load <8 x i32>, <8 x i32>* %31, align 4, !tbaa !57, !alias.scope !238
  %32 = getelementptr i32, i32* %28, i64 16
  %33 = bitcast i32* %32 to <8 x i32>*
  %wide.load130 = load <8 x i32>, <8 x i32>* %33, align 4, !tbaa !57, !alias.scope !238
  %34 = getelementptr i32, i32* %28, i64 24
  %35 = bitcast i32* %34 to <8 x i32>*
  %wide.load131 = load <8 x i32>, <8 x i32>* %35, align 4, !tbaa !57, !alias.scope !238
  %36 = mul i64 %26, %cLength
  %37 = add i64 %36, %add6
  %38 = getelementptr inbounds i32, i32* %21, i64 %37
  %39 = bitcast i32* %38 to <8 x i32>*
  store <8 x i32> %wide.load, <8 x i32>* %39, align 4, !tbaa !57, !alias.scope !241, !noalias !238
  %40 = getelementptr i32, i32* %38, i64 8
  %41 = bitcast i32* %40 to <8 x i32>*
  store <8 x i32> %wide.load129, <8 x i32>* %41, align 4, !tbaa !57, !alias.scope !241, !noalias !238
  %42 = getelementptr i32, i32* %38, i64 16
  %43 = bitcast i32* %42 to <8 x i32>*
  store <8 x i32> %wide.load130, <8 x i32>* %43, align 4, !tbaa !57, !alias.scope !241, !noalias !238
  %44 = getelementptr i32, i32* %38, i64 24
  %45 = bitcast i32* %44 to <8 x i32>*
  store <8 x i32> %wide.load131, <8 x i32>* %45, align 4, !tbaa !57, !alias.scope !241, !noalias !238
  %index.next = or i64 %index, 32
  %46 = add i64 %index.next, %cStart
  %47 = add i64 %46, %mul10
  %48 = getelementptr inbounds i32, i32* %20, i64 %47
  %49 = bitcast i32* %48 to <8 x i32>*
  %wide.load.1 = load <8 x i32>, <8 x i32>* %49, align 4, !tbaa !57, !alias.scope !238
  %50 = getelementptr i32, i32* %48, i64 8
  %51 = bitcast i32* %50 to <8 x i32>*
  %wide.load129.1 = load <8 x i32>, <8 x i32>* %51, align 4, !tbaa !57, !alias.scope !238
  %52 = getelementptr i32, i32* %48, i64 16
  %53 = bitcast i32* %52 to <8 x i32>*
  %wide.load130.1 = load <8 x i32>, <8 x i32>* %53, align 4, !tbaa !57, !alias.scope !238
  %54 = getelementptr i32, i32* %48, i64 24
  %55 = bitcast i32* %54 to <8 x i32>*
  %wide.load131.1 = load <8 x i32>, <8 x i32>* %55, align 4, !tbaa !57, !alias.scope !238
  %56 = mul i64 %46, %cLength
  %57 = add i64 %56, %add6
  %58 = getelementptr inbounds i32, i32* %21, i64 %57
  %59 = bitcast i32* %58 to <8 x i32>*
  store <8 x i32> %wide.load.1, <8 x i32>* %59, align 4, !tbaa !57, !alias.scope !241, !noalias !238
  %60 = getelementptr i32, i32* %58, i64 8
  %61 = bitcast i32* %60 to <8 x i32>*
  store <8 x i32> %wide.load129.1, <8 x i32>* %61, align 4, !tbaa !57, !alias.scope !241, !noalias !238
  %62 = getelementptr i32, i32* %58, i64 16
  %63 = bitcast i32* %62 to <8 x i32>*
  store <8 x i32> %wide.load130.1, <8 x i32>* %63, align 4, !tbaa !57, !alias.scope !241, !noalias !238
  %64 = getelementptr i32, i32* %58, i64 24
  %65 = bitcast i32* %64 to <8 x i32>*
  store <8 x i32> %wide.load131.1, <8 x i32>* %65, align 4, !tbaa !57, !alias.scope !241, !noalias !238
  %index.next.1 = add i64 %index, 64
  %niter.nsub.1 = add i64 %niter, -2
  %niter.ncmp.1 = icmp eq i64 %niter.nsub.1, 0
  br i1 %niter.ncmp.1, label %middle.block.unr-lcssa, label %vector.body, !llvm.loop !243

middle.block.unr-lcssa:                           ; preds = %vector.body, %vector.ph
  %index.unr = phi i64 [ 0, %vector.ph ], [ %index.next.1, %vector.body ]
  br i1 %lcmp.mod, label %middle.block, label %vector.body.epil

vector.body.epil:                                 ; preds = %middle.block.unr-lcssa
  %66 = add i64 %index.unr, %cStart
  %67 = add i64 %66, %mul10
  %68 = getelementptr inbounds i32, i32* %20, i64 %67
  %69 = bitcast i32* %68 to <8 x i32>*
  %wide.load.epil = load <8 x i32>, <8 x i32>* %69, align 4, !tbaa !57, !alias.scope !238
  %70 = getelementptr i32, i32* %68, i64 8
  %71 = bitcast i32* %70 to <8 x i32>*
  %wide.load129.epil = load <8 x i32>, <8 x i32>* %71, align 4, !tbaa !57, !alias.scope !238
  %72 = getelementptr i32, i32* %68, i64 16
  %73 = bitcast i32* %72 to <8 x i32>*
  %wide.load130.epil = load <8 x i32>, <8 x i32>* %73, align 4, !tbaa !57, !alias.scope !238
  %74 = getelementptr i32, i32* %68, i64 24
  %75 = bitcast i32* %74 to <8 x i32>*
  %wide.load131.epil = load <8 x i32>, <8 x i32>* %75, align 4, !tbaa !57, !alias.scope !238
  %76 = mul i64 %66, %cLength
  %77 = add i64 %76, %add6
  %78 = getelementptr inbounds i32, i32* %21, i64 %77
  %79 = bitcast i32* %78 to <8 x i32>*
  store <8 x i32> %wide.load.epil, <8 x i32>* %79, align 4, !tbaa !57, !alias.scope !241, !noalias !238
  %80 = getelementptr i32, i32* %78, i64 8
  %81 = bitcast i32* %80 to <8 x i32>*
  store <8 x i32> %wide.load129.epil, <8 x i32>* %81, align 4, !tbaa !57, !alias.scope !241, !noalias !238
  %82 = getelementptr i32, i32* %78, i64 16
  %83 = bitcast i32* %82 to <8 x i32>*
  store <8 x i32> %wide.load130.epil, <8 x i32>* %83, align 4, !tbaa !57, !alias.scope !241, !noalias !238
  %84 = getelementptr i32, i32* %78, i64 24
  %85 = bitcast i32* %84 to <8 x i32>*
  store <8 x i32> %wide.load131.epil, <8 x i32>* %85, align 4, !tbaa !57, !alias.scope !241, !noalias !238
  br label %middle.block

middle.block:                                     ; preds = %middle.block.unr-lcssa, %vector.body.epil
  br i1 %cmp.n, label %for.cond.cleanup, label %for.body.preheader

for.cond.cleanup:                                 ; preds = %for.body.prol.loopexit, %for.body, %middle.block, %pfor.body
  reattach within %syncreg22, label %pfor.inc

for.body:                                         ; preds = %for.body, %for.body.preheader.new
  %j.0115 = phi i64 [ %j.0115.unr, %for.body.preheader.new ], [ %inc.3, %for.body ]
  %add11 = add i64 %j.0115, %mul10
  %arrayidx = getelementptr inbounds i32, i32* %20, i64 %add11
  %86 = load i32, i32* %arrayidx, align 4, !tbaa !57
  %mul12 = mul i64 %j.0115, %cLength
  %add13 = add i64 %mul12, %add6
  %arrayidx14 = getelementptr inbounds i32, i32* %21, i64 %add13
  store i32 %86, i32* %arrayidx14, align 4, !tbaa !57
  %inc = add nuw i64 %j.0115, 1
  %add11.1 = add i64 %inc, %mul10
  %arrayidx.1 = getelementptr inbounds i32, i32* %20, i64 %add11.1
  %87 = load i32, i32* %arrayidx.1, align 4, !tbaa !57
  %mul12.1 = mul i64 %inc, %cLength
  %add13.1 = add i64 %mul12.1, %add6
  %arrayidx14.1 = getelementptr inbounds i32, i32* %21, i64 %add13.1
  store i32 %87, i32* %arrayidx14.1, align 4, !tbaa !57
  %inc.1 = add i64 %j.0115, 2
  %add11.2 = add i64 %inc.1, %mul10
  %arrayidx.2 = getelementptr inbounds i32, i32* %20, i64 %add11.2
  %88 = load i32, i32* %arrayidx.2, align 4, !tbaa !57
  %mul12.2 = mul i64 %inc.1, %cLength
  %add13.2 = add i64 %mul12.2, %add6
  %arrayidx14.2 = getelementptr inbounds i32, i32* %21, i64 %add13.2
  store i32 %88, i32* %arrayidx14.2, align 4, !tbaa !57
  %inc.2 = add i64 %j.0115, 3
  %add11.3 = add i64 %inc.2, %mul10
  %arrayidx.3 = getelementptr inbounds i32, i32* %20, i64 %add11.3
  %89 = load i32, i32* %arrayidx.3, align 4, !tbaa !57
  %mul12.3 = mul i64 %inc.2, %cLength
  %add13.3 = add i64 %mul12.3, %add6
  %arrayidx14.3 = getelementptr inbounds i32, i32* %21, i64 %add13.3
  store i32 %89, i32* %arrayidx14.3, align 4, !tbaa !57
  %inc.3 = add i64 %j.0115, 4
  %exitcond.3 = icmp eq i64 %inc.3, %add7
  br i1 %exitcond.3, label %for.cond.cleanup, label %for.body, !llvm.loop !244

pfor.inc:                                         ; preds = %for.cond.cleanup, %pfor.detach
  %inc15 = add nuw i64 %__begin.0117, 1
  %exitcond118 = icmp eq i64 %inc15, %rCount
  br i1 %exitcond118, label %pfor.cond.cleanup, label %pfor.detach, !llvm.loop !245

; CHECK: pfor.inc:
; CHECK: !llvm.loop ![[ORIGLOOPID:[0-9]+]]

if.else:                                          ; preds = %entry
  %cmp17 = icmp ugt i64 %cCount, %rCount
  br i1 %cmp17, label %if.then18, label %if.else31

if.then18:                                        ; preds = %if.else
  %div19 = lshr i64 %cCount, 1
  %sub21 = sub i64 %cCount, %div19
  detach within %syncreg22, label %det.achd, label %det.cont unwind label %lpad23

det.achd:                                         ; preds = %if.then18
  invoke void @_ZN4pbbs9transposeIjjE6transREmmmmmm(%"struct.pbbs::transpose"* %this, i64 %rStart, i64 %rCount, i64 %rLength, i64 %cStart, i64 %div19, i64 %cLength)
          to label %invoke.cont unwind label %lpad

invoke.cont:                                      ; preds = %det.achd
  reattach within %syncreg22, label %det.cont

det.cont:                                         ; preds = %if.then18, %invoke.cont
  %add27 = add i64 %div19, %cStart
  invoke void @_ZN4pbbs9transposeIjjE6transREmmmmmm(%"struct.pbbs::transpose"* %this, i64 %rStart, i64 %rCount, i64 %rLength, i64 %add27, i64 %sub21, i64 %cLength)
          to label %invoke.cont28 unwind label %lpad23

invoke.cont28:                                    ; preds = %det.cont
  sync within %syncreg22, label %sync.continue55

lpad:                                             ; preds = %det.achd
  %90 = landingpad { i8*, i32 }
          catch i8* null
  invoke void @llvm.detached.rethrow.sl_p0i8i32s(token %syncreg22, { i8*, i32 } %90)
          to label %det.rethrow.unreachable unwind label %lpad23

det.rethrow.unreachable:                          ; preds = %lpad
  unreachable

lpad23:                                           ; preds = %det.cont, %if.then18, %lpad
  %91 = landingpad { i8*, i32 }
          cleanup
  %92 = extractvalue { i8*, i32 } %91, 0
  %93 = extractvalue { i8*, i32 } %91, 1
  sync within %syncreg22, label %eh.resume

if.else31:                                        ; preds = %if.else
  %div33 = lshr i64 %rCount, 1
  %sub36 = sub i64 %rCount, %div33
  detach within %syncreg22, label %det.achd37, label %det.cont42 unwind label %lpad43

det.achd37:                                       ; preds = %if.else31
  invoke void @_ZN4pbbs9transposeIjjE6transREmmmmmm(%"struct.pbbs::transpose"* %this, i64 %rStart, i64 %div33, i64 %rLength, i64 %cStart, i64 %cCount, i64 %cLength)
          to label %invoke.cont41 unwind label %lpad38

invoke.cont41:                                    ; preds = %det.achd37
  reattach within %syncreg22, label %det.cont42

det.cont42:                                       ; preds = %if.else31, %invoke.cont41
  %add50 = add i64 %div33, %rStart
  invoke void @_ZN4pbbs9transposeIjjE6transREmmmmmm(%"struct.pbbs::transpose"* %this, i64 %add50, i64 %sub36, i64 %rLength, i64 %cStart, i64 %cCount, i64 %cLength)
          to label %invoke.cont51 unwind label %lpad43

invoke.cont51:                                    ; preds = %det.cont42
  sync within %syncreg22, label %sync.continue55

lpad38:                                           ; preds = %det.achd37
  %94 = landingpad { i8*, i32 }
          catch i8* null
  invoke void @llvm.detached.rethrow.sl_p0i8i32s(token %syncreg22, { i8*, i32 } %94)
          to label %det.rethrow.unreachable49 unwind label %lpad43

det.rethrow.unreachable49:                        ; preds = %lpad38
  unreachable

lpad43:                                           ; preds = %det.cont42, %if.else31, %lpad38
  %95 = landingpad { i8*, i32 }
          cleanup
  %96 = extractvalue { i8*, i32 } %95, 0
  %97 = extractvalue { i8*, i32 } %95, 1
  sync within %syncreg22, label %eh.resume

sync.continue55:                                  ; preds = %pfor.cond.cleanup, %invoke.cont28, %invoke.cont51
  ret void

eh.resume:                                        ; preds = %lpad43, %lpad23
  %ehselector.slot25.0 = phi i32 [ %93, %lpad23 ], [ %97, %lpad43 ]
  %exn.slot24.0 = phi i8* [ %92, %lpad23 ], [ %96, %lpad43 ]
  %lpad.val58 = insertvalue { i8*, i32 } undef, i8* %exn.slot24.0, 0
  %lpad.val59 = insertvalue { i8*, i32 } %lpad.val58, i32 %ehselector.slot25.0, 1
  resume { i8*, i32 } %lpad.val59
}

; CHECK: define private fastcc void @_ZN4pbbs9transposeIjjE6transREmmmmmm.outline_pfor.detach.ls1(
; CHECK: i64 %rStart.ls1
; CHECK: )
; CHECK: call fastcc void @_ZN4pbbs9transposeIjjE6transREmmmmmm.outline_pfor.detach.ls1(
; CHECK: i64 %rStart.ls1
; CHECK: )
; CHECK: pfor.inc.ls1:
; CHECK-NOT: !llvm.loop ![[OLDLOOPID]]

; CHECK-NOT: Tapir loop not transformed: failed to use divide-and-conquer loop spawning

declare i32 @__gxx_personality_v0(...)

; Function Attrs: argmemonly nounwind
declare token @llvm.syncregion.start() #3

; Function Attrs: argmemonly
declare void @llvm.detached.rethrow.sl_p0i8i32s(token, { i8*, i32 }) #13

!5 = !{!"omnipotent char", !6, i64 0}
!6 = !{!"Simple C++ TBAA"}
!10 = !{!"any pointer", !5, i64 0}
!13 = !{!"int", !5, i64 0}
!29 = !{!"tapir.loop.spawn.strategy", i32 1}
!32 = !{!"llvm.loop.unroll.disable"}
!57 = !{!13, !13, i64 0}
!60 = !{!"llvm.loop.isvectorized", i32 1}
!181 = !{!182, !10, i64 0}
!182 = !{!"_ZTSN4pbbs9transposeIjjEE", !10, i64 0, !10, i64 8}
!183 = !{!182, !10, i64 8}
!237 = distinct !{!237, !32}
!238 = !{!239}
!239 = distinct !{!239, !240}
!240 = distinct !{!240, !"LVerDomain"}
!241 = !{!242}
!242 = distinct !{!242, !240}
!243 = distinct !{!243, !60}
!244 = distinct !{!244, !60}
!245 = distinct !{!245, !29}
