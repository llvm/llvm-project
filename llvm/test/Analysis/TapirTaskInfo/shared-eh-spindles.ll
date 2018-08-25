; RUN: opt < %s -analyze -tasks -S 2>&1 | FileCheck %s

%struct.vertex = type { %class._point2d, %struct.tri*, %struct.tri*, i32, i32 }
%class._point2d = type { double, double }
%struct.tri = type { [3 x %struct.tri*], [3 x %struct.vertex*], i32, i8, i8 }
%class.gTreeNode = type { %class._point2d, double, %struct.nData, i32, [8 x %class.gTreeNode*], %struct.vertex**, %class.gTreeNode* }
%struct.nData = type { i32 }

$_ZN9gTreeNodeI8_point2dIdE7_vect2dIdE6vertex5nDataIS4_EE18buildRecursiveTreeE4_seqIPS4_EPiiPS7_SC_iii = comdat any

; Function Attrs: uwtable
define linkonce_odr void @_ZN9gTreeNodeI8_point2dIdE7_vect2dIdE6vertex5nDataIS4_EE18buildRecursiveTreeE4_seqIPS4_EPiiPS7_SC_iii(%class.gTreeNode* %this, %struct.vertex** %S.coerce0, i64 %S.coerce1, i32* %offsets, i32 %quadrants, %class.gTreeNode* %newNodes, %class.gTreeNode* %parent, i32 %nodesToLeft, i32 %height, i32 %depth) local_unnamed_addr #7 comdat align 2 personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*) {
entry:
  %syncreg = tail call token @llvm.syncregion.start()
  %count = getelementptr inbounds %class.gTreeNode, %class.gTreeNode* %parent, i64 0, i32 3
  store i32 0, i32* %count, align 4
  %cmp = icmp eq i32 %height, 1
  %size = getelementptr inbounds %class.gTreeNode, %class.gTreeNode* %parent, i64 0, i32 1
  %x.i = getelementptr inbounds %class.gTreeNode, %class.gTreeNode* %parent, i64 0, i32 0, i32 0
  %y.i = getelementptr inbounds %class.gTreeNode, %class.gTreeNode* %parent, i64 0, i32 0, i32 1
  %shl7 = shl i32 %nodesToLeft, 2
  br i1 %cmp, label %if.then, label %if.else

if.then:                                          ; preds = %entry
  %sub = add nsw i32 %quadrants, -1
  %0 = sext i32 %shl7 to i64
  %1 = load double, double* %size, align 8
  %div = fmul fast double %1, 2.500000e-01
  %2 = load double, double* %x.i, align 8
  %add.i = fsub fast double %2, %div
  %3 = load double, double* %y.i, align 8
  %add9.i = fsub fast double %3, %div
  %cmp8 = icmp eq i32 %shl7, %sub
  br i1 %cmp8, label %cond.end, label %cond.false

cond.false:                                       ; preds = %if.then
  %4 = or i64 %0, 1
  %arrayidx = getelementptr inbounds i32, i32* %offsets, i64 %4
  %5 = load i32, i32* %arrayidx, align 4
  %conv = sext i32 %5 to i64
  br label %cond.end

cond.end:                                         ; preds = %if.then, %cond.false
  %cond = phi i64 [ %conv, %cond.false ], [ %S.coerce1, %if.then ]
  %arrayidx11 = getelementptr inbounds i32, i32* %offsets, i64 %0
  %6 = load i32, i32* %arrayidx11, align 4
  %add.ptr20 = getelementptr inbounds %class.gTreeNode, %class.gTreeNode* %newNodes, i64 %0
  %arrayidx22 = getelementptr inbounds %class.gTreeNode, %class.gTreeNode* %parent, i64 0, i32 4, i64 0
  store %class.gTreeNode* %add.ptr20, %class.gTreeNode** %arrayidx22, align 8
  detach within %syncreg, label %det.achd, label %det.cont unwind label %lpad29.loopexit

det.achd:                                         ; preds = %cond.end
  %7 = trunc i64 %cond to i32
  %conv14 = sub i32 %7, %6
  %conv18 = sext i32 %conv14 to i64
  %idx.ext = sext i32 %6 to i64
  %add.ptr = getelementptr inbounds %struct.vertex*, %struct.vertex** %S.coerce0, i64 %idx.ext
  %div25 = fmul fast double %1, 5.000000e-01
  %call28 = invoke %class.gTreeNode* @_ZN9gTreeNodeI8_point2dIdE7_vect2dIdE6vertex5nDataIS4_EE7newTreeE4_seqIPS4_ES1_dPS7_i(%struct.vertex** %add.ptr, i64 %conv18, double %add.i, double %add9.i, double %div25, %class.gTreeNode* %add.ptr20, i32 1)
          to label %invoke.cont unwind label %lpad

invoke.cont:                                      ; preds = %det.achd
  reattach within %syncreg, label %det.cont

det.cont:                                         ; preds = %cond.end, %invoke.cont
  %8 = load double, double* %size, align 8
  %div.1 = fmul fast double %8, 2.500000e-01
  %9 = load double, double* %x.i, align 8
  %add.i.1 = fadd fast double %div.1, %9
  %10 = load double, double* %y.i, align 8
  %add9.i.1 = fsub fast double %10, %div.1
  %11 = or i64 %0, 1
  %12 = trunc i64 %11 to i32
  %cmp8.1 = icmp eq i32 %sub, %12
  br i1 %cmp8.1, label %cond.end.1, label %cond.false.1

lpad:                                             ; preds = %det.achd.3, %det.achd.2, %det.achd.1, %det.achd
  %13 = landingpad { i8*, i32 }
          catch i8* null
  invoke void @llvm.detached.rethrow.sl_p0i8i32s(token %syncreg, { i8*, i32 } %13)
          to label %det.rethrow.unreachable unwind label %lpad29.loopexit.split-lp

det.rethrow.unreachable:                          ; preds = %lpad
  unreachable

lpad29.loopexit:                                  ; preds = %cond.end.3, %cond.end.2, %cond.end.1, %cond.end
  %lpad.loopexit = landingpad { i8*, i32 }
          cleanup
  br label %lpad29

lpad29.loopexit.split-lp:                         ; preds = %lpad
  %lpad.loopexit.split-lp = landingpad { i8*, i32 }
          cleanup
  br label %lpad29

lpad29:                                           ; preds = %lpad29.loopexit.split-lp, %lpad29.loopexit
  %lpad.phi = phi { i8*, i32 } [ %lpad.loopexit, %lpad29.loopexit ], [ %lpad.loopexit.split-lp, %lpad29.loopexit.split-lp ]
  %14 = extractvalue { i8*, i32 } %lpad.phi, 0
  %15 = extractvalue { i8*, i32 } %lpad.phi, 1
  sync within %syncreg, label %eh.resume

if.else:                                          ; preds = %entry
  %idx.ext51 = sext i32 %shl7 to i64
  %add74 = add nsw i32 %depth, 1
  %sub73 = add nsw i32 %height, -1
  %mul62 = shl nsw i32 %depth, 1
  %shl63 = shl i32 1, %mul62
  %idx.ext64 = sext i32 %shl63 to i64
  %add.ptr65 = getelementptr inbounds %class.gTreeNode, %class.gTreeNode* %newNodes, i64 %idx.ext64
  %16 = load double, double* %size, align 8
  %div44 = fmul fast double %16, 2.500000e-01
  %17 = bitcast %class.gTreeNode* %parent to <2 x double>*
  %18 = load <2 x double>, <2 x double>* %17, align 8
  %19 = insertelement <2 x double> undef, double %div44, i32 0
  %20 = shufflevector <2 x double> %19, <2 x double> undef, <2 x i32> zeroinitializer
  %21 = fsub fast <2 x double> %18, %20
  %add.ptr52 = getelementptr inbounds %class.gTreeNode, %class.gTreeNode* %newNodes, i64 %idx.ext51
  %div55 = fmul fast double %16, 5.000000e-01
  %cnt.i.i = getelementptr inbounds %class.gTreeNode, %class.gTreeNode* %newNodes, i64 %idx.ext51, i32 2, i32 0
  store i32 0, i32* %cnt.i.i, align 4
  %count.i = getelementptr inbounds %class.gTreeNode, %class.gTreeNode* %newNodes, i64 %idx.ext51, i32 3
  store i32 0, i32* %count.i, align 4
  %size.i = getelementptr inbounds %class.gTreeNode, %class.gTreeNode* %newNodes, i64 %idx.ext51, i32 1
  store double %div55, double* %size.i, align 8
  %22 = bitcast %class.gTreeNode* %add.ptr52 to <2 x double>*
  store <2 x double> %21, <2 x double>* %22, align 8
  %vertices.i = getelementptr inbounds %class.gTreeNode, %class.gTreeNode* %newNodes, i64 %idx.ext51, i32 5
  %23 = bitcast %struct.vertex*** %vertices.i to i8*
  tail call void @llvm.memset.p0i8.i64(i8* nonnull %23, i8 0, i64 16, i32 8, i1 false)
  %arrayidx58 = getelementptr inbounds %class.gTreeNode, %class.gTreeNode* %parent, i64 0, i32 4, i64 0
  store %class.gTreeNode* %add.ptr52, %class.gTreeNode** %arrayidx58, align 8
  detach within %syncreg, label %det.achd75, label %det.cont80 unwind label %lpad81.loopexit

det.achd75:                                       ; preds = %if.else
  invoke void @_ZN9gTreeNodeI8_point2dIdE7_vect2dIdE6vertex5nDataIS4_EE18buildRecursiveTreeE4_seqIPS4_EPiiPS7_SC_iii(%class.gTreeNode* %this, %struct.vertex** %S.coerce0, i64 %S.coerce1, i32* %offsets, i32 %quadrants, %class.gTreeNode* nonnull %add.ptr65, %class.gTreeNode* %add.ptr52, i32 %shl7, i32 %sub73, i32 %add74)
          to label %invoke.cont79 unwind label %lpad76

invoke.cont79:                                    ; preds = %det.achd75
  reattach within %syncreg, label %det.cont80

det.cont80:                                       ; preds = %if.else, %invoke.cont79
  %24 = load double, double* %size, align 8
  %div44.1 = fmul fast double %24, 2.500000e-01
  %25 = load double, double* %x.i, align 8
  %add.i203.1 = fadd fast double %div44.1, %25
  %26 = load double, double* %y.i, align 8
  %add9.i208.1 = fsub fast double %26, %div44.1
  %add.ptr47.1 = getelementptr inbounds %class.gTreeNode, %class.gTreeNode* %newNodes, i64 1
  %add.ptr52.1 = getelementptr inbounds %class.gTreeNode, %class.gTreeNode* %add.ptr47.1, i64 %idx.ext51
  %div55.1 = fmul fast double %24, 5.000000e-01
  %cnt.i.i.1 = getelementptr inbounds %class.gTreeNode, %class.gTreeNode* %add.ptr52.1, i64 0, i32 2, i32 0
  store i32 0, i32* %cnt.i.i.1, align 4
  %count.i.1 = getelementptr inbounds %class.gTreeNode, %class.gTreeNode* %add.ptr52.1, i64 0, i32 3
  store i32 0, i32* %count.i.1, align 4
  %size.i.1 = getelementptr inbounds %class.gTreeNode, %class.gTreeNode* %add.ptr52.1, i64 0, i32 1
  store double %div55.1, double* %size.i.1, align 8
  %cnt.sroa.0.0..sroa_idx.i.1 = getelementptr inbounds %class.gTreeNode, %class.gTreeNode* %add.ptr52.1, i64 0, i32 0, i32 0
  store double %add.i203.1, double* %cnt.sroa.0.0..sroa_idx.i.1, align 8
  %cnt.sroa.2.0..sroa_idx3.i.1 = getelementptr inbounds %class.gTreeNode, %class.gTreeNode* %add.ptr52.1, i64 0, i32 0, i32 1
  store double %add9.i208.1, double* %cnt.sroa.2.0..sroa_idx3.i.1, align 8
  %vertices.i.1 = getelementptr inbounds %class.gTreeNode, %class.gTreeNode* %add.ptr52.1, i64 0, i32 5
  %27 = bitcast %struct.vertex*** %vertices.i.1 to i8*
  tail call void @llvm.memset.p0i8.i64(i8* nonnull %27, i8 0, i64 16, i32 8, i1 false)
  %arrayidx58.1 = getelementptr inbounds %class.gTreeNode, %class.gTreeNode* %parent, i64 0, i32 4, i64 1
  store %class.gTreeNode* %add.ptr52.1, %class.gTreeNode** %arrayidx58.1, align 8
  detach within %syncreg, label %det.achd75.1, label %det.cont80.1 unwind label %lpad81.loopexit

lpad76:                                           ; preds = %det.achd75.3, %det.achd75.2, %det.achd75.1, %det.achd75
  %28 = landingpad { i8*, i32 }
          catch i8* null
  invoke void @llvm.detached.rethrow.sl_p0i8i32s(token %syncreg, { i8*, i32 } %28)
          to label %det.rethrow.unreachable87 unwind label %lpad81.loopexit.split-lp

det.rethrow.unreachable87:                        ; preds = %lpad76
  unreachable

lpad81.loopexit:                                  ; preds = %det.cont80.2, %det.cont80.1, %det.cont80, %if.else
  %lpad.loopexit217 = landingpad { i8*, i32 }
          cleanup
  br label %lpad81

lpad81.loopexit.split-lp:                         ; preds = %lpad76
  %lpad.loopexit.split-lp218 = landingpad { i8*, i32 }
          cleanup
  br label %lpad81

lpad81:                                           ; preds = %lpad81.loopexit.split-lp, %lpad81.loopexit
  %lpad.phi219 = phi { i8*, i32 } [ %lpad.loopexit217, %lpad81.loopexit ], [ %lpad.loopexit.split-lp218, %lpad81.loopexit.split-lp ]
  %29 = extractvalue { i8*, i32 } %lpad.phi219, 0
  %30 = extractvalue { i8*, i32 } %lpad.phi219, 1
  sync within %syncreg, label %eh.resume

if.end:                                           ; preds = %det.cont80.2, %invoke.cont79.3, %cond.end.3, %invoke.cont.3
  sync within %syncreg, label %sync.continue92

sync.continue92:                                  ; preds = %if.end
  %cnt.i = getelementptr inbounds %class.gTreeNode, %class.gTreeNode* %parent, i64 0, i32 2, i32 0
  %arrayidx104 = getelementptr inbounds %class.gTreeNode, %class.gTreeNode* %parent, i64 0, i32 4, i64 0
  %31 = load %class.gTreeNode*, %class.gTreeNode** %arrayidx104, align 8
  %agg.tmp101.sroa.0.0..sroa_idx = getelementptr inbounds %class.gTreeNode, %class.gTreeNode* %31, i64 0, i32 2, i32 0
  %agg.tmp101.sroa.0.0.copyload = load i32, i32* %agg.tmp101.sroa.0.0..sroa_idx, align 4
  %32 = load i32, i32* %cnt.i, align 4
  %add.i211 = add nsw i32 %32, %agg.tmp101.sroa.0.0.copyload
  store i32 %add.i211, i32* %cnt.i, align 4
  %count112 = getelementptr inbounds %class.gTreeNode, %class.gTreeNode* %31, i64 0, i32 3
  %33 = load i32, i32* %count112, align 4
  %34 = load i32, i32* %count, align 4
  %add114 = add nsw i32 %34, %33
  store i32 %add114, i32* %count, align 4
  %arrayidx104.1 = getelementptr inbounds %class.gTreeNode, %class.gTreeNode* %parent, i64 0, i32 4, i64 1
  %35 = load %class.gTreeNode*, %class.gTreeNode** %arrayidx104.1, align 8
  %agg.tmp101.sroa.0.0..sroa_idx.1 = getelementptr inbounds %class.gTreeNode, %class.gTreeNode* %35, i64 0, i32 2, i32 0
  %agg.tmp101.sroa.0.0.copyload.1 = load i32, i32* %agg.tmp101.sroa.0.0..sroa_idx.1, align 4
  %add.i211.1 = add nsw i32 %add.i211, %agg.tmp101.sroa.0.0.copyload.1
  store i32 %add.i211.1, i32* %cnt.i, align 4
  %count112.1 = getelementptr inbounds %class.gTreeNode, %class.gTreeNode* %35, i64 0, i32 3
  %36 = load i32, i32* %count112.1, align 4
  %add114.1 = add nsw i32 %add114, %36
  store i32 %add114.1, i32* %count, align 4
  %arrayidx104.2 = getelementptr inbounds %class.gTreeNode, %class.gTreeNode* %parent, i64 0, i32 4, i64 2
  %37 = load %class.gTreeNode*, %class.gTreeNode** %arrayidx104.2, align 8
  %agg.tmp101.sroa.0.0..sroa_idx.2 = getelementptr inbounds %class.gTreeNode, %class.gTreeNode* %37, i64 0, i32 2, i32 0
  %agg.tmp101.sroa.0.0.copyload.2 = load i32, i32* %agg.tmp101.sroa.0.0..sroa_idx.2, align 4
  %add.i211.2 = add nsw i32 %add.i211.1, %agg.tmp101.sroa.0.0.copyload.2
  store i32 %add.i211.2, i32* %cnt.i, align 4
  %count112.2 = getelementptr inbounds %class.gTreeNode, %class.gTreeNode* %37, i64 0, i32 3
  %38 = load i32, i32* %count112.2, align 4
  %add114.2 = add nsw i32 %add114.1, %38
  store i32 %add114.2, i32* %count, align 4
  %arrayidx104.3 = getelementptr inbounds %class.gTreeNode, %class.gTreeNode* %parent, i64 0, i32 4, i64 3
  %39 = load %class.gTreeNode*, %class.gTreeNode** %arrayidx104.3, align 8
  %agg.tmp101.sroa.0.0..sroa_idx.3 = getelementptr inbounds %class.gTreeNode, %class.gTreeNode* %39, i64 0, i32 2, i32 0
  %agg.tmp101.sroa.0.0.copyload.3 = load i32, i32* %agg.tmp101.sroa.0.0..sroa_idx.3, align 4
  %add.i211.3 = add nsw i32 %add.i211.2, %agg.tmp101.sroa.0.0.copyload.3
  store i32 %add.i211.3, i32* %cnt.i, align 4
  %count112.3 = getelementptr inbounds %class.gTreeNode, %class.gTreeNode* %39, i64 0, i32 3
  %40 = load i32, i32* %count112.3, align 4
  %add114.3 = add nsw i32 %add114.2, %40
  store i32 %add114.3, i32* %count, align 4
  %cmp119 = icmp eq i32 %add114.3, 0
  br i1 %cmp119, label %if.then120, label %if.end122

if.then120:                                       ; preds = %sync.continue92
  %vertices = getelementptr inbounds %class.gTreeNode, %class.gTreeNode* %parent, i64 0, i32 5
  store %struct.vertex** %S.coerce0, %struct.vertex*** %vertices, align 8
  br label %if.end122

if.end122:                                        ; preds = %if.then120, %sync.continue92
  ret void

eh.resume:                                        ; preds = %lpad29, %lpad81
  %ehselector.slot31.0 = phi i32 [ %30, %lpad81 ], [ %15, %lpad29 ]
  %exn.slot30.0 = phi i8* [ %29, %lpad81 ], [ %14, %lpad29 ]
  %lpad.val126 = insertvalue { i8*, i32 } undef, i8* %exn.slot30.0, 0
  %lpad.val127 = insertvalue { i8*, i32 } %lpad.val126, i32 %ehselector.slot31.0, 1
  resume { i8*, i32 } %lpad.val127

cond.false.1:                                     ; preds = %det.cont
  %41 = add nsw i64 %11, 1
  %arrayidx.1 = getelementptr inbounds i32, i32* %offsets, i64 %41
  %42 = load i32, i32* %arrayidx.1, align 4
  %conv.1 = sext i32 %42 to i64
  br label %cond.end.1

cond.end.1:                                       ; preds = %cond.false.1, %det.cont
  %cond.1 = phi i64 [ %conv.1, %cond.false.1 ], [ %S.coerce1, %det.cont ]
  %arrayidx11.1 = getelementptr inbounds i32, i32* %offsets, i64 %11
  %43 = load i32, i32* %arrayidx11.1, align 4
  %add.ptr20.1 = getelementptr inbounds %class.gTreeNode, %class.gTreeNode* %newNodes, i64 %11
  %arrayidx22.1 = getelementptr inbounds %class.gTreeNode, %class.gTreeNode* %parent, i64 0, i32 4, i64 1
  store %class.gTreeNode* %add.ptr20.1, %class.gTreeNode** %arrayidx22.1, align 8
  detach within %syncreg, label %det.achd.1, label %det.cont.1 unwind label %lpad29.loopexit

det.achd.1:                                       ; preds = %cond.end.1
  %44 = trunc i64 %cond.1 to i32
  %conv14.1 = sub i32 %44, %43
  %conv18.1 = sext i32 %conv14.1 to i64
  %idx.ext.1 = sext i32 %43 to i64
  %add.ptr.1 = getelementptr inbounds %struct.vertex*, %struct.vertex** %S.coerce0, i64 %idx.ext.1
  %div25.1 = fmul fast double %8, 5.000000e-01
  %call28.1 = invoke %class.gTreeNode* @_ZN9gTreeNodeI8_point2dIdE7_vect2dIdE6vertex5nDataIS4_EE7newTreeE4_seqIPS4_ES1_dPS7_i(%struct.vertex** %add.ptr.1, i64 %conv18.1, double %add.i.1, double %add9.i.1, double %div25.1, %class.gTreeNode* nonnull %add.ptr20.1, i32 1)
          to label %invoke.cont.1 unwind label %lpad

invoke.cont.1:                                    ; preds = %det.achd.1
  reattach within %syncreg, label %det.cont.1

det.cont.1:                                       ; preds = %invoke.cont.1, %cond.end.1
  %45 = load double, double* %size, align 8
  %div.2 = fmul fast double %45, 2.500000e-01
  %46 = load double, double* %x.i, align 8
  %add.i.2 = fsub fast double %46, %div.2
  %47 = load double, double* %y.i, align 8
  %add9.i.2 = fadd fast double %47, %div.2
  %48 = or i64 %0, 2
  %49 = trunc i64 %48 to i32
  %cmp8.2 = icmp eq i32 %sub, %49
  br i1 %cmp8.2, label %cond.end.2, label %cond.false.2

cond.false.2:                                     ; preds = %det.cont.1
  %50 = or i64 %0, 3
  %arrayidx.2 = getelementptr inbounds i32, i32* %offsets, i64 %50
  %51 = load i32, i32* %arrayidx.2, align 4
  %conv.2 = sext i32 %51 to i64
  br label %cond.end.2

cond.end.2:                                       ; preds = %cond.false.2, %det.cont.1
  %cond.2 = phi i64 [ %conv.2, %cond.false.2 ], [ %S.coerce1, %det.cont.1 ]
  %arrayidx11.2 = getelementptr inbounds i32, i32* %offsets, i64 %48
  %52 = load i32, i32* %arrayidx11.2, align 4
  %add.ptr20.2 = getelementptr inbounds %class.gTreeNode, %class.gTreeNode* %newNodes, i64 %48
  %arrayidx22.2 = getelementptr inbounds %class.gTreeNode, %class.gTreeNode* %parent, i64 0, i32 4, i64 2
  store %class.gTreeNode* %add.ptr20.2, %class.gTreeNode** %arrayidx22.2, align 8
  detach within %syncreg, label %det.achd.2, label %det.cont.2 unwind label %lpad29.loopexit

det.achd.2:                                       ; preds = %cond.end.2
  %53 = trunc i64 %cond.2 to i32
  %conv14.2 = sub i32 %53, %52
  %conv18.2 = sext i32 %conv14.2 to i64
  %idx.ext.2 = sext i32 %52 to i64
  %add.ptr.2 = getelementptr inbounds %struct.vertex*, %struct.vertex** %S.coerce0, i64 %idx.ext.2
  %div25.2 = fmul fast double %45, 5.000000e-01
  %call28.2 = invoke %class.gTreeNode* @_ZN9gTreeNodeI8_point2dIdE7_vect2dIdE6vertex5nDataIS4_EE7newTreeE4_seqIPS4_ES1_dPS7_i(%struct.vertex** %add.ptr.2, i64 %conv18.2, double %add.i.2, double %add9.i.2, double %div25.2, %class.gTreeNode* nonnull %add.ptr20.2, i32 1)
          to label %invoke.cont.2 unwind label %lpad

invoke.cont.2:                                    ; preds = %det.achd.2
  reattach within %syncreg, label %det.cont.2

det.cont.2:                                       ; preds = %invoke.cont.2, %cond.end.2
  %54 = load double, double* %size, align 8
  %div.3 = fmul fast double %54, 2.500000e-01
  %55 = load double, double* %x.i, align 8
  %add.i.3 = fadd fast double %div.3, %55
  %56 = load double, double* %y.i, align 8
  %add9.i.3 = fadd fast double %56, %div.3
  %57 = or i64 %0, 3
  %58 = trunc i64 %57 to i32
  %cmp8.3 = icmp eq i32 %sub, %58
  br i1 %cmp8.3, label %cond.end.3, label %cond.false.3

cond.false.3:                                     ; preds = %det.cont.2
  %59 = add nsw i64 %57, 1
  %arrayidx.3 = getelementptr inbounds i32, i32* %offsets, i64 %59
  %60 = load i32, i32* %arrayidx.3, align 4
  %conv.3 = sext i32 %60 to i64
  br label %cond.end.3

cond.end.3:                                       ; preds = %cond.false.3, %det.cont.2
  %cond.3 = phi i64 [ %conv.3, %cond.false.3 ], [ %S.coerce1, %det.cont.2 ]
  %arrayidx11.3 = getelementptr inbounds i32, i32* %offsets, i64 %57
  %61 = load i32, i32* %arrayidx11.3, align 4
  %add.ptr20.3 = getelementptr inbounds %class.gTreeNode, %class.gTreeNode* %newNodes, i64 %57
  %arrayidx22.3 = getelementptr inbounds %class.gTreeNode, %class.gTreeNode* %parent, i64 0, i32 4, i64 3
  store %class.gTreeNode* %add.ptr20.3, %class.gTreeNode** %arrayidx22.3, align 8
  detach within %syncreg, label %det.achd.3, label %if.end unwind label %lpad29.loopexit

det.achd.3:                                       ; preds = %cond.end.3
  %62 = trunc i64 %cond.3 to i32
  %conv14.3 = sub i32 %62, %61
  %conv18.3 = sext i32 %conv14.3 to i64
  %idx.ext.3 = sext i32 %61 to i64
  %add.ptr.3 = getelementptr inbounds %struct.vertex*, %struct.vertex** %S.coerce0, i64 %idx.ext.3
  %div25.3 = fmul fast double %54, 5.000000e-01
  %call28.3 = invoke %class.gTreeNode* @_ZN9gTreeNodeI8_point2dIdE7_vect2dIdE6vertex5nDataIS4_EE7newTreeE4_seqIPS4_ES1_dPS7_i(%struct.vertex** %add.ptr.3, i64 %conv18.3, double %add.i.3, double %add9.i.3, double %div25.3, %class.gTreeNode* nonnull %add.ptr20.3, i32 1)
          to label %invoke.cont.3 unwind label %lpad

invoke.cont.3:                                    ; preds = %det.achd.3
  reattach within %syncreg, label %if.end

det.achd75.1:                                     ; preds = %det.cont80
  %63 = or i32 %shl7, 1
  invoke void @_ZN9gTreeNodeI8_point2dIdE7_vect2dIdE6vertex5nDataIS4_EE18buildRecursiveTreeE4_seqIPS4_EPiiPS7_SC_iii(%class.gTreeNode* %this, %struct.vertex** %S.coerce0, i64 %S.coerce1, i32* %offsets, i32 %quadrants, %class.gTreeNode* nonnull %add.ptr65, %class.gTreeNode* nonnull %add.ptr52.1, i32 %63, i32 %sub73, i32 %add74)
          to label %invoke.cont79.1 unwind label %lpad76

invoke.cont79.1:                                  ; preds = %det.achd75.1
  reattach within %syncreg, label %det.cont80.1

det.cont80.1:                                     ; preds = %invoke.cont79.1, %det.cont80
  %64 = load double, double* %size, align 8
  %div44.2 = fmul fast double %64, 2.500000e-01
  %65 = load double, double* %x.i, align 8
  %add.i203.2 = fsub fast double %65, %div44.2
  %66 = load double, double* %y.i, align 8
  %add9.i208.2 = fadd fast double %66, %div44.2
  %add.ptr47.2 = getelementptr inbounds %class.gTreeNode, %class.gTreeNode* %newNodes, i64 2
  %add.ptr52.2 = getelementptr inbounds %class.gTreeNode, %class.gTreeNode* %add.ptr47.2, i64 %idx.ext51
  %div55.2 = fmul fast double %64, 5.000000e-01
  %cnt.i.i.2 = getelementptr inbounds %class.gTreeNode, %class.gTreeNode* %add.ptr52.2, i64 0, i32 2, i32 0
  store i32 0, i32* %cnt.i.i.2, align 4
  %count.i.2 = getelementptr inbounds %class.gTreeNode, %class.gTreeNode* %add.ptr52.2, i64 0, i32 3
  store i32 0, i32* %count.i.2, align 4
  %size.i.2 = getelementptr inbounds %class.gTreeNode, %class.gTreeNode* %add.ptr52.2, i64 0, i32 1
  store double %div55.2, double* %size.i.2, align 8
  %cnt.sroa.0.0..sroa_idx.i.2 = getelementptr inbounds %class.gTreeNode, %class.gTreeNode* %add.ptr52.2, i64 0, i32 0, i32 0
  store double %add.i203.2, double* %cnt.sroa.0.0..sroa_idx.i.2, align 8
  %cnt.sroa.2.0..sroa_idx3.i.2 = getelementptr inbounds %class.gTreeNode, %class.gTreeNode* %add.ptr52.2, i64 0, i32 0, i32 1
  store double %add9.i208.2, double* %cnt.sroa.2.0..sroa_idx3.i.2, align 8
  %vertices.i.2 = getelementptr inbounds %class.gTreeNode, %class.gTreeNode* %add.ptr52.2, i64 0, i32 5
  %67 = bitcast %struct.vertex*** %vertices.i.2 to i8*
  tail call void @llvm.memset.p0i8.i64(i8* nonnull %67, i8 0, i64 16, i32 8, i1 false)
  %arrayidx58.2 = getelementptr inbounds %class.gTreeNode, %class.gTreeNode* %parent, i64 0, i32 4, i64 2
  store %class.gTreeNode* %add.ptr52.2, %class.gTreeNode** %arrayidx58.2, align 8
  detach within %syncreg, label %det.achd75.2, label %det.cont80.2 unwind label %lpad81.loopexit

det.achd75.2:                                     ; preds = %det.cont80.1
  %68 = or i32 %shl7, 2
  invoke void @_ZN9gTreeNodeI8_point2dIdE7_vect2dIdE6vertex5nDataIS4_EE18buildRecursiveTreeE4_seqIPS4_EPiiPS7_SC_iii(%class.gTreeNode* %this, %struct.vertex** %S.coerce0, i64 %S.coerce1, i32* %offsets, i32 %quadrants, %class.gTreeNode* nonnull %add.ptr65, %class.gTreeNode* nonnull %add.ptr52.2, i32 %68, i32 %sub73, i32 %add74)
          to label %invoke.cont79.2 unwind label %lpad76

invoke.cont79.2:                                  ; preds = %det.achd75.2
  reattach within %syncreg, label %det.cont80.2

det.cont80.2:                                     ; preds = %invoke.cont79.2, %det.cont80.1
  %69 = load double, double* %size, align 8
  %div44.3 = fmul fast double %69, 2.500000e-01
  %70 = bitcast %class.gTreeNode* %parent to <2 x double>*
  %71 = load <2 x double>, <2 x double>* %70, align 8
  %72 = insertelement <2 x double> undef, double %div44.3, i32 0
  %73 = shufflevector <2 x double> %72, <2 x double> undef, <2 x i32> zeroinitializer
  %74 = fadd fast <2 x double> %73, %71
  %add.ptr47.3 = getelementptr inbounds %class.gTreeNode, %class.gTreeNode* %newNodes, i64 3
  %add.ptr52.3 = getelementptr inbounds %class.gTreeNode, %class.gTreeNode* %add.ptr47.3, i64 %idx.ext51
  %div55.3 = fmul fast double %69, 5.000000e-01
  %cnt.i.i.3 = getelementptr inbounds %class.gTreeNode, %class.gTreeNode* %add.ptr52.3, i64 0, i32 2, i32 0
  store i32 0, i32* %cnt.i.i.3, align 4
  %count.i.3 = getelementptr inbounds %class.gTreeNode, %class.gTreeNode* %add.ptr52.3, i64 0, i32 3
  store i32 0, i32* %count.i.3, align 4
  %size.i.3 = getelementptr inbounds %class.gTreeNode, %class.gTreeNode* %add.ptr52.3, i64 0, i32 1
  store double %div55.3, double* %size.i.3, align 8
  %75 = bitcast %class.gTreeNode* %add.ptr52.3 to <2 x double>*
  store <2 x double> %74, <2 x double>* %75, align 8
  %vertices.i.3 = getelementptr inbounds %class.gTreeNode, %class.gTreeNode* %add.ptr52.3, i64 0, i32 5
  %76 = bitcast %struct.vertex*** %vertices.i.3 to i8*
  tail call void @llvm.memset.p0i8.i64(i8* nonnull %76, i8 0, i64 16, i32 8, i1 false)
  %arrayidx58.3 = getelementptr inbounds %class.gTreeNode, %class.gTreeNode* %parent, i64 0, i32 4, i64 3
  store %class.gTreeNode* %add.ptr52.3, %class.gTreeNode** %arrayidx58.3, align 8
  detach within %syncreg, label %det.achd75.3, label %if.end unwind label %lpad81.loopexit

det.achd75.3:                                     ; preds = %det.cont80.2
  %77 = or i32 %shl7, 3
  invoke void @_ZN9gTreeNodeI8_point2dIdE7_vect2dIdE6vertex5nDataIS4_EE18buildRecursiveTreeE4_seqIPS4_EPiiPS7_SC_iii(%class.gTreeNode* %this, %struct.vertex** %S.coerce0, i64 %S.coerce1, i32* %offsets, i32 %quadrants, %class.gTreeNode* nonnull %add.ptr65, %class.gTreeNode* nonnull %add.ptr52.3, i32 %77, i32 %sub73, i32 %add74)
          to label %invoke.cont79.3 unwind label %lpad76

invoke.cont79.3:                                  ; preds = %det.achd75.3
  reattach within %syncreg, label %if.end
}

; Function Attrs: argmemonly nounwind
declare token @llvm.syncregion.start() #4

declare i32 @__gxx_personality_v0(...)

; Function Attrs: argmemonly
declare void @llvm.detached.rethrow.sl_p0i8i32s(token, { i8*, i32 }) #9

; Function Attrs: argmemonly nounwind
declare void @llvm.memset.p0i8.i64(i8* nocapture writeonly, i8, i64, i32, i1) #4

declare %class.gTreeNode* @_ZN9gTreeNodeI8_point2dIdE7_vect2dIdE6vertex5nDataIS4_EE7newTreeE4_seqIPS4_ES1_dPS7_i(%struct.vertex** %S.coerce0, i64 %S.coerce1, double %cnt.coerce0, double %cnt.coerce1, double %sz, %class.gTreeNode* %newNodes, i32 %numNewNodes) local_unnamed_addr #7

; CHECK: task at depth 0 containing: <task entry><func sp entry>%entry,%if.else<sp exit>
; CHECK-DAG: <shared EH><phi sp entry>%lpad,%lpad29.loopexit.split-lp<sp exit>,%det.rethrow.unreachable<sp exit><task EH exit>
; CHECK-DAG: <shared EH><phi sp entry>%lpad76,%lpad81.loopexit.split-lp<sp exit>,%det.rethrow.unreachable87<sp exit><task EH exit>
; CHECK: task at depth 1 containing: <task entry><task sp entry>%det.achd75.3<sp exit>
; CHECK: <phi sp entry>%lpad76,%lpad81.loopexit.split-lp<sp exit>,%det.rethrow.unreachable87<sp exit><task EH exit>
; CHECK: task at depth 1 containing: <task entry><task sp entry>%det.achd75.2<sp exit>
; CHECK: <phi sp entry>%lpad76,%lpad81.loopexit.split-lp<sp exit>,%det.rethrow.unreachable87<sp exit><task EH exit>
; CHECK: task at depth 1 containing: <task entry><task sp entry>%det.achd75.1<sp exit>
; CHECK: <phi sp entry>%lpad76,%lpad81.loopexit.split-lp<sp exit>,%det.rethrow.unreachable87<sp exit><task EH exit>
; CHECK: task at depth 1 containing: <task entry><task sp entry>%det.achd75<sp exit>
; CHECK: <phi sp entry>%lpad76,%lpad81.loopexit.split-lp<sp exit>,%det.rethrow.unreachable87<sp exit><task EH exit>
; CHECK: task at depth 1 containing: <task entry><task sp entry>%det.achd.3<sp exit>
; CHECK: <phi sp entry>%lpad,%lpad29.loopexit.split-lp<sp exit>,%det.rethrow.unreachable<sp exit><task EH exit>
; CHECK: task at depth 1 containing: <task entry><task sp entry>%det.achd.2<sp exit>
; CHECK: <phi sp entry>%lpad,%lpad29.loopexit.split-lp<sp exit>,%det.rethrow.unreachable<sp exit><task EH exit>
; CHECK: task at depth 1 containing: <task entry><task sp entry>%det.achd.1<sp exit>
; CHECK: <phi sp entry>%lpad,%lpad29.loopexit.split-lp<sp exit>,%det.rethrow.unreachable<sp exit><task EH exit>
; CHECK: task at depth 1 containing: <task entry><task sp entry>%det.achd<sp exit>
; CHECK: <phi sp entry>%lpad,%lpad29.loopexit.split-lp<sp exit>,%det.rethrow.unreachable<sp exit><task EH exit>
