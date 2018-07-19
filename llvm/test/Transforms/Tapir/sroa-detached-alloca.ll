; RUN: opt < %s -sroa -S | FileCheck %s

%"class.std::basic_ostream" = type { i32 (...)**, %"class.std::basic_ios" }
%"class.std::basic_ios" = type { %"class.std::ios_base", %"class.std::basic_ostream"*, i8, i8, %"class.std::basic_streambuf"*, %"class.std::ctype"*, %"class.std::num_put"*, %"class.std::num_get"* }%struct.tri = type { [3 x %struct.tri*], [3 x %struct.vertex*], i32, i8, i8 }
%"class.std::ios_base" = type { i32 (...)**, i64, i64, i32, i32, i32, %"struct.std::ios_base::_Callback_list"*, %"struct.std::ios_base::_Words", [8 x %"struct.std::ios_base::_Words"], i32, %"struct.std::ios_base::_Words"*, %"class.std::locale" }
%"struct.std::ios_base::_Callback_list" = type { %"struct.std::ios_base::_Callback_list"*, void (i32, %"class.std::ios_base"*, i32)*, i32, i32 }
%"struct.std::ios_base::_Words" = type { i8*, i64 }
%"class.std::locale" = type { %"class.std::locale::_Impl"* }
%"class.std::locale::_Impl" = type { i32, %"class.std::locale::facet"**, i64, %"class.std::locale::facet"**, i8** }
%"class.std::locale::facet" = type <{ i32 (...)**, i32, [4 x i8] }>
%"class.std::basic_streambuf" = type { i32 (...)**, i8*, i8*, i8*, i8*, i8*, i8*, %"class.std::locale" }
%"class.std::ctype" = type <{ %"class.std::locale::facet.base", [4 x i8], %struct.__locale_struct*, i8, [7 x i8], i32*, i32*, i16*, i8, [256 x i8], [256 x i8], i8, [6 x i8] }>
%"class.std::locale::facet.base" = type <{ i32 (...)**, i32 }>
%struct.__locale_struct = type { [13 x %struct.__locale_data*], i16*, i32*, i32*, [13 x i8*] }
%struct.__locale_data = type opaque
%"class.std::num_put" = type { %"class.std::locale::facet.base", [4 x i8] }
%"class.std::num_get" = type { %"class.std::locale::facet.base", [4 x i8] }
%struct.vertex = type { %class._point2d, %struct.tri*, %struct.tri*, i32, i32 }
%class._point2d = type { double, double }
%struct.simplex = type <{ %struct.tri*, i32, i8, [3 x i8] }>

@_ZSt4cout = external global %"class.std::basic_ostream", align 8
@.str = private unnamed_addr constant [53 x i8] c"delaunayCheck: neighbor inside triangle at triangle \00", align 1
@.str.4 = private unnamed_addr constant [33 x i8] c"In Circle Violation at triangle \00", align 1
@.str.5 = private unnamed_addr constant [46 x i8] c"did not locate back pointer in triangulation\0A\00", align 1

; Function Attrs: uwtable
define zeroext i1 @_Z13checkDelaunayP3triii(%struct.tri* %triangs, i32 %n, i32 %boundarySize) local_unnamed_addr #0 personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*) {
entry:
  %syncreg = call token @llvm.syncregion.start()
  %insideOutError = alloca i32, align 4
  %inCircleError = alloca i32, align 4
  %syncreg5 = call token @llvm.syncregion.start()
  %conv = sext i32 %n to i64
  %mul = shl nsw i64 %conv, 2
  %call = call noalias i8* @malloc(i64 %mul) #3
  %0 = bitcast i8* %call to i32*
  br label %pfor.cond

pfor.cond:                                        ; preds = %pfor.inc, %entry
  %__begin.0 = phi i32 [ 0, %entry ], [ %inc, %pfor.inc ]
  %cmp = icmp slt i32 %__begin.0, %n
  br i1 %cmp, label %pfor.detach, label %pfor.cond.cleanup

pfor.cond.cleanup:                                ; preds = %pfor.cond
  sync within %syncreg, label %sync.continue

pfor.detach:                                      ; preds = %pfor.cond
  detach within %syncreg, label %pfor.body.entry, label %pfor.inc

pfor.body.entry:                                  ; preds = %pfor.detach
  br label %pfor.body

pfor.body:                                        ; preds = %pfor.body.entry
  %1 = zext i32 %__begin.0 to i64
  %arrayidx = getelementptr inbounds i32, i32* %0, i64 %1
  store i32 0, i32* %arrayidx, align 4
  reattach within %syncreg, label %pfor.inc

pfor.inc:                                         ; preds = %pfor.body, %pfor.detach
  %inc = add nuw nsw i32 %__begin.0, 1
  br label %pfor.cond

sync.continue:                                    ; preds = %pfor.cond.cleanup
  %2 = bitcast i32* %insideOutError to i8*
  call void @llvm.lifetime.start.p0i8(i64 4, i8* %2) #3
  store i32 %n, i32* %insideOutError, align 4
  %3 = bitcast i32* %inCircleError to i8*
  call void @llvm.lifetime.start.p0i8(i64 4, i8* %3) #3
  store i32 %n, i32* %inCircleError, align 4
  br label %pfor.cond13

pfor.cond13:                                      ; preds = %pfor.inc110, %sync.continue
  %__begin7.0 = phi i32 [ 0, %sync.continue ], [ %inc111, %pfor.inc110 ]
  %cmp14 = icmp slt i32 %__begin7.0, %n
  br i1 %cmp14, label %pfor.detach16, label %pfor.cond.cleanup15

pfor.cond.cleanup15:                              ; preds = %pfor.cond13
  sync within %syncreg5, label %sync.continue116

pfor.detach16:                                    ; preds = %pfor.cond13
  detach within %syncreg5, label %pfor.body.entry19, label %pfor.inc110 unwind label %lpad112

pfor.body.entry19:                                ; preds = %pfor.detach16
  %t = alloca %struct.simplex, align 8
  %a = alloca { %struct.tri*, i64 }, align 8
  %tmpcast = bitcast { %struct.tri*, i64 }* %a to %struct.simplex*
  %ref.tmp = alloca { %struct.tri*, i64 }, align 8
  %tmpcast207 = bitcast { %struct.tri*, i64 }* %ref.tmp to %struct.simplex*
  br label %pfor.body20

; CHECK: pfor.body.entry19:
; CHECK-NOT: %t = alloca

pfor.body20:                                      ; preds = %pfor.body.entry19
  %4 = zext i32 %__begin7.0 to i64
  %initialized = getelementptr inbounds %struct.tri, %struct.tri* %triangs, i64 %4, i32 3
  %5 = load i8, i8* %initialized, align 4
  %tobool = icmp eq i8 %5, 0
  br i1 %tobool, label %pfor.preattach108, label %if.then

if.then:                                          ; preds = %pfor.body20
  %arrayidx22 = getelementptr inbounds %struct.tri, %struct.tri* %triangs, i64 %4
  %6 = bitcast %struct.simplex* %t to i8*
  call void @llvm.lifetime.start.p0i8(i64 16, i8* %6) #3
  %t.i = getelementptr inbounds %struct.simplex, %struct.simplex* %t, i64 0, i32 0
  store %struct.tri* %arrayidx22, %struct.tri** %t.i, align 8
  %o.i = getelementptr inbounds %struct.simplex, %struct.simplex* %t, i64 0, i32 1
  store i32 0, i32* %o.i, align 8
  %boundary.i = getelementptr inbounds %struct.simplex, %struct.simplex* %t, i64 0, i32 2
  store i8 0, i8* %boundary.i, align 4
  br label %for.cond

; CHECK: if.then:
; CHECK-NOT: bitcast %struct.simplex* %t
; CHECK-NOT: %t.i = getelementptr inbounds %struct.simplex, %struct.simplex* %t, i64 0, i32 0
; CHECK-NOT: store %struct.tri* %arrayidx22, %struct.tri** %t.i, align 8
; CHECK-NOT: %boundary.i = getelementptr inbounds %struct.simplex, %struct.simplex* %t, i64 0, i32 2
; CHECK: %t.sroa.11.0.insert.mask = and i40 undef, -4294967296
; CHECK: %t.sroa.11.4.insert.mask = and i40 %t.sroa.11.0.insert.mask, 4294967295

for.cond:                                         ; preds = %invoke.cont100, %if.then
  %j25.0 = phi i32 [ 0, %if.then ], [ %inc104, %invoke.cont100 ]
  %cmp26 = icmp ult i32 %j25.0, 3
  br i1 %cmp26, label %for.body, label %for.cond.cleanup

; CHECK: for.cond:
; CHECK: %t.sroa.0.0 = phi %struct.tri* [ %arrayidx22, %if.then ], [ %52, %invoke.cont100 ]
; CHECK: %t.sroa.11.0 = phi i40 [ %t.sroa.11.4.insert.mask, %if.then ], [ %ref.tmp98.sroa.5.0.extract.trunc, %invoke.cont100 ]

for.cond.cleanup:                                 ; preds = %for.cond
  call void @llvm.lifetime.end.p0i8(i64 16, i8* %6) #3
  br label %pfor.preattach108

; CHECK: for.cond.cleanup:
; CHECK-NOT: call void @llvm.lifetime.end.p0i8(i64 16, i8* %6) #3

for.body:                                         ; preds = %for.cond
  %7 = bitcast { %struct.tri*, i64 }* %a to i8*
  call void @llvm.lifetime.start.p0i8(i64 16, i8* %7) #3
  %t.i209 = getelementptr inbounds %struct.simplex, %struct.simplex* %t, i64 0, i32 0
  %8 = load %struct.tri*, %struct.tri** %t.i209, align 8
  %o.i210 = getelementptr inbounds %struct.simplex, %struct.simplex* %t, i64 0, i32 1
  %9 = load i32, i32* %o.i210, align 8
  %idxprom.i = sext i32 %9 to i64
  %arrayidx.i = getelementptr inbounds %struct.tri, %struct.tri* %8, i64 0, i32 0, i64 %idxprom.i
  %10 = load %struct.tri*, %struct.tri** %arrayidx.i, align 8
  %cmp.i = icmp eq %struct.tri* %10, null
  br i1 %cmp.i, label %if.else.i, label %if.then.i

; CHECK: for.body:
; CHECK-NOT: %t.i209 = getelementptr inbounds %struct.simplex, %struct.simplex* %t, i64 0, i32 0
; CHECK-NOT: %8 = load %struct.tri*, %struct.tri** %t.i209, align 8
; CHECK-NOT: %o.i210 = getelementptr inbounds %struct.simplex, %struct.simplex* %t, i64 0, i32 1
; CHECK-NOT: %9 = load i32, i32* %o.i210, align 8
; CHECK-NOT: %idxprom.i = sext i32 %9 to i64
; CHECK-NOT: %arrayidx.i = getelementptr inbounds %struct.tri, %struct.tri* %8, i64 0, i32 0, i64 %idxprom.i
; CHECK-NOT: %10 = load %struct.tri*, %struct.tri** %arrayidx.i, align 8, !tbaa !21
; CHECK: %t.sroa.11.0.extract.trunc27 = trunc i40 %t.sroa.11.0 to i32
; CHECK: %idxprom.i = sext i32 %t.sroa.11.0.extract.trunc27 to i64
; CHECK: %arrayidx.i = getelementptr inbounds %struct.tri, %struct.tri* %t.sroa.0.0, i64 0, i32 0, i64 %idxprom.i
; CHECK: %6 = load %struct.tri*, %struct.tri** %arrayidx.i, align 8

if.then.i:                                        ; preds = %for.body
  %arrayidx.i.i = getelementptr inbounds %struct.tri, %struct.tri* %10, i64 0, i32 0, i64 0
  %11 = load %struct.tri*, %struct.tri** %arrayidx.i.i, align 8
  %cmp2.i.i = icmp eq %struct.tri* %11, %8
  br i1 %cmp2.i.i, label %_ZN7simplex6acrossEv.exit, label %for.inc.i.i

; CHECK: if.then.i:
; CHECK-NOT: %cmp2.i.i = icmp eq %struct.tri* %11, %8
; CHECK: %cmp2.i.i = icmp eq %struct.tri* %7, %t.sroa.0.0

for.inc.i.i:                                      ; preds = %if.then.i
  %arrayidx.1.i.i = getelementptr inbounds %struct.tri, %struct.tri* %10, i64 0, i32 0, i64 1
  %12 = load %struct.tri*, %struct.tri** %arrayidx.1.i.i, align 8
  %cmp2.1.i.i = icmp eq %struct.tri* %12, %8
  br i1 %cmp2.1.i.i, label %_ZN7simplex6acrossEv.exit, label %for.inc.1.i.i

; CHECK: for.inc.i.i:
; CHECK-NOT: %cmp2.1.i.i = icmp eq %struct.tri* %12, %8
; CHECK: %cmp2.1.i.i = icmp eq %struct.tri* %8, %t.sroa.0.0

for.inc.1.i.i:                                    ; preds = %for.inc.i.i
  %arrayidx.2.i.i = getelementptr inbounds %struct.tri, %struct.tri* %10, i64 0, i32 0, i64 2
  %13 = load %struct.tri*, %struct.tri** %arrayidx.2.i.i, align 8
  %cmp2.2.i.i = icmp eq %struct.tri* %13, %8
  br i1 %cmp2.2.i.i, label %_ZN7simplex6acrossEv.exit, label %for.inc.2.i.i

; CHECK: for.inc.1.i.i:
; CHECK-NOT: %cmp2.2.i.i = icmp eq %struct.tri* %13, %8
; CHECK: %cmp2.2.i.i = icmp eq %struct.tri* %9, %t.sroa.0.0

for.inc.2.i.i:                                    ; preds = %for.inc.1.i.i
  %call.i.i211 = invoke dereferenceable(272) %"class.std::basic_ostream"* @_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc(%"class.std::basic_ostream"* nonnull dereferenceable(272) @_ZSt4cout, i8* getelementptr inbounds ([46 x i8], [46 x i8]* @.str.5, i64 0, i64 0))
          to label %call.i.i.noexc unwind label %lpad28

call.i.i.noexc:                                   ; preds = %for.inc.2.i.i
  call void @abort() #16
  unreachable

if.else.i:                                        ; preds = %for.body
  %retval.sroa.3.8.insert.ext13.i = zext i32 %9 to i64
  %retval.sroa.3.12.insert.insert.i = or i64 %retval.sroa.3.8.insert.ext13.i, 4294967296
  br label %_ZN7simplex6acrossEv.exit

; CHECK: if.else.i:
; CHECK-NOT: %retval.sroa.3.8.insert.ext13.i = zext i32 %9 to i64
; CHECK: %retval.sroa.3.8.insert.ext13.i = zext i32 %t.sroa.11.0.extract.trunc27 to i64

_ZN7simplex6acrossEv.exit:                        ; preds = %if.then.i, %for.inc.i.i, %for.inc.1.i.i, %if.else.i
  %retval.sroa.3.0.i = phi i64 [ %retval.sroa.3.12.insert.insert.i, %if.else.i ], [ 0, %if.then.i ], [ 1, %for.inc.i.i ], [ 2, %for.inc.1.i.i ]
  %retval.sroa.0.0.i = phi %struct.tri* [ %8, %if.else.i ], [ %10, %if.then.i ], [ %10, %for.inc.i.i ], [ %10, %for.inc.1.i.i ]
  %.fca.0.insert.i = insertvalue { %struct.tri*, i64 } undef, %struct.tri* %retval.sroa.0.0.i, 0
  %.fca.1.insert.i = insertvalue { %struct.tri*, i64 } %.fca.0.insert.i, i64 %retval.sroa.3.0.i, 1
  br label %invoke.cont29

; CHECK: _ZN7simplex6acrossEv.exit:
; CHECK-NOT: %retval.sroa.0.0.i = phi %struct.tri* [ %8, %if.else.i ], [ %10, %if.then.i ], [ %10, %for.inc.i.i ], [ %10, %for.inc.1.i.i ]
; CHECK: %retval.sroa.0.0.i = phi %struct.tri* [ %t.sroa.0.0, %if.else.i ], [ %6, %if.then.i ], [ %6, %for.inc.i.i ], [ %6, %for.inc.1.i.i ]

invoke.cont29:                                    ; preds = %_ZN7simplex6acrossEv.exit
  %14 = getelementptr inbounds { %struct.tri*, i64 }, { %struct.tri*, i64 }* %a, i64 0, i32 0
  %15 = extractvalue { %struct.tri*, i64 } %.fca.1.insert.i, 0
  store %struct.tri* %15, %struct.tri** %14, align 8
  %16 = getelementptr inbounds { %struct.tri*, i64 }, { %struct.tri*, i64 }* %a, i64 0, i32 1
  %17 = extractvalue { %struct.tri*, i64 } %.fca.1.insert.i, 1
  store i64 %17, i64* %16, align 8
  %boundary.i212 = getelementptr inbounds %struct.simplex, %struct.simplex* %tmpcast, i64 0, i32 2
  %18 = load i8, i8* %boundary.i212, align 4
  %tobool.i = icmp eq i8 %18, 0
  br i1 %tobool.i, label %if.then33, label %if.else

if.then33:                                        ; preds = %invoke.cont29
  %19 = bitcast { %struct.tri*, i64 }* %ref.tmp to i8*
  call void @llvm.lifetime.start.p0i8(i64 16, i8* %19) #3
  %t.i213 = getelementptr inbounds %struct.simplex, %struct.simplex* %tmpcast, i64 0, i32 0
  %20 = load %struct.tri*, %struct.tri** %t.i213, align 8
  %o.i214 = getelementptr inbounds %struct.simplex, %struct.simplex* %tmpcast, i64 0, i32 1
  %21 = load i32, i32* %o.i214, align 8
  %cmp.i.i = icmp sgt i32 %21, 1
  %cond.i.v.i = select i1 %cmp.i.i, i32 -2, i32 1
  %cond.i.i = add i32 %cond.i.v.i, %21
  %retval.sroa.2.8.insert.ext.i = zext i32 %cond.i.i to i64
  %.fca.0.insert.i215 = insertvalue { %struct.tri*, i64 } undef, %struct.tri* %20, 0
  %.fca.1.insert.i216 = insertvalue { %struct.tri*, i64 } %.fca.0.insert.i215, i64 %retval.sroa.2.8.insert.ext.i, 1
  br label %invoke.cont35

invoke.cont35:                                    ; preds = %if.then33
  %22 = getelementptr inbounds { %struct.tri*, i64 }, { %struct.tri*, i64 }* %ref.tmp, i64 0, i32 0
  %23 = extractvalue { %struct.tri*, i64 } %.fca.1.insert.i216, 0
  store %struct.tri* %23, %struct.tri** %22, align 8
  %24 = getelementptr inbounds { %struct.tri*, i64 }, { %struct.tri*, i64 }* %ref.tmp, i64 0, i32 1
  %25 = extractvalue { %struct.tri*, i64 } %.fca.1.insert.i216, 1
  store i64 %25, i64* %24, align 8
  %t.i217 = getelementptr inbounds %struct.simplex, %struct.simplex* %tmpcast207, i64 0, i32 0
  %26 = load %struct.tri*, %struct.tri** %t.i217, align 8
  %o.i218 = getelementptr inbounds %struct.simplex, %struct.simplex* %tmpcast207, i64 0, i32 1
  %27 = load i32, i32* %o.i218, align 8
  %idxprom.i219 = sext i32 %27 to i64
  %arrayidx.i220 = getelementptr inbounds %struct.tri, %struct.tri* %26, i64 0, i32 1, i64 %idxprom.i219
  %28 = load %struct.vertex*, %struct.vertex** %arrayidx.i220, align 8
  call void @llvm.lifetime.end.p0i8(i64 16, i8* %19) #3
  %boundary.i227 = getelementptr inbounds %struct.simplex, %struct.simplex* %t, i64 0, i32 2
  %29 = load i8, i8* %boundary.i227, align 4
  %tobool.i228 = icmp eq i8 %29, 0
  br i1 %tobool.i228, label %lor.lhs.false.i, label %_ZN7simplex7outsideEP6vertex.exit

; CHECK: invoke.cont35:
; CHECK-NOT: %boundary.i227 = getelementptr inbounds %struct.simplex, %struct.simplex* %t, i64 0, i32 2
; CHECK-NOT: %29 = load i8, i8* %boundary.i227, align 4
; CHECK: %t.sroa.11.4.extract.shift34 = lshr i40 %t.sroa.11.0, 32
; CHECK: %t.sroa.11.4.extract.trunc35 = trunc i40 %t.sroa.11.4.extract.shift34 to i8

lor.lhs.false.i:                                  ; preds = %invoke.cont35
  %t.i229 = getelementptr inbounds %struct.simplex, %struct.simplex* %t, i64 0, i32 0
  %30 = load %struct.tri*, %struct.tri** %t.i229, align 8
  %cmp.i230 = icmp eq %struct.tri* %30, null
  br i1 %cmp.i230, label %_ZN7simplex7outsideEP6vertex.exit, label %if.end.i237

; CHECK: lor.lhs.false.i:
; CHECK-NOT: %t.i229 = getelementptr inbounds %struct.simplex, %struct.simplex* %t, i64 0, i32 0
; CHECK-NOT: %30 = load %struct.tri*, %struct.tri** %t.i229, align 8
; CHECK: %cmp.i230 = icmp eq %struct.tri* %t.sroa.0.0, null

if.end.i237:                                      ; preds = %lor.lhs.false.i
  %o.i231 = getelementptr inbounds %struct.simplex, %struct.simplex* %t, i64 0, i32 1
  %31 = load i32, i32* %o.i231, align 8
  %cmp.i.i232 = icmp sgt i32 %31, 0
  %cond.i.v.i233 = select i1 %cmp.i.i232, i32 -1, i32 2
  %cond.i.i234 = add i32 %cond.i.v.i233, %31
  %idxprom.i235 = sext i32 %cond.i.i234 to i64
  %arrayidx.i236 = getelementptr inbounds %struct.tri, %struct.tri* %30, i64 0, i32 1, i64 %idxprom.i235
  %32 = load %struct.vertex*, %struct.vertex** %arrayidx.i236, align 8
  %agg.tmp.sroa.0.0..sroa_idx.i = getelementptr inbounds %struct.vertex, %struct.vertex* %32, i64 0, i32 0, i32 0
  %agg.tmp.sroa.0.0.copyload.i = load double, double* %agg.tmp.sroa.0.0..sroa_idx.i, align 8
  %agg.tmp.sroa.2.0..sroa_idx16.i = getelementptr inbounds %struct.vertex, %struct.vertex* %32, i64 0, i32 0, i32 1
  %agg.tmp.sroa.2.0.copyload.i = load double, double* %agg.tmp.sroa.2.0..sroa_idx16.i, align 8
  %agg.tmp3.sroa.0.0..sroa_idx.i = getelementptr inbounds %struct.vertex, %struct.vertex* %28, i64 0, i32 0, i32 0
  %agg.tmp3.sroa.0.0.copyload.i = load double, double* %agg.tmp3.sroa.0.0..sroa_idx.i, align 8
  %agg.tmp3.sroa.2.0..sroa_idx15.i = getelementptr inbounds %struct.vertex, %struct.vertex* %28, i64 0, i32 0, i32 1
  %agg.tmp3.sroa.2.0.copyload.i = load double, double* %agg.tmp3.sroa.2.0..sroa_idx15.i, align 8
  %idxprom9.i = sext i32 %31 to i64
  %arrayidx10.i = getelementptr inbounds %struct.tri, %struct.tri* %30, i64 0, i32 1, i64 %idxprom9.i
  %33 = load %struct.vertex*, %struct.vertex** %arrayidx10.i, align 8
  %agg.tmp5.sroa.0.0..sroa_idx.i = getelementptr inbounds %struct.vertex, %struct.vertex* %33, i64 0, i32 0, i32 0
  %agg.tmp5.sroa.0.0.copyload.i = load double, double* %agg.tmp5.sroa.0.0..sroa_idx.i, align 8
  %agg.tmp5.sroa.2.0..sroa_idx14.i = getelementptr inbounds %struct.vertex, %struct.vertex* %33, i64 0, i32 0, i32 1
  %agg.tmp5.sroa.2.0.copyload.i = load double, double* %agg.tmp5.sroa.2.0..sroa_idx14.i, align 8
  %sub.i.i.i = fsub fast double %agg.tmp3.sroa.0.0.copyload.i, %agg.tmp.sroa.0.0.copyload.i
  %sub4.i.i.i = fsub fast double %agg.tmp3.sroa.2.0.copyload.i, %agg.tmp.sroa.2.0.copyload.i
  %sub.i10.i.i = fsub fast double %agg.tmp5.sroa.0.0.copyload.i, %agg.tmp.sroa.0.0.copyload.i
  %sub4.i12.i.i = fsub fast double %agg.tmp5.sroa.2.0.copyload.i, %agg.tmp.sroa.2.0.copyload.i
  %mul.i.i.i = fmul fast double %sub4.i12.i.i, %sub.i.i.i
  %mul4.i.i.i = fmul fast double %sub.i10.i.i, %sub4.i.i.i
  %sub.i8.i.i = fsub fast double %mul.i.i.i, %mul4.i.i.i
  %cmp.i17.i = fcmp fast ogt double %sub.i8.i.i, 0.000000e+00
  br label %_ZN7simplex7outsideEP6vertex.exit

; CHECK: if.end.i237:
; CHECK-NOT: %o.i231 = getelementptr inbounds %struct.simplex, %struct.simplex* %t, i64 0, i32 1
; CHECK-NOT: %31 = load i32, i32* %o.i231, align 8
; CHECK: %t.sroa.11.0.extract.trunc25 = trunc i40 %t.sroa.11.0 to i32
; CHECK: %cmp.i.i232 = icmp sgt i32 %t.sroa.11.0.extract.trunc25, 0
; CHECK: %cond.i.i234 = add i32 %cond.i.v.i233, %t.sroa.11.0.extract.trunc25

_ZN7simplex7outsideEP6vertex.exit:                ; preds = %invoke.cont35, %lor.lhs.false.i, %if.end.i237
  %retval.0.i238 = phi i1 [ %cmp.i17.i, %if.end.i237 ], [ false, %invoke.cont35 ], [ false, %lor.lhs.false.i ]
  br label %invoke.cont40

invoke.cont40:                                    ; preds = %_ZN7simplex7outsideEP6vertex.exit
  br i1 %retval.0.i238, label %if.end63, label %if.then42

if.then42:                                        ; preds = %invoke.cont40
  %t43 = getelementptr inbounds %struct.simplex, %struct.simplex* %t, i64 0, i32 0
  %34 = load %struct.tri*, %struct.tri** %t43, align 8
  %o = getelementptr inbounds %struct.simplex, %struct.simplex* %t, i64 0, i32 1
  %35 = load i32, i32* %o, align 8
  %add44 = add nsw i32 %35, 2
  %rem = srem i32 %add44, 3
  %idxprom45 = sext i32 %rem to i64
  %arrayidx46 = getelementptr inbounds %struct.tri, %struct.tri* %34, i64 0, i32 1, i64 %idxprom45
  %36 = load %struct.vertex*, %struct.vertex** %arrayidx46, align 8
  %agg.tmp.sroa.0.0..sroa_idx = getelementptr inbounds %struct.vertex, %struct.vertex* %36, i64 0, i32 0, i32 0
  %agg.tmp.sroa.0.0.copyload = load double, double* %agg.tmp.sroa.0.0..sroa_idx, align 8
  %agg.tmp.sroa.2.0..sroa_idx162 = getelementptr inbounds %struct.vertex, %struct.vertex* %36, i64 0, i32 0, i32 1
  %agg.tmp.sroa.2.0.copyload = load double, double* %agg.tmp.sroa.2.0..sroa_idx162, align 8
  %agg.tmp47.sroa.0.0..sroa_idx = getelementptr inbounds %struct.vertex, %struct.vertex* %28, i64 0, i32 0, i32 0
  %agg.tmp47.sroa.0.0.copyload = load double, double* %agg.tmp47.sroa.0.0..sroa_idx, align 8
  %agg.tmp47.sroa.2.0..sroa_idx161 = getelementptr inbounds %struct.vertex, %struct.vertex* %28, i64 0, i32 0, i32 1
  %agg.tmp47.sroa.2.0.copyload = load double, double* %agg.tmp47.sroa.2.0..sroa_idx161, align 8
  %idxprom53 = sext i32 %35 to i64
  %arrayidx54 = getelementptr inbounds %struct.tri, %struct.tri* %34, i64 0, i32 1, i64 %idxprom53
  %37 = load %struct.vertex*, %struct.vertex** %arrayidx54, align 8
  %agg.tmp49.sroa.0.0..sroa_idx = getelementptr inbounds %struct.vertex, %struct.vertex* %37, i64 0, i32 0, i32 0
  %agg.tmp49.sroa.0.0.copyload = load double, double* %agg.tmp49.sroa.0.0..sroa_idx, align 8
  %agg.tmp49.sroa.2.0..sroa_idx160 = getelementptr inbounds %struct.vertex, %struct.vertex* %37, i64 0, i32 0, i32 1
  %agg.tmp49.sroa.2.0.copyload = load double, double* %agg.tmp49.sroa.2.0..sroa_idx160, align 8
  %sub.i.i.i239 = fsub fast double %agg.tmp47.sroa.0.0.copyload, %agg.tmp.sroa.0.0.copyload
  %sub4.i.i.i240 = fsub fast double %agg.tmp47.sroa.2.0.copyload, %agg.tmp.sroa.2.0.copyload
  %sub.i10.i.i241 = fsub fast double %agg.tmp49.sroa.0.0.copyload, %agg.tmp.sroa.0.0.copyload
  %sub4.i12.i.i242 = fsub fast double %agg.tmp49.sroa.2.0.copyload, %agg.tmp.sroa.2.0.copyload
  %mul.i.i.i243 = fmul fast double %sub4.i12.i.i242, %sub.i.i.i239
  %mul4.i.i.i244 = fmul fast double %sub.i10.i.i241, %sub4.i.i.i240
  %sub.i8.i.i245 = fsub fast double %mul.i.i.i243, %mul4.i.i.i244
  %mul.i18.i = fmul fast double %sub.i.i.i239, %sub.i.i.i239
  %mul4.i20.i = fmul fast double %sub4.i.i.i240, %sub4.i.i.i240
  %add.i21.i = fadd fast double %mul4.i20.i, %mul.i18.i
  %mul.i.i = fmul fast double %sub.i10.i.i241, %sub.i10.i.i241
  %mul4.i.i = fmul fast double %sub4.i12.i.i242, %sub4.i12.i.i242
  %add.i.i = fadd fast double %mul4.i.i, %mul.i.i
  %38 = fmul fast double %add.i.i, %add.i21.i
  %39 = call fast double @llvm.sqrt.f64(double %38)
  %div.i = fdiv fast double %sub.i8.i.i245, %39
  br label %invoke.cont57

; CHECK: if.then42:
; CHECK-NOT: %o = getelementptr inbounds %struct.simplex, %struct.simplex* %t, i64 0, i32 1
; CHECK-NOT: %35 = load i32, i32* %o, align 8
; CHECK: %t.sroa.11.0.extract.trunc = trunc i40 %t.sroa.11.0 to i32

invoke.cont57:                                    ; preds = %if.then42
  %cmp59 = fcmp fast olt double %div.i, -1.000000e-10
  br i1 %cmp59, label %if.then60, label %if.end63

if.then60:                                        ; preds = %invoke.cont57
  br label %do.body.i

do.body.i:                                        ; preds = %land.rhs.i, %if.then60
  %40 = load i32, i32* %insideOutError, align 4
  %cmp.i249 = icmp sgt i32 %40, %__begin7.0
  br i1 %cmp.i249, label %land.rhs.i, label %_ZN5utils8writeMinIiEEbPT_S1_.exit

land.rhs.i:                                       ; preds = %do.body.i
  %41 = cmpxchg i32* %insideOutError, i32 %40, i32 %__begin7.0 seq_cst seq_cst
  %42 = extractvalue { i32, i1 } %41, 1
  br i1 %42, label %_ZN5utils8writeMinIiEEbPT_S1_.exit, label %do.body.i

_ZN5utils8writeMinIiEEbPT_S1_.exit:               ; preds = %do.body.i, %land.rhs.i
  %r.1.i = phi i1 [ true, %land.rhs.i ], [ false, %do.body.i ]
  br label %if.end63

lpad28:                                           ; preds = %for.inc.2.i.i
  %43 = landingpad { i8*, i32 }
          catch i8* null
  %44 = extractvalue { i8*, i32 } %43, 0
  %45 = extractvalue { i8*, i32 } %43, 1
  br label %ehcleanup103

lpad34:                                           ; No predecessors!
  %46 = landingpad { i8*, i32 }
          catch i8* null
  %47 = extractvalue { i8*, i32 } %46, 0
  %48 = extractvalue { i8*, i32 } %46, 1
  call void @llvm.lifetime.end.p0i8(i64 16, i8* %19) #3
  br label %ehcleanup103

lpad39:                                           ; No predecessors!
  %49 = landingpad { i8*, i32 }
          catch i8* null
  %50 = extractvalue { i8*, i32 } %49, 0
  %51 = extractvalue { i8*, i32 } %49, 1
  br label %ehcleanup103

lpad56:                                           ; No predecessors!
  %52 = landingpad { i8*, i32 }
          catch i8* null
  %53 = extractvalue { i8*, i32 } %52, 0
  %54 = extractvalue { i8*, i32 } %52, 1
  br label %ehcleanup103

if.end63:                                         ; preds = %_ZN5utils8writeMinIiEEbPT_S1_.exit, %invoke.cont57, %invoke.cont40
  %boundary.i253 = getelementptr inbounds %struct.simplex, %struct.simplex* %t, i64 0, i32 2
  %55 = load i8, i8* %boundary.i253, align 4
  %tobool.i254 = icmp eq i8 %55, 0
  br i1 %tobool.i254, label %lor.lhs.false.i257, label %_ZN7simplex6inCircEP6vertex.exit

; CHECK: if.end63:
; CHECK-NOT: %boundary.i253 = getelementptr inbounds %struct.simplex, %struct.simplex* %t, i64 0, i32 2
; CHECK-NOT: %55 = load i8, i8* %boundary.i253, align 4
; CHECK: %t.sroa.11.4.extract.shift = lshr i40 %t.sroa.11.0, 32
; CHECK: %t.sroa.11.4.extract.trunc = trunc i40 %t.sroa.11.4.extract.shift to i8

lor.lhs.false.i257:                               ; preds = %if.end63
  %t.i255 = getelementptr inbounds %struct.simplex, %struct.simplex* %t, i64 0, i32 0
  %56 = load %struct.tri*, %struct.tri** %t.i255, align 8
  %cmp.i256 = icmp eq %struct.tri* %56, null
  br i1 %cmp.i256, label %_ZN7simplex6inCircEP6vertex.exit, label %if.end.i270

; CHECK: lor.lhs.false.i257:
; CHECK-NOT: %t.i255 = getelementptr inbounds %struct.simplex, %struct.simplex* %t, i64 0, i32 0
; CHECK-NOT: %56 = load %struct.tri*, %struct.tri** %t.i255, align 8
; CHECK: %cmp.i256 = icmp eq %struct.tri* %t.sroa.0.0, null

if.end.i270:                                      ; preds = %lor.lhs.false.i257
  %arrayidx.i258 = getelementptr inbounds %struct.tri, %struct.tri* %56, i64 0, i32 1, i64 0
  %57 = load %struct.vertex*, %struct.vertex** %arrayidx.i258, align 8
  %agg.tmp.sroa.0.0..sroa_idx.i259 = getelementptr inbounds %struct.vertex, %struct.vertex* %57, i64 0, i32 0, i32 0
  %agg.tmp.sroa.0.0.copyload.i260 = load double, double* %agg.tmp.sroa.0.0..sroa_idx.i259, align 8
  %agg.tmp.sroa.2.0..sroa_idx18.i = getelementptr inbounds %struct.vertex, %struct.vertex* %57, i64 0, i32 0, i32 1
  %agg.tmp.sroa.2.0.copyload.i261 = load double, double* %agg.tmp.sroa.2.0..sroa_idx18.i, align 8
  %arrayidx6.i = getelementptr inbounds %struct.tri, %struct.tri* %56, i64 0, i32 1, i64 1
  %58 = load %struct.vertex*, %struct.vertex** %arrayidx6.i, align 8
  %agg.tmp3.sroa.0.0..sroa_idx.i262 = getelementptr inbounds %struct.vertex, %struct.vertex* %58, i64 0, i32 0, i32 0
  %agg.tmp3.sroa.0.0.copyload.i263 = load double, double* %agg.tmp3.sroa.0.0..sroa_idx.i262, align 8
  %agg.tmp3.sroa.2.0..sroa_idx17.i = getelementptr inbounds %struct.vertex, %struct.vertex* %58, i64 0, i32 0, i32 1
  %agg.tmp3.sroa.2.0.copyload.i264 = load double, double* %agg.tmp3.sroa.2.0..sroa_idx17.i, align 8
  %arrayidx11.i = getelementptr inbounds %struct.tri, %struct.tri* %56, i64 0, i32 1, i64 2
  %59 = load %struct.vertex*, %struct.vertex** %arrayidx11.i, align 8
  %agg.tmp8.sroa.0.0..sroa_idx.i = getelementptr inbounds %struct.vertex, %struct.vertex* %59, i64 0, i32 0, i32 0
  %agg.tmp8.sroa.0.0.copyload.i = load double, double* %agg.tmp8.sroa.0.0..sroa_idx.i, align 8
  %agg.tmp8.sroa.2.0..sroa_idx16.i = getelementptr inbounds %struct.vertex, %struct.vertex* %59, i64 0, i32 0, i32 1
  %agg.tmp8.sroa.2.0.copyload.i = load double, double* %agg.tmp8.sroa.2.0..sroa_idx16.i, align 8
  %agg.tmp13.sroa.0.0..sroa_idx.i = getelementptr inbounds %struct.vertex, %struct.vertex* %28, i64 0, i32 0, i32 0
  %agg.tmp13.sroa.0.0.copyload.i = load double, double* %agg.tmp13.sroa.0.0..sroa_idx.i, align 8
  %agg.tmp13.sroa.2.0..sroa_idx15.i = getelementptr inbounds %struct.vertex, %struct.vertex* %28, i64 0, i32 0, i32 1
  %agg.tmp13.sroa.2.0.copyload.i = load double, double* %agg.tmp13.sroa.2.0..sroa_idx15.i, align 8
  %sub.i.i.i265 = fsub fast double %agg.tmp.sroa.0.0.copyload.i260, %agg.tmp13.sroa.0.0.copyload.i
  %sub4.i.i.i266 = fsub fast double %agg.tmp.sroa.2.0.copyload.i261, %agg.tmp13.sroa.2.0.copyload.i
  %mul.i49.i.i = fmul fast double %sub.i.i.i265, %sub.i.i.i265
  %mul5.i50.i.i = fmul fast double %sub4.i.i.i266, %sub4.i.i.i266
  %add.i51.i.i = fadd fast double %mul5.i50.i.i, %mul.i49.i.i
  %sub.i44.i.i = fsub fast double %agg.tmp3.sroa.0.0.copyload.i263, %agg.tmp13.sroa.0.0.copyload.i
  %sub4.i46.i.i = fsub fast double %agg.tmp3.sroa.2.0.copyload.i264, %agg.tmp13.sroa.2.0.copyload.i
  %mul.i37.i.i = fmul fast double %sub.i44.i.i, %sub.i44.i.i
  %mul5.i38.i.i = fmul fast double %sub4.i46.i.i, %sub4.i46.i.i
  %add.i39.i.i = fadd fast double %mul5.i38.i.i, %mul.i37.i.i
  %sub.i32.i.i = fsub fast double %agg.tmp8.sroa.0.0.copyload.i, %agg.tmp13.sroa.0.0.copyload.i
  %sub4.i34.i.i = fsub fast double %agg.tmp8.sroa.2.0.copyload.i, %agg.tmp13.sroa.2.0.copyload.i
  %mul.i26.i.i = fmul fast double %sub.i32.i.i, %sub.i32.i.i
  %mul5.i.i.i = fmul fast double %sub4.i34.i.i, %sub4.i34.i.i
  %add.i27.i.i = fadd fast double %mul5.i.i.i, %mul.i26.i.i
  %mul.i20.i.i = fmul fast double %add.i39.i.i, %sub4.i.i.i266
  %mul4.i22.i.i = fmul fast double %add.i51.i.i, %sub4.i46.i.i
  %sub.i23.i.i = fsub fast double %mul.i20.i.i, %mul4.i22.i.i
  %mul6.i25.i.i = fmul fast double %add.i51.i.i, %sub.i44.i.i
  %mul9.i.i.i = fmul fast double %add.i39.i.i, %sub.i.i.i265
  %sub10.i.i.i = fsub fast double %mul6.i25.i.i, %mul9.i.i.i
  %mul13.i.i.i = fmul fast double %sub4.i46.i.i, %sub.i.i.i265
  %mul16.i.i.i = fmul fast double %sub4.i.i.i266, %sub.i44.i.i
  %sub17.i.i.i = fsub fast double %mul13.i.i.i, %mul16.i.i.i
  %mul.i.i.i267 = fmul fast double %sub.i23.i.i, %sub.i32.i.i
  %mul4.i.i.i268 = fmul fast double %sub10.i.i.i, %sub4.i34.i.i
  %mul6.i.i.i = fmul fast double %add.i27.i.i, %sub17.i.i.i
  %add.i.i.i = fadd fast double %mul.i.i.i267, %mul6.i.i.i
  %add7.i.i.i = fadd fast double %add.i.i.i, %mul4.i.i.i268
  %cmp.i.i269 = fcmp fast ogt double %add7.i.i.i, 0.000000e+00
  br label %_ZN7simplex6inCircEP6vertex.exit

_ZN7simplex6inCircEP6vertex.exit:                 ; preds = %if.end63, %lor.lhs.false.i257, %if.end.i270
  %retval.0.i271 = phi i1 [ %cmp.i.i269, %if.end.i270 ], [ false, %if.end63 ], [ false, %lor.lhs.false.i257 ]
  br label %invoke.cont64

invoke.cont64:                                    ; preds = %_ZN7simplex6inCircEP6vertex.exit
  br i1 %retval.0.i271, label %if.then66, label %if.end97

if.then66:                                        ; preds = %invoke.cont64
  %t69 = getelementptr inbounds %struct.simplex, %struct.simplex* %t, i64 0, i32 0
  %60 = load %struct.tri*, %struct.tri** %t69, align 8
  %arrayidx71 = getelementptr inbounds %struct.tri, %struct.tri* %60, i64 0, i32 1, i64 0
  %61 = load %struct.vertex*, %struct.vertex** %arrayidx71, align 8
  %agg.tmp68.sroa.0.0..sroa_idx = getelementptr inbounds %struct.vertex, %struct.vertex* %61, i64 0, i32 0, i32 0
  %agg.tmp68.sroa.0.0.copyload = load double, double* %agg.tmp68.sroa.0.0..sroa_idx, align 8
  %agg.tmp68.sroa.2.0..sroa_idx157 = getelementptr inbounds %struct.vertex, %struct.vertex* %61, i64 0, i32 0, i32 1
  %agg.tmp68.sroa.2.0.copyload = load double, double* %agg.tmp68.sroa.2.0..sroa_idx157, align 8
  %arrayidx76 = getelementptr inbounds %struct.tri, %struct.tri* %60, i64 0, i32 1, i64 1
  %62 = load %struct.vertex*, %struct.vertex** %arrayidx76, align 8
  %agg.tmp73.sroa.0.0..sroa_idx = getelementptr inbounds %struct.vertex, %struct.vertex* %62, i64 0, i32 0, i32 0
  %agg.tmp73.sroa.0.0.copyload = load double, double* %agg.tmp73.sroa.0.0..sroa_idx, align 8
  %agg.tmp73.sroa.2.0..sroa_idx156 = getelementptr inbounds %struct.vertex, %struct.vertex* %62, i64 0, i32 0, i32 1
  %agg.tmp73.sroa.2.0.copyload = load double, double* %agg.tmp73.sroa.2.0..sroa_idx156, align 8
  %arrayidx81 = getelementptr inbounds %struct.tri, %struct.tri* %60, i64 0, i32 1, i64 2
  %63 = load %struct.vertex*, %struct.vertex** %arrayidx81, align 8
  %agg.tmp78.sroa.0.0..sroa_idx = getelementptr inbounds %struct.vertex, %struct.vertex* %63, i64 0, i32 0, i32 0
  %agg.tmp78.sroa.0.0.copyload = load double, double* %agg.tmp78.sroa.0.0..sroa_idx, align 8
  %agg.tmp78.sroa.2.0..sroa_idx155 = getelementptr inbounds %struct.vertex, %struct.vertex* %63, i64 0, i32 0, i32 1
  %agg.tmp78.sroa.2.0.copyload = load double, double* %agg.tmp78.sroa.2.0..sroa_idx155, align 8
  %agg.tmp83.sroa.0.0..sroa_idx = getelementptr inbounds %struct.vertex, %struct.vertex* %28, i64 0, i32 0, i32 0
  %agg.tmp83.sroa.0.0.copyload = load double, double* %agg.tmp83.sroa.0.0..sroa_idx, align 8
  %agg.tmp83.sroa.2.0..sroa_idx154 = getelementptr inbounds %struct.vertex, %struct.vertex* %28, i64 0, i32 0, i32 1
  %agg.tmp83.sroa.2.0.copyload = load double, double* %agg.tmp83.sroa.2.0..sroa_idx154, align 8
  %sub.i.i = fsub fast double %agg.tmp68.sroa.0.0.copyload, %agg.tmp83.sroa.0.0.copyload
  %sub4.i.i = fsub fast double %agg.tmp68.sroa.2.0.copyload, %agg.tmp83.sroa.2.0.copyload
  %mul.i77.i = fmul fast double %sub.i.i, %sub.i.i
  %mul5.i78.i = fmul fast double %sub4.i.i, %sub4.i.i
  %add.i79.i = fadd fast double %mul5.i78.i, %mul.i77.i
  %sub.i72.i = fsub fast double %agg.tmp73.sroa.0.0.copyload, %agg.tmp83.sroa.0.0.copyload
  %sub4.i74.i = fsub fast double %agg.tmp73.sroa.2.0.copyload, %agg.tmp83.sroa.2.0.copyload
  %mul.i65.i = fmul fast double %sub.i72.i, %sub.i72.i
  %mul5.i66.i = fmul fast double %sub4.i74.i, %sub4.i74.i
  %add.i67.i = fadd fast double %mul5.i66.i, %mul.i65.i
  %sub.i60.i = fsub fast double %agg.tmp78.sroa.0.0.copyload, %agg.tmp83.sroa.0.0.copyload
  %sub4.i62.i = fsub fast double %agg.tmp78.sroa.2.0.copyload, %agg.tmp83.sroa.2.0.copyload
  %mul.i54.i = fmul fast double %sub.i60.i, %sub.i60.i
  %mul5.i.i = fmul fast double %sub4.i62.i, %sub4.i62.i
  %add.i55.i = fadd fast double %mul5.i.i, %mul.i54.i
  %mul.i48.i = fmul fast double %add.i67.i, %sub4.i.i
  %mul4.i50.i = fmul fast double %add.i79.i, %sub4.i74.i
  %sub.i51.i = fsub fast double %mul.i48.i, %mul4.i50.i
  %mul6.i53.i = fmul fast double %add.i79.i, %sub.i72.i
  %mul9.i.i = fmul fast double %add.i67.i, %sub.i.i
  %sub10.i.i = fsub fast double %mul6.i53.i, %mul9.i.i
  %mul13.i.i = fmul fast double %sub4.i74.i, %sub.i.i
  %mul16.i.i = fmul fast double %sub4.i.i, %sub.i72.i
  %sub17.i.i = fsub fast double %mul13.i.i, %mul16.i.i
  %mul.i38.i = fmul fast double %sub.i51.i, %sub.i60.i
  %mul4.i40.i = fmul fast double %sub10.i.i, %sub4.i62.i
  %mul6.i43.i = fmul fast double %add.i55.i, %sub17.i.i
  %add.i41.i = fadd fast double %mul.i38.i, %mul6.i43.i
  %add7.i44.i = fadd fast double %add.i41.i, %mul4.i40.i
  %mul6.i34.i = fmul fast double %add.i79.i, %add.i79.i
  %add7.i35.i = fadd fast double %mul6.i34.i, %add.i79.i
  %mul6.i26.i = fmul fast double %add.i67.i, %add.i67.i
  %add7.i27.i = fadd fast double %mul6.i26.i, %add.i67.i
  %64 = fmul fast double %add7.i35.i, %add7.i27.i
  %mul6.i.i = fmul fast double %add.i55.i, %add.i55.i
  %add7.i.i = fadd fast double %mul6.i.i, %add.i55.i
  %65 = fmul fast double %64, %add7.i.i
  %66 = call fast double @llvm.sqrt.f64(double %65)
  %div.i272 = fdiv fast double %add7.i44.i, %66
  br label %invoke.cont86

invoke.cont86:                                    ; preds = %if.then66
  %cmp88 = fcmp fast ogt double %div.i272, 1.000000e-10
  br i1 %cmp88, label %if.then89, label %if.end97

if.then89:                                        ; preds = %invoke.cont86
  br label %do.body.i282

do.body.i282:                                     ; preds = %land.rhs.i283, %if.then89
  %67 = load i32, i32* %inCircleError, align 4
  %cmp.i281 = icmp sgt i32 %67, %__begin7.0
  br i1 %cmp.i281, label %land.rhs.i283, label %_ZN5utils8writeMinIiEEbPT_S1_.exit285

land.rhs.i283:                                    ; preds = %do.body.i282
  %68 = cmpxchg i32* %inCircleError, i32 %67, i32 %__begin7.0 seq_cst seq_cst
  %69 = extractvalue { i32, i1 } %68, 1
  br i1 %69, label %_ZN5utils8writeMinIiEEbPT_S1_.exit285, label %do.body.i282

_ZN5utils8writeMinIiEEbPT_S1_.exit285:            ; preds = %do.body.i282, %land.rhs.i283
  %r.1.i284 = phi i1 [ true, %land.rhs.i283 ], [ false, %do.body.i282 ]
  br label %if.end97

lpad85:                                           ; No predecessors!
  %70 = landingpad { i8*, i32 }
          catch i8* null
  %71 = extractvalue { i8*, i32 } %70, 0
  %72 = extractvalue { i8*, i32 } %70, 1
  br label %ehcleanup103

if.else:                                          ; preds = %invoke.cont29
  %arrayidx95 = getelementptr inbounds i32, i32* %0, i64 %4
  %73 = load i32, i32* %arrayidx95, align 4
  %inc96 = add nsw i32 %73, 1
  store i32 %inc96, i32* %arrayidx95, align 4
  br label %if.end97

if.end97:                                         ; preds = %_ZN5utils8writeMinIiEEbPT_S1_.exit285, %invoke.cont64, %invoke.cont86, %if.else
  %t.i286 = getelementptr inbounds %struct.simplex, %struct.simplex* %t, i64 0, i32 0
  %74 = load %struct.tri*, %struct.tri** %t.i286, align 8
  %o.i287 = getelementptr inbounds %struct.simplex, %struct.simplex* %t, i64 0, i32 1
  %75 = load i32, i32* %o.i287, align 8
  %cmp.i.i288 = icmp sgt i32 %75, 1
  %cond.i.v.i289 = select i1 %cmp.i.i288, i32 -2, i32 1
  %cond.i.i290 = add i32 %cond.i.v.i289, %75
  %retval.sroa.2.8.insert.ext.i291 = zext i32 %cond.i.i290 to i64
  %.fca.0.insert.i292 = insertvalue { %struct.tri*, i64 } undef, %struct.tri* %74, 0
  %.fca.1.insert.i293 = insertvalue { %struct.tri*, i64 } %.fca.0.insert.i292, i64 %retval.sroa.2.8.insert.ext.i291, 1
  br label %invoke.cont100

; CHECK: if.end97:
; CHECK-NOT: %o.i287 = getelementptr inbounds %struct.simplex, %struct.simplex* %t, i64 0, i32 1
; CHECK-NOT: %75 = load i32, i32* %o.i287, align 8
; CHECK: %t.sroa.11.0.extract.trunc29 = trunc i40 %t.sroa.11.0 to i32
; CHECK: %cond.i.i290 = add i32 %cond.i.v.i289, %t.sroa.11.0.extract.trunc29

invoke.cont100:                                   ; preds = %if.end97
  %76 = extractvalue { %struct.tri*, i64 } %.fca.1.insert.i293, 0
  %77 = extractvalue { %struct.tri*, i64 } %.fca.1.insert.i293, 1
  %ref.tmp98.sroa.0.0..sroa_idx = getelementptr inbounds %struct.simplex, %struct.simplex* %t, i64 0, i32 0
  store %struct.tri* %76, %struct.tri** %ref.tmp98.sroa.0.0..sroa_idx, align 8
  %ref.tmp98.sroa.5.0..sroa_idx152 = getelementptr inbounds %struct.simplex, %struct.simplex* %t, i64 0, i32 1
  %ref.tmp98.sroa.5.0..sroa_cast153 = bitcast i32* %ref.tmp98.sroa.5.0..sroa_idx152 to i40*
  %ref.tmp98.sroa.5.0.extract.trunc = trunc i64 %77 to i40
  store i40 %ref.tmp98.sroa.5.0.extract.trunc, i40* %ref.tmp98.sroa.5.0..sroa_cast153, align 8
  call void @llvm.lifetime.end.p0i8(i64 16, i8* %7) #3
  %inc104 = add nuw nsw i32 %j25.0, 1
  br label %for.cond

lpad99:                                           ; No predecessors!
  %78 = landingpad { i8*, i32 }
          catch i8* null
  %79 = extractvalue { i8*, i32 } %78, 0
  %80 = extractvalue { i8*, i32 } %78, 1
  br label %ehcleanup103

ehcleanup103:                                     ; preds = %lpad34, %lpad39, %lpad56, %lpad85, %lpad99, %lpad28
  %ehselector.slot.1 = phi i32 [ %80, %lpad99 ], [ %45, %lpad28 ], [ %72, %lpad85 ], [ %51, %lpad39 ], [ %54, %lpad56 ], [ %48, %lpad34 ]
  %exn.slot.1 = phi i8* [ %79, %lpad99 ], [ %44, %lpad28 ], [ %71, %lpad85 ], [ %50, %lpad39 ], [ %53, %lpad56 ], [ %47, %lpad34 ]
  call void @llvm.lifetime.end.p0i8(i64 16, i8* %7) #3
  call void @llvm.lifetime.end.p0i8(i64 16, i8* %6) #3
  %lpad.val = insertvalue { i8*, i32 } undef, i8* %exn.slot.1, 0
  %lpad.val115 = insertvalue { i8*, i32 } %lpad.val, i32 %ehselector.slot.1, 1
  invoke void @llvm.detached.rethrow.sl_p0i8i32s(token %syncreg5, { i8*, i32 } %lpad.val115)
          to label %det.rethrow.unreachable unwind label %lpad112

det.rethrow.unreachable:                          ; preds = %ehcleanup103
  unreachable

pfor.preattach108:                                ; preds = %pfor.body20, %for.cond.cleanup
  reattach within %syncreg5, label %pfor.inc110

; CHECK: pfor.preattach108:
; CHECK: %t.sroa.11.1 = phi i40 [ undef, %pfor.body20 ], [ %t.sroa.11.0, %for.cond.cleanup ]

pfor.inc110:                                      ; preds = %pfor.detach16, %pfor.preattach108
  %inc111 = add nuw nsw i32 %__begin7.0, 1
  br label %pfor.cond13

lpad112:                                          ; preds = %pfor.detach16, %ehcleanup103
  %81 = landingpad { i8*, i32 }
          cleanup
  sync within %syncreg5, label %sync.continue118

sync.continue116:                                 ; preds = %pfor.cond.cleanup15
  %call.i306 = call i32 @_ZN8sequence6reduceIiiN5utils4addFIiEENS_4getAIiiEEEET_T0_S7_T1_T2_(i32 0, i32 %n, i32* %0)
  call void @free(i8* %call) #3
  %82 = load i32, i32* %insideOutError, align 4
  %cmp124 = icmp slt i32 %82, %n
  br i1 %cmp124, label %if.then125, label %if.end130

sync.continue118:                                 ; preds = %lpad112
  call void @llvm.lifetime.end.p0i8(i64 4, i8* %3) #3
  call void @llvm.lifetime.end.p0i8(i64 4, i8* %2) #3
  resume { i8*, i32 } %81

if.then125:                                       ; preds = %sync.continue116
  %call.i.i307 = call i64 @strlen(i8* nonnull getelementptr inbounds ([53 x i8], [53 x i8]* @.str, i64 0, i64 0)) #3
  %call1.i308 = call dereferenceable(272) %"class.std::basic_ostream"* @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(%"class.std::basic_ostream"* nonnull dereferenceable(272) @_ZSt4cout, i8* nonnull getelementptr inbounds ([53 x i8], [53 x i8]* @.str, i64 0, i64 0), i64 %call.i.i307)
  %83 = load i32, i32* %inCircleError, align 4
  %call127 = call dereferenceable(272) %"class.std::basic_ostream"* @_ZNSolsEi(%"class.std::basic_ostream"* nonnull @_ZSt4cout, i32 %83)
  %84 = bitcast %"class.std::basic_ostream"* %call127 to i8**
  %vtable.i274 = load i8*, i8** %84, align 8
  %vbase.offset.ptr.i275 = getelementptr i8, i8* %vtable.i274, i64 -24
  %85 = bitcast i8* %vbase.offset.ptr.i275 to i64*
  %vbase.offset.i276 = load i64, i64* %85, align 8
  %86 = bitcast %"class.std::basic_ostream"* %call127 to i8*
  %add.ptr.i277 = getelementptr inbounds i8, i8* %86, i64 %vbase.offset.i276
  %87 = bitcast i8* %add.ptr.i277 to %"class.std::basic_ios"*
  %_M_ctype.i294 = getelementptr inbounds %"class.std::basic_ios", %"class.std::basic_ios"* %87, i64 0, i32 5
  %88 = load %"class.std::ctype"*, %"class.std::ctype"** %_M_ctype.i294, align 8
  %tobool.i311 = icmp eq %"class.std::ctype"* %88, null
  br i1 %tobool.i311, label %if.then.i312, label %_ZSt13__check_facetISt5ctypeIcEERKT_PS3_.exit314

if.then.i312:                                     ; preds = %if.then125
  call void @_ZSt16__throw_bad_castv() #17
  unreachable

_ZSt13__check_facetISt5ctypeIcEERKT_PS3_.exit314: ; preds = %if.then125
  %_M_widen_ok.i296 = getelementptr inbounds %"class.std::ctype", %"class.std::ctype"* %88, i64 0, i32 8
  %89 = load i8, i8* %_M_widen_ok.i296, align 8
  %tobool.i297 = icmp eq i8 %89, 0
  br i1 %tobool.i297, label %if.end.i303, label %if.then.i299

if.then.i299:                                     ; preds = %_ZSt13__check_facetISt5ctypeIcEERKT_PS3_.exit314
  %arrayidx.i298 = getelementptr inbounds %"class.std::ctype", %"class.std::ctype"* %88, i64 0, i32 9, i64 10
  %90 = load i8, i8* %arrayidx.i298, align 1
  br label %_ZNKSt5ctypeIcE5widenEc.exit305

if.end.i303:                                      ; preds = %_ZSt13__check_facetISt5ctypeIcEERKT_PS3_.exit314
  call void @_ZNKSt5ctypeIcE13_M_widen_initEv(%"class.std::ctype"* %88)
  %91 = bitcast %"class.std::ctype"* %88 to i8 (%"class.std::ctype"*, i8)***
  %vtable.i300 = load i8 (%"class.std::ctype"*, i8)**, i8 (%"class.std::ctype"*, i8)*** %91, align 8
  %vfn.i301 = getelementptr inbounds i8 (%"class.std::ctype"*, i8)*, i8 (%"class.std::ctype"*, i8)** %vtable.i300, i64 6
  %92 = load i8 (%"class.std::ctype"*, i8)*, i8 (%"class.std::ctype"*, i8)** %vfn.i301, align 8
  %call.i302 = call signext i8 %92(%"class.std::ctype"* %88, i8 signext 10)
  br label %_ZNKSt5ctypeIcE5widenEc.exit305

_ZNKSt5ctypeIcE5widenEc.exit305:                  ; preds = %if.then.i299, %if.end.i303
  %retval.0.i304 = phi i8 [ %90, %if.then.i299 ], [ %call.i302, %if.end.i303 ]
  %call1.i279 = call dereferenceable(272) %"class.std::basic_ostream"* @_ZNSo3putEc(%"class.std::basic_ostream"* nonnull %call127, i8 signext %retval.0.i304)
  %call.i280 = call dereferenceable(272) %"class.std::basic_ostream"* @_ZNSo5flushEv(%"class.std::basic_ostream"* nonnull %call1.i279)
  br label %cleanup

if.end130:                                        ; preds = %sync.continue116
  %93 = load i32, i32* %inCircleError, align 4
  %cmp131 = icmp slt i32 %93, %n
  br i1 %cmp131, label %if.then132, label %cleanup

if.then132:                                       ; preds = %if.end130
  %call.i.i = call i64 @strlen(i8* nonnull getelementptr inbounds ([33 x i8], [33 x i8]* @.str.4, i64 0, i64 0)) #3
  %call1.i250 = call dereferenceable(272) %"class.std::basic_ostream"* @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(%"class.std::basic_ostream"* nonnull dereferenceable(272) @_ZSt4cout, i8* nonnull getelementptr inbounds ([33 x i8], [33 x i8]* @.str.4, i64 0, i64 0), i64 %call.i.i)
  %94 = load i32, i32* %inCircleError, align 4
  %call134 = call dereferenceable(272) %"class.std::basic_ostream"* @_ZNSolsEi(%"class.std::basic_ostream"* nonnull @_ZSt4cout, i32 %94)
  %95 = bitcast %"class.std::basic_ostream"* %call134 to i8**
  %vtable.i = load i8*, i8** %95, align 8
  %vbase.offset.ptr.i = getelementptr i8, i8* %vtable.i, i64 -24
  %96 = bitcast i8* %vbase.offset.ptr.i to i64*
  %vbase.offset.i = load i64, i64* %96, align 8
  %97 = bitcast %"class.std::basic_ostream"* %call134 to i8*
  %add.ptr.i = getelementptr inbounds i8, i8* %97, i64 %vbase.offset.i
  %98 = bitcast i8* %add.ptr.i to %"class.std::basic_ios"*
  %_M_ctype.i = getelementptr inbounds %"class.std::basic_ios", %"class.std::basic_ios"* %98, i64 0, i32 5
  %99 = load %"class.std::ctype"*, %"class.std::ctype"** %_M_ctype.i, align 8
  %tobool.i246 = icmp eq %"class.std::ctype"* %99, null
  br i1 %tobool.i246, label %if.then.i247, label %_ZSt13__check_facetISt5ctypeIcEERKT_PS3_.exit

if.then.i247:                                     ; preds = %if.then132
  call void @_ZSt16__throw_bad_castv() #17
  unreachable

_ZSt13__check_facetISt5ctypeIcEERKT_PS3_.exit:    ; preds = %if.then132
  %_M_widen_ok.i = getelementptr inbounds %"class.std::ctype", %"class.std::ctype"* %99, i64 0, i32 8
  %100 = load i8, i8* %_M_widen_ok.i, align 8
  %tobool.i222 = icmp eq i8 %100, 0
  br i1 %tobool.i222, label %if.end.i, label %if.then.i224

if.then.i224:                                     ; preds = %_ZSt13__check_facetISt5ctypeIcEERKT_PS3_.exit
  %arrayidx.i223 = getelementptr inbounds %"class.std::ctype", %"class.std::ctype"* %99, i64 0, i32 9, i64 10
  %101 = load i8, i8* %arrayidx.i223, align 1
  br label %_ZNKSt5ctypeIcE5widenEc.exit

if.end.i:                                         ; preds = %_ZSt13__check_facetISt5ctypeIcEERKT_PS3_.exit
  call void @_ZNKSt5ctypeIcE13_M_widen_initEv(%"class.std::ctype"* %99)
  %102 = bitcast %"class.std::ctype"* %99 to i8 (%"class.std::ctype"*, i8)***
  %vtable.i225 = load i8 (%"class.std::ctype"*, i8)**, i8 (%"class.std::ctype"*, i8)*** %102, align 8
  %vfn.i = getelementptr inbounds i8 (%"class.std::ctype"*, i8)*, i8 (%"class.std::ctype"*, i8)** %vtable.i225, i64 6
  %103 = load i8 (%"class.std::ctype"*, i8)*, i8 (%"class.std::ctype"*, i8)** %vfn.i, align 8
  %call.i226 = call signext i8 %103(%"class.std::ctype"* %99, i8 signext 10)
  br label %_ZNKSt5ctypeIcE5widenEc.exit

_ZNKSt5ctypeIcE5widenEc.exit:                     ; preds = %if.then.i224, %if.end.i
  %retval.0.i = phi i8 [ %101, %if.then.i224 ], [ %call.i226, %if.end.i ]
  %call1.i = call dereferenceable(272) %"class.std::basic_ostream"* @_ZNSo3putEc(%"class.std::basic_ostream"* nonnull %call134, i8 signext %retval.0.i)
  %call.i = call dereferenceable(272) %"class.std::basic_ostream"* @_ZNSo5flushEv(%"class.std::basic_ostream"* nonnull %call1.i)
  br label %cleanup

cleanup:                                          ; preds = %if.end130, %_ZNKSt5ctypeIcE5widenEc.exit, %_ZNKSt5ctypeIcE5widenEc.exit305
  %retval.0 = phi i1 [ true, %_ZNKSt5ctypeIcE5widenEc.exit305 ], [ true, %_ZNKSt5ctypeIcE5widenEc.exit ], [ false, %if.end130 ]
  call void @llvm.lifetime.end.p0i8(i64 4, i8* %3) #3
  call void @llvm.lifetime.end.p0i8(i64 4, i8* %2) #3
  ret i1 %retval.0
}

; Function Attrs: argmemonly nounwind
declare void @llvm.lifetime.start.p0i8(i64, i8* nocapture) #4

; Function Attrs: nounwind
declare noalias i8* @malloc(i64) local_unnamed_addr #1

; Function Attrs: nounwind
declare void @free(i8* nocapture) local_unnamed_addr #2

; Function Attrs: argmemonly nounwind readonly
declare i64 @strlen(i8* nocapture) local_unnamed_addr #14

; Function Attrs: argmemonly nounwind
declare token @llvm.syncregion.start() #4

declare i32 @__gxx_personality_v0(...)

; Function Attrs: argmemonly nounwind
declare void @llvm.lifetime.end.p0i8(i64, i8* nocapture) #4

; Function Attrs: argmemonly
declare void @llvm.detached.rethrow.sl_p0i8i32s(token, { i8*, i32 }) #5

; Function Attrs: inlinehint uwtable
declare dereferenceable(272) %"class.std::basic_ostream"* @_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc(%"class.std::basic_ostream"* dereferenceable(272), i8*) local_unnamed_addr #7

declare dereferenceable(272) %"class.std::basic_ostream"* @_ZNSolsEi(%"class.std::basic_ostream"*, i32) local_unnamed_addr #0

; Function Attrs: noreturn nounwind
declare void @abort() local_unnamed_addr #8

; Function Attrs: nounwind readnone speculatable
declare double @llvm.sqrt.f64(double) #9

declare dereferenceable(272) %"class.std::basic_ostream"* @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(%"class.std::basic_ostream"* dereferenceable(272), i8*, i64) local_unnamed_addr #0

declare dereferenceable(272) %"class.std::basic_ostream"* @_ZNSo3putEc(%"class.std::basic_ostream"*, i8 signext) local_unnamed_addr #0

declare dereferenceable(272) %"class.std::basic_ostream"* @_ZNSo5flushEv(%"class.std::basic_ostream"*) local_unnamed_addr #0

; Function Attrs: noreturn
declare void @_ZSt16__throw_bad_castv() local_unnamed_addr #10

declare void @_ZNKSt5ctypeIcE13_M_widen_initEv(%"class.std::ctype"*) local_unnamed_addr #0

; Function Attrs: uwtable
declare i32 @_ZN8sequence6reduceIiiN5utils4addFIiEENS_4getAIiiEEEET_T0_S7_T1_T2_(i32 %s, i32 %e, i32* %g.coerce) local_unnamed_addr #3

attributes #0 = { "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="haswell" "target-features"="+aes,+avx,+avx2,+bmi,+bmi2,+cmov,+cx16,+f16c,+fma,+fsgsbase,+fxsr,+lzcnt,+mmx,+movbe,+pclmul,+popcnt,+rdrnd,+sse,+sse2,+sse3,+sse4.1,+sse4.2,+ssse3,+x87,+xsave,+xsaveopt,-adx,-avx512bitalg,-avx512bw,-avx512cd,-avx512dq,-avx512er,-avx512f,-avx512ifma,-avx512pf,-avx512vbmi,-avx512vbmi2,-avx512vl,-avx512vnni,-avx512vpopcntdq,-clflushopt,-clwb,-clzero,-fma4,-gfni,-ibt,-lwp,-mwaitx,-pku,-prefetchwt1,-prfchw,-rdseed,-rtm,-sgx,-sha,-shstk,-sse4a,-tbm,-vaes,-vpclmulqdq,-xop,-xsavec,-xsaves" "unsafe-fp-math"="true" "use-soft-float"="false" }
attributes #1 = { nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="haswell" "target-features"="+aes,+avx,+avx2,+bmi,+bmi2,+cmov,+cx16,+f16c,+fma,+fsgsbase,+fxsr,+lzcnt,+mmx,+movbe,+pclmul,+popcnt,+rdrnd,+sse,+sse2,+sse3,+sse4.1,+sse4.2,+ssse3,+x87,+xsave,+xsaveopt,-adx,-avx512bitalg,-avx512bw,-avx512cd,-avx512dq,-avx512er,-avx512f,-avx512ifma,-avx512pf,-avx512vbmi,-avx512vbmi2,-avx512vl,-avx512vnni,-avx512vpopcntdq,-clflushopt,-clwb,-clzero,-fma4,-gfni,-ibt,-lwp,-mwaitx,-pku,-prefetchwt1,-prfchw,-rdseed,-rtm,-sgx,-sha,-shstk,-sse4a,-tbm,-vaes,-vpclmulqdq,-xop,-xsavec,-xsaves" "unsafe-fp-math"="true" "use-soft-float"="false" }
attributes #2 = { nounwind }
attributes #3 = { uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="true" "no-jump-tables"="false" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="haswell" "target-features"="+aes,+avx,+avx2,+bmi,+bmi2,+cmov,+cx16,+f16c,+fma,+fsgsbase,+fxsr,+lzcnt,+mmx,+movbe,+pclmul,+popcnt,+rdrnd,+sse,+sse2,+sse3,+sse4.1,+sse4.2,+ssse3,+x87,+xsave,+xsaveopt,-adx,-avx512bitalg,-avx512bw,-avx512cd,-avx512dq,-avx512er,-avx512f,-avx512ifma,-avx512pf,-avx512vbmi,-avx512vbmi2,-avx512vl,-avx512vnni,-avx512vpopcntdq,-clflushopt,-clwb,-clzero,-fma4,-gfni,-ibt,-lwp,-mwaitx,-pku,-prefetchwt1,-prfchw,-rdseed,-rtm,-sgx,-sha,-shstk,-sse4a,-tbm,-vaes,-vpclmulqdq,-xop,-xsavec,-xsaves" "unsafe-fp-math"="true" "use-soft-float"="false" }
attributes #4 = { argmemonly nounwind }
attributes #5 = { argmemonly }
attributes #6 = { norecurse uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="true" "no-jump-tables"="false" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="haswell" "target-features"="+aes,+avx,+avx2,+bmi,+bmi2,+cmov,+cx16,+f16c,+fma,+fsgsbase,+fxsr,+lzcnt,+mmx,+movbe,+pclmul,+popcnt,+rdrnd,+sse,+sse2,+sse3,+sse4.1,+sse4.2,+ssse3,+x87,+xsave,+xsaveopt,-adx,-avx512bitalg,-avx512bw,-avx512cd,-avx512dq,-avx512er,-avx512f,-avx512ifma,-avx512pf,-avx512vbmi,-avx512vbmi2,-avx512vl,-avx512vnni,-avx512vpopcntdq,-clflushopt,-clwb,-clzero,-fma4,-gfni,-ibt,-lwp,-mwaitx,-pku,-prefetchwt1,-prfchw,-rdseed,-rtm,-sgx,-sha,-shstk,-sse4a,-tbm,-vaes,-vpclmulqdq,-xop,-xsavec,-xsaves" "unsafe-fp-math"="true" "use-soft-float"="false" }
attributes #7 = { inlinehint uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="true" "no-jump-tables"="false" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="haswell" "target-features"="+aes,+avx,+avx2,+bmi,+bmi2,+cmov,+cx16,+f16c,+fma,+fsgsbase,+fxsr,+lzcnt,+mmx,+movbe,+pclmul,+popcnt,+rdrnd,+sse,+sse2,+sse3,+sse4.1,+sse4.2,+ssse3,+x87,+xsave,+xsaveopt,-adx,-avx512bitalg,-avx512bw,-avx512cd,-avx512dq,-avx512er,-avx512f,-avx512ifma,-avx512pf,-avx512vbmi,-avx512vbmi2,-avx512vl,-avx512vnni,-avx512vpopcntdq,-clflushopt,-clwb,-clzero,-fma4,-gfni,-ibt,-lwp,-mwaitx,-pku,-prefetchwt1,-prfchw,-rdseed,-rtm,-sgx,-sha,-shstk,-sse4a,-tbm,-vaes,-vpclmulqdq,-xop,-xsavec,-xsaves" "unsafe-fp-math"="true" "use-soft-float"="false" }
attributes #8 = { noreturn nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="haswell" "target-features"="+aes,+avx,+avx2,+bmi,+bmi2,+cmov,+cx16,+f16c,+fma,+fsgsbase,+fxsr,+lzcnt,+mmx,+movbe,+pclmul,+popcnt,+rdrnd,+sse,+sse2,+sse3,+sse4.1,+sse4.2,+ssse3,+x87,+xsave,+xsaveopt,-adx,-avx512bitalg,-avx512bw,-avx512cd,-avx512dq,-avx512er,-avx512f,-avx512ifma,-avx512pf,-avx512vbmi,-avx512vbmi2,-avx512vl,-avx512vnni,-avx512vpopcntdq,-clflushopt,-clwb,-clzero,-fma4,-gfni,-ibt,-lwp,-mwaitx,-pku,-prefetchwt1,-prfchw,-rdseed,-rtm,-sgx,-sha,-shstk,-sse4a,-tbm,-vaes,-vpclmulqdq,-xop,-xsavec,-xsaves" "unsafe-fp-math"="true" "use-soft-float"="false" }
attributes #9 = { nounwind readnone speculatable }
attributes #10 = { noreturn "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="haswell" "target-features"="+aes,+avx,+avx2,+bmi,+bmi2,+cmov,+cx16,+f16c,+fma,+fsgsbase,+fxsr,+lzcnt,+mmx,+movbe,+pclmul,+popcnt,+rdrnd,+sse,+sse2,+sse3,+sse4.1,+sse4.2,+ssse3,+x87,+xsave,+xsaveopt,-adx,-avx512bitalg,-avx512bw,-avx512cd,-avx512dq,-avx512er,-avx512f,-avx512ifma,-avx512pf,-avx512vbmi,-avx512vbmi2,-avx512vl,-avx512vnni,-avx512vpopcntdq,-clflushopt,-clwb,-clzero,-fma4,-gfni,-ibt,-lwp,-mwaitx,-pku,-prefetchwt1,-prfchw,-rdseed,-rtm,-sgx,-sha,-shstk,-sse4a,-tbm,-vaes,-vpclmulqdq,-xop,-xsavec,-xsaves" "unsafe-fp-math"="true" "use-soft-float"="false" }
