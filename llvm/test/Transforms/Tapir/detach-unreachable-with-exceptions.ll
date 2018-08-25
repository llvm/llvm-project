; RUN: opt < %s -passes=simplify-cfg -S | FileCheck %s --check-prefix=SC
; RUN: opt < %s -passes=simplify-cfg,task-simplify -S | FileCheck %s --check-prefix=TS

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%"class.std::basic_ostream" = type { i32 (...)**, %"class.std::basic_ios" }
%"class.std::basic_ios" = type { %"class.std::ios_base", %"class.std::basic_ostream"*, i8, i8, %"class.std::basic_streambuf"*, %"class.std::ctype"*, %"class.std::num_put"*, %"class.std::num_get"* }
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
%class._point2d = type { double, double }
%struct.vertex = type { i32, %class._point2d, [1 x %struct.vertex*] }
%class.gTreeNode = type { %class._point2d, double, %struct.nData, i32, [8 x %class.gTreeNode*], %struct.vertex**, %class.gTreeNode* }
%struct.nData = type { i32 }
%"struct.kNearestNeighbor<vertex<_point2d<double>, 1>, 1>::kNN" = type { %struct.vertex*, [1 x %struct.vertex*], [1 x double], i32, i32 }

$_Z3ANNILi1E6vertexI8_point2dIdELi1EEEvPPT0_ii = comdat any

@_ZSt4cout = external global %"class.std::basic_ostream", align 8
@.str.26 = private unnamed_addr constant [19 x i8] c"k too large in kNN\00", align 1

; Function Attrs: uwtable
define linkonce_odr void @_Z3ANNILi1E6vertexI8_point2dIdELi1EEEvPPT0_ii(%struct.vertex** %v, i32 %n, i32 %k) local_unnamed_addr #5 comdat personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*) {
entry:
  %syncreg = tail call token @llvm.syncregion.start()
  %call.i = call %class.gTreeNode* @_ZN9gTreeNodeI8_point2dIdE7_vect2dIdE6vertexIS1_Li1EE5nDataIS5_EE5gTreeEPPS5_i(%struct.vertex** %v, i32 %n)
  %count.i.i = getelementptr inbounds %class.gTreeNode, %class.gTreeNode* %call.i, i64 0, i32 3
  %0 = load i32, i32* %count.i.i, align 4, !tbaa !95
  %1 = sext i32 %0 to i64
  %2 = call { i64, i1 } @llvm.umul.with.overflow.i64(i64 %1, i64 8)
  %3 = extractvalue { i64, i1 } %2, 1
  %4 = extractvalue { i64, i1 } %2, 0
  %5 = select i1 %3, i64 -1, i64 %4
  %call.i.i = call i8* @_Znam(i64 %5) #19
  %6 = bitcast i8* %call.i.i to %struct.vertex**
  call void @_ZN9gTreeNodeI8_point2dIdE7_vect2dIdE6vertexIS1_Li1EE5nDataIS5_EE10applyIndexINS8_10flatten_FAEEEviT_(%class.gTreeNode* %call.i, i32 0, %struct.vertex** %6)
  %cmp31 = icmp sgt i32 %n, 0
  br i1 %cmp31, label %pfor.detach.lr.ph, label %pfor.cond.cleanup

pfor.detach.lr.ph:                                ; preds = %entry
  %wide.trip.count = zext i32 %n to i64
  %cmp.i = icmp sgt i32 %k, 1
  %cmp24.i = icmp sgt i32 %k, 0
  %7 = zext i32 %k to i64
  %8 = shl nuw nsw i64 %7, 3
  %wide.trip.count.i = zext i32 %k to i64
  br i1 %cmp.i, label %pfor.detach.lr.ph.split.us, label %pfor.detach.lr.ph.pfor.detach.lr.ph.split_crit_edge

pfor.detach.lr.ph.pfor.detach.lr.ph.split_crit_edge: ; preds = %pfor.detach.lr.ph
  br label %pfor.detach.lr.ph.split

pfor.detach.lr.ph.split.us:                       ; preds = %pfor.detach.lr.ph
  br label %pfor.detach.us

pfor.detach.us:                                   ; preds = %pfor.inc.us, %pfor.detach.lr.ph.split.us
  %indvars.iv.us = phi i64 [ 0, %pfor.detach.lr.ph.split.us ], [ %indvars.iv.next.us, %pfor.inc.us ]
  detach within %syncreg, label %pfor.body.us, label %pfor.inc.us unwind label %lpad5.loopexit.us-lcssa.us

; SC: pfor.detach.us:
; SC: detach within %syncreg, label %pfor.body.us, label %pfor.inc.us unwind label %lpad5.loopexit.us-lcssa.us

; TS: pfor.detach.us:
; TS: br label %pfor.body.us

pfor.body.us:                                     ; preds = %pfor.detach.us
  %nn.i.us = alloca %"struct.kNearestNeighbor<vertex<_point2d<double>, 1>, 1>::kNN", align 8
  %arrayidx.us = getelementptr inbounds %struct.vertex*, %struct.vertex** %6, i64 %indvars.iv.us
  %9 = load %struct.vertex*, %struct.vertex** %arrayidx.us, align 8, !tbaa !20
  %arraydecay.us = getelementptr inbounds %struct.vertex, %struct.vertex* %9, i64 0, i32 2, i64 0
  %10 = bitcast %"struct.kNearestNeighbor<vertex<_point2d<double>, 1>, 1>::kNN"* %nn.i.us to i8*
  call void @llvm.lifetime.start.p0i8(i64 32, i8* nonnull %10) #2
  br i1 true, label %if.then.i.us-lcssa.us, label %if.end.i.us

; SC: pfor.body.us:
; SC: br label %if.then.i

if.end.i.us:                                      ; preds = %pfor.body.us
  %k.i.us = getelementptr inbounds %"struct.kNearestNeighbor<vertex<_point2d<double>, 1>, 1>::kNN", %"struct.kNearestNeighbor<vertex<_point2d<double>, 1>, 1>::kNN"* %nn.i.us, i64 0, i32 4
  store i32 %k, i32* %k.i.us, align 4, !tbaa !98
  %quads.i.us = getelementptr inbounds %"struct.kNearestNeighbor<vertex<_point2d<double>, 1>, 1>::kNN", %"struct.kNearestNeighbor<vertex<_point2d<double>, 1>, 1>::kNN"* %nn.i.us, i64 0, i32 3
  store i32 4, i32* %quads.i.us, align 8, !tbaa !100
  %ps.i.us = getelementptr inbounds %"struct.kNearestNeighbor<vertex<_point2d<double>, 1>, 1>::kNN", %"struct.kNearestNeighbor<vertex<_point2d<double>, 1>, 1>::kNN"* %nn.i.us, i64 0, i32 0
  store %struct.vertex* %9, %struct.vertex** %ps.i.us, align 8, !tbaa !101
  %11 = load i32, i32* %k.i.us, align 4, !tbaa !98
  %cmp515.i.us = icmp sgt i32 %11, 0
  br i1 %cmp515.i.us, label %for.body.preheader.i.us, label %.noexc.us

for.body.preheader.i.us:                          ; preds = %if.end.i.us
  br label %for.body.i.us

for.body.i.us:                                    ; preds = %for.body.i.us, %for.body.preheader.i.us
  %indvars.iv.i7.us = phi i64 [ %indvars.iv.next.i9.us, %for.body.i.us ], [ 0, %for.body.preheader.i.us ]
  %arrayidx.i.us = getelementptr inbounds %"struct.kNearestNeighbor<vertex<_point2d<double>, 1>, 1>::kNN", %"struct.kNearestNeighbor<vertex<_point2d<double>, 1>, 1>::kNN"* %nn.i.us, i64 0, i32 1, i64 %indvars.iv.i7.us
  store %struct.vertex* null, %struct.vertex** %arrayidx.i.us, align 8, !tbaa !20
  %arrayidx8.i8.us = getelementptr inbounds %"struct.kNearestNeighbor<vertex<_point2d<double>, 1>, 1>::kNN", %"struct.kNearestNeighbor<vertex<_point2d<double>, 1>, 1>::kNN"* %nn.i.us, i64 0, i32 2, i64 %indvars.iv.i7.us
  store double 0x7FEFFFFFFFFFFFFF, double* %arrayidx8.i8.us, align 8, !tbaa !82
  %indvars.iv.next.i9.us = add nuw nsw i64 %indvars.iv.i7.us, 1
  %12 = load i32, i32* %k.i.us, align 4, !tbaa !98
  %13 = sext i32 %12 to i64
  %cmp5.i.us = icmp slt i64 %indvars.iv.next.i9.us, %13
  br i1 %cmp5.i.us, label %for.body.i.us, label %.noexc.loopexit.us

.noexc.us:                                        ; preds = %.noexc.loopexit.us, %if.end.i.us
  invoke void @_ZN16kNearestNeighborI6vertexI8_point2dIdELi1EELi1EE3kNN10nearestNghEP9gTreeNodeIS2_7_vect2dIdES3_5nDataIS3_EE(%"struct.kNearestNeighbor<vertex<_point2d<double>, 1>, 1>::kNN"* nonnull %nn.i.us, %class.gTreeNode* %call.i)
          to label %.noexc3.us unwind label %lpad.loopexit20.us-lcssa.us

.noexc3.us:                                       ; preds = %.noexc.us
  br i1 %cmp24.i, label %for.body6.lr.ph.i.us, label %_ZN16kNearestNeighborI6vertexI8_point2dIdELi1EELi1EE8kNearestEPS3_PS5_i.exit.us

for.body6.lr.ph.i.us:                             ; preds = %.noexc3.us
  %result30.i.us = bitcast %struct.vertex** %arraydecay.us to i8*
  call void @llvm.memset.p0i8.i64(i8* %result30.i.us, i8 0, i64 %8, i32 8, i1 false)
  %k.i.i.us = getelementptr inbounds %"struct.kNearestNeighbor<vertex<_point2d<double>, 1>, 1>::kNN", %"struct.kNearestNeighbor<vertex<_point2d<double>, 1>, 1>::kNN"* %nn.i.us, i64 0, i32 4
  br label %for.body6.i.us

for.body6.i.us:                                   ; preds = %for.body6.i.us, %for.body6.lr.ph.i.us
  %indvars.iv.i.us = phi i64 [ 0, %for.body6.lr.ph.i.us ], [ %indvars.iv.next.i.us, %for.body6.i.us ]
  %14 = trunc i64 %indvars.iv.i.us to i32
  %15 = load i32, i32* %k.i.i.us, align 4, !tbaa !98
  %sub.i.i.us = xor i32 %14, -1
  %sub2.i.i.us = add i32 %15, %sub.i.i.us
  %idxprom.i.i.us = sext i32 %sub2.i.i.us to i64
  %arrayidx.i.i.us = getelementptr inbounds %"struct.kNearestNeighbor<vertex<_point2d<double>, 1>, 1>::kNN", %"struct.kNearestNeighbor<vertex<_point2d<double>, 1>, 1>::kNN"* %nn.i.us, i64 0, i32 1, i64 %idxprom.i.i.us
  %16 = load %struct.vertex*, %struct.vertex** %arrayidx.i.i.us, align 8, !tbaa !20
  %arrayidx8.i.us = getelementptr inbounds %struct.vertex*, %struct.vertex** %arraydecay.us, i64 %indvars.iv.i.us
  store %struct.vertex* %16, %struct.vertex** %arrayidx8.i.us, align 8, !tbaa !20
  %indvars.iv.next.i.us = add nuw nsw i64 %indvars.iv.i.us, 1
  %exitcond.i.us = icmp eq i64 %indvars.iv.next.i.us, %wide.trip.count.i
  br i1 %exitcond.i.us, label %_ZN16kNearestNeighborI6vertexI8_point2dIdELi1EELi1EE8kNearestEPS3_PS5_i.exit.loopexit.us, label %for.body6.i.us

_ZN16kNearestNeighborI6vertexI8_point2dIdELi1EELi1EE8kNearestEPS3_PS5_i.exit.us: ; preds = %_ZN16kNearestNeighborI6vertexI8_point2dIdELi1EELi1EE8kNearestEPS3_PS5_i.exit.loopexit.us, %.noexc3.us
  call void @llvm.lifetime.end.p0i8(i64 32, i8* nonnull %10) #2
  reattach within %syncreg, label %pfor.inc.us

pfor.inc.us:                                      ; preds = %_ZN16kNearestNeighborI6vertexI8_point2dIdELi1EELi1EE8kNearestEPS3_PS5_i.exit.us, %pfor.detach.us
  %indvars.iv.next.us = add nuw nsw i64 %indvars.iv.us, 1
  %exitcond.us = icmp eq i64 %indvars.iv.next.us, %wide.trip.count
  br i1 %exitcond.us, label %pfor.cond.cleanup.loopexit.us-lcssa.us, label %pfor.detach.us, !llvm.loop !102

; SC: pfor.inc.us:
; SC: preds =
; SC-NOT: _ZN16kNearestNeighborI6vertexI8_point2dIdELi1EELi1EE8kNearestEPS3_PS5_i.exit.us
; SC: %pfor.detach.us
; SC-NEXT: %indvars.iv.next.us = add nuw nsw i64 %indvars.iv.us, 1

_ZN16kNearestNeighborI6vertexI8_point2dIdELi1EELi1EE8kNearestEPS3_PS5_i.exit.loopexit.us: ; preds = %for.body6.i.us
  br label %_ZN16kNearestNeighborI6vertexI8_point2dIdELi1EELi1EE8kNearestEPS3_PS5_i.exit.us

.noexc.loopexit.us:                               ; preds = %for.body.i.us
  br label %.noexc.us

lpad5.loopexit.us-lcssa.us:                       ; preds = %pfor.detach.us
  %lpad.us-lcssa.us = landingpad { i8*, i32 }
          cleanup
  br label %lpad5.loopexit

; TS: lpad5.loopexit.us-lcssa.us.body:
; TS: br label %lpad5

if.then.i.us-lcssa.us:                            ; preds = %pfor.body.us
  br label %if.then.i

lpad.loopexit20.us-lcssa.us:                      ; preds = %.noexc.us
  %lpad.us-lcssa25.us = landingpad { i8*, i32 }
          catch i8* null
  br label %lpad.loopexit20

pfor.cond.cleanup.loopexit.us-lcssa.us:           ; preds = %pfor.inc.us
  br label %pfor.cond.cleanup.loopexit

pfor.detach.lr.ph.split:                          ; preds = %pfor.detach.lr.ph.pfor.detach.lr.ph.split_crit_edge
  br label %pfor.detach

pfor.cond.cleanup.loopexit.us-lcssa:              ; preds = %pfor.inc
  br label %pfor.cond.cleanup.loopexit

pfor.cond.cleanup.loopexit:                       ; preds = %pfor.cond.cleanup.loopexit.us-lcssa.us, %pfor.cond.cleanup.loopexit.us-lcssa
  br label %pfor.cond.cleanup

pfor.cond.cleanup:                                ; preds = %pfor.cond.cleanup.loopexit, %entry
  sync within %syncreg, label %sync.continue

pfor.detach:                                      ; preds = %pfor.inc, %pfor.detach.lr.ph.split
  %indvars.iv = phi i64 [ 0, %pfor.detach.lr.ph.split ], [ %indvars.iv.next, %pfor.inc ]
  detach within %syncreg, label %pfor.body, label %pfor.inc unwind label %lpad5.loopexit.us-lcssa

pfor.body:                                        ; preds = %pfor.detach
  %nn.i = alloca %"struct.kNearestNeighbor<vertex<_point2d<double>, 1>, 1>::kNN", align 8
  %arrayidx = getelementptr inbounds %struct.vertex*, %struct.vertex** %6, i64 %indvars.iv
  %17 = load %struct.vertex*, %struct.vertex** %arrayidx, align 8, !tbaa !20
  %arraydecay = getelementptr inbounds %struct.vertex, %struct.vertex* %17, i64 0, i32 2, i64 0
  %18 = bitcast %"struct.kNearestNeighbor<vertex<_point2d<double>, 1>, 1>::kNN"* %nn.i to i8*
  call void @llvm.lifetime.start.p0i8(i64 32, i8* nonnull %18) #2
  br i1 false, label %if.then.i.us-lcssa, label %if.end.i

if.then.i.us-lcssa:                               ; preds = %pfor.body
  br label %if.then.i

if.then.i:                                        ; preds = %if.then.i.us-lcssa.us, %if.then.i.us-lcssa
  %call.i.i12 = call i64 @strlen(i8* getelementptr inbounds ([19 x i8], [19 x i8]* @.str.26, i64 0, i64 0)) #2
  %call1.i14 = invoke dereferenceable(272) %"class.std::basic_ostream"* @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(%"class.std::basic_ostream"* nonnull dereferenceable(272) @_ZSt4cout, i8* nonnull getelementptr inbounds ([19 x i8], [19 x i8]* @.str.26, i64 0, i64 0), i64 %call.i.i12)
          to label %call.i5.noexc unwind label %lpad.loopexit.split-lp21

call.i5.noexc:                                    ; preds = %if.then.i
  %call.i.i611 = invoke dereferenceable(272) %"class.std::basic_ostream"* @_ZSt4endlIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_(%"class.std::basic_ostream"* dereferenceable(272) @_ZSt4cout)
          to label %call.i.i6.noexc unwind label %lpad.loopexit.split-lp21

call.i.i6.noexc:                                  ; preds = %call.i5.noexc
  call void @abort() #18
  unreachable

if.end.i:                                         ; preds = %pfor.body
  %k.i = getelementptr inbounds %"struct.kNearestNeighbor<vertex<_point2d<double>, 1>, 1>::kNN", %"struct.kNearestNeighbor<vertex<_point2d<double>, 1>, 1>::kNN"* %nn.i, i64 0, i32 4
  store i32 %k, i32* %k.i, align 4, !tbaa !98
  %quads.i = getelementptr inbounds %"struct.kNearestNeighbor<vertex<_point2d<double>, 1>, 1>::kNN", %"struct.kNearestNeighbor<vertex<_point2d<double>, 1>, 1>::kNN"* %nn.i, i64 0, i32 3
  store i32 4, i32* %quads.i, align 8, !tbaa !100
  %ps.i = getelementptr inbounds %"struct.kNearestNeighbor<vertex<_point2d<double>, 1>, 1>::kNN", %"struct.kNearestNeighbor<vertex<_point2d<double>, 1>, 1>::kNN"* %nn.i, i64 0, i32 0
  store %struct.vertex* %17, %struct.vertex** %ps.i, align 8, !tbaa !101
  %19 = load i32, i32* %k.i, align 4, !tbaa !98
  %cmp515.i = icmp sgt i32 %19, 0
  br i1 %cmp515.i, label %for.body.preheader.i, label %.noexc

for.body.preheader.i:                             ; preds = %if.end.i
  br label %for.body.i

for.body.i:                                       ; preds = %for.body.i, %for.body.preheader.i
  %indvars.iv.i7 = phi i64 [ %indvars.iv.next.i9, %for.body.i ], [ 0, %for.body.preheader.i ]
  %arrayidx.i = getelementptr inbounds %"struct.kNearestNeighbor<vertex<_point2d<double>, 1>, 1>::kNN", %"struct.kNearestNeighbor<vertex<_point2d<double>, 1>, 1>::kNN"* %nn.i, i64 0, i32 1, i64 %indvars.iv.i7
  store %struct.vertex* null, %struct.vertex** %arrayidx.i, align 8, !tbaa !20
  %arrayidx8.i8 = getelementptr inbounds %"struct.kNearestNeighbor<vertex<_point2d<double>, 1>, 1>::kNN", %"struct.kNearestNeighbor<vertex<_point2d<double>, 1>, 1>::kNN"* %nn.i, i64 0, i32 2, i64 %indvars.iv.i7
  store double 0x7FEFFFFFFFFFFFFF, double* %arrayidx8.i8, align 8, !tbaa !82
  %indvars.iv.next.i9 = add nuw nsw i64 %indvars.iv.i7, 1
  %20 = load i32, i32* %k.i, align 4, !tbaa !98
  %21 = sext i32 %20 to i64
  %cmp5.i = icmp slt i64 %indvars.iv.next.i9, %21
  br i1 %cmp5.i, label %for.body.i, label %.noexc.loopexit

.noexc.loopexit:                                  ; preds = %for.body.i
  br label %.noexc

.noexc:                                           ; preds = %.noexc.loopexit, %if.end.i
  invoke void @_ZN16kNearestNeighborI6vertexI8_point2dIdELi1EELi1EE3kNN10nearestNghEP9gTreeNodeIS2_7_vect2dIdES3_5nDataIS3_EE(%"struct.kNearestNeighbor<vertex<_point2d<double>, 1>, 1>::kNN"* nonnull %nn.i, %class.gTreeNode* %call.i)
          to label %.noexc3 unwind label %lpad.loopexit20.us-lcssa

.noexc3:                                          ; preds = %.noexc
  br i1 %cmp24.i, label %for.body6.lr.ph.i, label %_ZN16kNearestNeighborI6vertexI8_point2dIdELi1EELi1EE8kNearestEPS3_PS5_i.exit

for.body6.lr.ph.i:                                ; preds = %.noexc3
  %result30.i = bitcast %struct.vertex** %arraydecay to i8*
  call void @llvm.memset.p0i8.i64(i8* %result30.i, i8 0, i64 %8, i32 8, i1 false)
  %k.i.i = getelementptr inbounds %"struct.kNearestNeighbor<vertex<_point2d<double>, 1>, 1>::kNN", %"struct.kNearestNeighbor<vertex<_point2d<double>, 1>, 1>::kNN"* %nn.i, i64 0, i32 4
  br label %for.body6.i

for.body6.i:                                      ; preds = %for.body6.i, %for.body6.lr.ph.i
  %indvars.iv.i = phi i64 [ 0, %for.body6.lr.ph.i ], [ %indvars.iv.next.i, %for.body6.i ]
  %22 = trunc i64 %indvars.iv.i to i32
  %23 = load i32, i32* %k.i.i, align 4, !tbaa !98
  %sub.i.i = xor i32 %22, -1
  %sub2.i.i = add i32 %23, %sub.i.i
  %idxprom.i.i = sext i32 %sub2.i.i to i64
  %arrayidx.i.i = getelementptr inbounds %"struct.kNearestNeighbor<vertex<_point2d<double>, 1>, 1>::kNN", %"struct.kNearestNeighbor<vertex<_point2d<double>, 1>, 1>::kNN"* %nn.i, i64 0, i32 1, i64 %idxprom.i.i
  %24 = load %struct.vertex*, %struct.vertex** %arrayidx.i.i, align 8, !tbaa !20
  %arrayidx8.i = getelementptr inbounds %struct.vertex*, %struct.vertex** %arraydecay, i64 %indvars.iv.i
  store %struct.vertex* %24, %struct.vertex** %arrayidx8.i, align 8, !tbaa !20
  %indvars.iv.next.i = add nuw nsw i64 %indvars.iv.i, 1
  %exitcond.i = icmp eq i64 %indvars.iv.next.i, %wide.trip.count.i
  br i1 %exitcond.i, label %_ZN16kNearestNeighborI6vertexI8_point2dIdELi1EELi1EE8kNearestEPS3_PS5_i.exit.loopexit, label %for.body6.i

_ZN16kNearestNeighborI6vertexI8_point2dIdELi1EELi1EE8kNearestEPS3_PS5_i.exit.loopexit: ; preds = %for.body6.i
  br label %_ZN16kNearestNeighborI6vertexI8_point2dIdELi1EELi1EE8kNearestEPS3_PS5_i.exit

_ZN16kNearestNeighborI6vertexI8_point2dIdELi1EELi1EE8kNearestEPS3_PS5_i.exit: ; preds = %_ZN16kNearestNeighborI6vertexI8_point2dIdELi1EELi1EE8kNearestEPS3_PS5_i.exit.loopexit, %.noexc3
  call void @llvm.lifetime.end.p0i8(i64 32, i8* nonnull %18) #2
  reattach within %syncreg, label %pfor.inc

pfor.inc:                                         ; preds = %_ZN16kNearestNeighborI6vertexI8_point2dIdELi1EELi1EE8kNearestEPS3_PS5_i.exit, %pfor.detach
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, %wide.trip.count
  br i1 %exitcond, label %pfor.cond.cleanup.loopexit.us-lcssa, label %pfor.detach, !llvm.loop !102

lpad.loopexit20.us-lcssa:                         ; preds = %.noexc
  %lpad.us-lcssa25 = landingpad { i8*, i32 }
          catch i8* null
  br label %lpad.loopexit20

lpad.loopexit20:                                  ; preds = %lpad.loopexit20.us-lcssa.us, %lpad.loopexit20.us-lcssa
  %25 = phi { i8*, i32 } [ %lpad.us-lcssa25.us, %lpad.loopexit20.us-lcssa.us ], [ %lpad.us-lcssa25, %lpad.loopexit20.us-lcssa ]
  br label %lpad

lpad.loopexit.split-lp21:                         ; preds = %call.i5.noexc, %if.then.i
  %lpad.loopexit.split-lp23 = landingpad { i8*, i32 }
          catch i8* null
  br label %lpad

; TS: lpad.loopexit.split-lp21:
; TS-NEXT: landingpad
; TS-NEXT: cleanup
; TS-NEXT: catch i8* null
; TS: br label %lpad.sd

lpad:                                             ; preds = %lpad.loopexit.split-lp21, %lpad.loopexit20
  %lpad.phi24 = phi { i8*, i32 } [ %25, %lpad.loopexit20 ], [ %lpad.loopexit.split-lp23, %lpad.loopexit.split-lp21 ]
  invoke void @llvm.detached.rethrow.sl_p0i8i32s(token %syncreg, { i8*, i32 } %lpad.phi24)
          to label %det.rethrow.unreachable unwind label %lpad5.loopexit.split-lp

; TS: lpad:
; TS-NOT: = phi
; TS: invoke void @llvm.detached.rethrow.sl_p0i8i32s(token %syncreg

det.rethrow.unreachable:                          ; preds = %lpad
  unreachable

lpad5.loopexit.us-lcssa:                          ; preds = %pfor.detach
  %lpad.us-lcssa = landingpad { i8*, i32 }
          cleanup
  br label %lpad5.loopexit

lpad5.loopexit:                                   ; preds = %lpad5.loopexit.us-lcssa.us, %lpad5.loopexit.us-lcssa
  %26 = phi { i8*, i32 } [ %lpad.us-lcssa.us, %lpad5.loopexit.us-lcssa.us ], [ %lpad.us-lcssa, %lpad5.loopexit.us-lcssa ]
  br label %lpad5

lpad5.loopexit.split-lp:                          ; preds = %lpad
  %lpad.loopexit.split-lp = landingpad { i8*, i32 }
          cleanup
  br label %lpad5

lpad5:                                            ; preds = %lpad5.loopexit.split-lp, %lpad5.loopexit
  %lpad.phi = phi { i8*, i32 } [ %26, %lpad5.loopexit ], [ %lpad.loopexit.split-lp, %lpad5.loopexit.split-lp ]
  sync within %syncreg, label %sync.continue9

sync.continue:                                    ; preds = %pfor.cond.cleanup
  %27 = bitcast %struct.vertex** %6 to i8*
  call void @free(i8* %27) #2
  call void @_ZN9gTreeNodeI8_point2dIdE7_vect2dIdE6vertexIS1_Li1EE5nDataIS5_EE3delEv(%class.gTreeNode* %call.i)
  ret void

sync.continue9:                                   ; preds = %lpad5
  resume { i8*, i32 } %lpad.phi
}

; TS: lpad.sd:
; TS: br label %lpad5.loopexit.us-lcssa.us.body

; Function Attrs: uwtable
declare void @_ZN16kNearestNeighborI6vertexI8_point2dIdELi1EELi1EE3kNN10nearestNghEP9gTreeNodeIS2_7_vect2dIdES3_5nDataIS3_EE(%"struct.kNearestNeighbor<vertex<_point2d<double>, 1>, 1>::kNN"* %this, %class.gTreeNode* %T) local_unnamed_addr #5

; Function Attrs: uwtable
declare void @_ZN9gTreeNodeI8_point2dIdE7_vect2dIdE6vertexIS1_Li1EE5nDataIS5_EE10applyIndexINS8_10flatten_FAEEEviT_(%class.gTreeNode* %this, i32 %s, %struct.vertex** %f.coerce) local_unnamed_addr #5

; Function Attrs: uwtable
declare void @_ZN9gTreeNodeI8_point2dIdE7_vect2dIdE6vertexIS1_Li1EE5nDataIS5_EE3delEv(%class.gTreeNode* %this) local_unnamed_addr #5

; Function Attrs: uwtable
declare %class.gTreeNode* @_ZN9gTreeNodeI8_point2dIdE7_vect2dIdE6vertexIS1_Li1EE5nDataIS5_EE5gTreeEPPS5_i(%struct.vertex** %vv, i32 %n) local_unnamed_addr #5

declare dereferenceable(272) %"class.std::basic_ostream"* @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(%"class.std::basic_ostream"* dereferenceable(272), i8*, i64) local_unnamed_addr #0

; Function Attrs: inlinehint uwtable
declare dereferenceable(272) %"class.std::basic_ostream"* @_ZSt4endlIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_(%"class.std::basic_ostream"* dereferenceable(272) %__os) #7

; Function Attrs: nobuiltin
declare noalias nonnull i8* @_Znam(i64) local_unnamed_addr #14

; Function Attrs: argmemonly nounwind
declare token @llvm.syncregion.start() #4

declare i32 @__gxx_personality_v0(...)

; Function Attrs: argmemonly
declare void @llvm.detached.rethrow.sl_p0i8i32s(token, { i8*, i32 }) #6

; Function Attrs: nounwind
declare noalias i8* @malloc(i64) local_unnamed_addr #1

; Function Attrs: nounwind
declare void @free(i8* nocapture) local_unnamed_addr #1

; Function Attrs: noreturn nounwind
declare void @abort() local_unnamed_addr #8

; Function Attrs: argmemonly nounwind
declare void @llvm.lifetime.start.p0i8(i64, i8* nocapture) #4

; Function Attrs: argmemonly nounwind
declare void @llvm.lifetime.end.p0i8(i64, i8* nocapture) #4

; Function Attrs: argmemonly nounwind
declare void @llvm.memset.p0i8.i64(i8* nocapture writeonly, i8, i64, i32, i1) #4

; Function Attrs: nounwind readnone speculatable
declare { i64, i1 } @llvm.umul.with.overflow.i64(i64, i64) #15

; Function Attrs: argmemonly nounwind readonly
declare i64 @strlen(i8* nocapture) local_unnamed_addr #13

!4 = !{!"double", !5, i64 0}
!5 = !{!"omnipotent char", !6, i64 0}
!6 = !{!"Simple C++ TBAA"}
!14 = !{!"tapir.loop.spawn.strategy", i32 1}
!20 = !{!21, !21, i64 0}
!21 = !{!"any pointer", !5, i64 0}
!35 = !{!"int", !5, i64 0}
!63 = !{!"_ZTS8_point2dIdE", !4, i64 0, !4, i64 8}
!82 = !{!4, !4, i64 0}
!95 = !{!96, !35, i64 28}
!96 = !{!"_ZTS9gTreeNodeI8_point2dIdE7_vect2dIdE6vertexIS1_Li1EE5nDataIS5_EE", !63, i64 0, !4, i64 16, !97, i64 24, !35, i64 28, !5, i64 32, !21, i64 96, !21, i64 104}
!97 = !{!"_ZTS5nDataI6vertexI8_point2dIdELi1EEE", !35, i64 0}
!98 = !{!99, !35, i64 28}
!99 = !{!"_ZTSN16kNearestNeighborI6vertexI8_point2dIdELi1EELi1EE3kNNE", !21, i64 0, !5, i64 8, !5, i64 16, !35, i64 24, !35, i64 28}
!100 = !{!99, !35, i64 24}
!101 = !{!99, !21, i64 0}
!102 = distinct !{!102, !14, !103}
!103 = !{!"tapir.loop.grainsize", i32 1}

