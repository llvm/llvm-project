; Check that Tapir lowering to the Cilk or CilkR targets will decorate
; functions that can be stolen with the "stealable" attribute.
;
; RUN: opt < %s -loop-spawning-ti -simplifycfg -instcombine -tapir2target -tapir-target=cilk -simplifycfg -instcombine -S | FileCheck %s --check-prefix=LOWERING
; RUN: opt < %s -loop-spawning-ti -simplifycfg -instcombine -tapir2target -tapir-target=cilkr -simplifycfg -instcombine -S | FileCheck %s --check-prefix=LOWERING
; RUN: opt < %s -passes='loop-spawning,function(simplify-cfg,instcombine),tapir2target,function(simplify-cfg,instcombine)' -tapir-target=cilkr -instcombine -S | FileCheck %s --check-prefix=LOWERING
;
; Check that the X86 assembly produced for stealable functions does
; not index stack variables using %rsp.
;
; RUN: opt < %s -loop-spawning-ti -simplifycfg -instcombine -tapir2target -tapir-target=cilk -simplifycfg -instcombine | llc -O3 -mtriple=x86_64-unknown-linux-gnu | FileCheck %s --check-prefix=ASM
; RUN: opt < %s -loop-spawning-ti -simplifycfg -instcombine -tapir2target -tapir-target=cilkr -simplifycfg -instcombine | llc -O3 -mtriple=x86_64-unknown-linux-gnu | FileCheck %s --check-prefix=ASM
; RUN: opt < %s -passes='loop-spawning,function(simplify-cfg,instcombine),tapir2target,function(simplify-cfg,instcombine)' -tapir-target=cilkr | llc -O3 -mtriple=x86_64-unknown-linux-gnu | FileCheck %s --check-prefix=ASM

%class._point3d = type { double, double, double }
%struct.vertex.29 = type { i32, %class._point3d, [1 x %struct.vertex.29*] }
%"struct.std::pair.38" = type { i32, %struct.vertex.29* }

$_ZN9gTreeNodeI8_point3dIdE7_vect3dIdE6vertexIS1_Li1EE5nDataIS5_EE13sortBlocksBigEPPS5_iiidS1_Pi = comdat any

; Function Attrs: nounwind uwtable
define linkonce_odr void @_ZN9gTreeNodeI8_point3dIdE7_vect3dIdE6vertexIS1_Li1EE5nDataIS5_EE13sortBlocksBigEPPS5_iiidS1_Pi(%struct.vertex.29** %S, i32 %count, i32 %quadrants, i32 %logdivs, double %size, %class._point3d* byval align 8 %center, i32* %offsets) local_unnamed_addr #6 comdat align 2 {
entry:
  %syncreg = tail call token @llvm.syncregion.start()
  %syncreg18 = tail call token @llvm.syncregion.start()
  %conv = sext i32 %count to i64
  %mul = shl nsw i64 %conv, 4
  %call = tail call noalias i8* @malloc(i64 %mul) #2
  %0 = bitcast i8* %call to %"struct.std::pair.38"*
  %shl = shl i32 1, %logdivs
  %div2.neg.neg = fmul fast double %size, 5.000000e-01
  %cmp94 = icmp sgt i32 %count, 0
  br i1 %cmp94, label %pfor.detach.lr.ph, label %pfor.cond.cleanup

pfor.detach.lr.ph:                                ; preds = %entry
  %conv1 = sitofp i32 %shl to double
  %div = fdiv fast double %size, %conv1
  %z.i = getelementptr inbounds %class._point3d, %class._point3d* %center, i64 0, i32 2
  %1 = load double, double* %z.i, align 8, !tbaa !12, !noalias !417
  %y.i = getelementptr inbounds %class._point3d, %class._point3d* %center, i64 0, i32 1
  %2 = load double, double* %y.i, align 8, !tbaa !11, !noalias !417
  %x.i = getelementptr inbounds %class._point3d, %class._point3d* %center, i64 0, i32 0
  %3 = load double, double* %x.i, align 8, !tbaa !6, !noalias !417
  %add.i.neg = fsub fast double %div2.neg.neg, %3
  %4 = fdiv fast double 1.000000e+00, %div
  %sub3.i = add nsw i32 %shl, -1
  %add9.i.neg = fsub fast double %div2.neg.neg, %2
  %add17.i.neg = fsub fast double %div2.neg.neg, %1
  %cmp75.i = icmp sgt i32 %logdivs, 0
  %wide.trip.count99 = zext i32 %count to i64
  br i1 %cmp75.i, label %pfor.detach.us.preheader, label %pfor.detach.preheader

pfor.detach.preheader:                            ; preds = %pfor.detach.lr.ph
  br label %pfor.detach

pfor.detach.us.preheader:                         ; preds = %pfor.detach.lr.ph
  %min.iters.check = icmp ult i32 %logdivs, 16
  %n.vec = and i32 %logdivs, -16
  %cmp.n = icmp eq i32 %n.vec, %logdivs
  br label %pfor.detach.us

pfor.detach.us:                                   ; preds = %pfor.detach.us.preheader, %pfor.inc.us
  %indvars.iv97 = phi i64 [ %indvars.iv.next98, %pfor.inc.us ], [ 0, %pfor.detach.us.preheader ]
  detach within %syncreg, label %for.body.lr.ph.i.us, label %pfor.inc.us
; LOWERING: _ZN9gTreeNodeI8_point3dIdE7_vect3dIdE6vertexIS1_Li1EE5nDataIS5_EE13sortBlocksBigEPPS5_iiidS1_Pi.outline_pfor.detach.us.ls1({{[^)]*}}) unnamed_addr [[FUNCATTR:#[0-9]+]]
; LOWERING: attributes [[FUNCATTR]] = { {{[^}]*}}stealable
; ASM: _ZN9gTreeNodeI8_point3dIdE7_vect3dIdE6vertexIS1_Li1EE5nDataIS5_EE13sortBlocksBigEPPS5_iiidS1_Pi.outline_pfor.detach.us.ls1:
; ASM-NOT: (%rsp)
; ASM: @function

for.body.lr.ph.i.us:                              ; preds = %pfor.detach.us
  %arrayidx.us = getelementptr inbounds %struct.vertex.29*, %struct.vertex.29** %S, i64 %indvars.iv97
  %5 = load %struct.vertex.29*, %struct.vertex.29** %arrayidx.us, align 8, !tbaa !30
  %agg.tmp8.sroa.0.0..sroa_idx.us = getelementptr inbounds %struct.vertex.29, %struct.vertex.29* %5, i64 0, i32 1, i32 0
  %agg.tmp8.sroa.0.0.copyload.us = load double, double* %agg.tmp8.sroa.0.0..sroa_idx.us, align 8
  %agg.tmp8.sroa.2.0..sroa_idx78.us = getelementptr inbounds %struct.vertex.29, %struct.vertex.29* %5, i64 0, i32 1, i32 1
  %agg.tmp8.sroa.2.0.copyload.us = load double, double* %agg.tmp8.sroa.2.0..sroa_idx78.us, align 8
  %agg.tmp8.sroa.3.0..sroa_idx79.us = getelementptr inbounds %struct.vertex.29, %struct.vertex.29* %5, i64 0, i32 1, i32 2
  %agg.tmp8.sroa.3.0.copyload.us = load double, double* %agg.tmp8.sroa.3.0..sroa_idx79.us, align 8
  %sub.i72.us = fadd fast double %add.i.neg, %agg.tmp8.sroa.0.0.copyload.us
  %6 = fmul fast double %sub.i72.us, %4
  %conv.i.us = fptosi double %6 to i32
  %cmp.i.i.us = icmp slt i32 %sub3.i, %conv.i.us
  %.sroa.speculated70.i.us = select i1 %cmp.i.i.us, i32 %sub3.i, i32 %conv.i.us
  %sub6.i.us = fadd fast double %add9.i.neg, %agg.tmp8.sroa.2.0.copyload.us
  %7 = fmul fast double %sub6.i.us, %4
  %conv8.i.us = fptosi double %7 to i32
  %cmp.i56.i.us = icmp slt i32 %sub3.i, %conv8.i.us
  %.sroa.speculated64.i.us = select i1 %cmp.i56.i.us, i32 %sub3.i, i32 %conv8.i.us
  %sub14.i.us = fadd fast double %add17.i.neg, %agg.tmp8.sroa.3.0.copyload.us
  %8 = fmul fast double %sub14.i.us, %4
  %conv16.i.us = fptosi double %8 to i32
  %cmp.i54.i.us = icmp slt i32 %sub3.i, %conv16.i.us
  %.sroa.speculated.i.us = select i1 %cmp.i54.i.us, i32 %sub3.i, i32 %conv16.i.us
  %9 = ptrtoint %struct.vertex.29* %5 to i64
  br i1 %min.iters.check, label %for.body.i.us.preheader, label %vector.ph

vector.ph:                                        ; preds = %for.body.lr.ph.i.us
  %broadcast.splatinsert = insertelement <8 x i32> undef, i32 %.sroa.speculated70.i.us, i32 0
  %broadcast.splat = shufflevector <8 x i32> %broadcast.splatinsert, <8 x i32> undef, <8 x i32> zeroinitializer
  %broadcast.splatinsert109 = insertelement <8 x i32> undef, i32 %.sroa.speculated70.i.us, i32 0
  %broadcast.splat110 = shufflevector <8 x i32> %broadcast.splatinsert109, <8 x i32> undef, <8 x i32> zeroinitializer
  %broadcast.splatinsert111 = insertelement <8 x i32> undef, i32 %.sroa.speculated64.i.us, i32 0
  %broadcast.splat112 = shufflevector <8 x i32> %broadcast.splatinsert111, <8 x i32> undef, <8 x i32> zeroinitializer
  %broadcast.splatinsert113 = insertelement <8 x i32> undef, i32 %.sroa.speculated64.i.us, i32 0
  %broadcast.splat114 = shufflevector <8 x i32> %broadcast.splatinsert113, <8 x i32> undef, <8 x i32> zeroinitializer
  %broadcast.splatinsert115 = insertelement <8 x i32> undef, i32 %.sroa.speculated.i.us, i32 0
  %broadcast.splat116 = shufflevector <8 x i32> %broadcast.splatinsert115, <8 x i32> undef, <8 x i32> zeroinitializer
  %broadcast.splatinsert117 = insertelement <8 x i32> undef, i32 %.sroa.speculated.i.us, i32 0
  %broadcast.splat118 = shufflevector <8 x i32> %broadcast.splatinsert117, <8 x i32> undef, <8 x i32> zeroinitializer
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i32 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %vec.phi = phi <8 x i32> [ zeroinitializer, %vector.ph ], [ %34, %vector.body ]
  %vec.phi107 = phi <8 x i32> [ zeroinitializer, %vector.ph ], [ %35, %vector.body ]
  %vec.ind = phi <8 x i32> [ <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, %vector.ph ], [ %vec.ind.next, %vector.body ]
  %step.add = add <8 x i32> %vec.ind, <i32 8, i32 8, i32 8, i32 8, i32 8, i32 8, i32 8, i32 8>
  %10 = shl <8 x i32> <i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1>, %vec.ind
  %11 = shl <8 x i32> <i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1>, %step.add
  %12 = and <8 x i32> %10, %broadcast.splat
  %13 = and <8 x i32> %11, %broadcast.splat110
  %14 = shl nuw nsw <8 x i32> %vec.ind, <i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1>
  %15 = shl nuw nsw <8 x i32> %step.add, <i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1>
  %16 = shl <8 x i32> %12, %14
  %17 = shl <8 x i32> %13, %15
  %18 = add nsw <8 x i32> %16, %vec.phi
  %19 = add nsw <8 x i32> %17, %vec.phi107
  %20 = and <8 x i32> %10, %broadcast.splat112
  %21 = and <8 x i32> %11, %broadcast.splat114
  %22 = or <8 x i32> %14, <i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1>
  %23 = or <8 x i32> %15, <i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1>
  %24 = shl <8 x i32> %20, %22
  %25 = shl <8 x i32> %21, %23
  %26 = add nsw <8 x i32> %18, %24
  %27 = add nsw <8 x i32> %19, %25
  %28 = and <8 x i32> %10, %broadcast.splat116
  %29 = and <8 x i32> %11, %broadcast.splat118
  %30 = add nuw nsw <8 x i32> %14, <i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2>
  %31 = add nuw nsw <8 x i32> %15, <i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2>
  %32 = shl <8 x i32> %28, %30
  %33 = shl <8 x i32> %29, %31
  %34 = add nsw <8 x i32> %26, %32
  %35 = add nsw <8 x i32> %27, %33
  %index.next = add i32 %index, 16
  %vec.ind.next = add <8 x i32> %vec.ind, <i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16>
  %36 = icmp eq i32 %index.next, %n.vec
  br i1 %36, label %middle.block, label %vector.body, !llvm.loop !420

middle.block:                                     ; preds = %vector.body
  %bin.rdx = add <8 x i32> %35, %34
  %rdx.shuf = shufflevector <8 x i32> %bin.rdx, <8 x i32> undef, <8 x i32> <i32 4, i32 5, i32 6, i32 7, i32 undef, i32 undef, i32 undef, i32 undef>
  %bin.rdx119 = add <8 x i32> %bin.rdx, %rdx.shuf
  %rdx.shuf120 = shufflevector <8 x i32> %bin.rdx119, <8 x i32> undef, <8 x i32> <i32 2, i32 3, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  %bin.rdx121 = add <8 x i32> %bin.rdx119, %rdx.shuf120
  %rdx.shuf122 = shufflevector <8 x i32> %bin.rdx121, <8 x i32> undef, <8 x i32> <i32 1, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  %bin.rdx123 = add <8 x i32> %bin.rdx121, %rdx.shuf122
  %37 = extractelement <8 x i32> %bin.rdx123, i32 0
  br i1 %cmp.n, label %_Z11ptFindBlock8_point3dIdEdiS0_.exit.loopexit.us, label %for.body.i.us.preheader

for.body.i.us.preheader:                          ; preds = %middle.block, %for.body.lr.ph.i.us
  %result.077.i.us.ph = phi i32 [ 0, %for.body.lr.ph.i.us ], [ %37, %middle.block ]
  %i.076.i.us.ph = phi i32 [ 0, %for.body.lr.ph.i.us ], [ %n.vec, %middle.block ]
  br label %for.body.i.us

for.body.i.us:                                    ; preds = %for.body.i.us.preheader, %for.body.i.us
  %result.077.i.us = phi i32 [ %add31.i.us, %for.body.i.us ], [ %result.077.i.us.ph, %for.body.i.us.preheader ]
  %i.076.i.us = phi i32 [ %inc.i.us, %for.body.i.us ], [ %i.076.i.us.ph, %for.body.i.us.preheader ]
  %shl20.i.us = shl i32 1, %i.076.i.us
  %and.i.us = and i32 %shl20.i.us, %.sroa.speculated70.i.us
  %mul.i.us = shl nuw nsw i32 %i.076.i.us, 1
  %shl21.i.us = shl i32 %and.i.us, %mul.i.us
  %add.i75.us = add nsw i32 %shl21.i.us, %result.077.i.us
  %and22.i.us = and i32 %shl20.i.us, %.sroa.speculated64.i.us
  %add24.i.us = or i32 %mul.i.us, 1
  %shl25.i.us = shl i32 %and22.i.us, %add24.i.us
  %add26.i.us = add nsw i32 %add.i75.us, %shl25.i.us
  %and27.i.us = and i32 %shl20.i.us, %.sroa.speculated.i.us
  %add29.i.us = add nuw nsw i32 %mul.i.us, 2
  %shl30.i.us = shl i32 %and27.i.us, %add29.i.us
  %add31.i.us = add nsw i32 %add26.i.us, %shl30.i.us
  %inc.i.us = add nuw nsw i32 %i.076.i.us, 1
  %exitcond.i.us = icmp eq i32 %inc.i.us, %logdivs
  br i1 %exitcond.i.us, label %_Z11ptFindBlock8_point3dIdEdiS0_.exit.loopexit.us, label %for.body.i.us, !llvm.loop !421

pfor.inc.us:                                      ; preds = %_Z11ptFindBlock8_point3dIdEdiS0_.exit.loopexit.us, %pfor.detach.us
  %indvars.iv.next98 = add nuw nsw i64 %indvars.iv97, 1
  %exitcond100 = icmp eq i64 %indvars.iv.next98, %wide.trip.count99
  br i1 %exitcond100, label %pfor.cond.cleanup, label %pfor.detach.us, !llvm.loop !422

_Z11ptFindBlock8_point3dIdEdiS0_.exit.loopexit.us: ; preds = %for.body.i.us, %middle.block
  %add31.i.us.lcssa = phi i32 [ %37, %middle.block ], [ %add31.i.us, %for.body.i.us ]
  %first2.i.us = getelementptr inbounds %"struct.std::pair.38", %"struct.std::pair.38"* %0, i64 %indvars.iv97, i32 0
  store i32 %add31.i.us.lcssa, i32* %first2.i.us, align 8, !tbaa !423
  %second4.i.us = getelementptr inbounds %"struct.std::pair.38", %"struct.std::pair.38"* %0, i64 %indvars.iv97, i32 1
  %38 = bitcast %struct.vertex.29** %second4.i.us to i64*
  store i64 %9, i64* %38, align 8, !tbaa !425
  reattach within %syncreg, label %pfor.inc.us

pfor.cond.cleanup:                                ; preds = %pfor.inc, %pfor.inc.us, %entry
  sync within %syncreg, label %sync.continue

pfor.detach:                                      ; preds = %pfor.detach.preheader, %pfor.inc
  %indvars.iv101 = phi i64 [ %indvars.iv.next102, %pfor.inc ], [ 0, %pfor.detach.preheader ]
  detach within %syncreg, label %_Z11ptFindBlock8_point3dIdEdiS0_.exit, label %pfor.inc

_Z11ptFindBlock8_point3dIdEdiS0_.exit:            ; preds = %pfor.detach
  %arrayidx = getelementptr inbounds %struct.vertex.29*, %struct.vertex.29** %S, i64 %indvars.iv101
  %39 = bitcast %struct.vertex.29** %arrayidx to i64*
  %40 = load i64, i64* %39, align 8, !tbaa !30
  %first2.i = getelementptr inbounds %"struct.std::pair.38", %"struct.std::pair.38"* %0, i64 %indvars.iv101, i32 0
  store i32 0, i32* %first2.i, align 8, !tbaa !423
  %second4.i = getelementptr inbounds %"struct.std::pair.38", %"struct.std::pair.38"* %0, i64 %indvars.iv101, i32 1
  %41 = bitcast %struct.vertex.29** %second4.i to i64*
  store i64 %40, i64* %41, align 8, !tbaa !425
  reattach within %syncreg, label %pfor.inc

pfor.inc:                                         ; preds = %_Z11ptFindBlock8_point3dIdEdiS0_.exit, %pfor.detach
  %indvars.iv.next102 = add nuw nsw i64 %indvars.iv101, 1
  %exitcond104 = icmp eq i64 %indvars.iv.next102, %wide.trip.count99
  br i1 %exitcond104, label %pfor.cond.cleanup, label %pfor.detach, !llvm.loop !422

sync.continue:                                    ; preds = %pfor.cond.cleanup
  %conv16 = sext i32 %quadrants to i64
  %cmp.i.i.i = icmp eq i32 %count, 2147483647
  %cond.i.i.i = select i1 %cmp.i.i.i, i64 8, i64 4
  %div.i.i.i = sdiv i64 %conv, 2048
  %add.i.i.i = add nsw i64 %div.i.i.i, 1
  %mul112.i.i.i = or i64 %cond.i.i.i, 16
  %add2.i.i.i = mul nsw i64 %mul112.i.i.i, %conv
  %mul3.i.i.i = shl nuw nsw i64 %cond.i.i.i, 8
  %mul4.i.i.i = mul nsw i64 %mul3.i.i.i, %add.i.i.i
  %add5.i.i.i = add nsw i64 %add2.i.i.i, %mul4.i.i.i
  %call1.i.i = tail call noalias i8* @malloc(i64 %add5.i.i.i) #2
  tail call void @_ZN7intSort6iSortXIjSt4pairIiP6vertexI8_point3dIdELi1EEEN5utils6firstFIiS6_EEiEEvPT0_PT2_llbPcT1_(%"struct.std::pair.38"* %0, i32* %offsets, i64 %conv, i64 %conv16, i1 zeroext false, i8* %call1.i.i) #2
  tail call void @free(i8* %call1.i.i) #2
  br i1 %cmp94, label %pfor.detach29.lr.ph, label %pfor.cond.cleanup28

pfor.detach29.lr.ph:                              ; preds = %sync.continue
  %wide.trip.count = zext i32 %count to i64
  br label %pfor.detach29

pfor.cond.cleanup28:                              ; preds = %pfor.inc40, %sync.continue
  sync within %syncreg18, label %sync.continue42

pfor.detach29:                                    ; preds = %pfor.inc40, %pfor.detach29.lr.ph
  %indvars.iv = phi i64 [ 0, %pfor.detach29.lr.ph ], [ %indvars.iv.next, %pfor.inc40 ]
  detach within %syncreg18, label %pfor.body34, label %pfor.inc40

pfor.body34:                                      ; preds = %pfor.detach29
  %second = getelementptr inbounds %"struct.std::pair.38", %"struct.std::pair.38"* %0, i64 %indvars.iv, i32 1
  %42 = bitcast %struct.vertex.29** %second to i64*
  %43 = load i64, i64* %42, align 8, !tbaa !425
  %arrayidx38 = getelementptr inbounds %struct.vertex.29*, %struct.vertex.29** %S, i64 %indvars.iv
  %44 = bitcast %struct.vertex.29** %arrayidx38 to i64*
  store i64 %43, i64* %44, align 8, !tbaa !30
  reattach within %syncreg18, label %pfor.inc40

pfor.inc40:                                       ; preds = %pfor.body34, %pfor.detach29
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, %wide.trip.count
  br i1 %exitcond, label %pfor.cond.cleanup28, label %pfor.detach29, !llvm.loop !426

sync.continue42:                                  ; preds = %pfor.cond.cleanup28
  tail call void @free(i8* %call) #2
  ret void
}

declare void @_ZN7intSort6iSortXIjSt4pairIiP6vertexI8_point3dIdELi1EEEN5utils6firstFIiS6_EEiEEvPT0_PT2_llbPcT1_(%"struct.std::pair.38"* %A, i32* %bucketOffsets, i64 %n, i64 %m, i1 zeroext %bottomUp, i8* %tmpSpace) local_unnamed_addr #6

; Function Attrs: argmemonly nounwind
declare token @llvm.syncregion.start() #4

; Function Attrs: nounwind
declare noalias i8* @malloc(i64) local_unnamed_addr #1

; Function Attrs: nounwind
declare void @free(i8* nocapture) local_unnamed_addr #1

attributes #0 = { "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="haswell" "target-features"="+aes,+avx,+avx2,+bmi,+bmi2,+cmov,+cx16,+f16c,+fma,+fsgsbase,+fxsr,+lzcnt,+mmx,+movbe,+pclmul,+popcnt,+rdrnd,+sse,+sse2,+sse3,+sse4.1,+sse4.2,+ssse3,+x87,+xsave,+xsaveopt,-adx,-avx512bitalg,-avx512bw,-avx512cd,-avx512dq,-avx512er,-avx512f,-avx512ifma,-avx512pf,-avx512vbmi,-avx512vbmi2,-avx512vl,-avx512vnni,-avx512vpopcntdq,-clflushopt,-clwb,-clzero,-fma4,-gfni,-ibt,-lwp,-mwaitx,-pku,-prefetchwt1,-prfchw,-rdseed,-rtm,-sgx,-sha,-shstk,-sse4a,-tbm,-vaes,-vpclmulqdq,-xop,-xsavec,-xsaves" "unsafe-fp-math"="true" "use-soft-float"="false" }
attributes #1 = { nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="haswell" "target-features"="+aes,+avx,+avx2,+bmi,+bmi2,+cmov,+cx16,+f16c,+fma,+fsgsbase,+fxsr,+lzcnt,+mmx,+movbe,+pclmul,+popcnt,+rdrnd,+sse,+sse2,+sse3,+sse4.1,+sse4.2,+ssse3,+x87,+xsave,+xsaveopt,-adx,-avx512bitalg,-avx512bw,-avx512cd,-avx512dq,-avx512er,-avx512f,-avx512ifma,-avx512pf,-avx512vbmi,-avx512vbmi2,-avx512vl,-avx512vnni,-avx512vpopcntdq,-clflushopt,-clwb,-clzero,-fma4,-gfni,-ibt,-lwp,-mwaitx,-pku,-prefetchwt1,-prfchw,-rdseed,-rtm,-sgx,-sha,-shstk,-sse4a,-tbm,-vaes,-vpclmulqdq,-xop,-xsavec,-xsaves" "unsafe-fp-math"="true" "use-soft-float"="false" }
attributes #2 = { nounwind }
attributes #3 = { nounwind readnone uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="true" "no-jump-tables"="false" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="haswell" "target-features"="+aes,+avx,+avx2,+bmi,+bmi2,+cmov,+cx16,+f16c,+fma,+fsgsbase,+fxsr,+lzcnt,+mmx,+movbe,+pclmul,+popcnt,+rdrnd,+sse,+sse2,+sse3,+sse4.1,+sse4.2,+ssse3,+x87,+xsave,+xsaveopt,-adx,-avx512bitalg,-avx512bw,-avx512cd,-avx512dq,-avx512er,-avx512f,-avx512ifma,-avx512pf,-avx512vbmi,-avx512vbmi2,-avx512vl,-avx512vnni,-avx512vpopcntdq,-clflushopt,-clwb,-clzero,-fma4,-gfni,-ibt,-lwp,-mwaitx,-pku,-prefetchwt1,-prfchw,-rdseed,-rtm,-sgx,-sha,-shstk,-sse4a,-tbm,-vaes,-vpclmulqdq,-xop,-xsavec,-xsaves" "unsafe-fp-math"="true" "use-soft-float"="false" }
attributes #4 = { argmemonly nounwind }
attributes #5 = { nounwind readonly uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="true" "no-jump-tables"="false" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="haswell" "target-features"="+aes,+avx,+avx2,+bmi,+bmi2,+cmov,+cx16,+f16c,+fma,+fsgsbase,+fxsr,+lzcnt,+mmx,+movbe,+pclmul,+popcnt,+rdrnd,+sse,+sse2,+sse3,+sse4.1,+sse4.2,+ssse3,+x87,+xsave,+xsaveopt,-adx,-avx512bitalg,-avx512bw,-avx512cd,-avx512dq,-avx512er,-avx512f,-avx512ifma,-avx512pf,-avx512vbmi,-avx512vbmi2,-avx512vl,-avx512vnni,-avx512vpopcntdq,-clflushopt,-clwb,-clzero,-fma4,-gfni,-ibt,-lwp,-mwaitx,-pku,-prefetchwt1,-prfchw,-rdseed,-rtm,-sgx,-sha,-shstk,-sse4a,-tbm,-vaes,-vpclmulqdq,-xop,-xsavec,-xsaves" "unsafe-fp-math"="true" "use-soft-float"="false" }
attributes #6 = { nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="true" "no-jump-tables"="false" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="haswell" "target-features"="+aes,+avx,+avx2,+bmi,+bmi2,+cmov,+cx16,+f16c,+fma,+fsgsbase,+fxsr,+lzcnt,+mmx,+movbe,+pclmul,+popcnt,+rdrnd,+sse,+sse2,+sse3,+sse4.1,+sse4.2,+ssse3,+x87,+xsave,+xsaveopt,-adx,-avx512bitalg,-avx512bw,-avx512cd,-avx512dq,-avx512er,-avx512f,-avx512ifma,-avx512pf,-avx512vbmi,-avx512vbmi2,-avx512vl,-avx512vnni,-avx512vpopcntdq,-clflushopt,-clwb,-clzero,-fma4,-gfni,-ibt,-lwp,-mwaitx,-pku,-prefetchwt1,-prfchw,-rdseed,-rtm,-sgx,-sha,-shstk,-sse4a,-tbm,-vaes,-vpclmulqdq,-xop,-xsavec,-xsaves" "unsafe-fp-math"="true" "use-soft-float"="false" }
attributes #7 = { inlinehint nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="true" "no-jump-tables"="false" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="haswell" "target-features"="+aes,+avx,+avx2,+bmi,+bmi2,+cmov,+cx16,+f16c,+fma,+fsgsbase,+fxsr,+lzcnt,+mmx,+movbe,+pclmul,+popcnt,+rdrnd,+sse,+sse2,+sse3,+sse4.1,+sse4.2,+ssse3,+x87,+xsave,+xsaveopt,-adx,-avx512bitalg,-avx512bw,-avx512cd,-avx512dq,-avx512er,-avx512f,-avx512ifma,-avx512pf,-avx512vbmi,-avx512vbmi2,-avx512vl,-avx512vnni,-avx512vpopcntdq,-clflushopt,-clwb,-clzero,-fma4,-gfni,-ibt,-lwp,-mwaitx,-pku,-prefetchwt1,-prfchw,-rdseed,-rtm,-sgx,-sha,-shstk,-sse4a,-tbm,-vaes,-vpclmulqdq,-xop,-xsavec,-xsaves" "unsafe-fp-math"="true" "use-soft-float"="false" }
attributes #8 = { noreturn nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="haswell" "target-features"="+aes,+avx,+avx2,+bmi,+bmi2,+cmov,+cx16,+f16c,+fma,+fsgsbase,+fxsr,+lzcnt,+mmx,+movbe,+pclmul,+popcnt,+rdrnd,+sse,+sse2,+sse3,+sse4.1,+sse4.2,+ssse3,+x87,+xsave,+xsaveopt,-adx,-avx512bitalg,-avx512bw,-avx512cd,-avx512dq,-avx512er,-avx512f,-avx512ifma,-avx512pf,-avx512vbmi,-avx512vbmi2,-avx512vl,-avx512vnni,-avx512vpopcntdq,-clflushopt,-clwb,-clzero,-fma4,-gfni,-ibt,-lwp,-mwaitx,-pku,-prefetchwt1,-prfchw,-rdseed,-rtm,-sgx,-sha,-shstk,-sse4a,-tbm,-vaes,-vpclmulqdq,-xop,-xsavec,-xsaves" "unsafe-fp-math"="true" "use-soft-float"="false" }
attributes #9 = { norecurse nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="true" "no-jump-tables"="false" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="haswell" "target-features"="+aes,+avx,+avx2,+bmi,+bmi2,+cmov,+cx16,+f16c,+fma,+fsgsbase,+fxsr,+lzcnt,+mmx,+movbe,+pclmul,+popcnt,+rdrnd,+sse,+sse2,+sse3,+sse4.1,+sse4.2,+ssse3,+x87,+xsave,+xsaveopt,-adx,-avx512bitalg,-avx512bw,-avx512cd,-avx512dq,-avx512er,-avx512f,-avx512ifma,-avx512pf,-avx512vbmi,-avx512vbmi2,-avx512vl,-avx512vnni,-avx512vpopcntdq,-clflushopt,-clwb,-clzero,-fma4,-gfni,-ibt,-lwp,-mwaitx,-pku,-prefetchwt1,-prfchw,-rdseed,-rtm,-sgx,-sha,-shstk,-sse4a,-tbm,-vaes,-vpclmulqdq,-xop,-xsavec,-xsaves" "unsafe-fp-math"="true" "use-soft-float"="false" }
attributes #10 = { noreturn "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="haswell" "target-features"="+aes,+avx,+avx2,+bmi,+bmi2,+cmov,+cx16,+f16c,+fma,+fsgsbase,+fxsr,+lzcnt,+mmx,+movbe,+pclmul,+popcnt,+rdrnd,+sse,+sse2,+sse3,+sse4.1,+sse4.2,+ssse3,+x87,+xsave,+xsaveopt,-adx,-avx512bitalg,-avx512bw,-avx512cd,-avx512dq,-avx512er,-avx512f,-avx512ifma,-avx512pf,-avx512vbmi,-avx512vbmi2,-avx512vl,-avx512vnni,-avx512vpopcntdq,-clflushopt,-clwb,-clzero,-fma4,-gfni,-ibt,-lwp,-mwaitx,-pku,-prefetchwt1,-prfchw,-rdseed,-rtm,-sgx,-sha,-shstk,-sse4a,-tbm,-vaes,-vpclmulqdq,-xop,-xsavec,-xsaves" "unsafe-fp-math"="true" "use-soft-float"="false" }
attributes #11 = { nounwind readonly "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="haswell" "target-features"="+aes,+avx,+avx2,+bmi,+bmi2,+cmov,+cx16,+f16c,+fma,+fsgsbase,+fxsr,+lzcnt,+mmx,+movbe,+pclmul,+popcnt,+rdrnd,+sse,+sse2,+sse3,+sse4.1,+sse4.2,+ssse3,+x87,+xsave,+xsaveopt,-adx,-avx512bitalg,-avx512bw,-avx512cd,-avx512dq,-avx512er,-avx512f,-avx512ifma,-avx512pf,-avx512vbmi,-avx512vbmi2,-avx512vl,-avx512vnni,-avx512vpopcntdq,-clflushopt,-clwb,-clzero,-fma4,-gfni,-ibt,-lwp,-mwaitx,-pku,-prefetchwt1,-prfchw,-rdseed,-rtm,-sgx,-sha,-shstk,-sse4a,-tbm,-vaes,-vpclmulqdq,-xop,-xsavec,-xsaves" "unsafe-fp-math"="true" "use-soft-float"="false" }
attributes #12 = { nobuiltin nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="haswell" "target-features"="+aes,+avx,+avx2,+bmi,+bmi2,+cmov,+cx16,+f16c,+fma,+fsgsbase,+fxsr,+lzcnt,+mmx,+movbe,+pclmul,+popcnt,+rdrnd,+sse,+sse2,+sse3,+sse4.1,+sse4.2,+ssse3,+x87,+xsave,+xsaveopt,-adx,-avx512bitalg,-avx512bw,-avx512cd,-avx512dq,-avx512er,-avx512f,-avx512ifma,-avx512pf,-avx512vbmi,-avx512vbmi2,-avx512vl,-avx512vnni,-avx512vpopcntdq,-clflushopt,-clwb,-clzero,-fma4,-gfni,-ibt,-lwp,-mwaitx,-pku,-prefetchwt1,-prfchw,-rdseed,-rtm,-sgx,-sha,-shstk,-sse4a,-tbm,-vaes,-vpclmulqdq,-xop,-xsavec,-xsaves" "unsafe-fp-math"="true" "use-soft-float"="false" }
attributes #13 = { argmemonly nounwind readonly "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="haswell" "target-features"="+aes,+avx,+avx2,+bmi,+bmi2,+cmov,+cx16,+f16c,+fma,+fsgsbase,+fxsr,+lzcnt,+mmx,+movbe,+pclmul,+popcnt,+rdrnd,+sse,+sse2,+sse3,+sse4.1,+sse4.2,+ssse3,+x87,+xsave,+xsaveopt,-adx,-avx512bitalg,-avx512bw,-avx512cd,-avx512dq,-avx512er,-avx512f,-avx512ifma,-avx512pf,-avx512vbmi,-avx512vbmi2,-avx512vl,-avx512vnni,-avx512vpopcntdq,-clflushopt,-clwb,-clzero,-fma4,-gfni,-ibt,-lwp,-mwaitx,-pku,-prefetchwt1,-prfchw,-rdseed,-rtm,-sgx,-sha,-shstk,-sse4a,-tbm,-vaes,-vpclmulqdq,-xop,-xsavec,-xsaves" "unsafe-fp-math"="true" "use-soft-float"="false" }
attributes #14 = { nobuiltin "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="haswell" "target-features"="+aes,+avx,+avx2,+bmi,+bmi2,+cmov,+cx16,+f16c,+fma,+fsgsbase,+fxsr,+lzcnt,+mmx,+movbe,+pclmul,+popcnt,+rdrnd,+sse,+sse2,+sse3,+sse4.1,+sse4.2,+ssse3,+x87,+xsave,+xsaveopt,-adx,-avx512bitalg,-avx512bw,-avx512cd,-avx512dq,-avx512er,-avx512f,-avx512ifma,-avx512pf,-avx512vbmi,-avx512vbmi2,-avx512vl,-avx512vnni,-avx512vpopcntdq,-clflushopt,-clwb,-clzero,-fma4,-gfni,-ibt,-lwp,-mwaitx,-pku,-prefetchwt1,-prfchw,-rdseed,-rtm,-sgx,-sha,-shstk,-sse4a,-tbm,-vaes,-vpclmulqdq,-xop,-xsavec,-xsaves" "unsafe-fp-math"="true" "use-soft-float"="false" }
attributes #15 = { nounwind readnone speculatable }
attributes #16 = { nounwind readonly }
attributes #17 = { noreturn nounwind }
attributes #18 = { builtin nounwind }

!llvm.module.flags = !{!0}
!llvm.ident = !{!1}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{!"clang version 6.0.0 (git@github.com:wsmoses/Tapir-Clang.git d5d865dfb510d91f47fc5257febec4f52eb1afcb) (git@github.com:wsmoses/Tapir-LLVM.git 32f6e7798224fa34e525743d0b9bde4a7b6e8f9c)"}
!2 = distinct !{!2, !3}
!3 = !{!"llvm.loop.isvectorized", i32 1}
!4 = distinct !{!4, !5, !3}
!5 = !{!"llvm.loop.unroll.runtime.disable"}
!6 = !{!7, !8, i64 0}
!7 = !{!"_ZTS8_point3dIdE", !8, i64 0, !8, i64 8, !8, i64 16}
!8 = !{!"double", !9, i64 0}
!9 = !{!"omnipotent char", !10, i64 0}
!10 = !{!"Simple C++ TBAA"}
!11 = !{!7, !8, i64 8}
!12 = !{!7, !8, i64 16}
!13 = distinct !{!13, !3}
!14 = distinct !{!14, !5, !3}
!15 = distinct !{!15, !16}
!16 = !{!"llvm.loop.unroll.disable"}
!17 = distinct !{!17, !16}
!18 = !{!8, !8, i64 0}
!19 = !{!20}
!20 = distinct !{!20, !21, !"_ZN8_point3dIdE11offsetPointEid: %agg.result"}
!21 = distinct !{!21, !"_ZN8_point3dIdE11offsetPointEid"}
!22 = !{!9, !9, i64 0}
!23 = distinct !{!23, !24}
!24 = !{!"tapir.loop.spawn.strategy", i32 1}
!25 = !{!26, !26, i64 0}
!26 = !{!"bool", !9, i64 0}
!27 = distinct !{!27, !24}
!28 = !{!29, !29, i64 0}
!29 = !{!"long", !9, i64 0}
!30 = !{!31, !31, i64 0}
!31 = !{!"any pointer", !9, i64 0}
!32 = distinct !{!32, !24}
!33 = !{!34, !29, i64 0}
!34 = !{!"_ZTSN7benchIO5wordsE", !29, i64 0, !31, i64 8, !29, i64 16, !31, i64 24}
!35 = !{!34, !31, i64 8}
!36 = !{!34, !29, i64 16}
!37 = !{!34, !31, i64 24}
!38 = !{!39, !39, i64 0}
!39 = !{!"vtable pointer", !10, i64 0}
!40 = !{!41, !43, i64 32}
!41 = !{!"_ZTSSt8ios_base", !29, i64 8, !29, i64 16, !42, i64 24, !43, i64 28, !43, i64 32, !31, i64 40, !44, i64 48, !9, i64 64, !45, i64 192, !31, i64 200, !46, i64 208}
!42 = !{!"_ZTSSt13_Ios_Fmtflags", !9, i64 0}
!43 = !{!"_ZTSSt12_Ios_Iostate", !9, i64 0}
!44 = !{!"_ZTSNSt8ios_base6_WordsE", !31, i64 0, !29, i64 8}
!45 = !{!"int", !9, i64 0}
!46 = !{!"_ZTSSt6locale", !31, i64 0}
!47 = !{!48, !31, i64 240}
!48 = !{!"_ZTSSt9basic_iosIcSt11char_traitsIcEE", !31, i64 216, !9, i64 224, !26, i64 225, !31, i64 232, !31, i64 240, !31, i64 248, !31, i64 256}
!49 = !{!50, !9, i64 56}
!50 = !{!"_ZTSSt5ctypeIcE", !31, i64 16, !26, i64 24, !31, i64 32, !31, i64 40, !31, i64 48, !9, i64 56, !9, i64 57, !9, i64 313, !9, i64 569}
!51 = distinct !{!51, !24}
!52 = !{!53, !29, i64 8}
!53 = !{!"_ZTSSi", !29, i64 8}
!54 = !{!55, !31, i64 0}
!55 = !{!"_ZTSNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE12_Alloc_hiderE", !31, i64 0}
!56 = !{!57, !31, i64 0}
!57 = !{!"_ZTSNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE", !55, i64 0, !29, i64 8, !9, i64 16}
!58 = !{!57, !29, i64 8}
!59 = !{!60, !45, i64 0}
!60 = !{!"_ZTS9trianglesI8_point2dIdEE", !45, i64 0, !45, i64 4, !31, i64 8, !31, i64 16}
!61 = !{!60, !31, i64 8}
!62 = !{!60, !45, i64 4}
!63 = !{!60, !31, i64 16}
!64 = !{!45, !45, i64 0}
!65 = !{!66, !45, i64 0}
!66 = !{!"_ZTS11commandLine", !45, i64 0, !31, i64 8, !57, i64 16}
!67 = !{!66, !31, i64 8}
!68 = distinct !{!68, !24}
!69 = distinct !{!69, !24}
!70 = !{!71, !45, i64 0}
!71 = !{!"_ZTS6vertexI8_point2dIdELi1EE", !45, i64 0, !72, i64 8, !9, i64 24}
!72 = !{!"_ZTS8_point2dIdE", !8, i64 0, !8, i64 8}
!73 = distinct !{!73, !24}
!74 = distinct !{!74, !24}
!75 = distinct !{!75, !16}
!76 = !{!77, !26, i64 24}
!77 = !{!"_ZTS5timer", !8, i64 0, !8, i64 8, !8, i64 16, !26, i64 24, !78, i64 28}
!78 = !{!"_ZTS8timezone", !45, i64 0, !45, i64 4}
!79 = !{!80, !29, i64 0}
!80 = !{!"_ZTS7timeval", !29, i64 0, !29, i64 8}
!81 = !{!80, !29, i64 8}
!82 = !{!77, !8, i64 8}
!83 = !{i8 0, i8 2}
!84 = !{!77, !8, i64 0}
!85 = !{!86, !45, i64 0}
!86 = !{!"_ZTS6vertexI8_point2dIdELi10EE", !45, i64 0, !72, i64 8, !9, i64 24}
!87 = distinct !{!87, !24}
!88 = distinct !{!88, !24}
!89 = distinct !{!89, !16}
!90 = distinct !{!90, !24}
!91 = distinct !{!91, !24}
!92 = !{!93, !45, i64 0}
!93 = !{!"_ZTS6vertexI8_point3dIdELi1EE", !45, i64 0, !7, i64 8, !9, i64 32}
!94 = distinct !{!94, !24}
!95 = distinct !{!95, !24}
!96 = distinct !{!96, !16}
!97 = !{!98, !45, i64 0}
!98 = !{!"_ZTS6vertexI8_point3dIdELi10EE", !45, i64 0, !7, i64 8, !9, i64 32}
!99 = distinct !{!99, !24}
!100 = distinct !{!100, !24}
!101 = distinct !{!101, !16}
!102 = distinct !{!102, !3}
!103 = distinct !{!103, !16}
!104 = distinct !{!104, !5, !3}
!105 = distinct !{!105, !3}
!106 = distinct !{!106, !16}
!107 = distinct !{!107, !5, !3}
!108 = distinct !{!108, !24}
!109 = distinct !{!109, !3}
!110 = distinct !{!110, !16}
!111 = distinct !{!111, !5, !3}
!112 = distinct !{!112, !16}
!113 = distinct !{!113, !24}
!114 = distinct !{!114, !16}
!115 = distinct !{!115, !16}
!116 = distinct !{!116, !16}
!117 = distinct !{!117, !16}
!118 = distinct !{!118, !16}
!119 = distinct !{!119, !3}
!120 = distinct !{!120, !16}
!121 = distinct !{!121, !5, !3}
!122 = distinct !{!122, !24}
!123 = distinct !{!123, !16}
!124 = distinct !{!124, !16}
!125 = distinct !{!125, !24}
!126 = distinct !{!126, !16}
!127 = distinct !{!127, !16}
!128 = !{!129, !45, i64 28}
!129 = !{!"_ZTS9gTreeNodeI8_point2dIdE7_vect2dIdE6vertexIS1_Li1EE5nDataIS5_EE", !72, i64 0, !8, i64 16, !130, i64 24, !45, i64 28, !9, i64 32, !31, i64 96, !31, i64 104}
!130 = !{!"_ZTS5nDataI6vertexI8_point2dIdELi1EEE", !45, i64 0}
!131 = !{!132, !45, i64 28}
!132 = !{!"_ZTSN16kNearestNeighborI6vertexI8_point2dIdELi1EELi1EE3kNNE", !31, i64 0, !9, i64 8, !9, i64 16, !45, i64 24, !45, i64 28}
!133 = !{!132, !45, i64 24}
!134 = !{!132, !31, i64 0}
!135 = !{!136}
!136 = distinct !{!136, !137}
!137 = distinct !{!137, !"LVerDomain"}
!138 = !{!139}
!139 = distinct !{!139, !137}
!140 = distinct !{!140, !3}
!141 = distinct !{!141, !16}
!142 = distinct !{!142, !3}
!143 = distinct !{!143, !24, !144}
!144 = !{!"tapir.loop.grainsize", i32 1}
!145 = !{!41, !29, i64 8}
!146 = distinct !{!146, !24}
!147 = !{!130, !45, i64 0}
!148 = !{!129, !8, i64 16}
!149 = distinct !{!149, !16}
!150 = !{!129, !31, i64 104}
!151 = !{!129, !31, i64 96}
!152 = !{!72, !8, i64 0}
!153 = !{!72, !8, i64 8}
!154 = !{!155, !157}
!155 = distinct !{!155, !156, !"_ZN8sequence5getAFIP6vertexI8_point2dIdELi1EESt4pairIS3_S3_EiN9gTreeNodeIS3_7_vect2dIdES4_5nDataIS4_EE6toPairEEclEi: %agg.result"}
!156 = distinct !{!156, !"_ZN8sequence5getAFIP6vertexI8_point2dIdELi1EESt4pairIS3_S3_EiN9gTreeNodeIS3_7_vect2dIdES4_5nDataIS4_EE6toPairEEclEi"}
!157 = distinct !{!157, !158, !"_ZN8sequence12reduceSerialISt4pairI8_point2dIdES3_EiN9gTreeNodeIS3_7_vect2dIdE6vertexIS3_Li1EE5nDataIS9_EE6minMaxENS_5getAFIPS9_S4_iNSC_6toPairEEEEET_T0_SJ_T1_T2_: %agg.result"}
!158 = distinct !{!158, !"_ZN8sequence12reduceSerialISt4pairI8_point2dIdES3_EiN9gTreeNodeIS3_7_vect2dIdE6vertexIS3_Li1EE5nDataIS9_EE6minMaxENS_5getAFIPS9_S4_iNSC_6toPairEEEEET_T0_SJ_T1_T2_"}
!159 = !{i64 0, i64 8, !18, i64 8, i64 8, !18}
!160 = !{!157}
!161 = !{!162, !157}
!162 = distinct !{!162, !163, !"_ZN8sequence5getAFIP6vertexI8_point2dIdELi1EESt4pairIS3_S3_EiN9gTreeNodeIS3_7_vect2dIdES4_5nDataIS4_EE6toPairEEclEi: %agg.result"}
!163 = distinct !{!163, !"_ZN8sequence5getAFIP6vertexI8_point2dIdELi1EESt4pairIS3_S3_EiN9gTreeNodeIS3_7_vect2dIdES4_5nDataIS4_EE6toPairEEclEi"}
!164 = distinct !{!164, !16}
!165 = !{!166, !168}
!166 = distinct !{!166, !167, !"_ZN8sequence5getAFIP6vertexI8_point2dIdELi1EESt4pairIS3_S3_EiN9gTreeNodeIS3_7_vect2dIdES4_5nDataIS4_EE6toPairEEclEi: %agg.result"}
!167 = distinct !{!167, !"_ZN8sequence5getAFIP6vertexI8_point2dIdELi1EESt4pairIS3_S3_EiN9gTreeNodeIS3_7_vect2dIdES4_5nDataIS4_EE6toPairEEclEi"}
!168 = distinct !{!168, !169, !"_ZN8sequence12reduceSerialISt4pairI8_point2dIdES3_EiN9gTreeNodeIS3_7_vect2dIdE6vertexIS3_Li1EE5nDataIS9_EE6minMaxENS_5getAFIPS9_S4_iNSC_6toPairEEEEET_T0_SJ_T1_T2_: %agg.result"}
!169 = distinct !{!169, !"_ZN8sequence12reduceSerialISt4pairI8_point2dIdES3_EiN9gTreeNodeIS3_7_vect2dIdE6vertexIS3_Li1EE5nDataIS9_EE6minMaxENS_5getAFIPS9_S4_iNSC_6toPairEEEEET_T0_SJ_T1_T2_"}
!170 = !{!171, !168}
!171 = distinct !{!171, !172, !"_ZN8sequence5getAFIP6vertexI8_point2dIdELi1EESt4pairIS3_S3_EiN9gTreeNodeIS3_7_vect2dIdES4_5nDataIS4_EE6toPairEEclEi: %agg.result"}
!172 = distinct !{!172, !"_ZN8sequence5getAFIP6vertexI8_point2dIdELi1EESt4pairIS3_S3_EiN9gTreeNodeIS3_7_vect2dIdES4_5nDataIS4_EE6toPairEEclEi"}
!173 = !{!168}
!174 = distinct !{!174, !16}
!175 = distinct !{!175, !24}
!176 = !{i64 0, i64 8, !18, i64 8, i64 8, !18, i64 16, i64 8, !18, i64 24, i64 8, !18}
!177 = !{!178}
!178 = distinct !{!178, !179, !"_ZN8sequence12reduceSerialISt4pairI8_point2dIdES3_EiN9gTreeNodeIS3_7_vect2dIdE6vertexIS3_Li1EE5nDataIS9_EE6minMaxENS_4getAIS4_iEEEET_T0_SH_T1_T2_: %agg.result"}
!179 = distinct !{!179, !"_ZN8sequence12reduceSerialISt4pairI8_point2dIdES3_EiN9gTreeNodeIS3_7_vect2dIdE6vertexIS3_Li1EE5nDataIS9_EE6minMaxENS_4getAIS4_iEEEET_T0_SH_T1_T2_"}
!180 = distinct !{!180, !3}
!181 = distinct !{!181, !5, !3}
!182 = !{!183}
!183 = distinct !{!183, !184, !"_ZN8sequence12reduceSerialISt4pairI8_point2dIdES3_EiN9gTreeNodeIS3_7_vect2dIdE6vertexIS3_Li1EE5nDataIS9_EE6minMaxENS_4getAIS4_iEEEET_T0_SH_T1_T2_: %agg.result"}
!184 = distinct !{!184, !"_ZN8sequence12reduceSerialISt4pairI8_point2dIdES3_EiN9gTreeNodeIS3_7_vect2dIdE6vertexIS3_Li1EE5nDataIS9_EE6minMaxENS_4getAIS4_iEEEET_T0_SH_T1_T2_"}
!185 = distinct !{!185, !3}
!186 = distinct !{!186, !5, !3}
!187 = distinct !{!187, !24}
!188 = distinct !{!188, !3}
!189 = distinct !{!189, !5, !3}
!190 = distinct !{!190, !24}
!191 = !{!192, !45, i64 0}
!192 = !{!"_ZTSSt4pairIiP6vertexI8_point2dIdELi1EEE", !45, i64 0, !31, i64 8}
!193 = !{!192, !31, i64 8}
!194 = distinct !{!194, !24}
!195 = distinct !{!195, !3}
!196 = distinct !{!196, !16}
!197 = distinct !{!197, !3}
!198 = distinct !{!198, !5, !3}
!199 = distinct !{!199, !5, !3}
!200 = !{!201, !29, i64 8}
!201 = !{!"_ZTSN7intSort5eBitsISt4pairIiP6vertexI8_point2dIdELi1EEEN5utils6firstFIiS6_EEEE", !202, i64 0, !29, i64 8, !29, i64 16}
!202 = !{!"_ZTSN5utils6firstFIiP6vertexI8_point2dIdELi1EEEE"}
!203 = !{!201, !29, i64 16}
!204 = distinct !{!204, !24}
!205 = distinct !{!205, !24}
!206 = distinct !{!206, !24}
!207 = distinct !{!207, !16}
!208 = distinct !{!208, !24, !144}
!209 = distinct !{!209, !16}
!210 = !{!211, !31, i64 0}
!211 = !{!"_ZTS9transposeIjjE", !31, i64 0, !31, i64 8}
!212 = !{!211, !31, i64 8}
!213 = distinct !{!213, !16}
!214 = !{!215, !31, i64 0}
!215 = !{!"_ZTS10blockTransISt4pairIiP6vertexI8_point2dIdELi1EEEjE", !31, i64 0, !31, i64 8, !31, i64 16, !31, i64 24, !31, i64 32}
!216 = !{!215, !31, i64 8}
!217 = !{!215, !31, i64 16}
!218 = !{!215, !31, i64 24}
!219 = !{!215, !31, i64 32}
!220 = distinct !{!220, !16}
!221 = distinct !{!221, !16}
!222 = distinct !{!222, !24}
!223 = !{!224}
!224 = distinct !{!224, !225}
!225 = distinct !{!225, !"LVerDomain"}
!226 = !{!227}
!227 = distinct !{!227, !225}
!228 = distinct !{!228, !3}
!229 = distinct !{!229, !16}
!230 = distinct !{!230, !3}
!231 = distinct !{!231, !16}
!232 = distinct !{!232, !16}
!233 = distinct !{!233, !16}
!234 = distinct !{!234, !16}
!235 = distinct !{!235, !3}
!236 = distinct !{!236, !16}
!237 = distinct !{!237, !5, !3}
!238 = distinct !{!238, !24}
!239 = distinct !{!239, !16}
!240 = distinct !{!240, !16}
!241 = distinct !{!241, !24}
!242 = distinct !{!242, !16}
!243 = distinct !{!243, !16}
!244 = distinct !{!244, !16}
!245 = distinct !{!245, !16}
!246 = distinct !{!246, !16}
!247 = distinct !{!247, !16}
!248 = distinct !{!248, !16}
!249 = distinct !{!249, !3}
!250 = distinct !{!250, !5, !3}
!251 = distinct !{!251, !24}
!252 = distinct !{!252, !16}
!253 = distinct !{!253, !16}
!254 = distinct !{!254, !24}
!255 = distinct !{!255, !16}
!256 = distinct !{!256, !16}
!257 = distinct !{!257, !24}
!258 = distinct !{!258, !24}
!259 = distinct !{!259, !24}
!260 = distinct !{!260, !24}
!261 = distinct !{!261, !16}
!262 = distinct !{!262, !3}
!263 = distinct !{!263, !16}
!264 = distinct !{!264, !5, !3}
!265 = distinct !{!265, !3}
!266 = distinct !{!266, !16}
!267 = distinct !{!267, !5, !3}
!268 = distinct !{!268, !24}
!269 = distinct !{!269, !3}
!270 = distinct !{!270, !16}
!271 = distinct !{!271, !5, !3}
!272 = distinct !{!272, !16}
!273 = distinct !{!273, !24}
!274 = distinct !{!274, !16}
!275 = !{!276, !31, i64 0}
!276 = !{!"_ZTS16kNearestNeighborI6vertexI8_point2dIdELi10EELi10EE", !31, i64 0}
!277 = !{!278, !45, i64 28}
!278 = !{!"_ZTS9gTreeNodeI8_point2dIdE7_vect2dIdE6vertexIS1_Li10EE5nDataIS5_EE", !72, i64 0, !8, i64 16, !279, i64 24, !45, i64 28, !9, i64 32, !31, i64 96, !31, i64 104}
!279 = !{!"_ZTS5nDataI6vertexI8_point2dIdELi10EEE", !45, i64 0}
!280 = distinct !{!280, !24, !144}
!281 = !{!282, !45, i64 172}
!282 = !{!"_ZTSN16kNearestNeighborI6vertexI8_point2dIdELi10EELi10EE3kNNE", !31, i64 0, !9, i64 8, !9, i64 88, !45, i64 168, !45, i64 172}
!283 = !{!282, !45, i64 168}
!284 = !{!282, !31, i64 0}
!285 = distinct !{!285, !3}
!286 = distinct !{!286, !16}
!287 = distinct !{!287, !5, !3}
!288 = distinct !{!288, !16}
!289 = distinct !{!289, !3}
!290 = distinct !{!290, !3}
!291 = distinct !{!291, !24}
!292 = !{!279, !45, i64 0}
!293 = !{!278, !8, i64 16}
!294 = distinct !{!294, !16}
!295 = !{!278, !31, i64 104}
!296 = !{!278, !31, i64 96}
!297 = !{!298, !300}
!298 = distinct !{!298, !299, !"_ZN8sequence5getAFIP6vertexI8_point2dIdELi10EESt4pairIS3_S3_EiN9gTreeNodeIS3_7_vect2dIdES4_5nDataIS4_EE6toPairEEclEi: %agg.result"}
!299 = distinct !{!299, !"_ZN8sequence5getAFIP6vertexI8_point2dIdELi10EESt4pairIS3_S3_EiN9gTreeNodeIS3_7_vect2dIdES4_5nDataIS4_EE6toPairEEclEi"}
!300 = distinct !{!300, !301, !"_ZN8sequence12reduceSerialISt4pairI8_point2dIdES3_EiN9gTreeNodeIS3_7_vect2dIdE6vertexIS3_Li10EE5nDataIS9_EE6minMaxENS_5getAFIPS9_S4_iNSC_6toPairEEEEET_T0_SJ_T1_T2_: %agg.result"}
!301 = distinct !{!301, !"_ZN8sequence12reduceSerialISt4pairI8_point2dIdES3_EiN9gTreeNodeIS3_7_vect2dIdE6vertexIS3_Li10EE5nDataIS9_EE6minMaxENS_5getAFIPS9_S4_iNSC_6toPairEEEEET_T0_SJ_T1_T2_"}
!302 = !{!300}
!303 = !{!304, !300}
!304 = distinct !{!304, !305, !"_ZN8sequence5getAFIP6vertexI8_point2dIdELi10EESt4pairIS3_S3_EiN9gTreeNodeIS3_7_vect2dIdES4_5nDataIS4_EE6toPairEEclEi: %agg.result"}
!305 = distinct !{!305, !"_ZN8sequence5getAFIP6vertexI8_point2dIdELi10EESt4pairIS3_S3_EiN9gTreeNodeIS3_7_vect2dIdES4_5nDataIS4_EE6toPairEEclEi"}
!306 = distinct !{!306, !16}
!307 = !{!308, !310}
!308 = distinct !{!308, !309, !"_ZN8sequence5getAFIP6vertexI8_point2dIdELi10EESt4pairIS3_S3_EiN9gTreeNodeIS3_7_vect2dIdES4_5nDataIS4_EE6toPairEEclEi: %agg.result"}
!309 = distinct !{!309, !"_ZN8sequence5getAFIP6vertexI8_point2dIdELi10EESt4pairIS3_S3_EiN9gTreeNodeIS3_7_vect2dIdES4_5nDataIS4_EE6toPairEEclEi"}
!310 = distinct !{!310, !311, !"_ZN8sequence12reduceSerialISt4pairI8_point2dIdES3_EiN9gTreeNodeIS3_7_vect2dIdE6vertexIS3_Li10EE5nDataIS9_EE6minMaxENS_5getAFIPS9_S4_iNSC_6toPairEEEEET_T0_SJ_T1_T2_: %agg.result"}
!311 = distinct !{!311, !"_ZN8sequence12reduceSerialISt4pairI8_point2dIdES3_EiN9gTreeNodeIS3_7_vect2dIdE6vertexIS3_Li10EE5nDataIS9_EE6minMaxENS_5getAFIPS9_S4_iNSC_6toPairEEEEET_T0_SJ_T1_T2_"}
!312 = !{!313, !310}
!313 = distinct !{!313, !314, !"_ZN8sequence5getAFIP6vertexI8_point2dIdELi10EESt4pairIS3_S3_EiN9gTreeNodeIS3_7_vect2dIdES4_5nDataIS4_EE6toPairEEclEi: %agg.result"}
!314 = distinct !{!314, !"_ZN8sequence5getAFIP6vertexI8_point2dIdELi10EESt4pairIS3_S3_EiN9gTreeNodeIS3_7_vect2dIdES4_5nDataIS4_EE6toPairEEclEi"}
!315 = !{!310}
!316 = distinct !{!316, !16}
!317 = distinct !{!317, !24}
!318 = !{!319}
!319 = distinct !{!319, !320, !"_ZN8sequence12reduceSerialISt4pairI8_point2dIdES3_EiN9gTreeNodeIS3_7_vect2dIdE6vertexIS3_Li10EE5nDataIS9_EE6minMaxENS_4getAIS4_iEEEET_T0_SH_T1_T2_: %agg.result"}
!320 = distinct !{!320, !"_ZN8sequence12reduceSerialISt4pairI8_point2dIdES3_EiN9gTreeNodeIS3_7_vect2dIdE6vertexIS3_Li10EE5nDataIS9_EE6minMaxENS_4getAIS4_iEEEET_T0_SH_T1_T2_"}
!321 = distinct !{!321, !3}
!322 = distinct !{!322, !5, !3}
!323 = !{!324}
!324 = distinct !{!324, !325, !"_ZN8sequence12reduceSerialISt4pairI8_point2dIdES3_EiN9gTreeNodeIS3_7_vect2dIdE6vertexIS3_Li10EE5nDataIS9_EE6minMaxENS_4getAIS4_iEEEET_T0_SH_T1_T2_: %agg.result"}
!325 = distinct !{!325, !"_ZN8sequence12reduceSerialISt4pairI8_point2dIdES3_EiN9gTreeNodeIS3_7_vect2dIdE6vertexIS3_Li10EE5nDataIS9_EE6minMaxENS_4getAIS4_iEEEET_T0_SH_T1_T2_"}
!326 = distinct !{!326, !3}
!327 = distinct !{!327, !5, !3}
!328 = distinct !{!328, !24}
!329 = distinct !{!329, !3}
!330 = distinct !{!330, !5, !3}
!331 = distinct !{!331, !24}
!332 = !{!333, !45, i64 0}
!333 = !{!"_ZTSSt4pairIiP6vertexI8_point2dIdELi10EEE", !45, i64 0, !31, i64 8}
!334 = !{!333, !31, i64 8}
!335 = distinct !{!335, !24}
!336 = distinct !{!336, !3}
!337 = distinct !{!337, !16}
!338 = distinct !{!338, !3}
!339 = distinct !{!339, !5, !3}
!340 = distinct !{!340, !5, !3}
!341 = !{!342, !29, i64 8}
!342 = !{!"_ZTSN7intSort5eBitsISt4pairIiP6vertexI8_point2dIdELi10EEEN5utils6firstFIiS6_EEEE", !343, i64 0, !29, i64 8, !29, i64 16}
!343 = !{!"_ZTSN5utils6firstFIiP6vertexI8_point2dIdELi10EEEE"}
!344 = !{!342, !29, i64 16}
!345 = distinct !{!345, !24}
!346 = distinct !{!346, !24}
!347 = distinct !{!347, !24}
!348 = distinct !{!348, !16}
!349 = distinct !{!349, !24, !144}
!350 = distinct !{!350, !16}
!351 = distinct !{!351, !16}
!352 = !{!353, !31, i64 0}
!353 = !{!"_ZTS10blockTransISt4pairIiP6vertexI8_point2dIdELi10EEEjE", !31, i64 0, !31, i64 8, !31, i64 16, !31, i64 24, !31, i64 32}
!354 = !{!353, !31, i64 8}
!355 = !{!353, !31, i64 16}
!356 = !{!353, !31, i64 24}
!357 = !{!353, !31, i64 32}
!358 = distinct !{!358, !16}
!359 = distinct !{!359, !16}
!360 = distinct !{!360, !24}
!361 = distinct !{!361, !16}
!362 = !{!363, !45, i64 36}
!363 = !{!"_ZTS9gTreeNodeI8_point3dIdE7_vect3dIdE6vertexIS1_Li1EE5nDataIS5_EE", !7, i64 0, !8, i64 24, !364, i64 32, !45, i64 36, !9, i64 40, !31, i64 104, !31, i64 112}
!364 = !{!"_ZTS5nDataI6vertexI8_point3dIdELi1EEE", !45, i64 0}
!365 = !{!366, !45, i64 28}
!366 = !{!"_ZTSN16kNearestNeighborI6vertexI8_point3dIdELi1EELi1EE3kNNE", !31, i64 0, !9, i64 8, !9, i64 16, !45, i64 24, !45, i64 28}
!367 = !{!366, !45, i64 24}
!368 = !{!366, !31, i64 0}
!369 = !{!370}
!370 = distinct !{!370, !371}
!371 = distinct !{!371, !"LVerDomain"}
!372 = !{!373}
!373 = distinct !{!373, !371}
!374 = distinct !{!374, !3}
!375 = distinct !{!375, !16}
!376 = distinct !{!376, !3}
!377 = distinct !{!377, !24, !144}
!378 = distinct !{!378, !24}
!379 = !{!364, !45, i64 0}
!380 = !{!363, !8, i64 24}
!381 = !{i64 0, i64 8, !18, i64 8, i64 8, !18, i64 16, i64 8, !18}
!382 = distinct !{!382, !16}
!383 = !{!363, !31, i64 112}
!384 = !{!385}
!385 = distinct !{!385, !386, !"_ZN8_point3dIdE11offsetPointEid: %agg.result"}
!386 = distinct !{!386, !"_ZN8_point3dIdE11offsetPointEid"}
!387 = !{!363, !31, i64 104}
!388 = !{!389, !391}
!389 = distinct !{!389, !390, !"_ZN8sequence5getAFIP6vertexI8_point3dIdELi1EESt4pairIS3_S3_EiN9gTreeNodeIS3_7_vect3dIdES4_5nDataIS4_EE6toPairEEclEi: %agg.result"}
!390 = distinct !{!390, !"_ZN8sequence5getAFIP6vertexI8_point3dIdELi1EESt4pairIS3_S3_EiN9gTreeNodeIS3_7_vect3dIdES4_5nDataIS4_EE6toPairEEclEi"}
!391 = distinct !{!391, !392, !"_ZN8sequence12reduceSerialISt4pairI8_point3dIdES3_EiN9gTreeNodeIS3_7_vect3dIdE6vertexIS3_Li1EE5nDataIS9_EE6minMaxENS_5getAFIPS9_S4_iNSC_6toPairEEEEET_T0_SJ_T1_T2_: %agg.result"}
!392 = distinct !{!392, !"_ZN8sequence12reduceSerialISt4pairI8_point3dIdES3_EiN9gTreeNodeIS3_7_vect3dIdE6vertexIS3_Li1EE5nDataIS9_EE6minMaxENS_5getAFIPS9_S4_iNSC_6toPairEEEEET_T0_SJ_T1_T2_"}
!393 = !{!391}
!394 = !{!395, !391}
!395 = distinct !{!395, !396, !"_ZN8sequence5getAFIP6vertexI8_point3dIdELi1EESt4pairIS3_S3_EiN9gTreeNodeIS3_7_vect3dIdES4_5nDataIS4_EE6toPairEEclEi: %agg.result"}
!396 = distinct !{!396, !"_ZN8sequence5getAFIP6vertexI8_point3dIdELi1EESt4pairIS3_S3_EiN9gTreeNodeIS3_7_vect3dIdES4_5nDataIS4_EE6toPairEEclEi"}
!397 = distinct !{!397, !16}
!398 = !{!399, !401}
!399 = distinct !{!399, !400, !"_ZN8sequence5getAFIP6vertexI8_point3dIdELi1EESt4pairIS3_S3_EiN9gTreeNodeIS3_7_vect3dIdES4_5nDataIS4_EE6toPairEEclEi: %agg.result"}
!400 = distinct !{!400, !"_ZN8sequence5getAFIP6vertexI8_point3dIdELi1EESt4pairIS3_S3_EiN9gTreeNodeIS3_7_vect3dIdES4_5nDataIS4_EE6toPairEEclEi"}
!401 = distinct !{!401, !402, !"_ZN8sequence12reduceSerialISt4pairI8_point3dIdES3_EiN9gTreeNodeIS3_7_vect3dIdE6vertexIS3_Li1EE5nDataIS9_EE6minMaxENS_5getAFIPS9_S4_iNSC_6toPairEEEEET_T0_SJ_T1_T2_: %agg.result"}
!402 = distinct !{!402, !"_ZN8sequence12reduceSerialISt4pairI8_point3dIdES3_EiN9gTreeNodeIS3_7_vect3dIdE6vertexIS3_Li1EE5nDataIS9_EE6minMaxENS_5getAFIPS9_S4_iNSC_6toPairEEEEET_T0_SJ_T1_T2_"}
!403 = !{!404, !401}
!404 = distinct !{!404, !405, !"_ZN8sequence5getAFIP6vertexI8_point3dIdELi1EESt4pairIS3_S3_EiN9gTreeNodeIS3_7_vect3dIdES4_5nDataIS4_EE6toPairEEclEi: %agg.result"}
!405 = distinct !{!405, !"_ZN8sequence5getAFIP6vertexI8_point3dIdELi1EESt4pairIS3_S3_EiN9gTreeNodeIS3_7_vect3dIdES4_5nDataIS4_EE6toPairEEclEi"}
!406 = !{!401}
!407 = distinct !{!407, !16}
!408 = distinct !{!408, !24}
!409 = !{i64 0, i64 8, !18, i64 8, i64 8, !18, i64 16, i64 8, !18, i64 24, i64 8, !18, i64 32, i64 8, !18, i64 40, i64 8, !18}
!410 = !{!411}
!411 = distinct !{!411, !412, !"_ZN8sequence12reduceSerialISt4pairI8_point3dIdES3_EiN9gTreeNodeIS3_7_vect3dIdE6vertexIS3_Li1EE5nDataIS9_EE6minMaxENS_4getAIS4_iEEEET_T0_SH_T1_T2_: %agg.result"}
!412 = distinct !{!412, !"_ZN8sequence12reduceSerialISt4pairI8_point3dIdES3_EiN9gTreeNodeIS3_7_vect3dIdE6vertexIS3_Li1EE5nDataIS9_EE6minMaxENS_4getAIS4_iEEEET_T0_SH_T1_T2_"}
!413 = !{!414}
!414 = distinct !{!414, !415, !"_ZN8sequence12reduceSerialISt4pairI8_point3dIdES3_EiN9gTreeNodeIS3_7_vect3dIdE6vertexIS3_Li1EE5nDataIS9_EE6minMaxENS_4getAIS4_iEEEET_T0_SH_T1_T2_: %agg.result"}
!415 = distinct !{!415, !"_ZN8sequence12reduceSerialISt4pairI8_point3dIdES3_EiN9gTreeNodeIS3_7_vect3dIdE6vertexIS3_Li1EE5nDataIS9_EE6minMaxENS_4getAIS4_iEEEET_T0_SH_T1_T2_"}
!416 = distinct !{!416, !24}
!417 = !{!418}
!418 = distinct !{!418, !419, !"_ZN8_point3dIdE11offsetPointEid: %agg.result"}
!419 = distinct !{!419, !"_ZN8_point3dIdE11offsetPointEid"}
!420 = distinct !{!420, !3}
!421 = distinct !{!421, !5, !3}
!422 = distinct !{!422, !24}
!423 = !{!424, !45, i64 0}
!424 = !{!"_ZTSSt4pairIiP6vertexI8_point3dIdELi1EEE", !45, i64 0, !31, i64 8}
!425 = !{!424, !31, i64 8}
!426 = distinct !{!426, !24}
!427 = !{!428}
!428 = distinct !{!428, !429, !"_ZN8_point3dIdE11offsetPointEid: %agg.result"}
!429 = distinct !{!429, !"_ZN8_point3dIdE11offsetPointEid"}
!430 = !{!431}
!431 = distinct !{!431, !432, !"_ZN8_point3dIdE11offsetPointEid: %agg.result"}
!432 = distinct !{!432, !"_ZN8_point3dIdE11offsetPointEid"}
!433 = distinct !{!433, !3}
!434 = distinct !{!434, !16}
!435 = distinct !{!435, !3}
!436 = distinct !{!436, !5, !3}
!437 = distinct !{!437, !5, !3}
!438 = !{!439, !29, i64 8}
!439 = !{!"_ZTSN7intSort5eBitsISt4pairIiP6vertexI8_point3dIdELi1EEEN5utils6firstFIiS6_EEEE", !440, i64 0, !29, i64 8, !29, i64 16}
!440 = !{!"_ZTSN5utils6firstFIiP6vertexI8_point3dIdELi1EEEE"}
!441 = !{!439, !29, i64 16}
!442 = distinct !{!442, !24}
!443 = distinct !{!443, !24}
!444 = distinct !{!444, !24}
!445 = distinct !{!445, !16}
!446 = distinct !{!446, !24, !144}
!447 = distinct !{!447, !16}
!448 = distinct !{!448, !16}
!449 = !{!450, !31, i64 0}
!450 = !{!"_ZTS10blockTransISt4pairIiP6vertexI8_point3dIdELi1EEEjE", !31, i64 0, !31, i64 8, !31, i64 16, !31, i64 24, !31, i64 32}
!451 = !{!450, !31, i64 8}
!452 = !{!450, !31, i64 16}
!453 = !{!450, !31, i64 24}
!454 = !{!450, !31, i64 32}
!455 = distinct !{!455, !16}
!456 = distinct !{!456, !16}
!457 = distinct !{!457, !24}
!458 = distinct !{!458, !16}
!459 = !{!460}
!460 = distinct !{!460, !461, !"_ZN8_point3dIdEmiES0_: %agg.result"}
!461 = distinct !{!461, !"_ZN8_point3dIdEmiES0_"}
!462 = !{!463}
!463 = distinct !{!463, !464, !"_ZN8_point3dIdEmiES0_: %agg.result"}
!464 = distinct !{!464, !"_ZN8_point3dIdEmiES0_"}
!465 = !{!466, !31, i64 0}
!466 = !{!"_ZTS16kNearestNeighborI6vertexI8_point3dIdELi10EELi10EE", !31, i64 0}
!467 = !{!468, !45, i64 36}
!468 = !{!"_ZTS9gTreeNodeI8_point3dIdE7_vect3dIdE6vertexIS1_Li10EE5nDataIS5_EE", !7, i64 0, !8, i64 24, !469, i64 32, !45, i64 36, !9, i64 40, !31, i64 104, !31, i64 112}
!469 = !{!"_ZTS5nDataI6vertexI8_point3dIdELi10EEE", !45, i64 0}
!470 = distinct !{!470, !24, !144}
!471 = !{!472, !45, i64 172}
!472 = !{!"_ZTSN16kNearestNeighborI6vertexI8_point3dIdELi10EELi10EE3kNNE", !31, i64 0, !9, i64 8, !9, i64 88, !45, i64 168, !45, i64 172}
!473 = !{!472, !45, i64 168}
!474 = !{!472, !31, i64 0}
!475 = distinct !{!475, !3}
!476 = distinct !{!476, !16}
!477 = distinct !{!477, !5, !3}
!478 = distinct !{!478, !16}
!479 = distinct !{!479, !3}
!480 = distinct !{!480, !3}
!481 = distinct !{!481, !24}
!482 = !{!469, !45, i64 0}
!483 = !{!468, !8, i64 24}
!484 = distinct !{!484, !16}
!485 = !{!468, !31, i64 112}
!486 = !{!487}
!487 = distinct !{!487, !488, !"_ZN8_point3dIdE11offsetPointEid: %agg.result"}
!488 = distinct !{!488, !"_ZN8_point3dIdE11offsetPointEid"}
!489 = !{!468, !31, i64 104}
!490 = !{!491, !493}
!491 = distinct !{!491, !492, !"_ZN8sequence5getAFIP6vertexI8_point3dIdELi10EESt4pairIS3_S3_EiN9gTreeNodeIS3_7_vect3dIdES4_5nDataIS4_EE6toPairEEclEi: %agg.result"}
!492 = distinct !{!492, !"_ZN8sequence5getAFIP6vertexI8_point3dIdELi10EESt4pairIS3_S3_EiN9gTreeNodeIS3_7_vect3dIdES4_5nDataIS4_EE6toPairEEclEi"}
!493 = distinct !{!493, !494, !"_ZN8sequence12reduceSerialISt4pairI8_point3dIdES3_EiN9gTreeNodeIS3_7_vect3dIdE6vertexIS3_Li10EE5nDataIS9_EE6minMaxENS_5getAFIPS9_S4_iNSC_6toPairEEEEET_T0_SJ_T1_T2_: %agg.result"}
!494 = distinct !{!494, !"_ZN8sequence12reduceSerialISt4pairI8_point3dIdES3_EiN9gTreeNodeIS3_7_vect3dIdE6vertexIS3_Li10EE5nDataIS9_EE6minMaxENS_5getAFIPS9_S4_iNSC_6toPairEEEEET_T0_SJ_T1_T2_"}
!495 = !{!493}
!496 = !{!497, !493}
!497 = distinct !{!497, !498, !"_ZN8sequence5getAFIP6vertexI8_point3dIdELi10EESt4pairIS3_S3_EiN9gTreeNodeIS3_7_vect3dIdES4_5nDataIS4_EE6toPairEEclEi: %agg.result"}
!498 = distinct !{!498, !"_ZN8sequence5getAFIP6vertexI8_point3dIdELi10EESt4pairIS3_S3_EiN9gTreeNodeIS3_7_vect3dIdES4_5nDataIS4_EE6toPairEEclEi"}
!499 = distinct !{!499, !16}
!500 = !{!501, !503}
!501 = distinct !{!501, !502, !"_ZN8sequence5getAFIP6vertexI8_point3dIdELi10EESt4pairIS3_S3_EiN9gTreeNodeIS3_7_vect3dIdES4_5nDataIS4_EE6toPairEEclEi: %agg.result"}
!502 = distinct !{!502, !"_ZN8sequence5getAFIP6vertexI8_point3dIdELi10EESt4pairIS3_S3_EiN9gTreeNodeIS3_7_vect3dIdES4_5nDataIS4_EE6toPairEEclEi"}
!503 = distinct !{!503, !504, !"_ZN8sequence12reduceSerialISt4pairI8_point3dIdES3_EiN9gTreeNodeIS3_7_vect3dIdE6vertexIS3_Li10EE5nDataIS9_EE6minMaxENS_5getAFIPS9_S4_iNSC_6toPairEEEEET_T0_SJ_T1_T2_: %agg.result"}
!504 = distinct !{!504, !"_ZN8sequence12reduceSerialISt4pairI8_point3dIdES3_EiN9gTreeNodeIS3_7_vect3dIdE6vertexIS3_Li10EE5nDataIS9_EE6minMaxENS_5getAFIPS9_S4_iNSC_6toPairEEEEET_T0_SJ_T1_T2_"}
!505 = !{!506, !503}
!506 = distinct !{!506, !507, !"_ZN8sequence5getAFIP6vertexI8_point3dIdELi10EESt4pairIS3_S3_EiN9gTreeNodeIS3_7_vect3dIdES4_5nDataIS4_EE6toPairEEclEi: %agg.result"}
!507 = distinct !{!507, !"_ZN8sequence5getAFIP6vertexI8_point3dIdELi10EESt4pairIS3_S3_EiN9gTreeNodeIS3_7_vect3dIdES4_5nDataIS4_EE6toPairEEclEi"}
!508 = !{!503}
!509 = distinct !{!509, !16}
!510 = distinct !{!510, !24}
!511 = !{!512}
!512 = distinct !{!512, !513, !"_ZN8sequence12reduceSerialISt4pairI8_point3dIdES3_EiN9gTreeNodeIS3_7_vect3dIdE6vertexIS3_Li10EE5nDataIS9_EE6minMaxENS_4getAIS4_iEEEET_T0_SH_T1_T2_: %agg.result"}
!513 = distinct !{!513, !"_ZN8sequence12reduceSerialISt4pairI8_point3dIdES3_EiN9gTreeNodeIS3_7_vect3dIdE6vertexIS3_Li10EE5nDataIS9_EE6minMaxENS_4getAIS4_iEEEET_T0_SH_T1_T2_"}
!514 = !{!515}
!515 = distinct !{!515, !516, !"_ZN8sequence12reduceSerialISt4pairI8_point3dIdES3_EiN9gTreeNodeIS3_7_vect3dIdE6vertexIS3_Li10EE5nDataIS9_EE6minMaxENS_4getAIS4_iEEEET_T0_SH_T1_T2_: %agg.result"}
!516 = distinct !{!516, !"_ZN8sequence12reduceSerialISt4pairI8_point3dIdES3_EiN9gTreeNodeIS3_7_vect3dIdE6vertexIS3_Li10EE5nDataIS9_EE6minMaxENS_4getAIS4_iEEEET_T0_SH_T1_T2_"}
!517 = distinct !{!517, !24}
!518 = !{!519}
!519 = distinct !{!519, !520, !"_ZN8_point3dIdE11offsetPointEid: %agg.result"}
!520 = distinct !{!520, !"_ZN8_point3dIdE11offsetPointEid"}
!521 = distinct !{!521, !3}
!522 = distinct !{!522, !5, !3}
!523 = distinct !{!523, !24}
!524 = !{!525, !45, i64 0}
!525 = !{!"_ZTSSt4pairIiP6vertexI8_point3dIdELi10EEE", !45, i64 0, !31, i64 8}
!526 = !{!525, !31, i64 8}
!527 = distinct !{!527, !24}
!528 = !{!529}
!529 = distinct !{!529, !530, !"_ZN8_point3dIdE11offsetPointEid: %agg.result"}
!530 = distinct !{!530, !"_ZN8_point3dIdE11offsetPointEid"}
!531 = !{!532}
!532 = distinct !{!532, !533, !"_ZN8_point3dIdE11offsetPointEid: %agg.result"}
!533 = distinct !{!533, !"_ZN8_point3dIdE11offsetPointEid"}
!534 = distinct !{!534, !3}
!535 = distinct !{!535, !16}
!536 = distinct !{!536, !3}
!537 = distinct !{!537, !5, !3}
!538 = distinct !{!538, !5, !3}
!539 = !{!540, !29, i64 8}
!540 = !{!"_ZTSN7intSort5eBitsISt4pairIiP6vertexI8_point3dIdELi10EEEN5utils6firstFIiS6_EEEE", !541, i64 0, !29, i64 8, !29, i64 16}
!541 = !{!"_ZTSN5utils6firstFIiP6vertexI8_point3dIdELi10EEEE"}
!542 = !{!540, !29, i64 16}
!543 = distinct !{!543, !24}
!544 = distinct !{!544, !24}
!545 = distinct !{!545, !24}
!546 = distinct !{!546, !16}
!547 = distinct !{!547, !24, !144}
!548 = distinct !{!548, !16}
!549 = distinct !{!549, !16}
!550 = !{!551, !31, i64 0}
!551 = !{!"_ZTS10blockTransISt4pairIiP6vertexI8_point3dIdELi10EEEjE", !31, i64 0, !31, i64 8, !31, i64 16, !31, i64 24, !31, i64 32}
!552 = !{!551, !31, i64 8}
!553 = !{!551, !31, i64 16}
!554 = !{!551, !31, i64 24}
!555 = !{!551, !31, i64 32}
!556 = distinct !{!556, !16}
!557 = distinct !{!557, !16}
!558 = distinct !{!558, !24}
!559 = distinct !{!559, !16}
!560 = !{!561}
!561 = distinct !{!561, !562, !"_ZN8_point3dIdEmiES0_: %agg.result"}
!562 = distinct !{!562, !"_ZN8_point3dIdEmiES0_"}
!563 = !{!564}
!564 = distinct !{!564, !565, !"_ZN8_point3dIdEmiES0_: %agg.result"}
!565 = distinct !{!565, !"_ZN8_point3dIdEmiES0_"}
!566 = !{!77, !8, i64 16}
