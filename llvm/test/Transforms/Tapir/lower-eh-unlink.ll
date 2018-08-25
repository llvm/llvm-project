; RUN: opt < %s -tapir2target -tapir-target=cilk -debug-abi-calls -S | FileCheck %s
; RUN: opt < %s -passes=tapir2target -tapir-target=cilk -debug-abi-calls -S | FileCheck %s

%struct.vertex = type { %class._point2d, %struct.tri*, %struct.tri*, i32, i32 }
%class._point2d = type { double, double }
%struct.tri = type { [3 x %struct.tri*], [3 x %struct.vertex*], i32, i8, i8 }

$_ZN8sequence4packIP6vertexiNS_4getAIS2_iEEEE4_seqIT_EPS6_PbT0_SA_T1_ = comdat any

; Function Attrs: uwtable
define linkonce_odr { %struct.vertex**, i64 } @_ZN8sequence4packIP6vertexiNS_4getAIS2_iEEEE4_seqIT_EPS6_PbT0_SA_T1_(%struct.vertex** %Out, i8* %Fl, i32 %s, i32 %e, %struct.vertex** %f.coerce) local_unnamed_addr #7 comdat personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*) {
entry:
  %syncreg = tail call token @llvm.syncregion.start()
  %syncreg55 = tail call token @llvm.syncregion.start()
  %sub = sub nsw i32 %e, %s
  %sub1 = add nsw i32 %sub, -1
  %div = sdiv i32 %sub1, 2048
  %add = add nsw i32 %div, 1
  %cmp = icmp slt i32 %sub, 2049
  br i1 %cmp, label %if.then, label %pfor.detach.lr.ph

if.then:                                          ; preds = %entry
  %call = tail call { %struct.vertex**, i64 } @_ZN8sequence10packSerialIP6vertexiNS_4getAIS2_iEEEE4_seqIT_EPS6_PbT0_SA_T1_(%struct.vertex** %Out, i8* %Fl, i32 %s, i32 %e, %struct.vertex** %f.coerce)
  %0 = extractvalue { %struct.vertex**, i64 } %call, 0
  %1 = extractvalue { %struct.vertex**, i64 } %call, 1
  br label %cleanup

pfor.detach.lr.ph:                                ; preds = %entry
  %conv = sext i32 %add to i64
  %mul = shl nsw i64 %conv, 2
  %call3 = tail call noalias i8* @malloc(i64 %mul) #2
  %2 = bitcast i8* %call3 to i32*
  %3 = sext i32 %s to i64
  %4 = sext i32 %e to i64
  %5 = sext i32 %div to i64
  br label %pfor.detach

pfor.cond.cleanup:                                ; preds = %pfor.inc
  sync within %syncreg, label %sync.continue

pfor.detach:                                      ; preds = %pfor.inc, %pfor.detach.lr.ph
  %indvars.iv221 = phi i64 [ %indvars.iv.next222, %pfor.inc ], [ 0, %pfor.detach.lr.ph ]
  detach within %syncreg, label %pfor.body, label %pfor.inc unwind label %lpad27

pfor.body:                                        ; preds = %pfor.detach
  %6 = shl nsw i64 %indvars.iv221, 11
  %7 = add nsw i64 %6, %3
  %8 = add nsw i64 %7, 2048
  %cmp.i = icmp sgt i64 %8, %4
  %9 = trunc i64 %8 to i32
  %.sroa.speculated202 = select i1 %cmp.i, i32 %e, i32 %9
  %add.ptr = getelementptr inbounds i8, i8* %Fl, i64 %7
  %10 = trunc i64 %7 to i32
  %sub21 = sub nsw i32 %.sroa.speculated202, %10
  %cmp.i194 = icmp sgt i32 %sub21, 127
  %and.i = and i32 %sub21, 511
  %cmp1.i = icmp eq i32 %and.i, 0
  %or.cond.i = and i1 %cmp.i194, %cmp1.i
  br i1 %or.cond.i, label %land.lhs.true2.i, label %if.else.i

land.lhs.true2.i:                                 ; preds = %pfor.body
  %11 = ptrtoint i8* %add.ptr to i64
  %and3.i = and i64 %11, 3
  %cmp4.i = icmp eq i64 %and3.i, 0
  br i1 %cmp4.i, label %if.then.i, label %for.body28.lr.ph.i

if.then.i:                                        ; preds = %land.lhs.true2.i
  %shr74.i = lshr i32 %sub21, 9
  %cmp561.i = icmp sgt i32 %sub21, 511
  br i1 %cmp561.i, label %for.body.lr.ph.i, label %_ZN8sequence14sumFlagsSerialIiEET_PbS1_.exit

for.body.lr.ph.i:                                 ; preds = %if.then.i
  %12 = bitcast i8* %add.ptr to i32*
  br label %vector.ph

vector.ph:                                        ; preds = %for.body.lr.ph.i, %vector.ph
  %k.064.i = phi i32 [ 0, %for.body.lr.ph.i ], [ %inc22.i, %vector.ph ]
  %IFl.063.i = phi i32* [ %12, %for.body.lr.ph.i ], [ %add.ptr.i, %vector.ph ]
  %r.062.i = phi i32 [ 0, %for.body.lr.ph.i ], [ %add20.i, %vector.ph ]
  %13 = bitcast i32* %IFl.063.i to <8 x i32>*
  %wide.load = load <8 x i32>, <8 x i32>* %13, align 4
  %14 = getelementptr i32, i32* %IFl.063.i, i64 8
  %15 = bitcast i32* %14 to <8 x i32>*
  %wide.load234 = load <8 x i32>, <8 x i32>* %15, align 4
  %16 = getelementptr i32, i32* %IFl.063.i, i64 16
  %17 = bitcast i32* %16 to <8 x i32>*
  %wide.load235 = load <8 x i32>, <8 x i32>* %17, align 4
  %18 = getelementptr i32, i32* %IFl.063.i, i64 24
  %19 = bitcast i32* %18 to <8 x i32>*
  %wide.load236 = load <8 x i32>, <8 x i32>* %19, align 4
  %20 = getelementptr inbounds i32, i32* %IFl.063.i, i64 32
  %21 = bitcast i32* %20 to <8 x i32>*
  %wide.load.1 = load <8 x i32>, <8 x i32>* %21, align 4
  %22 = getelementptr i32, i32* %IFl.063.i, i64 40
  %23 = bitcast i32* %22 to <8 x i32>*
  %wide.load234.1 = load <8 x i32>, <8 x i32>* %23, align 4
  %24 = getelementptr i32, i32* %IFl.063.i, i64 48
  %25 = bitcast i32* %24 to <8 x i32>*
  %wide.load235.1 = load <8 x i32>, <8 x i32>* %25, align 4
  %26 = getelementptr i32, i32* %IFl.063.i, i64 56
  %27 = bitcast i32* %26 to <8 x i32>*
  %wide.load236.1 = load <8 x i32>, <8 x i32>* %27, align 4
  %28 = add nsw <8 x i32> %wide.load.1, %wide.load
  %29 = add nsw <8 x i32> %wide.load234.1, %wide.load234
  %30 = add nsw <8 x i32> %wide.load235.1, %wide.load235
  %31 = add nsw <8 x i32> %wide.load236.1, %wide.load236
  %32 = getelementptr inbounds i32, i32* %IFl.063.i, i64 64
  %33 = bitcast i32* %32 to <8 x i32>*
  %wide.load.2 = load <8 x i32>, <8 x i32>* %33, align 4
  %34 = getelementptr i32, i32* %IFl.063.i, i64 72
  %35 = bitcast i32* %34 to <8 x i32>*
  %wide.load234.2 = load <8 x i32>, <8 x i32>* %35, align 4
  %36 = getelementptr i32, i32* %IFl.063.i, i64 80
  %37 = bitcast i32* %36 to <8 x i32>*
  %wide.load235.2 = load <8 x i32>, <8 x i32>* %37, align 4
  %38 = getelementptr i32, i32* %IFl.063.i, i64 88
  %39 = bitcast i32* %38 to <8 x i32>*
  %wide.load236.2 = load <8 x i32>, <8 x i32>* %39, align 4
  %40 = add nsw <8 x i32> %wide.load.2, %28
  %41 = add nsw <8 x i32> %wide.load234.2, %29
  %42 = add nsw <8 x i32> %wide.load235.2, %30
  %43 = add nsw <8 x i32> %wide.load236.2, %31
  %44 = getelementptr inbounds i32, i32* %IFl.063.i, i64 96
  %45 = bitcast i32* %44 to <8 x i32>*
  %wide.load.3 = load <8 x i32>, <8 x i32>* %45, align 4
  %46 = getelementptr i32, i32* %IFl.063.i, i64 104
  %47 = bitcast i32* %46 to <8 x i32>*
  %wide.load234.3 = load <8 x i32>, <8 x i32>* %47, align 4
  %48 = getelementptr i32, i32* %IFl.063.i, i64 112
  %49 = bitcast i32* %48 to <8 x i32>*
  %wide.load235.3 = load <8 x i32>, <8 x i32>* %49, align 4
  %50 = getelementptr i32, i32* %IFl.063.i, i64 120
  %51 = bitcast i32* %50 to <8 x i32>*
  %wide.load236.3 = load <8 x i32>, <8 x i32>* %51, align 4
  %52 = add nsw <8 x i32> %wide.load.3, %40
  %53 = add nsw <8 x i32> %wide.load234.3, %41
  %54 = add nsw <8 x i32> %wide.load235.3, %42
  %55 = add nsw <8 x i32> %wide.load236.3, %43
  %bin.rdx = add <8 x i32> %53, %52
  %bin.rdx237 = add <8 x i32> %54, %bin.rdx
  %bin.rdx238 = add <8 x i32> %55, %bin.rdx237
  %rdx.shuf = shufflevector <8 x i32> %bin.rdx238, <8 x i32> undef, <8 x i32> <i32 4, i32 5, i32 6, i32 7, i32 undef, i32 undef, i32 undef, i32 undef>
  %bin.rdx239 = add <8 x i32> %bin.rdx238, %rdx.shuf
  %rdx.shuf240 = shufflevector <8 x i32> %bin.rdx239, <8 x i32> undef, <8 x i32> <i32 2, i32 3, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  %bin.rdx241 = add <8 x i32> %bin.rdx239, %rdx.shuf240
  %rdx.shuf242 = shufflevector <8 x i32> %bin.rdx241, <8 x i32> undef, <8 x i32> <i32 1, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  %bin.rdx243 = add <8 x i32> %bin.rdx241, %rdx.shuf242
  %56 = extractelement <8 x i32> %bin.rdx243, i32 0
  %and10.i = and i32 %56, 255
  %57 = lshr i32 %56, 8
  %and12.i = and i32 %57, 255
  %58 = lshr i32 %56, 16
  %and15.i = and i32 %58, 255
  %59 = lshr i32 %56, 24
  %add13.i = add i32 %and10.i, %r.062.i
  %add16.i = add i32 %add13.i, %59
  %add19.i = add i32 %add16.i, %and12.i
  %add20.i = add i32 %add19.i, %and15.i
  %add.ptr.i = getelementptr inbounds i32, i32* %IFl.063.i, i64 128
  %inc22.i = add nuw nsw i32 %k.064.i, 1
  %cmp5.i = icmp ult i32 %inc22.i, %shr74.i
  br i1 %cmp5.i, label %vector.ph, label %_ZN8sequence14sumFlagsSerialIiEET_PbS1_.exit

if.else.i:                                        ; preds = %pfor.body
  %cmp2665.i = icmp sgt i32 %sub21, 0
  br i1 %cmp2665.i, label %for.body28.lr.ph.i, label %_ZN8sequence14sumFlagsSerialIiEET_PbS1_.exit

for.body28.lr.ph.i:                               ; preds = %if.else.i, %land.lhs.true2.i
  %wide.trip.count.i = zext i32 %sub21 to i64
  %min.iters.check = icmp ult i32 %sub21, 32
  br i1 %min.iters.check, label %for.body28.i.preheader, label %vector.ph247

vector.ph247:                                     ; preds = %for.body28.lr.ph.i
  %n.vec = and i64 %wide.trip.count.i, 4294967264
  %60 = add nsw i64 %n.vec, -32
  %61 = lshr exact i64 %60, 5
  %62 = add nuw nsw i64 %61, 1
  %xtraiter = and i64 %62, 3
  %63 = icmp ult i64 %60, 96
  br i1 %63, label %middle.block245.unr-lcssa, label %vector.ph247.new

vector.ph247.new:                                 ; preds = %vector.ph247
  %unroll_iter = sub nsw i64 %62, %xtraiter
  br label %vector.body244

vector.body244:                                   ; preds = %vector.body244, %vector.ph247.new
  %index248 = phi i64 [ 0, %vector.ph247.new ], [ %index.next249.3, %vector.body244 ]
  %vec.phi258 = phi <8 x i32> [ zeroinitializer, %vector.ph247.new ], [ %124, %vector.body244 ]
  %vec.phi259 = phi <8 x i32> [ zeroinitializer, %vector.ph247.new ], [ %125, %vector.body244 ]
  %vec.phi260 = phi <8 x i32> [ zeroinitializer, %vector.ph247.new ], [ %126, %vector.body244 ]
  %vec.phi261 = phi <8 x i32> [ zeroinitializer, %vector.ph247.new ], [ %127, %vector.body244 ]
  %niter = phi i64 [ %unroll_iter, %vector.ph247.new ], [ %niter.nsub.3, %vector.body244 ]
  %64 = getelementptr inbounds i8, i8* %add.ptr, i64 %index248
  %65 = bitcast i8* %64 to <8 x i8>*
  %wide.load262 = load <8 x i8>, <8 x i8>* %65, align 1
  %66 = getelementptr i8, i8* %64, i64 8
  %67 = bitcast i8* %66 to <8 x i8>*
  %wide.load263 = load <8 x i8>, <8 x i8>* %67, align 1
  %68 = getelementptr i8, i8* %64, i64 16
  %69 = bitcast i8* %68 to <8 x i8>*
  %wide.load264 = load <8 x i8>, <8 x i8>* %69, align 1
  %70 = getelementptr i8, i8* %64, i64 24
  %71 = bitcast i8* %70 to <8 x i8>*
  %wide.load265 = load <8 x i8>, <8 x i8>* %71, align 1
  %72 = zext <8 x i8> %wide.load262 to <8 x i32>
  %73 = zext <8 x i8> %wide.load263 to <8 x i32>
  %74 = zext <8 x i8> %wide.load264 to <8 x i32>
  %75 = zext <8 x i8> %wide.load265 to <8 x i32>
  %76 = add nuw nsw <8 x i32> %vec.phi258, %72
  %77 = add nuw nsw <8 x i32> %vec.phi259, %73
  %78 = add nuw nsw <8 x i32> %vec.phi260, %74
  %79 = add nuw nsw <8 x i32> %vec.phi261, %75
  %index.next249 = or i64 %index248, 32
  %80 = getelementptr inbounds i8, i8* %add.ptr, i64 %index.next249
  %81 = bitcast i8* %80 to <8 x i8>*
  %wide.load262.1 = load <8 x i8>, <8 x i8>* %81, align 1
  %82 = getelementptr i8, i8* %80, i64 8
  %83 = bitcast i8* %82 to <8 x i8>*
  %wide.load263.1 = load <8 x i8>, <8 x i8>* %83, align 1
  %84 = getelementptr i8, i8* %80, i64 16
  %85 = bitcast i8* %84 to <8 x i8>*
  %wide.load264.1 = load <8 x i8>, <8 x i8>* %85, align 1
  %86 = getelementptr i8, i8* %80, i64 24
  %87 = bitcast i8* %86 to <8 x i8>*
  %wide.load265.1 = load <8 x i8>, <8 x i8>* %87, align 1
  %88 = zext <8 x i8> %wide.load262.1 to <8 x i32>
  %89 = zext <8 x i8> %wide.load263.1 to <8 x i32>
  %90 = zext <8 x i8> %wide.load264.1 to <8 x i32>
  %91 = zext <8 x i8> %wide.load265.1 to <8 x i32>
  %92 = add nuw nsw <8 x i32> %76, %88
  %93 = add nuw nsw <8 x i32> %77, %89
  %94 = add nuw nsw <8 x i32> %78, %90
  %95 = add nuw nsw <8 x i32> %79, %91
  %index.next249.1 = or i64 %index248, 64
  %96 = getelementptr inbounds i8, i8* %add.ptr, i64 %index.next249.1
  %97 = bitcast i8* %96 to <8 x i8>*
  %wide.load262.2 = load <8 x i8>, <8 x i8>* %97, align 1
  %98 = getelementptr i8, i8* %96, i64 8
  %99 = bitcast i8* %98 to <8 x i8>*
  %wide.load263.2 = load <8 x i8>, <8 x i8>* %99, align 1
  %100 = getelementptr i8, i8* %96, i64 16
  %101 = bitcast i8* %100 to <8 x i8>*
  %wide.load264.2 = load <8 x i8>, <8 x i8>* %101, align 1
  %102 = getelementptr i8, i8* %96, i64 24
  %103 = bitcast i8* %102 to <8 x i8>*
  %wide.load265.2 = load <8 x i8>, <8 x i8>* %103, align 1
  %104 = zext <8 x i8> %wide.load262.2 to <8 x i32>
  %105 = zext <8 x i8> %wide.load263.2 to <8 x i32>
  %106 = zext <8 x i8> %wide.load264.2 to <8 x i32>
  %107 = zext <8 x i8> %wide.load265.2 to <8 x i32>
  %108 = add nuw nsw <8 x i32> %92, %104
  %109 = add nuw nsw <8 x i32> %93, %105
  %110 = add nuw nsw <8 x i32> %94, %106
  %111 = add nuw nsw <8 x i32> %95, %107
  %index.next249.2 = or i64 %index248, 96
  %112 = getelementptr inbounds i8, i8* %add.ptr, i64 %index.next249.2
  %113 = bitcast i8* %112 to <8 x i8>*
  %wide.load262.3 = load <8 x i8>, <8 x i8>* %113, align 1
  %114 = getelementptr i8, i8* %112, i64 8
  %115 = bitcast i8* %114 to <8 x i8>*
  %wide.load263.3 = load <8 x i8>, <8 x i8>* %115, align 1
  %116 = getelementptr i8, i8* %112, i64 16
  %117 = bitcast i8* %116 to <8 x i8>*
  %wide.load264.3 = load <8 x i8>, <8 x i8>* %117, align 1
  %118 = getelementptr i8, i8* %112, i64 24
  %119 = bitcast i8* %118 to <8 x i8>*
  %wide.load265.3 = load <8 x i8>, <8 x i8>* %119, align 1
  %120 = zext <8 x i8> %wide.load262.3 to <8 x i32>
  %121 = zext <8 x i8> %wide.load263.3 to <8 x i32>
  %122 = zext <8 x i8> %wide.load264.3 to <8 x i32>
  %123 = zext <8 x i8> %wide.load265.3 to <8 x i32>
  %124 = add nuw nsw <8 x i32> %108, %120
  %125 = add nuw nsw <8 x i32> %109, %121
  %126 = add nuw nsw <8 x i32> %110, %122
  %127 = add nuw nsw <8 x i32> %111, %123
  %index.next249.3 = add i64 %index248, 128
  %niter.nsub.3 = add i64 %niter, -4
  %niter.ncmp.3 = icmp eq i64 %niter.nsub.3, 0
  br i1 %niter.ncmp.3, label %middle.block245.unr-lcssa, label %vector.body244, !llvm.loop !270

middle.block245.unr-lcssa:                        ; preds = %vector.body244, %vector.ph247
  %.lcssa279.ph = phi <8 x i32> [ undef, %vector.ph247 ], [ %124, %vector.body244 ]
  %.lcssa278.ph = phi <8 x i32> [ undef, %vector.ph247 ], [ %125, %vector.body244 ]
  %.lcssa277.ph = phi <8 x i32> [ undef, %vector.ph247 ], [ %126, %vector.body244 ]
  %.lcssa.ph = phi <8 x i32> [ undef, %vector.ph247 ], [ %127, %vector.body244 ]
  %index248.unr = phi i64 [ 0, %vector.ph247 ], [ %index.next249.3, %vector.body244 ]
  %vec.phi258.unr = phi <8 x i32> [ zeroinitializer, %vector.ph247 ], [ %124, %vector.body244 ]
  %vec.phi259.unr = phi <8 x i32> [ zeroinitializer, %vector.ph247 ], [ %125, %vector.body244 ]
  %vec.phi260.unr = phi <8 x i32> [ zeroinitializer, %vector.ph247 ], [ %126, %vector.body244 ]
  %vec.phi261.unr = phi <8 x i32> [ zeroinitializer, %vector.ph247 ], [ %127, %vector.body244 ]
  %lcmp.mod = icmp eq i64 %xtraiter, 0
  br i1 %lcmp.mod, label %middle.block245, label %vector.body244.epil.preheader

vector.body244.epil.preheader:                    ; preds = %middle.block245.unr-lcssa
  br label %vector.body244.epil

vector.body244.epil:                              ; preds = %vector.body244.epil, %vector.body244.epil.preheader
  %index248.epil = phi i64 [ %index248.unr, %vector.body244.epil.preheader ], [ %index.next249.epil, %vector.body244.epil ]
  %vec.phi258.epil = phi <8 x i32> [ %vec.phi258.unr, %vector.body244.epil.preheader ], [ %140, %vector.body244.epil ]
  %vec.phi259.epil = phi <8 x i32> [ %vec.phi259.unr, %vector.body244.epil.preheader ], [ %141, %vector.body244.epil ]
  %vec.phi260.epil = phi <8 x i32> [ %vec.phi260.unr, %vector.body244.epil.preheader ], [ %142, %vector.body244.epil ]
  %vec.phi261.epil = phi <8 x i32> [ %vec.phi261.unr, %vector.body244.epil.preheader ], [ %143, %vector.body244.epil ]
  %epil.iter = phi i64 [ %xtraiter, %vector.body244.epil.preheader ], [ %epil.iter.sub, %vector.body244.epil ]
  %128 = getelementptr inbounds i8, i8* %add.ptr, i64 %index248.epil
  %129 = bitcast i8* %128 to <8 x i8>*
  %wide.load262.epil = load <8 x i8>, <8 x i8>* %129, align 1
  %130 = getelementptr i8, i8* %128, i64 8
  %131 = bitcast i8* %130 to <8 x i8>*
  %wide.load263.epil = load <8 x i8>, <8 x i8>* %131, align 1
  %132 = getelementptr i8, i8* %128, i64 16
  %133 = bitcast i8* %132 to <8 x i8>*
  %wide.load264.epil = load <8 x i8>, <8 x i8>* %133, align 1
  %134 = getelementptr i8, i8* %128, i64 24
  %135 = bitcast i8* %134 to <8 x i8>*
  %wide.load265.epil = load <8 x i8>, <8 x i8>* %135, align 1
  %136 = zext <8 x i8> %wide.load262.epil to <8 x i32>
  %137 = zext <8 x i8> %wide.load263.epil to <8 x i32>
  %138 = zext <8 x i8> %wide.load264.epil to <8 x i32>
  %139 = zext <8 x i8> %wide.load265.epil to <8 x i32>
  %140 = add nuw nsw <8 x i32> %vec.phi258.epil, %136
  %141 = add nuw nsw <8 x i32> %vec.phi259.epil, %137
  %142 = add nuw nsw <8 x i32> %vec.phi260.epil, %138
  %143 = add nuw nsw <8 x i32> %vec.phi261.epil, %139
  %index.next249.epil = add i64 %index248.epil, 32
  %epil.iter.sub = add i64 %epil.iter, -1
  %epil.iter.cmp = icmp eq i64 %epil.iter.sub, 0
  br i1 %epil.iter.cmp, label %middle.block245, label %vector.body244.epil, !llvm.loop !271

middle.block245:                                  ; preds = %vector.body244.epil, %middle.block245.unr-lcssa
  %.lcssa279 = phi <8 x i32> [ %.lcssa279.ph, %middle.block245.unr-lcssa ], [ %140, %vector.body244.epil ]
  %.lcssa278 = phi <8 x i32> [ %.lcssa278.ph, %middle.block245.unr-lcssa ], [ %141, %vector.body244.epil ]
  %.lcssa277 = phi <8 x i32> [ %.lcssa277.ph, %middle.block245.unr-lcssa ], [ %142, %vector.body244.epil ]
  %.lcssa = phi <8 x i32> [ %.lcssa.ph, %middle.block245.unr-lcssa ], [ %143, %vector.body244.epil ]
  %bin.rdx266 = add nuw <8 x i32> %.lcssa278, %.lcssa279
  %bin.rdx267 = add <8 x i32> %.lcssa277, %bin.rdx266
  %bin.rdx268 = add <8 x i32> %.lcssa, %bin.rdx267
  %rdx.shuf269 = shufflevector <8 x i32> %bin.rdx268, <8 x i32> undef, <8 x i32> <i32 4, i32 5, i32 6, i32 7, i32 undef, i32 undef, i32 undef, i32 undef>
  %bin.rdx270 = add <8 x i32> %bin.rdx268, %rdx.shuf269
  %rdx.shuf271 = shufflevector <8 x i32> %bin.rdx270, <8 x i32> undef, <8 x i32> <i32 2, i32 3, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  %bin.rdx272 = add <8 x i32> %bin.rdx270, %rdx.shuf271
  %rdx.shuf273 = shufflevector <8 x i32> %bin.rdx272, <8 x i32> undef, <8 x i32> <i32 1, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  %bin.rdx274 = add <8 x i32> %bin.rdx272, %rdx.shuf273
  %144 = extractelement <8 x i32> %bin.rdx274, i32 0
  %cmp.n251 = icmp eq i64 %n.vec, %wide.trip.count.i
  br i1 %cmp.n251, label %_ZN8sequence14sumFlagsSerialIiEET_PbS1_.exit, label %for.body28.i.preheader

for.body28.i.preheader:                           ; preds = %middle.block245, %for.body28.lr.ph.i
  %indvars.iv70.i.ph = phi i64 [ 0, %for.body28.lr.ph.i ], [ %n.vec, %middle.block245 ]
  %r.166.i.ph = phi i32 [ 0, %for.body28.lr.ph.i ], [ %144, %middle.block245 ]
  br label %for.body28.i

for.body28.i:                                     ; preds = %for.body28.i.preheader, %for.body28.i
  %indvars.iv70.i = phi i64 [ %indvars.iv.next71.i, %for.body28.i ], [ %indvars.iv70.i.ph, %for.body28.i.preheader ]
  %r.166.i = phi i32 [ %add31.i, %for.body28.i ], [ %r.166.i.ph, %for.body28.i.preheader ]
  %arrayidx30.i = getelementptr inbounds i8, i8* %add.ptr, i64 %indvars.iv70.i
  %145 = load i8, i8* %arrayidx30.i, align 1
  %146 = zext i8 %145 to i32
  %add31.i = add nuw nsw i32 %r.166.i, %146
  %indvars.iv.next71.i = add nuw nsw i64 %indvars.iv70.i, 1
  %exitcond72.i = icmp eq i64 %indvars.iv.next71.i, %wide.trip.count.i
  br i1 %exitcond72.i, label %_ZN8sequence14sumFlagsSerialIiEET_PbS1_.exit, label %for.body28.i, !llvm.loop !272

_ZN8sequence14sumFlagsSerialIiEET_PbS1_.exit:     ; preds = %for.body28.i, %vector.ph, %middle.block245, %if.then.i, %if.else.i
  %r.2.i = phi i32 [ 0, %if.then.i ], [ 0, %if.else.i ], [ %144, %middle.block245 ], [ %add20.i, %vector.ph ], [ %add31.i, %for.body28.i ]
  %arrayidx = getelementptr inbounds i32, i32* %2, i64 %indvars.iv221
  store i32 %r.2.i, i32* %arrayidx, align 4
  reattach within %syncreg, label %pfor.inc

pfor.inc:                                         ; preds = %pfor.detach, %_ZN8sequence14sumFlagsSerialIiEET_PbS1_.exit
  %indvars.iv.next222 = add nuw nsw i64 %indvars.iv221, 1
  %cmp12 = icmp slt i64 %indvars.iv221, %5
  br i1 %cmp12, label %pfor.detach, label %pfor.cond.cleanup, !llvm.loop !273

lpad27:                                           ; preds = %pfor.detach
  %147 = landingpad { i8*, i32 }
          cleanup
  %148 = extractvalue { i8*, i32 } %147, 0
  %149 = extractvalue { i8*, i32 } %147, 1
  sync within %syncreg, label %ehcleanup119

sync.continue:                                    ; preds = %pfor.cond.cleanup
  %call.i = tail call i32 @_ZN8sequence4scanIiiN5utils4addFIiEENS_4getAIiiEEEET_PS6_T0_S8_T1_T2_S6_bb(i32* %2, i32 0, i32 %add, i32* %2, i32 0, i1 zeroext false, i1 zeroext false)
  %cmp41 = icmp eq %struct.vertex** %Out, null
  br i1 %cmp41, label %if.then42, label %if.end46

if.then42:                                        ; preds = %sync.continue
  %conv43 = sext i32 %call.i to i64
  %mul44 = shl nsw i64 %conv43, 3
  %call45 = tail call noalias i8* @malloc(i64 %mul44) #2
  %150 = bitcast i8* %call45 to %struct.vertex**
  br label %if.end46

if.end46:                                         ; preds = %if.then42, %sync.continue
  %Out.addr.0 = phi %struct.vertex** [ %150, %if.then42 ], [ %Out, %sync.continue ]
  %cmp64212 = icmp slt i32 %sub, -2046
  br i1 %cmp64212, label %pfor.cond.cleanup65, label %pfor.detach66.lr.ph

pfor.detach66.lr.ph:                              ; preds = %if.end46
  %151 = sext i32 %s to i64
  %152 = sext i32 %e to i64
  %153 = sext i32 %div to i64
  br label %pfor.detach66

pfor.cond.cleanup65:                              ; preds = %pfor.inc97, %if.end46
  sync within %syncreg55, label %sync.continue106

pfor.detach66:                                    ; preds = %pfor.inc97, %pfor.detach66.lr.ph
  %indvars.iv = phi i64 [ %indvars.iv.next, %pfor.inc97 ], [ 0, %pfor.detach66.lr.ph ]
  detach within %syncreg55, label %pfor.body71, label %pfor.inc97 unwind label %lpad99.loopexit

pfor.body71:                                      ; preds = %pfor.detach66
  %154 = shl nsw i64 %indvars.iv, 11
  %155 = add nsw i64 %154, %151
  %156 = add nsw i64 %155, 2048
  %cmp.i195 = icmp sgt i64 %156, %152
  %157 = trunc i64 %156 to i32
  %.sroa.speculated = select i1 %cmp.i195, i32 %e, i32 %157
  %arrayidx85 = getelementptr inbounds i32, i32* %2, i64 %indvars.iv
  %158 = load i32, i32* %arrayidx85, align 4
  %idx.ext86 = sext i32 %158 to i64
  %add.ptr87 = getelementptr inbounds %struct.vertex*, %struct.vertex** %Out.addr.0, i64 %idx.ext86
  %159 = trunc i64 %155 to i32
  %call92 = invoke { %struct.vertex**, i64 } @_ZN8sequence10packSerialIP6vertexiNS_4getAIS2_iEEEE4_seqIT_EPS6_PbT0_SA_T1_(%struct.vertex** %add.ptr87, i8* %Fl, i32 %159, i32 %.sroa.speculated, %struct.vertex** %f.coerce)
          to label %invoke.cont91 unwind label %lpad90

invoke.cont91:                                    ; preds = %pfor.body71
  reattach within %syncreg55, label %pfor.inc97

pfor.inc97:                                       ; preds = %pfor.detach66, %invoke.cont91
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %cmp64 = icmp slt i64 %indvars.iv, %153
  br i1 %cmp64, label %pfor.detach66, label %pfor.cond.cleanup65, !llvm.loop !274

lpad90:                                           ; preds = %pfor.body71
  %160 = landingpad { i8*, i32 }
          catch i8* null
  invoke void @llvm.detached.rethrow.sl_p0i8i32s(token %syncreg55, { i8*, i32 } %160)
          to label %det.rethrow.unreachable105 unwind label %lpad99.loopexit.split-lp

det.rethrow.unreachable105:                       ; preds = %lpad90
  unreachable

lpad99.loopexit:                                  ; preds = %pfor.detach66
  %lpad.loopexit = landingpad { i8*, i32 }
          cleanup
  br label %lpad99

lpad99.loopexit.split-lp:                         ; preds = %lpad90
  %lpad.loopexit.split-lp = landingpad { i8*, i32 }
          cleanup
  br label %lpad99

lpad99:                                           ; preds = %lpad99.loopexit.split-lp, %lpad99.loopexit
  %lpad.phi = phi { i8*, i32 } [ %lpad.loopexit, %lpad99.loopexit ], [ %lpad.loopexit.split-lp, %lpad99.loopexit.split-lp ]
  %161 = extractvalue { i8*, i32 } %lpad.phi, 0
  %162 = extractvalue { i8*, i32 } %lpad.phi, 1
  sync within %syncreg55, label %ehcleanup119

sync.continue106:                                 ; preds = %pfor.cond.cleanup65
  tail call void @free(i8* %call3) #2
  %conv117 = sext i32 %call.i to i64
  br label %cleanup

ehcleanup119:                                     ; preds = %lpad99, %lpad27
  %ehselector.slot29.0 = phi i32 [ %149, %lpad27 ], [ %162, %lpad99 ]
  %exn.slot28.0 = phi i8* [ %148, %lpad27 ], [ %161, %lpad99 ]
  %lpad.val124 = insertvalue { i8*, i32 } undef, i8* %exn.slot28.0, 0
  %lpad.val125 = insertvalue { i8*, i32 } %lpad.val124, i32 %ehselector.slot29.0, 1
  resume { i8*, i32 } %lpad.val125

cleanup:                                          ; preds = %sync.continue106, %if.then
  %retval.sroa.0.0 = phi %struct.vertex** [ %0, %if.then ], [ %Out.addr.0, %sync.continue106 ]
  %retval.sroa.3.0 = phi i64 [ %1, %if.then ], [ %conv117, %sync.continue106 ]
  %.fca.0.insert = insertvalue { %struct.vertex**, i64 } undef, %struct.vertex** %retval.sroa.0.0, 0
  %.fca.1.insert = insertvalue { %struct.vertex**, i64 } %.fca.0.insert, i64 %retval.sroa.3.0, 1
  ret { %struct.vertex**, i64 } %.fca.1.insert
}

; Function Attrs: argmemonly nounwind
declare token @llvm.syncregion.start() #4

declare i32 @__gxx_personality_v0(...)

declare { %struct.vertex**, i64 } @_ZN8sequence10packSerialIP6vertexiNS_4getAIS2_iEEEE4_seqIT_EPS6_PbT0_SA_T1_(%struct.vertex** %Out, i8* %Fl, i32 %s, i32 %e, %struct.vertex** %f.coerce) local_unnamed_addr #7

declare i32 @_ZN8sequence4scanIiiN5utils4addFIiEENS_4getAIiiEEEET_PS6_T0_S8_T1_T2_S6_bb(i32* %Out, i32 %s, i32 %e, i32* %g.coerce, i32 %zero, i1 zeroext %inclusive, i1 zeroext %back) local_unnamed_addr #7

; Function Attrs: nounwind
declare void @free(i8* nocapture) local_unnamed_addr #1

; Function Attrs: argmemonly
declare void @llvm.detached.rethrow.sl_p0i8i32s(token, { i8*, i32 }) #9

; Function Attrs: nounwind
declare noalias i8* @malloc(i64) local_unnamed_addr #1

; CHECK: define internal fastcc void @_ZN8sequence4packIP6vertexiNS_4getAIS2_iEEEE4_seqIT_EPS6_PbT0_SA_T1__pfor.body71.otd1(i64 %indvars.iv.otd1
; CHECK: lpad90.otd1:
; CHECK: %[[LPAD:.+]] = landingpad
; CHECK: call void @__cilk_parent_epilogue(%struct.__cilkrts_stack_frame* %__cilkrts_sf)
; CHECK: resume {{.+}} %[[LPAD]]

; CHECK: define internal fastcc void @_ZN8sequence4packIP6vertexiNS_4getAIS2_iEEEE4_seqIT_EPS6_PbT0_SA_T1__pfor.body.otd1(i64 %indvars.iv221.otd1

attributes #1 = { nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="haswell" "target-features"="+aes,+avx,+avx2,+bmi,+bmi2,+cmov,+cx16,+f16c,+fma,+fsgsbase,+fxsr,+lzcnt,+mmx,+movbe,+pclmul,+popcnt,+rdrnd,+sse,+sse2,+sse3,+sse4.1,+sse4.2,+ssse3,+x87,+xsave,+xsaveopt,-adx,-avx512bitalg,-avx512bw,-avx512cd,-avx512dq,-avx512er,-avx512f,-avx512ifma,-avx512pf,-avx512vbmi,-avx512vbmi2,-avx512vl,-avx512vnni,-avx512vpopcntdq,-clflushopt,-clwb,-clzero,-fma4,-gfni,-ibt,-lwp,-mwaitx,-pku,-prefetchwt1,-prfchw,-rdseed,-rtm,-sgx,-sha,-shstk,-sse4a,-tbm,-vaes,-vpclmulqdq,-xop,-xsavec,-xsaves" "unsafe-fp-math"="true" "use-soft-float"="false" }
attributes #4 = { argmemonly nounwind }
attributes #7 = { uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="true" "no-jump-tables"="false" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="haswell" "target-features"="+aes,+avx,+avx2,+bmi,+bmi2,+cmov,+cx16,+f16c,+fma,+fsgsbase,+fxsr,+lzcnt,+mmx,+movbe,+pclmul,+popcnt,+rdrnd,+sse,+sse2,+sse3,+sse4.1,+sse4.2,+ssse3,+x87,+xsave,+xsaveopt,-adx,-avx512bitalg,-avx512bw,-avx512cd,-avx512dq,-avx512er,-avx512f,-avx512ifma,-avx512pf,-avx512vbmi,-avx512vbmi2,-avx512vl,-avx512vnni,-avx512vpopcntdq,-clflushopt,-clwb,-clzero,-fma4,-gfni,-ibt,-lwp,-mwaitx,-pku,-prefetchwt1,-prfchw,-rdseed,-rtm,-sgx,-sha,-shstk,-sse4a,-tbm,-vaes,-vpclmulqdq,-xop,-xsavec,-xsaves" "unsafe-fp-math"="true" "use-soft-float"="false" }
attributes #9 = { argmemonly }

!3 = !{!"llvm.loop.isvectorized", i32 1}
!5 = !{!"llvm.loop.unroll.runtime.disable"}
!16 = !{!"llvm.loop.unroll.disable"}
!84 = !{!"tapir.loop.spawn.strategy", i32 1}
!270 = distinct !{!270, !3}
!271 = distinct !{!271, !16}
!272 = distinct !{!272, !5, !3}
!273 = distinct !{!273, !84}
!274 = distinct !{!274, !84}
