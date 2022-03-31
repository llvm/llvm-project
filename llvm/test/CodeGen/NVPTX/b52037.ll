; Reproducer for a bad performance regression triggered by switch to the new PM.
; `barney` ended up with the local variables not being optimized away and that
; had rather dramatic effect on some GPU code. See
; https://bugs.llvm.org/show_bug.cgi?id=52037 for the gory details.
;
; RUN: llc -mtriple=nvptx64-nvidia-cuda -mcpu=sm_70 -O3 -o - %s | FileCheck %s

; CHECK-LABEL: .visible .entry barney(
; CHECK-NOT:  .local{{.*}}__local_depot
; CHECK: ret;

source_filename = "reduced.1.ll"
target triple = "nvptx64-nvidia-cuda"

%char3 = type { i8, i8, i8 }
%float4 = type { float, float, float, float }
%float3 = type { float, float, float }
%int3 = type { i32, i32, i32 }
%struct.spam.2 = type { %struct.foo.3, i16*, float, float, i32, float }
%struct.foo.3 = type <{ %float4*, %float4*, %float4*, i32*, i32*, i32, i32, float }>
%struct.zot = type { %struct.bar, [8 x i8], %struct.foo, [12 x i8] }
%struct.bar = type { i32 (...)** }
%struct.foo = type <{ i16*, %float4, %int3, i32, %float3, [4 x i8], i64, i32, i8, [3 x i8], i32 }>

@global = external local_unnamed_addr addrspace(4) externally_initialized global [27 x %char3], align 1
@global_1 = linkonce_odr unnamed_addr constant { [3 x i8*] } { [3 x i8*] [i8* inttoptr (i64 16 to i8*), i8* null, i8* null] }, align 8

; Function Attrs: argmemonly mustprogress nofree nounwind willreturn
declare void @llvm.memcpy.p0i8.p0i8.i64(i8* noalias nocapture writeonly, i8* noalias nocapture readonly, i64, i1 immarg) #0

declare %float4 @snork(float) local_unnamed_addr

declare %float3 @bar_2(float, float) local_unnamed_addr

declare %float3 @zot() local_unnamed_addr

declare %int3 @hoge(i32, i32, i32) local_unnamed_addr

declare i64 @foo() local_unnamed_addr

define void @barney(%struct.spam.2* nocapture readonly %arg) local_unnamed_addr {
bb:
  tail call void asm sideeffect "// KEEP", ""() #1
  %tmp = alloca %struct.zot, align 16
  %tmp4 = getelementptr inbounds %struct.spam.2, %struct.spam.2* %arg, i64 0, i32 1
  %tmp5 = load i16*, i16** %tmp4, align 8
  %tmp6 = bitcast %struct.zot* %tmp to i8*
  %tmp9 = getelementptr inbounds %struct.zot, %struct.zot* %tmp, i64 0, i32 2, i32 1
  %0 = bitcast %float4* %tmp9 to i16**
  store i16* %tmp5, i16** %0, align 8
  %tmp10 = getelementptr inbounds %struct.zot, %struct.zot* %tmp, i64 0, i32 0, i32 0
  store i32 (...)** bitcast (i8** getelementptr inbounds ({ [3 x i8*] }, { [3 x i8*] }* @global_1, i64 0, inrange i32 0, i64 3) to i32 (...)**), i32 (...)*** %tmp10, align 16
  %tmp34 = getelementptr %struct.spam.2, %struct.spam.2* %arg, i64 0, i32 0, i32 0
  %tmp.i1 = tail call i64 @foo()
  %tmp44.i16 = getelementptr inbounds i16, i16* %tmp5, i64 undef
  %tmp45.i17 = load i16, i16* %tmp44.i16, align 2
  %tmp47.i18 = icmp eq i16 %tmp45.i17, -1
  br i1 %tmp47.i18, label %bb14, label %bb49.i.lr.ph

bb49.i.lr.ph:                                     ; preds = %bb
  %tmp16 = bitcast %struct.zot* %tmp to i8**
  %tmp7.i6 = getelementptr inbounds %struct.zot, %struct.zot* %tmp, i64 0, i32 2
  %extract = lshr i16 %tmp45.i17, 11
  %extract.t = trunc i16 %extract to i8
  %1 = getelementptr inbounds %struct.zot, %struct.zot* %tmp, i64 0, i32 2
  %tmp58.i = getelementptr inbounds %struct.zot, %struct.zot* %tmp, i64 0, i32 2, i32 1, i32 2
  %tmp59.i = getelementptr inbounds %struct.zot, %struct.zot* %tmp, i64 0, i32 2, i32 4, i32 2
  %tmp62.i = getelementptr inbounds %struct.zot, %struct.zot* %tmp, i64 0, i32 2, i32 2, i32 2
  %2 = getelementptr inbounds %struct.foo, %struct.foo* %1, i64 1
  %3 = bitcast %struct.foo* %2 to i8*
  %tmp64.i = getelementptr inbounds %struct.zot, %struct.zot* %tmp, i64 0, i32 2, i32 10
  %tmp19.i.i = load float, float* %tmp58.i, align 16
  %tmp23.i.i = getelementptr inbounds %struct.zot, %struct.zot* %tmp, i64 0, i32 2, i32 2
  %4 = bitcast %int3* %tmp23.i.i to float*
  %tmp24.i.i = load float, float* %4, align 8
  %5 = getelementptr inbounds %struct.zot, %struct.zot* %tmp, i64 0, i32 2, i32 5, i64 0
  %6 = bitcast i8* %5 to float*
  %.repack3.i = getelementptr inbounds %struct.zot, %struct.zot* %tmp, i64 0, i32 2, i32 6
  %7 = bitcast i64* %.repack3.i to float*
  %tmp41.i.i = load i32, i32* %tmp62.i, align 16
  %tmp48.i.i = getelementptr inbounds %struct.zot, %struct.zot* %tmp, i64 0, i32 2, i32 3
  %tmp49.i.i = load i32, i32* %tmp48.i.i, align 4
  %tmp54.i.i = getelementptr inbounds %struct.zot, %struct.zot* %tmp, i64 0, i32 2, i32 4
  %8 = bitcast %float3* %tmp54.i.i to i32*
  %tmp55.i.i = load i32, i32* %8, align 8
  %tmp9.i = getelementptr inbounds %struct.zot, %struct.zot* %tmp, i64 0, i32 2, i32 7
  %9 = bitcast i32* %tmp9.i to i64*
  %tmp40.i = getelementptr inbounds %struct.zot, %struct.zot* %tmp, i64 0, i32 2, i32 4, i32 1
  %10 = bitcast float* %tmp40.i to i32*
  %tmp41.i = load i32, i32* %10, align 4
  %tmp42.i = zext i32 %tmp41.i to i64
  %tmp7.i = getelementptr inbounds %struct.zot, %struct.zot* %tmp, i64 0, i32 2
  %tmp17.pre = load i8*, i8** %tmp16, align 16
  %tmp60.i.peel = bitcast %struct.foo* %tmp7.i6 to i32**
  %tmp61.i.peel = load i32*, i32** %tmp60.i.peel, align 16
  %tmp10.i.i.peel = add nsw i8 %extract.t, -1
  store i8 %tmp10.i.i.peel, i8* %3, align 4
  %tmp13.i.i.peel = tail call %float3 @zot() #1
  %tmp15.i.i.peel = extractvalue %float3 %tmp13.i.i.peel, 0
  %tmp22.i.i.peel = fsub contract float %tmp19.i.i, %tmp15.i.i.peel
  %tmp17.i.i.peel = extractvalue %float3 %tmp13.i.i.peel, 2
  %tmp27.i.i.peel = fsub contract float %tmp24.i.i, %tmp17.i.i.peel
  %tmp28.i.i.peel = tail call %float3 @bar_2(float %tmp22.i.i.peel, float %tmp27.i.i.peel) #1
  %tmp28.i.elt.i.peel = extractvalue %float3 %tmp28.i.i.peel, 0
  store float %tmp28.i.elt.i.peel, float* %tmp59.i, align 16
  %tmp28.i.elt2.i.peel = extractvalue %float3 %tmp28.i.i.peel, 1
  store float %tmp28.i.elt2.i.peel, float* %6, align 4
  %tmp28.i.elt4.i.peel = extractvalue %float3 %tmp28.i.i.peel, 2
  store float %tmp28.i.elt4.i.peel, float* %7, align 8
  %tmp38.i.i.peel = zext i8 %tmp10.i.i.peel to i64
  %tmp39.i5.i.peel = getelementptr inbounds [27 x %char3], [27 x %char3] addrspace(4)* @global, i64 0, i64 %tmp38.i.i.peel
  %tmp39.i.i.peel = addrspacecast %char3 addrspace(4)* %tmp39.i5.i.peel to %char3*
  %tmp42.i.i.peel = getelementptr inbounds %char3, %char3* %tmp39.i.i.peel, i64 0, i32 0
  %tmp43.i.i.peel = load i8, i8* %tmp42.i.i.peel, align 1
  %tmp44.i.i.peel = sext i8 %tmp43.i.i.peel to i32
  %tmp45.i.i.peel = add nsw i32 %tmp41.i.i, %tmp44.i.i.peel
  %tmp50.i.i.peel = getelementptr inbounds %char3, %char3* %tmp39.i.i.peel, i64 0, i32 1
  %tmp51.i.i.peel = load i8, i8* %tmp50.i.i.peel, align 1
  %tmp52.i.i.peel = sext i8 %tmp51.i.i.peel to i32
  %tmp53.i.i.peel = add nsw i32 %tmp49.i.i, %tmp52.i.i.peel
  %tmp56.i.i.peel = getelementptr inbounds %char3, %char3* %tmp39.i.i.peel, i64 0, i32 2
  %tmp57.i.i.peel = load i8, i8* %tmp56.i.i.peel, align 1
  %tmp58.i.i.peel = sext i8 %tmp57.i.i.peel to i32
  %tmp59.i.i.peel = add nsw i32 %tmp55.i.i, %tmp58.i.i.peel
  %tmp60.i.i.peel = tail call %int3 @hoge(i32 %tmp45.i.i.peel, i32 %tmp53.i.i.peel, i32 %tmp59.i.i.peel) #1
  %tmp61.i.i.peel = getelementptr inbounds i32, i32* %tmp61.i.peel, i64 undef
  %tmp62.i.i.peel = load i32, i32* %tmp61.i.i.peel, align 4
  store i32 %tmp62.i.i.peel, i32* %tmp64.i, align 8
  %tmp22.peel = getelementptr inbounds %struct.zot, %struct.zot* %tmp, i64 0, i32 2
  %11 = bitcast %struct.foo* %tmp22.peel to i8*
  %tmp24.peel = getelementptr inbounds i8, i8* %11, i64 80
  %12 = bitcast i8* %tmp24.peel to i32*
  %tmp25.peel = load i32, i32* %12, align 16
  %tmp36.peel = load %float4*, %float4** %tmp34, align 8
  %tmp37.peel = zext i32 %tmp25.peel to i64
  %tmp38.peel = getelementptr inbounds %float4, %float4* %tmp36.peel, i64 %tmp37.peel
  %tmp39.peel = bitcast %float4* %tmp38.peel to i8*
  tail call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 1 undef, i8* align 1 %tmp39.peel, i64 undef, i1 false)
  %tmp40.peel = getelementptr inbounds %struct.zot, %struct.zot* %tmp, i64 0, i32 2, i32 4, i32 2
  %tmp41.peel25 = getelementptr inbounds float, float* %tmp40.peel, i64 2
  %tmp42.peel = load float, float* %tmp41.peel25, align 8
  %tmp44.peel = load float, float* inttoptr (i64 8 to float*), align 8
  %tmp45.peel = fsub contract float %tmp42.peel, %tmp44.peel
  %tmp46.peel = tail call %float4 @snork(float %tmp45.peel)
  %tmp.i.peel = tail call i64 @foo()
  %tmp10.i.peel = load i64, i64* %9, align 16
  %tmp11.i.peel = add i64 %tmp10.i.peel, %tmp.i.peel
  store i64 %tmp11.i.peel, i64* %9, align 16, !tbaa !1
  %tmp43.i.peel = add i64 %tmp11.i.peel, %tmp42.i
  %tmp44.i.peel = getelementptr inbounds i16, i16* %tmp5, i64 %tmp43.i.peel
  %tmp45.i.peel = load i16, i16* %tmp44.i.peel, align 2
  %tmp47.i.peel = icmp eq i16 %tmp45.i.peel, -1
  %extract21.peel = lshr i16 %tmp45.i.peel, 11
  %extract.t22.peel = trunc i16 %extract21.peel to i8
  br i1 %tmp47.i.peel, label %bb14, label %bb49.i.lr.ph.peel.newph

bb49.i.lr.ph.peel.newph:                          ; preds = %bb49.i.lr.ph
  %tmp60.i = bitcast %struct.foo* %tmp7.i to i32**
  %tmp61.i = load i32*, i32** %tmp60.i, align 16
  %tmp61.i.i = getelementptr inbounds i32, i32* %tmp61.i, i64 undef
  %tmp18 = getelementptr i8, i8* %tmp17.pre, i64 -24
  %tmp19 = bitcast i8* %tmp18 to i64*
  br label %bb49.i

bb49.i:                                           ; preds = %bb49.i, %bb49.i.lr.ph.peel.newph
  %tmp45.i20.off11 = phi i8 [ %extract.t22.peel, %bb49.i.lr.ph.peel.newph ], [ %extract.t22, %bb49.i ]
  %tmp10.i.i = add nsw i8 %tmp45.i20.off11, -1
  store i8 %tmp10.i.i, i8* %3, align 4
  %tmp13.i.i = tail call %float3 @zot() #1
  %tmp15.i.i = extractvalue %float3 %tmp13.i.i, 0
  %tmp22.i.i = fsub contract float %tmp19.i.i, %tmp15.i.i
  %tmp17.i.i = extractvalue %float3 %tmp13.i.i, 2
  %tmp27.i.i = fsub contract float %tmp24.i.i, %tmp17.i.i
  %tmp28.i.i = tail call %float3 @bar_2(float %tmp22.i.i, float %tmp27.i.i) #1
  %tmp28.i.elt.i = extractvalue %float3 %tmp28.i.i, 0
  store float %tmp28.i.elt.i, float* %tmp59.i, align 16
  %tmp28.i.elt2.i = extractvalue %float3 %tmp28.i.i, 1
  store float %tmp28.i.elt2.i, float* %6, align 4
  %tmp28.i.elt4.i = extractvalue %float3 %tmp28.i.i, 2
  store float %tmp28.i.elt4.i, float* %7, align 8
  %tmp38.i.i = zext i8 %tmp10.i.i to i64
  %tmp39.i5.i = getelementptr inbounds [27 x %char3], [27 x %char3] addrspace(4)* @global, i64 0, i64 %tmp38.i.i
  %tmp39.i.i = addrspacecast %char3 addrspace(4)* %tmp39.i5.i to %char3*
  %tmp42.i.i = getelementptr inbounds %char3, %char3* %tmp39.i.i, i64 0, i32 0
  %tmp43.i.i = load i8, i8* %tmp42.i.i, align 1
  %tmp44.i.i = sext i8 %tmp43.i.i to i32
  %tmp45.i.i = add nsw i32 %tmp41.i.i, %tmp44.i.i
  %tmp50.i.i = getelementptr inbounds %char3, %char3* %tmp39.i.i, i64 0, i32 1
  %tmp51.i.i = load i8, i8* %tmp50.i.i, align 1
  %tmp52.i.i = sext i8 %tmp51.i.i to i32
  %tmp53.i.i = add nsw i32 %tmp49.i.i, %tmp52.i.i
  %tmp56.i.i = getelementptr inbounds %char3, %char3* %tmp39.i.i, i64 0, i32 2
  %tmp57.i.i = load i8, i8* %tmp56.i.i, align 1
  %tmp58.i.i = sext i8 %tmp57.i.i to i32
  %tmp59.i.i = add nsw i32 %tmp55.i.i, %tmp58.i.i
  %tmp60.i.i = tail call %int3 @hoge(i32 %tmp45.i.i, i32 %tmp53.i.i, i32 %tmp59.i.i) #1
  %tmp62.i.i = load i32, i32* %tmp61.i.i, align 4
  store i32 %tmp62.i.i, i32* %tmp64.i, align 8
  %tmp20 = load i64, i64* %tmp19, align 8
  %tmp22 = getelementptr inbounds i8, i8* %tmp6, i64 %tmp20
  %tmp24 = getelementptr inbounds i8, i8* %tmp22, i64 80
  %13 = bitcast i8* %tmp24 to i32*
  %tmp25 = load i32, i32* %13, align 4
  %tmp36 = load %float4*, %float4** %tmp34, align 8
  %tmp37 = zext i32 %tmp25 to i64
  %tmp38 = getelementptr inbounds %float4, %float4* %tmp36, i64 %tmp37
  %tmp39 = bitcast %float4* %tmp38 to i8*
  tail call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 1 undef, i8* align 1 %tmp39, i64 undef, i1 false)
  %tmp40 = getelementptr inbounds i8, i8* %tmp22, i64 48
  %tmp41 = getelementptr inbounds i8, i8* %tmp40, i64 8
  %14 = bitcast i8* %tmp41 to float*
  %tmp42 = load float, float* %14, align 4
  %tmp44 = load float, float* inttoptr (i64 8 to float*), align 8
  %tmp45 = fsub contract float %tmp42, %tmp44
  %tmp46 = tail call %float4 @snork(float %tmp45)
  %tmp.i = tail call i64 @foo()
  %tmp10.i = load i64, i64* %9, align 16
  %tmp11.i = add i64 %tmp10.i, %tmp.i
  store i64 %tmp11.i, i64* %9, align 16, !tbaa !1
  %tmp43.i = add i64 %tmp11.i, %tmp42.i
  %tmp44.i = getelementptr inbounds i16, i16* %tmp5, i64 %tmp43.i
  %tmp45.i = load i16, i16* %tmp44.i, align 2
  %tmp47.i = icmp eq i16 %tmp45.i, -1
  %extract21 = lshr i16 %tmp45.i, 11
  %extract.t22 = trunc i16 %extract21 to i8
  br i1 %tmp47.i, label %bb14, label %bb49.i, !llvm.loop !12

bb14:                                             ; preds = %bb49.i.lr.ph, %bb49.i, %bb
  ret void
}

attributes #0 = { argmemonly mustprogress nofree nounwind willreturn }
attributes #1 = { nounwind }

!nvvm.annotations = !{!0}

!0 = !{void (%struct.spam.2*)* @barney, !"kernel", i32 1}
!1 = !{!2, !11, i64 64}
!2 = !{!"_ZTSN7cuneibs22neiblist_iterator_coreE", !3, i64 0, !3, i64 8, !6, i64 16, !8, i64 32, !9, i64 44, !10, i64 48, !11, i64 64, !9, i64 72, !4, i64 76, !9, i64 80}
!3 = !{!"any pointer", !4, i64 0}
!4 = !{!"omnipotent char", !5, i64 0}
!5 = !{!"Simple C++ TBAA"}
!6 = !{!"_ZTS6float4", !7, i64 0, !7, i64 4, !7, i64 8, !7, i64 12}
!7 = !{!"float", !4, i64 0}
!8 = !{!"_ZTS4int3", !9, i64 0, !9, i64 4, !9, i64 8}
!9 = !{!"int", !4, i64 0}
!10 = !{!"_ZTS6float3", !7, i64 0, !7, i64 4, !7, i64 8}
!11 = !{!"long", !4, i64 0}
!12 = distinct !{!12, !13}
!13 = !{!"llvm.loop.peeled.count", i32 1}
