; RUN: llc  -mtriple=x86_64-unknown-unknown -mattr=+avx512f,+avx512bw,+avx512vl,+avx512dq -mcpu=znver5 < %s | FileCheck %s
; RUN: llc -update-baseIndex -mtriple=x86_64-unknown-unknown -mattr=+avx512f,+avx512bw,+avx512vl,+avx512dq -mcpu=znver5 < %s | FileCheck %s
; RUN: llc -update-baseIndex=false -mtriple=x86_64-unknown-unknown -mattr=+avx512f,+avx512bw,+avx512vl,+avx512dq -mcpu=znver5 < %s | FileCheck %s -check-prefix=OLD

; ModuleID = 'qwdemo.c'
source_filename = "qwdemo.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct.pt = type { float, float, float, i32 }

; Function Attrs: nofree nosync nounwind memory(argmem: readwrite) uwtable
define dso_local i32 @foo(float noundef %cut_coulsq, ptr noalias nocapture noundef readonly %jlist, i32 noundef %jnum, ptr noalias nocapture noundef readonly %x, ptr noalias nocapture noundef writeonly %trsq, ptr noalias nocapture noundef writeonly %tdelx, ptr noalias nocapture noundef writeonly %tdely, ptr noalias nocapture noundef writeonly %tdelz, ptr noalias nocapture noundef writeonly %tjtype, ptr noalias nocapture noundef writeonly %tj, ptr noalias nocapture noundef readnone %tx, ptr noalias nocapture noundef readnone %ty, ptr noalias nocapture noundef readnone %tz) local_unnamed_addr #0 {
entry:
  %0 = load float, ptr %x, align 4, !tbaa !5
  %y = getelementptr inbounds %struct.pt, ptr %x, i64 0, i32 1
  %1 = load float, ptr %y, align 4, !tbaa !11
  %z = getelementptr inbounds %struct.pt, ptr %x, i64 0, i32 2
  %2 = load float, ptr %z, align 4, !tbaa !12
  %cmp62 = icmp sgt i32 %jnum, 0
  br i1 %cmp62, label %for.body.preheader, label %for.cond.cleanup

for.body.preheader:                               ; preds = %entry
  %wide.trip.count = zext i32 %jnum to i64
  %min.iters.check = icmp ult i32 %jnum, 16
  br i1 %min.iters.check, label %for.body.preheader75, label %vector.ph

vector.ph:                                        ; preds = %for.body.preheader
  %n.vec = and i64 %wide.trip.count, 4294967280
  %broadcast.splatinsert = insertelement <16 x float> poison, float %0, i64 0
  %broadcast.splat = shufflevector <16 x float> %broadcast.splatinsert, <16 x float> poison, <16 x i32> zeroinitializer
  %broadcast.splatinsert67 = insertelement <16 x float> poison, float %1, i64 0
  %broadcast.splat68 = shufflevector <16 x float> %broadcast.splatinsert67, <16 x float> poison, <16 x i32> zeroinitializer
  %broadcast.splatinsert70 = insertelement <16 x float> poison, float %2, i64 0
  %broadcast.splat71 = shufflevector <16 x float> %broadcast.splatinsert70, <16 x float> poison, <16 x i32> zeroinitializer
  %broadcast.splatinsert72 = insertelement <16 x float> poison, float %cut_coulsq, i64 0
  %broadcast.splat73 = shufflevector <16 x float> %broadcast.splatinsert72, <16 x float> poison, <16 x i32> zeroinitializer
  br label %vector.body

; CHECK-LABEL: .LBB0_6:
; CHECK:   vgatherdps      (%rdx,%zmm12), %zmm13 {%k1}
; CHECK:   vgatherdps      (%rdx,%zmm14), %zmm15 {%k1}
; CHECK:   vgatherdps      (%rdx,%zmm17), %zmm16 {%k1}

; OLD-LABEL: .LBB0_6:

; OLD:  vgatherqps      (%rdx,%zmm12), %ymm15 {%k1}
; OLD:  vgatherqps      (%rdx,%zmm11), %ymm12 {%k1}
; OLD:  vgatherqps      4(,%zmm14), %ymm12 {%k1}
; OLD:  vgatherqps      4(,%zmm13), %ymm15 {%k1}
; OLD:  vgatherqps      8(,%zmm14), %ymm15 {%k1}
; OLD:  vgatherqps      8(,%zmm13), %ymm16 {%k1}

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %pred.index = phi i32 [ 0, %vector.ph ], [ %predphi, %vector.body ]
  %3 = getelementptr inbounds i32, ptr %jlist, i64 %index
  %wide.load = load <16 x i32>, ptr %3, align 4, !tbaa !13
  %4 = and <16 x i32> %wide.load, <i32 536870911, i32 536870911, i32 536870911, i32 536870911, i32 536870911, i32 536870911, i32 536870911, i32 536870911, i32 536870911, i32 536870911, i32 536870911, i32 536870911, i32 536870911, i32 536870911, i32 536870911, i32 536870911>
  %5 = zext <16 x i32> %4 to <16 x i64>
  %6 = getelementptr inbounds %struct.pt, ptr %x, <16 x i64> %5
  %wide.masked.gather = tail call <16 x float> @llvm.masked.gather.v16f32.v16p0(<16 x ptr> %6, i32 4, <16 x i1> <i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true>, <16 x float> poison), !tbaa !5
  %7 = fsub <16 x float> %broadcast.splat, %wide.masked.gather
  %8 = getelementptr inbounds %struct.pt, ptr %x, <16 x i64> %5, i32 1
  %wide.masked.gather66 = tail call <16 x float> @llvm.masked.gather.v16f32.v16p0(<16 x ptr> %8, i32 4, <16 x i1> <i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true>, <16 x float> poison), !tbaa !11
  %9 = fsub <16 x float> %broadcast.splat68, %wide.masked.gather66
  %10 = getelementptr inbounds %struct.pt, ptr %x, <16 x i64> %5, i32 2
  %wide.masked.gather69 = tail call <16 x float> @llvm.masked.gather.v16f32.v16p0(<16 x ptr> %10, i32 4, <16 x i1> <i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true>, <16 x float> poison), !tbaa !12
  %11 = fsub <16 x float> %broadcast.splat71, %wide.masked.gather69
  %12 = fmul <16 x float> %9, %9
  %13 = tail call <16 x float> @llvm.fmuladd.v16f32(<16 x float> %7, <16 x float> %7, <16 x float> %12)
  %14 = tail call <16 x float> @llvm.fmuladd.v16f32(<16 x float> %11, <16 x float> %11, <16 x float> %13)
  %15 = fcmp olt <16 x float> %14, %broadcast.splat73
  %16 = sext i32 %pred.index to i64
  %17 = getelementptr float, ptr %trsq, i64 %16
  tail call void @llvm.masked.compressstore.v16f32(<16 x float> %14, ptr %17, <16 x i1> %15), !tbaa !14
  %18 = getelementptr float, ptr %tdelx, i64 %16
  tail call void @llvm.masked.compressstore.v16f32(<16 x float> %7, ptr %18, <16 x i1> %15), !tbaa !14
  %19 = getelementptr float, ptr %tdely, i64 %16
  tail call void @llvm.masked.compressstore.v16f32(<16 x float> %9, ptr %19, <16 x i1> %15), !tbaa !14
  %20 = getelementptr float, ptr %tdelz, i64 %16
  tail call void @llvm.masked.compressstore.v16f32(<16 x float> %11, ptr %20, <16 x i1> %15), !tbaa !14
  %21 = getelementptr inbounds %struct.pt, ptr %x, <16 x i64> %5, i32 3
  %wide.masked.gather74 = tail call <16 x i32> @llvm.masked.gather.v16i32.v16p0(<16 x ptr> %21, i32 4, <16 x i1> %15, <16 x i32> poison), !tbaa !15
  %22 = getelementptr i32, ptr %tjtype, i64 %16
  tail call void @llvm.masked.compressstore.v16i32(<16 x i32> %wide.masked.gather74, ptr %22, <16 x i1> %15), !tbaa !13
  %23 = getelementptr i32, ptr %tj, i64 %16
  tail call void @llvm.masked.compressstore.v16i32(<16 x i32> %wide.load, ptr %23, <16 x i1> %15), !tbaa !13
  %24 = bitcast <16 x i1> %15 to i16
  %mask.popcnt = tail call i16 @llvm.ctpop.i16(i16 %24), !range !16
  %popcnt.cmp.not = icmp eq i16 %24, 0
  %narrow = select i1 %popcnt.cmp.not, i16 0, i16 %mask.popcnt
  %popcnt.inc = zext i16 %narrow to i32
  %predphi = add i32 %pred.index, %popcnt.inc
  %index.next = add nuw i64 %index, 16
  %25 = icmp eq i64 %index.next, %n.vec
  br i1 %25, label %middle.block, label %vector.body, !llvm.loop !17

middle.block:                                     ; preds = %vector.body
  %cmp.n = icmp eq i64 %n.vec, %wide.trip.count
  br i1 %cmp.n, label %for.cond.cleanup, label %for.body.preheader75

for.body.preheader75:                             ; preds = %for.body.preheader, %middle.block
  %indvars.iv.ph = phi i64 [ 0, %for.body.preheader ], [ %n.vec, %middle.block ]
  %ej.064.ph = phi i32 [ 0, %for.body.preheader ], [ %predphi, %middle.block ]
  br label %for.body

for.cond.cleanup:                                 ; preds = %if.end, %middle.block, %entry
  %ej.0.lcssa = phi i32 [ 0, %entry ], [ %predphi, %middle.block ], [ %ej.1, %if.end ]
  ret i32 %ej.0.lcssa

for.body:                                         ; preds = %for.body.preheader75, %if.end
  %indvars.iv = phi i64 [ %indvars.iv.next, %if.end ], [ %indvars.iv.ph, %for.body.preheader75 ]
  %ej.064 = phi i32 [ %ej.1, %if.end ], [ %ej.064.ph, %for.body.preheader75 ]
  %arrayidx4 = getelementptr inbounds i32, ptr %jlist, i64 %indvars.iv
  %26 = load i32, ptr %arrayidx4, align 4, !tbaa !13
  %and = and i32 %26, 536870911
  %idxprom5 = zext i32 %and to i64
  %arrayidx6 = getelementptr inbounds %struct.pt, ptr %x, i64 %idxprom5
  %27 = load float, ptr %arrayidx6, align 4, !tbaa !5
  %sub = fsub float %0, %27
  %y10 = getelementptr inbounds %struct.pt, ptr %x, i64 %idxprom5, i32 1
  %28 = load float, ptr %y10, align 4, !tbaa !11
  %sub11 = fsub float %1, %28
  %z14 = getelementptr inbounds %struct.pt, ptr %x, i64 %idxprom5, i32 2
  %29 = load float, ptr %z14, align 4, !tbaa !12
  %sub15 = fsub float %2, %29
  %mul16 = fmul float %sub11, %sub11
  %30 = tail call float @llvm.fmuladd.f32(float %sub, float %sub, float %mul16)
  %31 = tail call float @llvm.fmuladd.f32(float %sub15, float %sub15, float %30)
  %cmp17 = fcmp olt float %31, %cut_coulsq
  br i1 %cmp17, label %if.then, label %if.end

if.then:                                          ; preds = %for.body
  %idxprom18 = sext i32 %ej.064 to i64
  %arrayidx19 = getelementptr inbounds float, ptr %trsq, i64 %idxprom18
  store float %31, ptr %arrayidx19, align 4, !tbaa !14
  %arrayidx21 = getelementptr inbounds float, ptr %tdelx, i64 %idxprom18
  store float %sub, ptr %arrayidx21, align 4, !tbaa !14
  %arrayidx23 = getelementptr inbounds float, ptr %tdely, i64 %idxprom18
  store float %sub11, ptr %arrayidx23, align 4, !tbaa !14
  %arrayidx25 = getelementptr inbounds float, ptr %tdelz, i64 %idxprom18
  store float %sub15, ptr %arrayidx25, align 4, !tbaa !14
  %w = getelementptr inbounds %struct.pt, ptr %x, i64 %idxprom5, i32 3
  %32 = load i32, ptr %w, align 4, !tbaa !15
  %arrayidx29 = getelementptr inbounds i32, ptr %tjtype, i64 %idxprom18
  store i32 %32, ptr %arrayidx29, align 4, !tbaa !13
  %arrayidx33 = getelementptr inbounds i32, ptr %tj, i64 %idxprom18
  store i32 %26, ptr %arrayidx33, align 4, !tbaa !13
  %inc = add nsw i32 %ej.064, 1
  br label %if.end

if.end:                                           ; preds = %if.then, %for.body
  %ej.1 = phi i32 [ %inc, %if.then ], [ %ej.064, %for.body ]
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, %wide.trip.count
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body, !llvm.loop !21
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.fmuladd.f32(float, float, float) #1

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(read)
declare <16 x float> @llvm.masked.gather.v16f32.v16p0(<16 x ptr>, i32 immarg, <16 x i1>, <16 x float>) #2

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare <16 x float> @llvm.fmuladd.v16f32(<16 x float>, <16 x float>, <16 x float>) #3

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: write)
declare void @llvm.masked.compressstore.v16f32(<16 x float>, ptr nocapture, <16 x i1>) #4

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(read)
declare <16 x i32> @llvm.masked.gather.v16i32.v16p0(<16 x ptr>, i32 immarg, <16 x i1>, <16 x i32>) #2

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: write)
declare void @llvm.masked.compressstore.v16i32(<16 x i32>, ptr nocapture, <16 x i1>) #4

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare i16 @llvm.ctpop.i16(i16) #3

attributes #0 = { nofree nosync nounwind memory(argmem: readwrite) uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="znver5" "target-features"="+adx,+aes,+avx,+avx2,+avx512bf16,+avx512bitalg,+avx512bw,+avx512cd,+avx512dq,+avx512f,+avx512ifma,+avx512vbmi,+avx512vbmi2,+avx512vl,+avx512vnni,+avx512vp2intersect,+avx512vpopcntdq,+avxvnni,+bmi,+bmi2,+clflushopt,+clwb,+clzero,+crc32,+cx16,+cx8,+f16c,+fma,+fsgsbase,+fxsr,+gfni,+invpcid,+lzcnt,+mmx,+movbe,+movdir64b,+movdiri,+mwaitx,+pclmul,+pku,+popcnt,+prefetchi,+prfchw,+rdpid,+rdpru,+rdrnd,+rdseed,+sahf,+sha,+shstk,+sse,+sse2,+sse3,+sse4.1,+sse4.2,+sse4a,+ssse3,+vaes,+vpclmulqdq,+wbnoinvd,+x87,+xsave,+xsavec,+xsaveopt,+xsaves" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #2 = { nocallback nofree nosync nounwind willreturn memory(read) }
attributes #3 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #4 = { nocallback nofree nosync nounwind willreturn memory(argmem: write) }

!llvm.module.flags = !{!0, !1, !2, !3}
!llvm.ident = !{!4}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 8, !"PIC Level", i32 2}
!2 = !{i32 7, !"PIE Level", i32 2}
!3 = !{i32 7, !"uwtable", i32 2}
!4 = !{!"clang version 17.0.6 (CLANG: Unknown-Revision)"}
!5 = !{!6, !7, i64 0}
!6 = !{!"pt", !7, i64 0, !7, i64 4, !7, i64 8, !10, i64 12}
!7 = !{!"float", !8, i64 0}
!8 = !{!"omnipotent char", !9, i64 0}
!9 = !{!"Simple C/C++ TBAA"}
!10 = !{!"int", !8, i64 0}
!11 = !{!6, !7, i64 4}
!12 = !{!6, !7, i64 8}
!13 = !{!10, !10, i64 0}
!14 = !{!7, !7, i64 0}
!15 = !{!6, !10, i64 12}
!16 = !{i16 0, i16 17}
!17 = distinct !{!17, !18, !19, !20}
!18 = !{!"llvm.loop.mustprogress"}
!19 = !{!"llvm.loop.isvectorized", i32 1}
!20 = !{!"llvm.loop.unroll.runtime.disable"}
!21 = distinct !{!21, !18, !20, !19}
