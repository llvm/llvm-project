; RUN: opt < %s -verify-ipgo -debug-only=verify-ipgo -passes=loop-unroll -S -disable-output 2>&1 | FileCheck %s
; RUN: opt < %s -verify-ipgo -passes=loop-unroll -S -disable-output 2>&1 | FileCheck %s --check-prefix=VERIFY
; REQUIRES: asserts

; CHECK-LABEL: *** IPGO Verification After LoopUnrollPass ***
; CHECK: PGOVerify# Not able to determine Block frequency for worker, block entry

; VERIFY-LABEL: *** IPGO Verification After LoopUnrollPass ***
; ModuleID = 'short.ll'
source_filename = "psimplex.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct.network = type { [200 x i8], [200 x i8], i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, double, i64, ptr, ptr, ptr, ptr, ptr, ptr, ptr, i64, i64, i64, i64, i64 }
%struct.basket = type { ptr, i64, i64, i64 }

@perm_p = external hidden unnamed_addr global ptr, align 8
@basket_sizes = external hidden unnamed_addr global ptr, align 8
@basket = external hidden unnamed_addr global ptr, align 8
@opt = external hidden unnamed_addr global i1, align 8
@opt_basket = external hidden unnamed_addr global ptr, align 8

; Function Attrs: nounwind uwtable
declare dso_local void @markBaskets(i64 noundef) local_unnamed_addr #0

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.start.p0(i64 immarg, ptr captures(none)) #1

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.end.p0(i64 immarg, ptr captures(none)) #1

; Function Attrs: nounwind uwtable
define dso_local void @worker(ptr noundef %net, i32 noundef %thread, i32 noundef %num_threads) local_unnamed_addr #0 {
entry:
  %perm = alloca [4061 x ptr], align 16
  %end_arc = alloca ptr, align 8
  %arcs1 = getelementptr inbounds nuw %struct.network, ptr %net, i64 0, i32 23
  %0 = load ptr, ptr %arcs1, align 8, !tbaa !5
  %stop_arcs2 = getelementptr inbounds nuw %struct.network, ptr %net, i64 0, i32 24
  %1 = load ptr, ptr %stop_arcs2, align 8, !tbaa !14
  %m3 = getelementptr inbounds nuw %struct.network, ptr %net, i64 0, i32 5
  %2 = load i64, ptr %m3, align 8, !tbaa !15
  %iterations4 = getelementptr inbounds nuw %struct.network, ptr %net, i64 0, i32 28
  call void @llvm.lifetime.start.p0(i64 32488, ptr nonnull %perm) #3
  call void @llvm.lifetime.start.p0(i64 8, ptr nonnull %end_arc) #3
  store ptr %0, ptr %end_arc, align 8, !tbaa !16
  %3 = load ptr, ptr @basket_sizes, align 8, !tbaa !17
  %idxprom = sext i32 %thread to i64
  %arrayidx = getelementptr inbounds i64, ptr %3, i64 %idxprom
  store i64 0, ptr %arrayidx, align 8, !tbaa !19
  %div = sdiv i32 4000, %num_threads
  %add6 = add nsw i32 %div, 61
  %add7 = add nsw i32 %div, 261
  %mul = mul nsw i32 %thread, %add7
  %add8 = add nsw i32 %mul, 1
  %conv = sext i32 %add8 to i64
  br label %for.cond

for.cond:                                         ; preds = %for.body, %entry
  %i.0 = phi i64 [ 1, %entry ], [ %inc, %for.body ]
  %j.0 = phi i64 [ %conv, %entry ], [ %inc16, %for.body ]
  %conv12 = sext i32 %add6 to i64
  %cmp = icmp slt i64 %i.0, %conv12
  br i1 %cmp, label %for.body, label %while.cond

for.body:                                         ; preds = %for.cond
  %4 = load ptr, ptr @basket, align 8, !tbaa !20
  %arrayidx14 = getelementptr inbounds %struct.basket, ptr %4, i64 %j.0
  %arrayidx15 = getelementptr inbounds nuw [4061 x ptr], ptr %perm, i64 0, i64 %i.0
  store ptr %arrayidx14, ptr %arrayidx15, align 8, !tbaa !20
  %inc = add nuw nsw i64 %i.0, 1
  %inc16 = add nsw i64 %j.0, 1
  br label %for.cond, !llvm.loop !22

while.cond:                                       ; preds = %if.end, %for.cond
  %.b = load i1, ptr @opt, align 1
  br i1 %.b, label %while.end, label %while.body

while.body:                                       ; preds = %while.cond
  %5 = load ptr, ptr @basket_sizes, align 8, !tbaa !17
  %6 = load i64, ptr %iterations4, align 8, !tbaa !19
  %add18 = add nsw i64 %6, %idxprom
  %conv19 = sext i32 %num_threads to i64
  %rem = srem i64 %add18, %conv19
  %max_elems = getelementptr inbounds nuw %struct.network, ptr %net, i64 0, i32 32
  %7 = load i64, ptr %max_elems, align 8, !tbaa !24
  %call = call ptr @primal_bea_mpp(i64 noundef %2, ptr noundef %0, ptr noundef %1, ptr noundef %5, ptr noundef nonnull %perm, i32 noundef %thread, ptr noundef nonnull %end_arc, i64 noundef %rem, i64 noundef %conv19, i64 noundef %7) #3
  %8 = load ptr, ptr @opt_basket, align 8, !tbaa !25
  %arrayidx22 = getelementptr inbounds ptr, ptr %8, i64 %idxprom
  store ptr %call, ptr %arrayidx22, align 8, !tbaa !20
  %add.ptr = getelementptr inbounds nuw ptr, ptr %perm, i64 1
  %9 = load ptr, ptr @perm_p, align 8, !tbaa !28
  %arrayidx25 = getelementptr inbounds ptr, ptr %9, i64 %idxprom
  store ptr %add.ptr, ptr %arrayidx25, align 8, !tbaa !25
  %cmp26 = icmp eq i32 %thread, 1
  br i1 %cmp26, label %if.then, label %if.end

if.then:                                          ; preds = %while.body
  call void @markBaskets(i64 noundef %conv19)
  br label %if.end

if.end:                                           ; preds = %if.then, %while.body
  br label %while.cond, !llvm.loop !31

while.end:                                        ; preds = %while.cond
  call void @llvm.lifetime.end.p0(i64 8, ptr nonnull %end_arc) #3
  call void @llvm.lifetime.end.p0(i64 32488, ptr nonnull %perm) #3
  ret void
}

declare dso_local ptr @primal_bea_mpp(i64 noundef, ptr noundef, ptr noundef, ptr noundef, ptr noundef, i32 noundef, ptr noundef, i64 noundef, i64 noundef, i64 noundef) local_unnamed_addr #2

attributes #0 = { nounwind uwtable "approx-func-fp-math"="true" "min-legal-vector-width"="0" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="znver4" "target-features"="+adx,+aes,+avx,+avx2,+avx512bf16,+avx512bitalg,+avx512bw,+avx512cd,+avx512dq,+avx512f,+avx512ifma,+avx512vbmi,+avx512vbmi2,+avx512vl,+avx512vnni,+avx512vpopcntdq,+bmi,+bmi2,+clflushopt,+clwb,+clzero,+crc32,+cx16,+cx8,+evex512,+f16c,+fma,+fsgsbase,+fxsr,+gfni,+invpcid,+lzcnt,+mmx,+movbe,+mwaitx,+pclmul,+pku,+popcnt,+prfchw,+rdpid,+rdpru,+rdrnd,+rdseed,+sahf,+sha,+shstk,+sse,+sse2,+sse3,+sse4.1,+sse4.2,+sse4a,+ssse3,+vaes,+vpclmulqdq,+wbnoinvd,+x87,+xsave,+xsavec,+xsaveopt,+xsaves" "unsafe-fp-math"="true" }
attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
attributes #2 = { "approx-func-fp-math"="true" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="znver4" "target-features"="+adx,+aes,+avx,+avx2,+avx512bf16,+avx512bitalg,+avx512bw,+avx512cd,+avx512dq,+avx512f,+avx512ifma,+avx512vbmi,+avx512vbmi2,+avx512vl,+avx512vnni,+avx512vpopcntdq,+bmi,+bmi2,+clflushopt,+clwb,+clzero,+crc32,+cx16,+cx8,+evex512,+f16c,+fma,+fsgsbase,+fxsr,+gfni,+invpcid,+lzcnt,+mmx,+movbe,+mwaitx,+pclmul,+pku,+popcnt,+prfchw,+rdpid,+rdpru,+rdrnd,+rdseed,+sahf,+sha,+shstk,+sse,+sse2,+sse3,+sse4.1,+sse4.2,+sse4a,+ssse3,+vaes,+vpclmulqdq,+wbnoinvd,+x87,+xsave,+xsavec,+xsaveopt,+xsaves" "unsafe-fp-math"="true" }
attributes #3 = { nounwind }

!llvm.module.flags = !{!0, !1, !2, !3, !32}
!llvm.ident = !{!4}

!32 = !{i32 1, !"ProfileSummary", !33}
!33 = !{!34, !35, !36, !37, !38, !39, !40, !41}
!34 = !{!"ProfileFormat", !"InstrProf"}
!35 = !{!"TotalCount", i64 1}
!36 = !{!"MaxCount", i64 1}
!37 = !{!"MaxInternalCount", i64 1}
!38 = !{!"MaxFunctionCount", i64 1}
!39 = !{!"NumCounts", i64 1}
!40 = !{!"NumFunctions", i64 1}
!41 = !{!"DetailedSummary", !42}
!42 = !{!43}
!43 = !{i32 10000, i64 1, i32 1}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 7, !"uwtable", i32 2}
!2 = !{i32 1, !"ThinLTO", i32 0}
!3 = !{i32 1, !"EnableSplitLTOUnit", i32 1}
!4 = !{!"clang version 21.1.8 (CLANG: Unknown-Revision)"}
!5 = !{!6, !13, i64 568, i64 8}
!6 = !{!7, i64 648, !"network", !7, i64 0, i64 200, !7, i64 200, i64 200, !9, i64 400, i64 8, !9, i64 408, i64 8, !9, i64 416, i64 8, !9, i64 424, i64 8, !9, i64 432, i64 8, !9, i64 440, i64 8, !9, i64 448, i64 8, !9, i64 456, i64 8, !9, i64 464, i64 8, !9, i64 472, i64 8, !9, i64 480, i64 8, !9, i64 488, i64 8, !9, i64 496, i64 8, !9, i64 504, i64 8, !9, i64 512, i64 8, !9, i64 520, i64 8, !9, i64 528, i64 8, !10, i64 536, i64 8, !9, i64 544, i64 8, !11, i64 552, i64 8, !11, i64 560, i64 8, !13, i64 568, i64 8, !13, i64 576, i64 8, !13, i64 584, i64 8, !13, i64 592, i64 8, !13, i64 600, i64 8, !9, i64 608, i64 8, !9, i64 616, i64 8, !9, i64 624, i64 8, !9, i64 632, i64 8, !9, i64 640, i64 8}
!7 = !{!8, i64 1, !"omnipotent char"}
!8 = !{!"Simple C/C++ TBAA"}
!9 = !{!7, i64 8, !"long"}
!10 = !{!7, i64 8, !"double"}
!11 = !{!12, i64 8, !"p1 _ZTS4node"}
!12 = !{!7, i64 8, !"any pointer"}
!13 = !{!12, i64 8, !"p1 _ZTS3arc"}
!14 = !{!6, !13, i64 576, i64 8}
!15 = !{!6, !9, i64 424, i64 8}
!16 = !{!13, !13, i64 0, i64 8}
!17 = !{!18, !18, i64 0, i64 8}
!18 = !{!12, i64 8, !"p1 long"}
!19 = !{!9, !9, i64 0, i64 8}
!20 = !{!21, !21, i64 0, i64 8}
!21 = !{!12, i64 8, !"p1 _ZTS6basket"}
!22 = distinct !{!22, !23}
!23 = !{!"llvm.loop.mustprogress"}
!24 = !{!6, !9, i64 640, i64 8}
!25 = !{!26, !26, i64 0, i64 8}
!26 = !{!27, i64 8, !"p2 _ZTS6basket"}
!27 = !{!12, i64 8, !"any p2 pointer"}
!28 = !{!29, !29, i64 0, i64 8}
!29 = !{!30, i64 8, !"p3 _ZTS6basket"}
!30 = !{!27, i64 8, !"any p3 pointer"}
!31 = distinct !{!31, !23}
