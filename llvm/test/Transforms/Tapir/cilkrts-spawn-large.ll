; RUN: llc %s -o - | FileCheck %s
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux_gnu"

%struct.__cilkrts_stack_frame = type { i32, i32, %struct.__cilkrts_stack_frame*, %struct.__cilkrts_worker*, i8*, [5 x i8*], i32, i16, i16, { %struct.__cilkrts_pedigree } }
%struct.__cilkrts_worker = type { %struct.__cilkrts_stack_frame**, %struct.__cilkrts_stack_frame**, %struct.__cilkrts_stack_frame**, %struct.__cilkrts_stack_frame**, %struct.__cilkrts_stack_frame**, i32, i8*, i8*, i8*, %struct.__cilkrts_stack_frame*, i8*, i8*, %struct.__cilkrts_pedigree }
%struct.__cilkrts_pedigree = type { i64, %struct.__cilkrts_pedigree* }

; Function Attrs: argmemonly nounwind
define void @cluster_25__XlaCompiledKernel_true__XlaNumConstantArgs_1__XlaNumResourceArgs_0_.42(i8* nocapture readnone %retval, i8* noalias nocapture readnone %run_options, i8** noalias nocapture readnone %params, i8** noalias nocapture readonly %buffer_table, i64* noalias nocapture readnone %prof_counters) local_unnamed_addr #0 {
entry:
  %0 = getelementptr inbounds i8*, i8** %buffer_table, i64 7
  %1 = bitcast i8** %0 to [20 x [200 x float]]**
  %2 = load [20 x [200 x float]]*, [20 x [200 x float]]** %1, align 8, !invariant.load !0, !dereferenceable !1, !align !2
  %3 = getelementptr inbounds i8*, i8** %buffer_table, i64 8
  %4 = bitcast i8** %3 to [20 x [200 x float]]**
  %5 = load [20 x [200 x float]]*, [20 x [200 x float]]** %4, align 8, !invariant.load !0, !dereferenceable !1, !align !2
  %6 = getelementptr inbounds i8*, i8** %buffer_table, i64 1
  %7 = load i8*, i8** %6, align 8, !invariant.load !0, !dereferenceable !3, !align !2
  %concatenate.9 = bitcast i8* %7 to [20 x [400 x float]]*
  %8 = bitcast [20 x [200 x float]]* %2 to i8*
  tail call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %7, i8* align 16 %8, i64 800, i1 false)
  %9 = bitcast [20 x [200 x float]]* %5 to i8*
  %10 = getelementptr i8, i8* %7, i64 800
  tail call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %10, i8* align 16 %9, i64 800, i1 false)
  %target_region.1 = getelementptr inbounds i8, i8* %7, i64 1600
  %src_addr.1 = getelementptr inbounds [20 x [200 x float]], [20 x [200 x float]]* %2, i64 0, i64 1, i64 0
  %11 = bitcast float* %src_addr.1 to i8*
  tail call void @llvm.memcpy.p0i8.p0i8.i64(i8* nonnull align 16 %target_region.1, i8* nonnull align 16 %11, i64 800, i1 false)
  %src_addr1.1 = getelementptr inbounds [20 x [200 x float]], [20 x [200 x float]]* %5, i64 0, i64 1, i64 0
  %12 = bitcast float* %src_addr1.1 to i8*
  %13 = getelementptr i8, i8* %7, i64 2400
  tail call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %13, i8* nonnull align 16 %12, i64 800, i1 false)
  %target_region.2 = getelementptr inbounds i8, i8* %7, i64 3200
  %src_addr.2 = getelementptr inbounds [20 x [200 x float]], [20 x [200 x float]]* %2, i64 0, i64 2, i64 0
  %14 = bitcast float* %src_addr.2 to i8*
  tail call void @llvm.memcpy.p0i8.p0i8.i64(i8* nonnull align 16 %target_region.2, i8* nonnull align 16 %14, i64 800, i1 false)
  %src_addr1.2 = getelementptr inbounds [20 x [200 x float]], [20 x [200 x float]]* %5, i64 0, i64 2, i64 0
  %15 = bitcast float* %src_addr1.2 to i8*
  %16 = getelementptr i8, i8* %7, i64 4000
  tail call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %16, i8* nonnull align 16 %15, i64 800, i1 false)
  %target_region.3 = getelementptr inbounds i8, i8* %7, i64 4800
  %src_addr.3 = getelementptr inbounds [20 x [200 x float]], [20 x [200 x float]]* %2, i64 0, i64 3, i64 0
  %17 = bitcast float* %src_addr.3 to i8*
  tail call void @llvm.memcpy.p0i8.p0i8.i64(i8* nonnull align 16 %target_region.3, i8* nonnull align 16 %17, i64 800, i1 false)
  %src_addr1.3 = getelementptr inbounds [20 x [200 x float]], [20 x [200 x float]]* %5, i64 0, i64 3, i64 0
  %18 = bitcast float* %src_addr1.3 to i8*
  %19 = getelementptr i8, i8* %7, i64 5600
  tail call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %19, i8* nonnull align 16 %18, i64 800, i1 false)
  %target_region.4 = getelementptr inbounds i8, i8* %7, i64 6400
  %src_addr.4 = getelementptr inbounds [20 x [200 x float]], [20 x [200 x float]]* %2, i64 0, i64 4, i64 0
  %20 = bitcast float* %src_addr.4 to i8*
  tail call void @llvm.memcpy.p0i8.p0i8.i64(i8* nonnull align 16 %target_region.4, i8* nonnull align 16 %20, i64 800, i1 false)
  %src_addr1.4 = getelementptr inbounds [20 x [200 x float]], [20 x [200 x float]]* %5, i64 0, i64 4, i64 0
  %21 = bitcast float* %src_addr1.4 to i8*
  %22 = getelementptr i8, i8* %7, i64 7200
  tail call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %22, i8* nonnull align 16 %21, i64 800, i1 false)
  %target_region.5 = getelementptr inbounds i8, i8* %7, i64 8000
  %src_addr.5 = getelementptr inbounds [20 x [200 x float]], [20 x [200 x float]]* %2, i64 0, i64 5, i64 0
  %23 = bitcast float* %src_addr.5 to i8*
  tail call void @llvm.memcpy.p0i8.p0i8.i64(i8* nonnull align 16 %target_region.5, i8* nonnull align 16 %23, i64 800, i1 false)
  %src_addr1.5 = getelementptr inbounds [20 x [200 x float]], [20 x [200 x float]]* %5, i64 0, i64 5, i64 0
  %24 = bitcast float* %src_addr1.5 to i8*
  %25 = getelementptr i8, i8* %7, i64 8800
  tail call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %25, i8* nonnull align 16 %24, i64 800, i1 false)
  %target_region.6 = getelementptr inbounds i8, i8* %7, i64 9600
  %src_addr.6 = getelementptr inbounds [20 x [200 x float]], [20 x [200 x float]]* %2, i64 0, i64 6, i64 0
  %26 = bitcast float* %src_addr.6 to i8*
  tail call void @llvm.memcpy.p0i8.p0i8.i64(i8* nonnull align 16 %target_region.6, i8* nonnull align 16 %26, i64 800, i1 false)
  %src_addr1.6 = getelementptr inbounds [20 x [200 x float]], [20 x [200 x float]]* %5, i64 0, i64 6, i64 0
  %27 = bitcast float* %src_addr1.6 to i8*
  %28 = getelementptr i8, i8* %7, i64 10400
  tail call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %28, i8* nonnull align 16 %27, i64 800, i1 false)
  %target_region.7 = getelementptr inbounds i8, i8* %7, i64 11200
  %src_addr.7 = getelementptr inbounds [20 x [200 x float]], [20 x [200 x float]]* %2, i64 0, i64 7, i64 0
  %29 = bitcast float* %src_addr.7 to i8*
  tail call void @llvm.memcpy.p0i8.p0i8.i64(i8* nonnull align 16 %target_region.7, i8* nonnull align 16 %29, i64 800, i1 false)
  %src_addr1.7 = getelementptr inbounds [20 x [200 x float]], [20 x [200 x float]]* %5, i64 0, i64 7, i64 0
  %30 = bitcast float* %src_addr1.7 to i8*
  %31 = getelementptr i8, i8* %7, i64 12000
  tail call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %31, i8* nonnull align 16 %30, i64 800, i1 false)
  %target_region.8 = getelementptr inbounds i8, i8* %7, i64 12800
  %src_addr.8 = getelementptr inbounds [20 x [200 x float]], [20 x [200 x float]]* %2, i64 0, i64 8, i64 0
  %32 = bitcast float* %src_addr.8 to i8*
  tail call void @llvm.memcpy.p0i8.p0i8.i64(i8* nonnull align 16 %target_region.8, i8* nonnull align 16 %32, i64 800, i1 false)
  %src_addr1.8 = getelementptr inbounds [20 x [200 x float]], [20 x [200 x float]]* %5, i64 0, i64 8, i64 0
  %33 = bitcast float* %src_addr1.8 to i8*
  %34 = getelementptr i8, i8* %7, i64 13600
  tail call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %34, i8* nonnull align 16 %33, i64 800, i1 false)
  %target_region.9 = getelementptr inbounds i8, i8* %7, i64 14400
  %src_addr.9 = getelementptr inbounds [20 x [200 x float]], [20 x [200 x float]]* %2, i64 0, i64 9, i64 0
  %35 = bitcast float* %src_addr.9 to i8*
  tail call void @llvm.memcpy.p0i8.p0i8.i64(i8* nonnull align 16 %target_region.9, i8* nonnull align 16 %35, i64 800, i1 false)
  %src_addr1.9 = getelementptr inbounds [20 x [200 x float]], [20 x [200 x float]]* %5, i64 0, i64 9, i64 0
  %36 = bitcast float* %src_addr1.9 to i8*
  %37 = getelementptr i8, i8* %7, i64 15200
  tail call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %37, i8* nonnull align 16 %36, i64 800, i1 false)
  %target_region.10 = getelementptr inbounds i8, i8* %7, i64 16000
  %src_addr.10 = getelementptr inbounds [20 x [200 x float]], [20 x [200 x float]]* %2, i64 0, i64 10, i64 0
  %38 = bitcast float* %src_addr.10 to i8*
  tail call void @llvm.memcpy.p0i8.p0i8.i64(i8* nonnull align 16 %target_region.10, i8* nonnull align 16 %38, i64 800, i1 false)
  %src_addr1.10 = getelementptr inbounds [20 x [200 x float]], [20 x [200 x float]]* %5, i64 0, i64 10, i64 0
  %39 = bitcast float* %src_addr1.10 to i8*
  %40 = getelementptr i8, i8* %7, i64 16800
  tail call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %40, i8* nonnull align 16 %39, i64 800, i1 false)
  %target_region.11 = getelementptr inbounds i8, i8* %7, i64 17600
  %src_addr.11 = getelementptr inbounds [20 x [200 x float]], [20 x [200 x float]]* %2, i64 0, i64 11, i64 0
  %41 = bitcast float* %src_addr.11 to i8*
  tail call void @llvm.memcpy.p0i8.p0i8.i64(i8* nonnull align 16 %target_region.11, i8* nonnull align 16 %41, i64 800, i1 false)
  %src_addr1.11 = getelementptr inbounds [20 x [200 x float]], [20 x [200 x float]]* %5, i64 0, i64 11, i64 0
  %42 = bitcast float* %src_addr1.11 to i8*
  %43 = getelementptr i8, i8* %7, i64 18400
  tail call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %43, i8* nonnull align 16 %42, i64 800, i1 false)
  %target_region.12 = getelementptr inbounds i8, i8* %7, i64 19200
  %src_addr.12 = getelementptr inbounds [20 x [200 x float]], [20 x [200 x float]]* %2, i64 0, i64 12, i64 0
  %44 = bitcast float* %src_addr.12 to i8*
  tail call void @llvm.memcpy.p0i8.p0i8.i64(i8* nonnull align 16 %target_region.12, i8* nonnull align 16 %44, i64 800, i1 false)
  %src_addr1.12 = getelementptr inbounds [20 x [200 x float]], [20 x [200 x float]]* %5, i64 0, i64 12, i64 0
  %45 = bitcast float* %src_addr1.12 to i8*
  %46 = getelementptr i8, i8* %7, i64 20000
  tail call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %46, i8* nonnull align 16 %45, i64 800, i1 false)
  %target_region.13 = getelementptr inbounds i8, i8* %7, i64 20800
  %src_addr.13 = getelementptr inbounds [20 x [200 x float]], [20 x [200 x float]]* %2, i64 0, i64 13, i64 0
  %47 = bitcast float* %src_addr.13 to i8*
  tail call void @llvm.memcpy.p0i8.p0i8.i64(i8* nonnull align 16 %target_region.13, i8* nonnull align 16 %47, i64 800, i1 false)
  %src_addr1.13 = getelementptr inbounds [20 x [200 x float]], [20 x [200 x float]]* %5, i64 0, i64 13, i64 0
  %48 = bitcast float* %src_addr1.13 to i8*
  %49 = getelementptr i8, i8* %7, i64 21600
  tail call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %49, i8* nonnull align 16 %48, i64 800, i1 false)
  %target_region.14 = getelementptr inbounds i8, i8* %7, i64 22400
  %src_addr.14 = getelementptr inbounds [20 x [200 x float]], [20 x [200 x float]]* %2, i64 0, i64 14, i64 0
  %50 = bitcast float* %src_addr.14 to i8*
  tail call void @llvm.memcpy.p0i8.p0i8.i64(i8* nonnull align 16 %target_region.14, i8* nonnull align 16 %50, i64 800, i1 false)
  %src_addr1.14 = getelementptr inbounds [20 x [200 x float]], [20 x [200 x float]]* %5, i64 0, i64 14, i64 0
  %51 = bitcast float* %src_addr1.14 to i8*
  %52 = getelementptr i8, i8* %7, i64 23200
  tail call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %52, i8* nonnull align 16 %51, i64 800, i1 false)
  %target_region.15 = getelementptr inbounds i8, i8* %7, i64 24000
  %src_addr.15 = getelementptr inbounds [20 x [200 x float]], [20 x [200 x float]]* %2, i64 0, i64 15, i64 0
  %53 = bitcast float* %src_addr.15 to i8*
  tail call void @llvm.memcpy.p0i8.p0i8.i64(i8* nonnull align 16 %target_region.15, i8* nonnull align 16 %53, i64 800, i1 false)
  %src_addr1.15 = getelementptr inbounds [20 x [200 x float]], [20 x [200 x float]]* %5, i64 0, i64 15, i64 0
  %54 = bitcast float* %src_addr1.15 to i8*
  %55 = getelementptr i8, i8* %7, i64 24800
  tail call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %55, i8* nonnull align 16 %54, i64 800, i1 false)
  %target_region.16 = getelementptr inbounds i8, i8* %7, i64 25600
  %src_addr.16 = getelementptr inbounds [20 x [200 x float]], [20 x [200 x float]]* %2, i64 0, i64 16, i64 0
  %56 = bitcast float* %src_addr.16 to i8*
  tail call void @llvm.memcpy.p0i8.p0i8.i64(i8* nonnull align 16 %target_region.16, i8* nonnull align 16 %56, i64 800, i1 false)
  %src_addr1.16 = getelementptr inbounds [20 x [200 x float]], [20 x [200 x float]]* %5, i64 0, i64 16, i64 0
  %57 = bitcast float* %src_addr1.16 to i8*
  %58 = getelementptr i8, i8* %7, i64 26400
  tail call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %58, i8* nonnull align 16 %57, i64 800, i1 false)
  %target_region.17 = getelementptr inbounds i8, i8* %7, i64 27200
  %src_addr.17 = getelementptr inbounds [20 x [200 x float]], [20 x [200 x float]]* %2, i64 0, i64 17, i64 0
  %59 = bitcast float* %src_addr.17 to i8*
  tail call void @llvm.memcpy.p0i8.p0i8.i64(i8* nonnull align 16 %target_region.17, i8* nonnull align 16 %59, i64 800, i1 false)
  %src_addr1.17 = getelementptr inbounds [20 x [200 x float]], [20 x [200 x float]]* %5, i64 0, i64 17, i64 0
  %60 = bitcast float* %src_addr1.17 to i8*
  %61 = getelementptr i8, i8* %7, i64 28000
  tail call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %61, i8* nonnull align 16 %60, i64 800, i1 false)
  %target_region.18 = getelementptr inbounds i8, i8* %7, i64 28800
  %src_addr.18 = getelementptr inbounds [20 x [200 x float]], [20 x [200 x float]]* %2, i64 0, i64 18, i64 0
  %62 = bitcast float* %src_addr.18 to i8*
  tail call void @llvm.memcpy.p0i8.p0i8.i64(i8* nonnull align 16 %target_region.18, i8* nonnull align 16 %62, i64 800, i1 false)
  %src_addr1.18 = getelementptr inbounds [20 x [200 x float]], [20 x [200 x float]]* %5, i64 0, i64 18, i64 0
  %63 = bitcast float* %src_addr1.18 to i8*
  %64 = getelementptr i8, i8* %7, i64 29600
  tail call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %64, i8* nonnull align 16 %63, i64 800, i1 false)
  %target_region.19 = getelementptr inbounds i8, i8* %7, i64 30400
  %src_addr.19 = getelementptr inbounds [20 x [200 x float]], [20 x [200 x float]]* %2, i64 0, i64 19, i64 0
  %65 = bitcast float* %src_addr.19 to i8*
  tail call void @llvm.memcpy.p0i8.p0i8.i64(i8* nonnull align 16 %target_region.19, i8* nonnull align 16 %65, i64 800, i1 false)
  %src_addr1.19 = getelementptr inbounds [20 x [200 x float]], [20 x [200 x float]]* %5, i64 0, i64 19, i64 0
  %66 = bitcast float* %src_addr1.19 to i8*
  %67 = getelementptr i8, i8* %7, i64 31200
  tail call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %67, i8* nonnull align 16 %66, i64 800, i1 false)
  %68 = bitcast i8** %buffer_table to [400 x [800 x float]]**
  %69 = load [400 x [800 x float]]*, [400 x [800 x float]]** %68, align 8, !invariant.load !0, !dereferenceable !4, !align !2
  %70 = getelementptr inbounds i8*, i8** %buffer_table, i64 12
  %71 = bitcast i8** %70 to [20 x [800 x float]]**
  %72 = load [20 x [800 x float]]*, [20 x [800 x float]]** %71, align 8, !invariant.load !0, !dereferenceable !5, !align !2
  %73 = tail call i32 @__cilkrts_get_nworkers()
  %74 = shl i32 %73, 3
  %75 = zext i32 %74 to i64
  %76 = add nuw nsw i64 %75, 19
  %77 = udiv i64 %76, %75
  %78 = icmp ult i64 %77, 2048
  %79 = select i1 %78, i64 %77, i64 2048
  tail call fastcc void @cluster_25__XlaCompiledKernel_true__XlaNumConstantArgs_1__XlaNumResourceArgs_0_.42.outline_dot.10.loop_detach.lhs.0.ls1(i64 0, i64 20, i64 %79, [20 x [400 x float]]* %concatenate.9, [400 x [800 x float]]* %69, [20 x [800 x float]]* %72) #6
  %80 = getelementptr inbounds i8*, i8** %buffer_table, i64 9
  %81 = bitcast i8** %80 to [800 x float]**
  %82 = load [800 x float]*, [800 x float]** %81, align 8, !invariant.load !0, !dereferenceable !6, !align !2
  %83 = getelementptr inbounds i8*, i8** %buffer_table, i64 3
  %84 = load i8*, i8** %83, align 8, !invariant.load !0, !dereferenceable !1, !align !2
  %fusion.2 = bitcast i8* %84 to [20 x [200 x float]]*
  %scevgep71 = getelementptr [800 x float], [800 x float]* %82, i64 0, i64 200
  %scevgep7172 = bitcast float* %scevgep71 to i8*
  %scevgep73 = getelementptr [800 x float], [800 x float]* %82, i64 0, i64 400
  %scevgep7374 = bitcast float* %scevgep73 to i8*
  %85 = bitcast float* %scevgep71 to <8 x float>*
  %wide.load90 = load <8 x float>, <8 x float>* %85, align 16
  %86 = getelementptr inbounds [800 x float], [800 x float]* %82, i64 0, i64 208
  %87 = bitcast float* %86 to <8 x float>*
  %wide.load91 = load <8 x float>, <8 x float>* %87, align 16
  %88 = getelementptr inbounds [800 x float], [800 x float]* %82, i64 0, i64 216
  %89 = bitcast float* %88 to <8 x float>*
  %wide.load92 = load <8 x float>, <8 x float>* %89, align 16
  %90 = getelementptr inbounds [800 x float], [800 x float]* %82, i64 0, i64 224
  %91 = bitcast float* %90 to <8 x float>*
  %wide.load93 = load <8 x float>, <8 x float>* %91, align 16
  %92 = getelementptr inbounds [800 x float], [800 x float]* %82, i64 0, i64 232
  %93 = bitcast float* %92 to <8 x float>*
  %wide.load90.1 = load <8 x float>, <8 x float>* %93, align 16
  %94 = getelementptr inbounds [800 x float], [800 x float]* %82, i64 0, i64 240
  %95 = bitcast float* %94 to <8 x float>*
  %wide.load91.1 = load <8 x float>, <8 x float>* %95, align 16
  %96 = getelementptr inbounds [800 x float], [800 x float]* %82, i64 0, i64 248
  %97 = bitcast float* %96 to <8 x float>*
  %wide.load92.1 = load <8 x float>, <8 x float>* %97, align 16
  %98 = getelementptr inbounds [800 x float], [800 x float]* %82, i64 0, i64 256
  %99 = bitcast float* %98 to <8 x float>*
  %wide.load93.1 = load <8 x float>, <8 x float>* %99, align 16
  %100 = getelementptr inbounds [800 x float], [800 x float]* %82, i64 0, i64 264
  %101 = bitcast float* %100 to <8 x float>*
  %wide.load90.2 = load <8 x float>, <8 x float>* %101, align 16
  %102 = getelementptr inbounds [800 x float], [800 x float]* %82, i64 0, i64 272
  %103 = bitcast float* %102 to <8 x float>*
  %wide.load91.2 = load <8 x float>, <8 x float>* %103, align 16
  %104 = getelementptr inbounds [800 x float], [800 x float]* %82, i64 0, i64 280
  %105 = bitcast float* %104 to <8 x float>*
  %wide.load92.2 = load <8 x float>, <8 x float>* %105, align 16
  %106 = getelementptr inbounds [800 x float], [800 x float]* %82, i64 0, i64 288
  %107 = bitcast float* %106 to <8 x float>*
  %wide.load93.2 = load <8 x float>, <8 x float>* %107, align 16
  %108 = getelementptr inbounds [800 x float], [800 x float]* %82, i64 0, i64 296
  %109 = bitcast float* %108 to <8 x float>*
  %wide.load90.3 = load <8 x float>, <8 x float>* %109, align 16
  %110 = getelementptr inbounds [800 x float], [800 x float]* %82, i64 0, i64 304
  %111 = bitcast float* %110 to <8 x float>*
  %wide.load91.3 = load <8 x float>, <8 x float>* %111, align 16
  %112 = getelementptr inbounds [800 x float], [800 x float]* %82, i64 0, i64 312
  %113 = bitcast float* %112 to <8 x float>*
  %wide.load92.3 = load <8 x float>, <8 x float>* %113, align 16
  %114 = getelementptr inbounds [800 x float], [800 x float]* %82, i64 0, i64 320
  %115 = bitcast float* %114 to <8 x float>*
  %wide.load93.3 = load <8 x float>, <8 x float>* %115, align 16
  %116 = getelementptr inbounds [800 x float], [800 x float]* %82, i64 0, i64 328
  %117 = bitcast float* %116 to <8 x float>*
  %wide.load90.4 = load <8 x float>, <8 x float>* %117, align 16
  %118 = getelementptr inbounds [800 x float], [800 x float]* %82, i64 0, i64 336
  %119 = bitcast float* %118 to <8 x float>*
  %wide.load91.4 = load <8 x float>, <8 x float>* %119, align 16
  %120 = getelementptr inbounds [800 x float], [800 x float]* %82, i64 0, i64 344
  %121 = bitcast float* %120 to <8 x float>*
  %wide.load92.4 = load <8 x float>, <8 x float>* %121, align 16
  %122 = getelementptr inbounds [800 x float], [800 x float]* %82, i64 0, i64 352
  %123 = bitcast float* %122 to <8 x float>*
  %wide.load93.4 = load <8 x float>, <8 x float>* %123, align 16
  %124 = getelementptr inbounds [800 x float], [800 x float]* %82, i64 0, i64 360
  %125 = bitcast float* %124 to <8 x float>*
  %wide.load90.5 = load <8 x float>, <8 x float>* %125, align 16
  %126 = getelementptr inbounds [800 x float], [800 x float]* %82, i64 0, i64 368
  %127 = bitcast float* %126 to <8 x float>*
  %wide.load91.5 = load <8 x float>, <8 x float>* %127, align 16
  %128 = getelementptr inbounds [800 x float], [800 x float]* %82, i64 0, i64 376
  %129 = bitcast float* %128 to <8 x float>*
  %wide.load92.5 = load <8 x float>, <8 x float>* %129, align 16
  %130 = getelementptr inbounds [800 x float], [800 x float]* %82, i64 0, i64 384
  %131 = bitcast float* %130 to <8 x float>*
  %wide.load93.5 = load <8 x float>, <8 x float>* %131, align 16
  tail call fastcc void @cluster_25__XlaCompiledKernel_true__XlaNumConstantArgs_1__XlaNumResourceArgs_0_.42.outline_fusion.2.loop_detach.dim.0.ls1(i64 0, i64 20, i64 %79, <8 x float> %wide.load91.2, <8 x float> %wide.load92.5, <8 x float> %wide.load91.5, <8 x float> %wide.load90.5, <8 x float> %wide.load93.4, <8 x float> %wide.load92.4, <8 x float> %wide.load91.4, <8 x float> %wide.load90.4, <8 x float> %wide.load93.3, <8 x float> %wide.load92.3, <8 x float> %wide.load91.3, <8 x float> %wide.load90.3, <8 x float> %wide.load93.2, <8 x float> %wide.load92.2, <8 x float> %wide.load90.2, <8 x float> %wide.load93.1, <8 x float> %wide.load92.1, <8 x float> %wide.load91.1, <8 x float> %wide.load90.1, <8 x float> %wide.load93.5, <8 x float> %wide.load93, <8 x float> %wide.load92, <8 x float> %wide.load91, <8 x float> %wide.load90, i8* %84, [800 x float]* %82, i8* %scevgep7374, [20 x [200 x float]]* %fusion.2, [20 x [800 x float]]* %72, i8* %scevgep7172) #6
  %132 = bitcast [800 x float]* %82 to i8*
  %133 = getelementptr inbounds i8*, i8** %buffer_table, i64 6
  %134 = load i8*, i8** %133, align 8, !invariant.load !0, !dereferenceable !1, !align !2
  %fusion.4 = bitcast i8* %134 to [20 x [200 x float]]*
  %135 = bitcast [800 x float]* %82 to <8 x float>*
  %wide.load130 = load <8 x float>, <8 x float>* %135, align 16
  %136 = getelementptr inbounds [800 x float], [800 x float]* %82, i64 0, i64 8
  %137 = bitcast float* %136 to <8 x float>*
  %wide.load131 = load <8 x float>, <8 x float>* %137, align 16
  %138 = getelementptr inbounds [800 x float], [800 x float]* %82, i64 0, i64 16
  %139 = bitcast float* %138 to <8 x float>*
  %wide.load132 = load <8 x float>, <8 x float>* %139, align 16
  %140 = getelementptr inbounds [800 x float], [800 x float]* %82, i64 0, i64 24
  %141 = bitcast float* %140 to <8 x float>*
  %wide.load133 = load <8 x float>, <8 x float>* %141, align 16
  %142 = getelementptr inbounds [800 x float], [800 x float]* %82, i64 0, i64 32
  %143 = bitcast float* %142 to <8 x float>*
  %wide.load130.1 = load <8 x float>, <8 x float>* %143, align 16
  %144 = getelementptr inbounds [800 x float], [800 x float]* %82, i64 0, i64 40
  %145 = bitcast float* %144 to <8 x float>*
  %wide.load131.1 = load <8 x float>, <8 x float>* %145, align 16
  %146 = getelementptr inbounds [800 x float], [800 x float]* %82, i64 0, i64 48
  %147 = bitcast float* %146 to <8 x float>*
  %wide.load132.1 = load <8 x float>, <8 x float>* %147, align 16
  %148 = getelementptr inbounds [800 x float], [800 x float]* %82, i64 0, i64 56
  %149 = bitcast float* %148 to <8 x float>*
  %wide.load133.1 = load <8 x float>, <8 x float>* %149, align 16
  %150 = getelementptr inbounds [800 x float], [800 x float]* %82, i64 0, i64 64
  %151 = bitcast float* %150 to <8 x float>*
  %wide.load130.2 = load <8 x float>, <8 x float>* %151, align 16
  %152 = getelementptr inbounds [800 x float], [800 x float]* %82, i64 0, i64 72
  %153 = bitcast float* %152 to <8 x float>*
  %wide.load131.2 = load <8 x float>, <8 x float>* %153, align 16
  %154 = getelementptr inbounds [800 x float], [800 x float]* %82, i64 0, i64 80
  %155 = bitcast float* %154 to <8 x float>*
  %wide.load132.2 = load <8 x float>, <8 x float>* %155, align 16
  %156 = getelementptr inbounds [800 x float], [800 x float]* %82, i64 0, i64 88
  %157 = bitcast float* %156 to <8 x float>*
  %wide.load133.2 = load <8 x float>, <8 x float>* %157, align 16
  %158 = getelementptr inbounds [800 x float], [800 x float]* %82, i64 0, i64 96
  %159 = bitcast float* %158 to <8 x float>*
  %wide.load130.3 = load <8 x float>, <8 x float>* %159, align 16
  %160 = getelementptr inbounds [800 x float], [800 x float]* %82, i64 0, i64 104
  %161 = bitcast float* %160 to <8 x float>*
  %wide.load131.3 = load <8 x float>, <8 x float>* %161, align 16
  %162 = getelementptr inbounds [800 x float], [800 x float]* %82, i64 0, i64 112
  %163 = bitcast float* %162 to <8 x float>*
  %wide.load132.3 = load <8 x float>, <8 x float>* %163, align 16
  %164 = getelementptr inbounds [800 x float], [800 x float]* %82, i64 0, i64 120
  %165 = bitcast float* %164 to <8 x float>*
  %wide.load133.3 = load <8 x float>, <8 x float>* %165, align 16
  %166 = getelementptr inbounds [800 x float], [800 x float]* %82, i64 0, i64 128
  %167 = bitcast float* %166 to <8 x float>*
  %wide.load130.4 = load <8 x float>, <8 x float>* %167, align 16
  %168 = getelementptr inbounds [800 x float], [800 x float]* %82, i64 0, i64 136
  %169 = bitcast float* %168 to <8 x float>*
  %wide.load131.4 = load <8 x float>, <8 x float>* %169, align 16
  %170 = getelementptr inbounds [800 x float], [800 x float]* %82, i64 0, i64 144
  %171 = bitcast float* %170 to <8 x float>*
  %wide.load132.4 = load <8 x float>, <8 x float>* %171, align 16
  %172 = getelementptr inbounds [800 x float], [800 x float]* %82, i64 0, i64 152
  %173 = bitcast float* %172 to <8 x float>*
  %wide.load133.4 = load <8 x float>, <8 x float>* %173, align 16
  %174 = getelementptr inbounds [800 x float], [800 x float]* %82, i64 0, i64 160
  %175 = bitcast float* %174 to <8 x float>*
  %wide.load130.5 = load <8 x float>, <8 x float>* %175, align 16
  %176 = getelementptr inbounds [800 x float], [800 x float]* %82, i64 0, i64 168
  %177 = bitcast float* %176 to <8 x float>*
  %wide.load131.5 = load <8 x float>, <8 x float>* %177, align 16
  %178 = getelementptr inbounds [800 x float], [800 x float]* %82, i64 0, i64 176
  %179 = bitcast float* %178 to <8 x float>*
  %wide.load132.5 = load <8 x float>, <8 x float>* %179, align 16
  %180 = getelementptr inbounds [800 x float], [800 x float]* %82, i64 0, i64 184
  %181 = bitcast float* %180 to <8 x float>*
  %wide.load133.5 = load <8 x float>, <8 x float>* %181, align 16
  tail call fastcc void @cluster_25__XlaCompiledKernel_true__XlaNumConstantArgs_1__XlaNumResourceArgs_0_.42.outline_fusion.4.loop_detach.dim.0.ls1(i64 0, i64 20, i64 %79, <8 x float> %wide.load131.2, <8 x float> %wide.load132.5, <8 x float> %wide.load131.5, <8 x float> %wide.load130.5, <8 x float> %wide.load133.4, <8 x float> %wide.load132.4, <8 x float> %wide.load131.4, <8 x float> %wide.load130.4, <8 x float> %wide.load133.3, <8 x float> %wide.load132.3, <8 x float> %wide.load131.3, <8 x float> %wide.load130.3, <8 x float> %wide.load133.2, <8 x float> %wide.load132.2, <8 x float> %wide.load130.2, <8 x float> %wide.load133.1, <8 x float> %wide.load132.1, <8 x float> %wide.load131.1, <8 x float> %wide.load130.1, <8 x float> %wide.load133.5, <8 x float> %wide.load133, <8 x float> %wide.load132, <8 x float> %wide.load131, <8 x float> %wide.load130, i8* %134, [800 x float]* %82, i8* %scevgep7172, [20 x [200 x float]]* %fusion.4, [20 x [800 x float]]* %72, i8* %132) #6
  %182 = getelementptr inbounds i8*, i8** %buffer_table, i64 4
  %183 = load i8*, i8** %182, align 8, !invariant.load !0, !dereferenceable !1, !align !2
  %fusion.1 = bitcast i8* %183 to [20 x [200 x float]]*
  tail call fastcc void @cluster_25__XlaCompiledKernel_true__XlaNumConstantArgs_1__XlaNumResourceArgs_0_.42.outline_fusion.1.loop_detach.dim.0.ls1(i64 0, i64 20, i64 %79, [20 x [200 x float]]* %fusion.4, [20 x [200 x float]]* %fusion.2, [20 x [200 x float]]* %fusion.1) #6
  tail call fastcc void @cluster_25__XlaCompiledKernel_true__XlaNumConstantArgs_1__XlaNumResourceArgs_0_.42.outline_fusion.3.loop_detach.dim.0.ls1(i64 0, i64 20, i64 %79, [20 x [200 x float]]* %fusion.4) #6
  %184 = getelementptr inbounds i8*, i8** %buffer_table, i64 5
  %185 = load i8*, i8** %184, align 8, !invariant.load !0, !dereferenceable !1, !align !2
  %fusion = bitcast i8* %185 to [20 x [200 x float]]*
  %scevgep184 = getelementptr [800 x float], [800 x float]* %82, i64 0, i64 600
  %scevgep184185 = bitcast float* %scevgep184 to i8*
  %scevgep186 = getelementptr [800 x float], [800 x float]* %82, i64 1, i64 0
  %scevgep186187 = bitcast float* %scevgep186 to i8*
  tail call fastcc void @cluster_25__XlaCompiledKernel_true__XlaNumConstantArgs_1__XlaNumResourceArgs_0_.42.outline_fusion.loop_detach.dim.0.ls1(i64 0, i64 20, i64 %79, i8* %scevgep186187, i8* %scevgep184185, [20 x [800 x float]]* %72, [800 x float]* %82, [20 x [200 x float]]* %fusion, i8* %185) #6
  %186 = getelementptr inbounds i8*, i8** %buffer_table, i64 2
  %187 = load i8*, i8** %186, align 8, !invariant.load !0, !dereferenceable !1, !align !2
  %fusion.5 = bitcast i8* %187 to [20 x [200 x float]]*
  %188 = bitcast float* %scevgep73 to <8 x float>*
  %wide.load245 = load <8 x float>, <8 x float>* %188, align 16
  %189 = getelementptr inbounds [800 x float], [800 x float]* %82, i64 0, i64 408
  %190 = bitcast float* %189 to <8 x float>*
  %wide.load246 = load <8 x float>, <8 x float>* %190, align 16
  %191 = getelementptr inbounds [800 x float], [800 x float]* %82, i64 0, i64 416
  %192 = bitcast float* %191 to <8 x float>*
  %wide.load247 = load <8 x float>, <8 x float>* %192, align 16
  %193 = getelementptr inbounds [800 x float], [800 x float]* %82, i64 0, i64 424
  %194 = bitcast float* %193 to <8 x float>*
  %wide.load248 = load <8 x float>, <8 x float>* %194, align 16
  %195 = getelementptr inbounds [800 x float], [800 x float]* %82, i64 0, i64 432
  %196 = bitcast float* %195 to <8 x float>*
  %wide.load245.1 = load <8 x float>, <8 x float>* %196, align 16
  %197 = getelementptr inbounds [800 x float], [800 x float]* %82, i64 0, i64 440
  %198 = bitcast float* %197 to <8 x float>*
  %wide.load246.1 = load <8 x float>, <8 x float>* %198, align 16
  %199 = getelementptr inbounds [800 x float], [800 x float]* %82, i64 0, i64 448
  %200 = bitcast float* %199 to <8 x float>*
  %wide.load247.1 = load <8 x float>, <8 x float>* %200, align 16
  %201 = getelementptr inbounds [800 x float], [800 x float]* %82, i64 0, i64 456
  %202 = bitcast float* %201 to <8 x float>*
  %wide.load248.1 = load <8 x float>, <8 x float>* %202, align 16
  %203 = getelementptr inbounds [800 x float], [800 x float]* %82, i64 0, i64 464
  %204 = bitcast float* %203 to <8 x float>*
  %wide.load245.2 = load <8 x float>, <8 x float>* %204, align 16
  %205 = getelementptr inbounds [800 x float], [800 x float]* %82, i64 0, i64 472
  %206 = bitcast float* %205 to <8 x float>*
  %wide.load246.2 = load <8 x float>, <8 x float>* %206, align 16
  %207 = getelementptr inbounds [800 x float], [800 x float]* %82, i64 0, i64 480
  %208 = bitcast float* %207 to <8 x float>*
  %wide.load247.2 = load <8 x float>, <8 x float>* %208, align 16
  %209 = getelementptr inbounds [800 x float], [800 x float]* %82, i64 0, i64 488
  %210 = bitcast float* %209 to <8 x float>*
  %wide.load248.2 = load <8 x float>, <8 x float>* %210, align 16
  %211 = getelementptr inbounds [800 x float], [800 x float]* %82, i64 0, i64 496
  %212 = bitcast float* %211 to <8 x float>*
  %wide.load245.3 = load <8 x float>, <8 x float>* %212, align 16
  %213 = getelementptr inbounds [800 x float], [800 x float]* %82, i64 0, i64 504
  %214 = bitcast float* %213 to <8 x float>*
  %wide.load246.3 = load <8 x float>, <8 x float>* %214, align 16
  %215 = getelementptr inbounds [800 x float], [800 x float]* %82, i64 0, i64 512
  %216 = bitcast float* %215 to <8 x float>*
  %wide.load247.3 = load <8 x float>, <8 x float>* %216, align 16
  %217 = getelementptr inbounds [800 x float], [800 x float]* %82, i64 0, i64 520
  %218 = bitcast float* %217 to <8 x float>*
  %wide.load248.3 = load <8 x float>, <8 x float>* %218, align 16
  %219 = getelementptr inbounds [800 x float], [800 x float]* %82, i64 0, i64 528
  %220 = bitcast float* %219 to <8 x float>*
  %wide.load245.4 = load <8 x float>, <8 x float>* %220, align 16
  %221 = getelementptr inbounds [800 x float], [800 x float]* %82, i64 0, i64 536
  %222 = bitcast float* %221 to <8 x float>*
  %wide.load246.4 = load <8 x float>, <8 x float>* %222, align 16
  %223 = getelementptr inbounds [800 x float], [800 x float]* %82, i64 0, i64 544
  %224 = bitcast float* %223 to <8 x float>*
  %wide.load247.4 = load <8 x float>, <8 x float>* %224, align 16
  %225 = getelementptr inbounds [800 x float], [800 x float]* %82, i64 0, i64 552
  %226 = bitcast float* %225 to <8 x float>*
  %wide.load248.4 = load <8 x float>, <8 x float>* %226, align 16
  %227 = getelementptr inbounds [800 x float], [800 x float]* %82, i64 0, i64 560
  %228 = bitcast float* %227 to <8 x float>*
  %wide.load245.5 = load <8 x float>, <8 x float>* %228, align 16
  %229 = getelementptr inbounds [800 x float], [800 x float]* %82, i64 0, i64 568
  %230 = bitcast float* %229 to <8 x float>*
  %wide.load246.5 = load <8 x float>, <8 x float>* %230, align 16
  %231 = getelementptr inbounds [800 x float], [800 x float]* %82, i64 0, i64 576
  %232 = bitcast float* %231 to <8 x float>*
  %wide.load247.5 = load <8 x float>, <8 x float>* %232, align 16
  %233 = getelementptr inbounds [800 x float], [800 x float]* %82, i64 0, i64 584
  %234 = bitcast float* %233 to <8 x float>*
  %wide.load248.5 = load <8 x float>, <8 x float>* %234, align 16
  tail call fastcc void @cluster_25__XlaCompiledKernel_true__XlaNumConstantArgs_1__XlaNumResourceArgs_0_.42.outline_fusion.5.loop_detach.dim.0.ls1(i64 0, i64 20, i64 %79, <8 x float> %wide.load246.2, <8 x float> %wide.load247.5, <8 x float> %wide.load246.5, <8 x float> %wide.load245.5, <8 x float> %wide.load248.4, <8 x float> %wide.load247.4, <8 x float> %wide.load246.4, <8 x float> %wide.load245.4, <8 x float> %wide.load248.3, <8 x float> %wide.load247.3, <8 x float> %wide.load246.3, <8 x float> %wide.load245.3, <8 x float> %wide.load248.2, <8 x float> %wide.load247.2, <8 x float> %wide.load245.2, <8 x float> %wide.load248.1, <8 x float> %wide.load247.1, <8 x float> %wide.load246.1, <8 x float> %wide.load245.1, <8 x float> %wide.load248.5, <8 x float> %wide.load248, <8 x float> %wide.load247, <8 x float> %wide.load246, <8 x float> %wide.load245, i8* %187, [800 x float]* %82, i8* %scevgep184185, [20 x [200 x float]]* %fusion.5, [20 x [800 x float]]* %72, i8* %scevgep7374) #6
  %235 = getelementptr inbounds i8*, i8** %buffer_table, i64 10
  %236 = bitcast i8** %235 to [6 x i8*]**
  %237 = load [6 x i8*]*, [6 x i8*]** %236, align 8, !invariant.load !0, !dereferenceable !7, !align !8
  %238 = getelementptr inbounds [6 x i8*], [6 x i8*]* %237, i64 0, i64 0
  store i8* %187, i8** %238, align 8, !alias.scope !9, !noalias !12
  %239 = getelementptr inbounds [6 x i8*], [6 x i8*]* %237, i64 0, i64 1
  store i8* %183, i8** %239, align 8, !alias.scope !9, !noalias !12
  %240 = getelementptr inbounds [6 x i8*], [6 x i8*]* %237, i64 0, i64 2
  store i8* %185, i8** %240, align 8, !alias.scope !9, !noalias !12
  %241 = getelementptr inbounds [6 x i8*], [6 x i8*]* %237, i64 0, i64 3
  store i8* %134, i8** %241, align 8, !alias.scope !9, !noalias !12
  %242 = getelementptr inbounds [6 x i8*], [6 x i8*]* %237, i64 0, i64 4
  store i8* %84, i8** %242, align 8, !alias.scope !9, !noalias !12
  %243 = getelementptr inbounds [6 x i8*], [6 x i8*]* %237, i64 0, i64 5
  store i8* %7, i8** %243, align 8, !alias.scope !9, !noalias !12
  ret void
}

; Function Attrs: argmemonly nounwind
declare void @llvm.memcpy.p0i8.p0i8.i64(i8* nocapture writeonly, i8* nocapture readonly, i64, i1) #1

; Function Attrs: nounwind readnone
declare float @tanhf(float) local_unnamed_addr #2

; Function Attrs: nounwind readonly
declare <8 x float> @llvm.masked.gather.v8f32.v8p0f32(<8 x float*>, i32, <8 x i1>, <8 x float>) #3

; Function Attrs: argmemonly nounwind stealable
define private fastcc void @cluster_25__XlaCompiledKernel_true__XlaNumConstantArgs_1__XlaNumResourceArgs_0_.42.outline_fusion.5.loop_detach.dim.0.ls1(i64 %fusion.5.indvar_address.dim.0.019.start.ls1, i64 %end.ls1, i64 %grainsize.ls1, <8 x float> %wide.load246.2.ls1, <8 x float> %wide.load247.5.ls1, <8 x float> %wide.load246.5.ls1, <8 x float> %wide.load245.5.ls1, <8 x float> %wide.load248.4.ls1, <8 x float> %wide.load247.4.ls1, <8 x float> %wide.load246.4.ls1, <8 x float> %wide.load245.4.ls1, <8 x float> %wide.load248.3.ls1, <8 x float> %wide.load247.3.ls1, <8 x float> %wide.load246.3.ls1, <8 x float> %wide.load245.3.ls1, <8 x float> %wide.load248.2.ls1, <8 x float> %wide.load247.2.ls1, <8 x float> %wide.load245.2.ls1, <8 x float> %wide.load248.1.ls1, <8 x float> %wide.load247.1.ls1, <8 x float> %wide.load246.1.ls1, <8 x float> %wide.load245.1.ls1, <8 x float> %wide.load248.5.ls1, <8 x float> %wide.load248.ls1, <8 x float> %wide.load247.ls1, <8 x float> %wide.load246.ls1, <8 x float> %wide.load245.ls1, i8* readnone align 16 %.ls1, [800 x float]* nocapture readonly align 16 %.ls11, i8* readnone align 16 %scevgep219220.ls1, [20 x [200 x float]]* align 16 %fusion.5.ls1, [20 x [800 x float]]* readonly align 16 %.ls12, i8* readnone align 16 %scevgep217218.ls1) unnamed_addr #4 {
fusion.loop_exit.dim.0.ls1:
  %__cilkrts_sf = alloca %struct.__cilkrts_stack_frame, align 8
  %0 = call %struct.__cilkrts_worker* @__cilkrts_get_tls_worker() #6
  %1 = icmp eq %struct.__cilkrts_worker* %0, null
  br i1 %1, label %slowpath.i, label %fastpath.i

slowpath.i:                                       ; preds = %fusion.loop_exit.dim.0.ls1
  %2 = call %struct.__cilkrts_worker* @__cilkrts_bind_thread_1() #6
  %3 = getelementptr inbounds %struct.__cilkrts_stack_frame, %struct.__cilkrts_stack_frame* %__cilkrts_sf, i64 0, i32 0
  store atomic i32 16777344, i32* %3 release, align 8
  br label %__cilkrts_enter_frame_1.exit

fastpath.i:                                       ; preds = %fusion.loop_exit.dim.0.ls1
  %4 = getelementptr inbounds %struct.__cilkrts_stack_frame, %struct.__cilkrts_stack_frame* %__cilkrts_sf, i64 0, i32 0
  store atomic i32 16777216, i32* %4 release, align 8
  br label %__cilkrts_enter_frame_1.exit

__cilkrts_enter_frame_1.exit:                     ; preds = %slowpath.i, %fastpath.i
  %5 = phi %struct.__cilkrts_worker* [ %2, %slowpath.i ], [ %0, %fastpath.i ]
  %6 = getelementptr inbounds %struct.__cilkrts_worker, %struct.__cilkrts_worker* %5, i64 0, i32 9
  %7 = load atomic %struct.__cilkrts_stack_frame*, %struct.__cilkrts_stack_frame** %6 acquire, align 8
  %8 = getelementptr inbounds %struct.__cilkrts_stack_frame, %struct.__cilkrts_stack_frame* %__cilkrts_sf, i64 0, i32 2
  store atomic %struct.__cilkrts_stack_frame* %7, %struct.__cilkrts_stack_frame** %8 release, align 8
  %9 = getelementptr inbounds %struct.__cilkrts_stack_frame, %struct.__cilkrts_stack_frame* %__cilkrts_sf, i64 0, i32 3
  store atomic %struct.__cilkrts_worker* %5, %struct.__cilkrts_worker** %9 release, align 8
  store atomic %struct.__cilkrts_stack_frame* %__cilkrts_sf, %struct.__cilkrts_stack_frame** %6 release, align 8
  %itercount3 = sub i64 %end.ls1, %fusion.5.indvar_address.dim.0.019.start.ls1
  %10 = icmp ugt i64 %itercount3, %grainsize.ls1
  br i1 %10, label %.lr.ph.preheader, label %fusion.5.loop_detach.dim.0.ls1.preheader

.lr.ph.preheader:                                 ; preds = %__cilkrts_enter_frame_1.exit
  %11 = getelementptr inbounds %struct.__cilkrts_stack_frame, %struct.__cilkrts_stack_frame* %__cilkrts_sf, i64 0, i32 6
  %12 = getelementptr inbounds %struct.__cilkrts_stack_frame, %struct.__cilkrts_stack_frame* %__cilkrts_sf, i64 0, i32 7
  %13 = getelementptr inbounds %struct.__cilkrts_stack_frame, %struct.__cilkrts_stack_frame* %__cilkrts_sf, i64 0, i32 5
  %14 = getelementptr inbounds [5 x i8*], [5 x i8*]* %13, i64 0, i64 0
  %15 = getelementptr inbounds %struct.__cilkrts_stack_frame, %struct.__cilkrts_stack_frame* %__cilkrts_sf, i64 0, i32 5, i64 2
  %16 = bitcast [5 x i8*]* %13 to i8*
  br label %.lr.ph

fusion.5.loop_detach.dim.0.ls1.preheader:         ; preds = %.split.split, %__cilkrts_enter_frame_1.exit
  %fusion.5.indvar_address.dim.0.019.ls1.dac.lcssa = phi i64 [ %fusion.5.indvar_address.dim.0.019.start.ls1, %__cilkrts_enter_frame_1.exit ], [ %miditer, %.split.split ]
  br label %fusion.5.loop_detach.dim.0.ls1

.lr.ph:                                           ; preds = %.lr.ph.preheader, %.split.split
  %itercount5 = phi i64 [ %itercount, %.split.split ], [ %itercount3, %.lr.ph.preheader ]
  %fusion.5.indvar_address.dim.0.019.ls1.dac4 = phi i64 [ %miditer, %.split.split ], [ %fusion.5.indvar_address.dim.0.019.start.ls1, %.lr.ph.preheader ]
  %halfcount = lshr i64 %itercount5, 1
  %miditer = add nuw nsw i64 %fusion.5.indvar_address.dim.0.019.ls1.dac4, %halfcount
  call void asm sideeffect "stmxcsr $0\0A\09fnstcw $1", "*m,*m,~{dirflag},~{fpsr},~{flags}"(i32* nonnull %11, i16* nonnull %12) #6
  %17 = call i8* @llvm.frameaddress(i32 0)
  store volatile i8* %17, i8** %14, align 8
  %18 = call i8* @llvm.stacksave()
  store volatile i8* %18, i8** %15, align 8
  %19 = call i32 @llvm.eh.sjlj.setjmp(i8* nonnull %16) #7
  %20 = icmp eq i32 %19, 0
  br i1 %20, label %.lr.ph.split, label %.split.split

.lr.ph.split:                                     ; preds = %.lr.ph
  call fastcc void @cluster_25__XlaCompiledKernel_true__XlaNumConstantArgs_1__XlaNumResourceArgs_0_.42.outline_fusion.5.loop_detach.dim.0.ls1.outline_.split.otd1(<8 x float> %wide.load246.3.ls1, <8 x float> %wide.load245.ls1, <8 x float> %wide.load246.ls1, <8 x float> %wide.load247.ls1, <8 x float> %wide.load248.ls1, <8 x float> %wide.load248.5.ls1, <8 x float> %wide.load245.1.ls1, <8 x float> %wide.load246.1.ls1, <8 x float> %wide.load247.1.ls1, <8 x float> %wide.load248.1.ls1, <8 x float> %wide.load245.2.ls1, <8 x float> %wide.load247.2.ls1, <8 x float> %wide.load248.2.ls1, <8 x float> %wide.load245.3.ls1, <8 x float> %wide.load247.3.ls1, <8 x float> %wide.load248.3.ls1, <8 x float> %wide.load245.4.ls1, <8 x float> %wide.load246.4.ls1, <8 x float> %wide.load247.4.ls1, <8 x float> %wide.load248.4.ls1, <8 x float> %wide.load245.5.ls1, <8 x float> %wide.load246.5.ls1, <8 x float> %wide.load247.5.ls1, <8 x float> %wide.load246.2.ls1, i8* %scevgep219220.ls1, [20 x [200 x float]]* %fusion.5.ls1, i8* %scevgep217218.ls1, [20 x [800 x float]]* %.ls12, i64 %fusion.5.indvar_address.dim.0.019.ls1.dac4, [800 x float]* %.ls11, i8* %.ls1, i64 %grainsize.ls1, i64 %miditer) #6
  br label %.split.split

.split.split:                                     ; preds = %.lr.ph, %.lr.ph.split
  %itercount = sub i64 %end.ls1, %miditer
  %21 = icmp ugt i64 %itercount, %grainsize.ls1
  br i1 %21, label %.lr.ph, label %fusion.5.loop_detach.dim.0.ls1.preheader

fusion.5.loop_sync.dim.0.ls1:                     ; preds = %fusion.5.loop_inc.dim.0.ls1
  %22 = getelementptr inbounds %struct.__cilkrts_stack_frame, %struct.__cilkrts_stack_frame* %__cilkrts_sf, i64 0, i32 0
  %23 = load atomic i32, i32* %22 acquire, align 8
  %24 = and i32 %23, 2
  %25 = icmp eq i32 %24, 0
  br i1 %25, label %__cilk_sync_nothrow.exit, label %cilk.sync.savestate.i

cilk.sync.savestate.i:                            ; preds = %fusion.5.loop_sync.dim.0.ls1
  %26 = load atomic %struct.__cilkrts_worker*, %struct.__cilkrts_worker** %9 acquire, align 8
  %.elt = getelementptr inbounds %struct.__cilkrts_worker, %struct.__cilkrts_worker* %26, i64 0, i32 12, i32 0
  %.unpack = load i64, i64* %.elt, align 8
  %.elt8 = getelementptr inbounds %struct.__cilkrts_worker, %struct.__cilkrts_worker* %26, i64 0, i32 12, i32 1
  %27 = bitcast %struct.__cilkrts_pedigree** %.elt8 to i64*
  %.unpack910 = load i64, i64* %27, align 8
  %.fca.0.0.gep = getelementptr inbounds %struct.__cilkrts_stack_frame, %struct.__cilkrts_stack_frame* %__cilkrts_sf, i64 0, i32 9, i32 0, i32 0
  %.fca.0.1.gep = getelementptr inbounds %struct.__cilkrts_stack_frame, %struct.__cilkrts_stack_frame* %__cilkrts_sf, i64 0, i32 9, i32 0, i32 1
  store i64 %.unpack, i64* %.fca.0.0.gep, align 8
  %28 = bitcast %struct.__cilkrts_pedigree** %.fca.0.1.gep to i64*
  store i64 %.unpack910, i64* %28, align 8
  %29 = getelementptr inbounds %struct.__cilkrts_stack_frame, %struct.__cilkrts_stack_frame* %__cilkrts_sf, i64 0, i32 6
  %30 = getelementptr inbounds %struct.__cilkrts_stack_frame, %struct.__cilkrts_stack_frame* %__cilkrts_sf, i64 0, i32 7
  call void asm sideeffect "stmxcsr $0\0A\09fnstcw $1", "*m,*m,~{dirflag},~{fpsr},~{flags}"(i32* nonnull %29, i16* nonnull %30) #6
  %31 = getelementptr inbounds %struct.__cilkrts_stack_frame, %struct.__cilkrts_stack_frame* %__cilkrts_sf, i64 0, i32 5
  %32 = call i8* @llvm.frameaddress(i32 0) #6
  %33 = getelementptr inbounds [5 x i8*], [5 x i8*]* %31, i64 0, i64 0
  store volatile i8* %32, i8** %33, align 8
  %34 = call i8* @llvm.stacksave() #6
  %35 = getelementptr inbounds %struct.__cilkrts_stack_frame, %struct.__cilkrts_stack_frame* %__cilkrts_sf, i64 0, i32 5, i64 2
  store volatile i8* %34, i8** %35, align 8
  %36 = bitcast [5 x i8*]* %31 to i8*
  %37 = call i32 @llvm.eh.sjlj.setjmp(i8* nonnull %36) #8
  %38 = icmp eq i32 %37, 0
  br i1 %38, label %cilk.sync.runtimecall.i, label %__cilk_sync_nothrow.exit

cilk.sync.runtimecall.i:                          ; preds = %cilk.sync.savestate.i
  call void @__cilkrts_sync(%struct.__cilkrts_stack_frame* nonnull %__cilkrts_sf) #6
  br label %__cilk_sync_nothrow.exit

__cilk_sync_nothrow.exit:                         ; preds = %fusion.5.loop_sync.dim.0.ls1, %cilk.sync.savestate.i, %cilk.sync.runtimecall.i
  %39 = load atomic %struct.__cilkrts_worker*, %struct.__cilkrts_worker** %9 acquire, align 8
  %40 = getelementptr inbounds %struct.__cilkrts_worker, %struct.__cilkrts_worker* %39, i64 0, i32 12, i32 0
  %41 = load i64, i64* %40, align 8
  %42 = add i64 %41, 1
  store i64 %42, i64* %40, align 8
  %43 = load atomic %struct.__cilkrts_worker*, %struct.__cilkrts_worker** %9 acquire, align 8
  %44 = bitcast %struct.__cilkrts_stack_frame** %8 to i64*
  %45 = load i64, i64* %44, align 8
  %46 = getelementptr inbounds %struct.__cilkrts_worker, %struct.__cilkrts_worker* %43, i64 0, i32 9
  %47 = bitcast %struct.__cilkrts_stack_frame** %46 to i64*
  store atomic i64 %45, i64* %47 release, align 8
  store atomic %struct.__cilkrts_stack_frame* null, %struct.__cilkrts_stack_frame** %8 release, align 8
  %48 = load atomic i32, i32* %22 acquire, align 8
  %49 = icmp eq i32 %48, 16777216
  br i1 %49, label %__cilk_parent_epilogue.exit, label %body.i

body.i:                                           ; preds = %__cilk_sync_nothrow.exit
  call void @__cilkrts_leave_frame(%struct.__cilkrts_stack_frame* nonnull %__cilkrts_sf) #6
  br label %__cilk_parent_epilogue.exit

__cilk_parent_epilogue.exit:                      ; preds = %__cilk_sync_nothrow.exit, %body.i
  ret void

fusion.5.loop_detach.dim.0.ls1:                   ; preds = %fusion.5.loop_detach.dim.0.ls1.preheader, %fusion.5.loop_inc.dim.0.ls1
  %fusion.5.indvar_address.dim.0.019.ls1 = phi i64 [ %indvar.inc15.ls1, %fusion.5.loop_inc.dim.0.ls1 ], [ %fusion.5.indvar_address.dim.0.019.ls1.dac.lcssa, %fusion.5.loop_detach.dim.0.ls1.preheader ]
  %50 = mul nuw nsw i64 %fusion.5.indvar_address.dim.0.019.ls1, 800
  %scevgep211.ls1 = getelementptr i8, i8* %.ls1, i64 %50
  %51 = add nuw nsw i64 %50, 800
  %scevgep212.ls1 = getelementptr i8, i8* %.ls1, i64 %51
  %scevgep213.ls1 = getelementptr [20 x [800 x float]], [20 x [800 x float]]* %.ls12, i64 0, i64 %fusion.5.indvar_address.dim.0.019.ls1, i64 400
  %scevgep213214.ls1 = bitcast float* %scevgep213.ls1 to i8*
  %scevgep215.ls1 = getelementptr [20 x [800 x float]], [20 x [800 x float]]* %.ls12, i64 0, i64 %fusion.5.indvar_address.dim.0.019.ls1, i64 600
  %scevgep215216.ls1 = bitcast float* %scevgep215.ls1 to i8*
  %bound0221.ls1 = icmp ult i8* %scevgep211.ls1, %scevgep215216.ls1
  %bound1222.ls1 = icmp ugt i8* %scevgep212.ls1, %scevgep213214.ls1
  %found.conflict223.ls1 = and i1 %bound0221.ls1, %bound1222.ls1
  %bound0224.ls1 = icmp ult i8* %scevgep211.ls1, %scevgep219220.ls1
  %bound1225.ls1 = icmp ugt i8* %scevgep212.ls1, %scevgep217218.ls1
  %found.conflict226.ls1 = and i1 %bound0224.ls1, %bound1225.ls1
  %conflict.rdx227.ls1 = or i1 %found.conflict223.ls1, %found.conflict226.ls1
  br i1 %conflict.rdx227.ls1, label %fusion.5.loop_body.dim.1.preheader.ls1, label %vector.body207.ls1

vector.body207.ls1:                               ; preds = %fusion.5.loop_detach.dim.0.ls1
  %52 = bitcast float* %scevgep213.ls1 to <8 x float>*
  %wide.load241.ls1 = load <8 x float>, <8 x float>* %52, align 16, !alias.scope !19, !noalias !23
  %53 = getelementptr inbounds [20 x [800 x float]], [20 x [800 x float]]* %.ls12, i64 0, i64 %fusion.5.indvar_address.dim.0.019.ls1, i64 408
  %54 = bitcast float* %53 to <8 x float>*
  %wide.load242.ls1 = load <8 x float>, <8 x float>* %54, align 16, !alias.scope !19, !noalias !23
  %55 = getelementptr inbounds [20 x [800 x float]], [20 x [800 x float]]* %.ls12, i64 0, i64 %fusion.5.indvar_address.dim.0.019.ls1, i64 416
  %56 = bitcast float* %55 to <8 x float>*
  %wide.load243.ls1 = load <8 x float>, <8 x float>* %56, align 16, !alias.scope !19, !noalias !23
  %57 = getelementptr inbounds [20 x [800 x float]], [20 x [800 x float]]* %.ls12, i64 0, i64 %fusion.5.indvar_address.dim.0.019.ls1, i64 424
  %58 = bitcast float* %57 to <8 x float>*
  %wide.load244.ls1 = load <8 x float>, <8 x float>* %58, align 16, !alias.scope !19, !noalias !23
  %59 = fadd fast <8 x float> %wide.load241.ls1, %wide.load245.ls1
  %60 = fadd fast <8 x float> %wide.load242.ls1, %wide.load246.ls1
  %61 = fadd fast <8 x float> %wide.load243.ls1, %wide.load247.ls1
  %62 = fadd fast <8 x float> %wide.load244.ls1, %wide.load248.ls1
  %63 = getelementptr inbounds [20 x [200 x float]], [20 x [200 x float]]* %fusion.5.ls1, i64 0, i64 %fusion.5.indvar_address.dim.0.019.ls1, i64 0
  %64 = bitcast float* %63 to <8 x float>*
  store <8 x float> %59, <8 x float>* %64, align 16, !alias.scope !25, !noalias !27
  %65 = getelementptr inbounds [20 x [200 x float]], [20 x [200 x float]]* %fusion.5.ls1, i64 0, i64 %fusion.5.indvar_address.dim.0.019.ls1, i64 8
  %66 = bitcast float* %65 to <8 x float>*
  store <8 x float> %60, <8 x float>* %66, align 16, !alias.scope !25, !noalias !27
  %67 = getelementptr inbounds [20 x [200 x float]], [20 x [200 x float]]* %fusion.5.ls1, i64 0, i64 %fusion.5.indvar_address.dim.0.019.ls1, i64 16
  %68 = bitcast float* %67 to <8 x float>*
  store <8 x float> %61, <8 x float>* %68, align 16, !alias.scope !25, !noalias !27
  %69 = getelementptr inbounds [20 x [200 x float]], [20 x [200 x float]]* %fusion.5.ls1, i64 0, i64 %fusion.5.indvar_address.dim.0.019.ls1, i64 24
  %70 = bitcast float* %69 to <8 x float>*
  store <8 x float> %62, <8 x float>* %70, align 16, !alias.scope !25, !noalias !27
  %71 = getelementptr inbounds [20 x [800 x float]], [20 x [800 x float]]* %.ls12, i64 0, i64 %fusion.5.indvar_address.dim.0.019.ls1, i64 432
  %72 = bitcast float* %71 to <8 x float>*
  %wide.load241.1.ls1 = load <8 x float>, <8 x float>* %72, align 16, !alias.scope !19, !noalias !23
  %73 = getelementptr inbounds [20 x [800 x float]], [20 x [800 x float]]* %.ls12, i64 0, i64 %fusion.5.indvar_address.dim.0.019.ls1, i64 440
  %74 = bitcast float* %73 to <8 x float>*
  %wide.load242.1.ls1 = load <8 x float>, <8 x float>* %74, align 16, !alias.scope !19, !noalias !23
  %75 = getelementptr inbounds [20 x [800 x float]], [20 x [800 x float]]* %.ls12, i64 0, i64 %fusion.5.indvar_address.dim.0.019.ls1, i64 448
  %76 = bitcast float* %75 to <8 x float>*
  %wide.load243.1.ls1 = load <8 x float>, <8 x float>* %76, align 16, !alias.scope !19, !noalias !23
  %77 = getelementptr inbounds [20 x [800 x float]], [20 x [800 x float]]* %.ls12, i64 0, i64 %fusion.5.indvar_address.dim.0.019.ls1, i64 456
  %78 = bitcast float* %77 to <8 x float>*
  %wide.load244.1.ls1 = load <8 x float>, <8 x float>* %78, align 16, !alias.scope !19, !noalias !23
  %79 = fadd fast <8 x float> %wide.load241.1.ls1, %wide.load245.1.ls1
  %80 = fadd fast <8 x float> %wide.load242.1.ls1, %wide.load246.1.ls1
  %81 = fadd fast <8 x float> %wide.load243.1.ls1, %wide.load247.1.ls1
  %82 = fadd fast <8 x float> %wide.load244.1.ls1, %wide.load248.1.ls1
  %83 = getelementptr inbounds [20 x [200 x float]], [20 x [200 x float]]* %fusion.5.ls1, i64 0, i64 %fusion.5.indvar_address.dim.0.019.ls1, i64 32
  %84 = bitcast float* %83 to <8 x float>*
  store <8 x float> %79, <8 x float>* %84, align 16, !alias.scope !25, !noalias !27
  %85 = getelementptr inbounds [20 x [200 x float]], [20 x [200 x float]]* %fusion.5.ls1, i64 0, i64 %fusion.5.indvar_address.dim.0.019.ls1, i64 40
  %86 = bitcast float* %85 to <8 x float>*
  store <8 x float> %80, <8 x float>* %86, align 16, !alias.scope !25, !noalias !27
  %87 = getelementptr inbounds [20 x [200 x float]], [20 x [200 x float]]* %fusion.5.ls1, i64 0, i64 %fusion.5.indvar_address.dim.0.019.ls1, i64 48
  %88 = bitcast float* %87 to <8 x float>*
  store <8 x float> %81, <8 x float>* %88, align 16, !alias.scope !25, !noalias !27
  %89 = getelementptr inbounds [20 x [200 x float]], [20 x [200 x float]]* %fusion.5.ls1, i64 0, i64 %fusion.5.indvar_address.dim.0.019.ls1, i64 56
  %90 = bitcast float* %89 to <8 x float>*
  store <8 x float> %82, <8 x float>* %90, align 16, !alias.scope !25, !noalias !27
  %91 = getelementptr inbounds [20 x [800 x float]], [20 x [800 x float]]* %.ls12, i64 0, i64 %fusion.5.indvar_address.dim.0.019.ls1, i64 464
  %92 = bitcast float* %91 to <8 x float>*
  %wide.load241.2.ls1 = load <8 x float>, <8 x float>* %92, align 16, !alias.scope !19, !noalias !23
  %93 = getelementptr inbounds [20 x [800 x float]], [20 x [800 x float]]* %.ls12, i64 0, i64 %fusion.5.indvar_address.dim.0.019.ls1, i64 472
  %94 = bitcast float* %93 to <8 x float>*
  %wide.load242.2.ls1 = load <8 x float>, <8 x float>* %94, align 16, !alias.scope !19, !noalias !23
  %95 = getelementptr inbounds [20 x [800 x float]], [20 x [800 x float]]* %.ls12, i64 0, i64 %fusion.5.indvar_address.dim.0.019.ls1, i64 480
  %96 = bitcast float* %95 to <8 x float>*
  %wide.load243.2.ls1 = load <8 x float>, <8 x float>* %96, align 16, !alias.scope !19, !noalias !23
  %97 = getelementptr inbounds [20 x [800 x float]], [20 x [800 x float]]* %.ls12, i64 0, i64 %fusion.5.indvar_address.dim.0.019.ls1, i64 488
  %98 = bitcast float* %97 to <8 x float>*
  %wide.load244.2.ls1 = load <8 x float>, <8 x float>* %98, align 16, !alias.scope !19, !noalias !23
  %99 = fadd fast <8 x float> %wide.load241.2.ls1, %wide.load245.2.ls1
  %100 = fadd fast <8 x float> %wide.load242.2.ls1, %wide.load246.2.ls1
  %101 = fadd fast <8 x float> %wide.load243.2.ls1, %wide.load247.2.ls1
  %102 = fadd fast <8 x float> %wide.load244.2.ls1, %wide.load248.2.ls1
  %103 = getelementptr inbounds [20 x [200 x float]], [20 x [200 x float]]* %fusion.5.ls1, i64 0, i64 %fusion.5.indvar_address.dim.0.019.ls1, i64 64
  %104 = bitcast float* %103 to <8 x float>*
  store <8 x float> %99, <8 x float>* %104, align 16, !alias.scope !25, !noalias !27
  %105 = getelementptr inbounds [20 x [200 x float]], [20 x [200 x float]]* %fusion.5.ls1, i64 0, i64 %fusion.5.indvar_address.dim.0.019.ls1, i64 72
  %106 = bitcast float* %105 to <8 x float>*
  store <8 x float> %100, <8 x float>* %106, align 16, !alias.scope !25, !noalias !27
  %107 = getelementptr inbounds [20 x [200 x float]], [20 x [200 x float]]* %fusion.5.ls1, i64 0, i64 %fusion.5.indvar_address.dim.0.019.ls1, i64 80
  %108 = bitcast float* %107 to <8 x float>*
  store <8 x float> %101, <8 x float>* %108, align 16, !alias.scope !25, !noalias !27
  %109 = getelementptr inbounds [20 x [200 x float]], [20 x [200 x float]]* %fusion.5.ls1, i64 0, i64 %fusion.5.indvar_address.dim.0.019.ls1, i64 88
  %110 = bitcast float* %109 to <8 x float>*
  store <8 x float> %102, <8 x float>* %110, align 16, !alias.scope !25, !noalias !27
  %111 = getelementptr inbounds [20 x [800 x float]], [20 x [800 x float]]* %.ls12, i64 0, i64 %fusion.5.indvar_address.dim.0.019.ls1, i64 496
  %112 = bitcast float* %111 to <8 x float>*
  %wide.load241.3.ls1 = load <8 x float>, <8 x float>* %112, align 16, !alias.scope !19, !noalias !23
  %113 = getelementptr inbounds [20 x [800 x float]], [20 x [800 x float]]* %.ls12, i64 0, i64 %fusion.5.indvar_address.dim.0.019.ls1, i64 504
  %114 = bitcast float* %113 to <8 x float>*
  %wide.load242.3.ls1 = load <8 x float>, <8 x float>* %114, align 16, !alias.scope !19, !noalias !23
  %115 = getelementptr inbounds [20 x [800 x float]], [20 x [800 x float]]* %.ls12, i64 0, i64 %fusion.5.indvar_address.dim.0.019.ls1, i64 512
  %116 = bitcast float* %115 to <8 x float>*
  %wide.load243.3.ls1 = load <8 x float>, <8 x float>* %116, align 16, !alias.scope !19, !noalias !23
  %117 = getelementptr inbounds [20 x [800 x float]], [20 x [800 x float]]* %.ls12, i64 0, i64 %fusion.5.indvar_address.dim.0.019.ls1, i64 520
  %118 = bitcast float* %117 to <8 x float>*
  %wide.load244.3.ls1 = load <8 x float>, <8 x float>* %118, align 16, !alias.scope !19, !noalias !23
  %119 = fadd fast <8 x float> %wide.load241.3.ls1, %wide.load245.3.ls1
  %120 = fadd fast <8 x float> %wide.load242.3.ls1, %wide.load246.3.ls1
  %121 = fadd fast <8 x float> %wide.load243.3.ls1, %wide.load247.3.ls1
  %122 = fadd fast <8 x float> %wide.load244.3.ls1, %wide.load248.3.ls1
  %123 = getelementptr inbounds [20 x [200 x float]], [20 x [200 x float]]* %fusion.5.ls1, i64 0, i64 %fusion.5.indvar_address.dim.0.019.ls1, i64 96
  %124 = bitcast float* %123 to <8 x float>*
  store <8 x float> %119, <8 x float>* %124, align 16, !alias.scope !25, !noalias !27
  %125 = getelementptr inbounds [20 x [200 x float]], [20 x [200 x float]]* %fusion.5.ls1, i64 0, i64 %fusion.5.indvar_address.dim.0.019.ls1, i64 104
  %126 = bitcast float* %125 to <8 x float>*
  store <8 x float> %120, <8 x float>* %126, align 16, !alias.scope !25, !noalias !27
  %127 = getelementptr inbounds [20 x [200 x float]], [20 x [200 x float]]* %fusion.5.ls1, i64 0, i64 %fusion.5.indvar_address.dim.0.019.ls1, i64 112
  %128 = bitcast float* %127 to <8 x float>*
  store <8 x float> %121, <8 x float>* %128, align 16, !alias.scope !25, !noalias !27
  %129 = getelementptr inbounds [20 x [200 x float]], [20 x [200 x float]]* %fusion.5.ls1, i64 0, i64 %fusion.5.indvar_address.dim.0.019.ls1, i64 120
  %130 = bitcast float* %129 to <8 x float>*
  store <8 x float> %122, <8 x float>* %130, align 16, !alias.scope !25, !noalias !27
  %131 = getelementptr inbounds [20 x [800 x float]], [20 x [800 x float]]* %.ls12, i64 0, i64 %fusion.5.indvar_address.dim.0.019.ls1, i64 528
  %132 = bitcast float* %131 to <8 x float>*
  %wide.load241.4.ls1 = load <8 x float>, <8 x float>* %132, align 16, !alias.scope !19, !noalias !23
  %133 = getelementptr inbounds [20 x [800 x float]], [20 x [800 x float]]* %.ls12, i64 0, i64 %fusion.5.indvar_address.dim.0.019.ls1, i64 536
  %134 = bitcast float* %133 to <8 x float>*
  %wide.load242.4.ls1 = load <8 x float>, <8 x float>* %134, align 16, !alias.scope !19, !noalias !23
  %135 = getelementptr inbounds [20 x [800 x float]], [20 x [800 x float]]* %.ls12, i64 0, i64 %fusion.5.indvar_address.dim.0.019.ls1, i64 544
  %136 = bitcast float* %135 to <8 x float>*
  %wide.load243.4.ls1 = load <8 x float>, <8 x float>* %136, align 16, !alias.scope !19, !noalias !23
  %137 = getelementptr inbounds [20 x [800 x float]], [20 x [800 x float]]* %.ls12, i64 0, i64 %fusion.5.indvar_address.dim.0.019.ls1, i64 552
  %138 = bitcast float* %137 to <8 x float>*
  %wide.load244.4.ls1 = load <8 x float>, <8 x float>* %138, align 16, !alias.scope !19, !noalias !23
  %139 = fadd fast <8 x float> %wide.load241.4.ls1, %wide.load245.4.ls1
  %140 = fadd fast <8 x float> %wide.load242.4.ls1, %wide.load246.4.ls1
  %141 = fadd fast <8 x float> %wide.load243.4.ls1, %wide.load247.4.ls1
  %142 = fadd fast <8 x float> %wide.load244.4.ls1, %wide.load248.4.ls1
  %143 = getelementptr inbounds [20 x [200 x float]], [20 x [200 x float]]* %fusion.5.ls1, i64 0, i64 %fusion.5.indvar_address.dim.0.019.ls1, i64 128
  %144 = bitcast float* %143 to <8 x float>*
  store <8 x float> %139, <8 x float>* %144, align 16, !alias.scope !25, !noalias !27
  %145 = getelementptr inbounds [20 x [200 x float]], [20 x [200 x float]]* %fusion.5.ls1, i64 0, i64 %fusion.5.indvar_address.dim.0.019.ls1, i64 136
  %146 = bitcast float* %145 to <8 x float>*
  store <8 x float> %140, <8 x float>* %146, align 16, !alias.scope !25, !noalias !27
  %147 = getelementptr inbounds [20 x [200 x float]], [20 x [200 x float]]* %fusion.5.ls1, i64 0, i64 %fusion.5.indvar_address.dim.0.019.ls1, i64 144
  %148 = bitcast float* %147 to <8 x float>*
  store <8 x float> %141, <8 x float>* %148, align 16, !alias.scope !25, !noalias !27
  %149 = getelementptr inbounds [20 x [200 x float]], [20 x [200 x float]]* %fusion.5.ls1, i64 0, i64 %fusion.5.indvar_address.dim.0.019.ls1, i64 152
  %150 = bitcast float* %149 to <8 x float>*
  store <8 x float> %142, <8 x float>* %150, align 16, !alias.scope !25, !noalias !27
  %151 = getelementptr inbounds [20 x [800 x float]], [20 x [800 x float]]* %.ls12, i64 0, i64 %fusion.5.indvar_address.dim.0.019.ls1, i64 560
  %152 = bitcast float* %151 to <8 x float>*
  %wide.load241.5.ls1 = load <8 x float>, <8 x float>* %152, align 16, !alias.scope !19, !noalias !23
  %153 = getelementptr inbounds [20 x [800 x float]], [20 x [800 x float]]* %.ls12, i64 0, i64 %fusion.5.indvar_address.dim.0.019.ls1, i64 568
  %154 = bitcast float* %153 to <8 x float>*
  %wide.load242.5.ls1 = load <8 x float>, <8 x float>* %154, align 16, !alias.scope !19, !noalias !23
  %155 = getelementptr inbounds [20 x [800 x float]], [20 x [800 x float]]* %.ls12, i64 0, i64 %fusion.5.indvar_address.dim.0.019.ls1, i64 576
  %156 = bitcast float* %155 to <8 x float>*
  %wide.load243.5.ls1 = load <8 x float>, <8 x float>* %156, align 16, !alias.scope !19, !noalias !23
  %157 = getelementptr inbounds [20 x [800 x float]], [20 x [800 x float]]* %.ls12, i64 0, i64 %fusion.5.indvar_address.dim.0.019.ls1, i64 584
  %158 = bitcast float* %157 to <8 x float>*
  %wide.load244.5.ls1 = load <8 x float>, <8 x float>* %158, align 16, !alias.scope !19, !noalias !23
  %159 = fadd fast <8 x float> %wide.load241.5.ls1, %wide.load245.5.ls1
  %160 = fadd fast <8 x float> %wide.load242.5.ls1, %wide.load246.5.ls1
  %161 = fadd fast <8 x float> %wide.load243.5.ls1, %wide.load247.5.ls1
  %162 = fadd fast <8 x float> %wide.load244.5.ls1, %wide.load248.5.ls1
  %163 = getelementptr inbounds [20 x [200 x float]], [20 x [200 x float]]* %fusion.5.ls1, i64 0, i64 %fusion.5.indvar_address.dim.0.019.ls1, i64 160
  %164 = bitcast float* %163 to <8 x float>*
  store <8 x float> %159, <8 x float>* %164, align 16, !alias.scope !25, !noalias !27
  %165 = getelementptr inbounds [20 x [200 x float]], [20 x [200 x float]]* %fusion.5.ls1, i64 0, i64 %fusion.5.indvar_address.dim.0.019.ls1, i64 168
  %166 = bitcast float* %165 to <8 x float>*
  store <8 x float> %160, <8 x float>* %166, align 16, !alias.scope !25, !noalias !27
  %167 = getelementptr inbounds [20 x [200 x float]], [20 x [200 x float]]* %fusion.5.ls1, i64 0, i64 %fusion.5.indvar_address.dim.0.019.ls1, i64 176
  %168 = bitcast float* %167 to <8 x float>*
  store <8 x float> %161, <8 x float>* %168, align 16, !alias.scope !25, !noalias !27
  %169 = getelementptr inbounds [20 x [200 x float]], [20 x [200 x float]]* %fusion.5.ls1, i64 0, i64 %fusion.5.indvar_address.dim.0.019.ls1, i64 184
  %170 = bitcast float* %169 to <8 x float>*
  store <8 x float> %162, <8 x float>* %170, align 16, !alias.scope !25, !noalias !27
  br label %fusion.5.loop_body.dim.1.preheader.ls1

fusion.5.loop_body.dim.1.preheader.ls1:           ; preds = %vector.body207.ls1, %fusion.5.loop_detach.dim.0.ls1
  %fusion.5.indvar_address.dim.1.018.ph.ls1 = phi i64 [ 0, %fusion.5.loop_detach.dim.0.ls1 ], [ 192, %vector.body207.ls1 ]
  br label %fusion.5.loop_body.dim.1.ls1

fusion.5.loop_body.dim.1.ls1:                     ; preds = %fusion.5.loop_body.dim.1.ls1, %fusion.5.loop_body.dim.1.preheader.ls1
  %fusion.5.indvar_address.dim.1.018.ls1 = phi i64 [ %fusion.5.indvar_address.dim.1.018.ph.ls1, %fusion.5.loop_body.dim.1.preheader.ls1 ], [ %indvar.inc16.3.ls1, %fusion.5.loop_body.dim.1.ls1 ]
  %171 = add nuw nsw i64 %fusion.5.indvar_address.dim.1.018.ls1, 400
  %172 = getelementptr inbounds [20 x [800 x float]], [20 x [800 x float]]* %.ls12, i64 0, i64 %fusion.5.indvar_address.dim.0.019.ls1, i64 %171
  %173 = load float, float* %172, align 16, !alias.scope !29, !noalias !23
  %174 = getelementptr inbounds [800 x float], [800 x float]* %.ls11, i64 0, i64 %171
  %175 = load float, float* %174, align 16, !invariant.load !0, !noalias !30
  %176 = fadd fast float %175, %173
  %177 = getelementptr inbounds [20 x [200 x float]], [20 x [200 x float]]* %fusion.5.ls1, i64 0, i64 %fusion.5.indvar_address.dim.0.019.ls1, i64 %fusion.5.indvar_address.dim.1.018.ls1
  store float %176, float* %177, align 16, !alias.scope !31, !noalias !32
  %indvar.inc16.ls1 = or i64 %fusion.5.indvar_address.dim.1.018.ls1, 1
  %178 = add nuw nsw i64 %fusion.5.indvar_address.dim.1.018.ls1, 401
  %179 = getelementptr inbounds [20 x [800 x float]], [20 x [800 x float]]* %.ls12, i64 0, i64 %fusion.5.indvar_address.dim.0.019.ls1, i64 %178
  %180 = load float, float* %179, align 4, !alias.scope !29, !noalias !23
  %181 = getelementptr inbounds [800 x float], [800 x float]* %.ls11, i64 0, i64 %178
  %182 = load float, float* %181, align 4, !invariant.load !0, !noalias !30
  %183 = fadd fast float %182, %180
  %184 = getelementptr inbounds [20 x [200 x float]], [20 x [200 x float]]* %fusion.5.ls1, i64 0, i64 %fusion.5.indvar_address.dim.0.019.ls1, i64 %indvar.inc16.ls1
  store float %183, float* %184, align 4, !alias.scope !31, !noalias !32
  %indvar.inc16.1.ls1 = or i64 %fusion.5.indvar_address.dim.1.018.ls1, 2
  %185 = add nuw nsw i64 %fusion.5.indvar_address.dim.1.018.ls1, 402
  %186 = getelementptr inbounds [20 x [800 x float]], [20 x [800 x float]]* %.ls12, i64 0, i64 %fusion.5.indvar_address.dim.0.019.ls1, i64 %185
  %187 = load float, float* %186, align 8, !alias.scope !29, !noalias !23
  %188 = getelementptr inbounds [800 x float], [800 x float]* %.ls11, i64 0, i64 %185
  %189 = load float, float* %188, align 8, !invariant.load !0, !noalias !30
  %190 = fadd fast float %189, %187
  %191 = getelementptr inbounds [20 x [200 x float]], [20 x [200 x float]]* %fusion.5.ls1, i64 0, i64 %fusion.5.indvar_address.dim.0.019.ls1, i64 %indvar.inc16.1.ls1
  store float %190, float* %191, align 8, !alias.scope !31, !noalias !32
  %indvar.inc16.2.ls1 = or i64 %fusion.5.indvar_address.dim.1.018.ls1, 3
  %192 = add nuw nsw i64 %fusion.5.indvar_address.dim.1.018.ls1, 403
  %193 = getelementptr inbounds [20 x [800 x float]], [20 x [800 x float]]* %.ls12, i64 0, i64 %fusion.5.indvar_address.dim.0.019.ls1, i64 %192
  %194 = load float, float* %193, align 4, !alias.scope !29, !noalias !23
  %195 = getelementptr inbounds [800 x float], [800 x float]* %.ls11, i64 0, i64 %192
  %196 = load float, float* %195, align 4, !invariant.load !0, !noalias !30
  %197 = fadd fast float %196, %194
  %198 = getelementptr inbounds [20 x [200 x float]], [20 x [200 x float]]* %fusion.5.ls1, i64 0, i64 %fusion.5.indvar_address.dim.0.019.ls1, i64 %indvar.inc16.2.ls1
  store float %197, float* %198, align 4, !alias.scope !31, !noalias !32
  %indvar.inc16.3.ls1 = add nuw nsw i64 %fusion.5.indvar_address.dim.1.018.ls1, 4
  %exitcond.3.ls1 = icmp eq i64 %indvar.inc16.3.ls1, 200
  br i1 %exitcond.3.ls1, label %fusion.5.loop_inc.dim.0.ls1, label %fusion.5.loop_body.dim.1.ls1, !llvm.loop !33

fusion.5.loop_inc.dim.0.ls1:                      ; preds = %fusion.5.loop_body.dim.1.ls1
  %indvar.inc15.ls1 = add nuw nsw i64 %fusion.5.indvar_address.dim.0.019.ls1, 1
  %exitcond43.ls1 = icmp eq i64 %indvar.inc15.ls1, %end.ls1
  br i1 %exitcond43.ls1, label %fusion.5.loop_sync.dim.0.ls1, label %fusion.5.loop_detach.dim.0.ls1, !llvm.loop !36
}

; Function Attrs: argmemonly nounwind stealable
define private fastcc void @cluster_25__XlaCompiledKernel_true__XlaNumConstantArgs_1__XlaNumResourceArgs_0_.42.outline_fusion.loop_detach.dim.0.ls1(i64 %fusion.indvar_address.dim.0.022.start.ls1, i64 %end.ls1, i64 %grainsize.ls1, i8* readnone align 16 %scevgep186187.ls1, i8* readnone align 16 %scevgep184185.ls1, [20 x [800 x float]]* readonly align 16 %.ls1, [800 x float]* nocapture readonly align 16 %.ls11, [20 x [200 x float]]* nocapture align 16 %fusion.ls1, i8* readnone align 16 %.ls12) unnamed_addr #4 {
fusion.3.loop_exit.dim.0.ls1:
  %__cilkrts_sf = alloca %struct.__cilkrts_stack_frame, align 8
  %0 = call %struct.__cilkrts_worker* @__cilkrts_get_tls_worker() #6
  %1 = icmp eq %struct.__cilkrts_worker* %0, null
  br i1 %1, label %slowpath.i, label %fastpath.i

slowpath.i:                                       ; preds = %fusion.3.loop_exit.dim.0.ls1
  %2 = call %struct.__cilkrts_worker* @__cilkrts_bind_thread_1() #6
  %3 = getelementptr inbounds %struct.__cilkrts_stack_frame, %struct.__cilkrts_stack_frame* %__cilkrts_sf, i64 0, i32 0
  store atomic i32 16777344, i32* %3 release, align 8
  br label %__cilkrts_enter_frame_1.exit

fastpath.i:                                       ; preds = %fusion.3.loop_exit.dim.0.ls1
  %4 = getelementptr inbounds %struct.__cilkrts_stack_frame, %struct.__cilkrts_stack_frame* %__cilkrts_sf, i64 0, i32 0
  store atomic i32 16777216, i32* %4 release, align 8
  br label %__cilkrts_enter_frame_1.exit

__cilkrts_enter_frame_1.exit:                     ; preds = %slowpath.i, %fastpath.i
  %5 = phi %struct.__cilkrts_worker* [ %2, %slowpath.i ], [ %0, %fastpath.i ]
  %6 = getelementptr inbounds %struct.__cilkrts_worker, %struct.__cilkrts_worker* %5, i64 0, i32 9
  %7 = load atomic %struct.__cilkrts_stack_frame*, %struct.__cilkrts_stack_frame** %6 acquire, align 8
  %8 = getelementptr inbounds %struct.__cilkrts_stack_frame, %struct.__cilkrts_stack_frame* %__cilkrts_sf, i64 0, i32 2
  store atomic %struct.__cilkrts_stack_frame* %7, %struct.__cilkrts_stack_frame** %8 release, align 8
  %9 = getelementptr inbounds %struct.__cilkrts_stack_frame, %struct.__cilkrts_stack_frame* %__cilkrts_sf, i64 0, i32 3
  store atomic %struct.__cilkrts_worker* %5, %struct.__cilkrts_worker** %9 release, align 8
  store atomic %struct.__cilkrts_stack_frame* %__cilkrts_sf, %struct.__cilkrts_stack_frame** %6 release, align 8
  %itercount4 = sub i64 %end.ls1, %fusion.indvar_address.dim.0.022.start.ls1
  %10 = icmp ugt i64 %itercount4, %grainsize.ls1
  br i1 %10, label %.lr.ph.preheader, label %fusion.loop_detach.dim.0.ls1.preheader

.lr.ph.preheader:                                 ; preds = %__cilkrts_enter_frame_1.exit
  %11 = getelementptr inbounds %struct.__cilkrts_stack_frame, %struct.__cilkrts_stack_frame* %__cilkrts_sf, i64 0, i32 6
  %12 = getelementptr inbounds %struct.__cilkrts_stack_frame, %struct.__cilkrts_stack_frame* %__cilkrts_sf, i64 0, i32 7
  %13 = getelementptr inbounds %struct.__cilkrts_stack_frame, %struct.__cilkrts_stack_frame* %__cilkrts_sf, i64 0, i32 5
  %14 = getelementptr inbounds [5 x i8*], [5 x i8*]* %13, i64 0, i64 0
  %15 = getelementptr inbounds %struct.__cilkrts_stack_frame, %struct.__cilkrts_stack_frame* %__cilkrts_sf, i64 0, i32 5, i64 2
  %16 = bitcast [5 x i8*]* %13 to i8*
  br label %.lr.ph

fusion.loop_detach.dim.0.ls1.preheader:           ; preds = %.split.split, %__cilkrts_enter_frame_1.exit
  %fusion.indvar_address.dim.0.022.ls1.dac.lcssa = phi i64 [ %fusion.indvar_address.dim.0.022.start.ls1, %__cilkrts_enter_frame_1.exit ], [ %miditer, %.split.split ]
  br label %fusion.loop_detach.dim.0.ls1

.lr.ph:                                           ; preds = %.lr.ph.preheader, %.split.split
  %itercount6 = phi i64 [ %itercount, %.split.split ], [ %itercount4, %.lr.ph.preheader ]
  %fusion.indvar_address.dim.0.022.ls1.dac5 = phi i64 [ %miditer, %.split.split ], [ %fusion.indvar_address.dim.0.022.start.ls1, %.lr.ph.preheader ]
  %halfcount = lshr i64 %itercount6, 1
  %miditer = add nuw nsw i64 %fusion.indvar_address.dim.0.022.ls1.dac5, %halfcount
  call void asm sideeffect "stmxcsr $0\0A\09fnstcw $1", "*m,*m,~{dirflag},~{fpsr},~{flags}"(i32* nonnull %11, i16* nonnull %12) #6
  %17 = call i8* @llvm.frameaddress(i32 0)
  store volatile i8* %17, i8** %14, align 8
  %18 = call i8* @llvm.stacksave()
  store volatile i8* %18, i8** %15, align 8
  %19 = call i32 @llvm.eh.sjlj.setjmp(i8* nonnull %16) #7
  %20 = icmp eq i32 %19, 0
  br i1 %20, label %.lr.ph.split, label %.split.split

.lr.ph.split:                                     ; preds = %.lr.ph
  call fastcc void @cluster_25__XlaCompiledKernel_true__XlaNumConstantArgs_1__XlaNumResourceArgs_0_.42.outline_fusion.loop_detach.dim.0.ls1.outline_.split.otd1(i64 %fusion.indvar_address.dim.0.022.ls1.dac5, i64 %miditer, i64 %grainsize.ls1, i8* %scevgep186187.ls1, i8* %scevgep184185.ls1, [20 x [800 x float]]* %.ls1, [800 x float]* %.ls11, [20 x [200 x float]]* %fusion.ls1, i8* %.ls12) #6
  br label %.split.split

.split.split:                                     ; preds = %.lr.ph, %.lr.ph.split
  %itercount = sub i64 %end.ls1, %miditer
  %21 = icmp ugt i64 %itercount, %grainsize.ls1
  br i1 %21, label %.lr.ph, label %fusion.loop_detach.dim.0.ls1.preheader

fusion.loop_sync.dim.0.ls1:                       ; preds = %fusion.loop_inc.dim.0.ls1
  %22 = getelementptr inbounds %struct.__cilkrts_stack_frame, %struct.__cilkrts_stack_frame* %__cilkrts_sf, i64 0, i32 0
  %23 = load atomic i32, i32* %22 acquire, align 8
  %24 = and i32 %23, 2
  %25 = icmp eq i32 %24, 0
  br i1 %25, label %__cilk_sync_nothrow.exit, label %cilk.sync.savestate.i

cilk.sync.savestate.i:                            ; preds = %fusion.loop_sync.dim.0.ls1
  %26 = load atomic %struct.__cilkrts_worker*, %struct.__cilkrts_worker** %9 acquire, align 8
  %.elt = getelementptr inbounds %struct.__cilkrts_worker, %struct.__cilkrts_worker* %26, i64 0, i32 12, i32 0
  %.unpack = load i64, i64* %.elt, align 8
  %.elt12 = getelementptr inbounds %struct.__cilkrts_worker, %struct.__cilkrts_worker* %26, i64 0, i32 12, i32 1
  %27 = bitcast %struct.__cilkrts_pedigree** %.elt12 to i64*
  %.unpack1314 = load i64, i64* %27, align 8
  %.fca.0.0.gep = getelementptr inbounds %struct.__cilkrts_stack_frame, %struct.__cilkrts_stack_frame* %__cilkrts_sf, i64 0, i32 9, i32 0, i32 0
  %.fca.0.1.gep = getelementptr inbounds %struct.__cilkrts_stack_frame, %struct.__cilkrts_stack_frame* %__cilkrts_sf, i64 0, i32 9, i32 0, i32 1
  store i64 %.unpack, i64* %.fca.0.0.gep, align 8
  %28 = bitcast %struct.__cilkrts_pedigree** %.fca.0.1.gep to i64*
  store i64 %.unpack1314, i64* %28, align 8
  %29 = getelementptr inbounds %struct.__cilkrts_stack_frame, %struct.__cilkrts_stack_frame* %__cilkrts_sf, i64 0, i32 6
  %30 = getelementptr inbounds %struct.__cilkrts_stack_frame, %struct.__cilkrts_stack_frame* %__cilkrts_sf, i64 0, i32 7
  call void asm sideeffect "stmxcsr $0\0A\09fnstcw $1", "*m,*m,~{dirflag},~{fpsr},~{flags}"(i32* nonnull %29, i16* nonnull %30) #6
  %31 = getelementptr inbounds %struct.__cilkrts_stack_frame, %struct.__cilkrts_stack_frame* %__cilkrts_sf, i64 0, i32 5
  %32 = call i8* @llvm.frameaddress(i32 0) #6
  %33 = getelementptr inbounds [5 x i8*], [5 x i8*]* %31, i64 0, i64 0
  store volatile i8* %32, i8** %33, align 8
  %34 = call i8* @llvm.stacksave() #6
  %35 = getelementptr inbounds %struct.__cilkrts_stack_frame, %struct.__cilkrts_stack_frame* %__cilkrts_sf, i64 0, i32 5, i64 2
  store volatile i8* %34, i8** %35, align 8
  %36 = bitcast [5 x i8*]* %31 to i8*
  %37 = call i32 @llvm.eh.sjlj.setjmp(i8* nonnull %36) #8
  %38 = icmp eq i32 %37, 0
  br i1 %38, label %cilk.sync.runtimecall.i, label %__cilk_sync_nothrow.exit

cilk.sync.runtimecall.i:                          ; preds = %cilk.sync.savestate.i
  call void @__cilkrts_sync(%struct.__cilkrts_stack_frame* nonnull %__cilkrts_sf) #6
  br label %__cilk_sync_nothrow.exit

__cilk_sync_nothrow.exit:                         ; preds = %fusion.loop_sync.dim.0.ls1, %cilk.sync.savestate.i, %cilk.sync.runtimecall.i
  %39 = load atomic %struct.__cilkrts_worker*, %struct.__cilkrts_worker** %9 acquire, align 8
  %40 = getelementptr inbounds %struct.__cilkrts_worker, %struct.__cilkrts_worker* %39, i64 0, i32 12, i32 0
  %41 = load i64, i64* %40, align 8
  %42 = add i64 %41, 1
  store i64 %42, i64* %40, align 8
  %43 = load atomic %struct.__cilkrts_worker*, %struct.__cilkrts_worker** %9 acquire, align 8
  %44 = bitcast %struct.__cilkrts_stack_frame** %8 to i64*
  %45 = load i64, i64* %44, align 8
  %46 = getelementptr inbounds %struct.__cilkrts_worker, %struct.__cilkrts_worker* %43, i64 0, i32 9
  %47 = bitcast %struct.__cilkrts_stack_frame** %46 to i64*
  store atomic i64 %45, i64* %47 release, align 8
  store atomic %struct.__cilkrts_stack_frame* null, %struct.__cilkrts_stack_frame** %8 release, align 8
  %48 = load atomic i32, i32* %22 acquire, align 8
  %49 = icmp eq i32 %48, 16777216
  br i1 %49, label %__cilk_parent_epilogue.exit, label %body.i

body.i:                                           ; preds = %__cilk_sync_nothrow.exit
  call void @__cilkrts_leave_frame(%struct.__cilkrts_stack_frame* nonnull %__cilkrts_sf) #6
  br label %__cilk_parent_epilogue.exit

__cilk_parent_epilogue.exit:                      ; preds = %__cilk_sync_nothrow.exit, %body.i
  ret void

fusion.loop_detach.dim.0.ls1:                     ; preds = %fusion.loop_detach.dim.0.ls1.preheader, %fusion.loop_inc.dim.0.ls1
  %fusion.indvar_address.dim.0.022.ls1 = phi i64 [ %52, %fusion.loop_inc.dim.0.ls1 ], [ %fusion.indvar_address.dim.0.022.ls1.dac.lcssa, %fusion.loop_detach.dim.0.ls1.preheader ]
  %50 = mul nuw nsw i64 %fusion.indvar_address.dim.0.022.ls1, 800
  %scevgep178.ls1 = getelementptr i8, i8* %.ls12, i64 %50
  %51 = add nuw nsw i64 %50, 800
  %scevgep179.ls1 = getelementptr i8, i8* %.ls12, i64 %51
  %scevgep180.ls1 = getelementptr [20 x [800 x float]], [20 x [800 x float]]* %.ls1, i64 0, i64 %fusion.indvar_address.dim.0.022.ls1, i64 600
  %scevgep180181.ls1 = bitcast float* %scevgep180.ls1 to i8*
  %52 = add nuw nsw i64 %fusion.indvar_address.dim.0.022.ls1, 1
  %scevgep182.ls1 = getelementptr [20 x [800 x float]], [20 x [800 x float]]* %.ls1, i64 0, i64 %52, i64 0
  %scevgep182183.ls1 = bitcast float* %scevgep182.ls1 to i8*
  %bound0188.ls1 = icmp ult i8* %scevgep178.ls1, %scevgep182183.ls1
  %bound1189.ls1 = icmp ugt i8* %scevgep179.ls1, %scevgep180181.ls1
  %found.conflict190.ls1 = and i1 %bound0188.ls1, %bound1189.ls1
  %bound0191.ls1 = icmp ult i8* %scevgep178.ls1, %scevgep186187.ls1
  %bound1192.ls1 = icmp ugt i8* %scevgep179.ls1, %scevgep184185.ls1
  %found.conflict193.ls1 = and i1 %bound0191.ls1, %bound1192.ls1
  %conflict.rdx194.ls1 = or i1 %found.conflict190.ls1, %found.conflict193.ls1
  br i1 %conflict.rdx194.ls1, label %fusion.loop_body.dim.1.ls1, label %vector.body174.ls1

vector.body174.ls1:                               ; preds = %fusion.loop_detach.dim.0.ls1, %vector.body174.ls1
  %index198.ls1 = phi i64 [ %index.next199.ls1, %vector.body174.ls1 ], [ 0, %fusion.loop_detach.dim.0.ls1 ]
  %53 = add nuw nsw i64 %index198.ls1, 600
  %54 = getelementptr inbounds [20 x [800 x float]], [20 x [800 x float]]* %.ls1, i64 0, i64 %fusion.indvar_address.dim.0.022.ls1, i64 %53
  %55 = bitcast float* %54 to <8 x float>*
  %wide.load205.ls1 = load <8 x float>, <8 x float>* %55, align 16, !alias.scope !38, !noalias !23
  %56 = getelementptr inbounds [800 x float], [800 x float]* %.ls11, i64 0, i64 %53
  %57 = bitcast float* %56 to <8 x float>*
  %wide.load206.ls1 = load <8 x float>, <8 x float>* %57, align 16, !invariant.load !0, !alias.scope !41, !noalias !30
  %58 = fadd fast <8 x float> %wide.load206.ls1, %wide.load205.ls1
  %59 = fmul fast <8 x float> %58, <float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01>
  %60 = fcmp fast uge <8 x float> %59, <float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00>
  %61 = select <8 x i1> %60, <8 x float> %59, <8 x float> <float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00>
  %62 = fcmp fast ule <8 x float> %61, <float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00>
  %63 = select <8 x i1> %62, <8 x float> %61, <8 x float> <float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00>
  %64 = fmul fast <8 x float> %63, %63
  %65 = fmul fast <8 x float> %64, <float 0xBCB3E4B800000000, float 0xBCB3E4B800000000, float 0xBCB3E4B800000000, float 0xBCB3E4B800000000, float 0xBCB3E4B800000000, float 0xBCB3E4B800000000, float 0xBCB3E4B800000000, float 0xBCB3E4B800000000>
  %66 = fadd fast <8 x float> %65, <float 0x3D4C266FC0000000, float 0x3D4C266FC0000000, float 0x3D4C266FC0000000, float 0x3D4C266FC0000000, float 0x3D4C266FC0000000, float 0x3D4C266FC0000000, float 0x3D4C266FC0000000, float 0x3D4C266FC0000000>
  %67 = fmul fast <8 x float> %64, %66
  %68 = fadd fast <8 x float> %67, <float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000>
  %69 = fmul fast <8 x float> %64, %68
  %70 = fadd fast <8 x float> %69, <float 0x3E6B800820000000, float 0x3E6B800820000000, float 0x3E6B800820000000, float 0x3E6B800820000000, float 0x3E6B800820000000, float 0x3E6B800820000000, float 0x3E6B800820000000, float 0x3E6B800820000000>
  %71 = fmul fast <8 x float> %64, %70
  %72 = fadd fast <8 x float> %71, <float 0x3EEF286940000000, float 0x3EEF286940000000, float 0x3EEF286940000000, float 0x3EEF286940000000, float 0x3EEF286940000000, float 0x3EEF286940000000, float 0x3EEF286940000000, float 0x3EEF286940000000>
  %73 = fmul fast <8 x float> %64, %72
  %74 = fadd fast <8 x float> %73, <float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000>
  %75 = fmul fast <8 x float> %64, %74
  %76 = fadd fast <8 x float> %75, <float 0x3F740B3B80000000, float 0x3F740B3B80000000, float 0x3F740B3B80000000, float 0x3F740B3B80000000, float 0x3F740B3B80000000, float 0x3F740B3B80000000, float 0x3F740B3B80000000, float 0x3F740B3B80000000>
  %77 = fmul fast <8 x float> %63, %76
  %78 = fmul fast <8 x float> %64, <float 0x3EB41A7B00000000, float 0x3EB41A7B00000000, float 0x3EB41A7B00000000, float 0x3EB41A7B00000000, float 0x3EB41A7B00000000, float 0x3EB41A7B00000000, float 0x3EB41A7B00000000, float 0x3EB41A7B00000000>
  %79 = fadd fast <8 x float> %78, <float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000>
  %80 = fmul fast <8 x float> %64, %79
  %81 = fadd fast <8 x float> %80, <float 0x3F629540A0000000, float 0x3F629540A0000000, float 0x3F629540A0000000, float 0x3F629540A0000000, float 0x3F629540A0000000, float 0x3F629540A0000000, float 0x3F629540A0000000, float 0x3F629540A0000000>
  %82 = fmul fast <8 x float> %64, %81
  %83 = fadd fast <8 x float> %82, <float 0x3F740B3BA0000000, float 0x3F740B3BA0000000, float 0x3F740B3BA0000000, float 0x3F740B3BA0000000, float 0x3F740B3BA0000000, float 0x3F740B3BA0000000, float 0x3F740B3BA0000000, float 0x3F740B3BA0000000>
  %84 = fdiv fast <8 x float> %77, %83
  %85 = fmul fast <8 x float> %84, <float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01>
  %86 = fadd fast <8 x float> %85, <float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01>
  %87 = getelementptr inbounds [20 x [200 x float]], [20 x [200 x float]]* %fusion.ls1, i64 0, i64 %fusion.indvar_address.dim.0.022.ls1, i64 %index198.ls1
  %88 = bitcast float* %87 to <8 x float>*
  store <8 x float> %86, <8 x float>* %88, align 16, !alias.scope !43, !noalias !45
  %index.next199.ls1 = add nuw nsw i64 %index198.ls1, 8
  %89 = icmp eq i64 %index.next199.ls1, 200
  br i1 %89, label %fusion.loop_inc.dim.0.ls1, label %vector.body174.ls1, !llvm.loop !46

fusion.loop_body.dim.1.ls1:                       ; preds = %fusion.loop_detach.dim.0.ls1, %fusion.loop_body.dim.1.ls1
  %fusion.indvar_address.dim.1.021.ls1 = phi i64 [ %indvar.inc14.ls1, %fusion.loop_body.dim.1.ls1 ], [ 0, %fusion.loop_detach.dim.0.ls1 ]
  %90 = add nuw nsw i64 %fusion.indvar_address.dim.1.021.ls1, 600
  %91 = getelementptr inbounds [20 x [800 x float]], [20 x [800 x float]]* %.ls1, i64 0, i64 %fusion.indvar_address.dim.0.022.ls1, i64 %90
  %92 = load float, float* %91, align 4, !alias.scope !29, !noalias !23
  %93 = getelementptr inbounds [800 x float], [800 x float]* %.ls11, i64 0, i64 %90
  %94 = load float, float* %93, align 4, !invariant.load !0, !noalias !30
  %95 = fadd fast float %94, %92
  %96 = fmul fast float %95, 5.000000e-01
  %97 = tail call fast float @tanhf(float %96)
  %98 = fmul fast float %97, 5.000000e-01
  %99 = fadd fast float %98, 5.000000e-01
  %100 = getelementptr inbounds [20 x [200 x float]], [20 x [200 x float]]* %fusion.ls1, i64 0, i64 %fusion.indvar_address.dim.0.022.ls1, i64 %fusion.indvar_address.dim.1.021.ls1
  store float %99, float* %100, align 4, !alias.scope !47, !noalias !48
  %indvar.inc14.ls1 = add nuw nsw i64 %fusion.indvar_address.dim.1.021.ls1, 1
  %exitcond44.ls1 = icmp eq i64 %indvar.inc14.ls1, 200
  br i1 %exitcond44.ls1, label %fusion.loop_inc.dim.0.ls1, label %fusion.loop_body.dim.1.ls1, !llvm.loop !49

fusion.loop_inc.dim.0.ls1:                        ; preds = %vector.body174.ls1, %fusion.loop_body.dim.1.ls1
  %exitcond45.ls1 = icmp eq i64 %52, %end.ls1
  br i1 %exitcond45.ls1, label %fusion.loop_sync.dim.0.ls1, label %fusion.loop_detach.dim.0.ls1, !llvm.loop !50
}

; Function Attrs: argmemonly nounwind stealable
define private fastcc void @cluster_25__XlaCompiledKernel_true__XlaNumConstantArgs_1__XlaNumResourceArgs_0_.42.outline_fusion.3.loop_detach.dim.0.ls1(i64 %fusion.3.indvar_address.dim.0.025.start.ls1, i64 %end.ls1, i64 %grainsize.ls1, [20 x [200 x float]]* align 16 %fusion.4.ls1) unnamed_addr #4 {
fusion.1.loop_sync.dim.0.split.ls1:
  %__cilkrts_sf = alloca %struct.__cilkrts_stack_frame, align 8
  %0 = call %struct.__cilkrts_worker* @__cilkrts_get_tls_worker() #6
  %1 = icmp eq %struct.__cilkrts_worker* %0, null
  br i1 %1, label %slowpath.i, label %fastpath.i

slowpath.i:                                       ; preds = %fusion.1.loop_sync.dim.0.split.ls1
  %2 = call %struct.__cilkrts_worker* @__cilkrts_bind_thread_1() #6
  %3 = getelementptr inbounds %struct.__cilkrts_stack_frame, %struct.__cilkrts_stack_frame* %__cilkrts_sf, i64 0, i32 0
  store atomic i32 16777344, i32* %3 release, align 8
  br label %__cilkrts_enter_frame_1.exit

fastpath.i:                                       ; preds = %fusion.1.loop_sync.dim.0.split.ls1
  %4 = getelementptr inbounds %struct.__cilkrts_stack_frame, %struct.__cilkrts_stack_frame* %__cilkrts_sf, i64 0, i32 0
  store atomic i32 16777216, i32* %4 release, align 8
  br label %__cilkrts_enter_frame_1.exit

__cilkrts_enter_frame_1.exit:                     ; preds = %slowpath.i, %fastpath.i
  %5 = phi %struct.__cilkrts_worker* [ %2, %slowpath.i ], [ %0, %fastpath.i ]
  %6 = getelementptr inbounds %struct.__cilkrts_worker, %struct.__cilkrts_worker* %5, i64 0, i32 9
  %7 = load atomic %struct.__cilkrts_stack_frame*, %struct.__cilkrts_stack_frame** %6 acquire, align 8
  %8 = getelementptr inbounds %struct.__cilkrts_stack_frame, %struct.__cilkrts_stack_frame* %__cilkrts_sf, i64 0, i32 2
  store atomic %struct.__cilkrts_stack_frame* %7, %struct.__cilkrts_stack_frame** %8 release, align 8
  %9 = getelementptr inbounds %struct.__cilkrts_stack_frame, %struct.__cilkrts_stack_frame* %__cilkrts_sf, i64 0, i32 3
  store atomic %struct.__cilkrts_worker* %5, %struct.__cilkrts_worker** %9 release, align 8
  store atomic %struct.__cilkrts_stack_frame* %__cilkrts_sf, %struct.__cilkrts_stack_frame** %6 release, align 8
  %itercount1 = sub i64 %end.ls1, %fusion.3.indvar_address.dim.0.025.start.ls1
  %10 = icmp ugt i64 %itercount1, %grainsize.ls1
  br i1 %10, label %.lr.ph.preheader, label %fusion.3.loop_detach.dim.0.ls1.preheader

.lr.ph.preheader:                                 ; preds = %__cilkrts_enter_frame_1.exit
  %11 = getelementptr inbounds %struct.__cilkrts_stack_frame, %struct.__cilkrts_stack_frame* %__cilkrts_sf, i64 0, i32 6
  %12 = getelementptr inbounds %struct.__cilkrts_stack_frame, %struct.__cilkrts_stack_frame* %__cilkrts_sf, i64 0, i32 7
  %13 = getelementptr inbounds %struct.__cilkrts_stack_frame, %struct.__cilkrts_stack_frame* %__cilkrts_sf, i64 0, i32 5
  %14 = getelementptr inbounds [5 x i8*], [5 x i8*]* %13, i64 0, i64 0
  %15 = getelementptr inbounds %struct.__cilkrts_stack_frame, %struct.__cilkrts_stack_frame* %__cilkrts_sf, i64 0, i32 5, i64 2
  %16 = bitcast [5 x i8*]* %13 to i8*
  br label %.lr.ph

fusion.3.loop_detach.dim.0.ls1.preheader:         ; preds = %.split.split, %__cilkrts_enter_frame_1.exit
  %fusion.3.indvar_address.dim.0.025.ls1.dac.lcssa = phi i64 [ %fusion.3.indvar_address.dim.0.025.start.ls1, %__cilkrts_enter_frame_1.exit ], [ %miditer, %.split.split ]
  br label %fusion.3.loop_detach.dim.0.ls1

.lr.ph:                                           ; preds = %.lr.ph.preheader, %.split.split
  %itercount3 = phi i64 [ %itercount, %.split.split ], [ %itercount1, %.lr.ph.preheader ]
  %fusion.3.indvar_address.dim.0.025.ls1.dac2 = phi i64 [ %miditer, %.split.split ], [ %fusion.3.indvar_address.dim.0.025.start.ls1, %.lr.ph.preheader ]
  %halfcount = lshr i64 %itercount3, 1
  %miditer = add nuw nsw i64 %fusion.3.indvar_address.dim.0.025.ls1.dac2, %halfcount
  call void asm sideeffect "stmxcsr $0\0A\09fnstcw $1", "*m,*m,~{dirflag},~{fpsr},~{flags}"(i32* nonnull %11, i16* nonnull %12) #6
  %17 = call i8* @llvm.frameaddress(i32 0)
  store volatile i8* %17, i8** %14, align 8
  %18 = call i8* @llvm.stacksave()
  store volatile i8* %18, i8** %15, align 8
  %19 = call i32 @llvm.eh.sjlj.setjmp(i8* nonnull %16) #7
  %20 = icmp eq i32 %19, 0
  br i1 %20, label %.lr.ph.split, label %.split.split

.lr.ph.split:                                     ; preds = %.lr.ph
  call fastcc void @cluster_25__XlaCompiledKernel_true__XlaNumConstantArgs_1__XlaNumResourceArgs_0_.42.outline_fusion.3.loop_detach.dim.0.ls1.outline_.split.otd1(i64 %fusion.3.indvar_address.dim.0.025.ls1.dac2, i64 %miditer, i64 %grainsize.ls1, [20 x [200 x float]]* %fusion.4.ls1) #6
  br label %.split.split

.split.split:                                     ; preds = %.lr.ph, %.lr.ph.split
  %itercount = sub i64 %end.ls1, %miditer
  %21 = icmp ugt i64 %itercount, %grainsize.ls1
  br i1 %21, label %.lr.ph, label %fusion.3.loop_detach.dim.0.ls1.preheader

fusion.3.loop_sync.dim.0.ls1:                     ; preds = %fusion.3.loop_detach.dim.0.ls1
  %22 = getelementptr inbounds %struct.__cilkrts_stack_frame, %struct.__cilkrts_stack_frame* %__cilkrts_sf, i64 0, i32 0
  %23 = load atomic i32, i32* %22 acquire, align 8
  %24 = and i32 %23, 2
  %25 = icmp eq i32 %24, 0
  br i1 %25, label %__cilk_sync_nothrow.exit, label %cilk.sync.savestate.i

cilk.sync.savestate.i:                            ; preds = %fusion.3.loop_sync.dim.0.ls1
  %26 = load atomic %struct.__cilkrts_worker*, %struct.__cilkrts_worker** %9 acquire, align 8
  %.elt = getelementptr inbounds %struct.__cilkrts_worker, %struct.__cilkrts_worker* %26, i64 0, i32 12, i32 0
  %.unpack = load i64, i64* %.elt, align 8
  %.elt6 = getelementptr inbounds %struct.__cilkrts_worker, %struct.__cilkrts_worker* %26, i64 0, i32 12, i32 1
  %27 = bitcast %struct.__cilkrts_pedigree** %.elt6 to i64*
  %.unpack78 = load i64, i64* %27, align 8
  %.fca.0.0.gep = getelementptr inbounds %struct.__cilkrts_stack_frame, %struct.__cilkrts_stack_frame* %__cilkrts_sf, i64 0, i32 9, i32 0, i32 0
  %.fca.0.1.gep = getelementptr inbounds %struct.__cilkrts_stack_frame, %struct.__cilkrts_stack_frame* %__cilkrts_sf, i64 0, i32 9, i32 0, i32 1
  store i64 %.unpack, i64* %.fca.0.0.gep, align 8
  %28 = bitcast %struct.__cilkrts_pedigree** %.fca.0.1.gep to i64*
  store i64 %.unpack78, i64* %28, align 8
  %29 = getelementptr inbounds %struct.__cilkrts_stack_frame, %struct.__cilkrts_stack_frame* %__cilkrts_sf, i64 0, i32 6
  %30 = getelementptr inbounds %struct.__cilkrts_stack_frame, %struct.__cilkrts_stack_frame* %__cilkrts_sf, i64 0, i32 7
  call void asm sideeffect "stmxcsr $0\0A\09fnstcw $1", "*m,*m,~{dirflag},~{fpsr},~{flags}"(i32* nonnull %29, i16* nonnull %30) #6
  %31 = getelementptr inbounds %struct.__cilkrts_stack_frame, %struct.__cilkrts_stack_frame* %__cilkrts_sf, i64 0, i32 5
  %32 = call i8* @llvm.frameaddress(i32 0) #6
  %33 = getelementptr inbounds [5 x i8*], [5 x i8*]* %31, i64 0, i64 0
  store volatile i8* %32, i8** %33, align 8
  %34 = call i8* @llvm.stacksave() #6
  %35 = getelementptr inbounds %struct.__cilkrts_stack_frame, %struct.__cilkrts_stack_frame* %__cilkrts_sf, i64 0, i32 5, i64 2
  store volatile i8* %34, i8** %35, align 8
  %36 = bitcast [5 x i8*]* %31 to i8*
  %37 = call i32 @llvm.eh.sjlj.setjmp(i8* nonnull %36) #8
  %38 = icmp eq i32 %37, 0
  br i1 %38, label %cilk.sync.runtimecall.i, label %__cilk_sync_nothrow.exit

cilk.sync.runtimecall.i:                          ; preds = %cilk.sync.savestate.i
  call void @__cilkrts_sync(%struct.__cilkrts_stack_frame* nonnull %__cilkrts_sf) #6
  br label %__cilk_sync_nothrow.exit

__cilk_sync_nothrow.exit:                         ; preds = %fusion.3.loop_sync.dim.0.ls1, %cilk.sync.savestate.i, %cilk.sync.runtimecall.i
  %39 = load atomic %struct.__cilkrts_worker*, %struct.__cilkrts_worker** %9 acquire, align 8
  %40 = getelementptr inbounds %struct.__cilkrts_worker, %struct.__cilkrts_worker* %39, i64 0, i32 12, i32 0
  %41 = load i64, i64* %40, align 8
  %42 = add i64 %41, 1
  store i64 %42, i64* %40, align 8
  %43 = load atomic %struct.__cilkrts_worker*, %struct.__cilkrts_worker** %9 acquire, align 8
  %44 = bitcast %struct.__cilkrts_stack_frame** %8 to i64*
  %45 = load i64, i64* %44, align 8
  %46 = getelementptr inbounds %struct.__cilkrts_worker, %struct.__cilkrts_worker* %43, i64 0, i32 9
  %47 = bitcast %struct.__cilkrts_stack_frame** %46 to i64*
  store atomic i64 %45, i64* %47 release, align 8
  store atomic %struct.__cilkrts_stack_frame* null, %struct.__cilkrts_stack_frame** %8 release, align 8
  %48 = load atomic i32, i32* %22 acquire, align 8
  %49 = icmp eq i32 %48, 16777216
  br i1 %49, label %__cilk_parent_epilogue.exit, label %body.i

body.i:                                           ; preds = %__cilk_sync_nothrow.exit
  call void @__cilkrts_leave_frame(%struct.__cilkrts_stack_frame* nonnull %__cilkrts_sf) #6
  br label %__cilk_parent_epilogue.exit

__cilk_parent_epilogue.exit:                      ; preds = %__cilk_sync_nothrow.exit, %body.i
  ret void

fusion.3.loop_detach.dim.0.ls1:                   ; preds = %fusion.3.loop_detach.dim.0.ls1.preheader, %fusion.3.loop_detach.dim.0.ls1
  %fusion.3.indvar_address.dim.0.025.ls1 = phi i64 [ %indvar.inc11.ls1, %fusion.3.loop_detach.dim.0.ls1 ], [ %fusion.3.indvar_address.dim.0.025.ls1.dac.lcssa, %fusion.3.loop_detach.dim.0.ls1.preheader ]
  %50 = getelementptr inbounds [20 x [200 x float]], [20 x [200 x float]]* %fusion.4.ls1, i64 0, i64 %fusion.3.indvar_address.dim.0.025.ls1, i64 0
  %51 = bitcast float* %50 to <8 x float>*
  %wide.load170.ls1 = load <8 x float>, <8 x float>* %51, align 16, !alias.scope !51, !noalias !52
  %52 = getelementptr inbounds [20 x [200 x float]], [20 x [200 x float]]* %fusion.4.ls1, i64 0, i64 %fusion.3.indvar_address.dim.0.025.ls1, i64 8
  %53 = bitcast float* %52 to <8 x float>*
  %wide.load171.ls1 = load <8 x float>, <8 x float>* %53, align 16, !alias.scope !51, !noalias !52
  %54 = getelementptr inbounds [20 x [200 x float]], [20 x [200 x float]]* %fusion.4.ls1, i64 0, i64 %fusion.3.indvar_address.dim.0.025.ls1, i64 16
  %55 = bitcast float* %54 to <8 x float>*
  %wide.load172.ls1 = load <8 x float>, <8 x float>* %55, align 16, !alias.scope !51, !noalias !52
  %56 = getelementptr inbounds [20 x [200 x float]], [20 x [200 x float]]* %fusion.4.ls1, i64 0, i64 %fusion.3.indvar_address.dim.0.025.ls1, i64 24
  %57 = bitcast float* %56 to <8 x float>*
  %wide.load173.ls1 = load <8 x float>, <8 x float>* %57, align 16, !alias.scope !51, !noalias !52
  %58 = fmul fast <8 x float> %wide.load170.ls1, <float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01>
  %59 = fmul fast <8 x float> %wide.load171.ls1, <float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01>
  %60 = fmul fast <8 x float> %wide.load172.ls1, <float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01>
  %61 = fmul fast <8 x float> %wide.load173.ls1, <float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01>
  %62 = fadd fast <8 x float> %58, <float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01>
  %63 = fadd fast <8 x float> %59, <float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01>
  %64 = fadd fast <8 x float> %60, <float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01>
  %65 = fadd fast <8 x float> %61, <float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01>
  store <8 x float> %62, <8 x float>* %51, align 16, !alias.scope !51, !noalias !52
  store <8 x float> %63, <8 x float>* %53, align 16, !alias.scope !51, !noalias !52
  store <8 x float> %64, <8 x float>* %55, align 16, !alias.scope !51, !noalias !52
  store <8 x float> %65, <8 x float>* %57, align 16, !alias.scope !51, !noalias !52
  %66 = getelementptr inbounds [20 x [200 x float]], [20 x [200 x float]]* %fusion.4.ls1, i64 0, i64 %fusion.3.indvar_address.dim.0.025.ls1, i64 32
  %67 = bitcast float* %66 to <8 x float>*
  %wide.load170.1.ls1 = load <8 x float>, <8 x float>* %67, align 16, !alias.scope !51, !noalias !52
  %68 = getelementptr inbounds [20 x [200 x float]], [20 x [200 x float]]* %fusion.4.ls1, i64 0, i64 %fusion.3.indvar_address.dim.0.025.ls1, i64 40
  %69 = bitcast float* %68 to <8 x float>*
  %wide.load171.1.ls1 = load <8 x float>, <8 x float>* %69, align 16, !alias.scope !51, !noalias !52
  %70 = getelementptr inbounds [20 x [200 x float]], [20 x [200 x float]]* %fusion.4.ls1, i64 0, i64 %fusion.3.indvar_address.dim.0.025.ls1, i64 48
  %71 = bitcast float* %70 to <8 x float>*
  %wide.load172.1.ls1 = load <8 x float>, <8 x float>* %71, align 16, !alias.scope !51, !noalias !52
  %72 = getelementptr inbounds [20 x [200 x float]], [20 x [200 x float]]* %fusion.4.ls1, i64 0, i64 %fusion.3.indvar_address.dim.0.025.ls1, i64 56
  %73 = bitcast float* %72 to <8 x float>*
  %wide.load173.1.ls1 = load <8 x float>, <8 x float>* %73, align 16, !alias.scope !51, !noalias !52
  %74 = fmul fast <8 x float> %wide.load170.1.ls1, <float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01>
  %75 = fmul fast <8 x float> %wide.load171.1.ls1, <float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01>
  %76 = fmul fast <8 x float> %wide.load172.1.ls1, <float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01>
  %77 = fmul fast <8 x float> %wide.load173.1.ls1, <float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01>
  %78 = fadd fast <8 x float> %74, <float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01>
  %79 = fadd fast <8 x float> %75, <float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01>
  %80 = fadd fast <8 x float> %76, <float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01>
  %81 = fadd fast <8 x float> %77, <float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01>
  store <8 x float> %78, <8 x float>* %67, align 16, !alias.scope !51, !noalias !52
  store <8 x float> %79, <8 x float>* %69, align 16, !alias.scope !51, !noalias !52
  store <8 x float> %80, <8 x float>* %71, align 16, !alias.scope !51, !noalias !52
  store <8 x float> %81, <8 x float>* %73, align 16, !alias.scope !51, !noalias !52
  %82 = getelementptr inbounds [20 x [200 x float]], [20 x [200 x float]]* %fusion.4.ls1, i64 0, i64 %fusion.3.indvar_address.dim.0.025.ls1, i64 64
  %83 = bitcast float* %82 to <8 x float>*
  %wide.load170.2.ls1 = load <8 x float>, <8 x float>* %83, align 16, !alias.scope !51, !noalias !52
  %84 = getelementptr inbounds [20 x [200 x float]], [20 x [200 x float]]* %fusion.4.ls1, i64 0, i64 %fusion.3.indvar_address.dim.0.025.ls1, i64 72
  %85 = bitcast float* %84 to <8 x float>*
  %wide.load171.2.ls1 = load <8 x float>, <8 x float>* %85, align 16, !alias.scope !51, !noalias !52
  %86 = getelementptr inbounds [20 x [200 x float]], [20 x [200 x float]]* %fusion.4.ls1, i64 0, i64 %fusion.3.indvar_address.dim.0.025.ls1, i64 80
  %87 = bitcast float* %86 to <8 x float>*
  %wide.load172.2.ls1 = load <8 x float>, <8 x float>* %87, align 16, !alias.scope !51, !noalias !52
  %88 = getelementptr inbounds [20 x [200 x float]], [20 x [200 x float]]* %fusion.4.ls1, i64 0, i64 %fusion.3.indvar_address.dim.0.025.ls1, i64 88
  %89 = bitcast float* %88 to <8 x float>*
  %wide.load173.2.ls1 = load <8 x float>, <8 x float>* %89, align 16, !alias.scope !51, !noalias !52
  %90 = fmul fast <8 x float> %wide.load170.2.ls1, <float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01>
  %91 = fmul fast <8 x float> %wide.load171.2.ls1, <float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01>
  %92 = fmul fast <8 x float> %wide.load172.2.ls1, <float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01>
  %93 = fmul fast <8 x float> %wide.load173.2.ls1, <float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01>
  %94 = fadd fast <8 x float> %90, <float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01>
  %95 = fadd fast <8 x float> %91, <float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01>
  %96 = fadd fast <8 x float> %92, <float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01>
  %97 = fadd fast <8 x float> %93, <float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01>
  store <8 x float> %94, <8 x float>* %83, align 16, !alias.scope !51, !noalias !52
  store <8 x float> %95, <8 x float>* %85, align 16, !alias.scope !51, !noalias !52
  store <8 x float> %96, <8 x float>* %87, align 16, !alias.scope !51, !noalias !52
  store <8 x float> %97, <8 x float>* %89, align 16, !alias.scope !51, !noalias !52
  %98 = getelementptr inbounds [20 x [200 x float]], [20 x [200 x float]]* %fusion.4.ls1, i64 0, i64 %fusion.3.indvar_address.dim.0.025.ls1, i64 96
  %99 = bitcast float* %98 to <8 x float>*
  %wide.load170.3.ls1 = load <8 x float>, <8 x float>* %99, align 16, !alias.scope !51, !noalias !52
  %100 = getelementptr inbounds [20 x [200 x float]], [20 x [200 x float]]* %fusion.4.ls1, i64 0, i64 %fusion.3.indvar_address.dim.0.025.ls1, i64 104
  %101 = bitcast float* %100 to <8 x float>*
  %wide.load171.3.ls1 = load <8 x float>, <8 x float>* %101, align 16, !alias.scope !51, !noalias !52
  %102 = getelementptr inbounds [20 x [200 x float]], [20 x [200 x float]]* %fusion.4.ls1, i64 0, i64 %fusion.3.indvar_address.dim.0.025.ls1, i64 112
  %103 = bitcast float* %102 to <8 x float>*
  %wide.load172.3.ls1 = load <8 x float>, <8 x float>* %103, align 16, !alias.scope !51, !noalias !52
  %104 = getelementptr inbounds [20 x [200 x float]], [20 x [200 x float]]* %fusion.4.ls1, i64 0, i64 %fusion.3.indvar_address.dim.0.025.ls1, i64 120
  %105 = bitcast float* %104 to <8 x float>*
  %wide.load173.3.ls1 = load <8 x float>, <8 x float>* %105, align 16, !alias.scope !51, !noalias !52
  %106 = fmul fast <8 x float> %wide.load170.3.ls1, <float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01>
  %107 = fmul fast <8 x float> %wide.load171.3.ls1, <float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01>
  %108 = fmul fast <8 x float> %wide.load172.3.ls1, <float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01>
  %109 = fmul fast <8 x float> %wide.load173.3.ls1, <float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01>
  %110 = fadd fast <8 x float> %106, <float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01>
  %111 = fadd fast <8 x float> %107, <float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01>
  %112 = fadd fast <8 x float> %108, <float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01>
  %113 = fadd fast <8 x float> %109, <float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01>
  store <8 x float> %110, <8 x float>* %99, align 16, !alias.scope !51, !noalias !52
  store <8 x float> %111, <8 x float>* %101, align 16, !alias.scope !51, !noalias !52
  store <8 x float> %112, <8 x float>* %103, align 16, !alias.scope !51, !noalias !52
  store <8 x float> %113, <8 x float>* %105, align 16, !alias.scope !51, !noalias !52
  %114 = getelementptr inbounds [20 x [200 x float]], [20 x [200 x float]]* %fusion.4.ls1, i64 0, i64 %fusion.3.indvar_address.dim.0.025.ls1, i64 128
  %115 = bitcast float* %114 to <8 x float>*
  %wide.load170.4.ls1 = load <8 x float>, <8 x float>* %115, align 16, !alias.scope !51, !noalias !52
  %116 = getelementptr inbounds [20 x [200 x float]], [20 x [200 x float]]* %fusion.4.ls1, i64 0, i64 %fusion.3.indvar_address.dim.0.025.ls1, i64 136
  %117 = bitcast float* %116 to <8 x float>*
  %wide.load171.4.ls1 = load <8 x float>, <8 x float>* %117, align 16, !alias.scope !51, !noalias !52
  %118 = getelementptr inbounds [20 x [200 x float]], [20 x [200 x float]]* %fusion.4.ls1, i64 0, i64 %fusion.3.indvar_address.dim.0.025.ls1, i64 144
  %119 = bitcast float* %118 to <8 x float>*
  %wide.load172.4.ls1 = load <8 x float>, <8 x float>* %119, align 16, !alias.scope !51, !noalias !52
  %120 = getelementptr inbounds [20 x [200 x float]], [20 x [200 x float]]* %fusion.4.ls1, i64 0, i64 %fusion.3.indvar_address.dim.0.025.ls1, i64 152
  %121 = bitcast float* %120 to <8 x float>*
  %wide.load173.4.ls1 = load <8 x float>, <8 x float>* %121, align 16, !alias.scope !51, !noalias !52
  %122 = fmul fast <8 x float> %wide.load170.4.ls1, <float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01>
  %123 = fmul fast <8 x float> %wide.load171.4.ls1, <float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01>
  %124 = fmul fast <8 x float> %wide.load172.4.ls1, <float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01>
  %125 = fmul fast <8 x float> %wide.load173.4.ls1, <float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01>
  %126 = fadd fast <8 x float> %122, <float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01>
  %127 = fadd fast <8 x float> %123, <float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01>
  %128 = fadd fast <8 x float> %124, <float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01>
  %129 = fadd fast <8 x float> %125, <float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01>
  store <8 x float> %126, <8 x float>* %115, align 16, !alias.scope !51, !noalias !52
  store <8 x float> %127, <8 x float>* %117, align 16, !alias.scope !51, !noalias !52
  store <8 x float> %128, <8 x float>* %119, align 16, !alias.scope !51, !noalias !52
  store <8 x float> %129, <8 x float>* %121, align 16, !alias.scope !51, !noalias !52
  %130 = getelementptr inbounds [20 x [200 x float]], [20 x [200 x float]]* %fusion.4.ls1, i64 0, i64 %fusion.3.indvar_address.dim.0.025.ls1, i64 160
  %131 = bitcast float* %130 to <8 x float>*
  %wide.load170.5.ls1 = load <8 x float>, <8 x float>* %131, align 16, !alias.scope !51, !noalias !52
  %132 = getelementptr inbounds [20 x [200 x float]], [20 x [200 x float]]* %fusion.4.ls1, i64 0, i64 %fusion.3.indvar_address.dim.0.025.ls1, i64 168
  %133 = bitcast float* %132 to <8 x float>*
  %wide.load171.5.ls1 = load <8 x float>, <8 x float>* %133, align 16, !alias.scope !51, !noalias !52
  %134 = getelementptr inbounds [20 x [200 x float]], [20 x [200 x float]]* %fusion.4.ls1, i64 0, i64 %fusion.3.indvar_address.dim.0.025.ls1, i64 176
  %135 = bitcast float* %134 to <8 x float>*
  %wide.load172.5.ls1 = load <8 x float>, <8 x float>* %135, align 16, !alias.scope !51, !noalias !52
  %136 = getelementptr inbounds [20 x [200 x float]], [20 x [200 x float]]* %fusion.4.ls1, i64 0, i64 %fusion.3.indvar_address.dim.0.025.ls1, i64 184
  %137 = bitcast float* %136 to <8 x float>*
  %wide.load173.5.ls1 = load <8 x float>, <8 x float>* %137, align 16, !alias.scope !51, !noalias !52
  %138 = fmul fast <8 x float> %wide.load170.5.ls1, <float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01>
  %139 = fmul fast <8 x float> %wide.load171.5.ls1, <float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01>
  %140 = fmul fast <8 x float> %wide.load172.5.ls1, <float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01>
  %141 = fmul fast <8 x float> %wide.load173.5.ls1, <float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01>
  %142 = fadd fast <8 x float> %138, <float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01>
  %143 = fadd fast <8 x float> %139, <float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01>
  %144 = fadd fast <8 x float> %140, <float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01>
  %145 = fadd fast <8 x float> %141, <float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01>
  store <8 x float> %142, <8 x float>* %131, align 16, !alias.scope !51, !noalias !52
  store <8 x float> %143, <8 x float>* %133, align 16, !alias.scope !51, !noalias !52
  store <8 x float> %144, <8 x float>* %135, align 16, !alias.scope !51, !noalias !52
  store <8 x float> %145, <8 x float>* %137, align 16, !alias.scope !51, !noalias !52
  %146 = getelementptr inbounds [20 x [200 x float]], [20 x [200 x float]]* %fusion.4.ls1, i64 0, i64 %fusion.3.indvar_address.dim.0.025.ls1, i64 192
  %147 = load float, float* %146, align 16, !alias.scope !51, !noalias !52
  %148 = fmul fast float %147, 5.000000e-01
  %149 = fadd fast float %148, 5.000000e-01
  store float %149, float* %146, align 16, !alias.scope !51, !noalias !52
  %150 = getelementptr inbounds [20 x [200 x float]], [20 x [200 x float]]* %fusion.4.ls1, i64 0, i64 %fusion.3.indvar_address.dim.0.025.ls1, i64 193
  %151 = load float, float* %150, align 4, !alias.scope !51, !noalias !52
  %152 = fmul fast float %151, 5.000000e-01
  %153 = fadd fast float %152, 5.000000e-01
  store float %153, float* %150, align 4, !alias.scope !51, !noalias !52
  %154 = getelementptr inbounds [20 x [200 x float]], [20 x [200 x float]]* %fusion.4.ls1, i64 0, i64 %fusion.3.indvar_address.dim.0.025.ls1, i64 194
  %155 = load float, float* %154, align 8, !alias.scope !51, !noalias !52
  %156 = fmul fast float %155, 5.000000e-01
  %157 = fadd fast float %156, 5.000000e-01
  store float %157, float* %154, align 8, !alias.scope !51, !noalias !52
  %158 = getelementptr inbounds [20 x [200 x float]], [20 x [200 x float]]* %fusion.4.ls1, i64 0, i64 %fusion.3.indvar_address.dim.0.025.ls1, i64 195
  %159 = load float, float* %158, align 4, !alias.scope !51, !noalias !52
  %160 = fmul fast float %159, 5.000000e-01
  %161 = fadd fast float %160, 5.000000e-01
  store float %161, float* %158, align 4, !alias.scope !51, !noalias !52
  %162 = getelementptr inbounds [20 x [200 x float]], [20 x [200 x float]]* %fusion.4.ls1, i64 0, i64 %fusion.3.indvar_address.dim.0.025.ls1, i64 196
  %163 = load float, float* %162, align 16, !alias.scope !51, !noalias !52
  %164 = fmul fast float %163, 5.000000e-01
  %165 = fadd fast float %164, 5.000000e-01
  store float %165, float* %162, align 16, !alias.scope !51, !noalias !52
  %166 = getelementptr inbounds [20 x [200 x float]], [20 x [200 x float]]* %fusion.4.ls1, i64 0, i64 %fusion.3.indvar_address.dim.0.025.ls1, i64 197
  %167 = load float, float* %166, align 4, !alias.scope !51, !noalias !52
  %168 = fmul fast float %167, 5.000000e-01
  %169 = fadd fast float %168, 5.000000e-01
  store float %169, float* %166, align 4, !alias.scope !51, !noalias !52
  %170 = getelementptr inbounds [20 x [200 x float]], [20 x [200 x float]]* %fusion.4.ls1, i64 0, i64 %fusion.3.indvar_address.dim.0.025.ls1, i64 198
  %171 = load float, float* %170, align 8, !alias.scope !51, !noalias !52
  %172 = fmul fast float %171, 5.000000e-01
  %173 = fadd fast float %172, 5.000000e-01
  store float %173, float* %170, align 8, !alias.scope !51, !noalias !52
  %174 = getelementptr inbounds [20 x [200 x float]], [20 x [200 x float]]* %fusion.4.ls1, i64 0, i64 %fusion.3.indvar_address.dim.0.025.ls1, i64 199
  %175 = load float, float* %174, align 4, !alias.scope !51, !noalias !52
  %176 = fmul fast float %175, 5.000000e-01
  %177 = fadd fast float %176, 5.000000e-01
  store float %177, float* %174, align 4, !alias.scope !51, !noalias !52
  %indvar.inc11.ls1 = add nuw nsw i64 %fusion.3.indvar_address.dim.0.025.ls1, 1
  %exitcond47.ls1 = icmp eq i64 %indvar.inc11.ls1, %end.ls1
  br i1 %exitcond47.ls1, label %fusion.3.loop_sync.dim.0.ls1, label %fusion.3.loop_detach.dim.0.ls1, !llvm.loop !53
}

; Function Attrs: argmemonly nounwind stealable
define private fastcc void @cluster_25__XlaCompiledKernel_true__XlaNumConstantArgs_1__XlaNumResourceArgs_0_.42.outline_fusion.1.loop_detach.dim.0.ls1(i64 %fusion.1.indvar_address.dim.0.028.start.ls1, i64 %end.ls1, i64 %grainsize.ls1, [20 x [200 x float]]* readonly align 16 %fusion.4.ls1, [20 x [200 x float]]* readonly align 16 %fusion.2.ls1, [20 x [200 x float]]* align 16 %fusion.1.ls1) unnamed_addr #4 {
fusion.4.loop_exit.dim.0.ls1:
  %__cilkrts_sf = alloca %struct.__cilkrts_stack_frame, align 8
  %0 = call %struct.__cilkrts_worker* @__cilkrts_get_tls_worker() #6
  %1 = icmp eq %struct.__cilkrts_worker* %0, null
  br i1 %1, label %slowpath.i, label %fastpath.i

slowpath.i:                                       ; preds = %fusion.4.loop_exit.dim.0.ls1
  %2 = call %struct.__cilkrts_worker* @__cilkrts_bind_thread_1() #6
  %3 = getelementptr inbounds %struct.__cilkrts_stack_frame, %struct.__cilkrts_stack_frame* %__cilkrts_sf, i64 0, i32 0
  store atomic i32 16777344, i32* %3 release, align 8
  br label %__cilkrts_enter_frame_1.exit

fastpath.i:                                       ; preds = %fusion.4.loop_exit.dim.0.ls1
  %4 = getelementptr inbounds %struct.__cilkrts_stack_frame, %struct.__cilkrts_stack_frame* %__cilkrts_sf, i64 0, i32 0
  store atomic i32 16777216, i32* %4 release, align 8
  br label %__cilkrts_enter_frame_1.exit

__cilkrts_enter_frame_1.exit:                     ; preds = %slowpath.i, %fastpath.i
  %5 = phi %struct.__cilkrts_worker* [ %2, %slowpath.i ], [ %0, %fastpath.i ]
  %6 = getelementptr inbounds %struct.__cilkrts_worker, %struct.__cilkrts_worker* %5, i64 0, i32 9
  %7 = load atomic %struct.__cilkrts_stack_frame*, %struct.__cilkrts_stack_frame** %6 acquire, align 8
  %8 = getelementptr inbounds %struct.__cilkrts_stack_frame, %struct.__cilkrts_stack_frame* %__cilkrts_sf, i64 0, i32 2
  store atomic %struct.__cilkrts_stack_frame* %7, %struct.__cilkrts_stack_frame** %8 release, align 8
  %9 = getelementptr inbounds %struct.__cilkrts_stack_frame, %struct.__cilkrts_stack_frame* %__cilkrts_sf, i64 0, i32 3
  store atomic %struct.__cilkrts_worker* %5, %struct.__cilkrts_worker** %9 release, align 8
  store atomic %struct.__cilkrts_stack_frame* %__cilkrts_sf, %struct.__cilkrts_stack_frame** %6 release, align 8
  %itercount1 = sub i64 %end.ls1, %fusion.1.indvar_address.dim.0.028.start.ls1
  %10 = icmp ugt i64 %itercount1, %grainsize.ls1
  br i1 %10, label %.lr.ph.preheader, label %fusion.1.loop_detach.dim.0.ls1.preheader

.lr.ph.preheader:                                 ; preds = %__cilkrts_enter_frame_1.exit
  %11 = getelementptr inbounds %struct.__cilkrts_stack_frame, %struct.__cilkrts_stack_frame* %__cilkrts_sf, i64 0, i32 6
  %12 = getelementptr inbounds %struct.__cilkrts_stack_frame, %struct.__cilkrts_stack_frame* %__cilkrts_sf, i64 0, i32 7
  %13 = getelementptr inbounds %struct.__cilkrts_stack_frame, %struct.__cilkrts_stack_frame* %__cilkrts_sf, i64 0, i32 5
  %14 = getelementptr inbounds [5 x i8*], [5 x i8*]* %13, i64 0, i64 0
  %15 = getelementptr inbounds %struct.__cilkrts_stack_frame, %struct.__cilkrts_stack_frame* %__cilkrts_sf, i64 0, i32 5, i64 2
  %16 = bitcast [5 x i8*]* %13 to i8*
  br label %.lr.ph

fusion.1.loop_detach.dim.0.ls1.preheader:         ; preds = %.split.split, %__cilkrts_enter_frame_1.exit
  %fusion.1.indvar_address.dim.0.028.ls1.dac.lcssa = phi i64 [ %fusion.1.indvar_address.dim.0.028.start.ls1, %__cilkrts_enter_frame_1.exit ], [ %miditer, %.split.split ]
  br label %fusion.1.loop_detach.dim.0.ls1

.lr.ph:                                           ; preds = %.lr.ph.preheader, %.split.split
  %itercount3 = phi i64 [ %itercount, %.split.split ], [ %itercount1, %.lr.ph.preheader ]
  %fusion.1.indvar_address.dim.0.028.ls1.dac2 = phi i64 [ %miditer, %.split.split ], [ %fusion.1.indvar_address.dim.0.028.start.ls1, %.lr.ph.preheader ]
  %halfcount = lshr i64 %itercount3, 1
  %miditer = add nuw nsw i64 %fusion.1.indvar_address.dim.0.028.ls1.dac2, %halfcount
  call void asm sideeffect "stmxcsr $0\0A\09fnstcw $1", "*m,*m,~{dirflag},~{fpsr},~{flags}"(i32* nonnull %11, i16* nonnull %12) #6
  %17 = call i8* @llvm.frameaddress(i32 0)
  store volatile i8* %17, i8** %14, align 8
  %18 = call i8* @llvm.stacksave()
  store volatile i8* %18, i8** %15, align 8
  %19 = call i32 @llvm.eh.sjlj.setjmp(i8* nonnull %16) #7
  %20 = icmp eq i32 %19, 0
  br i1 %20, label %.lr.ph.split, label %.split.split

.lr.ph.split:                                     ; preds = %.lr.ph
  call fastcc void @cluster_25__XlaCompiledKernel_true__XlaNumConstantArgs_1__XlaNumResourceArgs_0_.42.outline_fusion.1.loop_detach.dim.0.ls1.outline_.split.otd1(i64 %fusion.1.indvar_address.dim.0.028.ls1.dac2, i64 %miditer, i64 %grainsize.ls1, [20 x [200 x float]]* %fusion.4.ls1, [20 x [200 x float]]* %fusion.2.ls1, [20 x [200 x float]]* %fusion.1.ls1) #6
  br label %.split.split

.split.split:                                     ; preds = %.lr.ph, %.lr.ph.split
  %itercount = sub i64 %end.ls1, %miditer
  %21 = icmp ugt i64 %itercount, %grainsize.ls1
  br i1 %21, label %.lr.ph, label %fusion.1.loop_detach.dim.0.ls1.preheader

fusion.1.loop_sync.dim.0.ls1:                     ; preds = %fusion.1.loop_detach.dim.0.ls1
  %22 = getelementptr inbounds %struct.__cilkrts_stack_frame, %struct.__cilkrts_stack_frame* %__cilkrts_sf, i64 0, i32 0
  %23 = load atomic i32, i32* %22 acquire, align 8
  %24 = and i32 %23, 2
  %25 = icmp eq i32 %24, 0
  br i1 %25, label %__cilk_sync_nothrow.exit, label %cilk.sync.savestate.i

cilk.sync.savestate.i:                            ; preds = %fusion.1.loop_sync.dim.0.ls1
  %26 = load atomic %struct.__cilkrts_worker*, %struct.__cilkrts_worker** %9 acquire, align 8
  %.elt = getelementptr inbounds %struct.__cilkrts_worker, %struct.__cilkrts_worker* %26, i64 0, i32 12, i32 0
  %.unpack = load i64, i64* %.elt, align 8
  %.elt6 = getelementptr inbounds %struct.__cilkrts_worker, %struct.__cilkrts_worker* %26, i64 0, i32 12, i32 1
  %27 = bitcast %struct.__cilkrts_pedigree** %.elt6 to i64*
  %.unpack78 = load i64, i64* %27, align 8
  %.fca.0.0.gep = getelementptr inbounds %struct.__cilkrts_stack_frame, %struct.__cilkrts_stack_frame* %__cilkrts_sf, i64 0, i32 9, i32 0, i32 0
  %.fca.0.1.gep = getelementptr inbounds %struct.__cilkrts_stack_frame, %struct.__cilkrts_stack_frame* %__cilkrts_sf, i64 0, i32 9, i32 0, i32 1
  store i64 %.unpack, i64* %.fca.0.0.gep, align 8
  %28 = bitcast %struct.__cilkrts_pedigree** %.fca.0.1.gep to i64*
  store i64 %.unpack78, i64* %28, align 8
  %29 = getelementptr inbounds %struct.__cilkrts_stack_frame, %struct.__cilkrts_stack_frame* %__cilkrts_sf, i64 0, i32 6
  %30 = getelementptr inbounds %struct.__cilkrts_stack_frame, %struct.__cilkrts_stack_frame* %__cilkrts_sf, i64 0, i32 7
  call void asm sideeffect "stmxcsr $0\0A\09fnstcw $1", "*m,*m,~{dirflag},~{fpsr},~{flags}"(i32* nonnull %29, i16* nonnull %30) #6
  %31 = getelementptr inbounds %struct.__cilkrts_stack_frame, %struct.__cilkrts_stack_frame* %__cilkrts_sf, i64 0, i32 5
  %32 = call i8* @llvm.frameaddress(i32 0) #6
  %33 = getelementptr inbounds [5 x i8*], [5 x i8*]* %31, i64 0, i64 0
  store volatile i8* %32, i8** %33, align 8
  %34 = call i8* @llvm.stacksave() #6
  %35 = getelementptr inbounds %struct.__cilkrts_stack_frame, %struct.__cilkrts_stack_frame* %__cilkrts_sf, i64 0, i32 5, i64 2
  store volatile i8* %34, i8** %35, align 8
  %36 = bitcast [5 x i8*]* %31 to i8*
  %37 = call i32 @llvm.eh.sjlj.setjmp(i8* nonnull %36) #8
  %38 = icmp eq i32 %37, 0
  br i1 %38, label %cilk.sync.runtimecall.i, label %__cilk_sync_nothrow.exit

cilk.sync.runtimecall.i:                          ; preds = %cilk.sync.savestate.i
  call void @__cilkrts_sync(%struct.__cilkrts_stack_frame* nonnull %__cilkrts_sf) #6
  br label %__cilk_sync_nothrow.exit

__cilk_sync_nothrow.exit:                         ; preds = %fusion.1.loop_sync.dim.0.ls1, %cilk.sync.savestate.i, %cilk.sync.runtimecall.i
  %39 = load atomic %struct.__cilkrts_worker*, %struct.__cilkrts_worker** %9 acquire, align 8
  %40 = getelementptr inbounds %struct.__cilkrts_worker, %struct.__cilkrts_worker* %39, i64 0, i32 12, i32 0
  %41 = load i64, i64* %40, align 8
  %42 = add i64 %41, 1
  store i64 %42, i64* %40, align 8
  %43 = load atomic %struct.__cilkrts_worker*, %struct.__cilkrts_worker** %9 acquire, align 8
  %44 = bitcast %struct.__cilkrts_stack_frame** %8 to i64*
  %45 = load i64, i64* %44, align 8
  %46 = getelementptr inbounds %struct.__cilkrts_worker, %struct.__cilkrts_worker* %43, i64 0, i32 9
  %47 = bitcast %struct.__cilkrts_stack_frame** %46 to i64*
  store atomic i64 %45, i64* %47 release, align 8
  store atomic %struct.__cilkrts_stack_frame* null, %struct.__cilkrts_stack_frame** %8 release, align 8
  %48 = load atomic i32, i32* %22 acquire, align 8
  %49 = icmp eq i32 %48, 16777216
  br i1 %49, label %__cilk_parent_epilogue.exit, label %body.i

body.i:                                           ; preds = %__cilk_sync_nothrow.exit
  call void @__cilkrts_leave_frame(%struct.__cilkrts_stack_frame* nonnull %__cilkrts_sf) #6
  br label %__cilk_parent_epilogue.exit

__cilk_parent_epilogue.exit:                      ; preds = %__cilk_sync_nothrow.exit, %body.i
  ret void

fusion.1.loop_detach.dim.0.ls1:                   ; preds = %fusion.1.loop_detach.dim.0.ls1.preheader, %fusion.1.loop_detach.dim.0.ls1
  %fusion.1.indvar_address.dim.0.028.ls1 = phi i64 [ %indvar.inc9.ls1, %fusion.1.loop_detach.dim.0.ls1 ], [ %fusion.1.indvar_address.dim.0.028.ls1.dac.lcssa, %fusion.1.loop_detach.dim.0.ls1.preheader ]
  %50 = getelementptr inbounds [20 x [200 x float]], [20 x [200 x float]]* %fusion.4.ls1, i64 0, i64 %fusion.1.indvar_address.dim.0.028.ls1, i64 0
  %51 = bitcast float* %50 to <8 x float>*
  %wide.load148.ls1 = load <8 x float>, <8 x float>* %51, align 16, !alias.scope !51, !noalias !52
  %52 = getelementptr inbounds [20 x [200 x float]], [20 x [200 x float]]* %fusion.4.ls1, i64 0, i64 %fusion.1.indvar_address.dim.0.028.ls1, i64 8
  %53 = bitcast float* %52 to <8 x float>*
  %wide.load149.ls1 = load <8 x float>, <8 x float>* %53, align 16, !alias.scope !51, !noalias !52
  %54 = getelementptr inbounds [20 x [200 x float]], [20 x [200 x float]]* %fusion.4.ls1, i64 0, i64 %fusion.1.indvar_address.dim.0.028.ls1, i64 16
  %55 = bitcast float* %54 to <8 x float>*
  %wide.load150.ls1 = load <8 x float>, <8 x float>* %55, align 16, !alias.scope !51, !noalias !52
  %56 = getelementptr inbounds [20 x [200 x float]], [20 x [200 x float]]* %fusion.4.ls1, i64 0, i64 %fusion.1.indvar_address.dim.0.028.ls1, i64 24
  %57 = bitcast float* %56 to <8 x float>*
  %wide.load151.ls1 = load <8 x float>, <8 x float>* %57, align 16, !alias.scope !51, !noalias !52
  %58 = fmul fast <8 x float> %wide.load148.ls1, <float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01>
  %59 = fmul fast <8 x float> %wide.load149.ls1, <float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01>
  %60 = fmul fast <8 x float> %wide.load150.ls1, <float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01>
  %61 = fmul fast <8 x float> %wide.load151.ls1, <float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01>
  %62 = fadd fast <8 x float> %58, <float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01>
  %63 = fadd fast <8 x float> %59, <float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01>
  %64 = fadd fast <8 x float> %60, <float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01>
  %65 = fadd fast <8 x float> %61, <float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01>
  %66 = getelementptr inbounds [20 x [200 x float]], [20 x [200 x float]]* %fusion.2.ls1, i64 0, i64 %fusion.1.indvar_address.dim.0.028.ls1, i64 0
  %67 = bitcast float* %66 to <8 x float>*
  %wide.load152.ls1 = load <8 x float>, <8 x float>* %67, align 16, !alias.scope !54, !noalias !55
  %68 = getelementptr inbounds [20 x [200 x float]], [20 x [200 x float]]* %fusion.2.ls1, i64 0, i64 %fusion.1.indvar_address.dim.0.028.ls1, i64 8
  %69 = bitcast float* %68 to <8 x float>*
  %wide.load153.ls1 = load <8 x float>, <8 x float>* %69, align 16, !alias.scope !54, !noalias !55
  %70 = getelementptr inbounds [20 x [200 x float]], [20 x [200 x float]]* %fusion.2.ls1, i64 0, i64 %fusion.1.indvar_address.dim.0.028.ls1, i64 16
  %71 = bitcast float* %70 to <8 x float>*
  %wide.load154.ls1 = load <8 x float>, <8 x float>* %71, align 16, !alias.scope !54, !noalias !55
  %72 = getelementptr inbounds [20 x [200 x float]], [20 x [200 x float]]* %fusion.2.ls1, i64 0, i64 %fusion.1.indvar_address.dim.0.028.ls1, i64 24
  %73 = bitcast float* %72 to <8 x float>*
  %wide.load155.ls1 = load <8 x float>, <8 x float>* %73, align 16, !alias.scope !54, !noalias !55
  %74 = fmul fast <8 x float> %wide.load152.ls1, %62
  %75 = fmul fast <8 x float> %wide.load153.ls1, %63
  %76 = fmul fast <8 x float> %wide.load154.ls1, %64
  %77 = fmul fast <8 x float> %wide.load155.ls1, %65
  %78 = getelementptr inbounds [20 x [200 x float]], [20 x [200 x float]]* %fusion.1.ls1, i64 0, i64 %fusion.1.indvar_address.dim.0.028.ls1, i64 0
  %79 = bitcast float* %78 to <8 x float>*
  store <8 x float> %74, <8 x float>* %79, align 16, !alias.scope !56, !noalias !57
  %80 = getelementptr inbounds [20 x [200 x float]], [20 x [200 x float]]* %fusion.1.ls1, i64 0, i64 %fusion.1.indvar_address.dim.0.028.ls1, i64 8
  %81 = bitcast float* %80 to <8 x float>*
  store <8 x float> %75, <8 x float>* %81, align 16, !alias.scope !56, !noalias !57
  %82 = getelementptr inbounds [20 x [200 x float]], [20 x [200 x float]]* %fusion.1.ls1, i64 0, i64 %fusion.1.indvar_address.dim.0.028.ls1, i64 16
  %83 = bitcast float* %82 to <8 x float>*
  store <8 x float> %76, <8 x float>* %83, align 16, !alias.scope !56, !noalias !57
  %84 = getelementptr inbounds [20 x [200 x float]], [20 x [200 x float]]* %fusion.1.ls1, i64 0, i64 %fusion.1.indvar_address.dim.0.028.ls1, i64 24
  %85 = bitcast float* %84 to <8 x float>*
  store <8 x float> %77, <8 x float>* %85, align 16, !alias.scope !56, !noalias !57
  %86 = getelementptr inbounds [20 x [200 x float]], [20 x [200 x float]]* %fusion.4.ls1, i64 0, i64 %fusion.1.indvar_address.dim.0.028.ls1, i64 32
  %87 = bitcast float* %86 to <8 x float>*
  %wide.load148.1.ls1 = load <8 x float>, <8 x float>* %87, align 16, !alias.scope !51, !noalias !52
  %88 = getelementptr inbounds [20 x [200 x float]], [20 x [200 x float]]* %fusion.4.ls1, i64 0, i64 %fusion.1.indvar_address.dim.0.028.ls1, i64 40
  %89 = bitcast float* %88 to <8 x float>*
  %wide.load149.1.ls1 = load <8 x float>, <8 x float>* %89, align 16, !alias.scope !51, !noalias !52
  %90 = getelementptr inbounds [20 x [200 x float]], [20 x [200 x float]]* %fusion.4.ls1, i64 0, i64 %fusion.1.indvar_address.dim.0.028.ls1, i64 48
  %91 = bitcast float* %90 to <8 x float>*
  %wide.load150.1.ls1 = load <8 x float>, <8 x float>* %91, align 16, !alias.scope !51, !noalias !52
  %92 = getelementptr inbounds [20 x [200 x float]], [20 x [200 x float]]* %fusion.4.ls1, i64 0, i64 %fusion.1.indvar_address.dim.0.028.ls1, i64 56
  %93 = bitcast float* %92 to <8 x float>*
  %wide.load151.1.ls1 = load <8 x float>, <8 x float>* %93, align 16, !alias.scope !51, !noalias !52
  %94 = fmul fast <8 x float> %wide.load148.1.ls1, <float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01>
  %95 = fmul fast <8 x float> %wide.load149.1.ls1, <float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01>
  %96 = fmul fast <8 x float> %wide.load150.1.ls1, <float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01>
  %97 = fmul fast <8 x float> %wide.load151.1.ls1, <float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01>
  %98 = fadd fast <8 x float> %94, <float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01>
  %99 = fadd fast <8 x float> %95, <float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01>
  %100 = fadd fast <8 x float> %96, <float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01>
  %101 = fadd fast <8 x float> %97, <float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01>
  %102 = getelementptr inbounds [20 x [200 x float]], [20 x [200 x float]]* %fusion.2.ls1, i64 0, i64 %fusion.1.indvar_address.dim.0.028.ls1, i64 32
  %103 = bitcast float* %102 to <8 x float>*
  %wide.load152.1.ls1 = load <8 x float>, <8 x float>* %103, align 16, !alias.scope !54, !noalias !55
  %104 = getelementptr inbounds [20 x [200 x float]], [20 x [200 x float]]* %fusion.2.ls1, i64 0, i64 %fusion.1.indvar_address.dim.0.028.ls1, i64 40
  %105 = bitcast float* %104 to <8 x float>*
  %wide.load153.1.ls1 = load <8 x float>, <8 x float>* %105, align 16, !alias.scope !54, !noalias !55
  %106 = getelementptr inbounds [20 x [200 x float]], [20 x [200 x float]]* %fusion.2.ls1, i64 0, i64 %fusion.1.indvar_address.dim.0.028.ls1, i64 48
  %107 = bitcast float* %106 to <8 x float>*
  %wide.load154.1.ls1 = load <8 x float>, <8 x float>* %107, align 16, !alias.scope !54, !noalias !55
  %108 = getelementptr inbounds [20 x [200 x float]], [20 x [200 x float]]* %fusion.2.ls1, i64 0, i64 %fusion.1.indvar_address.dim.0.028.ls1, i64 56
  %109 = bitcast float* %108 to <8 x float>*
  %wide.load155.1.ls1 = load <8 x float>, <8 x float>* %109, align 16, !alias.scope !54, !noalias !55
  %110 = fmul fast <8 x float> %wide.load152.1.ls1, %98
  %111 = fmul fast <8 x float> %wide.load153.1.ls1, %99
  %112 = fmul fast <8 x float> %wide.load154.1.ls1, %100
  %113 = fmul fast <8 x float> %wide.load155.1.ls1, %101
  %114 = getelementptr inbounds [20 x [200 x float]], [20 x [200 x float]]* %fusion.1.ls1, i64 0, i64 %fusion.1.indvar_address.dim.0.028.ls1, i64 32
  %115 = bitcast float* %114 to <8 x float>*
  store <8 x float> %110, <8 x float>* %115, align 16, !alias.scope !56, !noalias !57
  %116 = getelementptr inbounds [20 x [200 x float]], [20 x [200 x float]]* %fusion.1.ls1, i64 0, i64 %fusion.1.indvar_address.dim.0.028.ls1, i64 40
  %117 = bitcast float* %116 to <8 x float>*
  store <8 x float> %111, <8 x float>* %117, align 16, !alias.scope !56, !noalias !57
  %118 = getelementptr inbounds [20 x [200 x float]], [20 x [200 x float]]* %fusion.1.ls1, i64 0, i64 %fusion.1.indvar_address.dim.0.028.ls1, i64 48
  %119 = bitcast float* %118 to <8 x float>*
  store <8 x float> %112, <8 x float>* %119, align 16, !alias.scope !56, !noalias !57
  %120 = getelementptr inbounds [20 x [200 x float]], [20 x [200 x float]]* %fusion.1.ls1, i64 0, i64 %fusion.1.indvar_address.dim.0.028.ls1, i64 56
  %121 = bitcast float* %120 to <8 x float>*
  store <8 x float> %113, <8 x float>* %121, align 16, !alias.scope !56, !noalias !57
  %122 = getelementptr inbounds [20 x [200 x float]], [20 x [200 x float]]* %fusion.4.ls1, i64 0, i64 %fusion.1.indvar_address.dim.0.028.ls1, i64 64
  %123 = bitcast float* %122 to <8 x float>*
  %wide.load148.2.ls1 = load <8 x float>, <8 x float>* %123, align 16, !alias.scope !51, !noalias !52
  %124 = getelementptr inbounds [20 x [200 x float]], [20 x [200 x float]]* %fusion.4.ls1, i64 0, i64 %fusion.1.indvar_address.dim.0.028.ls1, i64 72
  %125 = bitcast float* %124 to <8 x float>*
  %wide.load149.2.ls1 = load <8 x float>, <8 x float>* %125, align 16, !alias.scope !51, !noalias !52
  %126 = getelementptr inbounds [20 x [200 x float]], [20 x [200 x float]]* %fusion.4.ls1, i64 0, i64 %fusion.1.indvar_address.dim.0.028.ls1, i64 80
  %127 = bitcast float* %126 to <8 x float>*
  %wide.load150.2.ls1 = load <8 x float>, <8 x float>* %127, align 16, !alias.scope !51, !noalias !52
  %128 = getelementptr inbounds [20 x [200 x float]], [20 x [200 x float]]* %fusion.4.ls1, i64 0, i64 %fusion.1.indvar_address.dim.0.028.ls1, i64 88
  %129 = bitcast float* %128 to <8 x float>*
  %wide.load151.2.ls1 = load <8 x float>, <8 x float>* %129, align 16, !alias.scope !51, !noalias !52
  %130 = fmul fast <8 x float> %wide.load148.2.ls1, <float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01>
  %131 = fmul fast <8 x float> %wide.load149.2.ls1, <float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01>
  %132 = fmul fast <8 x float> %wide.load150.2.ls1, <float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01>
  %133 = fmul fast <8 x float> %wide.load151.2.ls1, <float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01>
  %134 = fadd fast <8 x float> %130, <float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01>
  %135 = fadd fast <8 x float> %131, <float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01>
  %136 = fadd fast <8 x float> %132, <float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01>
  %137 = fadd fast <8 x float> %133, <float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01>
  %138 = getelementptr inbounds [20 x [200 x float]], [20 x [200 x float]]* %fusion.2.ls1, i64 0, i64 %fusion.1.indvar_address.dim.0.028.ls1, i64 64
  %139 = bitcast float* %138 to <8 x float>*
  %wide.load152.2.ls1 = load <8 x float>, <8 x float>* %139, align 16, !alias.scope !54, !noalias !55
  %140 = getelementptr inbounds [20 x [200 x float]], [20 x [200 x float]]* %fusion.2.ls1, i64 0, i64 %fusion.1.indvar_address.dim.0.028.ls1, i64 72
  %141 = bitcast float* %140 to <8 x float>*
  %wide.load153.2.ls1 = load <8 x float>, <8 x float>* %141, align 16, !alias.scope !54, !noalias !55
  %142 = getelementptr inbounds [20 x [200 x float]], [20 x [200 x float]]* %fusion.2.ls1, i64 0, i64 %fusion.1.indvar_address.dim.0.028.ls1, i64 80
  %143 = bitcast float* %142 to <8 x float>*
  %wide.load154.2.ls1 = load <8 x float>, <8 x float>* %143, align 16, !alias.scope !54, !noalias !55
  %144 = getelementptr inbounds [20 x [200 x float]], [20 x [200 x float]]* %fusion.2.ls1, i64 0, i64 %fusion.1.indvar_address.dim.0.028.ls1, i64 88
  %145 = bitcast float* %144 to <8 x float>*
  %wide.load155.2.ls1 = load <8 x float>, <8 x float>* %145, align 16, !alias.scope !54, !noalias !55
  %146 = fmul fast <8 x float> %wide.load152.2.ls1, %134
  %147 = fmul fast <8 x float> %wide.load153.2.ls1, %135
  %148 = fmul fast <8 x float> %wide.load154.2.ls1, %136
  %149 = fmul fast <8 x float> %wide.load155.2.ls1, %137
  %150 = getelementptr inbounds [20 x [200 x float]], [20 x [200 x float]]* %fusion.1.ls1, i64 0, i64 %fusion.1.indvar_address.dim.0.028.ls1, i64 64
  %151 = bitcast float* %150 to <8 x float>*
  store <8 x float> %146, <8 x float>* %151, align 16, !alias.scope !56, !noalias !57
  %152 = getelementptr inbounds [20 x [200 x float]], [20 x [200 x float]]* %fusion.1.ls1, i64 0, i64 %fusion.1.indvar_address.dim.0.028.ls1, i64 72
  %153 = bitcast float* %152 to <8 x float>*
  store <8 x float> %147, <8 x float>* %153, align 16, !alias.scope !56, !noalias !57
  %154 = getelementptr inbounds [20 x [200 x float]], [20 x [200 x float]]* %fusion.1.ls1, i64 0, i64 %fusion.1.indvar_address.dim.0.028.ls1, i64 80
  %155 = bitcast float* %154 to <8 x float>*
  store <8 x float> %148, <8 x float>* %155, align 16, !alias.scope !56, !noalias !57
  %156 = getelementptr inbounds [20 x [200 x float]], [20 x [200 x float]]* %fusion.1.ls1, i64 0, i64 %fusion.1.indvar_address.dim.0.028.ls1, i64 88
  %157 = bitcast float* %156 to <8 x float>*
  store <8 x float> %149, <8 x float>* %157, align 16, !alias.scope !56, !noalias !57
  %158 = getelementptr inbounds [20 x [200 x float]], [20 x [200 x float]]* %fusion.4.ls1, i64 0, i64 %fusion.1.indvar_address.dim.0.028.ls1, i64 96
  %159 = bitcast float* %158 to <8 x float>*
  %wide.load148.3.ls1 = load <8 x float>, <8 x float>* %159, align 16, !alias.scope !51, !noalias !52
  %160 = getelementptr inbounds [20 x [200 x float]], [20 x [200 x float]]* %fusion.4.ls1, i64 0, i64 %fusion.1.indvar_address.dim.0.028.ls1, i64 104
  %161 = bitcast float* %160 to <8 x float>*
  %wide.load149.3.ls1 = load <8 x float>, <8 x float>* %161, align 16, !alias.scope !51, !noalias !52
  %162 = getelementptr inbounds [20 x [200 x float]], [20 x [200 x float]]* %fusion.4.ls1, i64 0, i64 %fusion.1.indvar_address.dim.0.028.ls1, i64 112
  %163 = bitcast float* %162 to <8 x float>*
  %wide.load150.3.ls1 = load <8 x float>, <8 x float>* %163, align 16, !alias.scope !51, !noalias !52
  %164 = getelementptr inbounds [20 x [200 x float]], [20 x [200 x float]]* %fusion.4.ls1, i64 0, i64 %fusion.1.indvar_address.dim.0.028.ls1, i64 120
  %165 = bitcast float* %164 to <8 x float>*
  %wide.load151.3.ls1 = load <8 x float>, <8 x float>* %165, align 16, !alias.scope !51, !noalias !52
  %166 = fmul fast <8 x float> %wide.load148.3.ls1, <float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01>
  %167 = fmul fast <8 x float> %wide.load149.3.ls1, <float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01>
  %168 = fmul fast <8 x float> %wide.load150.3.ls1, <float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01>
  %169 = fmul fast <8 x float> %wide.load151.3.ls1, <float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01>
  %170 = fadd fast <8 x float> %166, <float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01>
  %171 = fadd fast <8 x float> %167, <float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01>
  %172 = fadd fast <8 x float> %168, <float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01>
  %173 = fadd fast <8 x float> %169, <float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01>
  %174 = getelementptr inbounds [20 x [200 x float]], [20 x [200 x float]]* %fusion.2.ls1, i64 0, i64 %fusion.1.indvar_address.dim.0.028.ls1, i64 96
  %175 = bitcast float* %174 to <8 x float>*
  %wide.load152.3.ls1 = load <8 x float>, <8 x float>* %175, align 16, !alias.scope !54, !noalias !55
  %176 = getelementptr inbounds [20 x [200 x float]], [20 x [200 x float]]* %fusion.2.ls1, i64 0, i64 %fusion.1.indvar_address.dim.0.028.ls1, i64 104
  %177 = bitcast float* %176 to <8 x float>*
  %wide.load153.3.ls1 = load <8 x float>, <8 x float>* %177, align 16, !alias.scope !54, !noalias !55
  %178 = getelementptr inbounds [20 x [200 x float]], [20 x [200 x float]]* %fusion.2.ls1, i64 0, i64 %fusion.1.indvar_address.dim.0.028.ls1, i64 112
  %179 = bitcast float* %178 to <8 x float>*
  %wide.load154.3.ls1 = load <8 x float>, <8 x float>* %179, align 16, !alias.scope !54, !noalias !55
  %180 = getelementptr inbounds [20 x [200 x float]], [20 x [200 x float]]* %fusion.2.ls1, i64 0, i64 %fusion.1.indvar_address.dim.0.028.ls1, i64 120
  %181 = bitcast float* %180 to <8 x float>*
  %wide.load155.3.ls1 = load <8 x float>, <8 x float>* %181, align 16, !alias.scope !54, !noalias !55
  %182 = fmul fast <8 x float> %wide.load152.3.ls1, %170
  %183 = fmul fast <8 x float> %wide.load153.3.ls1, %171
  %184 = fmul fast <8 x float> %wide.load154.3.ls1, %172
  %185 = fmul fast <8 x float> %wide.load155.3.ls1, %173
  %186 = getelementptr inbounds [20 x [200 x float]], [20 x [200 x float]]* %fusion.1.ls1, i64 0, i64 %fusion.1.indvar_address.dim.0.028.ls1, i64 96
  %187 = bitcast float* %186 to <8 x float>*
  store <8 x float> %182, <8 x float>* %187, align 16, !alias.scope !56, !noalias !57
  %188 = getelementptr inbounds [20 x [200 x float]], [20 x [200 x float]]* %fusion.1.ls1, i64 0, i64 %fusion.1.indvar_address.dim.0.028.ls1, i64 104
  %189 = bitcast float* %188 to <8 x float>*
  store <8 x float> %183, <8 x float>* %189, align 16, !alias.scope !56, !noalias !57
  %190 = getelementptr inbounds [20 x [200 x float]], [20 x [200 x float]]* %fusion.1.ls1, i64 0, i64 %fusion.1.indvar_address.dim.0.028.ls1, i64 112
  %191 = bitcast float* %190 to <8 x float>*
  store <8 x float> %184, <8 x float>* %191, align 16, !alias.scope !56, !noalias !57
  %192 = getelementptr inbounds [20 x [200 x float]], [20 x [200 x float]]* %fusion.1.ls1, i64 0, i64 %fusion.1.indvar_address.dim.0.028.ls1, i64 120
  %193 = bitcast float* %192 to <8 x float>*
  store <8 x float> %185, <8 x float>* %193, align 16, !alias.scope !56, !noalias !57
  %194 = getelementptr inbounds [20 x [200 x float]], [20 x [200 x float]]* %fusion.4.ls1, i64 0, i64 %fusion.1.indvar_address.dim.0.028.ls1, i64 128
  %195 = bitcast float* %194 to <8 x float>*
  %wide.load148.4.ls1 = load <8 x float>, <8 x float>* %195, align 16, !alias.scope !51, !noalias !52
  %196 = getelementptr inbounds [20 x [200 x float]], [20 x [200 x float]]* %fusion.4.ls1, i64 0, i64 %fusion.1.indvar_address.dim.0.028.ls1, i64 136
  %197 = bitcast float* %196 to <8 x float>*
  %wide.load149.4.ls1 = load <8 x float>, <8 x float>* %197, align 16, !alias.scope !51, !noalias !52
  %198 = getelementptr inbounds [20 x [200 x float]], [20 x [200 x float]]* %fusion.4.ls1, i64 0, i64 %fusion.1.indvar_address.dim.0.028.ls1, i64 144
  %199 = bitcast float* %198 to <8 x float>*
  %wide.load150.4.ls1 = load <8 x float>, <8 x float>* %199, align 16, !alias.scope !51, !noalias !52
  %200 = getelementptr inbounds [20 x [200 x float]], [20 x [200 x float]]* %fusion.4.ls1, i64 0, i64 %fusion.1.indvar_address.dim.0.028.ls1, i64 152
  %201 = bitcast float* %200 to <8 x float>*
  %wide.load151.4.ls1 = load <8 x float>, <8 x float>* %201, align 16, !alias.scope !51, !noalias !52
  %202 = fmul fast <8 x float> %wide.load148.4.ls1, <float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01>
  %203 = fmul fast <8 x float> %wide.load149.4.ls1, <float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01>
  %204 = fmul fast <8 x float> %wide.load150.4.ls1, <float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01>
  %205 = fmul fast <8 x float> %wide.load151.4.ls1, <float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01>
  %206 = fadd fast <8 x float> %202, <float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01>
  %207 = fadd fast <8 x float> %203, <float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01>
  %208 = fadd fast <8 x float> %204, <float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01>
  %209 = fadd fast <8 x float> %205, <float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01>
  %210 = getelementptr inbounds [20 x [200 x float]], [20 x [200 x float]]* %fusion.2.ls1, i64 0, i64 %fusion.1.indvar_address.dim.0.028.ls1, i64 128
  %211 = bitcast float* %210 to <8 x float>*
  %wide.load152.4.ls1 = load <8 x float>, <8 x float>* %211, align 16, !alias.scope !54, !noalias !55
  %212 = getelementptr inbounds [20 x [200 x float]], [20 x [200 x float]]* %fusion.2.ls1, i64 0, i64 %fusion.1.indvar_address.dim.0.028.ls1, i64 136
  %213 = bitcast float* %212 to <8 x float>*
  %wide.load153.4.ls1 = load <8 x float>, <8 x float>* %213, align 16, !alias.scope !54, !noalias !55
  %214 = getelementptr inbounds [20 x [200 x float]], [20 x [200 x float]]* %fusion.2.ls1, i64 0, i64 %fusion.1.indvar_address.dim.0.028.ls1, i64 144
  %215 = bitcast float* %214 to <8 x float>*
  %wide.load154.4.ls1 = load <8 x float>, <8 x float>* %215, align 16, !alias.scope !54, !noalias !55
  %216 = getelementptr inbounds [20 x [200 x float]], [20 x [200 x float]]* %fusion.2.ls1, i64 0, i64 %fusion.1.indvar_address.dim.0.028.ls1, i64 152
  %217 = bitcast float* %216 to <8 x float>*
  %wide.load155.4.ls1 = load <8 x float>, <8 x float>* %217, align 16, !alias.scope !54, !noalias !55
  %218 = fmul fast <8 x float> %wide.load152.4.ls1, %206
  %219 = fmul fast <8 x float> %wide.load153.4.ls1, %207
  %220 = fmul fast <8 x float> %wide.load154.4.ls1, %208
  %221 = fmul fast <8 x float> %wide.load155.4.ls1, %209
  %222 = getelementptr inbounds [20 x [200 x float]], [20 x [200 x float]]* %fusion.1.ls1, i64 0, i64 %fusion.1.indvar_address.dim.0.028.ls1, i64 128
  %223 = bitcast float* %222 to <8 x float>*
  store <8 x float> %218, <8 x float>* %223, align 16, !alias.scope !56, !noalias !57
  %224 = getelementptr inbounds [20 x [200 x float]], [20 x [200 x float]]* %fusion.1.ls1, i64 0, i64 %fusion.1.indvar_address.dim.0.028.ls1, i64 136
  %225 = bitcast float* %224 to <8 x float>*
  store <8 x float> %219, <8 x float>* %225, align 16, !alias.scope !56, !noalias !57
  %226 = getelementptr inbounds [20 x [200 x float]], [20 x [200 x float]]* %fusion.1.ls1, i64 0, i64 %fusion.1.indvar_address.dim.0.028.ls1, i64 144
  %227 = bitcast float* %226 to <8 x float>*
  store <8 x float> %220, <8 x float>* %227, align 16, !alias.scope !56, !noalias !57
  %228 = getelementptr inbounds [20 x [200 x float]], [20 x [200 x float]]* %fusion.1.ls1, i64 0, i64 %fusion.1.indvar_address.dim.0.028.ls1, i64 152
  %229 = bitcast float* %228 to <8 x float>*
  store <8 x float> %221, <8 x float>* %229, align 16, !alias.scope !56, !noalias !57
  %230 = getelementptr inbounds [20 x [200 x float]], [20 x [200 x float]]* %fusion.4.ls1, i64 0, i64 %fusion.1.indvar_address.dim.0.028.ls1, i64 160
  %231 = bitcast float* %230 to <8 x float>*
  %wide.load148.5.ls1 = load <8 x float>, <8 x float>* %231, align 16, !alias.scope !51, !noalias !52
  %232 = getelementptr inbounds [20 x [200 x float]], [20 x [200 x float]]* %fusion.4.ls1, i64 0, i64 %fusion.1.indvar_address.dim.0.028.ls1, i64 168
  %233 = bitcast float* %232 to <8 x float>*
  %wide.load149.5.ls1 = load <8 x float>, <8 x float>* %233, align 16, !alias.scope !51, !noalias !52
  %234 = getelementptr inbounds [20 x [200 x float]], [20 x [200 x float]]* %fusion.4.ls1, i64 0, i64 %fusion.1.indvar_address.dim.0.028.ls1, i64 176
  %235 = bitcast float* %234 to <8 x float>*
  %wide.load150.5.ls1 = load <8 x float>, <8 x float>* %235, align 16, !alias.scope !51, !noalias !52
  %236 = getelementptr inbounds [20 x [200 x float]], [20 x [200 x float]]* %fusion.4.ls1, i64 0, i64 %fusion.1.indvar_address.dim.0.028.ls1, i64 184
  %237 = bitcast float* %236 to <8 x float>*
  %wide.load151.5.ls1 = load <8 x float>, <8 x float>* %237, align 16, !alias.scope !51, !noalias !52
  %238 = fmul fast <8 x float> %wide.load148.5.ls1, <float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01>
  %239 = fmul fast <8 x float> %wide.load149.5.ls1, <float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01>
  %240 = fmul fast <8 x float> %wide.load150.5.ls1, <float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01>
  %241 = fmul fast <8 x float> %wide.load151.5.ls1, <float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01>
  %242 = fadd fast <8 x float> %238, <float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01>
  %243 = fadd fast <8 x float> %239, <float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01>
  %244 = fadd fast <8 x float> %240, <float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01>
  %245 = fadd fast <8 x float> %241, <float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01>
  %246 = getelementptr inbounds [20 x [200 x float]], [20 x [200 x float]]* %fusion.2.ls1, i64 0, i64 %fusion.1.indvar_address.dim.0.028.ls1, i64 160
  %247 = bitcast float* %246 to <8 x float>*
  %wide.load152.5.ls1 = load <8 x float>, <8 x float>* %247, align 16, !alias.scope !54, !noalias !55
  %248 = getelementptr inbounds [20 x [200 x float]], [20 x [200 x float]]* %fusion.2.ls1, i64 0, i64 %fusion.1.indvar_address.dim.0.028.ls1, i64 168
  %249 = bitcast float* %248 to <8 x float>*
  %wide.load153.5.ls1 = load <8 x float>, <8 x float>* %249, align 16, !alias.scope !54, !noalias !55
  %250 = getelementptr inbounds [20 x [200 x float]], [20 x [200 x float]]* %fusion.2.ls1, i64 0, i64 %fusion.1.indvar_address.dim.0.028.ls1, i64 176
  %251 = bitcast float* %250 to <8 x float>*
  %wide.load154.5.ls1 = load <8 x float>, <8 x float>* %251, align 16, !alias.scope !54, !noalias !55
  %252 = getelementptr inbounds [20 x [200 x float]], [20 x [200 x float]]* %fusion.2.ls1, i64 0, i64 %fusion.1.indvar_address.dim.0.028.ls1, i64 184
  %253 = bitcast float* %252 to <8 x float>*
  %wide.load155.5.ls1 = load <8 x float>, <8 x float>* %253, align 16, !alias.scope !54, !noalias !55
  %254 = fmul fast <8 x float> %wide.load152.5.ls1, %242
  %255 = fmul fast <8 x float> %wide.load153.5.ls1, %243
  %256 = fmul fast <8 x float> %wide.load154.5.ls1, %244
  %257 = fmul fast <8 x float> %wide.load155.5.ls1, %245
  %258 = getelementptr inbounds [20 x [200 x float]], [20 x [200 x float]]* %fusion.1.ls1, i64 0, i64 %fusion.1.indvar_address.dim.0.028.ls1, i64 160
  %259 = bitcast float* %258 to <8 x float>*
  store <8 x float> %254, <8 x float>* %259, align 16, !alias.scope !56, !noalias !57
  %260 = getelementptr inbounds [20 x [200 x float]], [20 x [200 x float]]* %fusion.1.ls1, i64 0, i64 %fusion.1.indvar_address.dim.0.028.ls1, i64 168
  %261 = bitcast float* %260 to <8 x float>*
  store <8 x float> %255, <8 x float>* %261, align 16, !alias.scope !56, !noalias !57
  %262 = getelementptr inbounds [20 x [200 x float]], [20 x [200 x float]]* %fusion.1.ls1, i64 0, i64 %fusion.1.indvar_address.dim.0.028.ls1, i64 176
  %263 = bitcast float* %262 to <8 x float>*
  store <8 x float> %256, <8 x float>* %263, align 16, !alias.scope !56, !noalias !57
  %264 = getelementptr inbounds [20 x [200 x float]], [20 x [200 x float]]* %fusion.1.ls1, i64 0, i64 %fusion.1.indvar_address.dim.0.028.ls1, i64 184
  %265 = bitcast float* %264 to <8 x float>*
  store <8 x float> %257, <8 x float>* %265, align 16, !alias.scope !56, !noalias !57
  %266 = getelementptr inbounds [20 x [200 x float]], [20 x [200 x float]]* %fusion.4.ls1, i64 0, i64 %fusion.1.indvar_address.dim.0.028.ls1, i64 192
  %267 = load float, float* %266, align 16, !alias.scope !51, !noalias !52
  %268 = fmul fast float %267, 5.000000e-01
  %269 = fadd fast float %268, 5.000000e-01
  %270 = getelementptr inbounds [20 x [200 x float]], [20 x [200 x float]]* %fusion.2.ls1, i64 0, i64 %fusion.1.indvar_address.dim.0.028.ls1, i64 192
  %271 = load float, float* %270, align 16, !alias.scope !54, !noalias !55
  %272 = fmul fast float %269, %271
  %273 = getelementptr inbounds [20 x [200 x float]], [20 x [200 x float]]* %fusion.1.ls1, i64 0, i64 %fusion.1.indvar_address.dim.0.028.ls1, i64 192
  store float %272, float* %273, align 16, !alias.scope !56, !noalias !57
  %274 = getelementptr inbounds [20 x [200 x float]], [20 x [200 x float]]* %fusion.4.ls1, i64 0, i64 %fusion.1.indvar_address.dim.0.028.ls1, i64 193
  %275 = load float, float* %274, align 4, !alias.scope !51, !noalias !52
  %276 = fmul fast float %275, 5.000000e-01
  %277 = fadd fast float %276, 5.000000e-01
  %278 = getelementptr inbounds [20 x [200 x float]], [20 x [200 x float]]* %fusion.2.ls1, i64 0, i64 %fusion.1.indvar_address.dim.0.028.ls1, i64 193
  %279 = load float, float* %278, align 4, !alias.scope !54, !noalias !55
  %280 = fmul fast float %277, %279
  %281 = getelementptr inbounds [20 x [200 x float]], [20 x [200 x float]]* %fusion.1.ls1, i64 0, i64 %fusion.1.indvar_address.dim.0.028.ls1, i64 193
  store float %280, float* %281, align 4, !alias.scope !56, !noalias !57
  %282 = getelementptr inbounds [20 x [200 x float]], [20 x [200 x float]]* %fusion.4.ls1, i64 0, i64 %fusion.1.indvar_address.dim.0.028.ls1, i64 194
  %283 = load float, float* %282, align 8, !alias.scope !51, !noalias !52
  %284 = fmul fast float %283, 5.000000e-01
  %285 = fadd fast float %284, 5.000000e-01
  %286 = getelementptr inbounds [20 x [200 x float]], [20 x [200 x float]]* %fusion.2.ls1, i64 0, i64 %fusion.1.indvar_address.dim.0.028.ls1, i64 194
  %287 = load float, float* %286, align 8, !alias.scope !54, !noalias !55
  %288 = fmul fast float %285, %287
  %289 = getelementptr inbounds [20 x [200 x float]], [20 x [200 x float]]* %fusion.1.ls1, i64 0, i64 %fusion.1.indvar_address.dim.0.028.ls1, i64 194
  store float %288, float* %289, align 8, !alias.scope !56, !noalias !57
  %290 = getelementptr inbounds [20 x [200 x float]], [20 x [200 x float]]* %fusion.4.ls1, i64 0, i64 %fusion.1.indvar_address.dim.0.028.ls1, i64 195
  %291 = load float, float* %290, align 4, !alias.scope !51, !noalias !52
  %292 = fmul fast float %291, 5.000000e-01
  %293 = fadd fast float %292, 5.000000e-01
  %294 = getelementptr inbounds [20 x [200 x float]], [20 x [200 x float]]* %fusion.2.ls1, i64 0, i64 %fusion.1.indvar_address.dim.0.028.ls1, i64 195
  %295 = load float, float* %294, align 4, !alias.scope !54, !noalias !55
  %296 = fmul fast float %293, %295
  %297 = getelementptr inbounds [20 x [200 x float]], [20 x [200 x float]]* %fusion.1.ls1, i64 0, i64 %fusion.1.indvar_address.dim.0.028.ls1, i64 195
  store float %296, float* %297, align 4, !alias.scope !56, !noalias !57
  %298 = getelementptr inbounds [20 x [200 x float]], [20 x [200 x float]]* %fusion.4.ls1, i64 0, i64 %fusion.1.indvar_address.dim.0.028.ls1, i64 196
  %299 = load float, float* %298, align 16, !alias.scope !51, !noalias !52
  %300 = fmul fast float %299, 5.000000e-01
  %301 = fadd fast float %300, 5.000000e-01
  %302 = getelementptr inbounds [20 x [200 x float]], [20 x [200 x float]]* %fusion.2.ls1, i64 0, i64 %fusion.1.indvar_address.dim.0.028.ls1, i64 196
  %303 = load float, float* %302, align 16, !alias.scope !54, !noalias !55
  %304 = fmul fast float %301, %303
  %305 = getelementptr inbounds [20 x [200 x float]], [20 x [200 x float]]* %fusion.1.ls1, i64 0, i64 %fusion.1.indvar_address.dim.0.028.ls1, i64 196
  store float %304, float* %305, align 16, !alias.scope !56, !noalias !57
  %306 = getelementptr inbounds [20 x [200 x float]], [20 x [200 x float]]* %fusion.4.ls1, i64 0, i64 %fusion.1.indvar_address.dim.0.028.ls1, i64 197
  %307 = load float, float* %306, align 4, !alias.scope !51, !noalias !52
  %308 = fmul fast float %307, 5.000000e-01
  %309 = fadd fast float %308, 5.000000e-01
  %310 = getelementptr inbounds [20 x [200 x float]], [20 x [200 x float]]* %fusion.2.ls1, i64 0, i64 %fusion.1.indvar_address.dim.0.028.ls1, i64 197
  %311 = load float, float* %310, align 4, !alias.scope !54, !noalias !55
  %312 = fmul fast float %309, %311
  %313 = getelementptr inbounds [20 x [200 x float]], [20 x [200 x float]]* %fusion.1.ls1, i64 0, i64 %fusion.1.indvar_address.dim.0.028.ls1, i64 197
  store float %312, float* %313, align 4, !alias.scope !56, !noalias !57
  %314 = getelementptr inbounds [20 x [200 x float]], [20 x [200 x float]]* %fusion.4.ls1, i64 0, i64 %fusion.1.indvar_address.dim.0.028.ls1, i64 198
  %315 = load float, float* %314, align 8, !alias.scope !51, !noalias !52
  %316 = fmul fast float %315, 5.000000e-01
  %317 = fadd fast float %316, 5.000000e-01
  %318 = getelementptr inbounds [20 x [200 x float]], [20 x [200 x float]]* %fusion.2.ls1, i64 0, i64 %fusion.1.indvar_address.dim.0.028.ls1, i64 198
  %319 = load float, float* %318, align 8, !alias.scope !54, !noalias !55
  %320 = fmul fast float %317, %319
  %321 = getelementptr inbounds [20 x [200 x float]], [20 x [200 x float]]* %fusion.1.ls1, i64 0, i64 %fusion.1.indvar_address.dim.0.028.ls1, i64 198
  store float %320, float* %321, align 8, !alias.scope !56, !noalias !57
  %322 = getelementptr inbounds [20 x [200 x float]], [20 x [200 x float]]* %fusion.4.ls1, i64 0, i64 %fusion.1.indvar_address.dim.0.028.ls1, i64 199
  %323 = load float, float* %322, align 4, !alias.scope !51, !noalias !52
  %324 = fmul fast float %323, 5.000000e-01
  %325 = fadd fast float %324, 5.000000e-01
  %326 = getelementptr inbounds [20 x [200 x float]], [20 x [200 x float]]* %fusion.2.ls1, i64 0, i64 %fusion.1.indvar_address.dim.0.028.ls1, i64 199
  %327 = load float, float* %326, align 4, !alias.scope !54, !noalias !55
  %328 = fmul fast float %325, %327
  %329 = getelementptr inbounds [20 x [200 x float]], [20 x [200 x float]]* %fusion.1.ls1, i64 0, i64 %fusion.1.indvar_address.dim.0.028.ls1, i64 199
  store float %328, float* %329, align 4, !alias.scope !56, !noalias !57
  %indvar.inc9.ls1 = add nuw nsw i64 %fusion.1.indvar_address.dim.0.028.ls1, 1
  %exitcond49.ls1 = icmp eq i64 %indvar.inc9.ls1, %end.ls1
  br i1 %exitcond49.ls1, label %fusion.1.loop_sync.dim.0.ls1, label %fusion.1.loop_detach.dim.0.ls1, !llvm.loop !58
}

; Function Attrs: argmemonly nounwind stealable
define private fastcc void @cluster_25__XlaCompiledKernel_true__XlaNumConstantArgs_1__XlaNumResourceArgs_0_.42.outline_fusion.4.loop_detach.dim.0.ls1(i64 %fusion.4.indvar_address.dim.0.031.start.ls1, i64 %end.ls1, i64 %grainsize.ls1, <8 x float> %wide.load131.2.ls1, <8 x float> %wide.load132.5.ls1, <8 x float> %wide.load131.5.ls1, <8 x float> %wide.load130.5.ls1, <8 x float> %wide.load133.4.ls1, <8 x float> %wide.load132.4.ls1, <8 x float> %wide.load131.4.ls1, <8 x float> %wide.load130.4.ls1, <8 x float> %wide.load133.3.ls1, <8 x float> %wide.load132.3.ls1, <8 x float> %wide.load131.3.ls1, <8 x float> %wide.load130.3.ls1, <8 x float> %wide.load133.2.ls1, <8 x float> %wide.load132.2.ls1, <8 x float> %wide.load130.2.ls1, <8 x float> %wide.load133.1.ls1, <8 x float> %wide.load132.1.ls1, <8 x float> %wide.load131.1.ls1, <8 x float> %wide.load130.1.ls1, <8 x float> %wide.load133.5.ls1, <8 x float> %wide.load133.ls1, <8 x float> %wide.load132.ls1, <8 x float> %wide.load131.ls1, <8 x float> %wide.load130.ls1, i8* readnone align 16 %.ls1, [800 x float]* nocapture readonly align 16 %.ls11, i8* readnone align 16 %scevgep104105.ls1, [20 x [200 x float]]* align 16 %fusion.4.ls1, [20 x [800 x float]]* readonly align 16 %.ls12, i8* readnone align 16 %.ls13) unnamed_addr #4 {
fusion.2.loop_exit.dim.0.ls1:
  %__cilkrts_sf = alloca %struct.__cilkrts_stack_frame, align 8
  %0 = call %struct.__cilkrts_worker* @__cilkrts_get_tls_worker() #6
  %1 = icmp eq %struct.__cilkrts_worker* %0, null
  br i1 %1, label %slowpath.i, label %fastpath.i

slowpath.i:                                       ; preds = %fusion.2.loop_exit.dim.0.ls1
  %2 = call %struct.__cilkrts_worker* @__cilkrts_bind_thread_1() #6
  %3 = getelementptr inbounds %struct.__cilkrts_stack_frame, %struct.__cilkrts_stack_frame* %__cilkrts_sf, i64 0, i32 0
  store atomic i32 16777344, i32* %3 release, align 8
  br label %__cilkrts_enter_frame_1.exit

fastpath.i:                                       ; preds = %fusion.2.loop_exit.dim.0.ls1
  %4 = getelementptr inbounds %struct.__cilkrts_stack_frame, %struct.__cilkrts_stack_frame* %__cilkrts_sf, i64 0, i32 0
  store atomic i32 16777216, i32* %4 release, align 8
  br label %__cilkrts_enter_frame_1.exit

__cilkrts_enter_frame_1.exit:                     ; preds = %slowpath.i, %fastpath.i
  %5 = phi %struct.__cilkrts_worker* [ %2, %slowpath.i ], [ %0, %fastpath.i ]
  %6 = getelementptr inbounds %struct.__cilkrts_worker, %struct.__cilkrts_worker* %5, i64 0, i32 9
  %7 = load atomic %struct.__cilkrts_stack_frame*, %struct.__cilkrts_stack_frame** %6 acquire, align 8
  %8 = getelementptr inbounds %struct.__cilkrts_stack_frame, %struct.__cilkrts_stack_frame* %__cilkrts_sf, i64 0, i32 2
  store atomic %struct.__cilkrts_stack_frame* %7, %struct.__cilkrts_stack_frame** %8 release, align 8
  %9 = getelementptr inbounds %struct.__cilkrts_stack_frame, %struct.__cilkrts_stack_frame* %__cilkrts_sf, i64 0, i32 3
  store atomic %struct.__cilkrts_worker* %5, %struct.__cilkrts_worker** %9 release, align 8
  store atomic %struct.__cilkrts_stack_frame* %__cilkrts_sf, %struct.__cilkrts_stack_frame** %6 release, align 8
  %itercount4 = sub i64 %end.ls1, %fusion.4.indvar_address.dim.0.031.start.ls1
  %10 = icmp ugt i64 %itercount4, %grainsize.ls1
  br i1 %10, label %.lr.ph.preheader, label %fusion.4.loop_detach.dim.0.ls1.preheader

.lr.ph.preheader:                                 ; preds = %__cilkrts_enter_frame_1.exit
  %11 = getelementptr inbounds %struct.__cilkrts_stack_frame, %struct.__cilkrts_stack_frame* %__cilkrts_sf, i64 0, i32 6
  %12 = getelementptr inbounds %struct.__cilkrts_stack_frame, %struct.__cilkrts_stack_frame* %__cilkrts_sf, i64 0, i32 7
  %13 = getelementptr inbounds %struct.__cilkrts_stack_frame, %struct.__cilkrts_stack_frame* %__cilkrts_sf, i64 0, i32 5
  %14 = getelementptr inbounds [5 x i8*], [5 x i8*]* %13, i64 0, i64 0
  %15 = getelementptr inbounds %struct.__cilkrts_stack_frame, %struct.__cilkrts_stack_frame* %__cilkrts_sf, i64 0, i32 5, i64 2
  %16 = bitcast [5 x i8*]* %13 to i8*
  br label %.lr.ph

fusion.4.loop_detach.dim.0.ls1.preheader:         ; preds = %.split.split, %__cilkrts_enter_frame_1.exit
  %fusion.4.indvar_address.dim.0.031.ls1.dac.lcssa = phi i64 [ %fusion.4.indvar_address.dim.0.031.start.ls1, %__cilkrts_enter_frame_1.exit ], [ %miditer, %.split.split ]
  br label %fusion.4.loop_detach.dim.0.ls1

.lr.ph:                                           ; preds = %.lr.ph.preheader, %.split.split
  %itercount6 = phi i64 [ %itercount, %.split.split ], [ %itercount4, %.lr.ph.preheader ]
  %fusion.4.indvar_address.dim.0.031.ls1.dac5 = phi i64 [ %miditer, %.split.split ], [ %fusion.4.indvar_address.dim.0.031.start.ls1, %.lr.ph.preheader ]
  %halfcount = lshr i64 %itercount6, 1
  %miditer = add nuw nsw i64 %fusion.4.indvar_address.dim.0.031.ls1.dac5, %halfcount
  call void asm sideeffect "stmxcsr $0\0A\09fnstcw $1", "*m,*m,~{dirflag},~{fpsr},~{flags}"(i32* nonnull %11, i16* nonnull %12) #6
  %17 = call i8* @llvm.frameaddress(i32 0)
  store volatile i8* %17, i8** %14, align 8
  %18 = call i8* @llvm.stacksave()
  store volatile i8* %18, i8** %15, align 8
  %19 = call i32 @llvm.eh.sjlj.setjmp(i8* nonnull %16) #7
  %20 = icmp eq i32 %19, 0
  br i1 %20, label %.lr.ph.split, label %.split.split

.lr.ph.split:                                     ; preds = %.lr.ph
  call fastcc void @cluster_25__XlaCompiledKernel_true__XlaNumConstantArgs_1__XlaNumResourceArgs_0_.42.outline_fusion.4.loop_detach.dim.0.ls1.outline_.split.otd1(<8 x float> %wide.load131.3.ls1, <8 x float> %wide.load130.ls1, <8 x float> %wide.load131.ls1, <8 x float> %wide.load132.ls1, <8 x float> %wide.load133.ls1, <8 x float> %wide.load133.5.ls1, <8 x float> %wide.load130.1.ls1, <8 x float> %wide.load131.1.ls1, <8 x float> %wide.load132.1.ls1, <8 x float> %wide.load133.1.ls1, <8 x float> %wide.load130.2.ls1, <8 x float> %wide.load132.2.ls1, <8 x float> %wide.load133.2.ls1, <8 x float> %wide.load130.3.ls1, <8 x float> %wide.load132.3.ls1, <8 x float> %wide.load133.3.ls1, <8 x float> %wide.load130.4.ls1, <8 x float> %wide.load131.4.ls1, <8 x float> %wide.load132.4.ls1, <8 x float> %wide.load133.4.ls1, <8 x float> %wide.load130.5.ls1, <8 x float> %wide.load131.5.ls1, <8 x float> %wide.load132.5.ls1, <8 x float> %wide.load131.2.ls1, i8* %scevgep104105.ls1, [20 x [200 x float]]* %fusion.4.ls1, i8* %.ls13, [20 x [800 x float]]* %.ls12, i64 %fusion.4.indvar_address.dim.0.031.ls1.dac5, [800 x float]* %.ls11, i8* %.ls1, i64 %grainsize.ls1, i64 %miditer) #6
  br label %.split.split

.split.split:                                     ; preds = %.lr.ph, %.lr.ph.split
  %itercount = sub i64 %end.ls1, %miditer
  %21 = icmp ugt i64 %itercount, %grainsize.ls1
  br i1 %21, label %.lr.ph, label %fusion.4.loop_detach.dim.0.ls1.preheader

fusion.4.loop_sync.dim.0.ls1:                     ; preds = %fusion.4.loop_inc.dim.0.ls1
  %22 = getelementptr inbounds %struct.__cilkrts_stack_frame, %struct.__cilkrts_stack_frame* %__cilkrts_sf, i64 0, i32 0
  %23 = load atomic i32, i32* %22 acquire, align 8
  %24 = and i32 %23, 2
  %25 = icmp eq i32 %24, 0
  br i1 %25, label %__cilk_sync_nothrow.exit, label %cilk.sync.savestate.i

cilk.sync.savestate.i:                            ; preds = %fusion.4.loop_sync.dim.0.ls1
  %26 = load atomic %struct.__cilkrts_worker*, %struct.__cilkrts_worker** %9 acquire, align 8
  %.elt = getelementptr inbounds %struct.__cilkrts_worker, %struct.__cilkrts_worker* %26, i64 0, i32 12, i32 0
  %.unpack = load i64, i64* %.elt, align 8
  %.elt9 = getelementptr inbounds %struct.__cilkrts_worker, %struct.__cilkrts_worker* %26, i64 0, i32 12, i32 1
  %27 = bitcast %struct.__cilkrts_pedigree** %.elt9 to i64*
  %.unpack1011 = load i64, i64* %27, align 8
  %.fca.0.0.gep = getelementptr inbounds %struct.__cilkrts_stack_frame, %struct.__cilkrts_stack_frame* %__cilkrts_sf, i64 0, i32 9, i32 0, i32 0
  %.fca.0.1.gep = getelementptr inbounds %struct.__cilkrts_stack_frame, %struct.__cilkrts_stack_frame* %__cilkrts_sf, i64 0, i32 9, i32 0, i32 1
  store i64 %.unpack, i64* %.fca.0.0.gep, align 8
  %28 = bitcast %struct.__cilkrts_pedigree** %.fca.0.1.gep to i64*
  store i64 %.unpack1011, i64* %28, align 8
  %29 = getelementptr inbounds %struct.__cilkrts_stack_frame, %struct.__cilkrts_stack_frame* %__cilkrts_sf, i64 0, i32 6
  %30 = getelementptr inbounds %struct.__cilkrts_stack_frame, %struct.__cilkrts_stack_frame* %__cilkrts_sf, i64 0, i32 7
  call void asm sideeffect "stmxcsr $0\0A\09fnstcw $1", "*m,*m,~{dirflag},~{fpsr},~{flags}"(i32* nonnull %29, i16* nonnull %30) #6
  %31 = getelementptr inbounds %struct.__cilkrts_stack_frame, %struct.__cilkrts_stack_frame* %__cilkrts_sf, i64 0, i32 5
  %32 = call i8* @llvm.frameaddress(i32 0) #6
  %33 = getelementptr inbounds [5 x i8*], [5 x i8*]* %31, i64 0, i64 0
  store volatile i8* %32, i8** %33, align 8
  %34 = call i8* @llvm.stacksave() #6
  %35 = getelementptr inbounds %struct.__cilkrts_stack_frame, %struct.__cilkrts_stack_frame* %__cilkrts_sf, i64 0, i32 5, i64 2
  store volatile i8* %34, i8** %35, align 8
  %36 = bitcast [5 x i8*]* %31 to i8*
  %37 = call i32 @llvm.eh.sjlj.setjmp(i8* nonnull %36) #8
  %38 = icmp eq i32 %37, 0
  br i1 %38, label %cilk.sync.runtimecall.i, label %__cilk_sync_nothrow.exit

cilk.sync.runtimecall.i:                          ; preds = %cilk.sync.savestate.i
  call void @__cilkrts_sync(%struct.__cilkrts_stack_frame* nonnull %__cilkrts_sf) #6
  br label %__cilk_sync_nothrow.exit

__cilk_sync_nothrow.exit:                         ; preds = %fusion.4.loop_sync.dim.0.ls1, %cilk.sync.savestate.i, %cilk.sync.runtimecall.i
  %39 = load atomic %struct.__cilkrts_worker*, %struct.__cilkrts_worker** %9 acquire, align 8
  %40 = getelementptr inbounds %struct.__cilkrts_worker, %struct.__cilkrts_worker* %39, i64 0, i32 12, i32 0
  %41 = load i64, i64* %40, align 8
  %42 = add i64 %41, 1
  store i64 %42, i64* %40, align 8
  %43 = load atomic %struct.__cilkrts_worker*, %struct.__cilkrts_worker** %9 acquire, align 8
  %44 = bitcast %struct.__cilkrts_stack_frame** %8 to i64*
  %45 = load i64, i64* %44, align 8
  %46 = getelementptr inbounds %struct.__cilkrts_worker, %struct.__cilkrts_worker* %43, i64 0, i32 9
  %47 = bitcast %struct.__cilkrts_stack_frame** %46 to i64*
  store atomic i64 %45, i64* %47 release, align 8
  store atomic %struct.__cilkrts_stack_frame* null, %struct.__cilkrts_stack_frame** %8 release, align 8
  %48 = load atomic i32, i32* %22 acquire, align 8
  %49 = icmp eq i32 %48, 16777216
  br i1 %49, label %__cilk_parent_epilogue.exit, label %body.i

body.i:                                           ; preds = %__cilk_sync_nothrow.exit
  call void @__cilkrts_leave_frame(%struct.__cilkrts_stack_frame* nonnull %__cilkrts_sf) #6
  br label %__cilk_parent_epilogue.exit

__cilk_parent_epilogue.exit:                      ; preds = %__cilk_sync_nothrow.exit, %body.i
  ret void

fusion.4.loop_detach.dim.0.ls1:                   ; preds = %fusion.4.loop_detach.dim.0.ls1.preheader, %fusion.4.loop_inc.dim.0.ls1
  %fusion.4.indvar_address.dim.0.031.ls1 = phi i64 [ %indvar.inc7.ls1, %fusion.4.loop_inc.dim.0.ls1 ], [ %fusion.4.indvar_address.dim.0.031.ls1.dac.lcssa, %fusion.4.loop_detach.dim.0.ls1.preheader ]
  %50 = mul nuw nsw i64 %fusion.4.indvar_address.dim.0.031.ls1, 800
  %scevgep98.ls1 = getelementptr i8, i8* %.ls1, i64 %50
  %51 = add nuw nsw i64 %50, 800
  %scevgep99.ls1 = getelementptr i8, i8* %.ls1, i64 %51
  %scevgep100.ls1 = getelementptr [20 x [800 x float]], [20 x [800 x float]]* %.ls12, i64 0, i64 %fusion.4.indvar_address.dim.0.031.ls1, i64 0
  %scevgep100101.ls1 = bitcast float* %scevgep100.ls1 to i8*
  %scevgep102.ls1 = getelementptr [20 x [800 x float]], [20 x [800 x float]]* %.ls12, i64 0, i64 %fusion.4.indvar_address.dim.0.031.ls1, i64 200
  %scevgep102103.ls1 = bitcast float* %scevgep102.ls1 to i8*
  %bound0106.ls1 = icmp ult i8* %scevgep98.ls1, %scevgep102103.ls1
  %bound1107.ls1 = icmp ugt i8* %scevgep99.ls1, %scevgep100101.ls1
  %found.conflict108.ls1 = and i1 %bound0106.ls1, %bound1107.ls1
  %bound0109.ls1 = icmp ult i8* %scevgep98.ls1, %scevgep104105.ls1
  %bound1110.ls1 = icmp ugt i8* %scevgep99.ls1, %.ls13
  %found.conflict111.ls1 = and i1 %bound0109.ls1, %bound1110.ls1
  %conflict.rdx112.ls1 = or i1 %found.conflict108.ls1, %found.conflict111.ls1
  br i1 %conflict.rdx112.ls1, label %fusion.4.loop_body.dim.1.preheader.ls1, label %vector.body94.ls1

vector.body94.ls1:                                ; preds = %fusion.4.loop_detach.dim.0.ls1
  %52 = bitcast float* %scevgep100.ls1 to <8 x float>*
  %wide.load126.ls1 = load <8 x float>, <8 x float>* %52, align 16, !alias.scope !59, !noalias !23
  %53 = getelementptr inbounds [20 x [800 x float]], [20 x [800 x float]]* %.ls12, i64 0, i64 %fusion.4.indvar_address.dim.0.031.ls1, i64 8
  %54 = bitcast float* %53 to <8 x float>*
  %wide.load127.ls1 = load <8 x float>, <8 x float>* %54, align 16, !alias.scope !59, !noalias !23
  %55 = getelementptr inbounds [20 x [800 x float]], [20 x [800 x float]]* %.ls12, i64 0, i64 %fusion.4.indvar_address.dim.0.031.ls1, i64 16
  %56 = bitcast float* %55 to <8 x float>*
  %wide.load128.ls1 = load <8 x float>, <8 x float>* %56, align 16, !alias.scope !59, !noalias !23
  %57 = getelementptr inbounds [20 x [800 x float]], [20 x [800 x float]]* %.ls12, i64 0, i64 %fusion.4.indvar_address.dim.0.031.ls1, i64 24
  %58 = bitcast float* %57 to <8 x float>*
  %wide.load129.ls1 = load <8 x float>, <8 x float>* %58, align 16, !alias.scope !59, !noalias !23
  %59 = fadd fast <8 x float> %wide.load126.ls1, %wide.load130.ls1
  %60 = fadd fast <8 x float> %wide.load127.ls1, %wide.load131.ls1
  %61 = fadd fast <8 x float> %wide.load128.ls1, %wide.load132.ls1
  %62 = fadd fast <8 x float> %wide.load129.ls1, %wide.load133.ls1
  %63 = fmul fast <8 x float> %59, <float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01>
  %64 = fmul fast <8 x float> %60, <float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01>
  %65 = fmul fast <8 x float> %61, <float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01>
  %66 = fmul fast <8 x float> %62, <float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01>
  %67 = fcmp fast uge <8 x float> %63, <float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00>
  %68 = select <8 x i1> %67, <8 x float> %63, <8 x float> <float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00>
  %69 = fcmp fast ule <8 x float> %68, <float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00>
  %70 = select <8 x i1> %69, <8 x float> %68, <8 x float> <float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00>
  %71 = fmul fast <8 x float> %70, %70
  %72 = fmul fast <8 x float> %71, <float 0xBCB3E4B800000000, float 0xBCB3E4B800000000, float 0xBCB3E4B800000000, float 0xBCB3E4B800000000, float 0xBCB3E4B800000000, float 0xBCB3E4B800000000, float 0xBCB3E4B800000000, float 0xBCB3E4B800000000>
  %73 = fadd fast <8 x float> %72, <float 0x3D4C266FC0000000, float 0x3D4C266FC0000000, float 0x3D4C266FC0000000, float 0x3D4C266FC0000000, float 0x3D4C266FC0000000, float 0x3D4C266FC0000000, float 0x3D4C266FC0000000, float 0x3D4C266FC0000000>
  %74 = fmul fast <8 x float> %71, %73
  %75 = fadd fast <8 x float> %74, <float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000>
  %76 = fmul fast <8 x float> %71, %75
  %77 = fadd fast <8 x float> %76, <float 0x3E6B800820000000, float 0x3E6B800820000000, float 0x3E6B800820000000, float 0x3E6B800820000000, float 0x3E6B800820000000, float 0x3E6B800820000000, float 0x3E6B800820000000, float 0x3E6B800820000000>
  %78 = fmul fast <8 x float> %71, %77
  %79 = fadd fast <8 x float> %78, <float 0x3EEF286940000000, float 0x3EEF286940000000, float 0x3EEF286940000000, float 0x3EEF286940000000, float 0x3EEF286940000000, float 0x3EEF286940000000, float 0x3EEF286940000000, float 0x3EEF286940000000>
  %80 = fmul fast <8 x float> %71, %79
  %81 = fadd fast <8 x float> %80, <float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000>
  %82 = fmul fast <8 x float> %71, %81
  %83 = fadd fast <8 x float> %82, <float 0x3F740B3B80000000, float 0x3F740B3B80000000, float 0x3F740B3B80000000, float 0x3F740B3B80000000, float 0x3F740B3B80000000, float 0x3F740B3B80000000, float 0x3F740B3B80000000, float 0x3F740B3B80000000>
  %84 = fmul fast <8 x float> %70, %83
  %85 = fmul fast <8 x float> %71, <float 0x3EB41A7B00000000, float 0x3EB41A7B00000000, float 0x3EB41A7B00000000, float 0x3EB41A7B00000000, float 0x3EB41A7B00000000, float 0x3EB41A7B00000000, float 0x3EB41A7B00000000, float 0x3EB41A7B00000000>
  %86 = fadd fast <8 x float> %85, <float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000>
  %87 = fmul fast <8 x float> %71, %86
  %88 = fadd fast <8 x float> %87, <float 0x3F629540A0000000, float 0x3F629540A0000000, float 0x3F629540A0000000, float 0x3F629540A0000000, float 0x3F629540A0000000, float 0x3F629540A0000000, float 0x3F629540A0000000, float 0x3F629540A0000000>
  %89 = fmul fast <8 x float> %71, %88
  %90 = fadd fast <8 x float> %89, <float 0x3F740B3BA0000000, float 0x3F740B3BA0000000, float 0x3F740B3BA0000000, float 0x3F740B3BA0000000, float 0x3F740B3BA0000000, float 0x3F740B3BA0000000, float 0x3F740B3BA0000000, float 0x3F740B3BA0000000>
  %91 = fdiv fast <8 x float> %84, %90
  %92 = fcmp fast uge <8 x float> %64, <float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00>
  %93 = select <8 x i1> %92, <8 x float> %64, <8 x float> <float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00>
  %94 = fcmp fast ule <8 x float> %93, <float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00>
  %95 = select <8 x i1> %94, <8 x float> %93, <8 x float> <float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00>
  %96 = fmul fast <8 x float> %95, %95
  %97 = fmul fast <8 x float> %96, <float 0xBCB3E4B800000000, float 0xBCB3E4B800000000, float 0xBCB3E4B800000000, float 0xBCB3E4B800000000, float 0xBCB3E4B800000000, float 0xBCB3E4B800000000, float 0xBCB3E4B800000000, float 0xBCB3E4B800000000>
  %98 = fadd fast <8 x float> %97, <float 0x3D4C266FC0000000, float 0x3D4C266FC0000000, float 0x3D4C266FC0000000, float 0x3D4C266FC0000000, float 0x3D4C266FC0000000, float 0x3D4C266FC0000000, float 0x3D4C266FC0000000, float 0x3D4C266FC0000000>
  %99 = fmul fast <8 x float> %96, %98
  %100 = fadd fast <8 x float> %99, <float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000>
  %101 = fmul fast <8 x float> %96, %100
  %102 = fadd fast <8 x float> %101, <float 0x3E6B800820000000, float 0x3E6B800820000000, float 0x3E6B800820000000, float 0x3E6B800820000000, float 0x3E6B800820000000, float 0x3E6B800820000000, float 0x3E6B800820000000, float 0x3E6B800820000000>
  %103 = fmul fast <8 x float> %96, %102
  %104 = fadd fast <8 x float> %103, <float 0x3EEF286940000000, float 0x3EEF286940000000, float 0x3EEF286940000000, float 0x3EEF286940000000, float 0x3EEF286940000000, float 0x3EEF286940000000, float 0x3EEF286940000000, float 0x3EEF286940000000>
  %105 = fmul fast <8 x float> %96, %104
  %106 = fadd fast <8 x float> %105, <float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000>
  %107 = fmul fast <8 x float> %96, %106
  %108 = fadd fast <8 x float> %107, <float 0x3F740B3B80000000, float 0x3F740B3B80000000, float 0x3F740B3B80000000, float 0x3F740B3B80000000, float 0x3F740B3B80000000, float 0x3F740B3B80000000, float 0x3F740B3B80000000, float 0x3F740B3B80000000>
  %109 = fmul fast <8 x float> %95, %108
  %110 = fmul fast <8 x float> %96, <float 0x3EB41A7B00000000, float 0x3EB41A7B00000000, float 0x3EB41A7B00000000, float 0x3EB41A7B00000000, float 0x3EB41A7B00000000, float 0x3EB41A7B00000000, float 0x3EB41A7B00000000, float 0x3EB41A7B00000000>
  %111 = fadd fast <8 x float> %110, <float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000>
  %112 = fmul fast <8 x float> %96, %111
  %113 = fadd fast <8 x float> %112, <float 0x3F629540A0000000, float 0x3F629540A0000000, float 0x3F629540A0000000, float 0x3F629540A0000000, float 0x3F629540A0000000, float 0x3F629540A0000000, float 0x3F629540A0000000, float 0x3F629540A0000000>
  %114 = fmul fast <8 x float> %96, %113
  %115 = fadd fast <8 x float> %114, <float 0x3F740B3BA0000000, float 0x3F740B3BA0000000, float 0x3F740B3BA0000000, float 0x3F740B3BA0000000, float 0x3F740B3BA0000000, float 0x3F740B3BA0000000, float 0x3F740B3BA0000000, float 0x3F740B3BA0000000>
  %116 = fdiv fast <8 x float> %109, %115
  %117 = fcmp fast uge <8 x float> %65, <float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00>
  %118 = select <8 x i1> %117, <8 x float> %65, <8 x float> <float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00>
  %119 = fcmp fast ule <8 x float> %118, <float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00>
  %120 = select <8 x i1> %119, <8 x float> %118, <8 x float> <float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00>
  %121 = fmul fast <8 x float> %120, %120
  %122 = fmul fast <8 x float> %121, <float 0xBCB3E4B800000000, float 0xBCB3E4B800000000, float 0xBCB3E4B800000000, float 0xBCB3E4B800000000, float 0xBCB3E4B800000000, float 0xBCB3E4B800000000, float 0xBCB3E4B800000000, float 0xBCB3E4B800000000>
  %123 = fadd fast <8 x float> %122, <float 0x3D4C266FC0000000, float 0x3D4C266FC0000000, float 0x3D4C266FC0000000, float 0x3D4C266FC0000000, float 0x3D4C266FC0000000, float 0x3D4C266FC0000000, float 0x3D4C266FC0000000, float 0x3D4C266FC0000000>
  %124 = fmul fast <8 x float> %121, %123
  %125 = fadd fast <8 x float> %124, <float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000>
  %126 = fmul fast <8 x float> %121, %125
  %127 = fadd fast <8 x float> %126, <float 0x3E6B800820000000, float 0x3E6B800820000000, float 0x3E6B800820000000, float 0x3E6B800820000000, float 0x3E6B800820000000, float 0x3E6B800820000000, float 0x3E6B800820000000, float 0x3E6B800820000000>
  %128 = fmul fast <8 x float> %121, %127
  %129 = fadd fast <8 x float> %128, <float 0x3EEF286940000000, float 0x3EEF286940000000, float 0x3EEF286940000000, float 0x3EEF286940000000, float 0x3EEF286940000000, float 0x3EEF286940000000, float 0x3EEF286940000000, float 0x3EEF286940000000>
  %130 = fmul fast <8 x float> %121, %129
  %131 = fadd fast <8 x float> %130, <float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000>
  %132 = fmul fast <8 x float> %121, %131
  %133 = fadd fast <8 x float> %132, <float 0x3F740B3B80000000, float 0x3F740B3B80000000, float 0x3F740B3B80000000, float 0x3F740B3B80000000, float 0x3F740B3B80000000, float 0x3F740B3B80000000, float 0x3F740B3B80000000, float 0x3F740B3B80000000>
  %134 = fmul fast <8 x float> %120, %133
  %135 = fmul fast <8 x float> %121, <float 0x3EB41A7B00000000, float 0x3EB41A7B00000000, float 0x3EB41A7B00000000, float 0x3EB41A7B00000000, float 0x3EB41A7B00000000, float 0x3EB41A7B00000000, float 0x3EB41A7B00000000, float 0x3EB41A7B00000000>
  %136 = fadd fast <8 x float> %135, <float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000>
  %137 = fmul fast <8 x float> %121, %136
  %138 = fadd fast <8 x float> %137, <float 0x3F629540A0000000, float 0x3F629540A0000000, float 0x3F629540A0000000, float 0x3F629540A0000000, float 0x3F629540A0000000, float 0x3F629540A0000000, float 0x3F629540A0000000, float 0x3F629540A0000000>
  %139 = fmul fast <8 x float> %121, %138
  %140 = fadd fast <8 x float> %139, <float 0x3F740B3BA0000000, float 0x3F740B3BA0000000, float 0x3F740B3BA0000000, float 0x3F740B3BA0000000, float 0x3F740B3BA0000000, float 0x3F740B3BA0000000, float 0x3F740B3BA0000000, float 0x3F740B3BA0000000>
  %141 = fdiv fast <8 x float> %134, %140
  %142 = fcmp fast uge <8 x float> %66, <float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00>
  %143 = select <8 x i1> %142, <8 x float> %66, <8 x float> <float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00>
  %144 = fcmp fast ule <8 x float> %143, <float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00>
  %145 = select <8 x i1> %144, <8 x float> %143, <8 x float> <float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00>
  %146 = fmul fast <8 x float> %145, %145
  %147 = fmul fast <8 x float> %146, <float 0xBCB3E4B800000000, float 0xBCB3E4B800000000, float 0xBCB3E4B800000000, float 0xBCB3E4B800000000, float 0xBCB3E4B800000000, float 0xBCB3E4B800000000, float 0xBCB3E4B800000000, float 0xBCB3E4B800000000>
  %148 = fadd fast <8 x float> %147, <float 0x3D4C266FC0000000, float 0x3D4C266FC0000000, float 0x3D4C266FC0000000, float 0x3D4C266FC0000000, float 0x3D4C266FC0000000, float 0x3D4C266FC0000000, float 0x3D4C266FC0000000, float 0x3D4C266FC0000000>
  %149 = fmul fast <8 x float> %146, %148
  %150 = fadd fast <8 x float> %149, <float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000>
  %151 = fmul fast <8 x float> %146, %150
  %152 = fadd fast <8 x float> %151, <float 0x3E6B800820000000, float 0x3E6B800820000000, float 0x3E6B800820000000, float 0x3E6B800820000000, float 0x3E6B800820000000, float 0x3E6B800820000000, float 0x3E6B800820000000, float 0x3E6B800820000000>
  %153 = fmul fast <8 x float> %146, %152
  %154 = fadd fast <8 x float> %153, <float 0x3EEF286940000000, float 0x3EEF286940000000, float 0x3EEF286940000000, float 0x3EEF286940000000, float 0x3EEF286940000000, float 0x3EEF286940000000, float 0x3EEF286940000000, float 0x3EEF286940000000>
  %155 = fmul fast <8 x float> %146, %154
  %156 = fadd fast <8 x float> %155, <float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000>
  %157 = fmul fast <8 x float> %146, %156
  %158 = fadd fast <8 x float> %157, <float 0x3F740B3B80000000, float 0x3F740B3B80000000, float 0x3F740B3B80000000, float 0x3F740B3B80000000, float 0x3F740B3B80000000, float 0x3F740B3B80000000, float 0x3F740B3B80000000, float 0x3F740B3B80000000>
  %159 = fmul fast <8 x float> %145, %158
  %160 = fmul fast <8 x float> %146, <float 0x3EB41A7B00000000, float 0x3EB41A7B00000000, float 0x3EB41A7B00000000, float 0x3EB41A7B00000000, float 0x3EB41A7B00000000, float 0x3EB41A7B00000000, float 0x3EB41A7B00000000, float 0x3EB41A7B00000000>
  %161 = fadd fast <8 x float> %160, <float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000>
  %162 = fmul fast <8 x float> %146, %161
  %163 = fadd fast <8 x float> %162, <float 0x3F629540A0000000, float 0x3F629540A0000000, float 0x3F629540A0000000, float 0x3F629540A0000000, float 0x3F629540A0000000, float 0x3F629540A0000000, float 0x3F629540A0000000, float 0x3F629540A0000000>
  %164 = fmul fast <8 x float> %146, %163
  %165 = fadd fast <8 x float> %164, <float 0x3F740B3BA0000000, float 0x3F740B3BA0000000, float 0x3F740B3BA0000000, float 0x3F740B3BA0000000, float 0x3F740B3BA0000000, float 0x3F740B3BA0000000, float 0x3F740B3BA0000000, float 0x3F740B3BA0000000>
  %166 = fdiv fast <8 x float> %159, %165
  %167 = getelementptr inbounds [20 x [200 x float]], [20 x [200 x float]]* %fusion.4.ls1, i64 0, i64 %fusion.4.indvar_address.dim.0.031.ls1, i64 0
  %168 = bitcast float* %167 to <8 x float>*
  store <8 x float> %91, <8 x float>* %168, align 16, !alias.scope !62, !noalias !64
  %169 = getelementptr inbounds [20 x [200 x float]], [20 x [200 x float]]* %fusion.4.ls1, i64 0, i64 %fusion.4.indvar_address.dim.0.031.ls1, i64 8
  %170 = bitcast float* %169 to <8 x float>*
  store <8 x float> %116, <8 x float>* %170, align 16, !alias.scope !62, !noalias !64
  %171 = getelementptr inbounds [20 x [200 x float]], [20 x [200 x float]]* %fusion.4.ls1, i64 0, i64 %fusion.4.indvar_address.dim.0.031.ls1, i64 16
  %172 = bitcast float* %171 to <8 x float>*
  store <8 x float> %141, <8 x float>* %172, align 16, !alias.scope !62, !noalias !64
  %173 = getelementptr inbounds [20 x [200 x float]], [20 x [200 x float]]* %fusion.4.ls1, i64 0, i64 %fusion.4.indvar_address.dim.0.031.ls1, i64 24
  %174 = bitcast float* %173 to <8 x float>*
  store <8 x float> %166, <8 x float>* %174, align 16, !alias.scope !62, !noalias !64
  %175 = getelementptr inbounds [20 x [800 x float]], [20 x [800 x float]]* %.ls12, i64 0, i64 %fusion.4.indvar_address.dim.0.031.ls1, i64 32
  %176 = bitcast float* %175 to <8 x float>*
  %wide.load126.1.ls1 = load <8 x float>, <8 x float>* %176, align 16, !alias.scope !59, !noalias !23
  %177 = getelementptr inbounds [20 x [800 x float]], [20 x [800 x float]]* %.ls12, i64 0, i64 %fusion.4.indvar_address.dim.0.031.ls1, i64 40
  %178 = bitcast float* %177 to <8 x float>*
  %wide.load127.1.ls1 = load <8 x float>, <8 x float>* %178, align 16, !alias.scope !59, !noalias !23
  %179 = getelementptr inbounds [20 x [800 x float]], [20 x [800 x float]]* %.ls12, i64 0, i64 %fusion.4.indvar_address.dim.0.031.ls1, i64 48
  %180 = bitcast float* %179 to <8 x float>*
  %wide.load128.1.ls1 = load <8 x float>, <8 x float>* %180, align 16, !alias.scope !59, !noalias !23
  %181 = getelementptr inbounds [20 x [800 x float]], [20 x [800 x float]]* %.ls12, i64 0, i64 %fusion.4.indvar_address.dim.0.031.ls1, i64 56
  %182 = bitcast float* %181 to <8 x float>*
  %wide.load129.1.ls1 = load <8 x float>, <8 x float>* %182, align 16, !alias.scope !59, !noalias !23
  %183 = fadd fast <8 x float> %wide.load126.1.ls1, %wide.load130.1.ls1
  %184 = fadd fast <8 x float> %wide.load127.1.ls1, %wide.load131.1.ls1
  %185 = fadd fast <8 x float> %wide.load128.1.ls1, %wide.load132.1.ls1
  %186 = fadd fast <8 x float> %wide.load129.1.ls1, %wide.load133.1.ls1
  %187 = fmul fast <8 x float> %183, <float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01>
  %188 = fmul fast <8 x float> %184, <float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01>
  %189 = fmul fast <8 x float> %185, <float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01>
  %190 = fmul fast <8 x float> %186, <float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01>
  %191 = fcmp fast uge <8 x float> %187, <float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00>
  %192 = select <8 x i1> %191, <8 x float> %187, <8 x float> <float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00>
  %193 = fcmp fast ule <8 x float> %192, <float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00>
  %194 = select <8 x i1> %193, <8 x float> %192, <8 x float> <float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00>
  %195 = fmul fast <8 x float> %194, %194
  %196 = fmul fast <8 x float> %195, <float 0xBCB3E4B800000000, float 0xBCB3E4B800000000, float 0xBCB3E4B800000000, float 0xBCB3E4B800000000, float 0xBCB3E4B800000000, float 0xBCB3E4B800000000, float 0xBCB3E4B800000000, float 0xBCB3E4B800000000>
  %197 = fadd fast <8 x float> %196, <float 0x3D4C266FC0000000, float 0x3D4C266FC0000000, float 0x3D4C266FC0000000, float 0x3D4C266FC0000000, float 0x3D4C266FC0000000, float 0x3D4C266FC0000000, float 0x3D4C266FC0000000, float 0x3D4C266FC0000000>
  %198 = fmul fast <8 x float> %195, %197
  %199 = fadd fast <8 x float> %198, <float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000>
  %200 = fmul fast <8 x float> %195, %199
  %201 = fadd fast <8 x float> %200, <float 0x3E6B800820000000, float 0x3E6B800820000000, float 0x3E6B800820000000, float 0x3E6B800820000000, float 0x3E6B800820000000, float 0x3E6B800820000000, float 0x3E6B800820000000, float 0x3E6B800820000000>
  %202 = fmul fast <8 x float> %195, %201
  %203 = fadd fast <8 x float> %202, <float 0x3EEF286940000000, float 0x3EEF286940000000, float 0x3EEF286940000000, float 0x3EEF286940000000, float 0x3EEF286940000000, float 0x3EEF286940000000, float 0x3EEF286940000000, float 0x3EEF286940000000>
  %204 = fmul fast <8 x float> %195, %203
  %205 = fadd fast <8 x float> %204, <float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000>
  %206 = fmul fast <8 x float> %195, %205
  %207 = fadd fast <8 x float> %206, <float 0x3F740B3B80000000, float 0x3F740B3B80000000, float 0x3F740B3B80000000, float 0x3F740B3B80000000, float 0x3F740B3B80000000, float 0x3F740B3B80000000, float 0x3F740B3B80000000, float 0x3F740B3B80000000>
  %208 = fmul fast <8 x float> %194, %207
  %209 = fmul fast <8 x float> %195, <float 0x3EB41A7B00000000, float 0x3EB41A7B00000000, float 0x3EB41A7B00000000, float 0x3EB41A7B00000000, float 0x3EB41A7B00000000, float 0x3EB41A7B00000000, float 0x3EB41A7B00000000, float 0x3EB41A7B00000000>
  %210 = fadd fast <8 x float> %209, <float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000>
  %211 = fmul fast <8 x float> %195, %210
  %212 = fadd fast <8 x float> %211, <float 0x3F629540A0000000, float 0x3F629540A0000000, float 0x3F629540A0000000, float 0x3F629540A0000000, float 0x3F629540A0000000, float 0x3F629540A0000000, float 0x3F629540A0000000, float 0x3F629540A0000000>
  %213 = fmul fast <8 x float> %195, %212
  %214 = fadd fast <8 x float> %213, <float 0x3F740B3BA0000000, float 0x3F740B3BA0000000, float 0x3F740B3BA0000000, float 0x3F740B3BA0000000, float 0x3F740B3BA0000000, float 0x3F740B3BA0000000, float 0x3F740B3BA0000000, float 0x3F740B3BA0000000>
  %215 = fdiv fast <8 x float> %208, %214
  %216 = fcmp fast uge <8 x float> %188, <float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00>
  %217 = select <8 x i1> %216, <8 x float> %188, <8 x float> <float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00>
  %218 = fcmp fast ule <8 x float> %217, <float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00>
  %219 = select <8 x i1> %218, <8 x float> %217, <8 x float> <float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00>
  %220 = fmul fast <8 x float> %219, %219
  %221 = fmul fast <8 x float> %220, <float 0xBCB3E4B800000000, float 0xBCB3E4B800000000, float 0xBCB3E4B800000000, float 0xBCB3E4B800000000, float 0xBCB3E4B800000000, float 0xBCB3E4B800000000, float 0xBCB3E4B800000000, float 0xBCB3E4B800000000>
  %222 = fadd fast <8 x float> %221, <float 0x3D4C266FC0000000, float 0x3D4C266FC0000000, float 0x3D4C266FC0000000, float 0x3D4C266FC0000000, float 0x3D4C266FC0000000, float 0x3D4C266FC0000000, float 0x3D4C266FC0000000, float 0x3D4C266FC0000000>
  %223 = fmul fast <8 x float> %220, %222
  %224 = fadd fast <8 x float> %223, <float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000>
  %225 = fmul fast <8 x float> %220, %224
  %226 = fadd fast <8 x float> %225, <float 0x3E6B800820000000, float 0x3E6B800820000000, float 0x3E6B800820000000, float 0x3E6B800820000000, float 0x3E6B800820000000, float 0x3E6B800820000000, float 0x3E6B800820000000, float 0x3E6B800820000000>
  %227 = fmul fast <8 x float> %220, %226
  %228 = fadd fast <8 x float> %227, <float 0x3EEF286940000000, float 0x3EEF286940000000, float 0x3EEF286940000000, float 0x3EEF286940000000, float 0x3EEF286940000000, float 0x3EEF286940000000, float 0x3EEF286940000000, float 0x3EEF286940000000>
  %229 = fmul fast <8 x float> %220, %228
  %230 = fadd fast <8 x float> %229, <float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000>
  %231 = fmul fast <8 x float> %220, %230
  %232 = fadd fast <8 x float> %231, <float 0x3F740B3B80000000, float 0x3F740B3B80000000, float 0x3F740B3B80000000, float 0x3F740B3B80000000, float 0x3F740B3B80000000, float 0x3F740B3B80000000, float 0x3F740B3B80000000, float 0x3F740B3B80000000>
  %233 = fmul fast <8 x float> %219, %232
  %234 = fmul fast <8 x float> %220, <float 0x3EB41A7B00000000, float 0x3EB41A7B00000000, float 0x3EB41A7B00000000, float 0x3EB41A7B00000000, float 0x3EB41A7B00000000, float 0x3EB41A7B00000000, float 0x3EB41A7B00000000, float 0x3EB41A7B00000000>
  %235 = fadd fast <8 x float> %234, <float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000>
  %236 = fmul fast <8 x float> %220, %235
  %237 = fadd fast <8 x float> %236, <float 0x3F629540A0000000, float 0x3F629540A0000000, float 0x3F629540A0000000, float 0x3F629540A0000000, float 0x3F629540A0000000, float 0x3F629540A0000000, float 0x3F629540A0000000, float 0x3F629540A0000000>
  %238 = fmul fast <8 x float> %220, %237
  %239 = fadd fast <8 x float> %238, <float 0x3F740B3BA0000000, float 0x3F740B3BA0000000, float 0x3F740B3BA0000000, float 0x3F740B3BA0000000, float 0x3F740B3BA0000000, float 0x3F740B3BA0000000, float 0x3F740B3BA0000000, float 0x3F740B3BA0000000>
  %240 = fdiv fast <8 x float> %233, %239
  %241 = fcmp fast uge <8 x float> %189, <float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00>
  %242 = select <8 x i1> %241, <8 x float> %189, <8 x float> <float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00>
  %243 = fcmp fast ule <8 x float> %242, <float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00>
  %244 = select <8 x i1> %243, <8 x float> %242, <8 x float> <float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00>
  %245 = fmul fast <8 x float> %244, %244
  %246 = fmul fast <8 x float> %245, <float 0xBCB3E4B800000000, float 0xBCB3E4B800000000, float 0xBCB3E4B800000000, float 0xBCB3E4B800000000, float 0xBCB3E4B800000000, float 0xBCB3E4B800000000, float 0xBCB3E4B800000000, float 0xBCB3E4B800000000>
  %247 = fadd fast <8 x float> %246, <float 0x3D4C266FC0000000, float 0x3D4C266FC0000000, float 0x3D4C266FC0000000, float 0x3D4C266FC0000000, float 0x3D4C266FC0000000, float 0x3D4C266FC0000000, float 0x3D4C266FC0000000, float 0x3D4C266FC0000000>
  %248 = fmul fast <8 x float> %245, %247
  %249 = fadd fast <8 x float> %248, <float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000>
  %250 = fmul fast <8 x float> %245, %249
  %251 = fadd fast <8 x float> %250, <float 0x3E6B800820000000, float 0x3E6B800820000000, float 0x3E6B800820000000, float 0x3E6B800820000000, float 0x3E6B800820000000, float 0x3E6B800820000000, float 0x3E6B800820000000, float 0x3E6B800820000000>
  %252 = fmul fast <8 x float> %245, %251
  %253 = fadd fast <8 x float> %252, <float 0x3EEF286940000000, float 0x3EEF286940000000, float 0x3EEF286940000000, float 0x3EEF286940000000, float 0x3EEF286940000000, float 0x3EEF286940000000, float 0x3EEF286940000000, float 0x3EEF286940000000>
  %254 = fmul fast <8 x float> %245, %253
  %255 = fadd fast <8 x float> %254, <float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000>
  %256 = fmul fast <8 x float> %245, %255
  %257 = fadd fast <8 x float> %256, <float 0x3F740B3B80000000, float 0x3F740B3B80000000, float 0x3F740B3B80000000, float 0x3F740B3B80000000, float 0x3F740B3B80000000, float 0x3F740B3B80000000, float 0x3F740B3B80000000, float 0x3F740B3B80000000>
  %258 = fmul fast <8 x float> %244, %257
  %259 = fmul fast <8 x float> %245, <float 0x3EB41A7B00000000, float 0x3EB41A7B00000000, float 0x3EB41A7B00000000, float 0x3EB41A7B00000000, float 0x3EB41A7B00000000, float 0x3EB41A7B00000000, float 0x3EB41A7B00000000, float 0x3EB41A7B00000000>
  %260 = fadd fast <8 x float> %259, <float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000>
  %261 = fmul fast <8 x float> %245, %260
  %262 = fadd fast <8 x float> %261, <float 0x3F629540A0000000, float 0x3F629540A0000000, float 0x3F629540A0000000, float 0x3F629540A0000000, float 0x3F629540A0000000, float 0x3F629540A0000000, float 0x3F629540A0000000, float 0x3F629540A0000000>
  %263 = fmul fast <8 x float> %245, %262
  %264 = fadd fast <8 x float> %263, <float 0x3F740B3BA0000000, float 0x3F740B3BA0000000, float 0x3F740B3BA0000000, float 0x3F740B3BA0000000, float 0x3F740B3BA0000000, float 0x3F740B3BA0000000, float 0x3F740B3BA0000000, float 0x3F740B3BA0000000>
  %265 = fdiv fast <8 x float> %258, %264
  %266 = fcmp fast uge <8 x float> %190, <float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00>
  %267 = select <8 x i1> %266, <8 x float> %190, <8 x float> <float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00>
  %268 = fcmp fast ule <8 x float> %267, <float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00>
  %269 = select <8 x i1> %268, <8 x float> %267, <8 x float> <float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00>
  %270 = fmul fast <8 x float> %269, %269
  %271 = fmul fast <8 x float> %270, <float 0xBCB3E4B800000000, float 0xBCB3E4B800000000, float 0xBCB3E4B800000000, float 0xBCB3E4B800000000, float 0xBCB3E4B800000000, float 0xBCB3E4B800000000, float 0xBCB3E4B800000000, float 0xBCB3E4B800000000>
  %272 = fadd fast <8 x float> %271, <float 0x3D4C266FC0000000, float 0x3D4C266FC0000000, float 0x3D4C266FC0000000, float 0x3D4C266FC0000000, float 0x3D4C266FC0000000, float 0x3D4C266FC0000000, float 0x3D4C266FC0000000, float 0x3D4C266FC0000000>
  %273 = fmul fast <8 x float> %270, %272
  %274 = fadd fast <8 x float> %273, <float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000>
  %275 = fmul fast <8 x float> %270, %274
  %276 = fadd fast <8 x float> %275, <float 0x3E6B800820000000, float 0x3E6B800820000000, float 0x3E6B800820000000, float 0x3E6B800820000000, float 0x3E6B800820000000, float 0x3E6B800820000000, float 0x3E6B800820000000, float 0x3E6B800820000000>
  %277 = fmul fast <8 x float> %270, %276
  %278 = fadd fast <8 x float> %277, <float 0x3EEF286940000000, float 0x3EEF286940000000, float 0x3EEF286940000000, float 0x3EEF286940000000, float 0x3EEF286940000000, float 0x3EEF286940000000, float 0x3EEF286940000000, float 0x3EEF286940000000>
  %279 = fmul fast <8 x float> %270, %278
  %280 = fadd fast <8 x float> %279, <float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000>
  %281 = fmul fast <8 x float> %270, %280
  %282 = fadd fast <8 x float> %281, <float 0x3F740B3B80000000, float 0x3F740B3B80000000, float 0x3F740B3B80000000, float 0x3F740B3B80000000, float 0x3F740B3B80000000, float 0x3F740B3B80000000, float 0x3F740B3B80000000, float 0x3F740B3B80000000>
  %283 = fmul fast <8 x float> %269, %282
  %284 = fmul fast <8 x float> %270, <float 0x3EB41A7B00000000, float 0x3EB41A7B00000000, float 0x3EB41A7B00000000, float 0x3EB41A7B00000000, float 0x3EB41A7B00000000, float 0x3EB41A7B00000000, float 0x3EB41A7B00000000, float 0x3EB41A7B00000000>
  %285 = fadd fast <8 x float> %284, <float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000>
  %286 = fmul fast <8 x float> %270, %285
  %287 = fadd fast <8 x float> %286, <float 0x3F629540A0000000, float 0x3F629540A0000000, float 0x3F629540A0000000, float 0x3F629540A0000000, float 0x3F629540A0000000, float 0x3F629540A0000000, float 0x3F629540A0000000, float 0x3F629540A0000000>
  %288 = fmul fast <8 x float> %270, %287
  %289 = fadd fast <8 x float> %288, <float 0x3F740B3BA0000000, float 0x3F740B3BA0000000, float 0x3F740B3BA0000000, float 0x3F740B3BA0000000, float 0x3F740B3BA0000000, float 0x3F740B3BA0000000, float 0x3F740B3BA0000000, float 0x3F740B3BA0000000>
  %290 = fdiv fast <8 x float> %283, %289
  %291 = getelementptr inbounds [20 x [200 x float]], [20 x [200 x float]]* %fusion.4.ls1, i64 0, i64 %fusion.4.indvar_address.dim.0.031.ls1, i64 32
  %292 = bitcast float* %291 to <8 x float>*
  store <8 x float> %215, <8 x float>* %292, align 16, !alias.scope !62, !noalias !64
  %293 = getelementptr inbounds [20 x [200 x float]], [20 x [200 x float]]* %fusion.4.ls1, i64 0, i64 %fusion.4.indvar_address.dim.0.031.ls1, i64 40
  %294 = bitcast float* %293 to <8 x float>*
  store <8 x float> %240, <8 x float>* %294, align 16, !alias.scope !62, !noalias !64
  %295 = getelementptr inbounds [20 x [200 x float]], [20 x [200 x float]]* %fusion.4.ls1, i64 0, i64 %fusion.4.indvar_address.dim.0.031.ls1, i64 48
  %296 = bitcast float* %295 to <8 x float>*
  store <8 x float> %265, <8 x float>* %296, align 16, !alias.scope !62, !noalias !64
  %297 = getelementptr inbounds [20 x [200 x float]], [20 x [200 x float]]* %fusion.4.ls1, i64 0, i64 %fusion.4.indvar_address.dim.0.031.ls1, i64 56
  %298 = bitcast float* %297 to <8 x float>*
  store <8 x float> %290, <8 x float>* %298, align 16, !alias.scope !62, !noalias !64
  %299 = getelementptr inbounds [20 x [800 x float]], [20 x [800 x float]]* %.ls12, i64 0, i64 %fusion.4.indvar_address.dim.0.031.ls1, i64 64
  %300 = bitcast float* %299 to <8 x float>*
  %wide.load126.2.ls1 = load <8 x float>, <8 x float>* %300, align 16, !alias.scope !59, !noalias !23
  %301 = getelementptr inbounds [20 x [800 x float]], [20 x [800 x float]]* %.ls12, i64 0, i64 %fusion.4.indvar_address.dim.0.031.ls1, i64 72
  %302 = bitcast float* %301 to <8 x float>*
  %wide.load127.2.ls1 = load <8 x float>, <8 x float>* %302, align 16, !alias.scope !59, !noalias !23
  %303 = getelementptr inbounds [20 x [800 x float]], [20 x [800 x float]]* %.ls12, i64 0, i64 %fusion.4.indvar_address.dim.0.031.ls1, i64 80
  %304 = bitcast float* %303 to <8 x float>*
  %wide.load128.2.ls1 = load <8 x float>, <8 x float>* %304, align 16, !alias.scope !59, !noalias !23
  %305 = getelementptr inbounds [20 x [800 x float]], [20 x [800 x float]]* %.ls12, i64 0, i64 %fusion.4.indvar_address.dim.0.031.ls1, i64 88
  %306 = bitcast float* %305 to <8 x float>*
  %wide.load129.2.ls1 = load <8 x float>, <8 x float>* %306, align 16, !alias.scope !59, !noalias !23
  %307 = fadd fast <8 x float> %wide.load126.2.ls1, %wide.load130.2.ls1
  %308 = fadd fast <8 x float> %wide.load127.2.ls1, %wide.load131.2.ls1
  %309 = fadd fast <8 x float> %wide.load128.2.ls1, %wide.load132.2.ls1
  %310 = fadd fast <8 x float> %wide.load129.2.ls1, %wide.load133.2.ls1
  %311 = fmul fast <8 x float> %307, <float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01>
  %312 = fmul fast <8 x float> %308, <float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01>
  %313 = fmul fast <8 x float> %309, <float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01>
  %314 = fmul fast <8 x float> %310, <float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01>
  %315 = fcmp fast uge <8 x float> %311, <float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00>
  %316 = select <8 x i1> %315, <8 x float> %311, <8 x float> <float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00>
  %317 = fcmp fast ule <8 x float> %316, <float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00>
  %318 = select <8 x i1> %317, <8 x float> %316, <8 x float> <float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00>
  %319 = fmul fast <8 x float> %318, %318
  %320 = fmul fast <8 x float> %319, <float 0xBCB3E4B800000000, float 0xBCB3E4B800000000, float 0xBCB3E4B800000000, float 0xBCB3E4B800000000, float 0xBCB3E4B800000000, float 0xBCB3E4B800000000, float 0xBCB3E4B800000000, float 0xBCB3E4B800000000>
  %321 = fadd fast <8 x float> %320, <float 0x3D4C266FC0000000, float 0x3D4C266FC0000000, float 0x3D4C266FC0000000, float 0x3D4C266FC0000000, float 0x3D4C266FC0000000, float 0x3D4C266FC0000000, float 0x3D4C266FC0000000, float 0x3D4C266FC0000000>
  %322 = fmul fast <8 x float> %319, %321
  %323 = fadd fast <8 x float> %322, <float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000>
  %324 = fmul fast <8 x float> %319, %323
  %325 = fadd fast <8 x float> %324, <float 0x3E6B800820000000, float 0x3E6B800820000000, float 0x3E6B800820000000, float 0x3E6B800820000000, float 0x3E6B800820000000, float 0x3E6B800820000000, float 0x3E6B800820000000, float 0x3E6B800820000000>
  %326 = fmul fast <8 x float> %319, %325
  %327 = fadd fast <8 x float> %326, <float 0x3EEF286940000000, float 0x3EEF286940000000, float 0x3EEF286940000000, float 0x3EEF286940000000, float 0x3EEF286940000000, float 0x3EEF286940000000, float 0x3EEF286940000000, float 0x3EEF286940000000>
  %328 = fmul fast <8 x float> %319, %327
  %329 = fadd fast <8 x float> %328, <float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000>
  %330 = fmul fast <8 x float> %319, %329
  %331 = fadd fast <8 x float> %330, <float 0x3F740B3B80000000, float 0x3F740B3B80000000, float 0x3F740B3B80000000, float 0x3F740B3B80000000, float 0x3F740B3B80000000, float 0x3F740B3B80000000, float 0x3F740B3B80000000, float 0x3F740B3B80000000>
  %332 = fmul fast <8 x float> %318, %331
  %333 = fmul fast <8 x float> %319, <float 0x3EB41A7B00000000, float 0x3EB41A7B00000000, float 0x3EB41A7B00000000, float 0x3EB41A7B00000000, float 0x3EB41A7B00000000, float 0x3EB41A7B00000000, float 0x3EB41A7B00000000, float 0x3EB41A7B00000000>
  %334 = fadd fast <8 x float> %333, <float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000>
  %335 = fmul fast <8 x float> %319, %334
  %336 = fadd fast <8 x float> %335, <float 0x3F629540A0000000, float 0x3F629540A0000000, float 0x3F629540A0000000, float 0x3F629540A0000000, float 0x3F629540A0000000, float 0x3F629540A0000000, float 0x3F629540A0000000, float 0x3F629540A0000000>
  %337 = fmul fast <8 x float> %319, %336
  %338 = fadd fast <8 x float> %337, <float 0x3F740B3BA0000000, float 0x3F740B3BA0000000, float 0x3F740B3BA0000000, float 0x3F740B3BA0000000, float 0x3F740B3BA0000000, float 0x3F740B3BA0000000, float 0x3F740B3BA0000000, float 0x3F740B3BA0000000>
  %339 = fdiv fast <8 x float> %332, %338
  %340 = fcmp fast uge <8 x float> %312, <float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00>
  %341 = select <8 x i1> %340, <8 x float> %312, <8 x float> <float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00>
  %342 = fcmp fast ule <8 x float> %341, <float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00>
  %343 = select <8 x i1> %342, <8 x float> %341, <8 x float> <float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00>
  %344 = fmul fast <8 x float> %343, %343
  %345 = fmul fast <8 x float> %344, <float 0xBCB3E4B800000000, float 0xBCB3E4B800000000, float 0xBCB3E4B800000000, float 0xBCB3E4B800000000, float 0xBCB3E4B800000000, float 0xBCB3E4B800000000, float 0xBCB3E4B800000000, float 0xBCB3E4B800000000>
  %346 = fadd fast <8 x float> %345, <float 0x3D4C266FC0000000, float 0x3D4C266FC0000000, float 0x3D4C266FC0000000, float 0x3D4C266FC0000000, float 0x3D4C266FC0000000, float 0x3D4C266FC0000000, float 0x3D4C266FC0000000, float 0x3D4C266FC0000000>
  %347 = fmul fast <8 x float> %344, %346
  %348 = fadd fast <8 x float> %347, <float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000>
  %349 = fmul fast <8 x float> %344, %348
  %350 = fadd fast <8 x float> %349, <float 0x3E6B800820000000, float 0x3E6B800820000000, float 0x3E6B800820000000, float 0x3E6B800820000000, float 0x3E6B800820000000, float 0x3E6B800820000000, float 0x3E6B800820000000, float 0x3E6B800820000000>
  %351 = fmul fast <8 x float> %344, %350
  %352 = fadd fast <8 x float> %351, <float 0x3EEF286940000000, float 0x3EEF286940000000, float 0x3EEF286940000000, float 0x3EEF286940000000, float 0x3EEF286940000000, float 0x3EEF286940000000, float 0x3EEF286940000000, float 0x3EEF286940000000>
  %353 = fmul fast <8 x float> %344, %352
  %354 = fadd fast <8 x float> %353, <float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000>
  %355 = fmul fast <8 x float> %344, %354
  %356 = fadd fast <8 x float> %355, <float 0x3F740B3B80000000, float 0x3F740B3B80000000, float 0x3F740B3B80000000, float 0x3F740B3B80000000, float 0x3F740B3B80000000, float 0x3F740B3B80000000, float 0x3F740B3B80000000, float 0x3F740B3B80000000>
  %357 = fmul fast <8 x float> %343, %356
  %358 = fmul fast <8 x float> %344, <float 0x3EB41A7B00000000, float 0x3EB41A7B00000000, float 0x3EB41A7B00000000, float 0x3EB41A7B00000000, float 0x3EB41A7B00000000, float 0x3EB41A7B00000000, float 0x3EB41A7B00000000, float 0x3EB41A7B00000000>
  %359 = fadd fast <8 x float> %358, <float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000>
  %360 = fmul fast <8 x float> %344, %359
  %361 = fadd fast <8 x float> %360, <float 0x3F629540A0000000, float 0x3F629540A0000000, float 0x3F629540A0000000, float 0x3F629540A0000000, float 0x3F629540A0000000, float 0x3F629540A0000000, float 0x3F629540A0000000, float 0x3F629540A0000000>
  %362 = fmul fast <8 x float> %344, %361
  %363 = fadd fast <8 x float> %362, <float 0x3F740B3BA0000000, float 0x3F740B3BA0000000, float 0x3F740B3BA0000000, float 0x3F740B3BA0000000, float 0x3F740B3BA0000000, float 0x3F740B3BA0000000, float 0x3F740B3BA0000000, float 0x3F740B3BA0000000>
  %364 = fdiv fast <8 x float> %357, %363
  %365 = fcmp fast uge <8 x float> %313, <float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00>
  %366 = select <8 x i1> %365, <8 x float> %313, <8 x float> <float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00>
  %367 = fcmp fast ule <8 x float> %366, <float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00>
  %368 = select <8 x i1> %367, <8 x float> %366, <8 x float> <float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00>
  %369 = fmul fast <8 x float> %368, %368
  %370 = fmul fast <8 x float> %369, <float 0xBCB3E4B800000000, float 0xBCB3E4B800000000, float 0xBCB3E4B800000000, float 0xBCB3E4B800000000, float 0xBCB3E4B800000000, float 0xBCB3E4B800000000, float 0xBCB3E4B800000000, float 0xBCB3E4B800000000>
  %371 = fadd fast <8 x float> %370, <float 0x3D4C266FC0000000, float 0x3D4C266FC0000000, float 0x3D4C266FC0000000, float 0x3D4C266FC0000000, float 0x3D4C266FC0000000, float 0x3D4C266FC0000000, float 0x3D4C266FC0000000, float 0x3D4C266FC0000000>
  %372 = fmul fast <8 x float> %369, %371
  %373 = fadd fast <8 x float> %372, <float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000>
  %374 = fmul fast <8 x float> %369, %373
  %375 = fadd fast <8 x float> %374, <float 0x3E6B800820000000, float 0x3E6B800820000000, float 0x3E6B800820000000, float 0x3E6B800820000000, float 0x3E6B800820000000, float 0x3E6B800820000000, float 0x3E6B800820000000, float 0x3E6B800820000000>
  %376 = fmul fast <8 x float> %369, %375
  %377 = fadd fast <8 x float> %376, <float 0x3EEF286940000000, float 0x3EEF286940000000, float 0x3EEF286940000000, float 0x3EEF286940000000, float 0x3EEF286940000000, float 0x3EEF286940000000, float 0x3EEF286940000000, float 0x3EEF286940000000>
  %378 = fmul fast <8 x float> %369, %377
  %379 = fadd fast <8 x float> %378, <float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000>
  %380 = fmul fast <8 x float> %369, %379
  %381 = fadd fast <8 x float> %380, <float 0x3F740B3B80000000, float 0x3F740B3B80000000, float 0x3F740B3B80000000, float 0x3F740B3B80000000, float 0x3F740B3B80000000, float 0x3F740B3B80000000, float 0x3F740B3B80000000, float 0x3F740B3B80000000>
  %382 = fmul fast <8 x float> %368, %381
  %383 = fmul fast <8 x float> %369, <float 0x3EB41A7B00000000, float 0x3EB41A7B00000000, float 0x3EB41A7B00000000, float 0x3EB41A7B00000000, float 0x3EB41A7B00000000, float 0x3EB41A7B00000000, float 0x3EB41A7B00000000, float 0x3EB41A7B00000000>
  %384 = fadd fast <8 x float> %383, <float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000>
  %385 = fmul fast <8 x float> %369, %384
  %386 = fadd fast <8 x float> %385, <float 0x3F629540A0000000, float 0x3F629540A0000000, float 0x3F629540A0000000, float 0x3F629540A0000000, float 0x3F629540A0000000, float 0x3F629540A0000000, float 0x3F629540A0000000, float 0x3F629540A0000000>
  %387 = fmul fast <8 x float> %369, %386
  %388 = fadd fast <8 x float> %387, <float 0x3F740B3BA0000000, float 0x3F740B3BA0000000, float 0x3F740B3BA0000000, float 0x3F740B3BA0000000, float 0x3F740B3BA0000000, float 0x3F740B3BA0000000, float 0x3F740B3BA0000000, float 0x3F740B3BA0000000>
  %389 = fdiv fast <8 x float> %382, %388
  %390 = fcmp fast uge <8 x float> %314, <float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00>
  %391 = select <8 x i1> %390, <8 x float> %314, <8 x float> <float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00>
  %392 = fcmp fast ule <8 x float> %391, <float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00>
  %393 = select <8 x i1> %392, <8 x float> %391, <8 x float> <float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00>
  %394 = fmul fast <8 x float> %393, %393
  %395 = fmul fast <8 x float> %394, <float 0xBCB3E4B800000000, float 0xBCB3E4B800000000, float 0xBCB3E4B800000000, float 0xBCB3E4B800000000, float 0xBCB3E4B800000000, float 0xBCB3E4B800000000, float 0xBCB3E4B800000000, float 0xBCB3E4B800000000>
  %396 = fadd fast <8 x float> %395, <float 0x3D4C266FC0000000, float 0x3D4C266FC0000000, float 0x3D4C266FC0000000, float 0x3D4C266FC0000000, float 0x3D4C266FC0000000, float 0x3D4C266FC0000000, float 0x3D4C266FC0000000, float 0x3D4C266FC0000000>
  %397 = fmul fast <8 x float> %394, %396
  %398 = fadd fast <8 x float> %397, <float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000>
  %399 = fmul fast <8 x float> %394, %398
  %400 = fadd fast <8 x float> %399, <float 0x3E6B800820000000, float 0x3E6B800820000000, float 0x3E6B800820000000, float 0x3E6B800820000000, float 0x3E6B800820000000, float 0x3E6B800820000000, float 0x3E6B800820000000, float 0x3E6B800820000000>
  %401 = fmul fast <8 x float> %394, %400
  %402 = fadd fast <8 x float> %401, <float 0x3EEF286940000000, float 0x3EEF286940000000, float 0x3EEF286940000000, float 0x3EEF286940000000, float 0x3EEF286940000000, float 0x3EEF286940000000, float 0x3EEF286940000000, float 0x3EEF286940000000>
  %403 = fmul fast <8 x float> %394, %402
  %404 = fadd fast <8 x float> %403, <float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000>
  %405 = fmul fast <8 x float> %394, %404
  %406 = fadd fast <8 x float> %405, <float 0x3F740B3B80000000, float 0x3F740B3B80000000, float 0x3F740B3B80000000, float 0x3F740B3B80000000, float 0x3F740B3B80000000, float 0x3F740B3B80000000, float 0x3F740B3B80000000, float 0x3F740B3B80000000>
  %407 = fmul fast <8 x float> %393, %406
  %408 = fmul fast <8 x float> %394, <float 0x3EB41A7B00000000, float 0x3EB41A7B00000000, float 0x3EB41A7B00000000, float 0x3EB41A7B00000000, float 0x3EB41A7B00000000, float 0x3EB41A7B00000000, float 0x3EB41A7B00000000, float 0x3EB41A7B00000000>
  %409 = fadd fast <8 x float> %408, <float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000>
  %410 = fmul fast <8 x float> %394, %409
  %411 = fadd fast <8 x float> %410, <float 0x3F629540A0000000, float 0x3F629540A0000000, float 0x3F629540A0000000, float 0x3F629540A0000000, float 0x3F629540A0000000, float 0x3F629540A0000000, float 0x3F629540A0000000, float 0x3F629540A0000000>
  %412 = fmul fast <8 x float> %394, %411
  %413 = fadd fast <8 x float> %412, <float 0x3F740B3BA0000000, float 0x3F740B3BA0000000, float 0x3F740B3BA0000000, float 0x3F740B3BA0000000, float 0x3F740B3BA0000000, float 0x3F740B3BA0000000, float 0x3F740B3BA0000000, float 0x3F740B3BA0000000>
  %414 = fdiv fast <8 x float> %407, %413
  %415 = getelementptr inbounds [20 x [200 x float]], [20 x [200 x float]]* %fusion.4.ls1, i64 0, i64 %fusion.4.indvar_address.dim.0.031.ls1, i64 64
  %416 = bitcast float* %415 to <8 x float>*
  store <8 x float> %339, <8 x float>* %416, align 16, !alias.scope !62, !noalias !64
  %417 = getelementptr inbounds [20 x [200 x float]], [20 x [200 x float]]* %fusion.4.ls1, i64 0, i64 %fusion.4.indvar_address.dim.0.031.ls1, i64 72
  %418 = bitcast float* %417 to <8 x float>*
  store <8 x float> %364, <8 x float>* %418, align 16, !alias.scope !62, !noalias !64
  %419 = getelementptr inbounds [20 x [200 x float]], [20 x [200 x float]]* %fusion.4.ls1, i64 0, i64 %fusion.4.indvar_address.dim.0.031.ls1, i64 80
  %420 = bitcast float* %419 to <8 x float>*
  store <8 x float> %389, <8 x float>* %420, align 16, !alias.scope !62, !noalias !64
  %421 = getelementptr inbounds [20 x [200 x float]], [20 x [200 x float]]* %fusion.4.ls1, i64 0, i64 %fusion.4.indvar_address.dim.0.031.ls1, i64 88
  %422 = bitcast float* %421 to <8 x float>*
  store <8 x float> %414, <8 x float>* %422, align 16, !alias.scope !62, !noalias !64
  %423 = getelementptr inbounds [20 x [800 x float]], [20 x [800 x float]]* %.ls12, i64 0, i64 %fusion.4.indvar_address.dim.0.031.ls1, i64 96
  %424 = bitcast float* %423 to <8 x float>*
  %wide.load126.3.ls1 = load <8 x float>, <8 x float>* %424, align 16, !alias.scope !59, !noalias !23
  %425 = getelementptr inbounds [20 x [800 x float]], [20 x [800 x float]]* %.ls12, i64 0, i64 %fusion.4.indvar_address.dim.0.031.ls1, i64 104
  %426 = bitcast float* %425 to <8 x float>*
  %wide.load127.3.ls1 = load <8 x float>, <8 x float>* %426, align 16, !alias.scope !59, !noalias !23
  %427 = getelementptr inbounds [20 x [800 x float]], [20 x [800 x float]]* %.ls12, i64 0, i64 %fusion.4.indvar_address.dim.0.031.ls1, i64 112
  %428 = bitcast float* %427 to <8 x float>*
  %wide.load128.3.ls1 = load <8 x float>, <8 x float>* %428, align 16, !alias.scope !59, !noalias !23
  %429 = getelementptr inbounds [20 x [800 x float]], [20 x [800 x float]]* %.ls12, i64 0, i64 %fusion.4.indvar_address.dim.0.031.ls1, i64 120
  %430 = bitcast float* %429 to <8 x float>*
  %wide.load129.3.ls1 = load <8 x float>, <8 x float>* %430, align 16, !alias.scope !59, !noalias !23
  %431 = fadd fast <8 x float> %wide.load126.3.ls1, %wide.load130.3.ls1
  %432 = fadd fast <8 x float> %wide.load127.3.ls1, %wide.load131.3.ls1
  %433 = fadd fast <8 x float> %wide.load128.3.ls1, %wide.load132.3.ls1
  %434 = fadd fast <8 x float> %wide.load129.3.ls1, %wide.load133.3.ls1
  %435 = fmul fast <8 x float> %431, <float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01>
  %436 = fmul fast <8 x float> %432, <float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01>
  %437 = fmul fast <8 x float> %433, <float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01>
  %438 = fmul fast <8 x float> %434, <float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01>
  %439 = fcmp fast uge <8 x float> %435, <float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00>
  %440 = select <8 x i1> %439, <8 x float> %435, <8 x float> <float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00>
  %441 = fcmp fast ule <8 x float> %440, <float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00>
  %442 = select <8 x i1> %441, <8 x float> %440, <8 x float> <float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00>
  %443 = fmul fast <8 x float> %442, %442
  %444 = fmul fast <8 x float> %443, <float 0xBCB3E4B800000000, float 0xBCB3E4B800000000, float 0xBCB3E4B800000000, float 0xBCB3E4B800000000, float 0xBCB3E4B800000000, float 0xBCB3E4B800000000, float 0xBCB3E4B800000000, float 0xBCB3E4B800000000>
  %445 = fadd fast <8 x float> %444, <float 0x3D4C266FC0000000, float 0x3D4C266FC0000000, float 0x3D4C266FC0000000, float 0x3D4C266FC0000000, float 0x3D4C266FC0000000, float 0x3D4C266FC0000000, float 0x3D4C266FC0000000, float 0x3D4C266FC0000000>
  %446 = fmul fast <8 x float> %443, %445
  %447 = fadd fast <8 x float> %446, <float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000>
  %448 = fmul fast <8 x float> %443, %447
  %449 = fadd fast <8 x float> %448, <float 0x3E6B800820000000, float 0x3E6B800820000000, float 0x3E6B800820000000, float 0x3E6B800820000000, float 0x3E6B800820000000, float 0x3E6B800820000000, float 0x3E6B800820000000, float 0x3E6B800820000000>
  %450 = fmul fast <8 x float> %443, %449
  %451 = fadd fast <8 x float> %450, <float 0x3EEF286940000000, float 0x3EEF286940000000, float 0x3EEF286940000000, float 0x3EEF286940000000, float 0x3EEF286940000000, float 0x3EEF286940000000, float 0x3EEF286940000000, float 0x3EEF286940000000>
  %452 = fmul fast <8 x float> %443, %451
  %453 = fadd fast <8 x float> %452, <float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000>
  %454 = fmul fast <8 x float> %443, %453
  %455 = fadd fast <8 x float> %454, <float 0x3F740B3B80000000, float 0x3F740B3B80000000, float 0x3F740B3B80000000, float 0x3F740B3B80000000, float 0x3F740B3B80000000, float 0x3F740B3B80000000, float 0x3F740B3B80000000, float 0x3F740B3B80000000>
  %456 = fmul fast <8 x float> %442, %455
  %457 = fmul fast <8 x float> %443, <float 0x3EB41A7B00000000, float 0x3EB41A7B00000000, float 0x3EB41A7B00000000, float 0x3EB41A7B00000000, float 0x3EB41A7B00000000, float 0x3EB41A7B00000000, float 0x3EB41A7B00000000, float 0x3EB41A7B00000000>
  %458 = fadd fast <8 x float> %457, <float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000>
  %459 = fmul fast <8 x float> %443, %458
  %460 = fadd fast <8 x float> %459, <float 0x3F629540A0000000, float 0x3F629540A0000000, float 0x3F629540A0000000, float 0x3F629540A0000000, float 0x3F629540A0000000, float 0x3F629540A0000000, float 0x3F629540A0000000, float 0x3F629540A0000000>
  %461 = fmul fast <8 x float> %443, %460
  %462 = fadd fast <8 x float> %461, <float 0x3F740B3BA0000000, float 0x3F740B3BA0000000, float 0x3F740B3BA0000000, float 0x3F740B3BA0000000, float 0x3F740B3BA0000000, float 0x3F740B3BA0000000, float 0x3F740B3BA0000000, float 0x3F740B3BA0000000>
  %463 = fdiv fast <8 x float> %456, %462
  %464 = fcmp fast uge <8 x float> %436, <float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00>
  %465 = select <8 x i1> %464, <8 x float> %436, <8 x float> <float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00>
  %466 = fcmp fast ule <8 x float> %465, <float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00>
  %467 = select <8 x i1> %466, <8 x float> %465, <8 x float> <float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00>
  %468 = fmul fast <8 x float> %467, %467
  %469 = fmul fast <8 x float> %468, <float 0xBCB3E4B800000000, float 0xBCB3E4B800000000, float 0xBCB3E4B800000000, float 0xBCB3E4B800000000, float 0xBCB3E4B800000000, float 0xBCB3E4B800000000, float 0xBCB3E4B800000000, float 0xBCB3E4B800000000>
  %470 = fadd fast <8 x float> %469, <float 0x3D4C266FC0000000, float 0x3D4C266FC0000000, float 0x3D4C266FC0000000, float 0x3D4C266FC0000000, float 0x3D4C266FC0000000, float 0x3D4C266FC0000000, float 0x3D4C266FC0000000, float 0x3D4C266FC0000000>
  %471 = fmul fast <8 x float> %468, %470
  %472 = fadd fast <8 x float> %471, <float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000>
  %473 = fmul fast <8 x float> %468, %472
  %474 = fadd fast <8 x float> %473, <float 0x3E6B800820000000, float 0x3E6B800820000000, float 0x3E6B800820000000, float 0x3E6B800820000000, float 0x3E6B800820000000, float 0x3E6B800820000000, float 0x3E6B800820000000, float 0x3E6B800820000000>
  %475 = fmul fast <8 x float> %468, %474
  %476 = fadd fast <8 x float> %475, <float 0x3EEF286940000000, float 0x3EEF286940000000, float 0x3EEF286940000000, float 0x3EEF286940000000, float 0x3EEF286940000000, float 0x3EEF286940000000, float 0x3EEF286940000000, float 0x3EEF286940000000>
  %477 = fmul fast <8 x float> %468, %476
  %478 = fadd fast <8 x float> %477, <float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000>
  %479 = fmul fast <8 x float> %468, %478
  %480 = fadd fast <8 x float> %479, <float 0x3F740B3B80000000, float 0x3F740B3B80000000, float 0x3F740B3B80000000, float 0x3F740B3B80000000, float 0x3F740B3B80000000, float 0x3F740B3B80000000, float 0x3F740B3B80000000, float 0x3F740B3B80000000>
  %481 = fmul fast <8 x float> %467, %480
  %482 = fmul fast <8 x float> %468, <float 0x3EB41A7B00000000, float 0x3EB41A7B00000000, float 0x3EB41A7B00000000, float 0x3EB41A7B00000000, float 0x3EB41A7B00000000, float 0x3EB41A7B00000000, float 0x3EB41A7B00000000, float 0x3EB41A7B00000000>
  %483 = fadd fast <8 x float> %482, <float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000>
  %484 = fmul fast <8 x float> %468, %483
  %485 = fadd fast <8 x float> %484, <float 0x3F629540A0000000, float 0x3F629540A0000000, float 0x3F629540A0000000, float 0x3F629540A0000000, float 0x3F629540A0000000, float 0x3F629540A0000000, float 0x3F629540A0000000, float 0x3F629540A0000000>
  %486 = fmul fast <8 x float> %468, %485
  %487 = fadd fast <8 x float> %486, <float 0x3F740B3BA0000000, float 0x3F740B3BA0000000, float 0x3F740B3BA0000000, float 0x3F740B3BA0000000, float 0x3F740B3BA0000000, float 0x3F740B3BA0000000, float 0x3F740B3BA0000000, float 0x3F740B3BA0000000>
  %488 = fdiv fast <8 x float> %481, %487
  %489 = fcmp fast uge <8 x float> %437, <float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00>
  %490 = select <8 x i1> %489, <8 x float> %437, <8 x float> <float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00>
  %491 = fcmp fast ule <8 x float> %490, <float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00>
  %492 = select <8 x i1> %491, <8 x float> %490, <8 x float> <float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00>
  %493 = fmul fast <8 x float> %492, %492
  %494 = fmul fast <8 x float> %493, <float 0xBCB3E4B800000000, float 0xBCB3E4B800000000, float 0xBCB3E4B800000000, float 0xBCB3E4B800000000, float 0xBCB3E4B800000000, float 0xBCB3E4B800000000, float 0xBCB3E4B800000000, float 0xBCB3E4B800000000>
  %495 = fadd fast <8 x float> %494, <float 0x3D4C266FC0000000, float 0x3D4C266FC0000000, float 0x3D4C266FC0000000, float 0x3D4C266FC0000000, float 0x3D4C266FC0000000, float 0x3D4C266FC0000000, float 0x3D4C266FC0000000, float 0x3D4C266FC0000000>
  %496 = fmul fast <8 x float> %493, %495
  %497 = fadd fast <8 x float> %496, <float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000>
  %498 = fmul fast <8 x float> %493, %497
  %499 = fadd fast <8 x float> %498, <float 0x3E6B800820000000, float 0x3E6B800820000000, float 0x3E6B800820000000, float 0x3E6B800820000000, float 0x3E6B800820000000, float 0x3E6B800820000000, float 0x3E6B800820000000, float 0x3E6B800820000000>
  %500 = fmul fast <8 x float> %493, %499
  %501 = fadd fast <8 x float> %500, <float 0x3EEF286940000000, float 0x3EEF286940000000, float 0x3EEF286940000000, float 0x3EEF286940000000, float 0x3EEF286940000000, float 0x3EEF286940000000, float 0x3EEF286940000000, float 0x3EEF286940000000>
  %502 = fmul fast <8 x float> %493, %501
  %503 = fadd fast <8 x float> %502, <float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000>
  %504 = fmul fast <8 x float> %493, %503
  %505 = fadd fast <8 x float> %504, <float 0x3F740B3B80000000, float 0x3F740B3B80000000, float 0x3F740B3B80000000, float 0x3F740B3B80000000, float 0x3F740B3B80000000, float 0x3F740B3B80000000, float 0x3F740B3B80000000, float 0x3F740B3B80000000>
  %506 = fmul fast <8 x float> %492, %505
  %507 = fmul fast <8 x float> %493, <float 0x3EB41A7B00000000, float 0x3EB41A7B00000000, float 0x3EB41A7B00000000, float 0x3EB41A7B00000000, float 0x3EB41A7B00000000, float 0x3EB41A7B00000000, float 0x3EB41A7B00000000, float 0x3EB41A7B00000000>
  %508 = fadd fast <8 x float> %507, <float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000>
  %509 = fmul fast <8 x float> %493, %508
  %510 = fadd fast <8 x float> %509, <float 0x3F629540A0000000, float 0x3F629540A0000000, float 0x3F629540A0000000, float 0x3F629540A0000000, float 0x3F629540A0000000, float 0x3F629540A0000000, float 0x3F629540A0000000, float 0x3F629540A0000000>
  %511 = fmul fast <8 x float> %493, %510
  %512 = fadd fast <8 x float> %511, <float 0x3F740B3BA0000000, float 0x3F740B3BA0000000, float 0x3F740B3BA0000000, float 0x3F740B3BA0000000, float 0x3F740B3BA0000000, float 0x3F740B3BA0000000, float 0x3F740B3BA0000000, float 0x3F740B3BA0000000>
  %513 = fdiv fast <8 x float> %506, %512
  %514 = fcmp fast uge <8 x float> %438, <float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00>
  %515 = select <8 x i1> %514, <8 x float> %438, <8 x float> <float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00>
  %516 = fcmp fast ule <8 x float> %515, <float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00>
  %517 = select <8 x i1> %516, <8 x float> %515, <8 x float> <float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00>
  %518 = fmul fast <8 x float> %517, %517
  %519 = fmul fast <8 x float> %518, <float 0xBCB3E4B800000000, float 0xBCB3E4B800000000, float 0xBCB3E4B800000000, float 0xBCB3E4B800000000, float 0xBCB3E4B800000000, float 0xBCB3E4B800000000, float 0xBCB3E4B800000000, float 0xBCB3E4B800000000>
  %520 = fadd fast <8 x float> %519, <float 0x3D4C266FC0000000, float 0x3D4C266FC0000000, float 0x3D4C266FC0000000, float 0x3D4C266FC0000000, float 0x3D4C266FC0000000, float 0x3D4C266FC0000000, float 0x3D4C266FC0000000, float 0x3D4C266FC0000000>
  %521 = fmul fast <8 x float> %518, %520
  %522 = fadd fast <8 x float> %521, <float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000>
  %523 = fmul fast <8 x float> %518, %522
  %524 = fadd fast <8 x float> %523, <float 0x3E6B800820000000, float 0x3E6B800820000000, float 0x3E6B800820000000, float 0x3E6B800820000000, float 0x3E6B800820000000, float 0x3E6B800820000000, float 0x3E6B800820000000, float 0x3E6B800820000000>
  %525 = fmul fast <8 x float> %518, %524
  %526 = fadd fast <8 x float> %525, <float 0x3EEF286940000000, float 0x3EEF286940000000, float 0x3EEF286940000000, float 0x3EEF286940000000, float 0x3EEF286940000000, float 0x3EEF286940000000, float 0x3EEF286940000000, float 0x3EEF286940000000>
  %527 = fmul fast <8 x float> %518, %526
  %528 = fadd fast <8 x float> %527, <float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000>
  %529 = fmul fast <8 x float> %518, %528
  %530 = fadd fast <8 x float> %529, <float 0x3F740B3B80000000, float 0x3F740B3B80000000, float 0x3F740B3B80000000, float 0x3F740B3B80000000, float 0x3F740B3B80000000, float 0x3F740B3B80000000, float 0x3F740B3B80000000, float 0x3F740B3B80000000>
  %531 = fmul fast <8 x float> %517, %530
  %532 = fmul fast <8 x float> %518, <float 0x3EB41A7B00000000, float 0x3EB41A7B00000000, float 0x3EB41A7B00000000, float 0x3EB41A7B00000000, float 0x3EB41A7B00000000, float 0x3EB41A7B00000000, float 0x3EB41A7B00000000, float 0x3EB41A7B00000000>
  %533 = fadd fast <8 x float> %532, <float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000>
  %534 = fmul fast <8 x float> %518, %533
  %535 = fadd fast <8 x float> %534, <float 0x3F629540A0000000, float 0x3F629540A0000000, float 0x3F629540A0000000, float 0x3F629540A0000000, float 0x3F629540A0000000, float 0x3F629540A0000000, float 0x3F629540A0000000, float 0x3F629540A0000000>
  %536 = fmul fast <8 x float> %518, %535
  %537 = fadd fast <8 x float> %536, <float 0x3F740B3BA0000000, float 0x3F740B3BA0000000, float 0x3F740B3BA0000000, float 0x3F740B3BA0000000, float 0x3F740B3BA0000000, float 0x3F740B3BA0000000, float 0x3F740B3BA0000000, float 0x3F740B3BA0000000>
  %538 = fdiv fast <8 x float> %531, %537
  %539 = getelementptr inbounds [20 x [200 x float]], [20 x [200 x float]]* %fusion.4.ls1, i64 0, i64 %fusion.4.indvar_address.dim.0.031.ls1, i64 96
  %540 = bitcast float* %539 to <8 x float>*
  store <8 x float> %463, <8 x float>* %540, align 16, !alias.scope !62, !noalias !64
  %541 = getelementptr inbounds [20 x [200 x float]], [20 x [200 x float]]* %fusion.4.ls1, i64 0, i64 %fusion.4.indvar_address.dim.0.031.ls1, i64 104
  %542 = bitcast float* %541 to <8 x float>*
  store <8 x float> %488, <8 x float>* %542, align 16, !alias.scope !62, !noalias !64
  %543 = getelementptr inbounds [20 x [200 x float]], [20 x [200 x float]]* %fusion.4.ls1, i64 0, i64 %fusion.4.indvar_address.dim.0.031.ls1, i64 112
  %544 = bitcast float* %543 to <8 x float>*
  store <8 x float> %513, <8 x float>* %544, align 16, !alias.scope !62, !noalias !64
  %545 = getelementptr inbounds [20 x [200 x float]], [20 x [200 x float]]* %fusion.4.ls1, i64 0, i64 %fusion.4.indvar_address.dim.0.031.ls1, i64 120
  %546 = bitcast float* %545 to <8 x float>*
  store <8 x float> %538, <8 x float>* %546, align 16, !alias.scope !62, !noalias !64
  %547 = getelementptr inbounds [20 x [800 x float]], [20 x [800 x float]]* %.ls12, i64 0, i64 %fusion.4.indvar_address.dim.0.031.ls1, i64 128
  %548 = bitcast float* %547 to <8 x float>*
  %wide.load126.4.ls1 = load <8 x float>, <8 x float>* %548, align 16, !alias.scope !59, !noalias !23
  %549 = getelementptr inbounds [20 x [800 x float]], [20 x [800 x float]]* %.ls12, i64 0, i64 %fusion.4.indvar_address.dim.0.031.ls1, i64 136
  %550 = bitcast float* %549 to <8 x float>*
  %wide.load127.4.ls1 = load <8 x float>, <8 x float>* %550, align 16, !alias.scope !59, !noalias !23
  %551 = getelementptr inbounds [20 x [800 x float]], [20 x [800 x float]]* %.ls12, i64 0, i64 %fusion.4.indvar_address.dim.0.031.ls1, i64 144
  %552 = bitcast float* %551 to <8 x float>*
  %wide.load128.4.ls1 = load <8 x float>, <8 x float>* %552, align 16, !alias.scope !59, !noalias !23
  %553 = getelementptr inbounds [20 x [800 x float]], [20 x [800 x float]]* %.ls12, i64 0, i64 %fusion.4.indvar_address.dim.0.031.ls1, i64 152
  %554 = bitcast float* %553 to <8 x float>*
  %wide.load129.4.ls1 = load <8 x float>, <8 x float>* %554, align 16, !alias.scope !59, !noalias !23
  %555 = fadd fast <8 x float> %wide.load126.4.ls1, %wide.load130.4.ls1
  %556 = fadd fast <8 x float> %wide.load127.4.ls1, %wide.load131.4.ls1
  %557 = fadd fast <8 x float> %wide.load128.4.ls1, %wide.load132.4.ls1
  %558 = fadd fast <8 x float> %wide.load129.4.ls1, %wide.load133.4.ls1
  %559 = fmul fast <8 x float> %555, <float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01>
  %560 = fmul fast <8 x float> %556, <float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01>
  %561 = fmul fast <8 x float> %557, <float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01>
  %562 = fmul fast <8 x float> %558, <float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01>
  %563 = fcmp fast uge <8 x float> %559, <float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00>
  %564 = select <8 x i1> %563, <8 x float> %559, <8 x float> <float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00>
  %565 = fcmp fast ule <8 x float> %564, <float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00>
  %566 = select <8 x i1> %565, <8 x float> %564, <8 x float> <float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00>
  %567 = fmul fast <8 x float> %566, %566
  %568 = fmul fast <8 x float> %567, <float 0xBCB3E4B800000000, float 0xBCB3E4B800000000, float 0xBCB3E4B800000000, float 0xBCB3E4B800000000, float 0xBCB3E4B800000000, float 0xBCB3E4B800000000, float 0xBCB3E4B800000000, float 0xBCB3E4B800000000>
  %569 = fadd fast <8 x float> %568, <float 0x3D4C266FC0000000, float 0x3D4C266FC0000000, float 0x3D4C266FC0000000, float 0x3D4C266FC0000000, float 0x3D4C266FC0000000, float 0x3D4C266FC0000000, float 0x3D4C266FC0000000, float 0x3D4C266FC0000000>
  %570 = fmul fast <8 x float> %567, %569
  %571 = fadd fast <8 x float> %570, <float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000>
  %572 = fmul fast <8 x float> %567, %571
  %573 = fadd fast <8 x float> %572, <float 0x3E6B800820000000, float 0x3E6B800820000000, float 0x3E6B800820000000, float 0x3E6B800820000000, float 0x3E6B800820000000, float 0x3E6B800820000000, float 0x3E6B800820000000, float 0x3E6B800820000000>
  %574 = fmul fast <8 x float> %567, %573
  %575 = fadd fast <8 x float> %574, <float 0x3EEF286940000000, float 0x3EEF286940000000, float 0x3EEF286940000000, float 0x3EEF286940000000, float 0x3EEF286940000000, float 0x3EEF286940000000, float 0x3EEF286940000000, float 0x3EEF286940000000>
  %576 = fmul fast <8 x float> %567, %575
  %577 = fadd fast <8 x float> %576, <float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000>
  %578 = fmul fast <8 x float> %567, %577
  %579 = fadd fast <8 x float> %578, <float 0x3F740B3B80000000, float 0x3F740B3B80000000, float 0x3F740B3B80000000, float 0x3F740B3B80000000, float 0x3F740B3B80000000, float 0x3F740B3B80000000, float 0x3F740B3B80000000, float 0x3F740B3B80000000>
  %580 = fmul fast <8 x float> %566, %579
  %581 = fmul fast <8 x float> %567, <float 0x3EB41A7B00000000, float 0x3EB41A7B00000000, float 0x3EB41A7B00000000, float 0x3EB41A7B00000000, float 0x3EB41A7B00000000, float 0x3EB41A7B00000000, float 0x3EB41A7B00000000, float 0x3EB41A7B00000000>
  %582 = fadd fast <8 x float> %581, <float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000>
  %583 = fmul fast <8 x float> %567, %582
  %584 = fadd fast <8 x float> %583, <float 0x3F629540A0000000, float 0x3F629540A0000000, float 0x3F629540A0000000, float 0x3F629540A0000000, float 0x3F629540A0000000, float 0x3F629540A0000000, float 0x3F629540A0000000, float 0x3F629540A0000000>
  %585 = fmul fast <8 x float> %567, %584
  %586 = fadd fast <8 x float> %585, <float 0x3F740B3BA0000000, float 0x3F740B3BA0000000, float 0x3F740B3BA0000000, float 0x3F740B3BA0000000, float 0x3F740B3BA0000000, float 0x3F740B3BA0000000, float 0x3F740B3BA0000000, float 0x3F740B3BA0000000>
  %587 = fdiv fast <8 x float> %580, %586
  %588 = fcmp fast uge <8 x float> %560, <float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00>
  %589 = select <8 x i1> %588, <8 x float> %560, <8 x float> <float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00>
  %590 = fcmp fast ule <8 x float> %589, <float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00>
  %591 = select <8 x i1> %590, <8 x float> %589, <8 x float> <float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00>
  %592 = fmul fast <8 x float> %591, %591
  %593 = fmul fast <8 x float> %592, <float 0xBCB3E4B800000000, float 0xBCB3E4B800000000, float 0xBCB3E4B800000000, float 0xBCB3E4B800000000, float 0xBCB3E4B800000000, float 0xBCB3E4B800000000, float 0xBCB3E4B800000000, float 0xBCB3E4B800000000>
  %594 = fadd fast <8 x float> %593, <float 0x3D4C266FC0000000, float 0x3D4C266FC0000000, float 0x3D4C266FC0000000, float 0x3D4C266FC0000000, float 0x3D4C266FC0000000, float 0x3D4C266FC0000000, float 0x3D4C266FC0000000, float 0x3D4C266FC0000000>
  %595 = fmul fast <8 x float> %592, %594
  %596 = fadd fast <8 x float> %595, <float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000>
  %597 = fmul fast <8 x float> %592, %596
  %598 = fadd fast <8 x float> %597, <float 0x3E6B800820000000, float 0x3E6B800820000000, float 0x3E6B800820000000, float 0x3E6B800820000000, float 0x3E6B800820000000, float 0x3E6B800820000000, float 0x3E6B800820000000, float 0x3E6B800820000000>
  %599 = fmul fast <8 x float> %592, %598
  %600 = fadd fast <8 x float> %599, <float 0x3EEF286940000000, float 0x3EEF286940000000, float 0x3EEF286940000000, float 0x3EEF286940000000, float 0x3EEF286940000000, float 0x3EEF286940000000, float 0x3EEF286940000000, float 0x3EEF286940000000>
  %601 = fmul fast <8 x float> %592, %600
  %602 = fadd fast <8 x float> %601, <float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000>
  %603 = fmul fast <8 x float> %592, %602
  %604 = fadd fast <8 x float> %603, <float 0x3F740B3B80000000, float 0x3F740B3B80000000, float 0x3F740B3B80000000, float 0x3F740B3B80000000, float 0x3F740B3B80000000, float 0x3F740B3B80000000, float 0x3F740B3B80000000, float 0x3F740B3B80000000>
  %605 = fmul fast <8 x float> %591, %604
  %606 = fmul fast <8 x float> %592, <float 0x3EB41A7B00000000, float 0x3EB41A7B00000000, float 0x3EB41A7B00000000, float 0x3EB41A7B00000000, float 0x3EB41A7B00000000, float 0x3EB41A7B00000000, float 0x3EB41A7B00000000, float 0x3EB41A7B00000000>
  %607 = fadd fast <8 x float> %606, <float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000>
  %608 = fmul fast <8 x float> %592, %607
  %609 = fadd fast <8 x float> %608, <float 0x3F629540A0000000, float 0x3F629540A0000000, float 0x3F629540A0000000, float 0x3F629540A0000000, float 0x3F629540A0000000, float 0x3F629540A0000000, float 0x3F629540A0000000, float 0x3F629540A0000000>
  %610 = fmul fast <8 x float> %592, %609
  %611 = fadd fast <8 x float> %610, <float 0x3F740B3BA0000000, float 0x3F740B3BA0000000, float 0x3F740B3BA0000000, float 0x3F740B3BA0000000, float 0x3F740B3BA0000000, float 0x3F740B3BA0000000, float 0x3F740B3BA0000000, float 0x3F740B3BA0000000>
  %612 = fdiv fast <8 x float> %605, %611
  %613 = fcmp fast uge <8 x float> %561, <float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00>
  %614 = select <8 x i1> %613, <8 x float> %561, <8 x float> <float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00>
  %615 = fcmp fast ule <8 x float> %614, <float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00>
  %616 = select <8 x i1> %615, <8 x float> %614, <8 x float> <float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00>
  %617 = fmul fast <8 x float> %616, %616
  %618 = fmul fast <8 x float> %617, <float 0xBCB3E4B800000000, float 0xBCB3E4B800000000, float 0xBCB3E4B800000000, float 0xBCB3E4B800000000, float 0xBCB3E4B800000000, float 0xBCB3E4B800000000, float 0xBCB3E4B800000000, float 0xBCB3E4B800000000>
  %619 = fadd fast <8 x float> %618, <float 0x3D4C266FC0000000, float 0x3D4C266FC0000000, float 0x3D4C266FC0000000, float 0x3D4C266FC0000000, float 0x3D4C266FC0000000, float 0x3D4C266FC0000000, float 0x3D4C266FC0000000, float 0x3D4C266FC0000000>
  %620 = fmul fast <8 x float> %617, %619
  %621 = fadd fast <8 x float> %620, <float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000>
  %622 = fmul fast <8 x float> %617, %621
  %623 = fadd fast <8 x float> %622, <float 0x3E6B800820000000, float 0x3E6B800820000000, float 0x3E6B800820000000, float 0x3E6B800820000000, float 0x3E6B800820000000, float 0x3E6B800820000000, float 0x3E6B800820000000, float 0x3E6B800820000000>
  %624 = fmul fast <8 x float> %617, %623
  %625 = fadd fast <8 x float> %624, <float 0x3EEF286940000000, float 0x3EEF286940000000, float 0x3EEF286940000000, float 0x3EEF286940000000, float 0x3EEF286940000000, float 0x3EEF286940000000, float 0x3EEF286940000000, float 0x3EEF286940000000>
  %626 = fmul fast <8 x float> %617, %625
  %627 = fadd fast <8 x float> %626, <float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000>
  %628 = fmul fast <8 x float> %617, %627
  %629 = fadd fast <8 x float> %628, <float 0x3F740B3B80000000, float 0x3F740B3B80000000, float 0x3F740B3B80000000, float 0x3F740B3B80000000, float 0x3F740B3B80000000, float 0x3F740B3B80000000, float 0x3F740B3B80000000, float 0x3F740B3B80000000>
  %630 = fmul fast <8 x float> %616, %629
  %631 = fmul fast <8 x float> %617, <float 0x3EB41A7B00000000, float 0x3EB41A7B00000000, float 0x3EB41A7B00000000, float 0x3EB41A7B00000000, float 0x3EB41A7B00000000, float 0x3EB41A7B00000000, float 0x3EB41A7B00000000, float 0x3EB41A7B00000000>
  %632 = fadd fast <8 x float> %631, <float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000>
  %633 = fmul fast <8 x float> %617, %632
  %634 = fadd fast <8 x float> %633, <float 0x3F629540A0000000, float 0x3F629540A0000000, float 0x3F629540A0000000, float 0x3F629540A0000000, float 0x3F629540A0000000, float 0x3F629540A0000000, float 0x3F629540A0000000, float 0x3F629540A0000000>
  %635 = fmul fast <8 x float> %617, %634
  %636 = fadd fast <8 x float> %635, <float 0x3F740B3BA0000000, float 0x3F740B3BA0000000, float 0x3F740B3BA0000000, float 0x3F740B3BA0000000, float 0x3F740B3BA0000000, float 0x3F740B3BA0000000, float 0x3F740B3BA0000000, float 0x3F740B3BA0000000>
  %637 = fdiv fast <8 x float> %630, %636
  %638 = fcmp fast uge <8 x float> %562, <float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00>
  %639 = select <8 x i1> %638, <8 x float> %562, <8 x float> <float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00>
  %640 = fcmp fast ule <8 x float> %639, <float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00>
  %641 = select <8 x i1> %640, <8 x float> %639, <8 x float> <float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00>
  %642 = fmul fast <8 x float> %641, %641
  %643 = fmul fast <8 x float> %642, <float 0xBCB3E4B800000000, float 0xBCB3E4B800000000, float 0xBCB3E4B800000000, float 0xBCB3E4B800000000, float 0xBCB3E4B800000000, float 0xBCB3E4B800000000, float 0xBCB3E4B800000000, float 0xBCB3E4B800000000>
  %644 = fadd fast <8 x float> %643, <float 0x3D4C266FC0000000, float 0x3D4C266FC0000000, float 0x3D4C266FC0000000, float 0x3D4C266FC0000000, float 0x3D4C266FC0000000, float 0x3D4C266FC0000000, float 0x3D4C266FC0000000, float 0x3D4C266FC0000000>
  %645 = fmul fast <8 x float> %642, %644
  %646 = fadd fast <8 x float> %645, <float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000>
  %647 = fmul fast <8 x float> %642, %646
  %648 = fadd fast <8 x float> %647, <float 0x3E6B800820000000, float 0x3E6B800820000000, float 0x3E6B800820000000, float 0x3E6B800820000000, float 0x3E6B800820000000, float 0x3E6B800820000000, float 0x3E6B800820000000, float 0x3E6B800820000000>
  %649 = fmul fast <8 x float> %642, %648
  %650 = fadd fast <8 x float> %649, <float 0x3EEF286940000000, float 0x3EEF286940000000, float 0x3EEF286940000000, float 0x3EEF286940000000, float 0x3EEF286940000000, float 0x3EEF286940000000, float 0x3EEF286940000000, float 0x3EEF286940000000>
  %651 = fmul fast <8 x float> %642, %650
  %652 = fadd fast <8 x float> %651, <float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000>
  %653 = fmul fast <8 x float> %642, %652
  %654 = fadd fast <8 x float> %653, <float 0x3F740B3B80000000, float 0x3F740B3B80000000, float 0x3F740B3B80000000, float 0x3F740B3B80000000, float 0x3F740B3B80000000, float 0x3F740B3B80000000, float 0x3F740B3B80000000, float 0x3F740B3B80000000>
  %655 = fmul fast <8 x float> %641, %654
  %656 = fmul fast <8 x float> %642, <float 0x3EB41A7B00000000, float 0x3EB41A7B00000000, float 0x3EB41A7B00000000, float 0x3EB41A7B00000000, float 0x3EB41A7B00000000, float 0x3EB41A7B00000000, float 0x3EB41A7B00000000, float 0x3EB41A7B00000000>
  %657 = fadd fast <8 x float> %656, <float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000>
  %658 = fmul fast <8 x float> %642, %657
  %659 = fadd fast <8 x float> %658, <float 0x3F629540A0000000, float 0x3F629540A0000000, float 0x3F629540A0000000, float 0x3F629540A0000000, float 0x3F629540A0000000, float 0x3F629540A0000000, float 0x3F629540A0000000, float 0x3F629540A0000000>
  %660 = fmul fast <8 x float> %642, %659
  %661 = fadd fast <8 x float> %660, <float 0x3F740B3BA0000000, float 0x3F740B3BA0000000, float 0x3F740B3BA0000000, float 0x3F740B3BA0000000, float 0x3F740B3BA0000000, float 0x3F740B3BA0000000, float 0x3F740B3BA0000000, float 0x3F740B3BA0000000>
  %662 = fdiv fast <8 x float> %655, %661
  %663 = getelementptr inbounds [20 x [200 x float]], [20 x [200 x float]]* %fusion.4.ls1, i64 0, i64 %fusion.4.indvar_address.dim.0.031.ls1, i64 128
  %664 = bitcast float* %663 to <8 x float>*
  store <8 x float> %587, <8 x float>* %664, align 16, !alias.scope !62, !noalias !64
  %665 = getelementptr inbounds [20 x [200 x float]], [20 x [200 x float]]* %fusion.4.ls1, i64 0, i64 %fusion.4.indvar_address.dim.0.031.ls1, i64 136
  %666 = bitcast float* %665 to <8 x float>*
  store <8 x float> %612, <8 x float>* %666, align 16, !alias.scope !62, !noalias !64
  %667 = getelementptr inbounds [20 x [200 x float]], [20 x [200 x float]]* %fusion.4.ls1, i64 0, i64 %fusion.4.indvar_address.dim.0.031.ls1, i64 144
  %668 = bitcast float* %667 to <8 x float>*
  store <8 x float> %637, <8 x float>* %668, align 16, !alias.scope !62, !noalias !64
  %669 = getelementptr inbounds [20 x [200 x float]], [20 x [200 x float]]* %fusion.4.ls1, i64 0, i64 %fusion.4.indvar_address.dim.0.031.ls1, i64 152
  %670 = bitcast float* %669 to <8 x float>*
  store <8 x float> %662, <8 x float>* %670, align 16, !alias.scope !62, !noalias !64
  %671 = getelementptr inbounds [20 x [800 x float]], [20 x [800 x float]]* %.ls12, i64 0, i64 %fusion.4.indvar_address.dim.0.031.ls1, i64 160
  %672 = bitcast float* %671 to <8 x float>*
  %wide.load126.5.ls1 = load <8 x float>, <8 x float>* %672, align 16, !alias.scope !59, !noalias !23
  %673 = getelementptr inbounds [20 x [800 x float]], [20 x [800 x float]]* %.ls12, i64 0, i64 %fusion.4.indvar_address.dim.0.031.ls1, i64 168
  %674 = bitcast float* %673 to <8 x float>*
  %wide.load127.5.ls1 = load <8 x float>, <8 x float>* %674, align 16, !alias.scope !59, !noalias !23
  %675 = getelementptr inbounds [20 x [800 x float]], [20 x [800 x float]]* %.ls12, i64 0, i64 %fusion.4.indvar_address.dim.0.031.ls1, i64 176
  %676 = bitcast float* %675 to <8 x float>*
  %wide.load128.5.ls1 = load <8 x float>, <8 x float>* %676, align 16, !alias.scope !59, !noalias !23
  %677 = getelementptr inbounds [20 x [800 x float]], [20 x [800 x float]]* %.ls12, i64 0, i64 %fusion.4.indvar_address.dim.0.031.ls1, i64 184
  %678 = bitcast float* %677 to <8 x float>*
  %wide.load129.5.ls1 = load <8 x float>, <8 x float>* %678, align 16, !alias.scope !59, !noalias !23
  %679 = fadd fast <8 x float> %wide.load126.5.ls1, %wide.load130.5.ls1
  %680 = fadd fast <8 x float> %wide.load127.5.ls1, %wide.load131.5.ls1
  %681 = fadd fast <8 x float> %wide.load128.5.ls1, %wide.load132.5.ls1
  %682 = fadd fast <8 x float> %wide.load129.5.ls1, %wide.load133.5.ls1
  %683 = fmul fast <8 x float> %679, <float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01>
  %684 = fmul fast <8 x float> %680, <float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01>
  %685 = fmul fast <8 x float> %681, <float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01>
  %686 = fmul fast <8 x float> %682, <float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01>
  %687 = fcmp fast uge <8 x float> %683, <float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00>
  %688 = select <8 x i1> %687, <8 x float> %683, <8 x float> <float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00>
  %689 = fcmp fast ule <8 x float> %688, <float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00>
  %690 = select <8 x i1> %689, <8 x float> %688, <8 x float> <float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00>
  %691 = fmul fast <8 x float> %690, %690
  %692 = fmul fast <8 x float> %691, <float 0xBCB3E4B800000000, float 0xBCB3E4B800000000, float 0xBCB3E4B800000000, float 0xBCB3E4B800000000, float 0xBCB3E4B800000000, float 0xBCB3E4B800000000, float 0xBCB3E4B800000000, float 0xBCB3E4B800000000>
  %693 = fadd fast <8 x float> %692, <float 0x3D4C266FC0000000, float 0x3D4C266FC0000000, float 0x3D4C266FC0000000, float 0x3D4C266FC0000000, float 0x3D4C266FC0000000, float 0x3D4C266FC0000000, float 0x3D4C266FC0000000, float 0x3D4C266FC0000000>
  %694 = fmul fast <8 x float> %691, %693
  %695 = fadd fast <8 x float> %694, <float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000>
  %696 = fmul fast <8 x float> %691, %695
  %697 = fadd fast <8 x float> %696, <float 0x3E6B800820000000, float 0x3E6B800820000000, float 0x3E6B800820000000, float 0x3E6B800820000000, float 0x3E6B800820000000, float 0x3E6B800820000000, float 0x3E6B800820000000, float 0x3E6B800820000000>
  %698 = fmul fast <8 x float> %691, %697
  %699 = fadd fast <8 x float> %698, <float 0x3EEF286940000000, float 0x3EEF286940000000, float 0x3EEF286940000000, float 0x3EEF286940000000, float 0x3EEF286940000000, float 0x3EEF286940000000, float 0x3EEF286940000000, float 0x3EEF286940000000>
  %700 = fmul fast <8 x float> %691, %699
  %701 = fadd fast <8 x float> %700, <float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000>
  %702 = fmul fast <8 x float> %691, %701
  %703 = fadd fast <8 x float> %702, <float 0x3F740B3B80000000, float 0x3F740B3B80000000, float 0x3F740B3B80000000, float 0x3F740B3B80000000, float 0x3F740B3B80000000, float 0x3F740B3B80000000, float 0x3F740B3B80000000, float 0x3F740B3B80000000>
  %704 = fmul fast <8 x float> %690, %703
  %705 = fmul fast <8 x float> %691, <float 0x3EB41A7B00000000, float 0x3EB41A7B00000000, float 0x3EB41A7B00000000, float 0x3EB41A7B00000000, float 0x3EB41A7B00000000, float 0x3EB41A7B00000000, float 0x3EB41A7B00000000, float 0x3EB41A7B00000000>
  %706 = fadd fast <8 x float> %705, <float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000>
  %707 = fmul fast <8 x float> %691, %706
  %708 = fadd fast <8 x float> %707, <float 0x3F629540A0000000, float 0x3F629540A0000000, float 0x3F629540A0000000, float 0x3F629540A0000000, float 0x3F629540A0000000, float 0x3F629540A0000000, float 0x3F629540A0000000, float 0x3F629540A0000000>
  %709 = fmul fast <8 x float> %691, %708
  %710 = fadd fast <8 x float> %709, <float 0x3F740B3BA0000000, float 0x3F740B3BA0000000, float 0x3F740B3BA0000000, float 0x3F740B3BA0000000, float 0x3F740B3BA0000000, float 0x3F740B3BA0000000, float 0x3F740B3BA0000000, float 0x3F740B3BA0000000>
  %711 = fdiv fast <8 x float> %704, %710
  %712 = fcmp fast uge <8 x float> %684, <float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00>
  %713 = select <8 x i1> %712, <8 x float> %684, <8 x float> <float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00>
  %714 = fcmp fast ule <8 x float> %713, <float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00>
  %715 = select <8 x i1> %714, <8 x float> %713, <8 x float> <float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00>
  %716 = fmul fast <8 x float> %715, %715
  %717 = fmul fast <8 x float> %716, <float 0xBCB3E4B800000000, float 0xBCB3E4B800000000, float 0xBCB3E4B800000000, float 0xBCB3E4B800000000, float 0xBCB3E4B800000000, float 0xBCB3E4B800000000, float 0xBCB3E4B800000000, float 0xBCB3E4B800000000>
  %718 = fadd fast <8 x float> %717, <float 0x3D4C266FC0000000, float 0x3D4C266FC0000000, float 0x3D4C266FC0000000, float 0x3D4C266FC0000000, float 0x3D4C266FC0000000, float 0x3D4C266FC0000000, float 0x3D4C266FC0000000, float 0x3D4C266FC0000000>
  %719 = fmul fast <8 x float> %716, %718
  %720 = fadd fast <8 x float> %719, <float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000>
  %721 = fmul fast <8 x float> %716, %720
  %722 = fadd fast <8 x float> %721, <float 0x3E6B800820000000, float 0x3E6B800820000000, float 0x3E6B800820000000, float 0x3E6B800820000000, float 0x3E6B800820000000, float 0x3E6B800820000000, float 0x3E6B800820000000, float 0x3E6B800820000000>
  %723 = fmul fast <8 x float> %716, %722
  %724 = fadd fast <8 x float> %723, <float 0x3EEF286940000000, float 0x3EEF286940000000, float 0x3EEF286940000000, float 0x3EEF286940000000, float 0x3EEF286940000000, float 0x3EEF286940000000, float 0x3EEF286940000000, float 0x3EEF286940000000>
  %725 = fmul fast <8 x float> %716, %724
  %726 = fadd fast <8 x float> %725, <float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000>
  %727 = fmul fast <8 x float> %716, %726
  %728 = fadd fast <8 x float> %727, <float 0x3F740B3B80000000, float 0x3F740B3B80000000, float 0x3F740B3B80000000, float 0x3F740B3B80000000, float 0x3F740B3B80000000, float 0x3F740B3B80000000, float 0x3F740B3B80000000, float 0x3F740B3B80000000>
  %729 = fmul fast <8 x float> %715, %728
  %730 = fmul fast <8 x float> %716, <float 0x3EB41A7B00000000, float 0x3EB41A7B00000000, float 0x3EB41A7B00000000, float 0x3EB41A7B00000000, float 0x3EB41A7B00000000, float 0x3EB41A7B00000000, float 0x3EB41A7B00000000, float 0x3EB41A7B00000000>
  %731 = fadd fast <8 x float> %730, <float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000>
  %732 = fmul fast <8 x float> %716, %731
  %733 = fadd fast <8 x float> %732, <float 0x3F629540A0000000, float 0x3F629540A0000000, float 0x3F629540A0000000, float 0x3F629540A0000000, float 0x3F629540A0000000, float 0x3F629540A0000000, float 0x3F629540A0000000, float 0x3F629540A0000000>
  %734 = fmul fast <8 x float> %716, %733
  %735 = fadd fast <8 x float> %734, <float 0x3F740B3BA0000000, float 0x3F740B3BA0000000, float 0x3F740B3BA0000000, float 0x3F740B3BA0000000, float 0x3F740B3BA0000000, float 0x3F740B3BA0000000, float 0x3F740B3BA0000000, float 0x3F740B3BA0000000>
  %736 = fdiv fast <8 x float> %729, %735
  %737 = fcmp fast uge <8 x float> %685, <float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00>
  %738 = select <8 x i1> %737, <8 x float> %685, <8 x float> <float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00>
  %739 = fcmp fast ule <8 x float> %738, <float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00>
  %740 = select <8 x i1> %739, <8 x float> %738, <8 x float> <float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00>
  %741 = fmul fast <8 x float> %740, %740
  %742 = fmul fast <8 x float> %741, <float 0xBCB3E4B800000000, float 0xBCB3E4B800000000, float 0xBCB3E4B800000000, float 0xBCB3E4B800000000, float 0xBCB3E4B800000000, float 0xBCB3E4B800000000, float 0xBCB3E4B800000000, float 0xBCB3E4B800000000>
  %743 = fadd fast <8 x float> %742, <float 0x3D4C266FC0000000, float 0x3D4C266FC0000000, float 0x3D4C266FC0000000, float 0x3D4C266FC0000000, float 0x3D4C266FC0000000, float 0x3D4C266FC0000000, float 0x3D4C266FC0000000, float 0x3D4C266FC0000000>
  %744 = fmul fast <8 x float> %741, %743
  %745 = fadd fast <8 x float> %744, <float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000>
  %746 = fmul fast <8 x float> %741, %745
  %747 = fadd fast <8 x float> %746, <float 0x3E6B800820000000, float 0x3E6B800820000000, float 0x3E6B800820000000, float 0x3E6B800820000000, float 0x3E6B800820000000, float 0x3E6B800820000000, float 0x3E6B800820000000, float 0x3E6B800820000000>
  %748 = fmul fast <8 x float> %741, %747
  %749 = fadd fast <8 x float> %748, <float 0x3EEF286940000000, float 0x3EEF286940000000, float 0x3EEF286940000000, float 0x3EEF286940000000, float 0x3EEF286940000000, float 0x3EEF286940000000, float 0x3EEF286940000000, float 0x3EEF286940000000>
  %750 = fmul fast <8 x float> %741, %749
  %751 = fadd fast <8 x float> %750, <float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000>
  %752 = fmul fast <8 x float> %741, %751
  %753 = fadd fast <8 x float> %752, <float 0x3F740B3B80000000, float 0x3F740B3B80000000, float 0x3F740B3B80000000, float 0x3F740B3B80000000, float 0x3F740B3B80000000, float 0x3F740B3B80000000, float 0x3F740B3B80000000, float 0x3F740B3B80000000>
  %754 = fmul fast <8 x float> %740, %753
  %755 = fmul fast <8 x float> %741, <float 0x3EB41A7B00000000, float 0x3EB41A7B00000000, float 0x3EB41A7B00000000, float 0x3EB41A7B00000000, float 0x3EB41A7B00000000, float 0x3EB41A7B00000000, float 0x3EB41A7B00000000, float 0x3EB41A7B00000000>
  %756 = fadd fast <8 x float> %755, <float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000>
  %757 = fmul fast <8 x float> %741, %756
  %758 = fadd fast <8 x float> %757, <float 0x3F629540A0000000, float 0x3F629540A0000000, float 0x3F629540A0000000, float 0x3F629540A0000000, float 0x3F629540A0000000, float 0x3F629540A0000000, float 0x3F629540A0000000, float 0x3F629540A0000000>
  %759 = fmul fast <8 x float> %741, %758
  %760 = fadd fast <8 x float> %759, <float 0x3F740B3BA0000000, float 0x3F740B3BA0000000, float 0x3F740B3BA0000000, float 0x3F740B3BA0000000, float 0x3F740B3BA0000000, float 0x3F740B3BA0000000, float 0x3F740B3BA0000000, float 0x3F740B3BA0000000>
  %761 = fdiv fast <8 x float> %754, %760
  %762 = fcmp fast uge <8 x float> %686, <float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00>
  %763 = select <8 x i1> %762, <8 x float> %686, <8 x float> <float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00>
  %764 = fcmp fast ule <8 x float> %763, <float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00>
  %765 = select <8 x i1> %764, <8 x float> %763, <8 x float> <float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00>
  %766 = fmul fast <8 x float> %765, %765
  %767 = fmul fast <8 x float> %766, <float 0xBCB3E4B800000000, float 0xBCB3E4B800000000, float 0xBCB3E4B800000000, float 0xBCB3E4B800000000, float 0xBCB3E4B800000000, float 0xBCB3E4B800000000, float 0xBCB3E4B800000000, float 0xBCB3E4B800000000>
  %768 = fadd fast <8 x float> %767, <float 0x3D4C266FC0000000, float 0x3D4C266FC0000000, float 0x3D4C266FC0000000, float 0x3D4C266FC0000000, float 0x3D4C266FC0000000, float 0x3D4C266FC0000000, float 0x3D4C266FC0000000, float 0x3D4C266FC0000000>
  %769 = fmul fast <8 x float> %766, %768
  %770 = fadd fast <8 x float> %769, <float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000>
  %771 = fmul fast <8 x float> %766, %770
  %772 = fadd fast <8 x float> %771, <float 0x3E6B800820000000, float 0x3E6B800820000000, float 0x3E6B800820000000, float 0x3E6B800820000000, float 0x3E6B800820000000, float 0x3E6B800820000000, float 0x3E6B800820000000, float 0x3E6B800820000000>
  %773 = fmul fast <8 x float> %766, %772
  %774 = fadd fast <8 x float> %773, <float 0x3EEF286940000000, float 0x3EEF286940000000, float 0x3EEF286940000000, float 0x3EEF286940000000, float 0x3EEF286940000000, float 0x3EEF286940000000, float 0x3EEF286940000000, float 0x3EEF286940000000>
  %775 = fmul fast <8 x float> %766, %774
  %776 = fadd fast <8 x float> %775, <float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000>
  %777 = fmul fast <8 x float> %766, %776
  %778 = fadd fast <8 x float> %777, <float 0x3F740B3B80000000, float 0x3F740B3B80000000, float 0x3F740B3B80000000, float 0x3F740B3B80000000, float 0x3F740B3B80000000, float 0x3F740B3B80000000, float 0x3F740B3B80000000, float 0x3F740B3B80000000>
  %779 = fmul fast <8 x float> %765, %778
  %780 = fmul fast <8 x float> %766, <float 0x3EB41A7B00000000, float 0x3EB41A7B00000000, float 0x3EB41A7B00000000, float 0x3EB41A7B00000000, float 0x3EB41A7B00000000, float 0x3EB41A7B00000000, float 0x3EB41A7B00000000, float 0x3EB41A7B00000000>
  %781 = fadd fast <8 x float> %780, <float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000>
  %782 = fmul fast <8 x float> %766, %781
  %783 = fadd fast <8 x float> %782, <float 0x3F629540A0000000, float 0x3F629540A0000000, float 0x3F629540A0000000, float 0x3F629540A0000000, float 0x3F629540A0000000, float 0x3F629540A0000000, float 0x3F629540A0000000, float 0x3F629540A0000000>
  %784 = fmul fast <8 x float> %766, %783
  %785 = fadd fast <8 x float> %784, <float 0x3F740B3BA0000000, float 0x3F740B3BA0000000, float 0x3F740B3BA0000000, float 0x3F740B3BA0000000, float 0x3F740B3BA0000000, float 0x3F740B3BA0000000, float 0x3F740B3BA0000000, float 0x3F740B3BA0000000>
  %786 = fdiv fast <8 x float> %779, %785
  %787 = getelementptr inbounds [20 x [200 x float]], [20 x [200 x float]]* %fusion.4.ls1, i64 0, i64 %fusion.4.indvar_address.dim.0.031.ls1, i64 160
  %788 = bitcast float* %787 to <8 x float>*
  store <8 x float> %711, <8 x float>* %788, align 16, !alias.scope !62, !noalias !64
  %789 = getelementptr inbounds [20 x [200 x float]], [20 x [200 x float]]* %fusion.4.ls1, i64 0, i64 %fusion.4.indvar_address.dim.0.031.ls1, i64 168
  %790 = bitcast float* %789 to <8 x float>*
  store <8 x float> %736, <8 x float>* %790, align 16, !alias.scope !62, !noalias !64
  %791 = getelementptr inbounds [20 x [200 x float]], [20 x [200 x float]]* %fusion.4.ls1, i64 0, i64 %fusion.4.indvar_address.dim.0.031.ls1, i64 176
  %792 = bitcast float* %791 to <8 x float>*
  store <8 x float> %761, <8 x float>* %792, align 16, !alias.scope !62, !noalias !64
  %793 = getelementptr inbounds [20 x [200 x float]], [20 x [200 x float]]* %fusion.4.ls1, i64 0, i64 %fusion.4.indvar_address.dim.0.031.ls1, i64 184
  %794 = bitcast float* %793 to <8 x float>*
  store <8 x float> %786, <8 x float>* %794, align 16, !alias.scope !62, !noalias !64
  br label %fusion.4.loop_body.dim.1.preheader.ls1

fusion.4.loop_body.dim.1.preheader.ls1:           ; preds = %vector.body94.ls1, %fusion.4.loop_detach.dim.0.ls1
  %fusion.4.indvar_address.dim.1.030.ph.ls1 = phi i64 [ 0, %fusion.4.loop_detach.dim.0.ls1 ], [ 192, %vector.body94.ls1 ]
  br label %fusion.4.loop_body.dim.1.ls1

fusion.4.loop_body.dim.1.ls1:                     ; preds = %fusion.4.loop_body.dim.1.ls1, %fusion.4.loop_body.dim.1.preheader.ls1
  %fusion.4.indvar_address.dim.1.030.ls1 = phi i64 [ %indvar.inc8.ls1, %fusion.4.loop_body.dim.1.ls1 ], [ %fusion.4.indvar_address.dim.1.030.ph.ls1, %fusion.4.loop_body.dim.1.preheader.ls1 ]
  %795 = getelementptr inbounds [20 x [800 x float]], [20 x [800 x float]]* %.ls12, i64 0, i64 %fusion.4.indvar_address.dim.0.031.ls1, i64 %fusion.4.indvar_address.dim.1.030.ls1
  %796 = load float, float* %795, align 4, !alias.scope !29, !noalias !23
  %797 = getelementptr inbounds [800 x float], [800 x float]* %.ls11, i64 0, i64 %fusion.4.indvar_address.dim.1.030.ls1
  %798 = load float, float* %797, align 4, !invariant.load !0, !noalias !30
  %799 = fadd fast float %798, %796
  %800 = fmul fast float %799, 5.000000e-01
  %801 = tail call fast float @tanhf(float %800)
  %802 = getelementptr inbounds [20 x [200 x float]], [20 x [200 x float]]* %fusion.4.ls1, i64 0, i64 %fusion.4.indvar_address.dim.0.031.ls1, i64 %fusion.4.indvar_address.dim.1.030.ls1
  store float %801, float* %802, align 4, !alias.scope !51, !noalias !52
  %indvar.inc8.ls1 = add nuw nsw i64 %fusion.4.indvar_address.dim.1.030.ls1, 1
  %exitcond50.ls1 = icmp eq i64 %indvar.inc8.ls1, 200
  br i1 %exitcond50.ls1, label %fusion.4.loop_inc.dim.0.ls1, label %fusion.4.loop_body.dim.1.ls1, !llvm.loop !66

fusion.4.loop_inc.dim.0.ls1:                      ; preds = %fusion.4.loop_body.dim.1.ls1
  %indvar.inc7.ls1 = add nuw nsw i64 %fusion.4.indvar_address.dim.0.031.ls1, 1
  %exitcond51.ls1 = icmp eq i64 %indvar.inc7.ls1, %end.ls1
  br i1 %exitcond51.ls1, label %fusion.4.loop_sync.dim.0.ls1, label %fusion.4.loop_detach.dim.0.ls1, !llvm.loop !67
}

; CHECK: .Lcluster_25__XlaCompiledKernel_true__XlaNumConstantArgs_1__XlaNumResourceArgs_0_.42.outline_fusion.4.loop_detach.dim.0.ls1,@function
; CHECK: stmxcsr
; CHECK: movq %rbp
; CHECK: movq %rsp
; CHECK: movl $1, %eax
; CHECK: testl %eax, %eax
; CHECK: jne
; CHECK: subq $672, %rsp
; CHECK: callq .Lcluster_25__XlaCompiledKernel_true__XlaNumConstantArgs_1__XlaNumResourceArgs_0_.42.outline_fusion.4.loop_detach.dim.0.ls1.outline_.split.otd1

; Function Attrs: argmemonly nounwind stealable
define private fastcc void @cluster_25__XlaCompiledKernel_true__XlaNumConstantArgs_1__XlaNumResourceArgs_0_.42.outline_fusion.2.loop_detach.dim.0.ls1(i64 %fusion.2.indvar_address.dim.0.034.start.ls1, i64 %end.ls1, i64 %grainsize.ls1, <8 x float> %wide.load91.2.ls1, <8 x float> %wide.load92.5.ls1, <8 x float> %wide.load91.5.ls1, <8 x float> %wide.load90.5.ls1, <8 x float> %wide.load93.4.ls1, <8 x float> %wide.load92.4.ls1, <8 x float> %wide.load91.4.ls1, <8 x float> %wide.load90.4.ls1, <8 x float> %wide.load93.3.ls1, <8 x float> %wide.load92.3.ls1, <8 x float> %wide.load91.3.ls1, <8 x float> %wide.load90.3.ls1, <8 x float> %wide.load93.2.ls1, <8 x float> %wide.load92.2.ls1, <8 x float> %wide.load90.2.ls1, <8 x float> %wide.load93.1.ls1, <8 x float> %wide.load92.1.ls1, <8 x float> %wide.load91.1.ls1, <8 x float> %wide.load90.1.ls1, <8 x float> %wide.load93.5.ls1, <8 x float> %wide.load93.ls1, <8 x float> %wide.load92.ls1, <8 x float> %wide.load91.ls1, <8 x float> %wide.load90.ls1, i8* readnone align 16 %.ls1, [800 x float]* nocapture readonly align 16 %.ls11, i8* readnone align 16 %scevgep7374.ls1, [20 x [200 x float]]* align 16 %fusion.2.ls1, [20 x [800 x float]]* readonly align 16 %.ls12, i8* readnone align 16 %scevgep7172.ls1) unnamed_addr #4 {
dot.10.loop_exit.lhs.0.ls1:
  %__cilkrts_sf = alloca %struct.__cilkrts_stack_frame, align 8
  %0 = call %struct.__cilkrts_worker* @__cilkrts_get_tls_worker() #6
  %1 = icmp eq %struct.__cilkrts_worker* %0, null
  br i1 %1, label %slowpath.i, label %fastpath.i

slowpath.i:                                       ; preds = %dot.10.loop_exit.lhs.0.ls1
  %2 = call %struct.__cilkrts_worker* @__cilkrts_bind_thread_1() #6
  %3 = getelementptr inbounds %struct.__cilkrts_stack_frame, %struct.__cilkrts_stack_frame* %__cilkrts_sf, i64 0, i32 0
  store atomic i32 16777344, i32* %3 release, align 8
  br label %__cilkrts_enter_frame_1.exit

fastpath.i:                                       ; preds = %dot.10.loop_exit.lhs.0.ls1
  %4 = getelementptr inbounds %struct.__cilkrts_stack_frame, %struct.__cilkrts_stack_frame* %__cilkrts_sf, i64 0, i32 0
  store atomic i32 16777216, i32* %4 release, align 8
  br label %__cilkrts_enter_frame_1.exit

__cilkrts_enter_frame_1.exit:                     ; preds = %slowpath.i, %fastpath.i
  %5 = phi %struct.__cilkrts_worker* [ %2, %slowpath.i ], [ %0, %fastpath.i ]
  %6 = getelementptr inbounds %struct.__cilkrts_worker, %struct.__cilkrts_worker* %5, i64 0, i32 9
  %7 = load atomic %struct.__cilkrts_stack_frame*, %struct.__cilkrts_stack_frame** %6 acquire, align 8
  %8 = getelementptr inbounds %struct.__cilkrts_stack_frame, %struct.__cilkrts_stack_frame* %__cilkrts_sf, i64 0, i32 2
  store atomic %struct.__cilkrts_stack_frame* %7, %struct.__cilkrts_stack_frame** %8 release, align 8
  %9 = getelementptr inbounds %struct.__cilkrts_stack_frame, %struct.__cilkrts_stack_frame* %__cilkrts_sf, i64 0, i32 3
  store atomic %struct.__cilkrts_worker* %5, %struct.__cilkrts_worker** %9 release, align 8
  store atomic %struct.__cilkrts_stack_frame* %__cilkrts_sf, %struct.__cilkrts_stack_frame** %6 release, align 8
  %itercount3 = sub i64 %end.ls1, %fusion.2.indvar_address.dim.0.034.start.ls1
  %10 = icmp ugt i64 %itercount3, %grainsize.ls1
  br i1 %10, label %.lr.ph.preheader, label %fusion.2.loop_detach.dim.0.ls1.preheader

.lr.ph.preheader:                                 ; preds = %__cilkrts_enter_frame_1.exit
  %11 = getelementptr inbounds %struct.__cilkrts_stack_frame, %struct.__cilkrts_stack_frame* %__cilkrts_sf, i64 0, i32 6
  %12 = getelementptr inbounds %struct.__cilkrts_stack_frame, %struct.__cilkrts_stack_frame* %__cilkrts_sf, i64 0, i32 7
  %13 = getelementptr inbounds %struct.__cilkrts_stack_frame, %struct.__cilkrts_stack_frame* %__cilkrts_sf, i64 0, i32 5
  %14 = getelementptr inbounds [5 x i8*], [5 x i8*]* %13, i64 0, i64 0
  %15 = getelementptr inbounds %struct.__cilkrts_stack_frame, %struct.__cilkrts_stack_frame* %__cilkrts_sf, i64 0, i32 5, i64 2
  %16 = bitcast [5 x i8*]* %13 to i8*
  br label %.lr.ph

fusion.2.loop_detach.dim.0.ls1.preheader:         ; preds = %.split.split, %__cilkrts_enter_frame_1.exit
  %fusion.2.indvar_address.dim.0.034.ls1.dac.lcssa = phi i64 [ %fusion.2.indvar_address.dim.0.034.start.ls1, %__cilkrts_enter_frame_1.exit ], [ %miditer, %.split.split ]
  br label %fusion.2.loop_detach.dim.0.ls1

.lr.ph:                                           ; preds = %.lr.ph.preheader, %.split.split
  %itercount5 = phi i64 [ %itercount, %.split.split ], [ %itercount3, %.lr.ph.preheader ]
  %fusion.2.indvar_address.dim.0.034.ls1.dac4 = phi i64 [ %miditer, %.split.split ], [ %fusion.2.indvar_address.dim.0.034.start.ls1, %.lr.ph.preheader ]
  %halfcount = lshr i64 %itercount5, 1
  %miditer = add nuw nsw i64 %fusion.2.indvar_address.dim.0.034.ls1.dac4, %halfcount
  call void asm sideeffect "stmxcsr $0\0A\09fnstcw $1", "*m,*m,~{dirflag},~{fpsr},~{flags}"(i32* nonnull %11, i16* nonnull %12) #6
  %17 = call i8* @llvm.frameaddress(i32 0)
  store volatile i8* %17, i8** %14, align 8
  %18 = call i8* @llvm.stacksave()
  store volatile i8* %18, i8** %15, align 8
  %19 = call i32 @llvm.eh.sjlj.setjmp(i8* nonnull %16) #7
  %20 = icmp eq i32 %19, 0
  br i1 %20, label %.lr.ph.split, label %.split.split

.lr.ph.split:                                     ; preds = %.lr.ph
  call fastcc void @cluster_25__XlaCompiledKernel_true__XlaNumConstantArgs_1__XlaNumResourceArgs_0_.42.outline_fusion.2.loop_detach.dim.0.ls1.outline_.split.otd1(<8 x float> %wide.load91.3.ls1, <8 x float> %wide.load90.ls1, <8 x float> %wide.load91.ls1, <8 x float> %wide.load92.ls1, <8 x float> %wide.load93.ls1, <8 x float> %wide.load93.5.ls1, <8 x float> %wide.load90.1.ls1, <8 x float> %wide.load91.1.ls1, <8 x float> %wide.load92.1.ls1, <8 x float> %wide.load93.1.ls1, <8 x float> %wide.load90.2.ls1, <8 x float> %wide.load92.2.ls1, <8 x float> %wide.load93.2.ls1, <8 x float> %wide.load90.3.ls1, <8 x float> %wide.load92.3.ls1, <8 x float> %wide.load93.3.ls1, <8 x float> %wide.load90.4.ls1, <8 x float> %wide.load91.4.ls1, <8 x float> %wide.load92.4.ls1, <8 x float> %wide.load93.4.ls1, <8 x float> %wide.load90.5.ls1, <8 x float> %wide.load91.5.ls1, <8 x float> %wide.load92.5.ls1, <8 x float> %wide.load91.2.ls1, i8* %scevgep7374.ls1, [20 x [200 x float]]* %fusion.2.ls1, i8* %scevgep7172.ls1, [20 x [800 x float]]* %.ls12, i64 %fusion.2.indvar_address.dim.0.034.ls1.dac4, [800 x float]* %.ls11, i8* %.ls1, i64 %grainsize.ls1, i64 %miditer) #6
  br label %.split.split

.split.split:                                     ; preds = %.lr.ph, %.lr.ph.split
  %itercount = sub i64 %end.ls1, %miditer
  %21 = icmp ugt i64 %itercount, %grainsize.ls1
  br i1 %21, label %.lr.ph, label %fusion.2.loop_detach.dim.0.ls1.preheader

fusion.2.loop_sync.dim.0.ls1:                     ; preds = %fusion.2.loop_inc.dim.0.ls1
  %22 = getelementptr inbounds %struct.__cilkrts_stack_frame, %struct.__cilkrts_stack_frame* %__cilkrts_sf, i64 0, i32 0
  %23 = load atomic i32, i32* %22 acquire, align 8
  %24 = and i32 %23, 2
  %25 = icmp eq i32 %24, 0
  br i1 %25, label %__cilk_sync_nothrow.exit, label %cilk.sync.savestate.i

cilk.sync.savestate.i:                            ; preds = %fusion.2.loop_sync.dim.0.ls1
  %26 = load atomic %struct.__cilkrts_worker*, %struct.__cilkrts_worker** %9 acquire, align 8
  %.elt = getelementptr inbounds %struct.__cilkrts_worker, %struct.__cilkrts_worker* %26, i64 0, i32 12, i32 0
  %.unpack = load i64, i64* %.elt, align 8
  %.elt8 = getelementptr inbounds %struct.__cilkrts_worker, %struct.__cilkrts_worker* %26, i64 0, i32 12, i32 1
  %27 = bitcast %struct.__cilkrts_pedigree** %.elt8 to i64*
  %.unpack910 = load i64, i64* %27, align 8
  %.fca.0.0.gep = getelementptr inbounds %struct.__cilkrts_stack_frame, %struct.__cilkrts_stack_frame* %__cilkrts_sf, i64 0, i32 9, i32 0, i32 0
  %.fca.0.1.gep = getelementptr inbounds %struct.__cilkrts_stack_frame, %struct.__cilkrts_stack_frame* %__cilkrts_sf, i64 0, i32 9, i32 0, i32 1
  store i64 %.unpack, i64* %.fca.0.0.gep, align 8
  %28 = bitcast %struct.__cilkrts_pedigree** %.fca.0.1.gep to i64*
  store i64 %.unpack910, i64* %28, align 8
  %29 = getelementptr inbounds %struct.__cilkrts_stack_frame, %struct.__cilkrts_stack_frame* %__cilkrts_sf, i64 0, i32 6
  %30 = getelementptr inbounds %struct.__cilkrts_stack_frame, %struct.__cilkrts_stack_frame* %__cilkrts_sf, i64 0, i32 7
  call void asm sideeffect "stmxcsr $0\0A\09fnstcw $1", "*m,*m,~{dirflag},~{fpsr},~{flags}"(i32* nonnull %29, i16* nonnull %30) #6
  %31 = getelementptr inbounds %struct.__cilkrts_stack_frame, %struct.__cilkrts_stack_frame* %__cilkrts_sf, i64 0, i32 5
  %32 = call i8* @llvm.frameaddress(i32 0) #6
  %33 = getelementptr inbounds [5 x i8*], [5 x i8*]* %31, i64 0, i64 0
  store volatile i8* %32, i8** %33, align 8
  %34 = call i8* @llvm.stacksave() #6
  %35 = getelementptr inbounds %struct.__cilkrts_stack_frame, %struct.__cilkrts_stack_frame* %__cilkrts_sf, i64 0, i32 5, i64 2
  store volatile i8* %34, i8** %35, align 8
  %36 = bitcast [5 x i8*]* %31 to i8*
  %37 = call i32 @llvm.eh.sjlj.setjmp(i8* nonnull %36) #8
  %38 = icmp eq i32 %37, 0
  br i1 %38, label %cilk.sync.runtimecall.i, label %__cilk_sync_nothrow.exit

cilk.sync.runtimecall.i:                          ; preds = %cilk.sync.savestate.i
  call void @__cilkrts_sync(%struct.__cilkrts_stack_frame* nonnull %__cilkrts_sf) #6
  br label %__cilk_sync_nothrow.exit

__cilk_sync_nothrow.exit:                         ; preds = %fusion.2.loop_sync.dim.0.ls1, %cilk.sync.savestate.i, %cilk.sync.runtimecall.i
  %39 = load atomic %struct.__cilkrts_worker*, %struct.__cilkrts_worker** %9 acquire, align 8
  %40 = getelementptr inbounds %struct.__cilkrts_worker, %struct.__cilkrts_worker* %39, i64 0, i32 12, i32 0
  %41 = load i64, i64* %40, align 8
  %42 = add i64 %41, 1
  store i64 %42, i64* %40, align 8
  %43 = load atomic %struct.__cilkrts_worker*, %struct.__cilkrts_worker** %9 acquire, align 8
  %44 = bitcast %struct.__cilkrts_stack_frame** %8 to i64*
  %45 = load i64, i64* %44, align 8
  %46 = getelementptr inbounds %struct.__cilkrts_worker, %struct.__cilkrts_worker* %43, i64 0, i32 9
  %47 = bitcast %struct.__cilkrts_stack_frame** %46 to i64*
  store atomic i64 %45, i64* %47 release, align 8
  store atomic %struct.__cilkrts_stack_frame* null, %struct.__cilkrts_stack_frame** %8 release, align 8
  %48 = load atomic i32, i32* %22 acquire, align 8
  %49 = icmp eq i32 %48, 16777216
  br i1 %49, label %__cilk_parent_epilogue.exit, label %body.i

body.i:                                           ; preds = %__cilk_sync_nothrow.exit
  call void @__cilkrts_leave_frame(%struct.__cilkrts_stack_frame* nonnull %__cilkrts_sf) #6
  br label %__cilk_parent_epilogue.exit

__cilk_parent_epilogue.exit:                      ; preds = %__cilk_sync_nothrow.exit, %body.i
  ret void

fusion.2.loop_detach.dim.0.ls1:                   ; preds = %fusion.2.loop_detach.dim.0.ls1.preheader, %fusion.2.loop_inc.dim.0.ls1
  %fusion.2.indvar_address.dim.0.034.ls1 = phi i64 [ %indvar.inc5.ls1, %fusion.2.loop_inc.dim.0.ls1 ], [ %fusion.2.indvar_address.dim.0.034.ls1.dac.lcssa, %fusion.2.loop_detach.dim.0.ls1.preheader ]
  %50 = mul nuw nsw i64 %fusion.2.indvar_address.dim.0.034.ls1, 800
  %scevgep.ls1 = getelementptr i8, i8* %.ls1, i64 %50
  %51 = add nuw nsw i64 %50, 800
  %scevgep66.ls1 = getelementptr i8, i8* %.ls1, i64 %51
  %scevgep67.ls1 = getelementptr [20 x [800 x float]], [20 x [800 x float]]* %.ls12, i64 0, i64 %fusion.2.indvar_address.dim.0.034.ls1, i64 200
  %scevgep6768.ls1 = bitcast float* %scevgep67.ls1 to i8*
  %scevgep69.ls1 = getelementptr [20 x [800 x float]], [20 x [800 x float]]* %.ls12, i64 0, i64 %fusion.2.indvar_address.dim.0.034.ls1, i64 400
  %scevgep6970.ls1 = bitcast float* %scevgep69.ls1 to i8*
  %bound0.ls1 = icmp ult i8* %scevgep.ls1, %scevgep6970.ls1
  %bound1.ls1 = icmp ugt i8* %scevgep66.ls1, %scevgep6768.ls1
  %found.conflict.ls1 = and i1 %bound0.ls1, %bound1.ls1
  %bound075.ls1 = icmp ult i8* %scevgep.ls1, %scevgep7374.ls1
  %bound176.ls1 = icmp ugt i8* %scevgep66.ls1, %scevgep7172.ls1
  %found.conflict77.ls1 = and i1 %bound075.ls1, %bound176.ls1
  %conflict.rdx.ls1 = or i1 %found.conflict.ls1, %found.conflict77.ls1
  br i1 %conflict.rdx.ls1, label %fusion.2.loop_body.dim.1.preheader.ls1, label %vector.body62.ls1

vector.body62.ls1:                                ; preds = %fusion.2.loop_detach.dim.0.ls1
  %52 = bitcast float* %scevgep67.ls1 to <8 x float>*
  %wide.load86.ls1 = load <8 x float>, <8 x float>* %52, align 16, !alias.scope !68, !noalias !23
  %53 = getelementptr inbounds [20 x [800 x float]], [20 x [800 x float]]* %.ls12, i64 0, i64 %fusion.2.indvar_address.dim.0.034.ls1, i64 208
  %54 = bitcast float* %53 to <8 x float>*
  %wide.load87.ls1 = load <8 x float>, <8 x float>* %54, align 16, !alias.scope !68, !noalias !23
  %55 = getelementptr inbounds [20 x [800 x float]], [20 x [800 x float]]* %.ls12, i64 0, i64 %fusion.2.indvar_address.dim.0.034.ls1, i64 216
  %56 = bitcast float* %55 to <8 x float>*
  %wide.load88.ls1 = load <8 x float>, <8 x float>* %56, align 16, !alias.scope !68, !noalias !23
  %57 = getelementptr inbounds [20 x [800 x float]], [20 x [800 x float]]* %.ls12, i64 0, i64 %fusion.2.indvar_address.dim.0.034.ls1, i64 224
  %58 = bitcast float* %57 to <8 x float>*
  %wide.load89.ls1 = load <8 x float>, <8 x float>* %58, align 16, !alias.scope !68, !noalias !23
  %59 = fadd fast <8 x float> %wide.load86.ls1, %wide.load90.ls1
  %60 = fadd fast <8 x float> %wide.load87.ls1, %wide.load91.ls1
  %61 = fadd fast <8 x float> %wide.load88.ls1, %wide.load92.ls1
  %62 = fadd fast <8 x float> %wide.load89.ls1, %wide.load93.ls1
  %63 = fcmp fast uge <8 x float> %59, <float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00>
  %64 = select <8 x i1> %63, <8 x float> %59, <8 x float> <float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00>
  %65 = fcmp fast ule <8 x float> %64, <float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00>
  %66 = select <8 x i1> %65, <8 x float> %64, <8 x float> <float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00>
  %67 = fmul fast <8 x float> %66, %66
  %68 = fmul fast <8 x float> %67, <float 0xBCB3E4B800000000, float 0xBCB3E4B800000000, float 0xBCB3E4B800000000, float 0xBCB3E4B800000000, float 0xBCB3E4B800000000, float 0xBCB3E4B800000000, float 0xBCB3E4B800000000, float 0xBCB3E4B800000000>
  %69 = fadd fast <8 x float> %68, <float 0x3D4C266FC0000000, float 0x3D4C266FC0000000, float 0x3D4C266FC0000000, float 0x3D4C266FC0000000, float 0x3D4C266FC0000000, float 0x3D4C266FC0000000, float 0x3D4C266FC0000000, float 0x3D4C266FC0000000>
  %70 = fmul fast <8 x float> %67, %69
  %71 = fadd fast <8 x float> %70, <float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000>
  %72 = fmul fast <8 x float> %67, %71
  %73 = fadd fast <8 x float> %72, <float 0x3E6B800820000000, float 0x3E6B800820000000, float 0x3E6B800820000000, float 0x3E6B800820000000, float 0x3E6B800820000000, float 0x3E6B800820000000, float 0x3E6B800820000000, float 0x3E6B800820000000>
  %74 = fmul fast <8 x float> %67, %73
  %75 = fadd fast <8 x float> %74, <float 0x3EEF286940000000, float 0x3EEF286940000000, float 0x3EEF286940000000, float 0x3EEF286940000000, float 0x3EEF286940000000, float 0x3EEF286940000000, float 0x3EEF286940000000, float 0x3EEF286940000000>
  %76 = fmul fast <8 x float> %67, %75
  %77 = fadd fast <8 x float> %76, <float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000>
  %78 = fmul fast <8 x float> %67, %77
  %79 = fadd fast <8 x float> %78, <float 0x3F740B3B80000000, float 0x3F740B3B80000000, float 0x3F740B3B80000000, float 0x3F740B3B80000000, float 0x3F740B3B80000000, float 0x3F740B3B80000000, float 0x3F740B3B80000000, float 0x3F740B3B80000000>
  %80 = fmul fast <8 x float> %66, %79
  %81 = fmul fast <8 x float> %67, <float 0x3EB41A7B00000000, float 0x3EB41A7B00000000, float 0x3EB41A7B00000000, float 0x3EB41A7B00000000, float 0x3EB41A7B00000000, float 0x3EB41A7B00000000, float 0x3EB41A7B00000000, float 0x3EB41A7B00000000>
  %82 = fadd fast <8 x float> %81, <float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000>
  %83 = fmul fast <8 x float> %67, %82
  %84 = fadd fast <8 x float> %83, <float 0x3F629540A0000000, float 0x3F629540A0000000, float 0x3F629540A0000000, float 0x3F629540A0000000, float 0x3F629540A0000000, float 0x3F629540A0000000, float 0x3F629540A0000000, float 0x3F629540A0000000>
  %85 = fmul fast <8 x float> %67, %84
  %86 = fadd fast <8 x float> %85, <float 0x3F740B3BA0000000, float 0x3F740B3BA0000000, float 0x3F740B3BA0000000, float 0x3F740B3BA0000000, float 0x3F740B3BA0000000, float 0x3F740B3BA0000000, float 0x3F740B3BA0000000, float 0x3F740B3BA0000000>
  %87 = fdiv fast <8 x float> %80, %86
  %88 = fcmp fast uge <8 x float> %60, <float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00>
  %89 = select <8 x i1> %88, <8 x float> %60, <8 x float> <float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00>
  %90 = fcmp fast ule <8 x float> %89, <float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00>
  %91 = select <8 x i1> %90, <8 x float> %89, <8 x float> <float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00>
  %92 = fmul fast <8 x float> %91, %91
  %93 = fmul fast <8 x float> %92, <float 0xBCB3E4B800000000, float 0xBCB3E4B800000000, float 0xBCB3E4B800000000, float 0xBCB3E4B800000000, float 0xBCB3E4B800000000, float 0xBCB3E4B800000000, float 0xBCB3E4B800000000, float 0xBCB3E4B800000000>
  %94 = fadd fast <8 x float> %93, <float 0x3D4C266FC0000000, float 0x3D4C266FC0000000, float 0x3D4C266FC0000000, float 0x3D4C266FC0000000, float 0x3D4C266FC0000000, float 0x3D4C266FC0000000, float 0x3D4C266FC0000000, float 0x3D4C266FC0000000>
  %95 = fmul fast <8 x float> %92, %94
  %96 = fadd fast <8 x float> %95, <float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000>
  %97 = fmul fast <8 x float> %92, %96
  %98 = fadd fast <8 x float> %97, <float 0x3E6B800820000000, float 0x3E6B800820000000, float 0x3E6B800820000000, float 0x3E6B800820000000, float 0x3E6B800820000000, float 0x3E6B800820000000, float 0x3E6B800820000000, float 0x3E6B800820000000>
  %99 = fmul fast <8 x float> %92, %98
  %100 = fadd fast <8 x float> %99, <float 0x3EEF286940000000, float 0x3EEF286940000000, float 0x3EEF286940000000, float 0x3EEF286940000000, float 0x3EEF286940000000, float 0x3EEF286940000000, float 0x3EEF286940000000, float 0x3EEF286940000000>
  %101 = fmul fast <8 x float> %92, %100
  %102 = fadd fast <8 x float> %101, <float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000>
  %103 = fmul fast <8 x float> %92, %102
  %104 = fadd fast <8 x float> %103, <float 0x3F740B3B80000000, float 0x3F740B3B80000000, float 0x3F740B3B80000000, float 0x3F740B3B80000000, float 0x3F740B3B80000000, float 0x3F740B3B80000000, float 0x3F740B3B80000000, float 0x3F740B3B80000000>
  %105 = fmul fast <8 x float> %91, %104
  %106 = fmul fast <8 x float> %92, <float 0x3EB41A7B00000000, float 0x3EB41A7B00000000, float 0x3EB41A7B00000000, float 0x3EB41A7B00000000, float 0x3EB41A7B00000000, float 0x3EB41A7B00000000, float 0x3EB41A7B00000000, float 0x3EB41A7B00000000>
  %107 = fadd fast <8 x float> %106, <float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000>
  %108 = fmul fast <8 x float> %92, %107
  %109 = fadd fast <8 x float> %108, <float 0x3F629540A0000000, float 0x3F629540A0000000, float 0x3F629540A0000000, float 0x3F629540A0000000, float 0x3F629540A0000000, float 0x3F629540A0000000, float 0x3F629540A0000000, float 0x3F629540A0000000>
  %110 = fmul fast <8 x float> %92, %109
  %111 = fadd fast <8 x float> %110, <float 0x3F740B3BA0000000, float 0x3F740B3BA0000000, float 0x3F740B3BA0000000, float 0x3F740B3BA0000000, float 0x3F740B3BA0000000, float 0x3F740B3BA0000000, float 0x3F740B3BA0000000, float 0x3F740B3BA0000000>
  %112 = fdiv fast <8 x float> %105, %111
  %113 = fcmp fast uge <8 x float> %61, <float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00>
  %114 = select <8 x i1> %113, <8 x float> %61, <8 x float> <float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00>
  %115 = fcmp fast ule <8 x float> %114, <float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00>
  %116 = select <8 x i1> %115, <8 x float> %114, <8 x float> <float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00>
  %117 = fmul fast <8 x float> %116, %116
  %118 = fmul fast <8 x float> %117, <float 0xBCB3E4B800000000, float 0xBCB3E4B800000000, float 0xBCB3E4B800000000, float 0xBCB3E4B800000000, float 0xBCB3E4B800000000, float 0xBCB3E4B800000000, float 0xBCB3E4B800000000, float 0xBCB3E4B800000000>
  %119 = fadd fast <8 x float> %118, <float 0x3D4C266FC0000000, float 0x3D4C266FC0000000, float 0x3D4C266FC0000000, float 0x3D4C266FC0000000, float 0x3D4C266FC0000000, float 0x3D4C266FC0000000, float 0x3D4C266FC0000000, float 0x3D4C266FC0000000>
  %120 = fmul fast <8 x float> %117, %119
  %121 = fadd fast <8 x float> %120, <float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000>
  %122 = fmul fast <8 x float> %117, %121
  %123 = fadd fast <8 x float> %122, <float 0x3E6B800820000000, float 0x3E6B800820000000, float 0x3E6B800820000000, float 0x3E6B800820000000, float 0x3E6B800820000000, float 0x3E6B800820000000, float 0x3E6B800820000000, float 0x3E6B800820000000>
  %124 = fmul fast <8 x float> %117, %123
  %125 = fadd fast <8 x float> %124, <float 0x3EEF286940000000, float 0x3EEF286940000000, float 0x3EEF286940000000, float 0x3EEF286940000000, float 0x3EEF286940000000, float 0x3EEF286940000000, float 0x3EEF286940000000, float 0x3EEF286940000000>
  %126 = fmul fast <8 x float> %117, %125
  %127 = fadd fast <8 x float> %126, <float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000>
  %128 = fmul fast <8 x float> %117, %127
  %129 = fadd fast <8 x float> %128, <float 0x3F740B3B80000000, float 0x3F740B3B80000000, float 0x3F740B3B80000000, float 0x3F740B3B80000000, float 0x3F740B3B80000000, float 0x3F740B3B80000000, float 0x3F740B3B80000000, float 0x3F740B3B80000000>
  %130 = fmul fast <8 x float> %116, %129
  %131 = fmul fast <8 x float> %117, <float 0x3EB41A7B00000000, float 0x3EB41A7B00000000, float 0x3EB41A7B00000000, float 0x3EB41A7B00000000, float 0x3EB41A7B00000000, float 0x3EB41A7B00000000, float 0x3EB41A7B00000000, float 0x3EB41A7B00000000>
  %132 = fadd fast <8 x float> %131, <float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000>
  %133 = fmul fast <8 x float> %117, %132
  %134 = fadd fast <8 x float> %133, <float 0x3F629540A0000000, float 0x3F629540A0000000, float 0x3F629540A0000000, float 0x3F629540A0000000, float 0x3F629540A0000000, float 0x3F629540A0000000, float 0x3F629540A0000000, float 0x3F629540A0000000>
  %135 = fmul fast <8 x float> %117, %134
  %136 = fadd fast <8 x float> %135, <float 0x3F740B3BA0000000, float 0x3F740B3BA0000000, float 0x3F740B3BA0000000, float 0x3F740B3BA0000000, float 0x3F740B3BA0000000, float 0x3F740B3BA0000000, float 0x3F740B3BA0000000, float 0x3F740B3BA0000000>
  %137 = fdiv fast <8 x float> %130, %136
  %138 = fcmp fast uge <8 x float> %62, <float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00>
  %139 = select <8 x i1> %138, <8 x float> %62, <8 x float> <float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00>
  %140 = fcmp fast ule <8 x float> %139, <float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00>
  %141 = select <8 x i1> %140, <8 x float> %139, <8 x float> <float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00>
  %142 = fmul fast <8 x float> %141, %141
  %143 = fmul fast <8 x float> %142, <float 0xBCB3E4B800000000, float 0xBCB3E4B800000000, float 0xBCB3E4B800000000, float 0xBCB3E4B800000000, float 0xBCB3E4B800000000, float 0xBCB3E4B800000000, float 0xBCB3E4B800000000, float 0xBCB3E4B800000000>
  %144 = fadd fast <8 x float> %143, <float 0x3D4C266FC0000000, float 0x3D4C266FC0000000, float 0x3D4C266FC0000000, float 0x3D4C266FC0000000, float 0x3D4C266FC0000000, float 0x3D4C266FC0000000, float 0x3D4C266FC0000000, float 0x3D4C266FC0000000>
  %145 = fmul fast <8 x float> %142, %144
  %146 = fadd fast <8 x float> %145, <float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000>
  %147 = fmul fast <8 x float> %142, %146
  %148 = fadd fast <8 x float> %147, <float 0x3E6B800820000000, float 0x3E6B800820000000, float 0x3E6B800820000000, float 0x3E6B800820000000, float 0x3E6B800820000000, float 0x3E6B800820000000, float 0x3E6B800820000000, float 0x3E6B800820000000>
  %149 = fmul fast <8 x float> %142, %148
  %150 = fadd fast <8 x float> %149, <float 0x3EEF286940000000, float 0x3EEF286940000000, float 0x3EEF286940000000, float 0x3EEF286940000000, float 0x3EEF286940000000, float 0x3EEF286940000000, float 0x3EEF286940000000, float 0x3EEF286940000000>
  %151 = fmul fast <8 x float> %142, %150
  %152 = fadd fast <8 x float> %151, <float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000>
  %153 = fmul fast <8 x float> %142, %152
  %154 = fadd fast <8 x float> %153, <float 0x3F740B3B80000000, float 0x3F740B3B80000000, float 0x3F740B3B80000000, float 0x3F740B3B80000000, float 0x3F740B3B80000000, float 0x3F740B3B80000000, float 0x3F740B3B80000000, float 0x3F740B3B80000000>
  %155 = fmul fast <8 x float> %141, %154
  %156 = fmul fast <8 x float> %142, <float 0x3EB41A7B00000000, float 0x3EB41A7B00000000, float 0x3EB41A7B00000000, float 0x3EB41A7B00000000, float 0x3EB41A7B00000000, float 0x3EB41A7B00000000, float 0x3EB41A7B00000000, float 0x3EB41A7B00000000>
  %157 = fadd fast <8 x float> %156, <float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000>
  %158 = fmul fast <8 x float> %142, %157
  %159 = fadd fast <8 x float> %158, <float 0x3F629540A0000000, float 0x3F629540A0000000, float 0x3F629540A0000000, float 0x3F629540A0000000, float 0x3F629540A0000000, float 0x3F629540A0000000, float 0x3F629540A0000000, float 0x3F629540A0000000>
  %160 = fmul fast <8 x float> %142, %159
  %161 = fadd fast <8 x float> %160, <float 0x3F740B3BA0000000, float 0x3F740B3BA0000000, float 0x3F740B3BA0000000, float 0x3F740B3BA0000000, float 0x3F740B3BA0000000, float 0x3F740B3BA0000000, float 0x3F740B3BA0000000, float 0x3F740B3BA0000000>
  %162 = fdiv fast <8 x float> %155, %161
  %163 = getelementptr inbounds [20 x [200 x float]], [20 x [200 x float]]* %fusion.2.ls1, i64 0, i64 %fusion.2.indvar_address.dim.0.034.ls1, i64 0
  %164 = bitcast float* %163 to <8 x float>*
  store <8 x float> %87, <8 x float>* %164, align 16, !alias.scope !71, !noalias !73
  %165 = getelementptr inbounds [20 x [200 x float]], [20 x [200 x float]]* %fusion.2.ls1, i64 0, i64 %fusion.2.indvar_address.dim.0.034.ls1, i64 8
  %166 = bitcast float* %165 to <8 x float>*
  store <8 x float> %112, <8 x float>* %166, align 16, !alias.scope !71, !noalias !73
  %167 = getelementptr inbounds [20 x [200 x float]], [20 x [200 x float]]* %fusion.2.ls1, i64 0, i64 %fusion.2.indvar_address.dim.0.034.ls1, i64 16
  %168 = bitcast float* %167 to <8 x float>*
  store <8 x float> %137, <8 x float>* %168, align 16, !alias.scope !71, !noalias !73
  %169 = getelementptr inbounds [20 x [200 x float]], [20 x [200 x float]]* %fusion.2.ls1, i64 0, i64 %fusion.2.indvar_address.dim.0.034.ls1, i64 24
  %170 = bitcast float* %169 to <8 x float>*
  store <8 x float> %162, <8 x float>* %170, align 16, !alias.scope !71, !noalias !73
  %171 = getelementptr inbounds [20 x [800 x float]], [20 x [800 x float]]* %.ls12, i64 0, i64 %fusion.2.indvar_address.dim.0.034.ls1, i64 232
  %172 = bitcast float* %171 to <8 x float>*
  %wide.load86.1.ls1 = load <8 x float>, <8 x float>* %172, align 16, !alias.scope !68, !noalias !23
  %173 = getelementptr inbounds [20 x [800 x float]], [20 x [800 x float]]* %.ls12, i64 0, i64 %fusion.2.indvar_address.dim.0.034.ls1, i64 240
  %174 = bitcast float* %173 to <8 x float>*
  %wide.load87.1.ls1 = load <8 x float>, <8 x float>* %174, align 16, !alias.scope !68, !noalias !23
  %175 = getelementptr inbounds [20 x [800 x float]], [20 x [800 x float]]* %.ls12, i64 0, i64 %fusion.2.indvar_address.dim.0.034.ls1, i64 248
  %176 = bitcast float* %175 to <8 x float>*
  %wide.load88.1.ls1 = load <8 x float>, <8 x float>* %176, align 16, !alias.scope !68, !noalias !23
  %177 = getelementptr inbounds [20 x [800 x float]], [20 x [800 x float]]* %.ls12, i64 0, i64 %fusion.2.indvar_address.dim.0.034.ls1, i64 256
  %178 = bitcast float* %177 to <8 x float>*
  %wide.load89.1.ls1 = load <8 x float>, <8 x float>* %178, align 16, !alias.scope !68, !noalias !23
  %179 = fadd fast <8 x float> %wide.load86.1.ls1, %wide.load90.1.ls1
  %180 = fadd fast <8 x float> %wide.load87.1.ls1, %wide.load91.1.ls1
  %181 = fadd fast <8 x float> %wide.load88.1.ls1, %wide.load92.1.ls1
  %182 = fadd fast <8 x float> %wide.load89.1.ls1, %wide.load93.1.ls1
  %183 = fcmp fast uge <8 x float> %179, <float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00>
  %184 = select <8 x i1> %183, <8 x float> %179, <8 x float> <float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00>
  %185 = fcmp fast ule <8 x float> %184, <float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00>
  %186 = select <8 x i1> %185, <8 x float> %184, <8 x float> <float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00>
  %187 = fmul fast <8 x float> %186, %186
  %188 = fmul fast <8 x float> %187, <float 0xBCB3E4B800000000, float 0xBCB3E4B800000000, float 0xBCB3E4B800000000, float 0xBCB3E4B800000000, float 0xBCB3E4B800000000, float 0xBCB3E4B800000000, float 0xBCB3E4B800000000, float 0xBCB3E4B800000000>
  %189 = fadd fast <8 x float> %188, <float 0x3D4C266FC0000000, float 0x3D4C266FC0000000, float 0x3D4C266FC0000000, float 0x3D4C266FC0000000, float 0x3D4C266FC0000000, float 0x3D4C266FC0000000, float 0x3D4C266FC0000000, float 0x3D4C266FC0000000>
  %190 = fmul fast <8 x float> %187, %189
  %191 = fadd fast <8 x float> %190, <float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000>
  %192 = fmul fast <8 x float> %187, %191
  %193 = fadd fast <8 x float> %192, <float 0x3E6B800820000000, float 0x3E6B800820000000, float 0x3E6B800820000000, float 0x3E6B800820000000, float 0x3E6B800820000000, float 0x3E6B800820000000, float 0x3E6B800820000000, float 0x3E6B800820000000>
  %194 = fmul fast <8 x float> %187, %193
  %195 = fadd fast <8 x float> %194, <float 0x3EEF286940000000, float 0x3EEF286940000000, float 0x3EEF286940000000, float 0x3EEF286940000000, float 0x3EEF286940000000, float 0x3EEF286940000000, float 0x3EEF286940000000, float 0x3EEF286940000000>
  %196 = fmul fast <8 x float> %187, %195
  %197 = fadd fast <8 x float> %196, <float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000>
  %198 = fmul fast <8 x float> %187, %197
  %199 = fadd fast <8 x float> %198, <float 0x3F740B3B80000000, float 0x3F740B3B80000000, float 0x3F740B3B80000000, float 0x3F740B3B80000000, float 0x3F740B3B80000000, float 0x3F740B3B80000000, float 0x3F740B3B80000000, float 0x3F740B3B80000000>
  %200 = fmul fast <8 x float> %186, %199
  %201 = fmul fast <8 x float> %187, <float 0x3EB41A7B00000000, float 0x3EB41A7B00000000, float 0x3EB41A7B00000000, float 0x3EB41A7B00000000, float 0x3EB41A7B00000000, float 0x3EB41A7B00000000, float 0x3EB41A7B00000000, float 0x3EB41A7B00000000>
  %202 = fadd fast <8 x float> %201, <float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000>
  %203 = fmul fast <8 x float> %187, %202
  %204 = fadd fast <8 x float> %203, <float 0x3F629540A0000000, float 0x3F629540A0000000, float 0x3F629540A0000000, float 0x3F629540A0000000, float 0x3F629540A0000000, float 0x3F629540A0000000, float 0x3F629540A0000000, float 0x3F629540A0000000>
  %205 = fmul fast <8 x float> %187, %204
  %206 = fadd fast <8 x float> %205, <float 0x3F740B3BA0000000, float 0x3F740B3BA0000000, float 0x3F740B3BA0000000, float 0x3F740B3BA0000000, float 0x3F740B3BA0000000, float 0x3F740B3BA0000000, float 0x3F740B3BA0000000, float 0x3F740B3BA0000000>
  %207 = fdiv fast <8 x float> %200, %206
  %208 = fcmp fast uge <8 x float> %180, <float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00>
  %209 = select <8 x i1> %208, <8 x float> %180, <8 x float> <float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00>
  %210 = fcmp fast ule <8 x float> %209, <float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00>
  %211 = select <8 x i1> %210, <8 x float> %209, <8 x float> <float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00>
  %212 = fmul fast <8 x float> %211, %211
  %213 = fmul fast <8 x float> %212, <float 0xBCB3E4B800000000, float 0xBCB3E4B800000000, float 0xBCB3E4B800000000, float 0xBCB3E4B800000000, float 0xBCB3E4B800000000, float 0xBCB3E4B800000000, float 0xBCB3E4B800000000, float 0xBCB3E4B800000000>
  %214 = fadd fast <8 x float> %213, <float 0x3D4C266FC0000000, float 0x3D4C266FC0000000, float 0x3D4C266FC0000000, float 0x3D4C266FC0000000, float 0x3D4C266FC0000000, float 0x3D4C266FC0000000, float 0x3D4C266FC0000000, float 0x3D4C266FC0000000>
  %215 = fmul fast <8 x float> %212, %214
  %216 = fadd fast <8 x float> %215, <float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000>
  %217 = fmul fast <8 x float> %212, %216
  %218 = fadd fast <8 x float> %217, <float 0x3E6B800820000000, float 0x3E6B800820000000, float 0x3E6B800820000000, float 0x3E6B800820000000, float 0x3E6B800820000000, float 0x3E6B800820000000, float 0x3E6B800820000000, float 0x3E6B800820000000>
  %219 = fmul fast <8 x float> %212, %218
  %220 = fadd fast <8 x float> %219, <float 0x3EEF286940000000, float 0x3EEF286940000000, float 0x3EEF286940000000, float 0x3EEF286940000000, float 0x3EEF286940000000, float 0x3EEF286940000000, float 0x3EEF286940000000, float 0x3EEF286940000000>
  %221 = fmul fast <8 x float> %212, %220
  %222 = fadd fast <8 x float> %221, <float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000>
  %223 = fmul fast <8 x float> %212, %222
  %224 = fadd fast <8 x float> %223, <float 0x3F740B3B80000000, float 0x3F740B3B80000000, float 0x3F740B3B80000000, float 0x3F740B3B80000000, float 0x3F740B3B80000000, float 0x3F740B3B80000000, float 0x3F740B3B80000000, float 0x3F740B3B80000000>
  %225 = fmul fast <8 x float> %211, %224
  %226 = fmul fast <8 x float> %212, <float 0x3EB41A7B00000000, float 0x3EB41A7B00000000, float 0x3EB41A7B00000000, float 0x3EB41A7B00000000, float 0x3EB41A7B00000000, float 0x3EB41A7B00000000, float 0x3EB41A7B00000000, float 0x3EB41A7B00000000>
  %227 = fadd fast <8 x float> %226, <float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000>
  %228 = fmul fast <8 x float> %212, %227
  %229 = fadd fast <8 x float> %228, <float 0x3F629540A0000000, float 0x3F629540A0000000, float 0x3F629540A0000000, float 0x3F629540A0000000, float 0x3F629540A0000000, float 0x3F629540A0000000, float 0x3F629540A0000000, float 0x3F629540A0000000>
  %230 = fmul fast <8 x float> %212, %229
  %231 = fadd fast <8 x float> %230, <float 0x3F740B3BA0000000, float 0x3F740B3BA0000000, float 0x3F740B3BA0000000, float 0x3F740B3BA0000000, float 0x3F740B3BA0000000, float 0x3F740B3BA0000000, float 0x3F740B3BA0000000, float 0x3F740B3BA0000000>
  %232 = fdiv fast <8 x float> %225, %231
  %233 = fcmp fast uge <8 x float> %181, <float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00>
  %234 = select <8 x i1> %233, <8 x float> %181, <8 x float> <float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00>
  %235 = fcmp fast ule <8 x float> %234, <float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00>
  %236 = select <8 x i1> %235, <8 x float> %234, <8 x float> <float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00>
  %237 = fmul fast <8 x float> %236, %236
  %238 = fmul fast <8 x float> %237, <float 0xBCB3E4B800000000, float 0xBCB3E4B800000000, float 0xBCB3E4B800000000, float 0xBCB3E4B800000000, float 0xBCB3E4B800000000, float 0xBCB3E4B800000000, float 0xBCB3E4B800000000, float 0xBCB3E4B800000000>
  %239 = fadd fast <8 x float> %238, <float 0x3D4C266FC0000000, float 0x3D4C266FC0000000, float 0x3D4C266FC0000000, float 0x3D4C266FC0000000, float 0x3D4C266FC0000000, float 0x3D4C266FC0000000, float 0x3D4C266FC0000000, float 0x3D4C266FC0000000>
  %240 = fmul fast <8 x float> %237, %239
  %241 = fadd fast <8 x float> %240, <float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000>
  %242 = fmul fast <8 x float> %237, %241
  %243 = fadd fast <8 x float> %242, <float 0x3E6B800820000000, float 0x3E6B800820000000, float 0x3E6B800820000000, float 0x3E6B800820000000, float 0x3E6B800820000000, float 0x3E6B800820000000, float 0x3E6B800820000000, float 0x3E6B800820000000>
  %244 = fmul fast <8 x float> %237, %243
  %245 = fadd fast <8 x float> %244, <float 0x3EEF286940000000, float 0x3EEF286940000000, float 0x3EEF286940000000, float 0x3EEF286940000000, float 0x3EEF286940000000, float 0x3EEF286940000000, float 0x3EEF286940000000, float 0x3EEF286940000000>
  %246 = fmul fast <8 x float> %237, %245
  %247 = fadd fast <8 x float> %246, <float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000>
  %248 = fmul fast <8 x float> %237, %247
  %249 = fadd fast <8 x float> %248, <float 0x3F740B3B80000000, float 0x3F740B3B80000000, float 0x3F740B3B80000000, float 0x3F740B3B80000000, float 0x3F740B3B80000000, float 0x3F740B3B80000000, float 0x3F740B3B80000000, float 0x3F740B3B80000000>
  %250 = fmul fast <8 x float> %236, %249
  %251 = fmul fast <8 x float> %237, <float 0x3EB41A7B00000000, float 0x3EB41A7B00000000, float 0x3EB41A7B00000000, float 0x3EB41A7B00000000, float 0x3EB41A7B00000000, float 0x3EB41A7B00000000, float 0x3EB41A7B00000000, float 0x3EB41A7B00000000>
  %252 = fadd fast <8 x float> %251, <float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000>
  %253 = fmul fast <8 x float> %237, %252
  %254 = fadd fast <8 x float> %253, <float 0x3F629540A0000000, float 0x3F629540A0000000, float 0x3F629540A0000000, float 0x3F629540A0000000, float 0x3F629540A0000000, float 0x3F629540A0000000, float 0x3F629540A0000000, float 0x3F629540A0000000>
  %255 = fmul fast <8 x float> %237, %254
  %256 = fadd fast <8 x float> %255, <float 0x3F740B3BA0000000, float 0x3F740B3BA0000000, float 0x3F740B3BA0000000, float 0x3F740B3BA0000000, float 0x3F740B3BA0000000, float 0x3F740B3BA0000000, float 0x3F740B3BA0000000, float 0x3F740B3BA0000000>
  %257 = fdiv fast <8 x float> %250, %256
  %258 = fcmp fast uge <8 x float> %182, <float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00>
  %259 = select <8 x i1> %258, <8 x float> %182, <8 x float> <float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00>
  %260 = fcmp fast ule <8 x float> %259, <float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00>
  %261 = select <8 x i1> %260, <8 x float> %259, <8 x float> <float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00>
  %262 = fmul fast <8 x float> %261, %261
  %263 = fmul fast <8 x float> %262, <float 0xBCB3E4B800000000, float 0xBCB3E4B800000000, float 0xBCB3E4B800000000, float 0xBCB3E4B800000000, float 0xBCB3E4B800000000, float 0xBCB3E4B800000000, float 0xBCB3E4B800000000, float 0xBCB3E4B800000000>
  %264 = fadd fast <8 x float> %263, <float 0x3D4C266FC0000000, float 0x3D4C266FC0000000, float 0x3D4C266FC0000000, float 0x3D4C266FC0000000, float 0x3D4C266FC0000000, float 0x3D4C266FC0000000, float 0x3D4C266FC0000000, float 0x3D4C266FC0000000>
  %265 = fmul fast <8 x float> %262, %264
  %266 = fadd fast <8 x float> %265, <float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000>
  %267 = fmul fast <8 x float> %262, %266
  %268 = fadd fast <8 x float> %267, <float 0x3E6B800820000000, float 0x3E6B800820000000, float 0x3E6B800820000000, float 0x3E6B800820000000, float 0x3E6B800820000000, float 0x3E6B800820000000, float 0x3E6B800820000000, float 0x3E6B800820000000>
  %269 = fmul fast <8 x float> %262, %268
  %270 = fadd fast <8 x float> %269, <float 0x3EEF286940000000, float 0x3EEF286940000000, float 0x3EEF286940000000, float 0x3EEF286940000000, float 0x3EEF286940000000, float 0x3EEF286940000000, float 0x3EEF286940000000, float 0x3EEF286940000000>
  %271 = fmul fast <8 x float> %262, %270
  %272 = fadd fast <8 x float> %271, <float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000>
  %273 = fmul fast <8 x float> %262, %272
  %274 = fadd fast <8 x float> %273, <float 0x3F740B3B80000000, float 0x3F740B3B80000000, float 0x3F740B3B80000000, float 0x3F740B3B80000000, float 0x3F740B3B80000000, float 0x3F740B3B80000000, float 0x3F740B3B80000000, float 0x3F740B3B80000000>
  %275 = fmul fast <8 x float> %261, %274
  %276 = fmul fast <8 x float> %262, <float 0x3EB41A7B00000000, float 0x3EB41A7B00000000, float 0x3EB41A7B00000000, float 0x3EB41A7B00000000, float 0x3EB41A7B00000000, float 0x3EB41A7B00000000, float 0x3EB41A7B00000000, float 0x3EB41A7B00000000>
  %277 = fadd fast <8 x float> %276, <float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000>
  %278 = fmul fast <8 x float> %262, %277
  %279 = fadd fast <8 x float> %278, <float 0x3F629540A0000000, float 0x3F629540A0000000, float 0x3F629540A0000000, float 0x3F629540A0000000, float 0x3F629540A0000000, float 0x3F629540A0000000, float 0x3F629540A0000000, float 0x3F629540A0000000>
  %280 = fmul fast <8 x float> %262, %279
  %281 = fadd fast <8 x float> %280, <float 0x3F740B3BA0000000, float 0x3F740B3BA0000000, float 0x3F740B3BA0000000, float 0x3F740B3BA0000000, float 0x3F740B3BA0000000, float 0x3F740B3BA0000000, float 0x3F740B3BA0000000, float 0x3F740B3BA0000000>
  %282 = fdiv fast <8 x float> %275, %281
  %283 = getelementptr inbounds [20 x [200 x float]], [20 x [200 x float]]* %fusion.2.ls1, i64 0, i64 %fusion.2.indvar_address.dim.0.034.ls1, i64 32
  %284 = bitcast float* %283 to <8 x float>*
  store <8 x float> %207, <8 x float>* %284, align 16, !alias.scope !71, !noalias !73
  %285 = getelementptr inbounds [20 x [200 x float]], [20 x [200 x float]]* %fusion.2.ls1, i64 0, i64 %fusion.2.indvar_address.dim.0.034.ls1, i64 40
  %286 = bitcast float* %285 to <8 x float>*
  store <8 x float> %232, <8 x float>* %286, align 16, !alias.scope !71, !noalias !73
  %287 = getelementptr inbounds [20 x [200 x float]], [20 x [200 x float]]* %fusion.2.ls1, i64 0, i64 %fusion.2.indvar_address.dim.0.034.ls1, i64 48
  %288 = bitcast float* %287 to <8 x float>*
  store <8 x float> %257, <8 x float>* %288, align 16, !alias.scope !71, !noalias !73
  %289 = getelementptr inbounds [20 x [200 x float]], [20 x [200 x float]]* %fusion.2.ls1, i64 0, i64 %fusion.2.indvar_address.dim.0.034.ls1, i64 56
  %290 = bitcast float* %289 to <8 x float>*
  store <8 x float> %282, <8 x float>* %290, align 16, !alias.scope !71, !noalias !73
  %291 = getelementptr inbounds [20 x [800 x float]], [20 x [800 x float]]* %.ls12, i64 0, i64 %fusion.2.indvar_address.dim.0.034.ls1, i64 264
  %292 = bitcast float* %291 to <8 x float>*
  %wide.load86.2.ls1 = load <8 x float>, <8 x float>* %292, align 16, !alias.scope !68, !noalias !23
  %293 = getelementptr inbounds [20 x [800 x float]], [20 x [800 x float]]* %.ls12, i64 0, i64 %fusion.2.indvar_address.dim.0.034.ls1, i64 272
  %294 = bitcast float* %293 to <8 x float>*
  %wide.load87.2.ls1 = load <8 x float>, <8 x float>* %294, align 16, !alias.scope !68, !noalias !23
  %295 = getelementptr inbounds [20 x [800 x float]], [20 x [800 x float]]* %.ls12, i64 0, i64 %fusion.2.indvar_address.dim.0.034.ls1, i64 280
  %296 = bitcast float* %295 to <8 x float>*
  %wide.load88.2.ls1 = load <8 x float>, <8 x float>* %296, align 16, !alias.scope !68, !noalias !23
  %297 = getelementptr inbounds [20 x [800 x float]], [20 x [800 x float]]* %.ls12, i64 0, i64 %fusion.2.indvar_address.dim.0.034.ls1, i64 288
  %298 = bitcast float* %297 to <8 x float>*
  %wide.load89.2.ls1 = load <8 x float>, <8 x float>* %298, align 16, !alias.scope !68, !noalias !23
  %299 = fadd fast <8 x float> %wide.load86.2.ls1, %wide.load90.2.ls1
  %300 = fadd fast <8 x float> %wide.load87.2.ls1, %wide.load91.2.ls1
  %301 = fadd fast <8 x float> %wide.load88.2.ls1, %wide.load92.2.ls1
  %302 = fadd fast <8 x float> %wide.load89.2.ls1, %wide.load93.2.ls1
  %303 = fcmp fast uge <8 x float> %299, <float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00>
  %304 = select <8 x i1> %303, <8 x float> %299, <8 x float> <float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00>
  %305 = fcmp fast ule <8 x float> %304, <float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00>
  %306 = select <8 x i1> %305, <8 x float> %304, <8 x float> <float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00>
  %307 = fmul fast <8 x float> %306, %306
  %308 = fmul fast <8 x float> %307, <float 0xBCB3E4B800000000, float 0xBCB3E4B800000000, float 0xBCB3E4B800000000, float 0xBCB3E4B800000000, float 0xBCB3E4B800000000, float 0xBCB3E4B800000000, float 0xBCB3E4B800000000, float 0xBCB3E4B800000000>
  %309 = fadd fast <8 x float> %308, <float 0x3D4C266FC0000000, float 0x3D4C266FC0000000, float 0x3D4C266FC0000000, float 0x3D4C266FC0000000, float 0x3D4C266FC0000000, float 0x3D4C266FC0000000, float 0x3D4C266FC0000000, float 0x3D4C266FC0000000>
  %310 = fmul fast <8 x float> %307, %309
  %311 = fadd fast <8 x float> %310, <float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000>
  %312 = fmul fast <8 x float> %307, %311
  %313 = fadd fast <8 x float> %312, <float 0x3E6B800820000000, float 0x3E6B800820000000, float 0x3E6B800820000000, float 0x3E6B800820000000, float 0x3E6B800820000000, float 0x3E6B800820000000, float 0x3E6B800820000000, float 0x3E6B800820000000>
  %314 = fmul fast <8 x float> %307, %313
  %315 = fadd fast <8 x float> %314, <float 0x3EEF286940000000, float 0x3EEF286940000000, float 0x3EEF286940000000, float 0x3EEF286940000000, float 0x3EEF286940000000, float 0x3EEF286940000000, float 0x3EEF286940000000, float 0x3EEF286940000000>
  %316 = fmul fast <8 x float> %307, %315
  %317 = fadd fast <8 x float> %316, <float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000>
  %318 = fmul fast <8 x float> %307, %317
  %319 = fadd fast <8 x float> %318, <float 0x3F740B3B80000000, float 0x3F740B3B80000000, float 0x3F740B3B80000000, float 0x3F740B3B80000000, float 0x3F740B3B80000000, float 0x3F740B3B80000000, float 0x3F740B3B80000000, float 0x3F740B3B80000000>
  %320 = fmul fast <8 x float> %306, %319
  %321 = fmul fast <8 x float> %307, <float 0x3EB41A7B00000000, float 0x3EB41A7B00000000, float 0x3EB41A7B00000000, float 0x3EB41A7B00000000, float 0x3EB41A7B00000000, float 0x3EB41A7B00000000, float 0x3EB41A7B00000000, float 0x3EB41A7B00000000>
  %322 = fadd fast <8 x float> %321, <float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000>
  %323 = fmul fast <8 x float> %307, %322
  %324 = fadd fast <8 x float> %323, <float 0x3F629540A0000000, float 0x3F629540A0000000, float 0x3F629540A0000000, float 0x3F629540A0000000, float 0x3F629540A0000000, float 0x3F629540A0000000, float 0x3F629540A0000000, float 0x3F629540A0000000>
  %325 = fmul fast <8 x float> %307, %324
  %326 = fadd fast <8 x float> %325, <float 0x3F740B3BA0000000, float 0x3F740B3BA0000000, float 0x3F740B3BA0000000, float 0x3F740B3BA0000000, float 0x3F740B3BA0000000, float 0x3F740B3BA0000000, float 0x3F740B3BA0000000, float 0x3F740B3BA0000000>
  %327 = fdiv fast <8 x float> %320, %326
  %328 = fcmp fast uge <8 x float> %300, <float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00>
  %329 = select <8 x i1> %328, <8 x float> %300, <8 x float> <float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00>
  %330 = fcmp fast ule <8 x float> %329, <float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00>
  %331 = select <8 x i1> %330, <8 x float> %329, <8 x float> <float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00>
  %332 = fmul fast <8 x float> %331, %331
  %333 = fmul fast <8 x float> %332, <float 0xBCB3E4B800000000, float 0xBCB3E4B800000000, float 0xBCB3E4B800000000, float 0xBCB3E4B800000000, float 0xBCB3E4B800000000, float 0xBCB3E4B800000000, float 0xBCB3E4B800000000, float 0xBCB3E4B800000000>
  %334 = fadd fast <8 x float> %333, <float 0x3D4C266FC0000000, float 0x3D4C266FC0000000, float 0x3D4C266FC0000000, float 0x3D4C266FC0000000, float 0x3D4C266FC0000000, float 0x3D4C266FC0000000, float 0x3D4C266FC0000000, float 0x3D4C266FC0000000>
  %335 = fmul fast <8 x float> %332, %334
  %336 = fadd fast <8 x float> %335, <float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000>
  %337 = fmul fast <8 x float> %332, %336
  %338 = fadd fast <8 x float> %337, <float 0x3E6B800820000000, float 0x3E6B800820000000, float 0x3E6B800820000000, float 0x3E6B800820000000, float 0x3E6B800820000000, float 0x3E6B800820000000, float 0x3E6B800820000000, float 0x3E6B800820000000>
  %339 = fmul fast <8 x float> %332, %338
  %340 = fadd fast <8 x float> %339, <float 0x3EEF286940000000, float 0x3EEF286940000000, float 0x3EEF286940000000, float 0x3EEF286940000000, float 0x3EEF286940000000, float 0x3EEF286940000000, float 0x3EEF286940000000, float 0x3EEF286940000000>
  %341 = fmul fast <8 x float> %332, %340
  %342 = fadd fast <8 x float> %341, <float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000>
  %343 = fmul fast <8 x float> %332, %342
  %344 = fadd fast <8 x float> %343, <float 0x3F740B3B80000000, float 0x3F740B3B80000000, float 0x3F740B3B80000000, float 0x3F740B3B80000000, float 0x3F740B3B80000000, float 0x3F740B3B80000000, float 0x3F740B3B80000000, float 0x3F740B3B80000000>
  %345 = fmul fast <8 x float> %331, %344
  %346 = fmul fast <8 x float> %332, <float 0x3EB41A7B00000000, float 0x3EB41A7B00000000, float 0x3EB41A7B00000000, float 0x3EB41A7B00000000, float 0x3EB41A7B00000000, float 0x3EB41A7B00000000, float 0x3EB41A7B00000000, float 0x3EB41A7B00000000>
  %347 = fadd fast <8 x float> %346, <float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000>
  %348 = fmul fast <8 x float> %332, %347
  %349 = fadd fast <8 x float> %348, <float 0x3F629540A0000000, float 0x3F629540A0000000, float 0x3F629540A0000000, float 0x3F629540A0000000, float 0x3F629540A0000000, float 0x3F629540A0000000, float 0x3F629540A0000000, float 0x3F629540A0000000>
  %350 = fmul fast <8 x float> %332, %349
  %351 = fadd fast <8 x float> %350, <float 0x3F740B3BA0000000, float 0x3F740B3BA0000000, float 0x3F740B3BA0000000, float 0x3F740B3BA0000000, float 0x3F740B3BA0000000, float 0x3F740B3BA0000000, float 0x3F740B3BA0000000, float 0x3F740B3BA0000000>
  %352 = fdiv fast <8 x float> %345, %351
  %353 = fcmp fast uge <8 x float> %301, <float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00>
  %354 = select <8 x i1> %353, <8 x float> %301, <8 x float> <float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00>
  %355 = fcmp fast ule <8 x float> %354, <float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00>
  %356 = select <8 x i1> %355, <8 x float> %354, <8 x float> <float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00>
  %357 = fmul fast <8 x float> %356, %356
  %358 = fmul fast <8 x float> %357, <float 0xBCB3E4B800000000, float 0xBCB3E4B800000000, float 0xBCB3E4B800000000, float 0xBCB3E4B800000000, float 0xBCB3E4B800000000, float 0xBCB3E4B800000000, float 0xBCB3E4B800000000, float 0xBCB3E4B800000000>
  %359 = fadd fast <8 x float> %358, <float 0x3D4C266FC0000000, float 0x3D4C266FC0000000, float 0x3D4C266FC0000000, float 0x3D4C266FC0000000, float 0x3D4C266FC0000000, float 0x3D4C266FC0000000, float 0x3D4C266FC0000000, float 0x3D4C266FC0000000>
  %360 = fmul fast <8 x float> %357, %359
  %361 = fadd fast <8 x float> %360, <float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000>
  %362 = fmul fast <8 x float> %357, %361
  %363 = fadd fast <8 x float> %362, <float 0x3E6B800820000000, float 0x3E6B800820000000, float 0x3E6B800820000000, float 0x3E6B800820000000, float 0x3E6B800820000000, float 0x3E6B800820000000, float 0x3E6B800820000000, float 0x3E6B800820000000>
  %364 = fmul fast <8 x float> %357, %363
  %365 = fadd fast <8 x float> %364, <float 0x3EEF286940000000, float 0x3EEF286940000000, float 0x3EEF286940000000, float 0x3EEF286940000000, float 0x3EEF286940000000, float 0x3EEF286940000000, float 0x3EEF286940000000, float 0x3EEF286940000000>
  %366 = fmul fast <8 x float> %357, %365
  %367 = fadd fast <8 x float> %366, <float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000>
  %368 = fmul fast <8 x float> %357, %367
  %369 = fadd fast <8 x float> %368, <float 0x3F740B3B80000000, float 0x3F740B3B80000000, float 0x3F740B3B80000000, float 0x3F740B3B80000000, float 0x3F740B3B80000000, float 0x3F740B3B80000000, float 0x3F740B3B80000000, float 0x3F740B3B80000000>
  %370 = fmul fast <8 x float> %356, %369
  %371 = fmul fast <8 x float> %357, <float 0x3EB41A7B00000000, float 0x3EB41A7B00000000, float 0x3EB41A7B00000000, float 0x3EB41A7B00000000, float 0x3EB41A7B00000000, float 0x3EB41A7B00000000, float 0x3EB41A7B00000000, float 0x3EB41A7B00000000>
  %372 = fadd fast <8 x float> %371, <float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000>
  %373 = fmul fast <8 x float> %357, %372
  %374 = fadd fast <8 x float> %373, <float 0x3F629540A0000000, float 0x3F629540A0000000, float 0x3F629540A0000000, float 0x3F629540A0000000, float 0x3F629540A0000000, float 0x3F629540A0000000, float 0x3F629540A0000000, float 0x3F629540A0000000>
  %375 = fmul fast <8 x float> %357, %374
  %376 = fadd fast <8 x float> %375, <float 0x3F740B3BA0000000, float 0x3F740B3BA0000000, float 0x3F740B3BA0000000, float 0x3F740B3BA0000000, float 0x3F740B3BA0000000, float 0x3F740B3BA0000000, float 0x3F740B3BA0000000, float 0x3F740B3BA0000000>
  %377 = fdiv fast <8 x float> %370, %376
  %378 = fcmp fast uge <8 x float> %302, <float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00>
  %379 = select <8 x i1> %378, <8 x float> %302, <8 x float> <float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00>
  %380 = fcmp fast ule <8 x float> %379, <float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00>
  %381 = select <8 x i1> %380, <8 x float> %379, <8 x float> <float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00>
  %382 = fmul fast <8 x float> %381, %381
  %383 = fmul fast <8 x float> %382, <float 0xBCB3E4B800000000, float 0xBCB3E4B800000000, float 0xBCB3E4B800000000, float 0xBCB3E4B800000000, float 0xBCB3E4B800000000, float 0xBCB3E4B800000000, float 0xBCB3E4B800000000, float 0xBCB3E4B800000000>
  %384 = fadd fast <8 x float> %383, <float 0x3D4C266FC0000000, float 0x3D4C266FC0000000, float 0x3D4C266FC0000000, float 0x3D4C266FC0000000, float 0x3D4C266FC0000000, float 0x3D4C266FC0000000, float 0x3D4C266FC0000000, float 0x3D4C266FC0000000>
  %385 = fmul fast <8 x float> %382, %384
  %386 = fadd fast <8 x float> %385, <float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000>
  %387 = fmul fast <8 x float> %382, %386
  %388 = fadd fast <8 x float> %387, <float 0x3E6B800820000000, float 0x3E6B800820000000, float 0x3E6B800820000000, float 0x3E6B800820000000, float 0x3E6B800820000000, float 0x3E6B800820000000, float 0x3E6B800820000000, float 0x3E6B800820000000>
  %389 = fmul fast <8 x float> %382, %388
  %390 = fadd fast <8 x float> %389, <float 0x3EEF286940000000, float 0x3EEF286940000000, float 0x3EEF286940000000, float 0x3EEF286940000000, float 0x3EEF286940000000, float 0x3EEF286940000000, float 0x3EEF286940000000, float 0x3EEF286940000000>
  %391 = fmul fast <8 x float> %382, %390
  %392 = fadd fast <8 x float> %391, <float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000>
  %393 = fmul fast <8 x float> %382, %392
  %394 = fadd fast <8 x float> %393, <float 0x3F740B3B80000000, float 0x3F740B3B80000000, float 0x3F740B3B80000000, float 0x3F740B3B80000000, float 0x3F740B3B80000000, float 0x3F740B3B80000000, float 0x3F740B3B80000000, float 0x3F740B3B80000000>
  %395 = fmul fast <8 x float> %381, %394
  %396 = fmul fast <8 x float> %382, <float 0x3EB41A7B00000000, float 0x3EB41A7B00000000, float 0x3EB41A7B00000000, float 0x3EB41A7B00000000, float 0x3EB41A7B00000000, float 0x3EB41A7B00000000, float 0x3EB41A7B00000000, float 0x3EB41A7B00000000>
  %397 = fadd fast <8 x float> %396, <float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000>
  %398 = fmul fast <8 x float> %382, %397
  %399 = fadd fast <8 x float> %398, <float 0x3F629540A0000000, float 0x3F629540A0000000, float 0x3F629540A0000000, float 0x3F629540A0000000, float 0x3F629540A0000000, float 0x3F629540A0000000, float 0x3F629540A0000000, float 0x3F629540A0000000>
  %400 = fmul fast <8 x float> %382, %399
  %401 = fadd fast <8 x float> %400, <float 0x3F740B3BA0000000, float 0x3F740B3BA0000000, float 0x3F740B3BA0000000, float 0x3F740B3BA0000000, float 0x3F740B3BA0000000, float 0x3F740B3BA0000000, float 0x3F740B3BA0000000, float 0x3F740B3BA0000000>
  %402 = fdiv fast <8 x float> %395, %401
  %403 = getelementptr inbounds [20 x [200 x float]], [20 x [200 x float]]* %fusion.2.ls1, i64 0, i64 %fusion.2.indvar_address.dim.0.034.ls1, i64 64
  %404 = bitcast float* %403 to <8 x float>*
  store <8 x float> %327, <8 x float>* %404, align 16, !alias.scope !71, !noalias !73
  %405 = getelementptr inbounds [20 x [200 x float]], [20 x [200 x float]]* %fusion.2.ls1, i64 0, i64 %fusion.2.indvar_address.dim.0.034.ls1, i64 72
  %406 = bitcast float* %405 to <8 x float>*
  store <8 x float> %352, <8 x float>* %406, align 16, !alias.scope !71, !noalias !73
  %407 = getelementptr inbounds [20 x [200 x float]], [20 x [200 x float]]* %fusion.2.ls1, i64 0, i64 %fusion.2.indvar_address.dim.0.034.ls1, i64 80
  %408 = bitcast float* %407 to <8 x float>*
  store <8 x float> %377, <8 x float>* %408, align 16, !alias.scope !71, !noalias !73
  %409 = getelementptr inbounds [20 x [200 x float]], [20 x [200 x float]]* %fusion.2.ls1, i64 0, i64 %fusion.2.indvar_address.dim.0.034.ls1, i64 88
  %410 = bitcast float* %409 to <8 x float>*
  store <8 x float> %402, <8 x float>* %410, align 16, !alias.scope !71, !noalias !73
  %411 = getelementptr inbounds [20 x [800 x float]], [20 x [800 x float]]* %.ls12, i64 0, i64 %fusion.2.indvar_address.dim.0.034.ls1, i64 296
  %412 = bitcast float* %411 to <8 x float>*
  %wide.load86.3.ls1 = load <8 x float>, <8 x float>* %412, align 16, !alias.scope !68, !noalias !23
  %413 = getelementptr inbounds [20 x [800 x float]], [20 x [800 x float]]* %.ls12, i64 0, i64 %fusion.2.indvar_address.dim.0.034.ls1, i64 304
  %414 = bitcast float* %413 to <8 x float>*
  %wide.load87.3.ls1 = load <8 x float>, <8 x float>* %414, align 16, !alias.scope !68, !noalias !23
  %415 = getelementptr inbounds [20 x [800 x float]], [20 x [800 x float]]* %.ls12, i64 0, i64 %fusion.2.indvar_address.dim.0.034.ls1, i64 312
  %416 = bitcast float* %415 to <8 x float>*
  %wide.load88.3.ls1 = load <8 x float>, <8 x float>* %416, align 16, !alias.scope !68, !noalias !23
  %417 = getelementptr inbounds [20 x [800 x float]], [20 x [800 x float]]* %.ls12, i64 0, i64 %fusion.2.indvar_address.dim.0.034.ls1, i64 320
  %418 = bitcast float* %417 to <8 x float>*
  %wide.load89.3.ls1 = load <8 x float>, <8 x float>* %418, align 16, !alias.scope !68, !noalias !23
  %419 = fadd fast <8 x float> %wide.load86.3.ls1, %wide.load90.3.ls1
  %420 = fadd fast <8 x float> %wide.load87.3.ls1, %wide.load91.3.ls1
  %421 = fadd fast <8 x float> %wide.load88.3.ls1, %wide.load92.3.ls1
  %422 = fadd fast <8 x float> %wide.load89.3.ls1, %wide.load93.3.ls1
  %423 = fcmp fast uge <8 x float> %419, <float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00>
  %424 = select <8 x i1> %423, <8 x float> %419, <8 x float> <float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00>
  %425 = fcmp fast ule <8 x float> %424, <float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00>
  %426 = select <8 x i1> %425, <8 x float> %424, <8 x float> <float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00>
  %427 = fmul fast <8 x float> %426, %426
  %428 = fmul fast <8 x float> %427, <float 0xBCB3E4B800000000, float 0xBCB3E4B800000000, float 0xBCB3E4B800000000, float 0xBCB3E4B800000000, float 0xBCB3E4B800000000, float 0xBCB3E4B800000000, float 0xBCB3E4B800000000, float 0xBCB3E4B800000000>
  %429 = fadd fast <8 x float> %428, <float 0x3D4C266FC0000000, float 0x3D4C266FC0000000, float 0x3D4C266FC0000000, float 0x3D4C266FC0000000, float 0x3D4C266FC0000000, float 0x3D4C266FC0000000, float 0x3D4C266FC0000000, float 0x3D4C266FC0000000>
  %430 = fmul fast <8 x float> %427, %429
  %431 = fadd fast <8 x float> %430, <float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000>
  %432 = fmul fast <8 x float> %427, %431
  %433 = fadd fast <8 x float> %432, <float 0x3E6B800820000000, float 0x3E6B800820000000, float 0x3E6B800820000000, float 0x3E6B800820000000, float 0x3E6B800820000000, float 0x3E6B800820000000, float 0x3E6B800820000000, float 0x3E6B800820000000>
  %434 = fmul fast <8 x float> %427, %433
  %435 = fadd fast <8 x float> %434, <float 0x3EEF286940000000, float 0x3EEF286940000000, float 0x3EEF286940000000, float 0x3EEF286940000000, float 0x3EEF286940000000, float 0x3EEF286940000000, float 0x3EEF286940000000, float 0x3EEF286940000000>
  %436 = fmul fast <8 x float> %427, %435
  %437 = fadd fast <8 x float> %436, <float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000>
  %438 = fmul fast <8 x float> %427, %437
  %439 = fadd fast <8 x float> %438, <float 0x3F740B3B80000000, float 0x3F740B3B80000000, float 0x3F740B3B80000000, float 0x3F740B3B80000000, float 0x3F740B3B80000000, float 0x3F740B3B80000000, float 0x3F740B3B80000000, float 0x3F740B3B80000000>
  %440 = fmul fast <8 x float> %426, %439
  %441 = fmul fast <8 x float> %427, <float 0x3EB41A7B00000000, float 0x3EB41A7B00000000, float 0x3EB41A7B00000000, float 0x3EB41A7B00000000, float 0x3EB41A7B00000000, float 0x3EB41A7B00000000, float 0x3EB41A7B00000000, float 0x3EB41A7B00000000>
  %442 = fadd fast <8 x float> %441, <float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000>
  %443 = fmul fast <8 x float> %427, %442
  %444 = fadd fast <8 x float> %443, <float 0x3F629540A0000000, float 0x3F629540A0000000, float 0x3F629540A0000000, float 0x3F629540A0000000, float 0x3F629540A0000000, float 0x3F629540A0000000, float 0x3F629540A0000000, float 0x3F629540A0000000>
  %445 = fmul fast <8 x float> %427, %444
  %446 = fadd fast <8 x float> %445, <float 0x3F740B3BA0000000, float 0x3F740B3BA0000000, float 0x3F740B3BA0000000, float 0x3F740B3BA0000000, float 0x3F740B3BA0000000, float 0x3F740B3BA0000000, float 0x3F740B3BA0000000, float 0x3F740B3BA0000000>
  %447 = fdiv fast <8 x float> %440, %446
  %448 = fcmp fast uge <8 x float> %420, <float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00>
  %449 = select <8 x i1> %448, <8 x float> %420, <8 x float> <float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00>
  %450 = fcmp fast ule <8 x float> %449, <float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00>
  %451 = select <8 x i1> %450, <8 x float> %449, <8 x float> <float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00>
  %452 = fmul fast <8 x float> %451, %451
  %453 = fmul fast <8 x float> %452, <float 0xBCB3E4B800000000, float 0xBCB3E4B800000000, float 0xBCB3E4B800000000, float 0xBCB3E4B800000000, float 0xBCB3E4B800000000, float 0xBCB3E4B800000000, float 0xBCB3E4B800000000, float 0xBCB3E4B800000000>
  %454 = fadd fast <8 x float> %453, <float 0x3D4C266FC0000000, float 0x3D4C266FC0000000, float 0x3D4C266FC0000000, float 0x3D4C266FC0000000, float 0x3D4C266FC0000000, float 0x3D4C266FC0000000, float 0x3D4C266FC0000000, float 0x3D4C266FC0000000>
  %455 = fmul fast <8 x float> %452, %454
  %456 = fadd fast <8 x float> %455, <float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000>
  %457 = fmul fast <8 x float> %452, %456
  %458 = fadd fast <8 x float> %457, <float 0x3E6B800820000000, float 0x3E6B800820000000, float 0x3E6B800820000000, float 0x3E6B800820000000, float 0x3E6B800820000000, float 0x3E6B800820000000, float 0x3E6B800820000000, float 0x3E6B800820000000>
  %459 = fmul fast <8 x float> %452, %458
  %460 = fadd fast <8 x float> %459, <float 0x3EEF286940000000, float 0x3EEF286940000000, float 0x3EEF286940000000, float 0x3EEF286940000000, float 0x3EEF286940000000, float 0x3EEF286940000000, float 0x3EEF286940000000, float 0x3EEF286940000000>
  %461 = fmul fast <8 x float> %452, %460
  %462 = fadd fast <8 x float> %461, <float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000>
  %463 = fmul fast <8 x float> %452, %462
  %464 = fadd fast <8 x float> %463, <float 0x3F740B3B80000000, float 0x3F740B3B80000000, float 0x3F740B3B80000000, float 0x3F740B3B80000000, float 0x3F740B3B80000000, float 0x3F740B3B80000000, float 0x3F740B3B80000000, float 0x3F740B3B80000000>
  %465 = fmul fast <8 x float> %451, %464
  %466 = fmul fast <8 x float> %452, <float 0x3EB41A7B00000000, float 0x3EB41A7B00000000, float 0x3EB41A7B00000000, float 0x3EB41A7B00000000, float 0x3EB41A7B00000000, float 0x3EB41A7B00000000, float 0x3EB41A7B00000000, float 0x3EB41A7B00000000>
  %467 = fadd fast <8 x float> %466, <float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000>
  %468 = fmul fast <8 x float> %452, %467
  %469 = fadd fast <8 x float> %468, <float 0x3F629540A0000000, float 0x3F629540A0000000, float 0x3F629540A0000000, float 0x3F629540A0000000, float 0x3F629540A0000000, float 0x3F629540A0000000, float 0x3F629540A0000000, float 0x3F629540A0000000>
  %470 = fmul fast <8 x float> %452, %469
  %471 = fadd fast <8 x float> %470, <float 0x3F740B3BA0000000, float 0x3F740B3BA0000000, float 0x3F740B3BA0000000, float 0x3F740B3BA0000000, float 0x3F740B3BA0000000, float 0x3F740B3BA0000000, float 0x3F740B3BA0000000, float 0x3F740B3BA0000000>
  %472 = fdiv fast <8 x float> %465, %471
  %473 = fcmp fast uge <8 x float> %421, <float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00>
  %474 = select <8 x i1> %473, <8 x float> %421, <8 x float> <float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00>
  %475 = fcmp fast ule <8 x float> %474, <float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00>
  %476 = select <8 x i1> %475, <8 x float> %474, <8 x float> <float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00>
  %477 = fmul fast <8 x float> %476, %476
  %478 = fmul fast <8 x float> %477, <float 0xBCB3E4B800000000, float 0xBCB3E4B800000000, float 0xBCB3E4B800000000, float 0xBCB3E4B800000000, float 0xBCB3E4B800000000, float 0xBCB3E4B800000000, float 0xBCB3E4B800000000, float 0xBCB3E4B800000000>
  %479 = fadd fast <8 x float> %478, <float 0x3D4C266FC0000000, float 0x3D4C266FC0000000, float 0x3D4C266FC0000000, float 0x3D4C266FC0000000, float 0x3D4C266FC0000000, float 0x3D4C266FC0000000, float 0x3D4C266FC0000000, float 0x3D4C266FC0000000>
  %480 = fmul fast <8 x float> %477, %479
  %481 = fadd fast <8 x float> %480, <float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000>
  %482 = fmul fast <8 x float> %477, %481
  %483 = fadd fast <8 x float> %482, <float 0x3E6B800820000000, float 0x3E6B800820000000, float 0x3E6B800820000000, float 0x3E6B800820000000, float 0x3E6B800820000000, float 0x3E6B800820000000, float 0x3E6B800820000000, float 0x3E6B800820000000>
  %484 = fmul fast <8 x float> %477, %483
  %485 = fadd fast <8 x float> %484, <float 0x3EEF286940000000, float 0x3EEF286940000000, float 0x3EEF286940000000, float 0x3EEF286940000000, float 0x3EEF286940000000, float 0x3EEF286940000000, float 0x3EEF286940000000, float 0x3EEF286940000000>
  %486 = fmul fast <8 x float> %477, %485
  %487 = fadd fast <8 x float> %486, <float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000>
  %488 = fmul fast <8 x float> %477, %487
  %489 = fadd fast <8 x float> %488, <float 0x3F740B3B80000000, float 0x3F740B3B80000000, float 0x3F740B3B80000000, float 0x3F740B3B80000000, float 0x3F740B3B80000000, float 0x3F740B3B80000000, float 0x3F740B3B80000000, float 0x3F740B3B80000000>
  %490 = fmul fast <8 x float> %476, %489
  %491 = fmul fast <8 x float> %477, <float 0x3EB41A7B00000000, float 0x3EB41A7B00000000, float 0x3EB41A7B00000000, float 0x3EB41A7B00000000, float 0x3EB41A7B00000000, float 0x3EB41A7B00000000, float 0x3EB41A7B00000000, float 0x3EB41A7B00000000>
  %492 = fadd fast <8 x float> %491, <float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000>
  %493 = fmul fast <8 x float> %477, %492
  %494 = fadd fast <8 x float> %493, <float 0x3F629540A0000000, float 0x3F629540A0000000, float 0x3F629540A0000000, float 0x3F629540A0000000, float 0x3F629540A0000000, float 0x3F629540A0000000, float 0x3F629540A0000000, float 0x3F629540A0000000>
  %495 = fmul fast <8 x float> %477, %494
  %496 = fadd fast <8 x float> %495, <float 0x3F740B3BA0000000, float 0x3F740B3BA0000000, float 0x3F740B3BA0000000, float 0x3F740B3BA0000000, float 0x3F740B3BA0000000, float 0x3F740B3BA0000000, float 0x3F740B3BA0000000, float 0x3F740B3BA0000000>
  %497 = fdiv fast <8 x float> %490, %496
  %498 = fcmp fast uge <8 x float> %422, <float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00>
  %499 = select <8 x i1> %498, <8 x float> %422, <8 x float> <float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00>
  %500 = fcmp fast ule <8 x float> %499, <float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00>
  %501 = select <8 x i1> %500, <8 x float> %499, <8 x float> <float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00>
  %502 = fmul fast <8 x float> %501, %501
  %503 = fmul fast <8 x float> %502, <float 0xBCB3E4B800000000, float 0xBCB3E4B800000000, float 0xBCB3E4B800000000, float 0xBCB3E4B800000000, float 0xBCB3E4B800000000, float 0xBCB3E4B800000000, float 0xBCB3E4B800000000, float 0xBCB3E4B800000000>
  %504 = fadd fast <8 x float> %503, <float 0x3D4C266FC0000000, float 0x3D4C266FC0000000, float 0x3D4C266FC0000000, float 0x3D4C266FC0000000, float 0x3D4C266FC0000000, float 0x3D4C266FC0000000, float 0x3D4C266FC0000000, float 0x3D4C266FC0000000>
  %505 = fmul fast <8 x float> %502, %504
  %506 = fadd fast <8 x float> %505, <float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000>
  %507 = fmul fast <8 x float> %502, %506
  %508 = fadd fast <8 x float> %507, <float 0x3E6B800820000000, float 0x3E6B800820000000, float 0x3E6B800820000000, float 0x3E6B800820000000, float 0x3E6B800820000000, float 0x3E6B800820000000, float 0x3E6B800820000000, float 0x3E6B800820000000>
  %509 = fmul fast <8 x float> %502, %508
  %510 = fadd fast <8 x float> %509, <float 0x3EEF286940000000, float 0x3EEF286940000000, float 0x3EEF286940000000, float 0x3EEF286940000000, float 0x3EEF286940000000, float 0x3EEF286940000000, float 0x3EEF286940000000, float 0x3EEF286940000000>
  %511 = fmul fast <8 x float> %502, %510
  %512 = fadd fast <8 x float> %511, <float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000>
  %513 = fmul fast <8 x float> %502, %512
  %514 = fadd fast <8 x float> %513, <float 0x3F740B3B80000000, float 0x3F740B3B80000000, float 0x3F740B3B80000000, float 0x3F740B3B80000000, float 0x3F740B3B80000000, float 0x3F740B3B80000000, float 0x3F740B3B80000000, float 0x3F740B3B80000000>
  %515 = fmul fast <8 x float> %501, %514
  %516 = fmul fast <8 x float> %502, <float 0x3EB41A7B00000000, float 0x3EB41A7B00000000, float 0x3EB41A7B00000000, float 0x3EB41A7B00000000, float 0x3EB41A7B00000000, float 0x3EB41A7B00000000, float 0x3EB41A7B00000000, float 0x3EB41A7B00000000>
  %517 = fadd fast <8 x float> %516, <float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000>
  %518 = fmul fast <8 x float> %502, %517
  %519 = fadd fast <8 x float> %518, <float 0x3F629540A0000000, float 0x3F629540A0000000, float 0x3F629540A0000000, float 0x3F629540A0000000, float 0x3F629540A0000000, float 0x3F629540A0000000, float 0x3F629540A0000000, float 0x3F629540A0000000>
  %520 = fmul fast <8 x float> %502, %519
  %521 = fadd fast <8 x float> %520, <float 0x3F740B3BA0000000, float 0x3F740B3BA0000000, float 0x3F740B3BA0000000, float 0x3F740B3BA0000000, float 0x3F740B3BA0000000, float 0x3F740B3BA0000000, float 0x3F740B3BA0000000, float 0x3F740B3BA0000000>
  %522 = fdiv fast <8 x float> %515, %521
  %523 = getelementptr inbounds [20 x [200 x float]], [20 x [200 x float]]* %fusion.2.ls1, i64 0, i64 %fusion.2.indvar_address.dim.0.034.ls1, i64 96
  %524 = bitcast float* %523 to <8 x float>*
  store <8 x float> %447, <8 x float>* %524, align 16, !alias.scope !71, !noalias !73
  %525 = getelementptr inbounds [20 x [200 x float]], [20 x [200 x float]]* %fusion.2.ls1, i64 0, i64 %fusion.2.indvar_address.dim.0.034.ls1, i64 104
  %526 = bitcast float* %525 to <8 x float>*
  store <8 x float> %472, <8 x float>* %526, align 16, !alias.scope !71, !noalias !73
  %527 = getelementptr inbounds [20 x [200 x float]], [20 x [200 x float]]* %fusion.2.ls1, i64 0, i64 %fusion.2.indvar_address.dim.0.034.ls1, i64 112
  %528 = bitcast float* %527 to <8 x float>*
  store <8 x float> %497, <8 x float>* %528, align 16, !alias.scope !71, !noalias !73
  %529 = getelementptr inbounds [20 x [200 x float]], [20 x [200 x float]]* %fusion.2.ls1, i64 0, i64 %fusion.2.indvar_address.dim.0.034.ls1, i64 120
  %530 = bitcast float* %529 to <8 x float>*
  store <8 x float> %522, <8 x float>* %530, align 16, !alias.scope !71, !noalias !73
  %531 = getelementptr inbounds [20 x [800 x float]], [20 x [800 x float]]* %.ls12, i64 0, i64 %fusion.2.indvar_address.dim.0.034.ls1, i64 328
  %532 = bitcast float* %531 to <8 x float>*
  %wide.load86.4.ls1 = load <8 x float>, <8 x float>* %532, align 16, !alias.scope !68, !noalias !23
  %533 = getelementptr inbounds [20 x [800 x float]], [20 x [800 x float]]* %.ls12, i64 0, i64 %fusion.2.indvar_address.dim.0.034.ls1, i64 336
  %534 = bitcast float* %533 to <8 x float>*
  %wide.load87.4.ls1 = load <8 x float>, <8 x float>* %534, align 16, !alias.scope !68, !noalias !23
  %535 = getelementptr inbounds [20 x [800 x float]], [20 x [800 x float]]* %.ls12, i64 0, i64 %fusion.2.indvar_address.dim.0.034.ls1, i64 344
  %536 = bitcast float* %535 to <8 x float>*
  %wide.load88.4.ls1 = load <8 x float>, <8 x float>* %536, align 16, !alias.scope !68, !noalias !23
  %537 = getelementptr inbounds [20 x [800 x float]], [20 x [800 x float]]* %.ls12, i64 0, i64 %fusion.2.indvar_address.dim.0.034.ls1, i64 352
  %538 = bitcast float* %537 to <8 x float>*
  %wide.load89.4.ls1 = load <8 x float>, <8 x float>* %538, align 16, !alias.scope !68, !noalias !23
  %539 = fadd fast <8 x float> %wide.load86.4.ls1, %wide.load90.4.ls1
  %540 = fadd fast <8 x float> %wide.load87.4.ls1, %wide.load91.4.ls1
  %541 = fadd fast <8 x float> %wide.load88.4.ls1, %wide.load92.4.ls1
  %542 = fadd fast <8 x float> %wide.load89.4.ls1, %wide.load93.4.ls1
  %543 = fcmp fast uge <8 x float> %539, <float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00>
  %544 = select <8 x i1> %543, <8 x float> %539, <8 x float> <float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00>
  %545 = fcmp fast ule <8 x float> %544, <float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00>
  %546 = select <8 x i1> %545, <8 x float> %544, <8 x float> <float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00>
  %547 = fmul fast <8 x float> %546, %546
  %548 = fmul fast <8 x float> %547, <float 0xBCB3E4B800000000, float 0xBCB3E4B800000000, float 0xBCB3E4B800000000, float 0xBCB3E4B800000000, float 0xBCB3E4B800000000, float 0xBCB3E4B800000000, float 0xBCB3E4B800000000, float 0xBCB3E4B800000000>
  %549 = fadd fast <8 x float> %548, <float 0x3D4C266FC0000000, float 0x3D4C266FC0000000, float 0x3D4C266FC0000000, float 0x3D4C266FC0000000, float 0x3D4C266FC0000000, float 0x3D4C266FC0000000, float 0x3D4C266FC0000000, float 0x3D4C266FC0000000>
  %550 = fmul fast <8 x float> %547, %549
  %551 = fadd fast <8 x float> %550, <float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000>
  %552 = fmul fast <8 x float> %547, %551
  %553 = fadd fast <8 x float> %552, <float 0x3E6B800820000000, float 0x3E6B800820000000, float 0x3E6B800820000000, float 0x3E6B800820000000, float 0x3E6B800820000000, float 0x3E6B800820000000, float 0x3E6B800820000000, float 0x3E6B800820000000>
  %554 = fmul fast <8 x float> %547, %553
  %555 = fadd fast <8 x float> %554, <float 0x3EEF286940000000, float 0x3EEF286940000000, float 0x3EEF286940000000, float 0x3EEF286940000000, float 0x3EEF286940000000, float 0x3EEF286940000000, float 0x3EEF286940000000, float 0x3EEF286940000000>
  %556 = fmul fast <8 x float> %547, %555
  %557 = fadd fast <8 x float> %556, <float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000>
  %558 = fmul fast <8 x float> %547, %557
  %559 = fadd fast <8 x float> %558, <float 0x3F740B3B80000000, float 0x3F740B3B80000000, float 0x3F740B3B80000000, float 0x3F740B3B80000000, float 0x3F740B3B80000000, float 0x3F740B3B80000000, float 0x3F740B3B80000000, float 0x3F740B3B80000000>
  %560 = fmul fast <8 x float> %546, %559
  %561 = fmul fast <8 x float> %547, <float 0x3EB41A7B00000000, float 0x3EB41A7B00000000, float 0x3EB41A7B00000000, float 0x3EB41A7B00000000, float 0x3EB41A7B00000000, float 0x3EB41A7B00000000, float 0x3EB41A7B00000000, float 0x3EB41A7B00000000>
  %562 = fadd fast <8 x float> %561, <float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000>
  %563 = fmul fast <8 x float> %547, %562
  %564 = fadd fast <8 x float> %563, <float 0x3F629540A0000000, float 0x3F629540A0000000, float 0x3F629540A0000000, float 0x3F629540A0000000, float 0x3F629540A0000000, float 0x3F629540A0000000, float 0x3F629540A0000000, float 0x3F629540A0000000>
  %565 = fmul fast <8 x float> %547, %564
  %566 = fadd fast <8 x float> %565, <float 0x3F740B3BA0000000, float 0x3F740B3BA0000000, float 0x3F740B3BA0000000, float 0x3F740B3BA0000000, float 0x3F740B3BA0000000, float 0x3F740B3BA0000000, float 0x3F740B3BA0000000, float 0x3F740B3BA0000000>
  %567 = fdiv fast <8 x float> %560, %566
  %568 = fcmp fast uge <8 x float> %540, <float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00>
  %569 = select <8 x i1> %568, <8 x float> %540, <8 x float> <float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00>
  %570 = fcmp fast ule <8 x float> %569, <float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00>
  %571 = select <8 x i1> %570, <8 x float> %569, <8 x float> <float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00>
  %572 = fmul fast <8 x float> %571, %571
  %573 = fmul fast <8 x float> %572, <float 0xBCB3E4B800000000, float 0xBCB3E4B800000000, float 0xBCB3E4B800000000, float 0xBCB3E4B800000000, float 0xBCB3E4B800000000, float 0xBCB3E4B800000000, float 0xBCB3E4B800000000, float 0xBCB3E4B800000000>
  %574 = fadd fast <8 x float> %573, <float 0x3D4C266FC0000000, float 0x3D4C266FC0000000, float 0x3D4C266FC0000000, float 0x3D4C266FC0000000, float 0x3D4C266FC0000000, float 0x3D4C266FC0000000, float 0x3D4C266FC0000000, float 0x3D4C266FC0000000>
  %575 = fmul fast <8 x float> %572, %574
  %576 = fadd fast <8 x float> %575, <float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000>
  %577 = fmul fast <8 x float> %572, %576
  %578 = fadd fast <8 x float> %577, <float 0x3E6B800820000000, float 0x3E6B800820000000, float 0x3E6B800820000000, float 0x3E6B800820000000, float 0x3E6B800820000000, float 0x3E6B800820000000, float 0x3E6B800820000000, float 0x3E6B800820000000>
  %579 = fmul fast <8 x float> %572, %578
  %580 = fadd fast <8 x float> %579, <float 0x3EEF286940000000, float 0x3EEF286940000000, float 0x3EEF286940000000, float 0x3EEF286940000000, float 0x3EEF286940000000, float 0x3EEF286940000000, float 0x3EEF286940000000, float 0x3EEF286940000000>
  %581 = fmul fast <8 x float> %572, %580
  %582 = fadd fast <8 x float> %581, <float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000>
  %583 = fmul fast <8 x float> %572, %582
  %584 = fadd fast <8 x float> %583, <float 0x3F740B3B80000000, float 0x3F740B3B80000000, float 0x3F740B3B80000000, float 0x3F740B3B80000000, float 0x3F740B3B80000000, float 0x3F740B3B80000000, float 0x3F740B3B80000000, float 0x3F740B3B80000000>
  %585 = fmul fast <8 x float> %571, %584
  %586 = fmul fast <8 x float> %572, <float 0x3EB41A7B00000000, float 0x3EB41A7B00000000, float 0x3EB41A7B00000000, float 0x3EB41A7B00000000, float 0x3EB41A7B00000000, float 0x3EB41A7B00000000, float 0x3EB41A7B00000000, float 0x3EB41A7B00000000>
  %587 = fadd fast <8 x float> %586, <float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000>
  %588 = fmul fast <8 x float> %572, %587
  %589 = fadd fast <8 x float> %588, <float 0x3F629540A0000000, float 0x3F629540A0000000, float 0x3F629540A0000000, float 0x3F629540A0000000, float 0x3F629540A0000000, float 0x3F629540A0000000, float 0x3F629540A0000000, float 0x3F629540A0000000>
  %590 = fmul fast <8 x float> %572, %589
  %591 = fadd fast <8 x float> %590, <float 0x3F740B3BA0000000, float 0x3F740B3BA0000000, float 0x3F740B3BA0000000, float 0x3F740B3BA0000000, float 0x3F740B3BA0000000, float 0x3F740B3BA0000000, float 0x3F740B3BA0000000, float 0x3F740B3BA0000000>
  %592 = fdiv fast <8 x float> %585, %591
  %593 = fcmp fast uge <8 x float> %541, <float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00>
  %594 = select <8 x i1> %593, <8 x float> %541, <8 x float> <float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00>
  %595 = fcmp fast ule <8 x float> %594, <float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00>
  %596 = select <8 x i1> %595, <8 x float> %594, <8 x float> <float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00>
  %597 = fmul fast <8 x float> %596, %596
  %598 = fmul fast <8 x float> %597, <float 0xBCB3E4B800000000, float 0xBCB3E4B800000000, float 0xBCB3E4B800000000, float 0xBCB3E4B800000000, float 0xBCB3E4B800000000, float 0xBCB3E4B800000000, float 0xBCB3E4B800000000, float 0xBCB3E4B800000000>
  %599 = fadd fast <8 x float> %598, <float 0x3D4C266FC0000000, float 0x3D4C266FC0000000, float 0x3D4C266FC0000000, float 0x3D4C266FC0000000, float 0x3D4C266FC0000000, float 0x3D4C266FC0000000, float 0x3D4C266FC0000000, float 0x3D4C266FC0000000>
  %600 = fmul fast <8 x float> %597, %599
  %601 = fadd fast <8 x float> %600, <float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000>
  %602 = fmul fast <8 x float> %597, %601
  %603 = fadd fast <8 x float> %602, <float 0x3E6B800820000000, float 0x3E6B800820000000, float 0x3E6B800820000000, float 0x3E6B800820000000, float 0x3E6B800820000000, float 0x3E6B800820000000, float 0x3E6B800820000000, float 0x3E6B800820000000>
  %604 = fmul fast <8 x float> %597, %603
  %605 = fadd fast <8 x float> %604, <float 0x3EEF286940000000, float 0x3EEF286940000000, float 0x3EEF286940000000, float 0x3EEF286940000000, float 0x3EEF286940000000, float 0x3EEF286940000000, float 0x3EEF286940000000, float 0x3EEF286940000000>
  %606 = fmul fast <8 x float> %597, %605
  %607 = fadd fast <8 x float> %606, <float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000>
  %608 = fmul fast <8 x float> %597, %607
  %609 = fadd fast <8 x float> %608, <float 0x3F740B3B80000000, float 0x3F740B3B80000000, float 0x3F740B3B80000000, float 0x3F740B3B80000000, float 0x3F740B3B80000000, float 0x3F740B3B80000000, float 0x3F740B3B80000000, float 0x3F740B3B80000000>
  %610 = fmul fast <8 x float> %596, %609
  %611 = fmul fast <8 x float> %597, <float 0x3EB41A7B00000000, float 0x3EB41A7B00000000, float 0x3EB41A7B00000000, float 0x3EB41A7B00000000, float 0x3EB41A7B00000000, float 0x3EB41A7B00000000, float 0x3EB41A7B00000000, float 0x3EB41A7B00000000>
  %612 = fadd fast <8 x float> %611, <float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000>
  %613 = fmul fast <8 x float> %597, %612
  %614 = fadd fast <8 x float> %613, <float 0x3F629540A0000000, float 0x3F629540A0000000, float 0x3F629540A0000000, float 0x3F629540A0000000, float 0x3F629540A0000000, float 0x3F629540A0000000, float 0x3F629540A0000000, float 0x3F629540A0000000>
  %615 = fmul fast <8 x float> %597, %614
  %616 = fadd fast <8 x float> %615, <float 0x3F740B3BA0000000, float 0x3F740B3BA0000000, float 0x3F740B3BA0000000, float 0x3F740B3BA0000000, float 0x3F740B3BA0000000, float 0x3F740B3BA0000000, float 0x3F740B3BA0000000, float 0x3F740B3BA0000000>
  %617 = fdiv fast <8 x float> %610, %616
  %618 = fcmp fast uge <8 x float> %542, <float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00>
  %619 = select <8 x i1> %618, <8 x float> %542, <8 x float> <float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00>
  %620 = fcmp fast ule <8 x float> %619, <float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00>
  %621 = select <8 x i1> %620, <8 x float> %619, <8 x float> <float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00>
  %622 = fmul fast <8 x float> %621, %621
  %623 = fmul fast <8 x float> %622, <float 0xBCB3E4B800000000, float 0xBCB3E4B800000000, float 0xBCB3E4B800000000, float 0xBCB3E4B800000000, float 0xBCB3E4B800000000, float 0xBCB3E4B800000000, float 0xBCB3E4B800000000, float 0xBCB3E4B800000000>
  %624 = fadd fast <8 x float> %623, <float 0x3D4C266FC0000000, float 0x3D4C266FC0000000, float 0x3D4C266FC0000000, float 0x3D4C266FC0000000, float 0x3D4C266FC0000000, float 0x3D4C266FC0000000, float 0x3D4C266FC0000000, float 0x3D4C266FC0000000>
  %625 = fmul fast <8 x float> %622, %624
  %626 = fadd fast <8 x float> %625, <float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000>
  %627 = fmul fast <8 x float> %622, %626
  %628 = fadd fast <8 x float> %627, <float 0x3E6B800820000000, float 0x3E6B800820000000, float 0x3E6B800820000000, float 0x3E6B800820000000, float 0x3E6B800820000000, float 0x3E6B800820000000, float 0x3E6B800820000000, float 0x3E6B800820000000>
  %629 = fmul fast <8 x float> %622, %628
  %630 = fadd fast <8 x float> %629, <float 0x3EEF286940000000, float 0x3EEF286940000000, float 0x3EEF286940000000, float 0x3EEF286940000000, float 0x3EEF286940000000, float 0x3EEF286940000000, float 0x3EEF286940000000, float 0x3EEF286940000000>
  %631 = fmul fast <8 x float> %622, %630
  %632 = fadd fast <8 x float> %631, <float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000>
  %633 = fmul fast <8 x float> %622, %632
  %634 = fadd fast <8 x float> %633, <float 0x3F740B3B80000000, float 0x3F740B3B80000000, float 0x3F740B3B80000000, float 0x3F740B3B80000000, float 0x3F740B3B80000000, float 0x3F740B3B80000000, float 0x3F740B3B80000000, float 0x3F740B3B80000000>
  %635 = fmul fast <8 x float> %621, %634
  %636 = fmul fast <8 x float> %622, <float 0x3EB41A7B00000000, float 0x3EB41A7B00000000, float 0x3EB41A7B00000000, float 0x3EB41A7B00000000, float 0x3EB41A7B00000000, float 0x3EB41A7B00000000, float 0x3EB41A7B00000000, float 0x3EB41A7B00000000>
  %637 = fadd fast <8 x float> %636, <float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000>
  %638 = fmul fast <8 x float> %622, %637
  %639 = fadd fast <8 x float> %638, <float 0x3F629540A0000000, float 0x3F629540A0000000, float 0x3F629540A0000000, float 0x3F629540A0000000, float 0x3F629540A0000000, float 0x3F629540A0000000, float 0x3F629540A0000000, float 0x3F629540A0000000>
  %640 = fmul fast <8 x float> %622, %639
  %641 = fadd fast <8 x float> %640, <float 0x3F740B3BA0000000, float 0x3F740B3BA0000000, float 0x3F740B3BA0000000, float 0x3F740B3BA0000000, float 0x3F740B3BA0000000, float 0x3F740B3BA0000000, float 0x3F740B3BA0000000, float 0x3F740B3BA0000000>
  %642 = fdiv fast <8 x float> %635, %641
  %643 = getelementptr inbounds [20 x [200 x float]], [20 x [200 x float]]* %fusion.2.ls1, i64 0, i64 %fusion.2.indvar_address.dim.0.034.ls1, i64 128
  %644 = bitcast float* %643 to <8 x float>*
  store <8 x float> %567, <8 x float>* %644, align 16, !alias.scope !71, !noalias !73
  %645 = getelementptr inbounds [20 x [200 x float]], [20 x [200 x float]]* %fusion.2.ls1, i64 0, i64 %fusion.2.indvar_address.dim.0.034.ls1, i64 136
  %646 = bitcast float* %645 to <8 x float>*
  store <8 x float> %592, <8 x float>* %646, align 16, !alias.scope !71, !noalias !73
  %647 = getelementptr inbounds [20 x [200 x float]], [20 x [200 x float]]* %fusion.2.ls1, i64 0, i64 %fusion.2.indvar_address.dim.0.034.ls1, i64 144
  %648 = bitcast float* %647 to <8 x float>*
  store <8 x float> %617, <8 x float>* %648, align 16, !alias.scope !71, !noalias !73
  %649 = getelementptr inbounds [20 x [200 x float]], [20 x [200 x float]]* %fusion.2.ls1, i64 0, i64 %fusion.2.indvar_address.dim.0.034.ls1, i64 152
  %650 = bitcast float* %649 to <8 x float>*
  store <8 x float> %642, <8 x float>* %650, align 16, !alias.scope !71, !noalias !73
  %651 = getelementptr inbounds [20 x [800 x float]], [20 x [800 x float]]* %.ls12, i64 0, i64 %fusion.2.indvar_address.dim.0.034.ls1, i64 360
  %652 = bitcast float* %651 to <8 x float>*
  %wide.load86.5.ls1 = load <8 x float>, <8 x float>* %652, align 16, !alias.scope !68, !noalias !23
  %653 = getelementptr inbounds [20 x [800 x float]], [20 x [800 x float]]* %.ls12, i64 0, i64 %fusion.2.indvar_address.dim.0.034.ls1, i64 368
  %654 = bitcast float* %653 to <8 x float>*
  %wide.load87.5.ls1 = load <8 x float>, <8 x float>* %654, align 16, !alias.scope !68, !noalias !23
  %655 = getelementptr inbounds [20 x [800 x float]], [20 x [800 x float]]* %.ls12, i64 0, i64 %fusion.2.indvar_address.dim.0.034.ls1, i64 376
  %656 = bitcast float* %655 to <8 x float>*
  %wide.load88.5.ls1 = load <8 x float>, <8 x float>* %656, align 16, !alias.scope !68, !noalias !23
  %657 = getelementptr inbounds [20 x [800 x float]], [20 x [800 x float]]* %.ls12, i64 0, i64 %fusion.2.indvar_address.dim.0.034.ls1, i64 384
  %658 = bitcast float* %657 to <8 x float>*
  %wide.load89.5.ls1 = load <8 x float>, <8 x float>* %658, align 16, !alias.scope !68, !noalias !23
  %659 = fadd fast <8 x float> %wide.load86.5.ls1, %wide.load90.5.ls1
  %660 = fadd fast <8 x float> %wide.load87.5.ls1, %wide.load91.5.ls1
  %661 = fadd fast <8 x float> %wide.load88.5.ls1, %wide.load92.5.ls1
  %662 = fadd fast <8 x float> %wide.load89.5.ls1, %wide.load93.5.ls1
  %663 = fcmp fast uge <8 x float> %659, <float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00>
  %664 = select <8 x i1> %663, <8 x float> %659, <8 x float> <float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00>
  %665 = fcmp fast ule <8 x float> %664, <float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00>
  %666 = select <8 x i1> %665, <8 x float> %664, <8 x float> <float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00>
  %667 = fmul fast <8 x float> %666, %666
  %668 = fmul fast <8 x float> %667, <float 0xBCB3E4B800000000, float 0xBCB3E4B800000000, float 0xBCB3E4B800000000, float 0xBCB3E4B800000000, float 0xBCB3E4B800000000, float 0xBCB3E4B800000000, float 0xBCB3E4B800000000, float 0xBCB3E4B800000000>
  %669 = fadd fast <8 x float> %668, <float 0x3D4C266FC0000000, float 0x3D4C266FC0000000, float 0x3D4C266FC0000000, float 0x3D4C266FC0000000, float 0x3D4C266FC0000000, float 0x3D4C266FC0000000, float 0x3D4C266FC0000000, float 0x3D4C266FC0000000>
  %670 = fmul fast <8 x float> %667, %669
  %671 = fadd fast <8 x float> %670, <float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000>
  %672 = fmul fast <8 x float> %667, %671
  %673 = fadd fast <8 x float> %672, <float 0x3E6B800820000000, float 0x3E6B800820000000, float 0x3E6B800820000000, float 0x3E6B800820000000, float 0x3E6B800820000000, float 0x3E6B800820000000, float 0x3E6B800820000000, float 0x3E6B800820000000>
  %674 = fmul fast <8 x float> %667, %673
  %675 = fadd fast <8 x float> %674, <float 0x3EEF286940000000, float 0x3EEF286940000000, float 0x3EEF286940000000, float 0x3EEF286940000000, float 0x3EEF286940000000, float 0x3EEF286940000000, float 0x3EEF286940000000, float 0x3EEF286940000000>
  %676 = fmul fast <8 x float> %667, %675
  %677 = fadd fast <8 x float> %676, <float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000>
  %678 = fmul fast <8 x float> %667, %677
  %679 = fadd fast <8 x float> %678, <float 0x3F740B3B80000000, float 0x3F740B3B80000000, float 0x3F740B3B80000000, float 0x3F740B3B80000000, float 0x3F740B3B80000000, float 0x3F740B3B80000000, float 0x3F740B3B80000000, float 0x3F740B3B80000000>
  %680 = fmul fast <8 x float> %666, %679
  %681 = fmul fast <8 x float> %667, <float 0x3EB41A7B00000000, float 0x3EB41A7B00000000, float 0x3EB41A7B00000000, float 0x3EB41A7B00000000, float 0x3EB41A7B00000000, float 0x3EB41A7B00000000, float 0x3EB41A7B00000000, float 0x3EB41A7B00000000>
  %682 = fadd fast <8 x float> %681, <float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000>
  %683 = fmul fast <8 x float> %667, %682
  %684 = fadd fast <8 x float> %683, <float 0x3F629540A0000000, float 0x3F629540A0000000, float 0x3F629540A0000000, float 0x3F629540A0000000, float 0x3F629540A0000000, float 0x3F629540A0000000, float 0x3F629540A0000000, float 0x3F629540A0000000>
  %685 = fmul fast <8 x float> %667, %684
  %686 = fadd fast <8 x float> %685, <float 0x3F740B3BA0000000, float 0x3F740B3BA0000000, float 0x3F740B3BA0000000, float 0x3F740B3BA0000000, float 0x3F740B3BA0000000, float 0x3F740B3BA0000000, float 0x3F740B3BA0000000, float 0x3F740B3BA0000000>
  %687 = fdiv fast <8 x float> %680, %686
  %688 = fcmp fast uge <8 x float> %660, <float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00>
  %689 = select <8 x i1> %688, <8 x float> %660, <8 x float> <float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00>
  %690 = fcmp fast ule <8 x float> %689, <float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00>
  %691 = select <8 x i1> %690, <8 x float> %689, <8 x float> <float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00>
  %692 = fmul fast <8 x float> %691, %691
  %693 = fmul fast <8 x float> %692, <float 0xBCB3E4B800000000, float 0xBCB3E4B800000000, float 0xBCB3E4B800000000, float 0xBCB3E4B800000000, float 0xBCB3E4B800000000, float 0xBCB3E4B800000000, float 0xBCB3E4B800000000, float 0xBCB3E4B800000000>
  %694 = fadd fast <8 x float> %693, <float 0x3D4C266FC0000000, float 0x3D4C266FC0000000, float 0x3D4C266FC0000000, float 0x3D4C266FC0000000, float 0x3D4C266FC0000000, float 0x3D4C266FC0000000, float 0x3D4C266FC0000000, float 0x3D4C266FC0000000>
  %695 = fmul fast <8 x float> %692, %694
  %696 = fadd fast <8 x float> %695, <float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000>
  %697 = fmul fast <8 x float> %692, %696
  %698 = fadd fast <8 x float> %697, <float 0x3E6B800820000000, float 0x3E6B800820000000, float 0x3E6B800820000000, float 0x3E6B800820000000, float 0x3E6B800820000000, float 0x3E6B800820000000, float 0x3E6B800820000000, float 0x3E6B800820000000>
  %699 = fmul fast <8 x float> %692, %698
  %700 = fadd fast <8 x float> %699, <float 0x3EEF286940000000, float 0x3EEF286940000000, float 0x3EEF286940000000, float 0x3EEF286940000000, float 0x3EEF286940000000, float 0x3EEF286940000000, float 0x3EEF286940000000, float 0x3EEF286940000000>
  %701 = fmul fast <8 x float> %692, %700
  %702 = fadd fast <8 x float> %701, <float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000>
  %703 = fmul fast <8 x float> %692, %702
  %704 = fadd fast <8 x float> %703, <float 0x3F740B3B80000000, float 0x3F740B3B80000000, float 0x3F740B3B80000000, float 0x3F740B3B80000000, float 0x3F740B3B80000000, float 0x3F740B3B80000000, float 0x3F740B3B80000000, float 0x3F740B3B80000000>
  %705 = fmul fast <8 x float> %691, %704
  %706 = fmul fast <8 x float> %692, <float 0x3EB41A7B00000000, float 0x3EB41A7B00000000, float 0x3EB41A7B00000000, float 0x3EB41A7B00000000, float 0x3EB41A7B00000000, float 0x3EB41A7B00000000, float 0x3EB41A7B00000000, float 0x3EB41A7B00000000>
  %707 = fadd fast <8 x float> %706, <float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000>
  %708 = fmul fast <8 x float> %692, %707
  %709 = fadd fast <8 x float> %708, <float 0x3F629540A0000000, float 0x3F629540A0000000, float 0x3F629540A0000000, float 0x3F629540A0000000, float 0x3F629540A0000000, float 0x3F629540A0000000, float 0x3F629540A0000000, float 0x3F629540A0000000>
  %710 = fmul fast <8 x float> %692, %709
  %711 = fadd fast <8 x float> %710, <float 0x3F740B3BA0000000, float 0x3F740B3BA0000000, float 0x3F740B3BA0000000, float 0x3F740B3BA0000000, float 0x3F740B3BA0000000, float 0x3F740B3BA0000000, float 0x3F740B3BA0000000, float 0x3F740B3BA0000000>
  %712 = fdiv fast <8 x float> %705, %711
  %713 = fcmp fast uge <8 x float> %661, <float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00>
  %714 = select <8 x i1> %713, <8 x float> %661, <8 x float> <float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00>
  %715 = fcmp fast ule <8 x float> %714, <float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00>
  %716 = select <8 x i1> %715, <8 x float> %714, <8 x float> <float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00>
  %717 = fmul fast <8 x float> %716, %716
  %718 = fmul fast <8 x float> %717, <float 0xBCB3E4B800000000, float 0xBCB3E4B800000000, float 0xBCB3E4B800000000, float 0xBCB3E4B800000000, float 0xBCB3E4B800000000, float 0xBCB3E4B800000000, float 0xBCB3E4B800000000, float 0xBCB3E4B800000000>
  %719 = fadd fast <8 x float> %718, <float 0x3D4C266FC0000000, float 0x3D4C266FC0000000, float 0x3D4C266FC0000000, float 0x3D4C266FC0000000, float 0x3D4C266FC0000000, float 0x3D4C266FC0000000, float 0x3D4C266FC0000000, float 0x3D4C266FC0000000>
  %720 = fmul fast <8 x float> %717, %719
  %721 = fadd fast <8 x float> %720, <float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000>
  %722 = fmul fast <8 x float> %717, %721
  %723 = fadd fast <8 x float> %722, <float 0x3E6B800820000000, float 0x3E6B800820000000, float 0x3E6B800820000000, float 0x3E6B800820000000, float 0x3E6B800820000000, float 0x3E6B800820000000, float 0x3E6B800820000000, float 0x3E6B800820000000>
  %724 = fmul fast <8 x float> %717, %723
  %725 = fadd fast <8 x float> %724, <float 0x3EEF286940000000, float 0x3EEF286940000000, float 0x3EEF286940000000, float 0x3EEF286940000000, float 0x3EEF286940000000, float 0x3EEF286940000000, float 0x3EEF286940000000, float 0x3EEF286940000000>
  %726 = fmul fast <8 x float> %717, %725
  %727 = fadd fast <8 x float> %726, <float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000>
  %728 = fmul fast <8 x float> %717, %727
  %729 = fadd fast <8 x float> %728, <float 0x3F740B3B80000000, float 0x3F740B3B80000000, float 0x3F740B3B80000000, float 0x3F740B3B80000000, float 0x3F740B3B80000000, float 0x3F740B3B80000000, float 0x3F740B3B80000000, float 0x3F740B3B80000000>
  %730 = fmul fast <8 x float> %716, %729
  %731 = fmul fast <8 x float> %717, <float 0x3EB41A7B00000000, float 0x3EB41A7B00000000, float 0x3EB41A7B00000000, float 0x3EB41A7B00000000, float 0x3EB41A7B00000000, float 0x3EB41A7B00000000, float 0x3EB41A7B00000000, float 0x3EB41A7B00000000>
  %732 = fadd fast <8 x float> %731, <float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000>
  %733 = fmul fast <8 x float> %717, %732
  %734 = fadd fast <8 x float> %733, <float 0x3F629540A0000000, float 0x3F629540A0000000, float 0x3F629540A0000000, float 0x3F629540A0000000, float 0x3F629540A0000000, float 0x3F629540A0000000, float 0x3F629540A0000000, float 0x3F629540A0000000>
  %735 = fmul fast <8 x float> %717, %734
  %736 = fadd fast <8 x float> %735, <float 0x3F740B3BA0000000, float 0x3F740B3BA0000000, float 0x3F740B3BA0000000, float 0x3F740B3BA0000000, float 0x3F740B3BA0000000, float 0x3F740B3BA0000000, float 0x3F740B3BA0000000, float 0x3F740B3BA0000000>
  %737 = fdiv fast <8 x float> %730, %736
  %738 = fcmp fast uge <8 x float> %662, <float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00>
  %739 = select <8 x i1> %738, <8 x float> %662, <8 x float> <float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00>
  %740 = fcmp fast ule <8 x float> %739, <float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00>
  %741 = select <8 x i1> %740, <8 x float> %739, <8 x float> <float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00, float 9.000000e+00>
  %742 = fmul fast <8 x float> %741, %741
  %743 = fmul fast <8 x float> %742, <float 0xBCB3E4B800000000, float 0xBCB3E4B800000000, float 0xBCB3E4B800000000, float 0xBCB3E4B800000000, float 0xBCB3E4B800000000, float 0xBCB3E4B800000000, float 0xBCB3E4B800000000, float 0xBCB3E4B800000000>
  %744 = fadd fast <8 x float> %743, <float 0x3D4C266FC0000000, float 0x3D4C266FC0000000, float 0x3D4C266FC0000000, float 0x3D4C266FC0000000, float 0x3D4C266FC0000000, float 0x3D4C266FC0000000, float 0x3D4C266FC0000000, float 0x3D4C266FC0000000>
  %745 = fmul fast <8 x float> %742, %744
  %746 = fadd fast <8 x float> %745, <float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000, float 0xBDD7A6FFE0000000>
  %747 = fmul fast <8 x float> %742, %746
  %748 = fadd fast <8 x float> %747, <float 0x3E6B800820000000, float 0x3E6B800820000000, float 0x3E6B800820000000, float 0x3E6B800820000000, float 0x3E6B800820000000, float 0x3E6B800820000000, float 0x3E6B800820000000, float 0x3E6B800820000000>
  %749 = fmul fast <8 x float> %742, %748
  %750 = fadd fast <8 x float> %749, <float 0x3EEF286940000000, float 0x3EEF286940000000, float 0x3EEF286940000000, float 0x3EEF286940000000, float 0x3EEF286940000000, float 0x3EEF286940000000, float 0x3EEF286940000000, float 0x3EEF286940000000>
  %751 = fmul fast <8 x float> %742, %750
  %752 = fadd fast <8 x float> %751, <float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000, float 0x3F44E1BDA0000000>
  %753 = fmul fast <8 x float> %742, %752
  %754 = fadd fast <8 x float> %753, <float 0x3F740B3B80000000, float 0x3F740B3B80000000, float 0x3F740B3B80000000, float 0x3F740B3B80000000, float 0x3F740B3B80000000, float 0x3F740B3B80000000, float 0x3F740B3B80000000, float 0x3F740B3B80000000>
  %755 = fmul fast <8 x float> %741, %754
  %756 = fmul fast <8 x float> %742, <float 0x3EB41A7B00000000, float 0x3EB41A7B00000000, float 0x3EB41A7B00000000, float 0x3EB41A7B00000000, float 0x3EB41A7B00000000, float 0x3EB41A7B00000000, float 0x3EB41A7B00000000, float 0x3EB41A7B00000000>
  %757 = fadd fast <8 x float> %756, <float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000, float 0x3F1F12BAC0000000>
  %758 = fmul fast <8 x float> %742, %757
  %759 = fadd fast <8 x float> %758, <float 0x3F629540A0000000, float 0x3F629540A0000000, float 0x3F629540A0000000, float 0x3F629540A0000000, float 0x3F629540A0000000, float 0x3F629540A0000000, float 0x3F629540A0000000, float 0x3F629540A0000000>
  %760 = fmul fast <8 x float> %742, %759
  %761 = fadd fast <8 x float> %760, <float 0x3F740B3BA0000000, float 0x3F740B3BA0000000, float 0x3F740B3BA0000000, float 0x3F740B3BA0000000, float 0x3F740B3BA0000000, float 0x3F740B3BA0000000, float 0x3F740B3BA0000000, float 0x3F740B3BA0000000>
  %762 = fdiv fast <8 x float> %755, %761
  %763 = getelementptr inbounds [20 x [200 x float]], [20 x [200 x float]]* %fusion.2.ls1, i64 0, i64 %fusion.2.indvar_address.dim.0.034.ls1, i64 160
  %764 = bitcast float* %763 to <8 x float>*
  store <8 x float> %687, <8 x float>* %764, align 16, !alias.scope !71, !noalias !73
  %765 = getelementptr inbounds [20 x [200 x float]], [20 x [200 x float]]* %fusion.2.ls1, i64 0, i64 %fusion.2.indvar_address.dim.0.034.ls1, i64 168
  %766 = bitcast float* %765 to <8 x float>*
  store <8 x float> %712, <8 x float>* %766, align 16, !alias.scope !71, !noalias !73
  %767 = getelementptr inbounds [20 x [200 x float]], [20 x [200 x float]]* %fusion.2.ls1, i64 0, i64 %fusion.2.indvar_address.dim.0.034.ls1, i64 176
  %768 = bitcast float* %767 to <8 x float>*
  store <8 x float> %737, <8 x float>* %768, align 16, !alias.scope !71, !noalias !73
  %769 = getelementptr inbounds [20 x [200 x float]], [20 x [200 x float]]* %fusion.2.ls1, i64 0, i64 %fusion.2.indvar_address.dim.0.034.ls1, i64 184
  %770 = bitcast float* %769 to <8 x float>*
  store <8 x float> %762, <8 x float>* %770, align 16, !alias.scope !71, !noalias !73
  br label %fusion.2.loop_body.dim.1.preheader.ls1

fusion.2.loop_body.dim.1.preheader.ls1:           ; preds = %vector.body62.ls1, %fusion.2.loop_detach.dim.0.ls1
  %fusion.2.indvar_address.dim.1.033.ph.ls1 = phi i64 [ 0, %fusion.2.loop_detach.dim.0.ls1 ], [ 192, %vector.body62.ls1 ]
  br label %fusion.2.loop_body.dim.1.ls1

fusion.2.loop_body.dim.1.ls1:                     ; preds = %fusion.2.loop_body.dim.1.ls1, %fusion.2.loop_body.dim.1.preheader.ls1
  %fusion.2.indvar_address.dim.1.033.ls1 = phi i64 [ %indvar.inc6.ls1, %fusion.2.loop_body.dim.1.ls1 ], [ %fusion.2.indvar_address.dim.1.033.ph.ls1, %fusion.2.loop_body.dim.1.preheader.ls1 ]
  %771 = add nuw nsw i64 %fusion.2.indvar_address.dim.1.033.ls1, 200
  %772 = getelementptr inbounds [20 x [800 x float]], [20 x [800 x float]]* %.ls12, i64 0, i64 %fusion.2.indvar_address.dim.0.034.ls1, i64 %771
  %773 = load float, float* %772, align 4, !alias.scope !29, !noalias !23
  %774 = getelementptr inbounds [800 x float], [800 x float]* %.ls11, i64 0, i64 %771
  %775 = load float, float* %774, align 4, !invariant.load !0, !noalias !30
  %776 = fadd fast float %775, %773
  %777 = tail call fast float @tanhf(float %776)
  %778 = getelementptr inbounds [20 x [200 x float]], [20 x [200 x float]]* %fusion.2.ls1, i64 0, i64 %fusion.2.indvar_address.dim.0.034.ls1, i64 %fusion.2.indvar_address.dim.1.033.ls1
  store float %777, float* %778, align 4, !alias.scope !54, !noalias !55
  %indvar.inc6.ls1 = add nuw nsw i64 %fusion.2.indvar_address.dim.1.033.ls1, 1
  %exitcond52.ls1 = icmp eq i64 %indvar.inc6.ls1, 200
  br i1 %exitcond52.ls1, label %fusion.2.loop_inc.dim.0.ls1, label %fusion.2.loop_body.dim.1.ls1, !llvm.loop !75

fusion.2.loop_inc.dim.0.ls1:                      ; preds = %fusion.2.loop_body.dim.1.ls1
  %indvar.inc5.ls1 = add nuw nsw i64 %fusion.2.indvar_address.dim.0.034.ls1, 1
  %exitcond53.ls1 = icmp eq i64 %indvar.inc5.ls1, %end.ls1
  br i1 %exitcond53.ls1, label %fusion.2.loop_sync.dim.0.ls1, label %fusion.2.loop_detach.dim.0.ls1, !llvm.loop !76
}

; Function Attrs: argmemonly nounwind stealable
define private fastcc void @cluster_25__XlaCompiledKernel_true__XlaNumConstantArgs_1__XlaNumResourceArgs_0_.42.outline_dot.10.loop_detach.rhs.1.ls2(i64 %dot.10.indvar_address.rhs.1.038.start.ls2, i64 %end.ls2, i64 %grainsize.ls2, [20 x [400 x float]]* nocapture readonly align 16 %concatenate.9.ls2, i64 %dot.10.indvar_address.lhs.0.040.ls2, [400 x [800 x float]]* align 16 %.ls2, [20 x [800 x float]]* nocapture align 16 %.ls21) unnamed_addr #4 {
dot.10.loop_body.lhs.0.ls2:
  %__cilkrts_sf = alloca %struct.__cilkrts_stack_frame, align 8
  %0 = call %struct.__cilkrts_worker* @__cilkrts_get_tls_worker() #6
  %1 = icmp eq %struct.__cilkrts_worker* %0, null
  br i1 %1, label %slowpath.i, label %fastpath.i

slowpath.i:                                       ; preds = %dot.10.loop_body.lhs.0.ls2
  %2 = call %struct.__cilkrts_worker* @__cilkrts_bind_thread_1() #6
  %3 = getelementptr inbounds %struct.__cilkrts_stack_frame, %struct.__cilkrts_stack_frame* %__cilkrts_sf, i64 0, i32 0
  store atomic i32 16777344, i32* %3 release, align 8
  br label %__cilkrts_enter_frame_1.exit

fastpath.i:                                       ; preds = %dot.10.loop_body.lhs.0.ls2
  %4 = getelementptr inbounds %struct.__cilkrts_stack_frame, %struct.__cilkrts_stack_frame* %__cilkrts_sf, i64 0, i32 0
  store atomic i32 16777216, i32* %4 release, align 8
  br label %__cilkrts_enter_frame_1.exit

__cilkrts_enter_frame_1.exit:                     ; preds = %slowpath.i, %fastpath.i
  %5 = phi %struct.__cilkrts_worker* [ %2, %slowpath.i ], [ %0, %fastpath.i ]
  %6 = getelementptr inbounds %struct.__cilkrts_worker, %struct.__cilkrts_worker* %5, i64 0, i32 9
  %7 = load atomic %struct.__cilkrts_stack_frame*, %struct.__cilkrts_stack_frame** %6 acquire, align 8
  %8 = getelementptr inbounds %struct.__cilkrts_stack_frame, %struct.__cilkrts_stack_frame* %__cilkrts_sf, i64 0, i32 2
  store atomic %struct.__cilkrts_stack_frame* %7, %struct.__cilkrts_stack_frame** %8 release, align 8
  %9 = getelementptr inbounds %struct.__cilkrts_stack_frame, %struct.__cilkrts_stack_frame* %__cilkrts_sf, i64 0, i32 3
  store atomic %struct.__cilkrts_worker* %5, %struct.__cilkrts_worker** %9 release, align 8
  store atomic %struct.__cilkrts_stack_frame* %__cilkrts_sf, %struct.__cilkrts_stack_frame** %6 release, align 8
  %itercount2 = sub i64 %end.ls2, %dot.10.indvar_address.rhs.1.038.start.ls2
  %10 = icmp ugt i64 %itercount2, %grainsize.ls2
  br i1 %10, label %.lr.ph.preheader, label %dot.10.loop_detach.rhs.1.ls2.preheader

.lr.ph.preheader:                                 ; preds = %__cilkrts_enter_frame_1.exit
  %11 = getelementptr inbounds %struct.__cilkrts_stack_frame, %struct.__cilkrts_stack_frame* %__cilkrts_sf, i64 0, i32 6
  %12 = getelementptr inbounds %struct.__cilkrts_stack_frame, %struct.__cilkrts_stack_frame* %__cilkrts_sf, i64 0, i32 7
  %13 = getelementptr inbounds %struct.__cilkrts_stack_frame, %struct.__cilkrts_stack_frame* %__cilkrts_sf, i64 0, i32 5
  %14 = getelementptr inbounds [5 x i8*], [5 x i8*]* %13, i64 0, i64 0
  %15 = getelementptr inbounds %struct.__cilkrts_stack_frame, %struct.__cilkrts_stack_frame* %__cilkrts_sf, i64 0, i32 5, i64 2
  %16 = bitcast [5 x i8*]* %13 to i8*
  br label %.lr.ph

dot.10.loop_detach.rhs.1.ls2.preheader:           ; preds = %.split.split, %__cilkrts_enter_frame_1.exit
  %dot.10.indvar_address.rhs.1.038.ls2.dac.lcssa = phi i64 [ %dot.10.indvar_address.rhs.1.038.start.ls2, %__cilkrts_enter_frame_1.exit ], [ %miditer, %.split.split ]
  %17 = getelementptr inbounds [20 x [400 x float]], [20 x [400 x float]]* %concatenate.9.ls2, i64 0, i64 %dot.10.indvar_address.lhs.0.040.ls2, i64 0
  %18 = bitcast float* %17 to <8 x float>*
  %wide.load.ls2 = load <8 x float>, <8 x float>* %18, align 16, !alias.scope !30, !noalias !77
  %19 = getelementptr inbounds [20 x [400 x float]], [20 x [400 x float]]* %concatenate.9.ls2, i64 0, i64 %dot.10.indvar_address.lhs.0.040.ls2, i64 8
  %20 = bitcast float* %19 to <8 x float>*
  %wide.load.1.ls2 = load <8 x float>, <8 x float>* %20, align 16, !alias.scope !30, !noalias !77
  %21 = getelementptr inbounds [20 x [400 x float]], [20 x [400 x float]]* %concatenate.9.ls2, i64 0, i64 %dot.10.indvar_address.lhs.0.040.ls2, i64 16
  %22 = bitcast float* %21 to <8 x float>*
  %wide.load.2.ls2 = load <8 x float>, <8 x float>* %22, align 16, !alias.scope !30, !noalias !77
  %23 = getelementptr inbounds [20 x [400 x float]], [20 x [400 x float]]* %concatenate.9.ls2, i64 0, i64 %dot.10.indvar_address.lhs.0.040.ls2, i64 24
  %24 = bitcast float* %23 to <8 x float>*
  %wide.load.3.ls2 = load <8 x float>, <8 x float>* %24, align 16, !alias.scope !30, !noalias !77
  %25 = getelementptr inbounds [20 x [400 x float]], [20 x [400 x float]]* %concatenate.9.ls2, i64 0, i64 %dot.10.indvar_address.lhs.0.040.ls2, i64 32
  %26 = bitcast float* %25 to <8 x float>*
  %wide.load.4.ls2 = load <8 x float>, <8 x float>* %26, align 16, !alias.scope !30, !noalias !77
  %27 = getelementptr inbounds [20 x [400 x float]], [20 x [400 x float]]* %concatenate.9.ls2, i64 0, i64 %dot.10.indvar_address.lhs.0.040.ls2, i64 40
  %28 = bitcast float* %27 to <8 x float>*
  %wide.load.ls2.1 = load <8 x float>, <8 x float>* %28, align 16, !alias.scope !30, !noalias !77
  %29 = getelementptr inbounds [20 x [400 x float]], [20 x [400 x float]]* %concatenate.9.ls2, i64 0, i64 %dot.10.indvar_address.lhs.0.040.ls2, i64 48
  %30 = bitcast float* %29 to <8 x float>*
  %wide.load.1.ls2.1 = load <8 x float>, <8 x float>* %30, align 16, !alias.scope !30, !noalias !77
  %31 = getelementptr inbounds [20 x [400 x float]], [20 x [400 x float]]* %concatenate.9.ls2, i64 0, i64 %dot.10.indvar_address.lhs.0.040.ls2, i64 56
  %32 = bitcast float* %31 to <8 x float>*
  %wide.load.2.ls2.1 = load <8 x float>, <8 x float>* %32, align 16, !alias.scope !30, !noalias !77
  %33 = getelementptr inbounds [20 x [400 x float]], [20 x [400 x float]]* %concatenate.9.ls2, i64 0, i64 %dot.10.indvar_address.lhs.0.040.ls2, i64 64
  %34 = bitcast float* %33 to <8 x float>*
  %wide.load.3.ls2.1 = load <8 x float>, <8 x float>* %34, align 16, !alias.scope !30, !noalias !77
  %35 = getelementptr inbounds [20 x [400 x float]], [20 x [400 x float]]* %concatenate.9.ls2, i64 0, i64 %dot.10.indvar_address.lhs.0.040.ls2, i64 72
  %36 = bitcast float* %35 to <8 x float>*
  %wide.load.4.ls2.1 = load <8 x float>, <8 x float>* %36, align 16, !alias.scope !30, !noalias !77
  %37 = getelementptr inbounds [20 x [400 x float]], [20 x [400 x float]]* %concatenate.9.ls2, i64 0, i64 %dot.10.indvar_address.lhs.0.040.ls2, i64 80
  %38 = bitcast float* %37 to <8 x float>*
  %wide.load.ls2.2 = load <8 x float>, <8 x float>* %38, align 16, !alias.scope !30, !noalias !77
  %39 = getelementptr inbounds [20 x [400 x float]], [20 x [400 x float]]* %concatenate.9.ls2, i64 0, i64 %dot.10.indvar_address.lhs.0.040.ls2, i64 88
  %40 = bitcast float* %39 to <8 x float>*
  %wide.load.1.ls2.2 = load <8 x float>, <8 x float>* %40, align 16, !alias.scope !30, !noalias !77
  %41 = getelementptr inbounds [20 x [400 x float]], [20 x [400 x float]]* %concatenate.9.ls2, i64 0, i64 %dot.10.indvar_address.lhs.0.040.ls2, i64 96
  %42 = bitcast float* %41 to <8 x float>*
  %wide.load.2.ls2.2 = load <8 x float>, <8 x float>* %42, align 16, !alias.scope !30, !noalias !77
  %43 = getelementptr inbounds [20 x [400 x float]], [20 x [400 x float]]* %concatenate.9.ls2, i64 0, i64 %dot.10.indvar_address.lhs.0.040.ls2, i64 104
  %44 = bitcast float* %43 to <8 x float>*
  %wide.load.3.ls2.2 = load <8 x float>, <8 x float>* %44, align 16, !alias.scope !30, !noalias !77
  %45 = getelementptr inbounds [20 x [400 x float]], [20 x [400 x float]]* %concatenate.9.ls2, i64 0, i64 %dot.10.indvar_address.lhs.0.040.ls2, i64 112
  %46 = bitcast float* %45 to <8 x float>*
  %wide.load.4.ls2.2 = load <8 x float>, <8 x float>* %46, align 16, !alias.scope !30, !noalias !77
  %47 = getelementptr inbounds [20 x [400 x float]], [20 x [400 x float]]* %concatenate.9.ls2, i64 0, i64 %dot.10.indvar_address.lhs.0.040.ls2, i64 120
  %48 = bitcast float* %47 to <8 x float>*
  %wide.load.ls2.3 = load <8 x float>, <8 x float>* %48, align 16, !alias.scope !30, !noalias !77
  %49 = getelementptr inbounds [20 x [400 x float]], [20 x [400 x float]]* %concatenate.9.ls2, i64 0, i64 %dot.10.indvar_address.lhs.0.040.ls2, i64 128
  %50 = bitcast float* %49 to <8 x float>*
  %wide.load.1.ls2.3 = load <8 x float>, <8 x float>* %50, align 16, !alias.scope !30, !noalias !77
  %51 = getelementptr inbounds [20 x [400 x float]], [20 x [400 x float]]* %concatenate.9.ls2, i64 0, i64 %dot.10.indvar_address.lhs.0.040.ls2, i64 136
  %52 = bitcast float* %51 to <8 x float>*
  %wide.load.2.ls2.3 = load <8 x float>, <8 x float>* %52, align 16, !alias.scope !30, !noalias !77
  %53 = getelementptr inbounds [20 x [400 x float]], [20 x [400 x float]]* %concatenate.9.ls2, i64 0, i64 %dot.10.indvar_address.lhs.0.040.ls2, i64 144
  %54 = bitcast float* %53 to <8 x float>*
  %wide.load.3.ls2.3 = load <8 x float>, <8 x float>* %54, align 16, !alias.scope !30, !noalias !77
  %55 = getelementptr inbounds [20 x [400 x float]], [20 x [400 x float]]* %concatenate.9.ls2, i64 0, i64 %dot.10.indvar_address.lhs.0.040.ls2, i64 152
  %56 = bitcast float* %55 to <8 x float>*
  %wide.load.4.ls2.3 = load <8 x float>, <8 x float>* %56, align 16, !alias.scope !30, !noalias !77
  %57 = getelementptr inbounds [20 x [400 x float]], [20 x [400 x float]]* %concatenate.9.ls2, i64 0, i64 %dot.10.indvar_address.lhs.0.040.ls2, i64 160
  %58 = bitcast float* %57 to <8 x float>*
  %wide.load.ls2.4 = load <8 x float>, <8 x float>* %58, align 16, !alias.scope !30, !noalias !77
  %59 = getelementptr inbounds [20 x [400 x float]], [20 x [400 x float]]* %concatenate.9.ls2, i64 0, i64 %dot.10.indvar_address.lhs.0.040.ls2, i64 168
  %60 = bitcast float* %59 to <8 x float>*
  %wide.load.1.ls2.4 = load <8 x float>, <8 x float>* %60, align 16, !alias.scope !30, !noalias !77
  %61 = getelementptr inbounds [20 x [400 x float]], [20 x [400 x float]]* %concatenate.9.ls2, i64 0, i64 %dot.10.indvar_address.lhs.0.040.ls2, i64 176
  %62 = bitcast float* %61 to <8 x float>*
  %wide.load.2.ls2.4 = load <8 x float>, <8 x float>* %62, align 16, !alias.scope !30, !noalias !77
  %63 = getelementptr inbounds [20 x [400 x float]], [20 x [400 x float]]* %concatenate.9.ls2, i64 0, i64 %dot.10.indvar_address.lhs.0.040.ls2, i64 184
  %64 = bitcast float* %63 to <8 x float>*
  %wide.load.3.ls2.4 = load <8 x float>, <8 x float>* %64, align 16, !alias.scope !30, !noalias !77
  %65 = getelementptr inbounds [20 x [400 x float]], [20 x [400 x float]]* %concatenate.9.ls2, i64 0, i64 %dot.10.indvar_address.lhs.0.040.ls2, i64 192
  %66 = bitcast float* %65 to <8 x float>*
  %wide.load.4.ls2.4 = load <8 x float>, <8 x float>* %66, align 16, !alias.scope !30, !noalias !77
  %67 = getelementptr inbounds [20 x [400 x float]], [20 x [400 x float]]* %concatenate.9.ls2, i64 0, i64 %dot.10.indvar_address.lhs.0.040.ls2, i64 200
  %68 = bitcast float* %67 to <8 x float>*
  %wide.load.ls2.5 = load <8 x float>, <8 x float>* %68, align 16, !alias.scope !30, !noalias !77
  %69 = getelementptr inbounds [20 x [400 x float]], [20 x [400 x float]]* %concatenate.9.ls2, i64 0, i64 %dot.10.indvar_address.lhs.0.040.ls2, i64 208
  %70 = bitcast float* %69 to <8 x float>*
  %wide.load.1.ls2.5 = load <8 x float>, <8 x float>* %70, align 16, !alias.scope !30, !noalias !77
  %71 = getelementptr inbounds [20 x [400 x float]], [20 x [400 x float]]* %concatenate.9.ls2, i64 0, i64 %dot.10.indvar_address.lhs.0.040.ls2, i64 216
  %72 = bitcast float* %71 to <8 x float>*
  %wide.load.2.ls2.5 = load <8 x float>, <8 x float>* %72, align 16, !alias.scope !30, !noalias !77
  %73 = getelementptr inbounds [20 x [400 x float]], [20 x [400 x float]]* %concatenate.9.ls2, i64 0, i64 %dot.10.indvar_address.lhs.0.040.ls2, i64 224
  %74 = bitcast float* %73 to <8 x float>*
  %wide.load.3.ls2.5 = load <8 x float>, <8 x float>* %74, align 16, !alias.scope !30, !noalias !77
  %75 = getelementptr inbounds [20 x [400 x float]], [20 x [400 x float]]* %concatenate.9.ls2, i64 0, i64 %dot.10.indvar_address.lhs.0.040.ls2, i64 232
  %76 = bitcast float* %75 to <8 x float>*
  %wide.load.4.ls2.5 = load <8 x float>, <8 x float>* %76, align 16, !alias.scope !30, !noalias !77
  %77 = getelementptr inbounds [20 x [400 x float]], [20 x [400 x float]]* %concatenate.9.ls2, i64 0, i64 %dot.10.indvar_address.lhs.0.040.ls2, i64 240
  %78 = bitcast float* %77 to <8 x float>*
  %wide.load.ls2.6 = load <8 x float>, <8 x float>* %78, align 16, !alias.scope !30, !noalias !77
  %79 = getelementptr inbounds [20 x [400 x float]], [20 x [400 x float]]* %concatenate.9.ls2, i64 0, i64 %dot.10.indvar_address.lhs.0.040.ls2, i64 248
  %80 = bitcast float* %79 to <8 x float>*
  %wide.load.1.ls2.6 = load <8 x float>, <8 x float>* %80, align 16, !alias.scope !30, !noalias !77
  %81 = getelementptr inbounds [20 x [400 x float]], [20 x [400 x float]]* %concatenate.9.ls2, i64 0, i64 %dot.10.indvar_address.lhs.0.040.ls2, i64 256
  %82 = bitcast float* %81 to <8 x float>*
  %wide.load.2.ls2.6 = load <8 x float>, <8 x float>* %82, align 16, !alias.scope !30, !noalias !77
  %83 = getelementptr inbounds [20 x [400 x float]], [20 x [400 x float]]* %concatenate.9.ls2, i64 0, i64 %dot.10.indvar_address.lhs.0.040.ls2, i64 264
  %84 = bitcast float* %83 to <8 x float>*
  %wide.load.3.ls2.6 = load <8 x float>, <8 x float>* %84, align 16, !alias.scope !30, !noalias !77
  %85 = getelementptr inbounds [20 x [400 x float]], [20 x [400 x float]]* %concatenate.9.ls2, i64 0, i64 %dot.10.indvar_address.lhs.0.040.ls2, i64 272
  %86 = bitcast float* %85 to <8 x float>*
  %wide.load.4.ls2.6 = load <8 x float>, <8 x float>* %86, align 16, !alias.scope !30, !noalias !77
  %87 = getelementptr inbounds [20 x [400 x float]], [20 x [400 x float]]* %concatenate.9.ls2, i64 0, i64 %dot.10.indvar_address.lhs.0.040.ls2, i64 280
  %88 = bitcast float* %87 to <8 x float>*
  %wide.load.ls2.7 = load <8 x float>, <8 x float>* %88, align 16, !alias.scope !30, !noalias !77
  %89 = getelementptr inbounds [20 x [400 x float]], [20 x [400 x float]]* %concatenate.9.ls2, i64 0, i64 %dot.10.indvar_address.lhs.0.040.ls2, i64 288
  %90 = bitcast float* %89 to <8 x float>*
  %wide.load.1.ls2.7 = load <8 x float>, <8 x float>* %90, align 16, !alias.scope !30, !noalias !77
  %91 = getelementptr inbounds [20 x [400 x float]], [20 x [400 x float]]* %concatenate.9.ls2, i64 0, i64 %dot.10.indvar_address.lhs.0.040.ls2, i64 296
  %92 = bitcast float* %91 to <8 x float>*
  %wide.load.2.ls2.7 = load <8 x float>, <8 x float>* %92, align 16, !alias.scope !30, !noalias !77
  %93 = getelementptr inbounds [20 x [400 x float]], [20 x [400 x float]]* %concatenate.9.ls2, i64 0, i64 %dot.10.indvar_address.lhs.0.040.ls2, i64 304
  %94 = bitcast float* %93 to <8 x float>*
  %wide.load.3.ls2.7 = load <8 x float>, <8 x float>* %94, align 16, !alias.scope !30, !noalias !77
  %95 = getelementptr inbounds [20 x [400 x float]], [20 x [400 x float]]* %concatenate.9.ls2, i64 0, i64 %dot.10.indvar_address.lhs.0.040.ls2, i64 312
  %96 = bitcast float* %95 to <8 x float>*
  %wide.load.4.ls2.7 = load <8 x float>, <8 x float>* %96, align 16, !alias.scope !30, !noalias !77
  %97 = getelementptr inbounds [20 x [400 x float]], [20 x [400 x float]]* %concatenate.9.ls2, i64 0, i64 %dot.10.indvar_address.lhs.0.040.ls2, i64 320
  %98 = bitcast float* %97 to <8 x float>*
  %wide.load.ls2.8 = load <8 x float>, <8 x float>* %98, align 16, !alias.scope !30, !noalias !77
  %99 = getelementptr inbounds [20 x [400 x float]], [20 x [400 x float]]* %concatenate.9.ls2, i64 0, i64 %dot.10.indvar_address.lhs.0.040.ls2, i64 328
  %100 = bitcast float* %99 to <8 x float>*
  %wide.load.1.ls2.8 = load <8 x float>, <8 x float>* %100, align 16, !alias.scope !30, !noalias !77
  %101 = getelementptr inbounds [20 x [400 x float]], [20 x [400 x float]]* %concatenate.9.ls2, i64 0, i64 %dot.10.indvar_address.lhs.0.040.ls2, i64 336
  %102 = bitcast float* %101 to <8 x float>*
  %wide.load.2.ls2.8 = load <8 x float>, <8 x float>* %102, align 16, !alias.scope !30, !noalias !77
  %103 = getelementptr inbounds [20 x [400 x float]], [20 x [400 x float]]* %concatenate.9.ls2, i64 0, i64 %dot.10.indvar_address.lhs.0.040.ls2, i64 344
  %104 = bitcast float* %103 to <8 x float>*
  %wide.load.3.ls2.8 = load <8 x float>, <8 x float>* %104, align 16, !alias.scope !30, !noalias !77
  %105 = getelementptr inbounds [20 x [400 x float]], [20 x [400 x float]]* %concatenate.9.ls2, i64 0, i64 %dot.10.indvar_address.lhs.0.040.ls2, i64 352
  %106 = bitcast float* %105 to <8 x float>*
  %wide.load.4.ls2.8 = load <8 x float>, <8 x float>* %106, align 16, !alias.scope !30, !noalias !77
  %107 = getelementptr inbounds [20 x [400 x float]], [20 x [400 x float]]* %concatenate.9.ls2, i64 0, i64 %dot.10.indvar_address.lhs.0.040.ls2, i64 360
  %108 = bitcast float* %107 to <8 x float>*
  %wide.load.ls2.9 = load <8 x float>, <8 x float>* %108, align 16, !alias.scope !30, !noalias !77
  %109 = getelementptr inbounds [20 x [400 x float]], [20 x [400 x float]]* %concatenate.9.ls2, i64 0, i64 %dot.10.indvar_address.lhs.0.040.ls2, i64 368
  %110 = bitcast float* %109 to <8 x float>*
  %wide.load.1.ls2.9 = load <8 x float>, <8 x float>* %110, align 16, !alias.scope !30, !noalias !77
  %111 = getelementptr inbounds [20 x [400 x float]], [20 x [400 x float]]* %concatenate.9.ls2, i64 0, i64 %dot.10.indvar_address.lhs.0.040.ls2, i64 376
  %112 = bitcast float* %111 to <8 x float>*
  %wide.load.2.ls2.9 = load <8 x float>, <8 x float>* %112, align 16, !alias.scope !30, !noalias !77
  %113 = getelementptr inbounds [20 x [400 x float]], [20 x [400 x float]]* %concatenate.9.ls2, i64 0, i64 %dot.10.indvar_address.lhs.0.040.ls2, i64 384
  %114 = bitcast float* %113 to <8 x float>*
  %wide.load.3.ls2.9 = load <8 x float>, <8 x float>* %114, align 16, !alias.scope !30, !noalias !77
  %115 = getelementptr inbounds [20 x [400 x float]], [20 x [400 x float]]* %concatenate.9.ls2, i64 0, i64 %dot.10.indvar_address.lhs.0.040.ls2, i64 392
  %116 = bitcast float* %115 to <8 x float>*
  %wide.load.4.ls2.9 = load <8 x float>, <8 x float>* %116, align 16, !alias.scope !30, !noalias !77
  br label %dot.10.loop_detach.rhs.1.ls2

.lr.ph:                                           ; preds = %.lr.ph.preheader, %.split.split
  %itercount4 = phi i64 [ %itercount, %.split.split ], [ %itercount2, %.lr.ph.preheader ]
  %dot.10.indvar_address.rhs.1.038.ls2.dac3 = phi i64 [ %miditer, %.split.split ], [ %dot.10.indvar_address.rhs.1.038.start.ls2, %.lr.ph.preheader ]
  %halfcount = lshr i64 %itercount4, 1
  %miditer = add nuw nsw i64 %dot.10.indvar_address.rhs.1.038.ls2.dac3, %halfcount
  call void asm sideeffect "stmxcsr $0\0A\09fnstcw $1", "*m,*m,~{dirflag},~{fpsr},~{flags}"(i32* nonnull %11, i16* nonnull %12) #6
  %117 = call i8* @llvm.frameaddress(i32 0)
  store volatile i8* %117, i8** %14, align 8
  %118 = call i8* @llvm.stacksave()
  store volatile i8* %118, i8** %15, align 8
  %119 = call i32 @llvm.eh.sjlj.setjmp(i8* nonnull %16) #7
  %120 = icmp eq i32 %119, 0
  br i1 %120, label %.lr.ph.split, label %.split.split

.lr.ph.split:                                     ; preds = %.lr.ph
  call fastcc void @cluster_25__XlaCompiledKernel_true__XlaNumConstantArgs_1__XlaNumResourceArgs_0_.42.outline_dot.10.loop_detach.rhs.1.ls2.outline_.split.otd1(i64 %dot.10.indvar_address.rhs.1.038.ls2.dac3, i64 %miditer, i64 %grainsize.ls2, [20 x [400 x float]]* %concatenate.9.ls2, i64 %dot.10.indvar_address.lhs.0.040.ls2, [400 x [800 x float]]* %.ls2, [20 x [800 x float]]* %.ls21) #6
  br label %.split.split

.split.split:                                     ; preds = %.lr.ph, %.lr.ph.split
  %itercount = sub i64 %end.ls2, %miditer
  %121 = icmp ugt i64 %itercount, %grainsize.ls2
  br i1 %121, label %.lr.ph, label %dot.10.loop_detach.rhs.1.ls2.preheader

dot.10.loop_sync.rhs.1.ls2:                       ; preds = %dot.10.loop_detach.rhs.1.ls2
  %122 = getelementptr inbounds %struct.__cilkrts_stack_frame, %struct.__cilkrts_stack_frame* %__cilkrts_sf, i64 0, i32 0
  %123 = load atomic i32, i32* %122 acquire, align 8
  %124 = and i32 %123, 2
  %125 = icmp eq i32 %124, 0
  br i1 %125, label %__cilk_sync_nothrow.exit, label %cilk.sync.savestate.i

cilk.sync.savestate.i:                            ; preds = %dot.10.loop_sync.rhs.1.ls2
  %126 = load atomic %struct.__cilkrts_worker*, %struct.__cilkrts_worker** %9 acquire, align 8
  %.elt = getelementptr inbounds %struct.__cilkrts_worker, %struct.__cilkrts_worker* %126, i64 0, i32 12, i32 0
  %.unpack = load i64, i64* %.elt, align 8
  %.elt7 = getelementptr inbounds %struct.__cilkrts_worker, %struct.__cilkrts_worker* %126, i64 0, i32 12, i32 1
  %127 = bitcast %struct.__cilkrts_pedigree** %.elt7 to i64*
  %.unpack89 = load i64, i64* %127, align 8
  %.fca.0.0.gep = getelementptr inbounds %struct.__cilkrts_stack_frame, %struct.__cilkrts_stack_frame* %__cilkrts_sf, i64 0, i32 9, i32 0, i32 0
  %.fca.0.1.gep = getelementptr inbounds %struct.__cilkrts_stack_frame, %struct.__cilkrts_stack_frame* %__cilkrts_sf, i64 0, i32 9, i32 0, i32 1
  store i64 %.unpack, i64* %.fca.0.0.gep, align 8
  %128 = bitcast %struct.__cilkrts_pedigree** %.fca.0.1.gep to i64*
  store i64 %.unpack89, i64* %128, align 8
  %129 = getelementptr inbounds %struct.__cilkrts_stack_frame, %struct.__cilkrts_stack_frame* %__cilkrts_sf, i64 0, i32 6
  %130 = getelementptr inbounds %struct.__cilkrts_stack_frame, %struct.__cilkrts_stack_frame* %__cilkrts_sf, i64 0, i32 7
  call void asm sideeffect "stmxcsr $0\0A\09fnstcw $1", "*m,*m,~{dirflag},~{fpsr},~{flags}"(i32* nonnull %129, i16* nonnull %130) #6
  %131 = getelementptr inbounds %struct.__cilkrts_stack_frame, %struct.__cilkrts_stack_frame* %__cilkrts_sf, i64 0, i32 5
  %132 = call i8* @llvm.frameaddress(i32 0) #6
  %133 = getelementptr inbounds [5 x i8*], [5 x i8*]* %131, i64 0, i64 0
  store volatile i8* %132, i8** %133, align 8
  %134 = call i8* @llvm.stacksave() #6
  %135 = getelementptr inbounds %struct.__cilkrts_stack_frame, %struct.__cilkrts_stack_frame* %__cilkrts_sf, i64 0, i32 5, i64 2
  store volatile i8* %134, i8** %135, align 8
  %136 = bitcast [5 x i8*]* %131 to i8*
  %137 = call i32 @llvm.eh.sjlj.setjmp(i8* nonnull %136) #8
  %138 = icmp eq i32 %137, 0
  br i1 %138, label %cilk.sync.runtimecall.i, label %__cilk_sync_nothrow.exit

cilk.sync.runtimecall.i:                          ; preds = %cilk.sync.savestate.i
  call void @__cilkrts_sync(%struct.__cilkrts_stack_frame* nonnull %__cilkrts_sf) #6
  br label %__cilk_sync_nothrow.exit

__cilk_sync_nothrow.exit:                         ; preds = %dot.10.loop_sync.rhs.1.ls2, %cilk.sync.savestate.i, %cilk.sync.runtimecall.i
  %139 = load atomic %struct.__cilkrts_worker*, %struct.__cilkrts_worker** %9 acquire, align 8
  %140 = getelementptr inbounds %struct.__cilkrts_worker, %struct.__cilkrts_worker* %139, i64 0, i32 12, i32 0
  %141 = load i64, i64* %140, align 8
  %142 = add i64 %141, 1
  store i64 %142, i64* %140, align 8
  %143 = load atomic %struct.__cilkrts_worker*, %struct.__cilkrts_worker** %9 acquire, align 8
  %144 = bitcast %struct.__cilkrts_stack_frame** %8 to i64*
  %145 = load i64, i64* %144, align 8
  %146 = getelementptr inbounds %struct.__cilkrts_worker, %struct.__cilkrts_worker* %143, i64 0, i32 9
  %147 = bitcast %struct.__cilkrts_stack_frame** %146 to i64*
  store atomic i64 %145, i64* %147 release, align 8
  store atomic %struct.__cilkrts_stack_frame* null, %struct.__cilkrts_stack_frame** %8 release, align 8
  %148 = load atomic i32, i32* %122 acquire, align 8
  %149 = icmp eq i32 %148, 16777216
  br i1 %149, label %__cilk_parent_epilogue.exit, label %body.i

body.i:                                           ; preds = %__cilk_sync_nothrow.exit
  call void @__cilkrts_leave_frame(%struct.__cilkrts_stack_frame* nonnull %__cilkrts_sf) #6
  br label %__cilk_parent_epilogue.exit

__cilk_parent_epilogue.exit:                      ; preds = %__cilk_sync_nothrow.exit, %body.i
  ret void

dot.10.loop_detach.rhs.1.ls2:                     ; preds = %dot.10.loop_detach.rhs.1.ls2.preheader, %dot.10.loop_detach.rhs.1.ls2
  %dot.10.indvar_address.rhs.1.038.ls2 = phi i64 [ %indvar.inc3.ls2, %dot.10.loop_detach.rhs.1.ls2 ], [ %dot.10.indvar_address.rhs.1.038.ls2.dac.lcssa, %dot.10.loop_detach.rhs.1.ls2.preheader ]
  %150 = getelementptr inbounds [400 x [800 x float]], [400 x [800 x float]]* %.ls2, i64 0, <8 x i64> <i64 0, i64 1, i64 2, i64 3, i64 4, i64 5, i64 6, i64 7>, i64 %dot.10.indvar_address.rhs.1.038.ls2
  %wide.masked.gather.ls2 = tail call <8 x float> @llvm.masked.gather.v8f32.v8p0f32(<8 x float*> %150, i32 4, <8 x i1> <i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true>, <8 x float> undef), !invariant.load !0, !noalias !30
  %151 = fmul fast <8 x float> %wide.masked.gather.ls2, %wide.load.ls2
  %152 = getelementptr inbounds [400 x [800 x float]], [400 x [800 x float]]* %.ls2, i64 0, <8 x i64> <i64 8, i64 9, i64 10, i64 11, i64 12, i64 13, i64 14, i64 15>, i64 %dot.10.indvar_address.rhs.1.038.ls2
  %wide.masked.gather.1.ls2 = tail call <8 x float> @llvm.masked.gather.v8f32.v8p0f32(<8 x float*> %152, i32 4, <8 x i1> <i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true>, <8 x float> undef), !invariant.load !0, !noalias !30
  %153 = fmul fast <8 x float> %wide.masked.gather.1.ls2, %wide.load.1.ls2
  %154 = fadd fast <8 x float> %153, %151
  %155 = getelementptr inbounds [400 x [800 x float]], [400 x [800 x float]]* %.ls2, i64 0, <8 x i64> <i64 16, i64 17, i64 18, i64 19, i64 20, i64 21, i64 22, i64 23>, i64 %dot.10.indvar_address.rhs.1.038.ls2
  %wide.masked.gather.2.ls2 = tail call <8 x float> @llvm.masked.gather.v8f32.v8p0f32(<8 x float*> %155, i32 4, <8 x i1> <i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true>, <8 x float> undef), !invariant.load !0, !noalias !30
  %156 = fmul fast <8 x float> %wide.masked.gather.2.ls2, %wide.load.2.ls2
  %157 = fadd fast <8 x float> %154, %156
  %158 = getelementptr inbounds [400 x [800 x float]], [400 x [800 x float]]* %.ls2, i64 0, <8 x i64> <i64 24, i64 25, i64 26, i64 27, i64 28, i64 29, i64 30, i64 31>, i64 %dot.10.indvar_address.rhs.1.038.ls2
  %wide.masked.gather.3.ls2 = tail call <8 x float> @llvm.masked.gather.v8f32.v8p0f32(<8 x float*> %158, i32 4, <8 x i1> <i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true>, <8 x float> undef), !invariant.load !0, !noalias !30
  %159 = fmul fast <8 x float> %wide.masked.gather.3.ls2, %wide.load.3.ls2
  %160 = fadd fast <8 x float> %157, %159
  %161 = getelementptr inbounds [400 x [800 x float]], [400 x [800 x float]]* %.ls2, i64 0, <8 x i64> <i64 32, i64 33, i64 34, i64 35, i64 36, i64 37, i64 38, i64 39>, i64 %dot.10.indvar_address.rhs.1.038.ls2
  %wide.masked.gather.4.ls2 = tail call <8 x float> @llvm.masked.gather.v8f32.v8p0f32(<8 x float*> %161, i32 4, <8 x i1> <i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true>, <8 x float> undef), !invariant.load !0, !noalias !30
  %162 = fmul fast <8 x float> %wide.masked.gather.4.ls2, %wide.load.4.ls2
  %163 = fadd fast <8 x float> %160, %162
  %164 = getelementptr inbounds [400 x [800 x float]], [400 x [800 x float]]* %.ls2, i64 0, <8 x i64> <i64 40, i64 41, i64 42, i64 43, i64 44, i64 45, i64 46, i64 47>, i64 %dot.10.indvar_address.rhs.1.038.ls2
  %wide.masked.gather.ls2.1 = tail call <8 x float> @llvm.masked.gather.v8f32.v8p0f32(<8 x float*> %164, i32 4, <8 x i1> <i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true>, <8 x float> undef), !invariant.load !0, !noalias !30
  %165 = fmul fast <8 x float> %wide.masked.gather.ls2.1, %wide.load.ls2.1
  %166 = fadd fast <8 x float> %163, %165
  %167 = getelementptr inbounds [400 x [800 x float]], [400 x [800 x float]]* %.ls2, i64 0, <8 x i64> <i64 48, i64 49, i64 50, i64 51, i64 52, i64 53, i64 54, i64 55>, i64 %dot.10.indvar_address.rhs.1.038.ls2
  %wide.masked.gather.1.ls2.1 = tail call <8 x float> @llvm.masked.gather.v8f32.v8p0f32(<8 x float*> %167, i32 4, <8 x i1> <i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true>, <8 x float> undef), !invariant.load !0, !noalias !30
  %168 = fmul fast <8 x float> %wide.masked.gather.1.ls2.1, %wide.load.1.ls2.1
  %169 = fadd fast <8 x float> %166, %168
  %170 = getelementptr inbounds [400 x [800 x float]], [400 x [800 x float]]* %.ls2, i64 0, <8 x i64> <i64 56, i64 57, i64 58, i64 59, i64 60, i64 61, i64 62, i64 63>, i64 %dot.10.indvar_address.rhs.1.038.ls2
  %wide.masked.gather.2.ls2.1 = tail call <8 x float> @llvm.masked.gather.v8f32.v8p0f32(<8 x float*> %170, i32 4, <8 x i1> <i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true>, <8 x float> undef), !invariant.load !0, !noalias !30
  %171 = fmul fast <8 x float> %wide.masked.gather.2.ls2.1, %wide.load.2.ls2.1
  %172 = fadd fast <8 x float> %169, %171
  %173 = getelementptr inbounds [400 x [800 x float]], [400 x [800 x float]]* %.ls2, i64 0, <8 x i64> <i64 64, i64 65, i64 66, i64 67, i64 68, i64 69, i64 70, i64 71>, i64 %dot.10.indvar_address.rhs.1.038.ls2
  %wide.masked.gather.3.ls2.1 = tail call <8 x float> @llvm.masked.gather.v8f32.v8p0f32(<8 x float*> %173, i32 4, <8 x i1> <i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true>, <8 x float> undef), !invariant.load !0, !noalias !30
  %174 = fmul fast <8 x float> %wide.masked.gather.3.ls2.1, %wide.load.3.ls2.1
  %175 = fadd fast <8 x float> %172, %174
  %176 = getelementptr inbounds [400 x [800 x float]], [400 x [800 x float]]* %.ls2, i64 0, <8 x i64> <i64 72, i64 73, i64 74, i64 75, i64 76, i64 77, i64 78, i64 79>, i64 %dot.10.indvar_address.rhs.1.038.ls2
  %wide.masked.gather.4.ls2.1 = tail call <8 x float> @llvm.masked.gather.v8f32.v8p0f32(<8 x float*> %176, i32 4, <8 x i1> <i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true>, <8 x float> undef), !invariant.load !0, !noalias !30
  %177 = fmul fast <8 x float> %wide.masked.gather.4.ls2.1, %wide.load.4.ls2.1
  %178 = fadd fast <8 x float> %175, %177
  %179 = getelementptr inbounds [400 x [800 x float]], [400 x [800 x float]]* %.ls2, i64 0, <8 x i64> <i64 80, i64 81, i64 82, i64 83, i64 84, i64 85, i64 86, i64 87>, i64 %dot.10.indvar_address.rhs.1.038.ls2
  %wide.masked.gather.ls2.2 = tail call <8 x float> @llvm.masked.gather.v8f32.v8p0f32(<8 x float*> %179, i32 4, <8 x i1> <i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true>, <8 x float> undef), !invariant.load !0, !noalias !30
  %180 = fmul fast <8 x float> %wide.masked.gather.ls2.2, %wide.load.ls2.2
  %181 = fadd fast <8 x float> %178, %180
  %182 = getelementptr inbounds [400 x [800 x float]], [400 x [800 x float]]* %.ls2, i64 0, <8 x i64> <i64 88, i64 89, i64 90, i64 91, i64 92, i64 93, i64 94, i64 95>, i64 %dot.10.indvar_address.rhs.1.038.ls2
  %wide.masked.gather.1.ls2.2 = tail call <8 x float> @llvm.masked.gather.v8f32.v8p0f32(<8 x float*> %182, i32 4, <8 x i1> <i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true>, <8 x float> undef), !invariant.load !0, !noalias !30
  %183 = fmul fast <8 x float> %wide.masked.gather.1.ls2.2, %wide.load.1.ls2.2
  %184 = fadd fast <8 x float> %181, %183
  %185 = getelementptr inbounds [400 x [800 x float]], [400 x [800 x float]]* %.ls2, i64 0, <8 x i64> <i64 96, i64 97, i64 98, i64 99, i64 100, i64 101, i64 102, i64 103>, i64 %dot.10.indvar_address.rhs.1.038.ls2
  %wide.masked.gather.2.ls2.2 = tail call <8 x float> @llvm.masked.gather.v8f32.v8p0f32(<8 x float*> %185, i32 4, <8 x i1> <i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true>, <8 x float> undef), !invariant.load !0, !noalias !30
  %186 = fmul fast <8 x float> %wide.masked.gather.2.ls2.2, %wide.load.2.ls2.2
  %187 = fadd fast <8 x float> %184, %186
  %188 = getelementptr inbounds [400 x [800 x float]], [400 x [800 x float]]* %.ls2, i64 0, <8 x i64> <i64 104, i64 105, i64 106, i64 107, i64 108, i64 109, i64 110, i64 111>, i64 %dot.10.indvar_address.rhs.1.038.ls2
  %wide.masked.gather.3.ls2.2 = tail call <8 x float> @llvm.masked.gather.v8f32.v8p0f32(<8 x float*> %188, i32 4, <8 x i1> <i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true>, <8 x float> undef), !invariant.load !0, !noalias !30
  %189 = fmul fast <8 x float> %wide.masked.gather.3.ls2.2, %wide.load.3.ls2.2
  %190 = fadd fast <8 x float> %187, %189
  %191 = getelementptr inbounds [400 x [800 x float]], [400 x [800 x float]]* %.ls2, i64 0, <8 x i64> <i64 112, i64 113, i64 114, i64 115, i64 116, i64 117, i64 118, i64 119>, i64 %dot.10.indvar_address.rhs.1.038.ls2
  %wide.masked.gather.4.ls2.2 = tail call <8 x float> @llvm.masked.gather.v8f32.v8p0f32(<8 x float*> %191, i32 4, <8 x i1> <i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true>, <8 x float> undef), !invariant.load !0, !noalias !30
  %192 = fmul fast <8 x float> %wide.masked.gather.4.ls2.2, %wide.load.4.ls2.2
  %193 = fadd fast <8 x float> %190, %192
  %194 = getelementptr inbounds [400 x [800 x float]], [400 x [800 x float]]* %.ls2, i64 0, <8 x i64> <i64 120, i64 121, i64 122, i64 123, i64 124, i64 125, i64 126, i64 127>, i64 %dot.10.indvar_address.rhs.1.038.ls2
  %wide.masked.gather.ls2.3 = tail call <8 x float> @llvm.masked.gather.v8f32.v8p0f32(<8 x float*> %194, i32 4, <8 x i1> <i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true>, <8 x float> undef), !invariant.load !0, !noalias !30
  %195 = fmul fast <8 x float> %wide.masked.gather.ls2.3, %wide.load.ls2.3
  %196 = fadd fast <8 x float> %193, %195
  %197 = getelementptr inbounds [400 x [800 x float]], [400 x [800 x float]]* %.ls2, i64 0, <8 x i64> <i64 128, i64 129, i64 130, i64 131, i64 132, i64 133, i64 134, i64 135>, i64 %dot.10.indvar_address.rhs.1.038.ls2
  %wide.masked.gather.1.ls2.3 = tail call <8 x float> @llvm.masked.gather.v8f32.v8p0f32(<8 x float*> %197, i32 4, <8 x i1> <i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true>, <8 x float> undef), !invariant.load !0, !noalias !30
  %198 = fmul fast <8 x float> %wide.masked.gather.1.ls2.3, %wide.load.1.ls2.3
  %199 = fadd fast <8 x float> %196, %198
  %200 = getelementptr inbounds [400 x [800 x float]], [400 x [800 x float]]* %.ls2, i64 0, <8 x i64> <i64 136, i64 137, i64 138, i64 139, i64 140, i64 141, i64 142, i64 143>, i64 %dot.10.indvar_address.rhs.1.038.ls2
  %wide.masked.gather.2.ls2.3 = tail call <8 x float> @llvm.masked.gather.v8f32.v8p0f32(<8 x float*> %200, i32 4, <8 x i1> <i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true>, <8 x float> undef), !invariant.load !0, !noalias !30
  %201 = fmul fast <8 x float> %wide.masked.gather.2.ls2.3, %wide.load.2.ls2.3
  %202 = fadd fast <8 x float> %199, %201
  %203 = getelementptr inbounds [400 x [800 x float]], [400 x [800 x float]]* %.ls2, i64 0, <8 x i64> <i64 144, i64 145, i64 146, i64 147, i64 148, i64 149, i64 150, i64 151>, i64 %dot.10.indvar_address.rhs.1.038.ls2
  %wide.masked.gather.3.ls2.3 = tail call <8 x float> @llvm.masked.gather.v8f32.v8p0f32(<8 x float*> %203, i32 4, <8 x i1> <i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true>, <8 x float> undef), !invariant.load !0, !noalias !30
  %204 = fmul fast <8 x float> %wide.masked.gather.3.ls2.3, %wide.load.3.ls2.3
  %205 = fadd fast <8 x float> %202, %204
  %206 = getelementptr inbounds [400 x [800 x float]], [400 x [800 x float]]* %.ls2, i64 0, <8 x i64> <i64 152, i64 153, i64 154, i64 155, i64 156, i64 157, i64 158, i64 159>, i64 %dot.10.indvar_address.rhs.1.038.ls2
  %wide.masked.gather.4.ls2.3 = tail call <8 x float> @llvm.masked.gather.v8f32.v8p0f32(<8 x float*> %206, i32 4, <8 x i1> <i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true>, <8 x float> undef), !invariant.load !0, !noalias !30
  %207 = fmul fast <8 x float> %wide.masked.gather.4.ls2.3, %wide.load.4.ls2.3
  %208 = fadd fast <8 x float> %205, %207
  %209 = getelementptr inbounds [400 x [800 x float]], [400 x [800 x float]]* %.ls2, i64 0, <8 x i64> <i64 160, i64 161, i64 162, i64 163, i64 164, i64 165, i64 166, i64 167>, i64 %dot.10.indvar_address.rhs.1.038.ls2
  %wide.masked.gather.ls2.4 = tail call <8 x float> @llvm.masked.gather.v8f32.v8p0f32(<8 x float*> %209, i32 4, <8 x i1> <i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true>, <8 x float> undef), !invariant.load !0, !noalias !30
  %210 = fmul fast <8 x float> %wide.masked.gather.ls2.4, %wide.load.ls2.4
  %211 = fadd fast <8 x float> %208, %210
  %212 = getelementptr inbounds [400 x [800 x float]], [400 x [800 x float]]* %.ls2, i64 0, <8 x i64> <i64 168, i64 169, i64 170, i64 171, i64 172, i64 173, i64 174, i64 175>, i64 %dot.10.indvar_address.rhs.1.038.ls2
  %wide.masked.gather.1.ls2.4 = tail call <8 x float> @llvm.masked.gather.v8f32.v8p0f32(<8 x float*> %212, i32 4, <8 x i1> <i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true>, <8 x float> undef), !invariant.load !0, !noalias !30
  %213 = fmul fast <8 x float> %wide.masked.gather.1.ls2.4, %wide.load.1.ls2.4
  %214 = fadd fast <8 x float> %211, %213
  %215 = getelementptr inbounds [400 x [800 x float]], [400 x [800 x float]]* %.ls2, i64 0, <8 x i64> <i64 176, i64 177, i64 178, i64 179, i64 180, i64 181, i64 182, i64 183>, i64 %dot.10.indvar_address.rhs.1.038.ls2
  %wide.masked.gather.2.ls2.4 = tail call <8 x float> @llvm.masked.gather.v8f32.v8p0f32(<8 x float*> %215, i32 4, <8 x i1> <i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true>, <8 x float> undef), !invariant.load !0, !noalias !30
  %216 = fmul fast <8 x float> %wide.masked.gather.2.ls2.4, %wide.load.2.ls2.4
  %217 = fadd fast <8 x float> %214, %216
  %218 = getelementptr inbounds [400 x [800 x float]], [400 x [800 x float]]* %.ls2, i64 0, <8 x i64> <i64 184, i64 185, i64 186, i64 187, i64 188, i64 189, i64 190, i64 191>, i64 %dot.10.indvar_address.rhs.1.038.ls2
  %wide.masked.gather.3.ls2.4 = tail call <8 x float> @llvm.masked.gather.v8f32.v8p0f32(<8 x float*> %218, i32 4, <8 x i1> <i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true>, <8 x float> undef), !invariant.load !0, !noalias !30
  %219 = fmul fast <8 x float> %wide.masked.gather.3.ls2.4, %wide.load.3.ls2.4
  %220 = fadd fast <8 x float> %217, %219
  %221 = getelementptr inbounds [400 x [800 x float]], [400 x [800 x float]]* %.ls2, i64 0, <8 x i64> <i64 192, i64 193, i64 194, i64 195, i64 196, i64 197, i64 198, i64 199>, i64 %dot.10.indvar_address.rhs.1.038.ls2
  %wide.masked.gather.4.ls2.4 = tail call <8 x float> @llvm.masked.gather.v8f32.v8p0f32(<8 x float*> %221, i32 4, <8 x i1> <i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true>, <8 x float> undef), !invariant.load !0, !noalias !30
  %222 = fmul fast <8 x float> %wide.masked.gather.4.ls2.4, %wide.load.4.ls2.4
  %223 = fadd fast <8 x float> %220, %222
  %224 = getelementptr inbounds [400 x [800 x float]], [400 x [800 x float]]* %.ls2, i64 0, <8 x i64> <i64 200, i64 201, i64 202, i64 203, i64 204, i64 205, i64 206, i64 207>, i64 %dot.10.indvar_address.rhs.1.038.ls2
  %wide.masked.gather.ls2.5 = tail call <8 x float> @llvm.masked.gather.v8f32.v8p0f32(<8 x float*> %224, i32 4, <8 x i1> <i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true>, <8 x float> undef), !invariant.load !0, !noalias !30
  %225 = fmul fast <8 x float> %wide.masked.gather.ls2.5, %wide.load.ls2.5
  %226 = fadd fast <8 x float> %223, %225
  %227 = getelementptr inbounds [400 x [800 x float]], [400 x [800 x float]]* %.ls2, i64 0, <8 x i64> <i64 208, i64 209, i64 210, i64 211, i64 212, i64 213, i64 214, i64 215>, i64 %dot.10.indvar_address.rhs.1.038.ls2
  %wide.masked.gather.1.ls2.5 = tail call <8 x float> @llvm.masked.gather.v8f32.v8p0f32(<8 x float*> %227, i32 4, <8 x i1> <i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true>, <8 x float> undef), !invariant.load !0, !noalias !30
  %228 = fmul fast <8 x float> %wide.masked.gather.1.ls2.5, %wide.load.1.ls2.5
  %229 = fadd fast <8 x float> %226, %228
  %230 = getelementptr inbounds [400 x [800 x float]], [400 x [800 x float]]* %.ls2, i64 0, <8 x i64> <i64 216, i64 217, i64 218, i64 219, i64 220, i64 221, i64 222, i64 223>, i64 %dot.10.indvar_address.rhs.1.038.ls2
  %wide.masked.gather.2.ls2.5 = tail call <8 x float> @llvm.masked.gather.v8f32.v8p0f32(<8 x float*> %230, i32 4, <8 x i1> <i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true>, <8 x float> undef), !invariant.load !0, !noalias !30
  %231 = fmul fast <8 x float> %wide.masked.gather.2.ls2.5, %wide.load.2.ls2.5
  %232 = fadd fast <8 x float> %229, %231
  %233 = getelementptr inbounds [400 x [800 x float]], [400 x [800 x float]]* %.ls2, i64 0, <8 x i64> <i64 224, i64 225, i64 226, i64 227, i64 228, i64 229, i64 230, i64 231>, i64 %dot.10.indvar_address.rhs.1.038.ls2
  %wide.masked.gather.3.ls2.5 = tail call <8 x float> @llvm.masked.gather.v8f32.v8p0f32(<8 x float*> %233, i32 4, <8 x i1> <i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true>, <8 x float> undef), !invariant.load !0, !noalias !30
  %234 = fmul fast <8 x float> %wide.masked.gather.3.ls2.5, %wide.load.3.ls2.5
  %235 = fadd fast <8 x float> %232, %234
  %236 = getelementptr inbounds [400 x [800 x float]], [400 x [800 x float]]* %.ls2, i64 0, <8 x i64> <i64 232, i64 233, i64 234, i64 235, i64 236, i64 237, i64 238, i64 239>, i64 %dot.10.indvar_address.rhs.1.038.ls2
  %wide.masked.gather.4.ls2.5 = tail call <8 x float> @llvm.masked.gather.v8f32.v8p0f32(<8 x float*> %236, i32 4, <8 x i1> <i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true>, <8 x float> undef), !invariant.load !0, !noalias !30
  %237 = fmul fast <8 x float> %wide.masked.gather.4.ls2.5, %wide.load.4.ls2.5
  %238 = fadd fast <8 x float> %235, %237
  %239 = getelementptr inbounds [400 x [800 x float]], [400 x [800 x float]]* %.ls2, i64 0, <8 x i64> <i64 240, i64 241, i64 242, i64 243, i64 244, i64 245, i64 246, i64 247>, i64 %dot.10.indvar_address.rhs.1.038.ls2
  %wide.masked.gather.ls2.6 = tail call <8 x float> @llvm.masked.gather.v8f32.v8p0f32(<8 x float*> %239, i32 4, <8 x i1> <i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true>, <8 x float> undef), !invariant.load !0, !noalias !30
  %240 = fmul fast <8 x float> %wide.masked.gather.ls2.6, %wide.load.ls2.6
  %241 = fadd fast <8 x float> %238, %240
  %242 = getelementptr inbounds [400 x [800 x float]], [400 x [800 x float]]* %.ls2, i64 0, <8 x i64> <i64 248, i64 249, i64 250, i64 251, i64 252, i64 253, i64 254, i64 255>, i64 %dot.10.indvar_address.rhs.1.038.ls2
  %wide.masked.gather.1.ls2.6 = tail call <8 x float> @llvm.masked.gather.v8f32.v8p0f32(<8 x float*> %242, i32 4, <8 x i1> <i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true>, <8 x float> undef), !invariant.load !0, !noalias !30
  %243 = fmul fast <8 x float> %wide.masked.gather.1.ls2.6, %wide.load.1.ls2.6
  %244 = fadd fast <8 x float> %241, %243
  %245 = getelementptr inbounds [400 x [800 x float]], [400 x [800 x float]]* %.ls2, i64 0, <8 x i64> <i64 256, i64 257, i64 258, i64 259, i64 260, i64 261, i64 262, i64 263>, i64 %dot.10.indvar_address.rhs.1.038.ls2
  %wide.masked.gather.2.ls2.6 = tail call <8 x float> @llvm.masked.gather.v8f32.v8p0f32(<8 x float*> %245, i32 4, <8 x i1> <i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true>, <8 x float> undef), !invariant.load !0, !noalias !30
  %246 = fmul fast <8 x float> %wide.masked.gather.2.ls2.6, %wide.load.2.ls2.6
  %247 = fadd fast <8 x float> %244, %246
  %248 = getelementptr inbounds [400 x [800 x float]], [400 x [800 x float]]* %.ls2, i64 0, <8 x i64> <i64 264, i64 265, i64 266, i64 267, i64 268, i64 269, i64 270, i64 271>, i64 %dot.10.indvar_address.rhs.1.038.ls2
  %wide.masked.gather.3.ls2.6 = tail call <8 x float> @llvm.masked.gather.v8f32.v8p0f32(<8 x float*> %248, i32 4, <8 x i1> <i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true>, <8 x float> undef), !invariant.load !0, !noalias !30
  %249 = fmul fast <8 x float> %wide.masked.gather.3.ls2.6, %wide.load.3.ls2.6
  %250 = fadd fast <8 x float> %247, %249
  %251 = getelementptr inbounds [400 x [800 x float]], [400 x [800 x float]]* %.ls2, i64 0, <8 x i64> <i64 272, i64 273, i64 274, i64 275, i64 276, i64 277, i64 278, i64 279>, i64 %dot.10.indvar_address.rhs.1.038.ls2
  %wide.masked.gather.4.ls2.6 = tail call <8 x float> @llvm.masked.gather.v8f32.v8p0f32(<8 x float*> %251, i32 4, <8 x i1> <i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true>, <8 x float> undef), !invariant.load !0, !noalias !30
  %252 = fmul fast <8 x float> %wide.masked.gather.4.ls2.6, %wide.load.4.ls2.6
  %253 = fadd fast <8 x float> %250, %252
  %254 = getelementptr inbounds [400 x [800 x float]], [400 x [800 x float]]* %.ls2, i64 0, <8 x i64> <i64 280, i64 281, i64 282, i64 283, i64 284, i64 285, i64 286, i64 287>, i64 %dot.10.indvar_address.rhs.1.038.ls2
  %wide.masked.gather.ls2.7 = tail call <8 x float> @llvm.masked.gather.v8f32.v8p0f32(<8 x float*> %254, i32 4, <8 x i1> <i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true>, <8 x float> undef), !invariant.load !0, !noalias !30
  %255 = fmul fast <8 x float> %wide.masked.gather.ls2.7, %wide.load.ls2.7
  %256 = fadd fast <8 x float> %253, %255
  %257 = getelementptr inbounds [400 x [800 x float]], [400 x [800 x float]]* %.ls2, i64 0, <8 x i64> <i64 288, i64 289, i64 290, i64 291, i64 292, i64 293, i64 294, i64 295>, i64 %dot.10.indvar_address.rhs.1.038.ls2
  %wide.masked.gather.1.ls2.7 = tail call <8 x float> @llvm.masked.gather.v8f32.v8p0f32(<8 x float*> %257, i32 4, <8 x i1> <i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true>, <8 x float> undef), !invariant.load !0, !noalias !30
  %258 = fmul fast <8 x float> %wide.masked.gather.1.ls2.7, %wide.load.1.ls2.7
  %259 = fadd fast <8 x float> %256, %258
  %260 = getelementptr inbounds [400 x [800 x float]], [400 x [800 x float]]* %.ls2, i64 0, <8 x i64> <i64 296, i64 297, i64 298, i64 299, i64 300, i64 301, i64 302, i64 303>, i64 %dot.10.indvar_address.rhs.1.038.ls2
  %wide.masked.gather.2.ls2.7 = tail call <8 x float> @llvm.masked.gather.v8f32.v8p0f32(<8 x float*> %260, i32 4, <8 x i1> <i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true>, <8 x float> undef), !invariant.load !0, !noalias !30
  %261 = fmul fast <8 x float> %wide.masked.gather.2.ls2.7, %wide.load.2.ls2.7
  %262 = fadd fast <8 x float> %259, %261
  %263 = getelementptr inbounds [400 x [800 x float]], [400 x [800 x float]]* %.ls2, i64 0, <8 x i64> <i64 304, i64 305, i64 306, i64 307, i64 308, i64 309, i64 310, i64 311>, i64 %dot.10.indvar_address.rhs.1.038.ls2
  %wide.masked.gather.3.ls2.7 = tail call <8 x float> @llvm.masked.gather.v8f32.v8p0f32(<8 x float*> %263, i32 4, <8 x i1> <i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true>, <8 x float> undef), !invariant.load !0, !noalias !30
  %264 = fmul fast <8 x float> %wide.masked.gather.3.ls2.7, %wide.load.3.ls2.7
  %265 = fadd fast <8 x float> %262, %264
  %266 = getelementptr inbounds [400 x [800 x float]], [400 x [800 x float]]* %.ls2, i64 0, <8 x i64> <i64 312, i64 313, i64 314, i64 315, i64 316, i64 317, i64 318, i64 319>, i64 %dot.10.indvar_address.rhs.1.038.ls2
  %wide.masked.gather.4.ls2.7 = tail call <8 x float> @llvm.masked.gather.v8f32.v8p0f32(<8 x float*> %266, i32 4, <8 x i1> <i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true>, <8 x float> undef), !invariant.load !0, !noalias !30
  %267 = fmul fast <8 x float> %wide.masked.gather.4.ls2.7, %wide.load.4.ls2.7
  %268 = fadd fast <8 x float> %265, %267
  %269 = getelementptr inbounds [400 x [800 x float]], [400 x [800 x float]]* %.ls2, i64 0, <8 x i64> <i64 320, i64 321, i64 322, i64 323, i64 324, i64 325, i64 326, i64 327>, i64 %dot.10.indvar_address.rhs.1.038.ls2
  %wide.masked.gather.ls2.8 = tail call <8 x float> @llvm.masked.gather.v8f32.v8p0f32(<8 x float*> %269, i32 4, <8 x i1> <i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true>, <8 x float> undef), !invariant.load !0, !noalias !30
  %270 = fmul fast <8 x float> %wide.masked.gather.ls2.8, %wide.load.ls2.8
  %271 = fadd fast <8 x float> %268, %270
  %272 = getelementptr inbounds [400 x [800 x float]], [400 x [800 x float]]* %.ls2, i64 0, <8 x i64> <i64 328, i64 329, i64 330, i64 331, i64 332, i64 333, i64 334, i64 335>, i64 %dot.10.indvar_address.rhs.1.038.ls2
  %wide.masked.gather.1.ls2.8 = tail call <8 x float> @llvm.masked.gather.v8f32.v8p0f32(<8 x float*> %272, i32 4, <8 x i1> <i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true>, <8 x float> undef), !invariant.load !0, !noalias !30
  %273 = fmul fast <8 x float> %wide.masked.gather.1.ls2.8, %wide.load.1.ls2.8
  %274 = fadd fast <8 x float> %271, %273
  %275 = getelementptr inbounds [400 x [800 x float]], [400 x [800 x float]]* %.ls2, i64 0, <8 x i64> <i64 336, i64 337, i64 338, i64 339, i64 340, i64 341, i64 342, i64 343>, i64 %dot.10.indvar_address.rhs.1.038.ls2
  %wide.masked.gather.2.ls2.8 = tail call <8 x float> @llvm.masked.gather.v8f32.v8p0f32(<8 x float*> %275, i32 4, <8 x i1> <i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true>, <8 x float> undef), !invariant.load !0, !noalias !30
  %276 = fmul fast <8 x float> %wide.masked.gather.2.ls2.8, %wide.load.2.ls2.8
  %277 = fadd fast <8 x float> %274, %276
  %278 = getelementptr inbounds [400 x [800 x float]], [400 x [800 x float]]* %.ls2, i64 0, <8 x i64> <i64 344, i64 345, i64 346, i64 347, i64 348, i64 349, i64 350, i64 351>, i64 %dot.10.indvar_address.rhs.1.038.ls2
  %wide.masked.gather.3.ls2.8 = tail call <8 x float> @llvm.masked.gather.v8f32.v8p0f32(<8 x float*> %278, i32 4, <8 x i1> <i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true>, <8 x float> undef), !invariant.load !0, !noalias !30
  %279 = fmul fast <8 x float> %wide.masked.gather.3.ls2.8, %wide.load.3.ls2.8
  %280 = fadd fast <8 x float> %277, %279
  %281 = getelementptr inbounds [400 x [800 x float]], [400 x [800 x float]]* %.ls2, i64 0, <8 x i64> <i64 352, i64 353, i64 354, i64 355, i64 356, i64 357, i64 358, i64 359>, i64 %dot.10.indvar_address.rhs.1.038.ls2
  %wide.masked.gather.4.ls2.8 = tail call <8 x float> @llvm.masked.gather.v8f32.v8p0f32(<8 x float*> %281, i32 4, <8 x i1> <i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true>, <8 x float> undef), !invariant.load !0, !noalias !30
  %282 = fmul fast <8 x float> %wide.masked.gather.4.ls2.8, %wide.load.4.ls2.8
  %283 = fadd fast <8 x float> %280, %282
  %284 = getelementptr inbounds [400 x [800 x float]], [400 x [800 x float]]* %.ls2, i64 0, <8 x i64> <i64 360, i64 361, i64 362, i64 363, i64 364, i64 365, i64 366, i64 367>, i64 %dot.10.indvar_address.rhs.1.038.ls2
  %wide.masked.gather.ls2.9 = tail call <8 x float> @llvm.masked.gather.v8f32.v8p0f32(<8 x float*> %284, i32 4, <8 x i1> <i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true>, <8 x float> undef), !invariant.load !0, !noalias !30
  %285 = fmul fast <8 x float> %wide.masked.gather.ls2.9, %wide.load.ls2.9
  %286 = fadd fast <8 x float> %283, %285
  %287 = getelementptr inbounds [400 x [800 x float]], [400 x [800 x float]]* %.ls2, i64 0, <8 x i64> <i64 368, i64 369, i64 370, i64 371, i64 372, i64 373, i64 374, i64 375>, i64 %dot.10.indvar_address.rhs.1.038.ls2
  %wide.masked.gather.1.ls2.9 = tail call <8 x float> @llvm.masked.gather.v8f32.v8p0f32(<8 x float*> %287, i32 4, <8 x i1> <i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true>, <8 x float> undef), !invariant.load !0, !noalias !30
  %288 = fmul fast <8 x float> %wide.masked.gather.1.ls2.9, %wide.load.1.ls2.9
  %289 = fadd fast <8 x float> %286, %288
  %290 = getelementptr inbounds [400 x [800 x float]], [400 x [800 x float]]* %.ls2, i64 0, <8 x i64> <i64 376, i64 377, i64 378, i64 379, i64 380, i64 381, i64 382, i64 383>, i64 %dot.10.indvar_address.rhs.1.038.ls2
  %wide.masked.gather.2.ls2.9 = tail call <8 x float> @llvm.masked.gather.v8f32.v8p0f32(<8 x float*> %290, i32 4, <8 x i1> <i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true>, <8 x float> undef), !invariant.load !0, !noalias !30
  %291 = fmul fast <8 x float> %wide.masked.gather.2.ls2.9, %wide.load.2.ls2.9
  %292 = fadd fast <8 x float> %289, %291
  %293 = getelementptr inbounds [400 x [800 x float]], [400 x [800 x float]]* %.ls2, i64 0, <8 x i64> <i64 384, i64 385, i64 386, i64 387, i64 388, i64 389, i64 390, i64 391>, i64 %dot.10.indvar_address.rhs.1.038.ls2
  %wide.masked.gather.3.ls2.9 = tail call <8 x float> @llvm.masked.gather.v8f32.v8p0f32(<8 x float*> %293, i32 4, <8 x i1> <i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true>, <8 x float> undef), !invariant.load !0, !noalias !30
  %294 = fmul fast <8 x float> %wide.masked.gather.3.ls2.9, %wide.load.3.ls2.9
  %295 = fadd fast <8 x float> %292, %294
  %296 = getelementptr inbounds [400 x [800 x float]], [400 x [800 x float]]* %.ls2, i64 0, <8 x i64> <i64 392, i64 393, i64 394, i64 395, i64 396, i64 397, i64 398, i64 399>, i64 %dot.10.indvar_address.rhs.1.038.ls2
  %wide.masked.gather.4.ls2.9 = tail call <8 x float> @llvm.masked.gather.v8f32.v8p0f32(<8 x float*> %296, i32 4, <8 x i1> <i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true>, <8 x float> undef), !invariant.load !0, !noalias !30
  %297 = fmul fast <8 x float> %wide.masked.gather.4.ls2.9, %wide.load.4.ls2.9
  %298 = fadd fast <8 x float> %295, %297
  %rdx.shuf.ls2 = shufflevector <8 x float> %298, <8 x float> undef, <8 x i32> <i32 4, i32 5, i32 6, i32 7, i32 undef, i32 undef, i32 undef, i32 undef>
  %bin.rdx.ls2 = fadd fast <8 x float> %rdx.shuf.ls2, %298
  %rdx.shuf58.ls2 = shufflevector <8 x float> %bin.rdx.ls2, <8 x float> undef, <8 x i32> <i32 2, i32 3, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  %bin.rdx59.ls2 = fadd fast <8 x float> %rdx.shuf58.ls2, %bin.rdx.ls2
  %rdx.shuf60.ls2 = shufflevector <8 x float> %bin.rdx59.ls2, <8 x float> undef, <8 x i32> <i32 1, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  %bin.rdx61.ls2 = fadd fast <8 x float> %rdx.shuf60.ls2, %bin.rdx59.ls2
  %299 = extractelement <8 x float> %bin.rdx61.ls2, i32 0
  %300 = getelementptr inbounds [20 x [800 x float]], [20 x [800 x float]]* %.ls21, i64 0, i64 %dot.10.indvar_address.lhs.0.040.ls2, i64 %dot.10.indvar_address.rhs.1.038.ls2
  store float %299, float* %300, align 4, !alias.scope !29, !noalias !23
  %indvar.inc3.ls2 = add nuw nsw i64 %dot.10.indvar_address.rhs.1.038.ls2, 1
  %exitcond55.ls2 = icmp eq i64 %indvar.inc3.ls2, %end.ls2
  br i1 %exitcond55.ls2, label %dot.10.loop_sync.rhs.1.ls2, label %dot.10.loop_detach.rhs.1.ls2, !llvm.loop !78
}

; Function Attrs: argmemonly nounwind stealable
define private fastcc void @cluster_25__XlaCompiledKernel_true__XlaNumConstantArgs_1__XlaNumResourceArgs_0_.42.outline_dot.10.loop_detach.lhs.0.ls1(i64 %dot.10.indvar_address.lhs.0.040.start.ls1, i64 %end.ls1, i64 %grainsize.ls1, [20 x [400 x float]]* nocapture readonly align 16 %concatenate.9.ls1, [400 x [800 x float]]* align 16 %.ls1, [20 x [800 x float]]* nocapture align 16 %.ls11) unnamed_addr #4 {
entry.ls1:
  %__cilkrts_sf = alloca %struct.__cilkrts_stack_frame, align 8
  %0 = call %struct.__cilkrts_worker* @__cilkrts_get_tls_worker() #6
  %1 = icmp eq %struct.__cilkrts_worker* %0, null
  br i1 %1, label %slowpath.i, label %fastpath.i

slowpath.i:                                       ; preds = %entry.ls1
  %2 = call %struct.__cilkrts_worker* @__cilkrts_bind_thread_1() #6
  %3 = getelementptr inbounds %struct.__cilkrts_stack_frame, %struct.__cilkrts_stack_frame* %__cilkrts_sf, i64 0, i32 0
  store atomic i32 16777344, i32* %3 release, align 8
  br label %__cilkrts_enter_frame_1.exit

fastpath.i:                                       ; preds = %entry.ls1
  %4 = getelementptr inbounds %struct.__cilkrts_stack_frame, %struct.__cilkrts_stack_frame* %__cilkrts_sf, i64 0, i32 0
  store atomic i32 16777216, i32* %4 release, align 8
  br label %__cilkrts_enter_frame_1.exit

__cilkrts_enter_frame_1.exit:                     ; preds = %slowpath.i, %fastpath.i
  %5 = phi %struct.__cilkrts_worker* [ %2, %slowpath.i ], [ %0, %fastpath.i ]
  %6 = getelementptr inbounds %struct.__cilkrts_worker, %struct.__cilkrts_worker* %5, i64 0, i32 9
  %7 = load atomic %struct.__cilkrts_stack_frame*, %struct.__cilkrts_stack_frame** %6 acquire, align 8
  %8 = getelementptr inbounds %struct.__cilkrts_stack_frame, %struct.__cilkrts_stack_frame* %__cilkrts_sf, i64 0, i32 2
  store atomic %struct.__cilkrts_stack_frame* %7, %struct.__cilkrts_stack_frame** %8 release, align 8
  %9 = getelementptr inbounds %struct.__cilkrts_stack_frame, %struct.__cilkrts_stack_frame* %__cilkrts_sf, i64 0, i32 3
  store atomic %struct.__cilkrts_worker* %5, %struct.__cilkrts_worker** %9 release, align 8
  store atomic %struct.__cilkrts_stack_frame* %__cilkrts_sf, %struct.__cilkrts_stack_frame** %6 release, align 8
  %itercount2 = sub i64 %end.ls1, %dot.10.indvar_address.lhs.0.040.start.ls1
  %10 = icmp ugt i64 %itercount2, %grainsize.ls1
  br i1 %10, label %.lr.ph.preheader, label %dot.10.loop_detach.lhs.0.ls1.preheader

.lr.ph.preheader:                                 ; preds = %__cilkrts_enter_frame_1.exit
  %11 = getelementptr inbounds %struct.__cilkrts_stack_frame, %struct.__cilkrts_stack_frame* %__cilkrts_sf, i64 0, i32 6
  %12 = getelementptr inbounds %struct.__cilkrts_stack_frame, %struct.__cilkrts_stack_frame* %__cilkrts_sf, i64 0, i32 7
  %13 = getelementptr inbounds %struct.__cilkrts_stack_frame, %struct.__cilkrts_stack_frame* %__cilkrts_sf, i64 0, i32 5
  %14 = getelementptr inbounds [5 x i8*], [5 x i8*]* %13, i64 0, i64 0
  %15 = getelementptr inbounds %struct.__cilkrts_stack_frame, %struct.__cilkrts_stack_frame* %__cilkrts_sf, i64 0, i32 5, i64 2
  %16 = bitcast [5 x i8*]* %13 to i8*
  br label %.lr.ph

dot.10.loop_detach.lhs.0.ls1.preheader:           ; preds = %.split.split, %__cilkrts_enter_frame_1.exit
  %dot.10.indvar_address.lhs.0.040.ls1.dac.lcssa = phi i64 [ %dot.10.indvar_address.lhs.0.040.start.ls1, %__cilkrts_enter_frame_1.exit ], [ %miditer, %.split.split ]
  %17 = call i32 @__cilkrts_get_nworkers()
  %18 = shl i32 %17, 3
  %19 = zext i32 %18 to i64
  %20 = add nuw nsw i64 %19, 799
  %21 = udiv i64 %20, %19
  %22 = icmp ult i64 %21, 2048
  %23 = select i1 %22, i64 %21, i64 2048
  br label %dot.10.loop_detach.lhs.0.ls1

.lr.ph:                                           ; preds = %.lr.ph.preheader, %.split.split
  %itercount4 = phi i64 [ %itercount, %.split.split ], [ %itercount2, %.lr.ph.preheader ]
  %dot.10.indvar_address.lhs.0.040.ls1.dac3 = phi i64 [ %miditer, %.split.split ], [ %dot.10.indvar_address.lhs.0.040.start.ls1, %.lr.ph.preheader ]
  %halfcount = lshr i64 %itercount4, 1
  %miditer = add nuw nsw i64 %dot.10.indvar_address.lhs.0.040.ls1.dac3, %halfcount
  call void asm sideeffect "stmxcsr $0\0A\09fnstcw $1", "*m,*m,~{dirflag},~{fpsr},~{flags}"(i32* nonnull %11, i16* nonnull %12) #6
  %24 = call i8* @llvm.frameaddress(i32 0)
  store volatile i8* %24, i8** %14, align 8
  %25 = call i8* @llvm.stacksave()
  store volatile i8* %25, i8** %15, align 8
  %26 = call i32 @llvm.eh.sjlj.setjmp(i8* nonnull %16) #7
  %27 = icmp eq i32 %26, 0
  br i1 %27, label %.lr.ph.split, label %.split.split

.lr.ph.split:                                     ; preds = %.lr.ph
  call fastcc void @cluster_25__XlaCompiledKernel_true__XlaNumConstantArgs_1__XlaNumResourceArgs_0_.42.outline_dot.10.loop_detach.lhs.0.ls1.outline_.split.otd1(i64 %dot.10.indvar_address.lhs.0.040.ls1.dac3, i64 %miditer, i64 %grainsize.ls1, [20 x [400 x float]]* %concatenate.9.ls1, [400 x [800 x float]]* %.ls1, [20 x [800 x float]]* %.ls11) #6
  br label %.split.split

.split.split:                                     ; preds = %.lr.ph, %.lr.ph.split
  %itercount = sub i64 %end.ls1, %miditer
  %28 = icmp ugt i64 %itercount, %grainsize.ls1
  br i1 %28, label %.lr.ph, label %dot.10.loop_detach.lhs.0.ls1.preheader

dot.10.loop_sync.lhs.0.ls1:                       ; preds = %dot.10.loop_detach.lhs.0.ls1
  %29 = getelementptr inbounds %struct.__cilkrts_stack_frame, %struct.__cilkrts_stack_frame* %__cilkrts_sf, i64 0, i32 0
  %30 = load atomic i32, i32* %29 acquire, align 8
  %31 = and i32 %30, 2
  %32 = icmp eq i32 %31, 0
  br i1 %32, label %__cilk_sync_nothrow.exit, label %cilk.sync.savestate.i

cilk.sync.savestate.i:                            ; preds = %dot.10.loop_sync.lhs.0.ls1
  %33 = load atomic %struct.__cilkrts_worker*, %struct.__cilkrts_worker** %9 acquire, align 8
  %.elt = getelementptr inbounds %struct.__cilkrts_worker, %struct.__cilkrts_worker* %33, i64 0, i32 12, i32 0
  %.unpack = load i64, i64* %.elt, align 8
  %.elt7 = getelementptr inbounds %struct.__cilkrts_worker, %struct.__cilkrts_worker* %33, i64 0, i32 12, i32 1
  %34 = bitcast %struct.__cilkrts_pedigree** %.elt7 to i64*
  %.unpack89 = load i64, i64* %34, align 8
  %.fca.0.0.gep = getelementptr inbounds %struct.__cilkrts_stack_frame, %struct.__cilkrts_stack_frame* %__cilkrts_sf, i64 0, i32 9, i32 0, i32 0
  %.fca.0.1.gep = getelementptr inbounds %struct.__cilkrts_stack_frame, %struct.__cilkrts_stack_frame* %__cilkrts_sf, i64 0, i32 9, i32 0, i32 1
  store i64 %.unpack, i64* %.fca.0.0.gep, align 8
  %35 = bitcast %struct.__cilkrts_pedigree** %.fca.0.1.gep to i64*
  store i64 %.unpack89, i64* %35, align 8
  %36 = getelementptr inbounds %struct.__cilkrts_stack_frame, %struct.__cilkrts_stack_frame* %__cilkrts_sf, i64 0, i32 6
  %37 = getelementptr inbounds %struct.__cilkrts_stack_frame, %struct.__cilkrts_stack_frame* %__cilkrts_sf, i64 0, i32 7
  call void asm sideeffect "stmxcsr $0\0A\09fnstcw $1", "*m,*m,~{dirflag},~{fpsr},~{flags}"(i32* nonnull %36, i16* nonnull %37) #6
  %38 = getelementptr inbounds %struct.__cilkrts_stack_frame, %struct.__cilkrts_stack_frame* %__cilkrts_sf, i64 0, i32 5
  %39 = call i8* @llvm.frameaddress(i32 0) #6
  %40 = getelementptr inbounds [5 x i8*], [5 x i8*]* %38, i64 0, i64 0
  store volatile i8* %39, i8** %40, align 8
  %41 = call i8* @llvm.stacksave() #6
  %42 = getelementptr inbounds %struct.__cilkrts_stack_frame, %struct.__cilkrts_stack_frame* %__cilkrts_sf, i64 0, i32 5, i64 2
  store volatile i8* %41, i8** %42, align 8
  %43 = bitcast [5 x i8*]* %38 to i8*
  %44 = call i32 @llvm.eh.sjlj.setjmp(i8* nonnull %43) #8
  %45 = icmp eq i32 %44, 0
  br i1 %45, label %cilk.sync.runtimecall.i, label %__cilk_sync_nothrow.exit

cilk.sync.runtimecall.i:                          ; preds = %cilk.sync.savestate.i
  call void @__cilkrts_sync(%struct.__cilkrts_stack_frame* nonnull %__cilkrts_sf) #6
  br label %__cilk_sync_nothrow.exit

__cilk_sync_nothrow.exit:                         ; preds = %dot.10.loop_sync.lhs.0.ls1, %cilk.sync.savestate.i, %cilk.sync.runtimecall.i
  %46 = load atomic %struct.__cilkrts_worker*, %struct.__cilkrts_worker** %9 acquire, align 8
  %47 = getelementptr inbounds %struct.__cilkrts_worker, %struct.__cilkrts_worker* %46, i64 0, i32 12, i32 0
  %48 = load i64, i64* %47, align 8
  %49 = add i64 %48, 1
  store i64 %49, i64* %47, align 8
  %50 = load atomic %struct.__cilkrts_worker*, %struct.__cilkrts_worker** %9 acquire, align 8
  %51 = bitcast %struct.__cilkrts_stack_frame** %8 to i64*
  %52 = load i64, i64* %51, align 8
  %53 = getelementptr inbounds %struct.__cilkrts_worker, %struct.__cilkrts_worker* %50, i64 0, i32 9
  %54 = bitcast %struct.__cilkrts_stack_frame** %53 to i64*
  store atomic i64 %52, i64* %54 release, align 8
  store atomic %struct.__cilkrts_stack_frame* null, %struct.__cilkrts_stack_frame** %8 release, align 8
  %55 = load atomic i32, i32* %29 acquire, align 8
  %56 = icmp eq i32 %55, 16777216
  br i1 %56, label %__cilk_parent_epilogue.exit, label %body.i

body.i:                                           ; preds = %__cilk_sync_nothrow.exit
  call void @__cilkrts_leave_frame(%struct.__cilkrts_stack_frame* nonnull %__cilkrts_sf) #6
  br label %__cilk_parent_epilogue.exit

__cilk_parent_epilogue.exit:                      ; preds = %__cilk_sync_nothrow.exit, %body.i
  ret void

dot.10.loop_detach.lhs.0.ls1:                     ; preds = %dot.10.loop_detach.lhs.0.ls1.preheader, %dot.10.loop_detach.lhs.0.ls1
  %dot.10.indvar_address.lhs.0.040.ls1 = phi i64 [ %indvar.inc2.ls1, %dot.10.loop_detach.lhs.0.ls1 ], [ %dot.10.indvar_address.lhs.0.040.ls1.dac.lcssa, %dot.10.loop_detach.lhs.0.ls1.preheader ]
  tail call fastcc void @cluster_25__XlaCompiledKernel_true__XlaNumConstantArgs_1__XlaNumResourceArgs_0_.42.outline_dot.10.loop_detach.rhs.1.ls2(i64 0, i64 800, i64 %23, [20 x [400 x float]]* %concatenate.9.ls1, i64 %dot.10.indvar_address.lhs.0.040.ls1, [400 x [800 x float]]* %.ls1, [20 x [800 x float]]* %.ls11) #6
  %indvar.inc2.ls1 = add nuw nsw i64 %dot.10.indvar_address.lhs.0.040.ls1, 1
  %exitcond56.ls1 = icmp eq i64 %indvar.inc2.ls1, %end.ls1
  br i1 %exitcond56.ls1, label %dot.10.loop_sync.lhs.0.ls1, label %dot.10.loop_detach.lhs.0.ls1, !llvm.loop !79
}

; Function Attrs: argmemonly noinline nounwind
define private fastcc void @cluster_25__XlaCompiledKernel_true__XlaNumConstantArgs_1__XlaNumResourceArgs_0_.42.outline_dot.10.loop_detach.lhs.0.ls1.outline_.split.otd1(i64 %dot.10.indvar_address.lhs.0.040.ls1.dac3.otd1, i64 %miditer.otd1, i64 %grainsize.ls1.otd1, [20 x [400 x float]]* nocapture readonly align 16 %concatenate.9.ls1.otd1, [400 x [800 x float]]* align 16 %.ls1.otd1, [20 x [800 x float]]* nocapture align 16 %.ls11.otd1) unnamed_addr #5 {
.lr.ph.otd1:
  %__cilkrts_sf = alloca %struct.__cilkrts_stack_frame, align 8
  %0 = tail call %struct.__cilkrts_worker* @__cilkrts_get_tls_worker_fast() #6
  %1 = getelementptr inbounds %struct.__cilkrts_stack_frame, %struct.__cilkrts_stack_frame* %__cilkrts_sf, i64 0, i32 0
  store atomic i32 16777216, i32* %1 release, align 8
  %2 = getelementptr inbounds %struct.__cilkrts_worker, %struct.__cilkrts_worker* %0, i64 0, i32 9
  %3 = load atomic %struct.__cilkrts_stack_frame*, %struct.__cilkrts_stack_frame** %2 acquire, align 8
  %4 = getelementptr inbounds %struct.__cilkrts_stack_frame, %struct.__cilkrts_stack_frame* %__cilkrts_sf, i64 0, i32 2
  store atomic %struct.__cilkrts_stack_frame* %3, %struct.__cilkrts_stack_frame** %4 release, align 8
  %5 = getelementptr inbounds %struct.__cilkrts_stack_frame, %struct.__cilkrts_stack_frame* %__cilkrts_sf, i64 0, i32 3
  store atomic %struct.__cilkrts_worker* %0, %struct.__cilkrts_worker** %5 release, align 8
  store atomic %struct.__cilkrts_stack_frame* %__cilkrts_sf, %struct.__cilkrts_stack_frame** %2 release, align 8
  %6 = load %struct.__cilkrts_worker*, %struct.__cilkrts_worker** %5, align 8
  %7 = load %struct.__cilkrts_stack_frame*, %struct.__cilkrts_stack_frame** %4, align 8
  %8 = getelementptr inbounds %struct.__cilkrts_worker, %struct.__cilkrts_worker* %6, i64 0, i32 0
  %9 = load atomic %struct.__cilkrts_stack_frame**, %struct.__cilkrts_stack_frame*** %8 acquire, align 8
  %.elt = getelementptr inbounds %struct.__cilkrts_worker, %struct.__cilkrts_worker* %6, i64 0, i32 12, i32 0
  %.unpack = load i64, i64* %.elt, align 8
  %.elt3 = getelementptr inbounds %struct.__cilkrts_worker, %struct.__cilkrts_worker* %6, i64 0, i32 12, i32 1
  %10 = bitcast %struct.__cilkrts_pedigree** %.elt3 to i64*
  %.unpack49 = load i64, i64* %10, align 8
  %11 = getelementptr inbounds %struct.__cilkrts_stack_frame, %struct.__cilkrts_stack_frame* %__cilkrts_sf, i64 0, i32 9
  %.fca.0.0.gep = getelementptr inbounds { %struct.__cilkrts_pedigree }, { %struct.__cilkrts_pedigree }* %11, i64 0, i32 0, i32 0
  %.fca.0.1.gep = getelementptr inbounds %struct.__cilkrts_stack_frame, %struct.__cilkrts_stack_frame* %__cilkrts_sf, i64 0, i32 9, i32 0, i32 1
  store i64 %.unpack, i64* %.fca.0.0.gep, align 8
  %12 = bitcast %struct.__cilkrts_pedigree** %.fca.0.1.gep to i64*
  store i64 %.unpack49, i64* %12, align 8
  %.repack = getelementptr inbounds %struct.__cilkrts_stack_frame, %struct.__cilkrts_stack_frame* %7, i64 0, i32 9, i32 0, i32 0
  store i64 %.unpack, i64* %.repack, align 8
  %13 = getelementptr inbounds %struct.__cilkrts_stack_frame, %struct.__cilkrts_stack_frame* %7, i64 0, i32 9, i32 0, i32 1
  %14 = bitcast %struct.__cilkrts_pedigree** %13 to i64*
  store i64 %.unpack49, i64* %14, align 8
  store atomic i64 0, i64* %.elt release, align 8
  %15 = getelementptr inbounds { %struct.__cilkrts_pedigree }, { %struct.__cilkrts_pedigree }* %11, i64 0, i32 0
  store atomic %struct.__cilkrts_pedigree* %15, %struct.__cilkrts_pedigree** %.elt3 release, align 8
  fence release
  store volatile %struct.__cilkrts_stack_frame* %7, %struct.__cilkrts_stack_frame** %9, align 8
  %16 = getelementptr %struct.__cilkrts_stack_frame*, %struct.__cilkrts_stack_frame** %9, i64 1
  store atomic %struct.__cilkrts_stack_frame** %16, %struct.__cilkrts_stack_frame*** %8 release, align 8
  %17 = load atomic i32, i32* %1 acquire, align 8
  %18 = or i32 %17, 4
  store atomic i32 %18, i32* %1 release, align 8
  tail call fastcc void @cluster_25__XlaCompiledKernel_true__XlaNumConstantArgs_1__XlaNumResourceArgs_0_.42.outline_dot.10.loop_detach.lhs.0.ls1(i64 %dot.10.indvar_address.lhs.0.040.ls1.dac3.otd1, i64 %miditer.otd1, i64 %grainsize.ls1.otd1, [20 x [400 x float]]* %concatenate.9.ls1.otd1, [400 x [800 x float]]* %.ls1.otd1, [20 x [800 x float]]* %.ls11.otd1) #6
  %19 = load atomic %struct.__cilkrts_worker*, %struct.__cilkrts_worker** %5 acquire, align 8
  %20 = bitcast %struct.__cilkrts_stack_frame** %4 to i64*
  %21 = load i64, i64* %20, align 8
  %22 = getelementptr inbounds %struct.__cilkrts_worker, %struct.__cilkrts_worker* %19, i64 0, i32 9
  %23 = bitcast %struct.__cilkrts_stack_frame** %22 to i64*
  store atomic i64 %21, i64* %23 release, align 8
  store atomic %struct.__cilkrts_stack_frame* null, %struct.__cilkrts_stack_frame** %4 release, align 8
  %24 = load atomic i32, i32* %1 acquire, align 8
  %25 = icmp eq i32 %24, 16777216
  br i1 %25, label %__cilk_parent_epilogue.exit, label %body.i

body.i:                                           ; preds = %.lr.ph.otd1
  call void @__cilkrts_leave_frame(%struct.__cilkrts_stack_frame* nonnull %__cilkrts_sf) #6
  br label %__cilk_parent_epilogue.exit

__cilk_parent_epilogue.exit:                      ; preds = %.lr.ph.otd1, %body.i
  ret void
}

declare %struct.__cilkrts_worker* @__cilkrts_get_tls_worker_fast() local_unnamed_addr

declare void @__cilkrts_leave_frame(%struct.__cilkrts_stack_frame*) local_unnamed_addr

declare %struct.__cilkrts_worker* @__cilkrts_get_tls_worker() local_unnamed_addr

declare %struct.__cilkrts_worker* @__cilkrts_bind_thread_1() local_unnamed_addr

; Function Attrs: nounwind readnone
declare i8* @llvm.frameaddress(i32) #2

; Function Attrs: nounwind
declare i8* @llvm.stacksave() #6

; Function Attrs: nounwind
declare i32 @llvm.eh.sjlj.setjmp(i8*) #6

; Function Attrs: nounwind readnone
declare i32 @__cilkrts_get_nworkers() local_unnamed_addr #2

declare void @__cilkrts_sync(%struct.__cilkrts_stack_frame*) local_unnamed_addr

; Function Attrs: argmemonly noinline nounwind
define private fastcc void @cluster_25__XlaCompiledKernel_true__XlaNumConstantArgs_1__XlaNumResourceArgs_0_.42.outline_dot.10.loop_detach.rhs.1.ls2.outline_.split.otd1(i64 %dot.10.indvar_address.rhs.1.038.ls2.dac3.otd1, i64 %miditer.otd1, i64 %grainsize.ls2.otd1, [20 x [400 x float]]* nocapture readonly align 16 %concatenate.9.ls2.otd1, i64 %dot.10.indvar_address.lhs.0.040.ls2.otd1, [400 x [800 x float]]* align 16 %.ls2.otd1, [20 x [800 x float]]* nocapture align 16 %.ls21.otd1) unnamed_addr #5 {
.lr.ph.otd1:
  %__cilkrts_sf = alloca %struct.__cilkrts_stack_frame, align 8
  %0 = tail call %struct.__cilkrts_worker* @__cilkrts_get_tls_worker_fast() #6
  %1 = getelementptr inbounds %struct.__cilkrts_stack_frame, %struct.__cilkrts_stack_frame* %__cilkrts_sf, i64 0, i32 0
  store atomic i32 16777216, i32* %1 release, align 8
  %2 = getelementptr inbounds %struct.__cilkrts_worker, %struct.__cilkrts_worker* %0, i64 0, i32 9
  %3 = load atomic %struct.__cilkrts_stack_frame*, %struct.__cilkrts_stack_frame** %2 acquire, align 8
  %4 = getelementptr inbounds %struct.__cilkrts_stack_frame, %struct.__cilkrts_stack_frame* %__cilkrts_sf, i64 0, i32 2
  store atomic %struct.__cilkrts_stack_frame* %3, %struct.__cilkrts_stack_frame** %4 release, align 8
  %5 = getelementptr inbounds %struct.__cilkrts_stack_frame, %struct.__cilkrts_stack_frame* %__cilkrts_sf, i64 0, i32 3
  store atomic %struct.__cilkrts_worker* %0, %struct.__cilkrts_worker** %5 release, align 8
  store atomic %struct.__cilkrts_stack_frame* %__cilkrts_sf, %struct.__cilkrts_stack_frame** %2 release, align 8
  %6 = load %struct.__cilkrts_worker*, %struct.__cilkrts_worker** %5, align 8
  %7 = load %struct.__cilkrts_stack_frame*, %struct.__cilkrts_stack_frame** %4, align 8
  %8 = getelementptr inbounds %struct.__cilkrts_worker, %struct.__cilkrts_worker* %6, i64 0, i32 0
  %9 = load atomic %struct.__cilkrts_stack_frame**, %struct.__cilkrts_stack_frame*** %8 acquire, align 8
  %.elt = getelementptr inbounds %struct.__cilkrts_worker, %struct.__cilkrts_worker* %6, i64 0, i32 12, i32 0
  %.unpack = load i64, i64* %.elt, align 8
  %.elt3 = getelementptr inbounds %struct.__cilkrts_worker, %struct.__cilkrts_worker* %6, i64 0, i32 12, i32 1
  %10 = bitcast %struct.__cilkrts_pedigree** %.elt3 to i64*
  %.unpack49 = load i64, i64* %10, align 8
  %11 = getelementptr inbounds %struct.__cilkrts_stack_frame, %struct.__cilkrts_stack_frame* %__cilkrts_sf, i64 0, i32 9
  %.fca.0.0.gep = getelementptr inbounds { %struct.__cilkrts_pedigree }, { %struct.__cilkrts_pedigree }* %11, i64 0, i32 0, i32 0
  %.fca.0.1.gep = getelementptr inbounds %struct.__cilkrts_stack_frame, %struct.__cilkrts_stack_frame* %__cilkrts_sf, i64 0, i32 9, i32 0, i32 1
  store i64 %.unpack, i64* %.fca.0.0.gep, align 8
  %12 = bitcast %struct.__cilkrts_pedigree** %.fca.0.1.gep to i64*
  store i64 %.unpack49, i64* %12, align 8
  %.repack = getelementptr inbounds %struct.__cilkrts_stack_frame, %struct.__cilkrts_stack_frame* %7, i64 0, i32 9, i32 0, i32 0
  store i64 %.unpack, i64* %.repack, align 8
  %13 = getelementptr inbounds %struct.__cilkrts_stack_frame, %struct.__cilkrts_stack_frame* %7, i64 0, i32 9, i32 0, i32 1
  %14 = bitcast %struct.__cilkrts_pedigree** %13 to i64*
  store i64 %.unpack49, i64* %14, align 8
  store atomic i64 0, i64* %.elt release, align 8
  %15 = getelementptr inbounds { %struct.__cilkrts_pedigree }, { %struct.__cilkrts_pedigree }* %11, i64 0, i32 0
  store atomic %struct.__cilkrts_pedigree* %15, %struct.__cilkrts_pedigree** %.elt3 release, align 8
  fence release
  store volatile %struct.__cilkrts_stack_frame* %7, %struct.__cilkrts_stack_frame** %9, align 8
  %16 = getelementptr %struct.__cilkrts_stack_frame*, %struct.__cilkrts_stack_frame** %9, i64 1
  store atomic %struct.__cilkrts_stack_frame** %16, %struct.__cilkrts_stack_frame*** %8 release, align 8
  %17 = load atomic i32, i32* %1 acquire, align 8
  %18 = or i32 %17, 4
  store atomic i32 %18, i32* %1 release, align 8
  tail call fastcc void @cluster_25__XlaCompiledKernel_true__XlaNumConstantArgs_1__XlaNumResourceArgs_0_.42.outline_dot.10.loop_detach.rhs.1.ls2(i64 %dot.10.indvar_address.rhs.1.038.ls2.dac3.otd1, i64 %miditer.otd1, i64 %grainsize.ls2.otd1, [20 x [400 x float]]* %concatenate.9.ls2.otd1, i64 %dot.10.indvar_address.lhs.0.040.ls2.otd1, [400 x [800 x float]]* %.ls2.otd1, [20 x [800 x float]]* %.ls21.otd1) #6
  %19 = load atomic %struct.__cilkrts_worker*, %struct.__cilkrts_worker** %5 acquire, align 8
  %20 = bitcast %struct.__cilkrts_stack_frame** %4 to i64*
  %21 = load i64, i64* %20, align 8
  %22 = getelementptr inbounds %struct.__cilkrts_worker, %struct.__cilkrts_worker* %19, i64 0, i32 9
  %23 = bitcast %struct.__cilkrts_stack_frame** %22 to i64*
  store atomic i64 %21, i64* %23 release, align 8
  store atomic %struct.__cilkrts_stack_frame* null, %struct.__cilkrts_stack_frame** %4 release, align 8
  %24 = load atomic i32, i32* %1 acquire, align 8
  %25 = icmp eq i32 %24, 16777216
  br i1 %25, label %__cilk_parent_epilogue.exit, label %body.i

body.i:                                           ; preds = %.lr.ph.otd1
  call void @__cilkrts_leave_frame(%struct.__cilkrts_stack_frame* nonnull %__cilkrts_sf) #6
  br label %__cilk_parent_epilogue.exit

__cilk_parent_epilogue.exit:                      ; preds = %.lr.ph.otd1, %body.i
  ret void
}

; Function Attrs: argmemonly noinline nounwind
define private fastcc void @cluster_25__XlaCompiledKernel_true__XlaNumConstantArgs_1__XlaNumResourceArgs_0_.42.outline_fusion.2.loop_detach.dim.0.ls1.outline_.split.otd1(<8 x float> %wide.load91.3.ls1.otd1, <8 x float> %wide.load90.ls1.otd1, <8 x float> %wide.load91.ls1.otd1, <8 x float> %wide.load92.ls1.otd1, <8 x float> %wide.load93.ls1.otd1, <8 x float> %wide.load93.5.ls1.otd1, <8 x float> %wide.load90.1.ls1.otd1, <8 x float> %wide.load91.1.ls1.otd1, <8 x float> %wide.load92.1.ls1.otd1, <8 x float> %wide.load93.1.ls1.otd1, <8 x float> %wide.load90.2.ls1.otd1, <8 x float> %wide.load92.2.ls1.otd1, <8 x float> %wide.load93.2.ls1.otd1, <8 x float> %wide.load90.3.ls1.otd1, <8 x float> %wide.load92.3.ls1.otd1, <8 x float> %wide.load93.3.ls1.otd1, <8 x float> %wide.load90.4.ls1.otd1, <8 x float> %wide.load91.4.ls1.otd1, <8 x float> %wide.load92.4.ls1.otd1, <8 x float> %wide.load93.4.ls1.otd1, <8 x float> %wide.load90.5.ls1.otd1, <8 x float> %wide.load91.5.ls1.otd1, <8 x float> %wide.load92.5.ls1.otd1, <8 x float> %wide.load91.2.ls1.otd1, i8* readnone align 16 %scevgep7374.ls1.otd1, [20 x [200 x float]]* align 16 %fusion.2.ls1.otd1, i8* readnone align 16 %scevgep7172.ls1.otd1, [20 x [800 x float]]* readonly align 16 %.ls12.otd1, i64 %fusion.2.indvar_address.dim.0.034.ls1.dac4.otd1, [800 x float]* nocapture readonly align 16 %.ls11.otd1, i8* readnone align 16 %.ls1.otd1, i64 %grainsize.ls1.otd1, i64 %miditer.otd1) unnamed_addr #5 {
.lr.ph.otd1:
  %__cilkrts_sf = alloca %struct.__cilkrts_stack_frame, align 8
  %0 = tail call %struct.__cilkrts_worker* @__cilkrts_get_tls_worker_fast() #6
  %1 = getelementptr inbounds %struct.__cilkrts_stack_frame, %struct.__cilkrts_stack_frame* %__cilkrts_sf, i64 0, i32 0
  store atomic i32 16777216, i32* %1 release, align 8
  %2 = getelementptr inbounds %struct.__cilkrts_worker, %struct.__cilkrts_worker* %0, i64 0, i32 9
  %3 = load atomic %struct.__cilkrts_stack_frame*, %struct.__cilkrts_stack_frame** %2 acquire, align 8
  %4 = getelementptr inbounds %struct.__cilkrts_stack_frame, %struct.__cilkrts_stack_frame* %__cilkrts_sf, i64 0, i32 2
  store atomic %struct.__cilkrts_stack_frame* %3, %struct.__cilkrts_stack_frame** %4 release, align 8
  %5 = getelementptr inbounds %struct.__cilkrts_stack_frame, %struct.__cilkrts_stack_frame* %__cilkrts_sf, i64 0, i32 3
  store atomic %struct.__cilkrts_worker* %0, %struct.__cilkrts_worker** %5 release, align 8
  store atomic %struct.__cilkrts_stack_frame* %__cilkrts_sf, %struct.__cilkrts_stack_frame** %2 release, align 8
  %6 = load %struct.__cilkrts_worker*, %struct.__cilkrts_worker** %5, align 8
  %7 = load %struct.__cilkrts_stack_frame*, %struct.__cilkrts_stack_frame** %4, align 8
  %8 = getelementptr inbounds %struct.__cilkrts_worker, %struct.__cilkrts_worker* %6, i64 0, i32 0
  %9 = load atomic %struct.__cilkrts_stack_frame**, %struct.__cilkrts_stack_frame*** %8 acquire, align 8
  %.elt = getelementptr inbounds %struct.__cilkrts_worker, %struct.__cilkrts_worker* %6, i64 0, i32 12, i32 0
  %.unpack = load i64, i64* %.elt, align 8
  %.elt3 = getelementptr inbounds %struct.__cilkrts_worker, %struct.__cilkrts_worker* %6, i64 0, i32 12, i32 1
  %10 = bitcast %struct.__cilkrts_pedigree** %.elt3 to i64*
  %.unpack49 = load i64, i64* %10, align 8
  %11 = getelementptr inbounds %struct.__cilkrts_stack_frame, %struct.__cilkrts_stack_frame* %__cilkrts_sf, i64 0, i32 9
  %.fca.0.0.gep = getelementptr inbounds { %struct.__cilkrts_pedigree }, { %struct.__cilkrts_pedigree }* %11, i64 0, i32 0, i32 0
  %.fca.0.1.gep = getelementptr inbounds %struct.__cilkrts_stack_frame, %struct.__cilkrts_stack_frame* %__cilkrts_sf, i64 0, i32 9, i32 0, i32 1
  store i64 %.unpack, i64* %.fca.0.0.gep, align 8
  %12 = bitcast %struct.__cilkrts_pedigree** %.fca.0.1.gep to i64*
  store i64 %.unpack49, i64* %12, align 8
  %.repack = getelementptr inbounds %struct.__cilkrts_stack_frame, %struct.__cilkrts_stack_frame* %7, i64 0, i32 9, i32 0, i32 0
  store i64 %.unpack, i64* %.repack, align 8
  %13 = getelementptr inbounds %struct.__cilkrts_stack_frame, %struct.__cilkrts_stack_frame* %7, i64 0, i32 9, i32 0, i32 1
  %14 = bitcast %struct.__cilkrts_pedigree** %13 to i64*
  store i64 %.unpack49, i64* %14, align 8
  store atomic i64 0, i64* %.elt release, align 8
  %15 = getelementptr inbounds { %struct.__cilkrts_pedigree }, { %struct.__cilkrts_pedigree }* %11, i64 0, i32 0
  store atomic %struct.__cilkrts_pedigree* %15, %struct.__cilkrts_pedigree** %.elt3 release, align 8
  fence release
  store volatile %struct.__cilkrts_stack_frame* %7, %struct.__cilkrts_stack_frame** %9, align 8
  %16 = getelementptr %struct.__cilkrts_stack_frame*, %struct.__cilkrts_stack_frame** %9, i64 1
  store atomic %struct.__cilkrts_stack_frame** %16, %struct.__cilkrts_stack_frame*** %8 release, align 8
  %17 = load atomic i32, i32* %1 acquire, align 8
  %18 = or i32 %17, 4
  store atomic i32 %18, i32* %1 release, align 8
  tail call fastcc void @cluster_25__XlaCompiledKernel_true__XlaNumConstantArgs_1__XlaNumResourceArgs_0_.42.outline_fusion.2.loop_detach.dim.0.ls1(i64 %fusion.2.indvar_address.dim.0.034.ls1.dac4.otd1, i64 %miditer.otd1, i64 %grainsize.ls1.otd1, <8 x float> %wide.load91.2.ls1.otd1, <8 x float> %wide.load92.5.ls1.otd1, <8 x float> %wide.load91.5.ls1.otd1, <8 x float> %wide.load90.5.ls1.otd1, <8 x float> %wide.load93.4.ls1.otd1, <8 x float> %wide.load92.4.ls1.otd1, <8 x float> %wide.load91.4.ls1.otd1, <8 x float> %wide.load90.4.ls1.otd1, <8 x float> %wide.load93.3.ls1.otd1, <8 x float> %wide.load92.3.ls1.otd1, <8 x float> %wide.load91.3.ls1.otd1, <8 x float> %wide.load90.3.ls1.otd1, <8 x float> %wide.load93.2.ls1.otd1, <8 x float> %wide.load92.2.ls1.otd1, <8 x float> %wide.load90.2.ls1.otd1, <8 x float> %wide.load93.1.ls1.otd1, <8 x float> %wide.load92.1.ls1.otd1, <8 x float> %wide.load91.1.ls1.otd1, <8 x float> %wide.load90.1.ls1.otd1, <8 x float> %wide.load93.5.ls1.otd1, <8 x float> %wide.load93.ls1.otd1, <8 x float> %wide.load92.ls1.otd1, <8 x float> %wide.load91.ls1.otd1, <8 x float> %wide.load90.ls1.otd1, i8* %.ls1.otd1, [800 x float]* %.ls11.otd1, i8* %scevgep7374.ls1.otd1, [20 x [200 x float]]* %fusion.2.ls1.otd1, [20 x [800 x float]]* %.ls12.otd1, i8* %scevgep7172.ls1.otd1) #6
  %19 = load atomic %struct.__cilkrts_worker*, %struct.__cilkrts_worker** %5 acquire, align 8
  %20 = bitcast %struct.__cilkrts_stack_frame** %4 to i64*
  %21 = load i64, i64* %20, align 8
  %22 = getelementptr inbounds %struct.__cilkrts_worker, %struct.__cilkrts_worker* %19, i64 0, i32 9
  %23 = bitcast %struct.__cilkrts_stack_frame** %22 to i64*
  store atomic i64 %21, i64* %23 release, align 8
  store atomic %struct.__cilkrts_stack_frame* null, %struct.__cilkrts_stack_frame** %4 release, align 8
  %24 = load atomic i32, i32* %1 acquire, align 8
  %25 = icmp eq i32 %24, 16777216
  br i1 %25, label %__cilk_parent_epilogue.exit, label %body.i

body.i:                                           ; preds = %.lr.ph.otd1
  call void @__cilkrts_leave_frame(%struct.__cilkrts_stack_frame* nonnull %__cilkrts_sf) #6
  br label %__cilk_parent_epilogue.exit

__cilk_parent_epilogue.exit:                      ; preds = %.lr.ph.otd1, %body.i
  ret void
}

; Function Attrs: argmemonly noinline nounwind
define private fastcc void @cluster_25__XlaCompiledKernel_true__XlaNumConstantArgs_1__XlaNumResourceArgs_0_.42.outline_fusion.4.loop_detach.dim.0.ls1.outline_.split.otd1(<8 x float> %wide.load131.3.ls1.otd1, <8 x float> %wide.load130.ls1.otd1, <8 x float> %wide.load131.ls1.otd1, <8 x float> %wide.load132.ls1.otd1, <8 x float> %wide.load133.ls1.otd1, <8 x float> %wide.load133.5.ls1.otd1, <8 x float> %wide.load130.1.ls1.otd1, <8 x float> %wide.load131.1.ls1.otd1, <8 x float> %wide.load132.1.ls1.otd1, <8 x float> %wide.load133.1.ls1.otd1, <8 x float> %wide.load130.2.ls1.otd1, <8 x float> %wide.load132.2.ls1.otd1, <8 x float> %wide.load133.2.ls1.otd1, <8 x float> %wide.load130.3.ls1.otd1, <8 x float> %wide.load132.3.ls1.otd1, <8 x float> %wide.load133.3.ls1.otd1, <8 x float> %wide.load130.4.ls1.otd1, <8 x float> %wide.load131.4.ls1.otd1, <8 x float> %wide.load132.4.ls1.otd1, <8 x float> %wide.load133.4.ls1.otd1, <8 x float> %wide.load130.5.ls1.otd1, <8 x float> %wide.load131.5.ls1.otd1, <8 x float> %wide.load132.5.ls1.otd1, <8 x float> %wide.load131.2.ls1.otd1, i8* readnone align 16 %scevgep104105.ls1.otd1, [20 x [200 x float]]* align 16 %fusion.4.ls1.otd1, i8* readnone align 16 %.ls13.otd1, [20 x [800 x float]]* readonly align 16 %.ls12.otd1, i64 %fusion.4.indvar_address.dim.0.031.ls1.dac5.otd1, [800 x float]* nocapture readonly align 16 %.ls11.otd1, i8* readnone align 16 %.ls1.otd1, i64 %grainsize.ls1.otd1, i64 %miditer.otd1) unnamed_addr #5 {
.lr.ph.otd1:
  %__cilkrts_sf = alloca %struct.__cilkrts_stack_frame, align 8
  %0 = tail call %struct.__cilkrts_worker* @__cilkrts_get_tls_worker_fast() #6
  %1 = getelementptr inbounds %struct.__cilkrts_stack_frame, %struct.__cilkrts_stack_frame* %__cilkrts_sf, i64 0, i32 0
  store atomic i32 16777216, i32* %1 release, align 8
  %2 = getelementptr inbounds %struct.__cilkrts_worker, %struct.__cilkrts_worker* %0, i64 0, i32 9
  %3 = load atomic %struct.__cilkrts_stack_frame*, %struct.__cilkrts_stack_frame** %2 acquire, align 8
  %4 = getelementptr inbounds %struct.__cilkrts_stack_frame, %struct.__cilkrts_stack_frame* %__cilkrts_sf, i64 0, i32 2
  store atomic %struct.__cilkrts_stack_frame* %3, %struct.__cilkrts_stack_frame** %4 release, align 8
  %5 = getelementptr inbounds %struct.__cilkrts_stack_frame, %struct.__cilkrts_stack_frame* %__cilkrts_sf, i64 0, i32 3
  store atomic %struct.__cilkrts_worker* %0, %struct.__cilkrts_worker** %5 release, align 8
  store atomic %struct.__cilkrts_stack_frame* %__cilkrts_sf, %struct.__cilkrts_stack_frame** %2 release, align 8
  %6 = load %struct.__cilkrts_worker*, %struct.__cilkrts_worker** %5, align 8
  %7 = load %struct.__cilkrts_stack_frame*, %struct.__cilkrts_stack_frame** %4, align 8
  %8 = getelementptr inbounds %struct.__cilkrts_worker, %struct.__cilkrts_worker* %6, i64 0, i32 0
  %9 = load atomic %struct.__cilkrts_stack_frame**, %struct.__cilkrts_stack_frame*** %8 acquire, align 8
  %.elt = getelementptr inbounds %struct.__cilkrts_worker, %struct.__cilkrts_worker* %6, i64 0, i32 12, i32 0
  %.unpack = load i64, i64* %.elt, align 8
  %.elt3 = getelementptr inbounds %struct.__cilkrts_worker, %struct.__cilkrts_worker* %6, i64 0, i32 12, i32 1
  %10 = bitcast %struct.__cilkrts_pedigree** %.elt3 to i64*
  %.unpack49 = load i64, i64* %10, align 8
  %11 = getelementptr inbounds %struct.__cilkrts_stack_frame, %struct.__cilkrts_stack_frame* %__cilkrts_sf, i64 0, i32 9
  %.fca.0.0.gep = getelementptr inbounds { %struct.__cilkrts_pedigree }, { %struct.__cilkrts_pedigree }* %11, i64 0, i32 0, i32 0
  %.fca.0.1.gep = getelementptr inbounds %struct.__cilkrts_stack_frame, %struct.__cilkrts_stack_frame* %__cilkrts_sf, i64 0, i32 9, i32 0, i32 1
  store i64 %.unpack, i64* %.fca.0.0.gep, align 8
  %12 = bitcast %struct.__cilkrts_pedigree** %.fca.0.1.gep to i64*
  store i64 %.unpack49, i64* %12, align 8
  %.repack = getelementptr inbounds %struct.__cilkrts_stack_frame, %struct.__cilkrts_stack_frame* %7, i64 0, i32 9, i32 0, i32 0
  store i64 %.unpack, i64* %.repack, align 8
  %13 = getelementptr inbounds %struct.__cilkrts_stack_frame, %struct.__cilkrts_stack_frame* %7, i64 0, i32 9, i32 0, i32 1
  %14 = bitcast %struct.__cilkrts_pedigree** %13 to i64*
  store i64 %.unpack49, i64* %14, align 8
  store atomic i64 0, i64* %.elt release, align 8
  %15 = getelementptr inbounds { %struct.__cilkrts_pedigree }, { %struct.__cilkrts_pedigree }* %11, i64 0, i32 0
  store atomic %struct.__cilkrts_pedigree* %15, %struct.__cilkrts_pedigree** %.elt3 release, align 8
  fence release
  store volatile %struct.__cilkrts_stack_frame* %7, %struct.__cilkrts_stack_frame** %9, align 8
  %16 = getelementptr %struct.__cilkrts_stack_frame*, %struct.__cilkrts_stack_frame** %9, i64 1
  store atomic %struct.__cilkrts_stack_frame** %16, %struct.__cilkrts_stack_frame*** %8 release, align 8
  %17 = load atomic i32, i32* %1 acquire, align 8
  %18 = or i32 %17, 4
  store atomic i32 %18, i32* %1 release, align 8
  tail call fastcc void @cluster_25__XlaCompiledKernel_true__XlaNumConstantArgs_1__XlaNumResourceArgs_0_.42.outline_fusion.4.loop_detach.dim.0.ls1(i64 %fusion.4.indvar_address.dim.0.031.ls1.dac5.otd1, i64 %miditer.otd1, i64 %grainsize.ls1.otd1, <8 x float> %wide.load131.2.ls1.otd1, <8 x float> %wide.load132.5.ls1.otd1, <8 x float> %wide.load131.5.ls1.otd1, <8 x float> %wide.load130.5.ls1.otd1, <8 x float> %wide.load133.4.ls1.otd1, <8 x float> %wide.load132.4.ls1.otd1, <8 x float> %wide.load131.4.ls1.otd1, <8 x float> %wide.load130.4.ls1.otd1, <8 x float> %wide.load133.3.ls1.otd1, <8 x float> %wide.load132.3.ls1.otd1, <8 x float> %wide.load131.3.ls1.otd1, <8 x float> %wide.load130.3.ls1.otd1, <8 x float> %wide.load133.2.ls1.otd1, <8 x float> %wide.load132.2.ls1.otd1, <8 x float> %wide.load130.2.ls1.otd1, <8 x float> %wide.load133.1.ls1.otd1, <8 x float> %wide.load132.1.ls1.otd1, <8 x float> %wide.load131.1.ls1.otd1, <8 x float> %wide.load130.1.ls1.otd1, <8 x float> %wide.load133.5.ls1.otd1, <8 x float> %wide.load133.ls1.otd1, <8 x float> %wide.load132.ls1.otd1, <8 x float> %wide.load131.ls1.otd1, <8 x float> %wide.load130.ls1.otd1, i8* %.ls1.otd1, [800 x float]* %.ls11.otd1, i8* %scevgep104105.ls1.otd1, [20 x [200 x float]]* %fusion.4.ls1.otd1, [20 x [800 x float]]* %.ls12.otd1, i8* %.ls13.otd1) #6
  %19 = load atomic %struct.__cilkrts_worker*, %struct.__cilkrts_worker** %5 acquire, align 8
  %20 = bitcast %struct.__cilkrts_stack_frame** %4 to i64*
  %21 = load i64, i64* %20, align 8
  %22 = getelementptr inbounds %struct.__cilkrts_worker, %struct.__cilkrts_worker* %19, i64 0, i32 9
  %23 = bitcast %struct.__cilkrts_stack_frame** %22 to i64*
  store atomic i64 %21, i64* %23 release, align 8
  store atomic %struct.__cilkrts_stack_frame* null, %struct.__cilkrts_stack_frame** %4 release, align 8
  %24 = load atomic i32, i32* %1 acquire, align 8
  %25 = icmp eq i32 %24, 16777216
  br i1 %25, label %__cilk_parent_epilogue.exit, label %body.i

body.i:                                           ; preds = %.lr.ph.otd1
  call void @__cilkrts_leave_frame(%struct.__cilkrts_stack_frame* nonnull %__cilkrts_sf) #6
  br label %__cilk_parent_epilogue.exit

__cilk_parent_epilogue.exit:                      ; preds = %.lr.ph.otd1, %body.i
  ret void
}

; Function Attrs: argmemonly noinline nounwind
define private fastcc void @cluster_25__XlaCompiledKernel_true__XlaNumConstantArgs_1__XlaNumResourceArgs_0_.42.outline_fusion.1.loop_detach.dim.0.ls1.outline_.split.otd1(i64 %fusion.1.indvar_address.dim.0.028.ls1.dac2.otd1, i64 %miditer.otd1, i64 %grainsize.ls1.otd1, [20 x [200 x float]]* readonly align 16 %fusion.4.ls1.otd1, [20 x [200 x float]]* readonly align 16 %fusion.2.ls1.otd1, [20 x [200 x float]]* align 16 %fusion.1.ls1.otd1) unnamed_addr #5 {
.lr.ph.otd1:
  %__cilkrts_sf = alloca %struct.__cilkrts_stack_frame, align 8
  %0 = tail call %struct.__cilkrts_worker* @__cilkrts_get_tls_worker_fast() #6
  %1 = getelementptr inbounds %struct.__cilkrts_stack_frame, %struct.__cilkrts_stack_frame* %__cilkrts_sf, i64 0, i32 0
  store atomic i32 16777216, i32* %1 release, align 8
  %2 = getelementptr inbounds %struct.__cilkrts_worker, %struct.__cilkrts_worker* %0, i64 0, i32 9
  %3 = load atomic %struct.__cilkrts_stack_frame*, %struct.__cilkrts_stack_frame** %2 acquire, align 8
  %4 = getelementptr inbounds %struct.__cilkrts_stack_frame, %struct.__cilkrts_stack_frame* %__cilkrts_sf, i64 0, i32 2
  store atomic %struct.__cilkrts_stack_frame* %3, %struct.__cilkrts_stack_frame** %4 release, align 8
  %5 = getelementptr inbounds %struct.__cilkrts_stack_frame, %struct.__cilkrts_stack_frame* %__cilkrts_sf, i64 0, i32 3
  store atomic %struct.__cilkrts_worker* %0, %struct.__cilkrts_worker** %5 release, align 8
  store atomic %struct.__cilkrts_stack_frame* %__cilkrts_sf, %struct.__cilkrts_stack_frame** %2 release, align 8
  %6 = load %struct.__cilkrts_worker*, %struct.__cilkrts_worker** %5, align 8
  %7 = load %struct.__cilkrts_stack_frame*, %struct.__cilkrts_stack_frame** %4, align 8
  %8 = getelementptr inbounds %struct.__cilkrts_worker, %struct.__cilkrts_worker* %6, i64 0, i32 0
  %9 = load atomic %struct.__cilkrts_stack_frame**, %struct.__cilkrts_stack_frame*** %8 acquire, align 8
  %.elt = getelementptr inbounds %struct.__cilkrts_worker, %struct.__cilkrts_worker* %6, i64 0, i32 12, i32 0
  %.unpack = load i64, i64* %.elt, align 8
  %.elt3 = getelementptr inbounds %struct.__cilkrts_worker, %struct.__cilkrts_worker* %6, i64 0, i32 12, i32 1
  %10 = bitcast %struct.__cilkrts_pedigree** %.elt3 to i64*
  %.unpack49 = load i64, i64* %10, align 8
  %11 = getelementptr inbounds %struct.__cilkrts_stack_frame, %struct.__cilkrts_stack_frame* %__cilkrts_sf, i64 0, i32 9
  %.fca.0.0.gep = getelementptr inbounds { %struct.__cilkrts_pedigree }, { %struct.__cilkrts_pedigree }* %11, i64 0, i32 0, i32 0
  %.fca.0.1.gep = getelementptr inbounds %struct.__cilkrts_stack_frame, %struct.__cilkrts_stack_frame* %__cilkrts_sf, i64 0, i32 9, i32 0, i32 1
  store i64 %.unpack, i64* %.fca.0.0.gep, align 8
  %12 = bitcast %struct.__cilkrts_pedigree** %.fca.0.1.gep to i64*
  store i64 %.unpack49, i64* %12, align 8
  %.repack = getelementptr inbounds %struct.__cilkrts_stack_frame, %struct.__cilkrts_stack_frame* %7, i64 0, i32 9, i32 0, i32 0
  store i64 %.unpack, i64* %.repack, align 8
  %13 = getelementptr inbounds %struct.__cilkrts_stack_frame, %struct.__cilkrts_stack_frame* %7, i64 0, i32 9, i32 0, i32 1
  %14 = bitcast %struct.__cilkrts_pedigree** %13 to i64*
  store i64 %.unpack49, i64* %14, align 8
  store atomic i64 0, i64* %.elt release, align 8
  %15 = getelementptr inbounds { %struct.__cilkrts_pedigree }, { %struct.__cilkrts_pedigree }* %11, i64 0, i32 0
  store atomic %struct.__cilkrts_pedigree* %15, %struct.__cilkrts_pedigree** %.elt3 release, align 8
  fence release
  store volatile %struct.__cilkrts_stack_frame* %7, %struct.__cilkrts_stack_frame** %9, align 8
  %16 = getelementptr %struct.__cilkrts_stack_frame*, %struct.__cilkrts_stack_frame** %9, i64 1
  store atomic %struct.__cilkrts_stack_frame** %16, %struct.__cilkrts_stack_frame*** %8 release, align 8
  %17 = load atomic i32, i32* %1 acquire, align 8
  %18 = or i32 %17, 4
  store atomic i32 %18, i32* %1 release, align 8
  tail call fastcc void @cluster_25__XlaCompiledKernel_true__XlaNumConstantArgs_1__XlaNumResourceArgs_0_.42.outline_fusion.1.loop_detach.dim.0.ls1(i64 %fusion.1.indvar_address.dim.0.028.ls1.dac2.otd1, i64 %miditer.otd1, i64 %grainsize.ls1.otd1, [20 x [200 x float]]* %fusion.4.ls1.otd1, [20 x [200 x float]]* %fusion.2.ls1.otd1, [20 x [200 x float]]* %fusion.1.ls1.otd1) #6
  %19 = load atomic %struct.__cilkrts_worker*, %struct.__cilkrts_worker** %5 acquire, align 8
  %20 = bitcast %struct.__cilkrts_stack_frame** %4 to i64*
  %21 = load i64, i64* %20, align 8
  %22 = getelementptr inbounds %struct.__cilkrts_worker, %struct.__cilkrts_worker* %19, i64 0, i32 9
  %23 = bitcast %struct.__cilkrts_stack_frame** %22 to i64*
  store atomic i64 %21, i64* %23 release, align 8
  store atomic %struct.__cilkrts_stack_frame* null, %struct.__cilkrts_stack_frame** %4 release, align 8
  %24 = load atomic i32, i32* %1 acquire, align 8
  %25 = icmp eq i32 %24, 16777216
  br i1 %25, label %__cilk_parent_epilogue.exit, label %body.i

body.i:                                           ; preds = %.lr.ph.otd1
  call void @__cilkrts_leave_frame(%struct.__cilkrts_stack_frame* nonnull %__cilkrts_sf) #6
  br label %__cilk_parent_epilogue.exit

__cilk_parent_epilogue.exit:                      ; preds = %.lr.ph.otd1, %body.i
  ret void
}

; Function Attrs: argmemonly noinline nounwind
define private fastcc void @cluster_25__XlaCompiledKernel_true__XlaNumConstantArgs_1__XlaNumResourceArgs_0_.42.outline_fusion.3.loop_detach.dim.0.ls1.outline_.split.otd1(i64 %fusion.3.indvar_address.dim.0.025.ls1.dac2.otd1, i64 %miditer.otd1, i64 %grainsize.ls1.otd1, [20 x [200 x float]]* align 16 %fusion.4.ls1.otd1) unnamed_addr #5 {
.lr.ph.otd1:
  %__cilkrts_sf = alloca %struct.__cilkrts_stack_frame, align 8
  %0 = tail call %struct.__cilkrts_worker* @__cilkrts_get_tls_worker_fast() #6
  %1 = getelementptr inbounds %struct.__cilkrts_stack_frame, %struct.__cilkrts_stack_frame* %__cilkrts_sf, i64 0, i32 0
  store atomic i32 16777216, i32* %1 release, align 8
  %2 = getelementptr inbounds %struct.__cilkrts_worker, %struct.__cilkrts_worker* %0, i64 0, i32 9
  %3 = load atomic %struct.__cilkrts_stack_frame*, %struct.__cilkrts_stack_frame** %2 acquire, align 8
  %4 = getelementptr inbounds %struct.__cilkrts_stack_frame, %struct.__cilkrts_stack_frame* %__cilkrts_sf, i64 0, i32 2
  store atomic %struct.__cilkrts_stack_frame* %3, %struct.__cilkrts_stack_frame** %4 release, align 8
  %5 = getelementptr inbounds %struct.__cilkrts_stack_frame, %struct.__cilkrts_stack_frame* %__cilkrts_sf, i64 0, i32 3
  store atomic %struct.__cilkrts_worker* %0, %struct.__cilkrts_worker** %5 release, align 8
  store atomic %struct.__cilkrts_stack_frame* %__cilkrts_sf, %struct.__cilkrts_stack_frame** %2 release, align 8
  %6 = load %struct.__cilkrts_worker*, %struct.__cilkrts_worker** %5, align 8
  %7 = load %struct.__cilkrts_stack_frame*, %struct.__cilkrts_stack_frame** %4, align 8
  %8 = getelementptr inbounds %struct.__cilkrts_worker, %struct.__cilkrts_worker* %6, i64 0, i32 0
  %9 = load atomic %struct.__cilkrts_stack_frame**, %struct.__cilkrts_stack_frame*** %8 acquire, align 8
  %.elt = getelementptr inbounds %struct.__cilkrts_worker, %struct.__cilkrts_worker* %6, i64 0, i32 12, i32 0
  %.unpack = load i64, i64* %.elt, align 8
  %.elt3 = getelementptr inbounds %struct.__cilkrts_worker, %struct.__cilkrts_worker* %6, i64 0, i32 12, i32 1
  %10 = bitcast %struct.__cilkrts_pedigree** %.elt3 to i64*
  %.unpack49 = load i64, i64* %10, align 8
  %11 = getelementptr inbounds %struct.__cilkrts_stack_frame, %struct.__cilkrts_stack_frame* %__cilkrts_sf, i64 0, i32 9
  %.fca.0.0.gep = getelementptr inbounds { %struct.__cilkrts_pedigree }, { %struct.__cilkrts_pedigree }* %11, i64 0, i32 0, i32 0
  %.fca.0.1.gep = getelementptr inbounds %struct.__cilkrts_stack_frame, %struct.__cilkrts_stack_frame* %__cilkrts_sf, i64 0, i32 9, i32 0, i32 1
  store i64 %.unpack, i64* %.fca.0.0.gep, align 8
  %12 = bitcast %struct.__cilkrts_pedigree** %.fca.0.1.gep to i64*
  store i64 %.unpack49, i64* %12, align 8
  %.repack = getelementptr inbounds %struct.__cilkrts_stack_frame, %struct.__cilkrts_stack_frame* %7, i64 0, i32 9, i32 0, i32 0
  store i64 %.unpack, i64* %.repack, align 8
  %13 = getelementptr inbounds %struct.__cilkrts_stack_frame, %struct.__cilkrts_stack_frame* %7, i64 0, i32 9, i32 0, i32 1
  %14 = bitcast %struct.__cilkrts_pedigree** %13 to i64*
  store i64 %.unpack49, i64* %14, align 8
  store atomic i64 0, i64* %.elt release, align 8
  %15 = getelementptr inbounds { %struct.__cilkrts_pedigree }, { %struct.__cilkrts_pedigree }* %11, i64 0, i32 0
  store atomic %struct.__cilkrts_pedigree* %15, %struct.__cilkrts_pedigree** %.elt3 release, align 8
  fence release
  store volatile %struct.__cilkrts_stack_frame* %7, %struct.__cilkrts_stack_frame** %9, align 8
  %16 = getelementptr %struct.__cilkrts_stack_frame*, %struct.__cilkrts_stack_frame** %9, i64 1
  store atomic %struct.__cilkrts_stack_frame** %16, %struct.__cilkrts_stack_frame*** %8 release, align 8
  %17 = load atomic i32, i32* %1 acquire, align 8
  %18 = or i32 %17, 4
  store atomic i32 %18, i32* %1 release, align 8
  tail call fastcc void @cluster_25__XlaCompiledKernel_true__XlaNumConstantArgs_1__XlaNumResourceArgs_0_.42.outline_fusion.3.loop_detach.dim.0.ls1(i64 %fusion.3.indvar_address.dim.0.025.ls1.dac2.otd1, i64 %miditer.otd1, i64 %grainsize.ls1.otd1, [20 x [200 x float]]* %fusion.4.ls1.otd1) #6
  %19 = load atomic %struct.__cilkrts_worker*, %struct.__cilkrts_worker** %5 acquire, align 8
  %20 = bitcast %struct.__cilkrts_stack_frame** %4 to i64*
  %21 = load i64, i64* %20, align 8
  %22 = getelementptr inbounds %struct.__cilkrts_worker, %struct.__cilkrts_worker* %19, i64 0, i32 9
  %23 = bitcast %struct.__cilkrts_stack_frame** %22 to i64*
  store atomic i64 %21, i64* %23 release, align 8
  store atomic %struct.__cilkrts_stack_frame* null, %struct.__cilkrts_stack_frame** %4 release, align 8
  %24 = load atomic i32, i32* %1 acquire, align 8
  %25 = icmp eq i32 %24, 16777216
  br i1 %25, label %__cilk_parent_epilogue.exit, label %body.i

body.i:                                           ; preds = %.lr.ph.otd1
  call void @__cilkrts_leave_frame(%struct.__cilkrts_stack_frame* nonnull %__cilkrts_sf) #6
  br label %__cilk_parent_epilogue.exit

__cilk_parent_epilogue.exit:                      ; preds = %.lr.ph.otd1, %body.i
  ret void
}

; Function Attrs: argmemonly noinline nounwind
define private fastcc void @cluster_25__XlaCompiledKernel_true__XlaNumConstantArgs_1__XlaNumResourceArgs_0_.42.outline_fusion.loop_detach.dim.0.ls1.outline_.split.otd1(i64 %fusion.indvar_address.dim.0.022.ls1.dac5.otd1, i64 %miditer.otd1, i64 %grainsize.ls1.otd1, i8* readnone align 16 %scevgep186187.ls1.otd1, i8* readnone align 16 %scevgep184185.ls1.otd1, [20 x [800 x float]]* readonly align 16 %.ls1.otd1, [800 x float]* nocapture readonly align 16 %.ls11.otd1, [20 x [200 x float]]* nocapture align 16 %fusion.ls1.otd1, i8* readnone align 16 %.ls12.otd1) unnamed_addr #5 {
.lr.ph.otd1:
  %__cilkrts_sf = alloca %struct.__cilkrts_stack_frame, align 8
  %0 = tail call %struct.__cilkrts_worker* @__cilkrts_get_tls_worker_fast() #6
  %1 = getelementptr inbounds %struct.__cilkrts_stack_frame, %struct.__cilkrts_stack_frame* %__cilkrts_sf, i64 0, i32 0
  store atomic i32 16777216, i32* %1 release, align 8
  %2 = getelementptr inbounds %struct.__cilkrts_worker, %struct.__cilkrts_worker* %0, i64 0, i32 9
  %3 = load atomic %struct.__cilkrts_stack_frame*, %struct.__cilkrts_stack_frame** %2 acquire, align 8
  %4 = getelementptr inbounds %struct.__cilkrts_stack_frame, %struct.__cilkrts_stack_frame* %__cilkrts_sf, i64 0, i32 2
  store atomic %struct.__cilkrts_stack_frame* %3, %struct.__cilkrts_stack_frame** %4 release, align 8
  %5 = getelementptr inbounds %struct.__cilkrts_stack_frame, %struct.__cilkrts_stack_frame* %__cilkrts_sf, i64 0, i32 3
  store atomic %struct.__cilkrts_worker* %0, %struct.__cilkrts_worker** %5 release, align 8
  store atomic %struct.__cilkrts_stack_frame* %__cilkrts_sf, %struct.__cilkrts_stack_frame** %2 release, align 8
  %6 = load %struct.__cilkrts_worker*, %struct.__cilkrts_worker** %5, align 8
  %7 = load %struct.__cilkrts_stack_frame*, %struct.__cilkrts_stack_frame** %4, align 8
  %8 = getelementptr inbounds %struct.__cilkrts_worker, %struct.__cilkrts_worker* %6, i64 0, i32 0
  %9 = load atomic %struct.__cilkrts_stack_frame**, %struct.__cilkrts_stack_frame*** %8 acquire, align 8
  %.elt = getelementptr inbounds %struct.__cilkrts_worker, %struct.__cilkrts_worker* %6, i64 0, i32 12, i32 0
  %.unpack = load i64, i64* %.elt, align 8
  %.elt3 = getelementptr inbounds %struct.__cilkrts_worker, %struct.__cilkrts_worker* %6, i64 0, i32 12, i32 1
  %10 = bitcast %struct.__cilkrts_pedigree** %.elt3 to i64*
  %.unpack49 = load i64, i64* %10, align 8
  %11 = getelementptr inbounds %struct.__cilkrts_stack_frame, %struct.__cilkrts_stack_frame* %__cilkrts_sf, i64 0, i32 9
  %.fca.0.0.gep = getelementptr inbounds { %struct.__cilkrts_pedigree }, { %struct.__cilkrts_pedigree }* %11, i64 0, i32 0, i32 0
  %.fca.0.1.gep = getelementptr inbounds %struct.__cilkrts_stack_frame, %struct.__cilkrts_stack_frame* %__cilkrts_sf, i64 0, i32 9, i32 0, i32 1
  store i64 %.unpack, i64* %.fca.0.0.gep, align 8
  %12 = bitcast %struct.__cilkrts_pedigree** %.fca.0.1.gep to i64*
  store i64 %.unpack49, i64* %12, align 8
  %.repack = getelementptr inbounds %struct.__cilkrts_stack_frame, %struct.__cilkrts_stack_frame* %7, i64 0, i32 9, i32 0, i32 0
  store i64 %.unpack, i64* %.repack, align 8
  %13 = getelementptr inbounds %struct.__cilkrts_stack_frame, %struct.__cilkrts_stack_frame* %7, i64 0, i32 9, i32 0, i32 1
  %14 = bitcast %struct.__cilkrts_pedigree** %13 to i64*
  store i64 %.unpack49, i64* %14, align 8
  store atomic i64 0, i64* %.elt release, align 8
  %15 = getelementptr inbounds { %struct.__cilkrts_pedigree }, { %struct.__cilkrts_pedigree }* %11, i64 0, i32 0
  store atomic %struct.__cilkrts_pedigree* %15, %struct.__cilkrts_pedigree** %.elt3 release, align 8
  fence release
  store volatile %struct.__cilkrts_stack_frame* %7, %struct.__cilkrts_stack_frame** %9, align 8
  %16 = getelementptr %struct.__cilkrts_stack_frame*, %struct.__cilkrts_stack_frame** %9, i64 1
  store atomic %struct.__cilkrts_stack_frame** %16, %struct.__cilkrts_stack_frame*** %8 release, align 8
  %17 = load atomic i32, i32* %1 acquire, align 8
  %18 = or i32 %17, 4
  store atomic i32 %18, i32* %1 release, align 8
  tail call fastcc void @cluster_25__XlaCompiledKernel_true__XlaNumConstantArgs_1__XlaNumResourceArgs_0_.42.outline_fusion.loop_detach.dim.0.ls1(i64 %fusion.indvar_address.dim.0.022.ls1.dac5.otd1, i64 %miditer.otd1, i64 %grainsize.ls1.otd1, i8* %scevgep186187.ls1.otd1, i8* %scevgep184185.ls1.otd1, [20 x [800 x float]]* %.ls1.otd1, [800 x float]* %.ls11.otd1, [20 x [200 x float]]* %fusion.ls1.otd1, i8* %.ls12.otd1) #6
  %19 = load atomic %struct.__cilkrts_worker*, %struct.__cilkrts_worker** %5 acquire, align 8
  %20 = bitcast %struct.__cilkrts_stack_frame** %4 to i64*
  %21 = load i64, i64* %20, align 8
  %22 = getelementptr inbounds %struct.__cilkrts_worker, %struct.__cilkrts_worker* %19, i64 0, i32 9
  %23 = bitcast %struct.__cilkrts_stack_frame** %22 to i64*
  store atomic i64 %21, i64* %23 release, align 8
  store atomic %struct.__cilkrts_stack_frame* null, %struct.__cilkrts_stack_frame** %4 release, align 8
  %24 = load atomic i32, i32* %1 acquire, align 8
  %25 = icmp eq i32 %24, 16777216
  br i1 %25, label %__cilk_parent_epilogue.exit, label %body.i

body.i:                                           ; preds = %.lr.ph.otd1
  call void @__cilkrts_leave_frame(%struct.__cilkrts_stack_frame* nonnull %__cilkrts_sf) #6
  br label %__cilk_parent_epilogue.exit

__cilk_parent_epilogue.exit:                      ; preds = %.lr.ph.otd1, %body.i
  ret void
}

; Function Attrs: argmemonly noinline nounwind
define private fastcc void @cluster_25__XlaCompiledKernel_true__XlaNumConstantArgs_1__XlaNumResourceArgs_0_.42.outline_fusion.5.loop_detach.dim.0.ls1.outline_.split.otd1(<8 x float> %wide.load246.3.ls1.otd1, <8 x float> %wide.load245.ls1.otd1, <8 x float> %wide.load246.ls1.otd1, <8 x float> %wide.load247.ls1.otd1, <8 x float> %wide.load248.ls1.otd1, <8 x float> %wide.load248.5.ls1.otd1, <8 x float> %wide.load245.1.ls1.otd1, <8 x float> %wide.load246.1.ls1.otd1, <8 x float> %wide.load247.1.ls1.otd1, <8 x float> %wide.load248.1.ls1.otd1, <8 x float> %wide.load245.2.ls1.otd1, <8 x float> %wide.load247.2.ls1.otd1, <8 x float> %wide.load248.2.ls1.otd1, <8 x float> %wide.load245.3.ls1.otd1, <8 x float> %wide.load247.3.ls1.otd1, <8 x float> %wide.load248.3.ls1.otd1, <8 x float> %wide.load245.4.ls1.otd1, <8 x float> %wide.load246.4.ls1.otd1, <8 x float> %wide.load247.4.ls1.otd1, <8 x float> %wide.load248.4.ls1.otd1, <8 x float> %wide.load245.5.ls1.otd1, <8 x float> %wide.load246.5.ls1.otd1, <8 x float> %wide.load247.5.ls1.otd1, <8 x float> %wide.load246.2.ls1.otd1, i8* readnone align 16 %scevgep219220.ls1.otd1, [20 x [200 x float]]* align 16 %fusion.5.ls1.otd1, i8* readnone align 16 %scevgep217218.ls1.otd1, [20 x [800 x float]]* readonly align 16 %.ls12.otd1, i64 %fusion.5.indvar_address.dim.0.019.ls1.dac4.otd1, [800 x float]* nocapture readonly align 16 %.ls11.otd1, i8* readnone align 16 %.ls1.otd1, i64 %grainsize.ls1.otd1, i64 %miditer.otd1) unnamed_addr #5 {
.lr.ph.otd1:
  %__cilkrts_sf = alloca %struct.__cilkrts_stack_frame, align 8
  %0 = tail call %struct.__cilkrts_worker* @__cilkrts_get_tls_worker_fast() #6
  %1 = getelementptr inbounds %struct.__cilkrts_stack_frame, %struct.__cilkrts_stack_frame* %__cilkrts_sf, i64 0, i32 0
  store atomic i32 16777216, i32* %1 release, align 8
  %2 = getelementptr inbounds %struct.__cilkrts_worker, %struct.__cilkrts_worker* %0, i64 0, i32 9
  %3 = load atomic %struct.__cilkrts_stack_frame*, %struct.__cilkrts_stack_frame** %2 acquire, align 8
  %4 = getelementptr inbounds %struct.__cilkrts_stack_frame, %struct.__cilkrts_stack_frame* %__cilkrts_sf, i64 0, i32 2
  store atomic %struct.__cilkrts_stack_frame* %3, %struct.__cilkrts_stack_frame** %4 release, align 8
  %5 = getelementptr inbounds %struct.__cilkrts_stack_frame, %struct.__cilkrts_stack_frame* %__cilkrts_sf, i64 0, i32 3
  store atomic %struct.__cilkrts_worker* %0, %struct.__cilkrts_worker** %5 release, align 8
  store atomic %struct.__cilkrts_stack_frame* %__cilkrts_sf, %struct.__cilkrts_stack_frame** %2 release, align 8
  %6 = load %struct.__cilkrts_worker*, %struct.__cilkrts_worker** %5, align 8
  %7 = load %struct.__cilkrts_stack_frame*, %struct.__cilkrts_stack_frame** %4, align 8
  %8 = getelementptr inbounds %struct.__cilkrts_worker, %struct.__cilkrts_worker* %6, i64 0, i32 0
  %9 = load atomic %struct.__cilkrts_stack_frame**, %struct.__cilkrts_stack_frame*** %8 acquire, align 8
  %.elt = getelementptr inbounds %struct.__cilkrts_worker, %struct.__cilkrts_worker* %6, i64 0, i32 12, i32 0
  %.unpack = load i64, i64* %.elt, align 8
  %.elt3 = getelementptr inbounds %struct.__cilkrts_worker, %struct.__cilkrts_worker* %6, i64 0, i32 12, i32 1
  %10 = bitcast %struct.__cilkrts_pedigree** %.elt3 to i64*
  %.unpack49 = load i64, i64* %10, align 8
  %11 = getelementptr inbounds %struct.__cilkrts_stack_frame, %struct.__cilkrts_stack_frame* %__cilkrts_sf, i64 0, i32 9
  %.fca.0.0.gep = getelementptr inbounds { %struct.__cilkrts_pedigree }, { %struct.__cilkrts_pedigree }* %11, i64 0, i32 0, i32 0
  %.fca.0.1.gep = getelementptr inbounds %struct.__cilkrts_stack_frame, %struct.__cilkrts_stack_frame* %__cilkrts_sf, i64 0, i32 9, i32 0, i32 1
  store i64 %.unpack, i64* %.fca.0.0.gep, align 8
  %12 = bitcast %struct.__cilkrts_pedigree** %.fca.0.1.gep to i64*
  store i64 %.unpack49, i64* %12, align 8
  %.repack = getelementptr inbounds %struct.__cilkrts_stack_frame, %struct.__cilkrts_stack_frame* %7, i64 0, i32 9, i32 0, i32 0
  store i64 %.unpack, i64* %.repack, align 8
  %13 = getelementptr inbounds %struct.__cilkrts_stack_frame, %struct.__cilkrts_stack_frame* %7, i64 0, i32 9, i32 0, i32 1
  %14 = bitcast %struct.__cilkrts_pedigree** %13 to i64*
  store i64 %.unpack49, i64* %14, align 8
  store atomic i64 0, i64* %.elt release, align 8
  %15 = getelementptr inbounds { %struct.__cilkrts_pedigree }, { %struct.__cilkrts_pedigree }* %11, i64 0, i32 0
  store atomic %struct.__cilkrts_pedigree* %15, %struct.__cilkrts_pedigree** %.elt3 release, align 8
  fence release
  store volatile %struct.__cilkrts_stack_frame* %7, %struct.__cilkrts_stack_frame** %9, align 8
  %16 = getelementptr %struct.__cilkrts_stack_frame*, %struct.__cilkrts_stack_frame** %9, i64 1
  store atomic %struct.__cilkrts_stack_frame** %16, %struct.__cilkrts_stack_frame*** %8 release, align 8
  %17 = load atomic i32, i32* %1 acquire, align 8
  %18 = or i32 %17, 4
  store atomic i32 %18, i32* %1 release, align 8
  tail call fastcc void @cluster_25__XlaCompiledKernel_true__XlaNumConstantArgs_1__XlaNumResourceArgs_0_.42.outline_fusion.5.loop_detach.dim.0.ls1(i64 %fusion.5.indvar_address.dim.0.019.ls1.dac4.otd1, i64 %miditer.otd1, i64 %grainsize.ls1.otd1, <8 x float> %wide.load246.2.ls1.otd1, <8 x float> %wide.load247.5.ls1.otd1, <8 x float> %wide.load246.5.ls1.otd1, <8 x float> %wide.load245.5.ls1.otd1, <8 x float> %wide.load248.4.ls1.otd1, <8 x float> %wide.load247.4.ls1.otd1, <8 x float> %wide.load246.4.ls1.otd1, <8 x float> %wide.load245.4.ls1.otd1, <8 x float> %wide.load248.3.ls1.otd1, <8 x float> %wide.load247.3.ls1.otd1, <8 x float> %wide.load246.3.ls1.otd1, <8 x float> %wide.load245.3.ls1.otd1, <8 x float> %wide.load248.2.ls1.otd1, <8 x float> %wide.load247.2.ls1.otd1, <8 x float> %wide.load245.2.ls1.otd1, <8 x float> %wide.load248.1.ls1.otd1, <8 x float> %wide.load247.1.ls1.otd1, <8 x float> %wide.load246.1.ls1.otd1, <8 x float> %wide.load245.1.ls1.otd1, <8 x float> %wide.load248.5.ls1.otd1, <8 x float> %wide.load248.ls1.otd1, <8 x float> %wide.load247.ls1.otd1, <8 x float> %wide.load246.ls1.otd1, <8 x float> %wide.load245.ls1.otd1, i8* %.ls1.otd1, [800 x float]* %.ls11.otd1, i8* %scevgep219220.ls1.otd1, [20 x [200 x float]]* %fusion.5.ls1.otd1, [20 x [800 x float]]* %.ls12.otd1, i8* %scevgep217218.ls1.otd1) #6
  %19 = load atomic %struct.__cilkrts_worker*, %struct.__cilkrts_worker** %5 acquire, align 8
  %20 = bitcast %struct.__cilkrts_stack_frame** %4 to i64*
  %21 = load i64, i64* %20, align 8
  %22 = getelementptr inbounds %struct.__cilkrts_worker, %struct.__cilkrts_worker* %19, i64 0, i32 9
  %23 = bitcast %struct.__cilkrts_stack_frame** %22 to i64*
  store atomic i64 %21, i64* %23 release, align 8
  store atomic %struct.__cilkrts_stack_frame* null, %struct.__cilkrts_stack_frame** %4 release, align 8
  %24 = load atomic i32, i32* %1 acquire, align 8
  %25 = icmp eq i32 %24, 16777216
  br i1 %25, label %__cilk_parent_epilogue.exit, label %body.i

body.i:                                           ; preds = %.lr.ph.otd1
  call void @__cilkrts_leave_frame(%struct.__cilkrts_stack_frame* nonnull %__cilkrts_sf) #6
  br label %__cilk_parent_epilogue.exit

__cilk_parent_epilogue.exit:                      ; preds = %.lr.ph.otd1, %body.i
  ret void
}

attributes #0 = { argmemonly nounwind "no-frame-pointer-elim"="false" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "unsafe-fp-math"="true" }
attributes #1 = { argmemonly nounwind }
attributes #2 = { nounwind readnone }
attributes #3 = { nounwind readonly }
attributes #4 = { argmemonly nounwind stealable "no-frame-pointer-elim"="false" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "unsafe-fp-math"="true" }
attributes #5 = { argmemonly noinline nounwind "no-frame-pointer-elim"="false" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "unsafe-fp-math"="true" }
attributes #6 = { nounwind }
attributes #7 = { returns_twice }
attributes #8 = { nounwind returns_twice }

!0 = !{}
!1 = !{i64 16000}
!2 = !{i64 16}
!3 = !{i64 32000}
!4 = !{i64 1280000}
!5 = !{i64 64000}
!6 = !{i64 3200}
!7 = !{i64 48}
!8 = !{i64 8}
!9 = !{!10}
!10 = !{!"buffer: {index:10, offset:0, size:48}", !11}
!11 = !{!"XLA global AA domain"}
!12 = !{!13, !14, !15, !16, !17, !18}
!13 = !{!"buffer: {index:1, offset:0, size:32000}", !11}
!14 = !{!"buffer: {index:2, offset:0, size:16000}", !11}
!15 = !{!"buffer: {index:3, offset:0, size:16000}", !11}
!16 = !{!"buffer: {index:4, offset:0, size:16000}", !11}
!17 = !{!"buffer: {index:5, offset:0, size:16000}", !11}
!18 = !{!"buffer: {index:6, offset:0, size:16000}", !11}
!19 = !{!20, !21}
!20 = !{!"buffer: {index:12, offset:0, size:64000}", !11}
!21 = distinct !{!21, !22}
!22 = distinct !{!22, !"LVerDomain"}
!23 = !{!13, !14, !15, !17, !18, !24}
!24 = !{!"buffer: {index:11, offset:0, size:4}", !11}
!25 = !{!14, !26}
!26 = distinct !{!26, !22}
!27 = !{!13, !15, !16, !17, !18, !10, !20, !21, !28}
!28 = distinct !{!28, !22}
!29 = !{!20}
!30 = !{!13}
!31 = !{!14}
!32 = !{!13, !15, !16, !17, !18, !10, !20}
!33 = distinct !{!33, !34, !35}
!34 = !{!"llvm.loop.from.tapir.loop"}
!35 = !{!"llvm.loop.isvectorized", i32 1}
!36 = distinct !{!36, !37}
!37 = !{!"tapir.loop.spawn.strategy", i32 0}
!38 = !{!20, !39}
!39 = distinct !{!39, !40}
!40 = distinct !{!40, !"LVerDomain"}
!41 = !{!42}
!42 = distinct !{!42, !40}
!43 = !{!17, !44}
!44 = distinct !{!44, !40}
!45 = !{!13, !14, !15, !16, !18, !10, !24, !20, !39, !42}
!46 = distinct !{!46, !34, !35}
!47 = !{!17}
!48 = !{!13, !14, !15, !16, !18, !10, !24, !20}
!49 = distinct !{!49, !34, !35}
!50 = distinct !{!50, !37}
!51 = !{!18}
!52 = !{!15, !16, !24, !20}
!53 = distinct !{!53, !37}
!54 = !{!15}
!55 = !{!13, !14, !16, !17, !18, !10, !24, !20}
!56 = !{!16}
!57 = !{!13, !14, !15, !17, !18, !10, !24}
!58 = distinct !{!58, !37}
!59 = !{!20, !60}
!60 = distinct !{!60, !61}
!61 = distinct !{!61, !"LVerDomain"}
!62 = !{!18, !63}
!63 = distinct !{!63, !61}
!64 = !{!15, !16, !24, !20, !60, !65}
!65 = distinct !{!65, !61}
!66 = distinct !{!66, !34, !35}
!67 = distinct !{!67, !37}
!68 = !{!20, !69}
!69 = distinct !{!69, !70}
!70 = distinct !{!70, !"LVerDomain"}
!71 = !{!15, !72}
!72 = distinct !{!72, !70}
!73 = !{!13, !14, !16, !17, !18, !10, !24, !20, !69, !74}
!74 = distinct !{!74, !70}
!75 = distinct !{!75, !34, !35}
!76 = distinct !{!76, !37}
!77 = !{!14, !15, !16, !17, !18, !10, !20}
!78 = distinct !{!78, !37}
!79 = distinct !{!79, !37}
