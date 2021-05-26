; RUN: llc < %s -mtriple=x86_64-unknown-unknown -mattr=+amx-int8 -mattr=+avx512f -verify-machineinstrs | FileCheck %s

; #include <immintrin.h>
;
; #define TILE_SZ 16
; void inner_product(int *A_mem, int *B_mem, int *C_mem, int M, int N, int K) {
;   const int m = M / TILE_SZ;
;   const int n = N / TILE_SZ;
;   const int k = K / TILE_SZ;
;
;   for (int i = 0; i < m; i++)
;     for (int j = 0; j < n; j++) {
;       __tile1024i c = {TILE_SZ, TILE_SZ*sizeof(int)};
;       __tile_zero(&c);
;       for (int l = 0; l < k; l++) {
;         __tile1024i a = {TILE_SZ, TILE_SZ*sizeof(int)};
;         __tile1024i b = {TILE_SZ, TILE_SZ*sizeof(int)};
;         __tile_loadd(&a, A_mem+(i*TILE_SZ)*K+l*TILE_SZ, K*sizeof(int));
;         __tile_loadd(&b, B_mem+(l*TILE_SZ)*N+j*TILE_SZ, N*sizeof(int));
;         __tile_dpbssd(&c, a, b);
;       }
;       __tile_stored(C_mem+(i*TILE_SZ)*M+j*TILE_SZ, N*sizeof(int), c);
;     }
; }

; CHECK:  ldtilecfg

; Function Attrs: noinline nounwind uwtable
define dso_local void @inner_product(i32* %A_mem, i32* %B_mem, i32* %C_mem, i32 %M, i32 %N, i32 %K) local_unnamed_addr {
entry:
  %mul = shl i32 %K, 4
  %conv = sext i32 %K to i64
  %mul15 = shl nsw i64 %conv, 2
  %conv23 = sext i32 %N to i64
  %mul24 = shl nsw i64 %conv23, 2
  %cmp8163 = icmp sgt i32 %K, 15
  %mul25 = shl i32 %M, 4
  %cmp4173 = icmp sgt i32 %N, 15
  %cmp187 = icmp sgt i32 %M, 15
  br i1 %cmp187, label %for.cond3.preheader.preheader, label %for.cond.cleanup

for.cond3.preheader.preheader:                    ; preds = %entry
  %div2 = sdiv i32 %K, 16
  %div1 = sdiv i32 %N, 16
  %div209 = lshr i32 %M, 4
  %wide.trip.count207 = zext i32 %div209 to i64
  %wide.trip.count203 = zext i32 %div1 to i64
  %wide.trip.count = zext i32 %div2 to i64
  %0 = add nsw i64 %wide.trip.count, -1
  %xtraiter = and i64 %wide.trip.count, 7
  %1 = icmp ult i64 %0, 7
  %unroll_iter = and i64 %wide.trip.count, 4294967288
  %lcmp.mod.not = icmp eq i64 %xtraiter, 0
  br label %for.cond3.preheader

for.cond3.preheader:                              ; preds = %for.cond3.preheader.preheader, %for.cond.cleanup5
  %indvars.iv205 = phi i64 [ 0, %for.cond3.preheader.preheader ], [ %indvars.iv.next206, %for.cond.cleanup5 ]
  %2 = trunc i64 %indvars.iv205 to i32
  %mul11 = mul i32 %mul, %2
  %idx.ext = sext i32 %mul11 to i64
  %add.ptr = getelementptr inbounds i32, i32* %A_mem, i64 %idx.ext
  %mul26 = mul i32 %mul25, %2
  %idx.ext27 = sext i32 %mul26 to i64
  %add.ptr28 = getelementptr inbounds i32, i32* %C_mem, i64 %idx.ext27
  br i1 %cmp4173, label %for.body6, label %for.cond.cleanup5

for.cond.cleanup:                                 ; preds = %for.cond.cleanup5, %entry
  ret void

for.cond.cleanup5:                                ; preds = %for.cond.cleanup9, %for.cond3.preheader
  %indvars.iv.next206 = add nuw nsw i64 %indvars.iv205, 1
  %exitcond208.not = icmp eq i64 %indvars.iv.next206, %wide.trip.count207
  br i1 %exitcond208.not, label %for.cond.cleanup, label %for.cond3.preheader

for.body6:                                        ; preds = %for.cond3.preheader, %for.cond.cleanup9
  %indvars.iv199 = phi i64 [ %indvars.iv.next200, %for.cond.cleanup9 ], [ 0, %for.cond3.preheader ]
  %3 = tail call x86_amx @llvm.x86.tilezero.internal(i16 16, i16 64)
  %4 = shl nsw i64 %indvars.iv199, 4
  br i1 %cmp8163, label %for.body10.preheader, label %for.cond.cleanup9

for.body10.preheader:                             ; preds = %for.body6
  %add.ptr19 = getelementptr inbounds i32, i32* %B_mem, i64 %4
  br i1 %1, label %for.cond.cleanup9.loopexit.unr-lcssa, label %for.body10

for.cond.cleanup9.loopexit.unr-lcssa:             ; preds = %for.body10, %for.body10.preheader
  %.lcssa.ph = phi x86_amx [ undef, %for.body10.preheader ], [ %68, %for.body10 ]
  %indvars.iv.unr = phi i64 [ 0, %for.body10.preheader ], [ %indvars.iv.next.7, %for.body10 ]
  %c.sroa.8127.2.in164.unr = phi x86_amx [ %3, %for.body10.preheader ], [ %68, %for.body10 ]
  br i1 %lcmp.mod.not, label %for.cond.cleanup9, label %for.body10.epil

for.body10.epil:                                  ; preds = %for.cond.cleanup9.loopexit.unr-lcssa, %for.body10.epil
  %indvars.iv.epil = phi i64 [ %indvars.iv.next.epil, %for.body10.epil ], [ %indvars.iv.unr, %for.cond.cleanup9.loopexit.unr-lcssa ]
  %c.sroa.8127.2.in164.epil = phi x86_amx [ %11, %for.body10.epil ], [ %c.sroa.8127.2.in164.unr, %for.cond.cleanup9.loopexit.unr-lcssa ]
  %epil.iter = phi i64 [ %epil.iter.sub, %for.body10.epil ], [ %xtraiter, %for.cond.cleanup9.loopexit.unr-lcssa ]
  %5 = shl nsw i64 %indvars.iv.epil, 4
  %add.ptr14.epil = getelementptr inbounds i32, i32* %add.ptr, i64 %5
  %6 = bitcast i32* %add.ptr14.epil to i8*
  %7 = tail call x86_amx @llvm.x86.tileloadd64.internal(i16 16, i16 64, i8* %6, i64 %mul15)
  %8 = mul nsw i64 %5, %conv23
  %add.ptr22.epil = getelementptr inbounds i32, i32* %add.ptr19, i64 %8
  %9 = bitcast i32* %add.ptr22.epil to i8*
  %10 = tail call x86_amx @llvm.x86.tileloadd64.internal(i16 16, i16 64, i8* %9, i64 %mul24)
  %11 = tail call x86_amx @llvm.x86.tdpbssd.internal(i16 16, i16 64, i16 64, x86_amx %c.sroa.8127.2.in164.epil, x86_amx %7, x86_amx %10)
  %indvars.iv.next.epil = add nuw nsw i64 %indvars.iv.epil, 1
  %epil.iter.sub = add i64 %epil.iter, -1
  %epil.iter.cmp.not = icmp eq i64 %epil.iter.sub, 0
  br i1 %epil.iter.cmp.not, label %for.cond.cleanup9, label %for.body10.epil

for.cond.cleanup9:                                ; preds = %for.cond.cleanup9.loopexit.unr-lcssa, %for.body10.epil, %for.body6
  %c.sroa.8127.2.in.lcssa = phi x86_amx [ %3, %for.body6 ], [ %.lcssa.ph, %for.cond.cleanup9.loopexit.unr-lcssa ], [ %11, %for.body10.epil ]
  %add.ptr31 = getelementptr inbounds i32, i32* %add.ptr28, i64 %4
  %12 = bitcast i32* %add.ptr31 to i8*
  tail call void @llvm.x86.tilestored64.internal(i16 16, i16 64, i8* %12, i64 %mul24, x86_amx %c.sroa.8127.2.in.lcssa)
  %indvars.iv.next200 = add nuw nsw i64 %indvars.iv199, 1
  %exitcond204.not = icmp eq i64 %indvars.iv.next200, %wide.trip.count203
  br i1 %exitcond204.not, label %for.cond.cleanup5, label %for.body6

for.body10:                                       ; preds = %for.body10.preheader, %for.body10
  %indvars.iv = phi i64 [ %indvars.iv.next.7, %for.body10 ], [ 0, %for.body10.preheader ]
  %c.sroa.8127.2.in164 = phi x86_amx [ %68, %for.body10 ], [ %3, %for.body10.preheader ]
  %niter = phi i64 [ %niter.nsub.7, %for.body10 ], [ %unroll_iter, %for.body10.preheader ]
  %13 = shl nsw i64 %indvars.iv, 4
  %add.ptr14 = getelementptr inbounds i32, i32* %add.ptr, i64 %13
  %14 = bitcast i32* %add.ptr14 to i8*
  %15 = tail call x86_amx @llvm.x86.tileloadd64.internal(i16 16, i16 64, i8* %14, i64 %mul15)
  %16 = mul nsw i64 %13, %conv23
  %add.ptr22 = getelementptr inbounds i32, i32* %add.ptr19, i64 %16
  %17 = bitcast i32* %add.ptr22 to i8*
  %18 = tail call x86_amx @llvm.x86.tileloadd64.internal(i16 16, i16 64, i8* %17, i64 %mul24)
  %19 = tail call x86_amx @llvm.x86.tdpbssd.internal(i16 16, i16 64, i16 64, x86_amx %c.sroa.8127.2.in164, x86_amx %15, x86_amx %18)
  %indvars.iv.next = shl i64 %indvars.iv, 4
  %20 = or i64 %indvars.iv.next, 16
  %add.ptr14.1 = getelementptr inbounds i32, i32* %add.ptr, i64 %20
  %21 = bitcast i32* %add.ptr14.1 to i8*
  %22 = tail call x86_amx @llvm.x86.tileloadd64.internal(i16 16, i16 64, i8* nonnull %21, i64 %mul15)
  %23 = mul nsw i64 %20, %conv23
  %add.ptr22.1 = getelementptr inbounds i32, i32* %add.ptr19, i64 %23
  %24 = bitcast i32* %add.ptr22.1 to i8*
  %25 = tail call x86_amx @llvm.x86.tileloadd64.internal(i16 16, i16 64, i8* nonnull %24, i64 %mul24)
  %26 = tail call x86_amx @llvm.x86.tdpbssd.internal(i16 16, i16 64, i16 64, x86_amx %19, x86_amx %22, x86_amx %25)
  %indvars.iv.next.1 = shl i64 %indvars.iv, 4
  %27 = or i64 %indvars.iv.next.1, 32
  %add.ptr14.2 = getelementptr inbounds i32, i32* %add.ptr, i64 %27
  %28 = bitcast i32* %add.ptr14.2 to i8*
  %29 = tail call x86_amx @llvm.x86.tileloadd64.internal(i16 16, i16 64, i8* nonnull %28, i64 %mul15)
  %30 = mul nsw i64 %27, %conv23
  %add.ptr22.2 = getelementptr inbounds i32, i32* %add.ptr19, i64 %30
  %31 = bitcast i32* %add.ptr22.2 to i8*
  %32 = tail call x86_amx @llvm.x86.tileloadd64.internal(i16 16, i16 64, i8* nonnull %31, i64 %mul24)
  %33 = tail call x86_amx @llvm.x86.tdpbssd.internal(i16 16, i16 64, i16 64, x86_amx %26, x86_amx %29, x86_amx %32)
  %indvars.iv.next.2 = shl i64 %indvars.iv, 4
  %34 = or i64 %indvars.iv.next.2, 48
  %add.ptr14.3 = getelementptr inbounds i32, i32* %add.ptr, i64 %34
  %35 = bitcast i32* %add.ptr14.3 to i8*
  %36 = tail call x86_amx @llvm.x86.tileloadd64.internal(i16 16, i16 64, i8* nonnull %35, i64 %mul15)
  %37 = mul nsw i64 %34, %conv23
  %add.ptr22.3 = getelementptr inbounds i32, i32* %add.ptr19, i64 %37
  %38 = bitcast i32* %add.ptr22.3 to i8*
  %39 = tail call x86_amx @llvm.x86.tileloadd64.internal(i16 16, i16 64, i8* nonnull %38, i64 %mul24)
  %40 = tail call x86_amx @llvm.x86.tdpbssd.internal(i16 16, i16 64, i16 64, x86_amx %33, x86_amx %36, x86_amx %39)
  %indvars.iv.next.3 = shl i64 %indvars.iv, 4
  %41 = or i64 %indvars.iv.next.3, 64
  %add.ptr14.4 = getelementptr inbounds i32, i32* %add.ptr, i64 %41
  %42 = bitcast i32* %add.ptr14.4 to i8*
  %43 = tail call x86_amx @llvm.x86.tileloadd64.internal(i16 16, i16 64, i8* nonnull %42, i64 %mul15)
  %44 = mul nsw i64 %41, %conv23
  %add.ptr22.4 = getelementptr inbounds i32, i32* %add.ptr19, i64 %44
  %45 = bitcast i32* %add.ptr22.4 to i8*
  %46 = tail call x86_amx @llvm.x86.tileloadd64.internal(i16 16, i16 64, i8* nonnull %45, i64 %mul24)
  %47 = tail call x86_amx @llvm.x86.tdpbssd.internal(i16 16, i16 64, i16 64, x86_amx %40, x86_amx %43, x86_amx %46)
  %indvars.iv.next.4 = shl i64 %indvars.iv, 4
  %48 = or i64 %indvars.iv.next.4, 80
  %add.ptr14.5 = getelementptr inbounds i32, i32* %add.ptr, i64 %48
  %49 = bitcast i32* %add.ptr14.5 to i8*
  %50 = tail call x86_amx @llvm.x86.tileloadd64.internal(i16 16, i16 64, i8* nonnull %49, i64 %mul15)
  %51 = mul nsw i64 %48, %conv23
  %add.ptr22.5 = getelementptr inbounds i32, i32* %add.ptr19, i64 %51
  %52 = bitcast i32* %add.ptr22.5 to i8*
  %53 = tail call x86_amx @llvm.x86.tileloadd64.internal(i16 16, i16 64, i8* nonnull %52, i64 %mul24)
  %54 = tail call x86_amx @llvm.x86.tdpbssd.internal(i16 16, i16 64, i16 64, x86_amx %47, x86_amx %50, x86_amx %53)
  %indvars.iv.next.5 = shl i64 %indvars.iv, 4
  %55 = or i64 %indvars.iv.next.5, 96
  %add.ptr14.6 = getelementptr inbounds i32, i32* %add.ptr, i64 %55
  %56 = bitcast i32* %add.ptr14.6 to i8*
  %57 = tail call x86_amx @llvm.x86.tileloadd64.internal(i16 16, i16 64, i8* nonnull %56, i64 %mul15)
  %58 = mul nsw i64 %55, %conv23
  %add.ptr22.6 = getelementptr inbounds i32, i32* %add.ptr19, i64 %58
  %59 = bitcast i32* %add.ptr22.6 to i8*
  %60 = tail call x86_amx @llvm.x86.tileloadd64.internal(i16 16, i16 64, i8* nonnull %59, i64 %mul24)
  %61 = tail call x86_amx @llvm.x86.tdpbssd.internal(i16 16, i16 64, i16 64, x86_amx %54, x86_amx %57, x86_amx %60)
  %indvars.iv.next.6 = shl i64 %indvars.iv, 4
  %62 = or i64 %indvars.iv.next.6, 112
  %add.ptr14.7 = getelementptr inbounds i32, i32* %add.ptr, i64 %62
  %63 = bitcast i32* %add.ptr14.7 to i8*
  %64 = tail call x86_amx @llvm.x86.tileloadd64.internal(i16 16, i16 64, i8* nonnull %63, i64 %mul15)
  %65 = mul nsw i64 %62, %conv23
  %add.ptr22.7 = getelementptr inbounds i32, i32* %add.ptr19, i64 %65
  %66 = bitcast i32* %add.ptr22.7 to i8*
  %67 = tail call x86_amx @llvm.x86.tileloadd64.internal(i16 16, i16 64, i8* nonnull %66, i64 %mul24)
  %68 = tail call x86_amx @llvm.x86.tdpbssd.internal(i16 16, i16 64, i16 64, x86_amx %61, x86_amx %64, x86_amx %67)
  %indvars.iv.next.7 = add nuw nsw i64 %indvars.iv, 8
  %niter.nsub.7 = add i64 %niter, -8
  %niter.ncmp.7 = icmp eq i64 %niter.nsub.7, 0
  br i1 %niter.ncmp.7, label %for.cond.cleanup9.loopexit.unr-lcssa, label %for.body10
}

declare x86_amx @llvm.x86.tilezero.internal(i16, i16)
declare x86_amx @llvm.x86.tileloadd64.internal(i16, i16, i8*, i64)
declare x86_amx @llvm.x86.tdpbssd.internal(i16, i16, i16, x86_amx, x86_amx, x86_amx)
declare void @llvm.x86.tilestored64.internal(i16, i16, i8*, i64, x86_amx)
