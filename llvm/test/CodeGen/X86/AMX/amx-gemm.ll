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

define dso_local void @inner_product(ptr %A_mem, ptr %B_mem, ptr %C_mem, i32 %M, i32 %N, i32 %K) local_unnamed_addr {
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
  %i = add nsw i64 %wide.trip.count, -1
  %xtraiter = and i64 %wide.trip.count, 7
  %i1 = icmp ult i64 %i, 7
  %unroll_iter = and i64 %wide.trip.count, 4294967288
  %lcmp.mod.not = icmp eq i64 %xtraiter, 0
  br label %for.cond3.preheader

for.cond3.preheader:                              ; preds = %for.cond.cleanup5, %for.cond3.preheader.preheader
  %indvars.iv205 = phi i64 [ 0, %for.cond3.preheader.preheader ], [ %indvars.iv.next206, %for.cond.cleanup5 ]
  %i2 = trunc i64 %indvars.iv205 to i32
  %mul11 = mul i32 %mul, %i2
  %idx.ext = sext i32 %mul11 to i64
  %add.ptr = getelementptr inbounds i32, ptr %A_mem, i64 %idx.ext
  %mul26 = mul i32 %mul25, %i2
  %idx.ext27 = sext i32 %mul26 to i64
  %add.ptr28 = getelementptr inbounds i32, ptr %C_mem, i64 %idx.ext27
  br i1 %cmp4173, label %for.body6, label %for.cond.cleanup5

for.cond.cleanup:                                 ; preds = %for.cond.cleanup5, %entry
  ret void

for.cond.cleanup5:                                ; preds = %for.cond.cleanup9, %for.cond3.preheader
  %indvars.iv.next206 = add nuw nsw i64 %indvars.iv205, 1
  %exitcond208.not = icmp eq i64 %indvars.iv.next206, %wide.trip.count207
  br i1 %exitcond208.not, label %for.cond.cleanup, label %for.cond3.preheader

for.body6:                                        ; preds = %for.cond.cleanup9, %for.cond3.preheader
  %indvars.iv199 = phi i64 [ %indvars.iv.next200, %for.cond.cleanup9 ], [ 0, %for.cond3.preheader ]
  %i3 = tail call x86_amx @llvm.x86.tilezero.internal(i16 16, i16 64)
  %i4 = shl nsw i64 %indvars.iv199, 4
  br i1 %cmp8163, label %for.body10.preheader, label %for.cond.cleanup9

for.body10.preheader:                             ; preds = %for.body6
  %add.ptr19 = getelementptr inbounds i32, ptr %B_mem, i64 %i4
  br i1 %i1, label %for.cond.cleanup9.loopexit.unr-lcssa, label %for.body10

for.cond.cleanup9.loopexit.unr-lcssa:             ; preds = %for.body10, %for.body10.preheader
  %.lcssa.ph = phi x86_amx [ undef, %for.body10.preheader ], [ %i68, %for.body10 ]
  %indvars.iv.unr = phi i64 [ 0, %for.body10.preheader ], [ %indvars.iv.next.7, %for.body10 ]
  %c.sroa.8127.2.in164.unr = phi x86_amx [ %i3, %for.body10.preheader ], [ %i68, %for.body10 ]
  br i1 %lcmp.mod.not, label %for.cond.cleanup9, label %for.body10.epil

for.body10.epil:                                  ; preds = %for.body10.epil, %for.cond.cleanup9.loopexit.unr-lcssa
  %indvars.iv.epil = phi i64 [ %indvars.iv.next.epil, %for.body10.epil ], [ %indvars.iv.unr, %for.cond.cleanup9.loopexit.unr-lcssa ]
  %c.sroa.8127.2.in164.epil = phi x86_amx [ %i11, %for.body10.epil ], [ %c.sroa.8127.2.in164.unr, %for.cond.cleanup9.loopexit.unr-lcssa ]
  %epil.iter = phi i64 [ %epil.iter.sub, %for.body10.epil ], [ %xtraiter, %for.cond.cleanup9.loopexit.unr-lcssa ]
  %i5 = shl nsw i64 %indvars.iv.epil, 4
  %add.ptr14.epil = getelementptr inbounds i32, ptr %add.ptr, i64 %i5
  %i7 = tail call x86_amx @llvm.x86.tileloadd64.internal(i16 16, i16 64, ptr %add.ptr14.epil, i64 %mul15)
  %i8 = mul nsw i64 %i5, %conv23
  %add.ptr22.epil = getelementptr inbounds i32, ptr %add.ptr19, i64 %i8
  %i10 = tail call x86_amx @llvm.x86.tileloadd64.internal(i16 16, i16 64, ptr %add.ptr22.epil, i64 %mul24)
  %i11 = tail call x86_amx @llvm.x86.tdpbssd.internal(i16 16, i16 64, i16 64, x86_amx %c.sroa.8127.2.in164.epil, x86_amx %i7, x86_amx %i10)
  %indvars.iv.next.epil = add nuw nsw i64 %indvars.iv.epil, 1
  %epil.iter.sub = add i64 %epil.iter, -1
  %epil.iter.cmp.not = icmp eq i64 %epil.iter.sub, 0
  br i1 %epil.iter.cmp.not, label %for.cond.cleanup9, label %for.body10.epil

for.cond.cleanup9:                                ; preds = %for.body10.epil, %for.cond.cleanup9.loopexit.unr-lcssa, %for.body6
  %c.sroa.8127.2.in.lcssa = phi x86_amx [ %i3, %for.body6 ], [ %.lcssa.ph, %for.cond.cleanup9.loopexit.unr-lcssa ], [ %i11, %for.body10.epil ]
  %add.ptr31 = getelementptr inbounds i32, ptr %add.ptr28, i64 %i4
  tail call void @llvm.x86.tilestored64.internal(i16 16, i16 64, ptr %add.ptr31, i64 %mul24, x86_amx %c.sroa.8127.2.in.lcssa)
  %indvars.iv.next200 = add nuw nsw i64 %indvars.iv199, 1
  %exitcond204.not = icmp eq i64 %indvars.iv.next200, %wide.trip.count203
  br i1 %exitcond204.not, label %for.cond.cleanup5, label %for.body6

for.body10:                                       ; preds = %for.body10, %for.body10.preheader
  %indvars.iv = phi i64 [ %indvars.iv.next.7, %for.body10 ], [ 0, %for.body10.preheader ]
  %c.sroa.8127.2.in164 = phi x86_amx [ %i68, %for.body10 ], [ %i3, %for.body10.preheader ]
  %niter = phi i64 [ %niter.nsub.7, %for.body10 ], [ %unroll_iter, %for.body10.preheader ]
  %i13 = shl nsw i64 %indvars.iv, 4
  %add.ptr14 = getelementptr inbounds i32, ptr %add.ptr, i64 %i13
  %i15 = tail call x86_amx @llvm.x86.tileloadd64.internal(i16 16, i16 64, ptr %add.ptr14, i64 %mul15)
  %i16 = mul nsw i64 %i13, %conv23
  %add.ptr22 = getelementptr inbounds i32, ptr %add.ptr19, i64 %i16
  %i18 = tail call x86_amx @llvm.x86.tileloadd64.internal(i16 16, i16 64, ptr %add.ptr22, i64 %mul24)
  %i19 = tail call x86_amx @llvm.x86.tdpbssd.internal(i16 16, i16 64, i16 64, x86_amx %c.sroa.8127.2.in164, x86_amx %i15, x86_amx %i18)
  %indvars.iv.next = shl i64 %indvars.iv, 4
  %i20 = or i64 %indvars.iv.next, 16
  %add.ptr14.1 = getelementptr inbounds i32, ptr %add.ptr, i64 %i20
  %i22 = tail call x86_amx @llvm.x86.tileloadd64.internal(i16 16, i16 64, ptr nonnull %add.ptr14.1, i64 %mul15)
  %i23 = mul nsw i64 %i20, %conv23
  %add.ptr22.1 = getelementptr inbounds i32, ptr %add.ptr19, i64 %i23
  %i25 = tail call x86_amx @llvm.x86.tileloadd64.internal(i16 16, i16 64, ptr nonnull %add.ptr22.1, i64 %mul24)
  %i26 = tail call x86_amx @llvm.x86.tdpbssd.internal(i16 16, i16 64, i16 64, x86_amx %i19, x86_amx %i22, x86_amx %i25)
  %indvars.iv.next.1 = shl i64 %indvars.iv, 4
  %i27 = or i64 %indvars.iv.next.1, 32
  %add.ptr14.2 = getelementptr inbounds i32, ptr %add.ptr, i64 %i27
  %i29 = tail call x86_amx @llvm.x86.tileloadd64.internal(i16 16, i16 64, ptr nonnull %add.ptr14.2, i64 %mul15)
  %i30 = mul nsw i64 %i27, %conv23
  %add.ptr22.2 = getelementptr inbounds i32, ptr %add.ptr19, i64 %i30
  %i32 = tail call x86_amx @llvm.x86.tileloadd64.internal(i16 16, i16 64, ptr nonnull %add.ptr22.2, i64 %mul24)
  %i33 = tail call x86_amx @llvm.x86.tdpbssd.internal(i16 16, i16 64, i16 64, x86_amx %i26, x86_amx %i29, x86_amx %i32)
  %indvars.iv.next.2 = shl i64 %indvars.iv, 4
  %i34 = or i64 %indvars.iv.next.2, 48
  %add.ptr14.3 = getelementptr inbounds i32, ptr %add.ptr, i64 %i34
  %i36 = tail call x86_amx @llvm.x86.tileloadd64.internal(i16 16, i16 64, ptr nonnull %add.ptr14.3, i64 %mul15)
  %i37 = mul nsw i64 %i34, %conv23
  %add.ptr22.3 = getelementptr inbounds i32, ptr %add.ptr19, i64 %i37
  %i39 = tail call x86_amx @llvm.x86.tileloadd64.internal(i16 16, i16 64, ptr nonnull %add.ptr22.3, i64 %mul24)
  %i40 = tail call x86_amx @llvm.x86.tdpbssd.internal(i16 16, i16 64, i16 64, x86_amx %i33, x86_amx %i36, x86_amx %i39)
  %indvars.iv.next.3 = shl i64 %indvars.iv, 4
  %i41 = or i64 %indvars.iv.next.3, 64
  %add.ptr14.4 = getelementptr inbounds i32, ptr %add.ptr, i64 %i41
  %i43 = tail call x86_amx @llvm.x86.tileloadd64.internal(i16 16, i16 64, ptr nonnull %add.ptr14.4, i64 %mul15)
  %i44 = mul nsw i64 %i41, %conv23
  %add.ptr22.4 = getelementptr inbounds i32, ptr %add.ptr19, i64 %i44
  %i46 = tail call x86_amx @llvm.x86.tileloadd64.internal(i16 16, i16 64, ptr nonnull %add.ptr22.4, i64 %mul24)
  %i47 = tail call x86_amx @llvm.x86.tdpbssd.internal(i16 16, i16 64, i16 64, x86_amx %i40, x86_amx %i43, x86_amx %i46)
  %indvars.iv.next.4 = shl i64 %indvars.iv, 4
  %i48 = or i64 %indvars.iv.next.4, 80
  %add.ptr14.5 = getelementptr inbounds i32, ptr %add.ptr, i64 %i48
  %i50 = tail call x86_amx @llvm.x86.tileloadd64.internal(i16 16, i16 64, ptr nonnull %add.ptr14.5, i64 %mul15)
  %i51 = mul nsw i64 %i48, %conv23
  %add.ptr22.5 = getelementptr inbounds i32, ptr %add.ptr19, i64 %i51
  %i53 = tail call x86_amx @llvm.x86.tileloadd64.internal(i16 16, i16 64, ptr nonnull %add.ptr22.5, i64 %mul24)
  %i54 = tail call x86_amx @llvm.x86.tdpbssd.internal(i16 16, i16 64, i16 64, x86_amx %i47, x86_amx %i50, x86_amx %i53)
  %indvars.iv.next.5 = shl i64 %indvars.iv, 4
  %i55 = or i64 %indvars.iv.next.5, 96
  %add.ptr14.6 = getelementptr inbounds i32, ptr %add.ptr, i64 %i55
  %i57 = tail call x86_amx @llvm.x86.tileloadd64.internal(i16 16, i16 64, ptr nonnull %add.ptr14.6, i64 %mul15)
  %i58 = mul nsw i64 %i55, %conv23
  %add.ptr22.6 = getelementptr inbounds i32, ptr %add.ptr19, i64 %i58
  %i60 = tail call x86_amx @llvm.x86.tileloadd64.internal(i16 16, i16 64, ptr nonnull %add.ptr22.6, i64 %mul24)
  %i61 = tail call x86_amx @llvm.x86.tdpbssd.internal(i16 16, i16 64, i16 64, x86_amx %i54, x86_amx %i57, x86_amx %i60)
  %indvars.iv.next.6 = shl i64 %indvars.iv, 4
  %i62 = or i64 %indvars.iv.next.6, 112
  %add.ptr14.7 = getelementptr inbounds i32, ptr %add.ptr, i64 %i62
  %i64 = tail call x86_amx @llvm.x86.tileloadd64.internal(i16 16, i16 64, ptr nonnull %add.ptr14.7, i64 %mul15)
  %i65 = mul nsw i64 %i62, %conv23
  %add.ptr22.7 = getelementptr inbounds i32, ptr %add.ptr19, i64 %i65
  %i67 = tail call x86_amx @llvm.x86.tileloadd64.internal(i16 16, i16 64, ptr nonnull %add.ptr22.7, i64 %mul24)
  %i68 = tail call x86_amx @llvm.x86.tdpbssd.internal(i16 16, i16 64, i16 64, x86_amx %i61, x86_amx %i64, x86_amx %i67)
  %indvars.iv.next.7 = add nuw nsw i64 %indvars.iv, 8
  %niter.nsub.7 = add i64 %niter, -8
  %niter.ncmp.7 = icmp eq i64 %niter.nsub.7, 0
  br i1 %niter.ncmp.7, label %for.cond.cleanup9.loopexit.unr-lcssa, label %for.body10
}

declare x86_amx @llvm.x86.tilezero.internal(i16, i16)
declare x86_amx @llvm.x86.tileloadd64.internal(i16, i16, ptr, i64)
declare x86_amx @llvm.x86.tdpbssd.internal(i16, i16, i16, x86_amx, x86_amx, x86_amx)
declare void @llvm.x86.tilestored64.internal(i16, i16, ptr, i64, x86_amx)
