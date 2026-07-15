; RUN: llc -march=hexagon -mattr=+hvxv73,+hvx-length128b,-long-calls -hexagon-allow-scatter-gather-hvx < %s | FileCheck %s

; CHECK-LABEL: Ripple_gather_32:
; CHECK: vtmp.w = vgather
; CHECK-LABEL: Ripple_gather_16:
; CHECK: vtmp.h = vgather
; CHECK-LABEL: Ripple_gather_8:
; CHECK: vand
; CHECK: vpacke

target datalayout = "e-m:e-p:32:32:32-a:0-n16:32-i64:64:64-i32:32:32-i16:16:16-i1:8:8-f32:32:32-f64:64:64-v32:32:32-v64:64:64-v512:512:512-v1024:1024:1024-v2048:2048:2048"
target triple = "hexagon"

; Function Attrs: nofree noinline norecurse nosync nounwind memory(argmem: readwrite, inaccessiblemem: readwrite)
define dso_local void @Ripple_gather_32(ptr nocapture noundef writeonly %destination, ptr nocapture noundef readonly %source, ptr nocapture noundef readonly %indexes) local_unnamed_addr #0 {
entry:
  %source.ripple.bcast.splatinsert = insertelement <32 x ptr> poison, ptr %source, i64 0
  %source.ripple.bcast.splat = shufflevector <32 x ptr> %source.ripple.bcast.splatinsert, <32 x ptr> poison, <32 x i32> zeroinitializer
  %0 = load <32 x i32>, ptr %indexes, align 4
  %arrayidx2.ripple.vectorized = getelementptr inbounds i32, <32 x ptr> %source.ripple.bcast.splat, <32 x i32> %0
  %1 = tail call <32 x i32> @llvm.masked.gather.v32i32.v32p0(<32 x ptr> %arrayidx2.ripple.vectorized, i32 4, <32 x i1> <i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true>, <32 x i32> poison)
  store <32 x i32> %1, ptr %destination, align 4
  ret void
}

; Function Attrs: nofree noinline norecurse nosync nounwind memory(argmem: readwrite, inaccessiblemem: readwrite)
define dso_local void @Ripple_gather_16(ptr nocapture noundef writeonly %destination, ptr nocapture noundef readonly %source, ptr nocapture noundef readonly %indexes) local_unnamed_addr #0 {
entry:
  %source.ripple.bcast.splatinsert = insertelement <64 x ptr> poison, ptr %source, i64 0
  %source.ripple.bcast.splat = shufflevector <64 x ptr> %source.ripple.bcast.splatinsert, <64 x ptr> poison, <64 x i32> zeroinitializer
  %0 = load <64 x i16>, ptr %indexes, align 2
  %idxprom.ripple.vectorized = zext <64 x i16> %0 to <64 x i32>
  %arrayidx2.ripple.vectorized = getelementptr inbounds i16, <64 x ptr> %source.ripple.bcast.splat, <64 x i32> %idxprom.ripple.vectorized
  %1 = tail call <64 x i16> @llvm.masked.gather.v64i16.v64p0(<64 x ptr> %arrayidx2.ripple.vectorized, i32 2, <64 x i1> <i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true>, <64 x i16> poison)
  store <64 x i16> %1, ptr %destination, align 2
  ret void
}

; Function Attrs: nofree noinline norecurse nosync nounwind memory(argmem: readwrite, inaccessiblemem: readwrite)
define dso_local void @Ripple_gather_8(ptr nocapture noundef writeonly %destination, ptr nocapture noundef readonly %source, ptr nocapture noundef readonly %indexes) local_unnamed_addr #0 {
entry:
  %source.ripple.bcast.splatinsert = insertelement <128 x ptr> poison, ptr %source, i64 0
  %source.ripple.bcast.splat = shufflevector <128 x ptr> %source.ripple.bcast.splatinsert, <128 x ptr> poison, <128 x i32> zeroinitializer
  %0 = load <128 x i8>, ptr %indexes, align 1
  %idxprom.ripple.vectorized = zext <128 x i8> %0 to <128 x i32>
  %arrayidx2.ripple.vectorized = getelementptr inbounds i8, <128 x ptr> %source.ripple.bcast.splat, <128 x i32> %idxprom.ripple.vectorized
  %1 = tail call <128 x i8> @llvm.masked.gather.v128i8.v128p0(<128 x ptr> %arrayidx2.ripple.vectorized, i32 1, <128 x i1> <i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true>, <128 x i8> poison)
  store <128 x i8> %1, ptr %destination, align 1
  ret void
}

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(read)
declare <32 x i32> @llvm.masked.gather.v32i32.v32p0(<32 x ptr>, i32 immarg, <32 x i1>, <32 x i32>) #1
declare <64 x i16> @llvm.masked.gather.v64i16.v64p0(<64 x ptr>, i32 immarg, <64 x i1>, <64 x i16>) #1
declare <128 x i8> @llvm.masked.gather.v128i8.v128p0(<128 x ptr> %0, i32 immarg %1, <128 x i1> %2, <128 x i8> %3) #1
