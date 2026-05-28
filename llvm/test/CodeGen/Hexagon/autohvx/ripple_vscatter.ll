; RUN: llc -march=hexagon -mattr=+hvx-length128b,+hvxv73,+v73,-long-calls -hexagon-allow-scatter-gather-hvx < %s | FileCheck %s

; CHECK-LABEL: Ripple_scatter_8:
; CHECK: if (q{{[0-9]+}}) vscatter(r{{[0-9]+}},m0,v{{[0-9]+}}.h).h
; CHECK: if (q{{[0-9]+}}) vscatter(r{{[0-9]+}},m0,v{{[0-9]+}}.h).h
; CHECK-LABEL: Ripple_scatter_16:
; CHECK: vscatter(r{{[0-9]+}},m0,v{{[0-9]+}}.h).h = v{{[0-9]+}}
; CHECK-LABEL: Ripple_scatter_32:
; CHECK: vscatter(r{{[0-9]+}},m0,v{{[0-9]+}}.w).w = v{{[0-9]+}}

target datalayout = "e-m:e-p:32:32:32-a:0-n16:32-i64:64:64-i32:32:32-i16:16:16-i1:8:8-f32:32:32-f64:64:64-v32:32:32-v64:64:64-v512:512:512-v1024:1024:1024-v2048:2048:2048"
target triple = "hexagon"

define dso_local void @Ripple_scatter_8(ptr nocapture noundef writeonly %destination, ptr nocapture noundef readonly %source, ptr nocapture noundef readonly %indexes) local_unnamed_addr #0 {
entry:
  %destination.ripple.bcast.splatinsert = insertelement <128 x ptr> poison, ptr %destination, i64 0
  %destination.ripple.bcast.splat = shufflevector <128 x ptr> %destination.ripple.bcast.splatinsert, <128 x ptr> poison, <128 x i32> zeroinitializer
  %.ripple.LS.instance11 = load <128 x i8>, ptr %source, align 1
  %.ripple.LS.instance = load <128 x i8>, ptr %indexes, align 1
  %idxprom.ripple.LS.instance = zext <128 x i8> %.ripple.LS.instance to <128 x i32>
  %arrayidx3.ripple.LS.instance = getelementptr inbounds i8, <128 x ptr> %destination.ripple.bcast.splat, <128 x i32> %idxprom.ripple.LS.instance
  %cst_ptr_to_i32 = ptrtoint ptr %destination to i32
  tail call void @llvm.masked.scatter.v128i8.v128p0(<128 x i8> %.ripple.LS.instance11, <128 x ptr> %arrayidx3.ripple.LS.instance, i32 1, <128 x i1> <i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true>)
  ret void
}

define dso_local void @Ripple_scatter_16(ptr nocapture noundef writeonly %destination, ptr nocapture noundef readonly %source, ptr nocapture noundef readonly %indexes) local_unnamed_addr #0 {
entry:
  %destination.ripple.bcast.splatinsert = insertelement <64 x ptr> poison, ptr %destination, i64 0
  %destination.ripple.bcast.splat = shufflevector <64 x ptr> %destination.ripple.bcast.splatinsert, <64 x ptr> poison, <64 x i32> zeroinitializer
  %.ripple.LS.instance11 = load <64 x i16>, ptr %source, align 2
  %.ripple.LS.instance = load <64 x i16>, ptr %indexes, align 2
  %idxprom.ripple.LS.instance = zext <64 x i16> %.ripple.LS.instance to <64 x i32>
  %arrayidx3.ripple.LS.instance = getelementptr inbounds i16, <64 x ptr> %destination.ripple.bcast.splat, <64 x i32> %idxprom.ripple.LS.instance
  tail call void @llvm.masked.scatter.v64i16.v64p0(<64 x i16> %.ripple.LS.instance11, <64 x ptr> %arrayidx3.ripple.LS.instance, i32 2, <64 x i1> <i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true>)
  ret void
}

define dso_local void @Ripple_scatter_32(ptr nocapture noundef writeonly %destination, ptr nocapture noundef readonly %source, ptr nocapture noundef readonly %indexes) local_unnamed_addr #0 {
entry:
  %destination.ripple.bcast.splatinsert = insertelement <32 x ptr> poison, ptr %destination, i64 0
  %destination.ripple.bcast.splat = shufflevector <32 x ptr> %destination.ripple.bcast.splatinsert, <32 x ptr> poison, <32 x i32> zeroinitializer
  %.ripple.LS.instance11 = load <32 x i32>, ptr %source, align 4
  %.ripple.LS.instance = load <32 x i32>, ptr %indexes, align 4
  %arrayidx3.ripple.LS.instance = getelementptr inbounds i32, <32 x ptr> %destination.ripple.bcast.splat, <32 x i32> %.ripple.LS.instance
  tail call void @llvm.masked.scatter.v32i32.v32p0(<32 x i32> %.ripple.LS.instance11, <32 x ptr> %arrayidx3.ripple.LS.instance, i32 4, <32 x i1> <i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true>)
  ret void
}

declare void @llvm.masked.scatter.v128i8.v128p0(<128 x i8> %0, <128 x ptr> %1, i32 immarg %2, <128 x i1> %3) #2
declare void @llvm.masked.scatter.v64i16.v64p0(<64 x i16> %0, <64 x ptr> %1, i32 immarg %2, <64 x i1> %3) #2
declare void @llvm.masked.scatter.v32i32.v32p0(<32 x i32> %0, <32 x ptr> %1, i32 immarg %2, <32 x i1> %3) #2
