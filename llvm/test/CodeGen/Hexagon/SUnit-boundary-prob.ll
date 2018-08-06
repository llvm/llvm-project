; REQUIRES: asserts
; RUN: llc -march=hexagon -O2 -mcpu=hexagonv60 --stats -o - 2>&1  < %s | FileCheck %s
; This was aborting while processing SUnits.

; CHECK: vmem

; CHECK-NOT: Number of node order issues found
; CHECK: Number of loops software pipelined
; CHECK-NOT: Number of node order issues found
source_filename = "bugpoint-output-bdb0052.bc"
target datalayout = "e-m:e-p:32:32:32-a:0-n16:32-i64:64:64-i32:32:32-i16:16:16-i1:8:8-f32:32:32-f64:64:64-v32:32:32-v64:64:64-v512:512:512-v1024:1024:1024-v2048:2048:2048"
target triple = "hexagon-unknown--elf"

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.lo(<32 x i32>) #0

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.hi(<32 x i32>) #0

; Function Attrs: nounwind readnone
declare <32 x i32> @llvm.hexagon.V6.vshuffvdd(<16 x i32>, <16 x i32>, i32) #0

; Function Attrs: nounwind readnone
declare <32 x i32> @llvm.hexagon.V6.vdealvdd(<16 x i32>, <16 x i32>, i32) #0

; Function Attrs: nounwind readnone
declare <32 x i32> @llvm.hexagon.V6.vcombine(<16 x i32>, <16 x i32>) #0

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.vshufeh(<16 x i32>, <16 x i32>) #0

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.vshufoh(<16 x i32>, <16 x i32>) #0

; Function Attrs: nounwind readnone
declare <32 x i32> @llvm.hexagon.V6.vmpyuhv(<16 x i32>, <16 x i32>) #0

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.vaslw.acc(<16 x i32>, <16 x i32>, i32) #0

define void @__error_op_vmpy_v__uh_v__uh__1() #1 {
entry:
  %in_u16.host181 = load i16*, i16** undef, align 4
  %in_u32.host182 = load i32*, i32** undef, align 4
  br label %"for op_vmpy_v__uh_v__uh__1.s0.y"

"for op_vmpy_v__uh_v__uh__1.s0.y":                ; preds = %"end for op_vmpy_v__uh_v__uh__1.s0.x.x", %entry
  %op_vmpy_v__uh_v__uh__1.s0.y = phi i32 [ 0, %entry ], [ %63, %"end for op_vmpy_v__uh_v__uh__1.s0.x.x" ]
  %0 = mul nuw nsw i32 %op_vmpy_v__uh_v__uh__1.s0.y, 768
  %1 = add nuw nsw i32 %0, 32
  %2 = add nuw nsw i32 %0, 64
  %3 = add nuw nsw i32 %0, 96
  br label %"for op_vmpy_v__uh_v__uh__1.s0.x.x"

"for op_vmpy_v__uh_v__uh__1.s0.x.x":              ; preds = %"for op_vmpy_v__uh_v__uh__1.s0.x.x", %"for op_vmpy_v__uh_v__uh__1.s0.y"
  %.phi210 = phi i32* [ %in_u32.host182, %"for op_vmpy_v__uh_v__uh__1.s0.y" ], [ %.inc211.3, %"for op_vmpy_v__uh_v__uh__1.s0.x.x" ]
  %.phi213 = phi i16* [ %in_u16.host181, %"for op_vmpy_v__uh_v__uh__1.s0.y" ], [ %.inc214.3, %"for op_vmpy_v__uh_v__uh__1.s0.x.x" ]
  %op_vmpy_v__uh_v__uh__1.s0.x.x = phi i32 [ 0, %"for op_vmpy_v__uh_v__uh__1.s0.y" ], [ %61, %"for op_vmpy_v__uh_v__uh__1.s0.x.x" ]
  %4 = mul nuw nsw i32 %op_vmpy_v__uh_v__uh__1.s0.x.x, 32
  %5 = bitcast i32* %.phi210 to <16 x i32>*
  %6 = load <16 x i32>, <16 x i32>* %5, align 64, !tbaa !1
  %7 = add nuw nsw i32 %4, 16
  %8 = getelementptr inbounds i32, i32* %in_u32.host182, i32 %7
  %9 = bitcast i32* %8 to <16 x i32>*
  %10 = load <16 x i32>, <16 x i32>* %9, align 64, !tbaa !1
  %11 = tail call <32 x i32> @llvm.hexagon.V6.vcombine(<16 x i32> %10, <16 x i32> %6)
  %e.i = tail call <16 x i32> @llvm.hexagon.V6.lo(<32 x i32> %11) #2
  %o.i = tail call <16 x i32> @llvm.hexagon.V6.hi(<32 x i32> %11) #2
  %r.i = tail call <32 x i32> @llvm.hexagon.V6.vdealvdd(<16 x i32> %o.i, <16 x i32> %e.i, i32 -4) #2
  %12 = bitcast i16* %.phi213 to <16 x i32>*
  %13 = load <16 x i32>, <16 x i32>* %12, align 64, !tbaa !4
  %a_lo.i = tail call <16 x i32> @llvm.hexagon.V6.lo(<32 x i32> %r.i) #2
  %a_hi.i = tail call <16 x i32> @llvm.hexagon.V6.hi(<32 x i32> %r.i) #2
  %a_e.i = tail call <16 x i32> @llvm.hexagon.V6.vshufeh(<16 x i32> %a_hi.i, <16 x i32> %a_lo.i) #2
  %a_o.i = tail call <16 x i32> @llvm.hexagon.V6.vshufoh(<16 x i32> %a_hi.i, <16 x i32> %a_lo.i) #2
  %ab_e.i = tail call <32 x i32> @llvm.hexagon.V6.vmpyuhv(<16 x i32> %a_e.i, <16 x i32> %13) #2
  %ab_o.i = tail call <32 x i32> @llvm.hexagon.V6.vmpyuhv(<16 x i32> %a_o.i, <16 x i32> %13) #2
  %a_lo.i.i = tail call <16 x i32> @llvm.hexagon.V6.lo(<32 x i32> %ab_e.i) #2
  %l_lo.i.i = tail call <16 x i32> @llvm.hexagon.V6.lo(<32 x i32> %ab_o.i) #2
  %s_lo.i.i = tail call <16 x i32> @llvm.hexagon.V6.vaslw.acc(<16 x i32> %a_lo.i.i, <16 x i32> %l_lo.i.i, i32 16) #2
  %l_hi.i.i = tail call <16 x i32> @llvm.hexagon.V6.hi(<32 x i32> %ab_o.i) #2
  %s_hi.i.i = tail call <16 x i32> @llvm.hexagon.V6.vaslw.acc(<16 x i32> undef, <16 x i32> %l_hi.i.i, i32 16) #2
  %s.i.i = tail call <32 x i32> @llvm.hexagon.V6.vcombine(<16 x i32> %s_hi.i.i, <16 x i32> %s_lo.i.i) #2
  %e.i189 = tail call <16 x i32> @llvm.hexagon.V6.lo(<32 x i32> %s.i.i) #2
  %o.i190 = tail call <16 x i32> @llvm.hexagon.V6.hi(<32 x i32> %s.i.i) #2
  %r.i191 = tail call <32 x i32> @llvm.hexagon.V6.vshuffvdd(<16 x i32> %o.i190, <16 x i32> %e.i189, i32 -4) #2
  %14 = tail call <16 x i32> @llvm.hexagon.V6.lo(<32 x i32> %r.i191)
  %15 = add nuw nsw i32 %4, %0
  %16 = getelementptr inbounds i32, i32* undef, i32 %15
  %17 = bitcast i32* %16 to <16 x i32>*
  store <16 x i32> %14, <16 x i32>* %17, align 64, !tbaa !6
  %18 = tail call <16 x i32> @llvm.hexagon.V6.hi(<32 x i32> %r.i191)
  store <16 x i32> %18, <16 x i32>* undef, align 64, !tbaa !6
  %.inc211 = getelementptr i32, i32* %.phi210, i32 32
  %.inc214 = getelementptr i16, i16* %.phi213, i32 32
  %19 = bitcast i32* %.inc211 to <16 x i32>*
  %20 = load <16 x i32>, <16 x i32>* %19, align 64, !tbaa !1
  %21 = add nuw nsw i32 %4, 48
  %22 = getelementptr inbounds i32, i32* %in_u32.host182, i32 %21
  %23 = bitcast i32* %22 to <16 x i32>*
  %24 = load <16 x i32>, <16 x i32>* %23, align 64, !tbaa !1
  %25 = tail call <32 x i32> @llvm.hexagon.V6.vcombine(<16 x i32> %24, <16 x i32> %20)
  %e.i.1 = tail call <16 x i32> @llvm.hexagon.V6.lo(<32 x i32> %25) #2
  %r.i.1 = tail call <32 x i32> @llvm.hexagon.V6.vdealvdd(<16 x i32> undef, <16 x i32> %e.i.1, i32 -4) #2
  %26 = bitcast i16* %.inc214 to <16 x i32>*
  %27 = load <16 x i32>, <16 x i32>* %26, align 64, !tbaa !4
  %a_lo.i.1 = tail call <16 x i32> @llvm.hexagon.V6.lo(<32 x i32> %r.i.1) #2
  %a_e.i.1 = tail call <16 x i32> @llvm.hexagon.V6.vshufeh(<16 x i32> undef, <16 x i32> %a_lo.i.1) #2
  %a_o.i.1 = tail call <16 x i32> @llvm.hexagon.V6.vshufoh(<16 x i32> undef, <16 x i32> %a_lo.i.1) #2
  %ab_e.i.1 = tail call <32 x i32> @llvm.hexagon.V6.vmpyuhv(<16 x i32> %a_e.i.1, <16 x i32> %27) #2
  %ab_o.i.1 = tail call <32 x i32> @llvm.hexagon.V6.vmpyuhv(<16 x i32> %a_o.i.1, <16 x i32> %27) #2
  %a_lo.i.i.1 = tail call <16 x i32> @llvm.hexagon.V6.lo(<32 x i32> %ab_e.i.1) #2
  %s_lo.i.i.1 = tail call <16 x i32> @llvm.hexagon.V6.vaslw.acc(<16 x i32> %a_lo.i.i.1, <16 x i32> undef, i32 16) #2
  %a_hi.i.i.1 = tail call <16 x i32> @llvm.hexagon.V6.hi(<32 x i32> %ab_e.i.1) #2
  %l_hi.i.i.1 = tail call <16 x i32> @llvm.hexagon.V6.hi(<32 x i32> %ab_o.i.1) #2
  %s_hi.i.i.1 = tail call <16 x i32> @llvm.hexagon.V6.vaslw.acc(<16 x i32> %a_hi.i.i.1, <16 x i32> %l_hi.i.i.1, i32 16) #2
  %s.i.i.1 = tail call <32 x i32> @llvm.hexagon.V6.vcombine(<16 x i32> %s_hi.i.i.1, <16 x i32> %s_lo.i.i.1) #2
  %e.i189.1 = tail call <16 x i32> @llvm.hexagon.V6.lo(<32 x i32> %s.i.i.1) #2
  %o.i190.1 = tail call <16 x i32> @llvm.hexagon.V6.hi(<32 x i32> %s.i.i.1) #2
  %r.i191.1 = tail call <32 x i32> @llvm.hexagon.V6.vshuffvdd(<16 x i32> %o.i190.1, <16 x i32> %e.i189.1, i32 -4) #2
  %28 = tail call <16 x i32> @llvm.hexagon.V6.lo(<32 x i32> %r.i191.1)
  %29 = add nuw nsw i32 %1, %4
  %30 = getelementptr inbounds i32, i32* undef, i32 %29
  %31 = bitcast i32* %30 to <16 x i32>*
  store <16 x i32> %28, <16 x i32>* %31, align 64, !tbaa !6
  %32 = tail call <16 x i32> @llvm.hexagon.V6.hi(<32 x i32> %r.i191.1)
  %33 = add nuw nsw i32 %29, 16
  %34 = getelementptr inbounds i32, i32* undef, i32 %33
  %35 = bitcast i32* %34 to <16 x i32>*
  store <16 x i32> %32, <16 x i32>* %35, align 64, !tbaa !6
  %.inc211.1 = getelementptr i32, i32* %.phi210, i32 64
  %.inc214.1 = getelementptr i16, i16* %.phi213, i32 64
  %36 = bitcast i32* %.inc211.1 to <16 x i32>*
  %37 = load <16 x i32>, <16 x i32>* %36, align 64, !tbaa !1
  %38 = add nuw nsw i32 %4, 80
  %39 = getelementptr inbounds i32, i32* %in_u32.host182, i32 %38
  %40 = bitcast i32* %39 to <16 x i32>*
  %41 = load <16 x i32>, <16 x i32>* %40, align 64, !tbaa !1
  %42 = tail call <32 x i32> @llvm.hexagon.V6.vcombine(<16 x i32> %41, <16 x i32> %37)
  %e.i.2 = tail call <16 x i32> @llvm.hexagon.V6.lo(<32 x i32> %42) #2
  %o.i.2 = tail call <16 x i32> @llvm.hexagon.V6.hi(<32 x i32> %42) #2
  %r.i.2 = tail call <32 x i32> @llvm.hexagon.V6.vdealvdd(<16 x i32> %o.i.2, <16 x i32> %e.i.2, i32 -4) #2
  %43 = bitcast i16* %.inc214.1 to <16 x i32>*
  %44 = load <16 x i32>, <16 x i32>* %43, align 64, !tbaa !4
  %a_lo.i.2 = tail call <16 x i32> @llvm.hexagon.V6.lo(<32 x i32> %r.i.2) #2
  %a_hi.i.2 = tail call <16 x i32> @llvm.hexagon.V6.hi(<32 x i32> %r.i.2) #2
  %a_e.i.2 = tail call <16 x i32> @llvm.hexagon.V6.vshufeh(<16 x i32> %a_hi.i.2, <16 x i32> %a_lo.i.2) #2
  %a_o.i.2 = tail call <16 x i32> @llvm.hexagon.V6.vshufoh(<16 x i32> %a_hi.i.2, <16 x i32> %a_lo.i.2) #2
  %ab_e.i.2 = tail call <32 x i32> @llvm.hexagon.V6.vmpyuhv(<16 x i32> %a_e.i.2, <16 x i32> %44) #2
  %ab_o.i.2 = tail call <32 x i32> @llvm.hexagon.V6.vmpyuhv(<16 x i32> %a_o.i.2, <16 x i32> %44) #2
  %l_lo.i.i.2 = tail call <16 x i32> @llvm.hexagon.V6.lo(<32 x i32> %ab_o.i.2) #2
  %s_lo.i.i.2 = tail call <16 x i32> @llvm.hexagon.V6.vaslw.acc(<16 x i32> undef, <16 x i32> %l_lo.i.i.2, i32 16) #2
  %a_hi.i.i.2 = tail call <16 x i32> @llvm.hexagon.V6.hi(<32 x i32> %ab_e.i.2) #2
  %l_hi.i.i.2 = tail call <16 x i32> @llvm.hexagon.V6.hi(<32 x i32> %ab_o.i.2) #2
  %s_hi.i.i.2 = tail call <16 x i32> @llvm.hexagon.V6.vaslw.acc(<16 x i32> %a_hi.i.i.2, <16 x i32> %l_hi.i.i.2, i32 16) #2
  %s.i.i.2 = tail call <32 x i32> @llvm.hexagon.V6.vcombine(<16 x i32> %s_hi.i.i.2, <16 x i32> %s_lo.i.i.2) #2
  %e.i189.2 = tail call <16 x i32> @llvm.hexagon.V6.lo(<32 x i32> %s.i.i.2) #2
  %o.i190.2 = tail call <16 x i32> @llvm.hexagon.V6.hi(<32 x i32> %s.i.i.2) #2
  %r.i191.2 = tail call <32 x i32> @llvm.hexagon.V6.vshuffvdd(<16 x i32> %o.i190.2, <16 x i32> %e.i189.2, i32 -4) #2
  %45 = tail call <16 x i32> @llvm.hexagon.V6.lo(<32 x i32> %r.i191.2)
  %46 = add nuw nsw i32 %2, %4
  %47 = getelementptr inbounds i32, i32* undef, i32 %46
  %48 = bitcast i32* %47 to <16 x i32>*
  store <16 x i32> %45, <16 x i32>* %48, align 64, !tbaa !6
  %49 = tail call <16 x i32> @llvm.hexagon.V6.hi(<32 x i32> %r.i191.2)
  %50 = add nuw nsw i32 %46, 16
  %51 = getelementptr inbounds i32, i32* undef, i32 %50
  %52 = bitcast i32* %51 to <16 x i32>*
  store <16 x i32> %49, <16 x i32>* %52, align 64, !tbaa !6
  %e.i189.3 = tail call <16 x i32> @llvm.hexagon.V6.lo(<32 x i32> undef) #2
  %r.i191.3 = tail call <32 x i32> @llvm.hexagon.V6.vshuffvdd(<16 x i32> undef, <16 x i32> %e.i189.3, i32 -4) #2
  %53 = tail call <16 x i32> @llvm.hexagon.V6.lo(<32 x i32> %r.i191.3)
  %54 = add nuw nsw i32 %3, %4
  %55 = getelementptr inbounds i32, i32* undef, i32 %54
  %56 = bitcast i32* %55 to <16 x i32>*
  store <16 x i32> %53, <16 x i32>* %56, align 64, !tbaa !6
  %57 = tail call <16 x i32> @llvm.hexagon.V6.hi(<32 x i32> %r.i191.3)
  %58 = add nuw nsw i32 %54, 16
  %59 = getelementptr inbounds i32, i32* undef, i32 %58
  %60 = bitcast i32* %59 to <16 x i32>*
  store <16 x i32> %57, <16 x i32>* %60, align 64, !tbaa !6
  %61 = add nuw nsw i32 %op_vmpy_v__uh_v__uh__1.s0.x.x, 4
  %62 = icmp eq i32 %61, 24
  %.inc211.3 = getelementptr i32, i32* %.phi210, i32 128
  %.inc214.3 = getelementptr i16, i16* %.phi213, i32 128
  br i1 %62, label %"end for op_vmpy_v__uh_v__uh__1.s0.x.x", label %"for op_vmpy_v__uh_v__uh__1.s0.x.x"

"end for op_vmpy_v__uh_v__uh__1.s0.x.x":          ; preds = %"for op_vmpy_v__uh_v__uh__1.s0.x.x"
  %63 = add nuw nsw i32 %op_vmpy_v__uh_v__uh__1.s0.y, 1
  br label %"for op_vmpy_v__uh_v__uh__1.s0.y"
}

attributes #0 = { nounwind readnone }
attributes #1 = { "target-cpu"="hexagonv60" "target-features"="+hvxv60,+hvx-length64b" }
attributes #2 = { nounwind }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"halide_mattrs", !"+hvx"}
!1 = !{!2, !2, i64 0}
!2 = !{!"in_u32", !3}
!3 = !{!"Halide buffer"}
!4 = !{!5, !5, i64 0}
!5 = !{!"in_u16", !3}
!6 = !{!7, !7, i64 0}
!7 = !{!"op_vmpy_v__uh_v__uh__1", !3}
