; RUN: llc -mtriple=hexagon -O2 -mcpu=hexagonv60 -mattr=+hvxv60,hvx-length64b < %s | FileCheck %s

; CHECK: vasr(v{{[0-9]+}}.h,v{{[0-9]+}}.h,r{{[0-7]+}}):sat

target datalayout = "e-p:32:32:32-i64:64:64-i32:32:32-i16:16:16-i1:32:32-f64:64:64-f32:32:32-v64:64:64-v32:32:32-a:0-n16:32"
target triple = "hexagon-unknown--elf"

%struct.buffer_t = type { i64, ptr, [4 x i32], [4 x i32], [4 x i32], i32, i8, i8, [6 x i8] }

; Function Attrs: norecurse nounwind
define i32 @__test_vasr(ptr noalias nocapture %f.buffer, ptr noalias nocapture %g.buffer, ptr noalias nocapture %res.buffer) #0 {
entry:
  %buf_host = getelementptr inbounds %struct.buffer_t, ptr %f.buffer, i32 0, i32 1
  %f.host = load ptr, ptr %buf_host, align 4
  %buf_dev = getelementptr inbounds %struct.buffer_t, ptr %f.buffer, i32 0, i32 0
  %f.dev = load i64, ptr %buf_dev, align 8
  %0 = icmp eq ptr %f.host, null
  %1 = icmp eq i64 %f.dev, 0
  %f.host_and_dev_are_null = and i1 %0, %1
  %buf_min = getelementptr inbounds %struct.buffer_t, ptr %f.buffer, i32 0, i32 4, i32 0
  %f.min.0 = load i32, ptr %buf_min, align 4
  %buf_host10 = getelementptr inbounds %struct.buffer_t, ptr %g.buffer, i32 0, i32 1
  %g.host = load ptr, ptr %buf_host10, align 4
  %buf_dev11 = getelementptr inbounds %struct.buffer_t, ptr %g.buffer, i32 0, i32 0
  %g.dev = load i64, ptr %buf_dev11, align 8
  %2 = icmp eq ptr %g.host, null
  %3 = icmp eq i64 %g.dev, 0
  %g.host_and_dev_are_null = and i1 %2, %3
  %buf_min22 = getelementptr inbounds %struct.buffer_t, ptr %g.buffer, i32 0, i32 4, i32 0
  %g.min.0 = load i32, ptr %buf_min22, align 4
  %buf_host27 = getelementptr inbounds %struct.buffer_t, ptr %res.buffer, i32 0, i32 1
  %res.host = load ptr, ptr %buf_host27, align 4
  %buf_dev28 = getelementptr inbounds %struct.buffer_t, ptr %res.buffer, i32 0, i32 0
  %res.dev = load i64, ptr %buf_dev28, align 8
  %4 = icmp eq ptr %res.host, null
  %5 = icmp eq i64 %res.dev, 0
  %res.host_and_dev_are_null = and i1 %4, %5
  %buf_extent31 = getelementptr inbounds %struct.buffer_t, ptr %res.buffer, i32 0, i32 2, i32 0
  %res.extent.0 = load i32, ptr %buf_extent31, align 4
  %buf_min39 = getelementptr inbounds %struct.buffer_t, ptr %res.buffer, i32 0, i32 4, i32 0
  %res.min.0 = load i32, ptr %buf_min39, align 4
  %6 = add nsw i32 %res.extent.0, -1
  %7 = and i32 %6, -64
  %8 = add i32 %res.min.0, 63
  %9 = add i32 %8, %7
  %10 = add nsw i32 %res.min.0, %res.extent.0
  %11 = add nsw i32 %10, -1
  %12 = icmp slt i32 %9, %11
  %13 = select i1 %12, i32 %9, i32 %11
  %14 = add nsw i32 %10, -64
  %15 = icmp slt i32 %res.min.0, %14
  %16 = select i1 %15, i32 %res.min.0, i32 %14
  %f.extent.0.required.s = sub nsw i32 %13, %16
  br i1 %f.host_and_dev_are_null, label %true_bb, label %after_bb

true_bb:                                          ; preds = %entry
  %buf_elem_size44 = getelementptr inbounds %struct.buffer_t, ptr %f.buffer, i32 0, i32 5
  store i32 1, ptr %buf_elem_size44, align 4
  store i32 %16, ptr %buf_min, align 4
  %17 = add nsw i32 %f.extent.0.required.s, 1
  %buf_extent46 = getelementptr inbounds %struct.buffer_t, ptr %f.buffer, i32 0, i32 2, i32 0
  store i32 %17, ptr %buf_extent46, align 4
  %buf_stride47 = getelementptr inbounds %struct.buffer_t, ptr %f.buffer, i32 0, i32 3, i32 0
  store i32 1, ptr %buf_stride47, align 4
  %buf_min48 = getelementptr inbounds %struct.buffer_t, ptr %f.buffer, i32 0, i32 4, i32 1
  store i32 0, ptr %buf_min48, align 4
  %buf_extent49 = getelementptr inbounds %struct.buffer_t, ptr %f.buffer, i32 0, i32 2, i32 1
  store i32 0, ptr %buf_extent49, align 4
  %buf_stride50 = getelementptr inbounds %struct.buffer_t, ptr %f.buffer, i32 0, i32 3, i32 1
  store i32 0, ptr %buf_stride50, align 4
  %buf_min51 = getelementptr inbounds %struct.buffer_t, ptr %f.buffer, i32 0, i32 4, i32 2
  store i32 0, ptr %buf_min51, align 4
  %buf_extent52 = getelementptr inbounds %struct.buffer_t, ptr %f.buffer, i32 0, i32 2, i32 2
  store i32 0, ptr %buf_extent52, align 4
  %buf_stride53 = getelementptr inbounds %struct.buffer_t, ptr %f.buffer, i32 0, i32 3, i32 2
  store i32 0, ptr %buf_stride53, align 4
  %buf_min54 = getelementptr inbounds %struct.buffer_t, ptr %f.buffer, i32 0, i32 4, i32 3
  store i32 0, ptr %buf_min54, align 4
  %buf_extent55 = getelementptr inbounds %struct.buffer_t, ptr %f.buffer, i32 0, i32 2, i32 3
  store i32 0, ptr %buf_extent55, align 4
  %buf_stride56 = getelementptr inbounds %struct.buffer_t, ptr %f.buffer, i32 0, i32 3, i32 3
  store i32 0, ptr %buf_stride56, align 4
  br label %after_bb

after_bb:                                         ; preds = %true_bb, %entry
  br i1 %g.host_and_dev_are_null, label %true_bb57, label %after_bb59

true_bb57:                                        ; preds = %after_bb
  %buf_elem_size60 = getelementptr inbounds %struct.buffer_t, ptr %g.buffer, i32 0, i32 5
  store i32 1, ptr %buf_elem_size60, align 4
  store i32 %16, ptr %buf_min22, align 4
  %18 = add nsw i32 %f.extent.0.required.s, 1
  %buf_extent62 = getelementptr inbounds %struct.buffer_t, ptr %g.buffer, i32 0, i32 2, i32 0
  store i32 %18, ptr %buf_extent62, align 4
  %buf_stride63 = getelementptr inbounds %struct.buffer_t, ptr %g.buffer, i32 0, i32 3, i32 0
  store i32 1, ptr %buf_stride63, align 4
  %buf_min64 = getelementptr inbounds %struct.buffer_t, ptr %g.buffer, i32 0, i32 4, i32 1
  store i32 0, ptr %buf_min64, align 4
  %buf_extent65 = getelementptr inbounds %struct.buffer_t, ptr %g.buffer, i32 0, i32 2, i32 1
  store i32 0, ptr %buf_extent65, align 4
  %buf_stride66 = getelementptr inbounds %struct.buffer_t, ptr %g.buffer, i32 0, i32 3, i32 1
  store i32 0, ptr %buf_stride66, align 4
  %buf_min67 = getelementptr inbounds %struct.buffer_t, ptr %g.buffer, i32 0, i32 4, i32 2
  store i32 0, ptr %buf_min67, align 4
  %buf_extent68 = getelementptr inbounds %struct.buffer_t, ptr %g.buffer, i32 0, i32 2, i32 2
  store i32 0, ptr %buf_extent68, align 4
  %buf_stride69 = getelementptr inbounds %struct.buffer_t, ptr %g.buffer, i32 0, i32 3, i32 2
  store i32 0, ptr %buf_stride69, align 4
  %buf_min70 = getelementptr inbounds %struct.buffer_t, ptr %g.buffer, i32 0, i32 4, i32 3
  store i32 0, ptr %buf_min70, align 4
  %buf_extent71 = getelementptr inbounds %struct.buffer_t, ptr %g.buffer, i32 0, i32 2, i32 3
  store i32 0, ptr %buf_extent71, align 4
  %buf_stride72 = getelementptr inbounds %struct.buffer_t, ptr %g.buffer, i32 0, i32 3, i32 3
  store i32 0, ptr %buf_stride72, align 4
  br label %after_bb59

after_bb59:                                       ; preds = %true_bb57, %after_bb
  br i1 %res.host_and_dev_are_null, label %after_bb75.thread, label %after_bb75

after_bb75.thread:                                ; preds = %after_bb59
  %buf_elem_size76 = getelementptr inbounds %struct.buffer_t, ptr %res.buffer, i32 0, i32 5
  store i32 1, ptr %buf_elem_size76, align 4
  store i32 %16, ptr %buf_min39, align 4
  %19 = add nsw i32 %f.extent.0.required.s, 1
  store i32 %19, ptr %buf_extent31, align 4
  %buf_stride79 = getelementptr inbounds %struct.buffer_t, ptr %res.buffer, i32 0, i32 3, i32 0
  store i32 1, ptr %buf_stride79, align 4
  %buf_min80 = getelementptr inbounds %struct.buffer_t, ptr %res.buffer, i32 0, i32 4, i32 1
  store i32 0, ptr %buf_min80, align 4
  %buf_extent81 = getelementptr inbounds %struct.buffer_t, ptr %res.buffer, i32 0, i32 2, i32 1
  store i32 0, ptr %buf_extent81, align 4
  %buf_stride82 = getelementptr inbounds %struct.buffer_t, ptr %res.buffer, i32 0, i32 3, i32 1
  store i32 0, ptr %buf_stride82, align 4
  %buf_min83 = getelementptr inbounds %struct.buffer_t, ptr %res.buffer, i32 0, i32 4, i32 2
  store i32 0, ptr %buf_min83, align 4
  %buf_extent84 = getelementptr inbounds %struct.buffer_t, ptr %res.buffer, i32 0, i32 2, i32 2
  store i32 0, ptr %buf_extent84, align 4
  %buf_stride85 = getelementptr inbounds %struct.buffer_t, ptr %res.buffer, i32 0, i32 3, i32 2
  store i32 0, ptr %buf_stride85, align 4
  %buf_min86 = getelementptr inbounds %struct.buffer_t, ptr %res.buffer, i32 0, i32 4, i32 3
  store i32 0, ptr %buf_min86, align 4
  %buf_extent87 = getelementptr inbounds %struct.buffer_t, ptr %res.buffer, i32 0, i32 2, i32 3
  store i32 0, ptr %buf_extent87, align 4
  %buf_stride88 = getelementptr inbounds %struct.buffer_t, ptr %res.buffer, i32 0, i32 3, i32 3
  store i32 0, ptr %buf_stride88, align 4
  br label %destructor_block

after_bb75:                                       ; preds = %after_bb59
  %20 = or i1 %f.host_and_dev_are_null, %g.host_and_dev_are_null
  br i1 %20, label %destructor_block, label %"produce res"

"produce res":                                    ; preds = %after_bb75
  %21 = ashr i32 %res.extent.0, 6
  %22 = icmp sgt i32 %21, 0
  br i1 %22, label %"for res.s0.x.x", label %"end for res.s0.x.x", !prof !4

"for res.s0.x.x":                                 ; preds = %"for res.s0.x.x", %"produce res"
  %res.s0.x.x = phi i32 [ %41, %"for res.s0.x.x" ], [ 0, %"produce res" ]
  %23 = shl nsw i32 %res.s0.x.x, 6
  %24 = add nsw i32 %23, %res.min.0
  %25 = sub nsw i32 %24, %f.min.0
  %26 = getelementptr inbounds i8, ptr %f.host, i32 %25
  %27 = bitcast ptr %26 to ptr
  %28 = load <16 x i32>, ptr %27, align 1, !tbaa !5
  %29 = tail call <32 x i32> @llvm.hexagon.V6.vzb(<16 x i32> %28)
  %30 = sub nsw i32 %24, %g.min.0
  %31 = getelementptr inbounds i8, ptr %g.host, i32 %30
  %32 = bitcast ptr %31 to ptr
  %33 = load <16 x i32>, ptr %32, align 1, !tbaa !8
  %34 = tail call <32 x i32> @llvm.hexagon.V6.vzb(<16 x i32> %33)
  %35 = tail call <32 x i32> @llvm.hexagon.V6.vaddh.dv(<32 x i32> %29, <32 x i32> %34)
  %36 = tail call <16 x i32> @llvm.hexagon.V6.hi(<32 x i32> %35)
  %37 = tail call <16 x i32> @llvm.hexagon.V6.lo(<32 x i32> %35)
  %38 = tail call <16 x i32> @llvm.hexagon.V6.vasrhubsat(<16 x i32> %36, <16 x i32> %37, i32 4)
  %39 = getelementptr inbounds i8, ptr %res.host, i32 %23
  %40 = bitcast ptr %39 to ptr
  store <16 x i32> %38, ptr %40, align 1, !tbaa !10
  %41 = add nuw nsw i32 %res.s0.x.x, 1
  %42 = icmp eq i32 %41, %21
  br i1 %42, label %"end for res.s0.x.x", label %"for res.s0.x.x"

"end for res.s0.x.x":                             ; preds = %"for res.s0.x.x", %"produce res"
  %43 = add nsw i32 %res.extent.0, 63
  %44 = ashr i32 %43, 6
  %45 = icmp sgt i32 %44, %21
  br i1 %45, label %"for res.s0.x.x92.preheader", label %destructor_block, !prof !4

"for res.s0.x.x92.preheader":                     ; preds = %"end for res.s0.x.x"
  %46 = sub i32 -64, %f.min.0
  %47 = add i32 %46, %10
  %48 = getelementptr inbounds i8, ptr %f.host, i32 %47
  %49 = bitcast ptr %48 to ptr
  %50 = load <16 x i32>, ptr %49, align 1
  %51 = tail call <32 x i32> @llvm.hexagon.V6.vzb(<16 x i32> %50)
  %52 = sub i32 -64, %g.min.0
  %53 = add i32 %52, %10
  %54 = getelementptr inbounds i8, ptr %g.host, i32 %53
  %55 = bitcast ptr %54 to ptr
  %56 = load <16 x i32>, ptr %55, align 1
  %57 = tail call <32 x i32> @llvm.hexagon.V6.vzb(<16 x i32> %56)
  %58 = tail call <32 x i32> @llvm.hexagon.V6.vaddh.dv(<32 x i32> %51, <32 x i32> %57)
  %59 = tail call <16 x i32> @llvm.hexagon.V6.lo(<32 x i32> %58)
  %60 = add nsw i32 %res.extent.0, -64
  %61 = getelementptr inbounds i8, ptr %res.host, i32 %60
  %62 = tail call <16 x i32> @llvm.hexagon.V6.hi(<32 x i32> %58)
  %63 = tail call <16 x i32> @llvm.hexagon.V6.vasrhubsat(<16 x i32> %62, <16 x i32> %59, i32 4)
  %64 = bitcast ptr %61 to ptr
  store <16 x i32> %63, ptr %64, align 1, !tbaa !10
  br label %destructor_block

destructor_block:                                 ; preds = %"for res.s0.x.x92.preheader", %"end for res.s0.x.x", %after_bb75, %after_bb75.thread
  ret i32 0
}

; Function Attrs: nounwind readnone
declare <32 x i32> @llvm.hexagon.V6.vaddh.dv(<32 x i32>, <32 x i32>) #1

; Function Attrs: nounwind readnone
declare <32 x i32> @llvm.hexagon.V6.vzb(<16 x i32>) #1

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.hi(<32 x i32>) #1

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.lo(<32 x i32>) #1

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.vasrhubsat(<16 x i32>, <16 x i32>, i32) #1

attributes #0 = { norecurse nounwind }
attributes #1 = { nounwind readnone }

!llvm.ident = !{!0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0}
!llvm.module.flags = !{!1, !2, !3}

!0 = !{!"Clang $LLVM_VERSION_MAJOR.$LLVM_VERSION_MINOR (based on LLVM 3.8.0)"}
!1 = !{i32 2, !"halide_use_soft_float_abi", i32 0}
!2 = !{i32 2, !"halide_mcpu", !"hexagonv60"}
!3 = !{i32 2, !"halide_mattrs", !"+hvx"}
!4 = !{!"branch_weights", i32 1073741824, i32 0}
!5 = !{!6, !6, i64 0}
!6 = !{!"f", !7}
!7 = !{!"Halide buffer"}
!8 = !{!9, !9, i64 0}
!9 = !{!"g", !7}
!10 = !{!11, !11, i64 0}
!11 = !{!"res", !7}
