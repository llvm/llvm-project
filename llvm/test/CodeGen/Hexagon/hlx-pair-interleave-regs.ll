; The HLX pair is supposed to be passed in the order of L(odd):L(even),
; where the odd register is paired with the immediately preceding even register.
; For instance sequence of registers L1:L0 is correct but not L1:L2.
; While passing a mix of single HLX and paired HLX objects to a function, the
; registers that might have been skipped by paired object will be reused by single
; HLX type.

; case1: (v1024i16, v1024i32, v1024i32, v1024i16) -> v1024i32
;   register order: $l0, $ll1, $ll2, $l1  (single, pair, pair, single)
;
; case2: (v1024i32, v1024i16, v1024i32, v1024i16) -> v1024i32
;   register order: $ll0, $l2, $ll2, $l3  (pair, single, pair, single)
; Note: ll0 -> l1:l0, ll1 -> l3:l2 ...

; RUN: llc -march=hexagon -mcpu=hexagonv85 -mattr=+hlx -start-before=hexagon-isel -stop-after=hexagon-isel -verify-machineinstrs -o - %s | FileCheck %s --check-prefix=CHECK-ISEL

; CHECK-ISEL-LABEL: name: test_interleave_case1
; CHECK-ISEL:       liveins: $l0, $ll1, $ll2, $l1
; CHECK-ISEL-DAG:   [[L0:%[0-9]+]]:hlxlr  = COPY $l0
; CHECK-ISEL-DAG:   [[LL1:%[0-9]+]]:hlxllr = COPY $ll1
; CHECK-ISEL-DAG:   [[LL2:%[0-9]+]]:hlxllr = COPY $ll2
; CHECK-ISEL-DAG:   [[L1:%[0-9]+]]:hlxlr  = COPY $l1
; CHECK-ISEL:       {{%[0-9]+}}:hlxllr = LX_add_wwuh [[LL1]], [[L0]]
; CHECK-ISEL:       {{%[0-9]+}}:hlxllr = LX_add_wwuh [[LL2]], [[L1]]

; CHECK-ISEL-LABEL: name: func1
; CHECK-ISEL:       [[S0:%[0-9]+]]:hlxlr  = LX_assign_u 0
; CHECK-ISEL:       [[S1:%[0-9]+]]:hlxlr  = LX_assign_u 1
; CHECK-ISEL:       [[S2:%[0-9]+]]:hlxlr  = LX_assign_u 2
; CHECK-ISEL:       [[S3:%[0-9]+]]:hlxlr  = LX_assign_u 3
; CHECK-ISEL:       [[PAIR:%[0-9]+]]:hlxllr = LX_combine_w [[S0]], [[S1]]
; CHECK-ISEL:       $l0  = COPY [[S2]]
; CHECK-ISEL:       $ll1 = COPY [[PAIR]]
; CHECK-ISEL:       $ll2 = COPY [[PAIR]]
; CHECK-ISEL:       $l1  = COPY [[S3]]
; CHECK-ISEL:       J2_call @test_interleave_case1, hexagoncsr_hlx
; CHECK-ISEL-SAME:  implicit $l0, implicit {{.*}}$ll1, implicit {{.*}}$ll2, implicit {{.*}}$l1

; CHECK-ISEL-LABEL: name: test_interleave_case2
; CHECK-ISEL:       liveins: $ll0, $l2, $ll2, $l3
; CHECK-ISEL-DAG:   [[LL0:%[0-9]+]]:hlxllr = COPY $ll0
; CHECK-ISEL-DAG:   [[L2:%[0-9]+]]:hlxlr  = COPY $l2
; CHECK-ISEL-DAG:   [[LL2:%[0-9]+]]:hlxllr = COPY $ll2
; CHECK-ISEL-DAG:   [[L3:%[0-9]+]]:hlxlr  = COPY $l3
; CHECK-ISEL:       {{%[0-9]+}}:hlxllr = LX_add_wwuh [[LL0]], [[L2]]
; CHECK-ISEL:       {{%[0-9]+}}:hlxllr = LX_add_wwuh [[LL2]], [[L3]]

; function signature: ((0,1), 2, (0,1), 3)
; CHECK-ISEL-LABEL: name: func2
; CHECK-ISEL:       [[S0:%[0-9]+]]:hlxlr  = LX_assign_u 0
; CHECK-ISEL:       [[S1:%[0-9]+]]:hlxlr  = LX_assign_u 1
; CHECK-ISEL:       [[S2:%[0-9]+]]:hlxlr  = LX_assign_u 2
; CHECK-ISEL:       [[S3:%[0-9]+]]:hlxlr  = LX_assign_u 3
; CHECK-ISEL:       [[PAIR:%[0-9]+]]:hlxllr = LX_combine_w [[S0]], [[S1]]
; CHECK-ISEL:       $ll0 = COPY [[PAIR]]
; CHECK-ISEL:       $l2  = COPY [[S2]]
; CHECK-ISEL:       $ll2 = COPY [[PAIR]]
; CHECK-ISEL:       $l3  = COPY [[S3]]
; CHECK-ISEL:       J2_call @test_interleave_case2, hexagoncsr_hlx
; CHECK-ISEL-SAME:  implicit $ll0, implicit {{.*}}$l2, implicit {{.*}}$ll2, implicit {{.*}}$l3

target datalayout = "e-m:e-p:32:32:32-a:0-n16:32-i64:64:64-i32:32:32-i16:16:16-i1:8:8-f32:32:32-f64:64:64-v32:32:32-v64:64:64-v512:512:512-v1024:1024:1024-v2048:2048:2048-v16384:16384:16384-v32768:16384:16384"
target triple = "hexagon"

@vtcm_base = dso_local local_unnamed_addr global ptr null, align 4

define dso_local inreg <1024 x i32> @test_interleave_case1(<1024 x i16> noundef %a, <1024 x i32> noundef %b, <1024 x i32> noundef %c, <1024 x i16> noundef %d) local_unnamed_addr #0 {
entry:
  %0 = tail call <1024 x i32> @llvm.hexagon.LX.add.wwuh(<1024 x i32> %b, <1024 x i16> %a)
  %1 = tail call <1024 x i32> @llvm.hexagon.LX.add.wwuh(<1024 x i32> %c, <1024 x i16> %d)
  %2 = tail call <1024 x i32> @llvm.hexagon.LX.add.w(<1024 x i32> %0, <1024 x i32> %1)
  ret <1024 x i32> %2
}

declare <1024 x i32> @llvm.hexagon.LX.add.w(<1024 x i32>, <1024 x i32>) #1

define dso_local noundef i32 @func1() local_unnamed_addr #2 {
entry:
  %0 = tail call <1024 x i16> @llvm.hexagon.LX.assign.u(i32 0)
  %1 = tail call <1024 x i16> @llvm.hexagon.LX.assign.u(i32 1)
  %2 = tail call <1024 x i16> @llvm.hexagon.LX.assign.u(i32 2)
  %3 = tail call <1024 x i16> @llvm.hexagon.LX.assign.u(i32 3)
  %4 = tail call <1024 x i32> @llvm.hexagon.LX.combine.w(<1024 x i16> %0, <1024 x i16> %1)
  %call = tail call inreg <1024 x i32> @test_interleave_case1(<1024 x i16> noundef %2, <1024 x i32> noundef %4, <1024 x i32> noundef %4, <1024 x i16> noundef %3)
  %5 = tail call <1024 x i16> @llvm.hexagon.LX.hi(<1024 x i32> %call)
  %6 = load ptr, ptr @vtcm_base, align 4, !tbaa !6
  store <1024 x i16> %5, ptr %6, align 2048, !tbaa !9
  ret i32 0
}

define dso_local inreg <1024 x i32> @test_interleave_case2(<1024 x i32> noundef %a, <1024 x i16> noundef %b, <1024 x i32> noundef %c, <1024 x i16> noundef %d) local_unnamed_addr #0 {
entry:
  %0 = tail call <1024 x i32> @llvm.hexagon.LX.add.wwuh(<1024 x i32> %a, <1024 x i16> %b)
  %1 = tail call <1024 x i32> @llvm.hexagon.LX.add.wwuh(<1024 x i32> %c, <1024 x i16> %d)
  %2 = tail call <1024 x i32> @llvm.hexagon.LX.add.w(<1024 x i32> %0, <1024 x i32> %1)
  ret <1024 x i32> %2
}

define dso_local noundef i32 @func2() local_unnamed_addr #2 {
entry:
  %0 = tail call <1024 x i16> @llvm.hexagon.LX.assign.u(i32 0)
  %1 = tail call <1024 x i16> @llvm.hexagon.LX.assign.u(i32 1)
  %2 = tail call <1024 x i16> @llvm.hexagon.LX.assign.u(i32 2)
  %3 = tail call <1024 x i16> @llvm.hexagon.LX.assign.u(i32 3)
  %4 = tail call <1024 x i32> @llvm.hexagon.LX.combine.w(<1024 x i16> %0, <1024 x i16> %1)
  %call = tail call inreg <1024 x i32> @test_interleave_case2(<1024 x i32> noundef %4, <1024 x i16> noundef %2, <1024 x i32> noundef %4, <1024 x i16> noundef %3)
  %5 = tail call <1024 x i16> @llvm.hexagon.LX.hi(<1024 x i32> %call)
  %6 = load ptr, ptr @vtcm_base, align 4, !tbaa !6
  store <1024 x i16> %5, ptr %6, align 2048, !tbaa !9
  ret i32 0
}

attributes #0 = { mustprogress nofree noinline norecurse nosync nounwind willreturn memory(none) "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="hexagonv85" "target-features"="+hlx,+hlx-length2048b,+hvx-length128b,+hvxv85,+v85,-long-calls" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind willreturn memory(none) }
attributes #2 = { nofree noinline nounwind "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="hexagonv85" "target-features"="+hlx,+hlx-length2048b,+hvx-length128b,+hvxv85,+v85,-long-calls" }

!llvm.module.flags = !{!0}

!0 = !{i32 7, !"frame-pointer", i32 2}
!4 = !{!"omnipotent char", !5, i64 0}
!5 = !{!"Simple C/C++ TBAA"}
!6 = !{!7, !7, i64 0}
!7 = !{!"p1 omnipotent char", !8, i64 0}
!8 = !{!"any pointer", !4, i64 0}
!9 = !{!4, !4, i64 0}
