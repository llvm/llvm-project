; Tests if the sf/hf = qf converts have been done correctly

; REQUIRES: asserts
; RUN: llc -mtriple=hexagon-unknown-elf -mhvx -mcpu=hexagonv79 -mattr=+hvxv79,+hvx-length128b,+hvx-qfloat \
; RUN: -enable-rem-conv=true -enable-xqf-gen=true -hexagon-qfloat-mode=ieee -verify-machineinstrs \
; RUN: -debug-print < %s 2>&1 -o /dev/null | FileCheck %s
; RUN: llc -mtriple=hexagon-unknown-elf -mhvx -mcpu=hexagonv79 -mattr=+hvxv79,+hvx-length128b,+hvx-qfloat \
; RUN: -enable-rem-conv=true -enable-xqf-gen=true -hexagon-qfloat-mode=lossy -verify-machineinstrs \
; RUN: -debug-print < %s 2>&1 -o /dev/null | FileCheck %s

; Single use of convert reg. The convert should be deleted.
define dso_local <32 x i32> @conv1_qf32(<32 x i32> noundef %input1, <32 x i32> noundef %input2) local_unnamed_addr #0 {
; CHECK: bb.0.entry
; CHECK: [[VREG2:%[0-9]+]]:hvxvr = V6_vadd_sf [[VREG0:%[0-9]+]]:hvxvr, %1:hvxvr
; CHECK-NOT: V6_vconv_sf_qf32 killed [[VREG2]]:hvxvr
; CHECK: V6_vadd_qf32_mix [[VREG2]]:hvxvr, [[VREG0]]:hvxvr
entry:
  %0 = tail call <32 x i32> @llvm.hexagon.V6.vadd.sf.128B(<32 x i32> %input1, <32 x i32> %input2)
  %1 = tail call <32 x i32> @llvm.hexagon.V6.vconv.sf.qf32.128B(<32 x i32> %0)
  %2 = tail call <32 x i32> @llvm.hexagon.V6.vadd.sf.128B(<32 x i32> %input1, <32 x i32> %1)
  ret <32 x i32> %2
}

; Double use of convert reg. The convert should not be deleted.
define dso_local <32 x i32> @conv2_qf32(<32 x i32> noundef %input1, <32 x i32> noundef %input2) local_unnamed_addr #0 {
; CHECK: bb.0.entry
; CHECK: [[VREG2:%[0-9]+]]:hvxvr = V6_vadd_sf [[VREG0:%[0-9]+]]:hvxvr, [[VREG1:%[0-9]+]]:hvxvr
; CHECK-NEXT: V6_vconv_sf_qf32 [[VREG2]]:hvxvr
; CHECK-NEXT: V6_vadd_qf32_mix [[VREG2]]:hvxvr, [[VREG0]]:hvxvr
; CHECK-NEXT: V6_vadd_qf32_mix [[VREG2]]:hvxvr, [[VREG1]]:hvxvr
entry:
  %0 = tail call <32 x i32> @llvm.hexagon.V6.vadd.sf.128B(<32 x i32> %input1, <32 x i32> %input2)
  %1 = tail call <32 x i32> @llvm.hexagon.V6.vconv.sf.qf32.128B(<32 x i32> %0)
  %2 = tail call <32 x i32> @llvm.hexagon.V6.vadd.sf.128B(<32 x i32> %input1, <32 x i32> %1)
  %3 = tail call <32 x i32> @llvm.hexagon.V6.vadd.sf.128B(<32 x i32> %input2, <32 x i32> %1)
  %4 = tail call <32 x i32> @llvm.hexagon.V6.vadd.qf32.128B(<32 x i32> %2, <32 x i32> %3)
  ret <32 x i32> %4
}

; Single use of convert reg. The convert should be deleted.
define dso_local <32 x i32> @conv1_qf16(<32 x i32> noundef %input1, <32 x i32> noundef %input2) local_unnamed_addr #0 {
; CHECK: bb.0.entry
; CHECK: [[VREG2:%[0-9]+]]:hvxvr = V6_vadd_hf [[VREG0:%[0-9]+]]:hvxvr, %1:hvxvr
; CHECK-NOT: V6_vconv_hf_qf16 killed [[VREG2]]:hvxvr
; CHECK: V6_vadd_qf16_mix [[VREG2]]:hvxvr, [[VREG0]]:hvxvr
entry:
  %0 = tail call <32 x i32> @llvm.hexagon.V6.vadd.hf.128B(<32 x i32> %input1, <32 x i32> %input2)
  %1 = tail call <32 x i32> @llvm.hexagon.V6.vconv.hf.qf16.128B(<32 x i32> %0)
  %2 = tail call <32 x i32> @llvm.hexagon.V6.vadd.hf.128B(<32 x i32> %input1, <32 x i32> %1)
  ret <32 x i32> %2
}

; Double use of convert reg. The convert should not be deleted.
define dso_local <32 x i32> @conv2_qf16(<32 x i32> noundef %input1, <32 x i32> noundef %input2) local_unnamed_addr #0 {
; CHECK: bb.0.entry
; CHECK: [[VREG2:%[0-9]+]]:hvxvr = V6_vadd_hf [[VREG0:%[0-9]+]]:hvxvr, [[VREG1:%[0-9]+]]:hvxvr
; CHECK-NEXT: V6_vconv_hf_qf16 [[VREG2]]:hvxvr
; CHECK-NEXT: V6_vadd_qf16_mix [[VREG2]]:hvxvr, [[VREG0]]:hvxvr
; CHECK-NEXT: V6_vadd_qf16_mix [[VREG2]]:hvxvr, [[VREG1]]:hvxvr
entry:
  %0 = tail call <32 x i32> @llvm.hexagon.V6.vadd.hf.128B(<32 x i32> %input1, <32 x i32> %input2)
  %1 = tail call <32 x i32> @llvm.hexagon.V6.vconv.hf.qf16.128B(<32 x i32> %0)
  %2 = tail call <32 x i32> @llvm.hexagon.V6.vadd.hf.128B(<32 x i32> %input1, <32 x i32> %1)
  %3 = tail call <32 x i32> @llvm.hexagon.V6.vadd.hf.128B(<32 x i32> %input2, <32 x i32> %1)
  %4 = tail call <32 x i32> @llvm.hexagon.V6.vadd.qf16.128B(<32 x i32> %2, <32 x i32> %3)
  ret <32 x i32> %4
}

declare <32 x i32> @llvm.hexagon.V6.vadd.sf.128B(<32 x i32>, <32 x i32>) #1
declare <32 x i32> @llvm.hexagon.V6.vadd.hf.128B(<32 x i32>, <32 x i32>) #1
declare <32 x i32> @llvm.hexagon.V6.vconv.sf.qf32.128B(<32 x i32>) #1
declare <32 x i32> @llvm.hexagon.V6.vconv.hf.qf16.128B(<32 x i32>) #1
declare <32 x i32> @llvm.hexagon.V6.vadd.qf32.128B(<32 x i32>, <32 x i32>) #1
declare <32 x i32> @llvm.hexagon.V6.vadd.qf16.128B(<32 x i32>, <32 x i32>) #1

attributes #0 = { nounwind "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-features"="-long-calls,-small-data" }
attributes #1 = { nocallback nofree nosync nounwind willreturn memory(none) }
