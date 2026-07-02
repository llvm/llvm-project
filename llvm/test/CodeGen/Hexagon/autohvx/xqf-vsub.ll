; Tests if correct vsub instructions are generated under different conditions

; RUN:   llc -mtriple=hexagon-unknown-elf -mhvx -mcpu=hexagonv79 -mattr=+hvxv79,+hvx-length128b,+hvx-qfloat -enable-xqf-gen=true \
; RUN: -enable-rem-conv=true -hexagon-qfloat-mode=ieee < %s | FileCheck %s

; The convert instruction before vsub should remain as it is.
define dso_local <32 x i32> @sub1_qf32(i32 noundef %input1, i32 noundef %input2, i32 noundef %size) local_unnamed_addr #0 {
; CHECK-LABEL: sub1_qf32
; CHECK: [[V3:v[0-9]+]].qf32 = vadd([[V1:v[0-9]+]].sf
; CHECK: [[V4:v[0-9]+]].sf = [[V3]].qf32
; CHECK: qf32 = vsub([[V1]].sf,[[V4]].sf)
entry:
  %0 = tail call <32 x i32> @llvm.hexagon.V6.lvsplatw.128B(i32 %input1)
  %1 = tail call <32 x i32> @llvm.hexagon.V6.lvsplatw.128B(i32 %input2)
  %2 = tail call <32 x i32> @llvm.hexagon.V6.vadd.sf.128B(<32 x i32> %0, <32 x i32> %1)
  %3 = tail call <32 x i32> @llvm.hexagon.V6.vconv.sf.qf32.128B(<32 x i32> %2)
  %4 = tail call <32 x i32> @llvm.hexagon.V6.vsub.sf.128B(<32 x i32> %0, <32 x i32> %3)
  ret <32 x i32> %4
}

; The convert instr before vsub can be removed and vsub opcode to be changed to take in qf32 type as op1.
define dso_local <32 x i32> @sub2_qf32(i32 noundef %input1, i32 noundef %input2, i32 noundef %size) local_unnamed_addr #0 {
; CHECK-LABEL: sub2_qf32
; CHECK: [[V3:v[0-9]+]].qf32 = vadd([[V1:v[0-9]+]].sf
; CHECK: qf32 = vsub([[V3]].qf32,[[V1]].sf)
entry:
  %0 = tail call <32 x i32> @llvm.hexagon.V6.lvsplatw.128B(i32 %input1)
  %1 = tail call <32 x i32> @llvm.hexagon.V6.lvsplatw.128B(i32 %input2)
  %2 = tail call <32 x i32> @llvm.hexagon.V6.vadd.sf.128B(<32 x i32> %0, <32 x i32> %1)
  %3 = tail call <32 x i32> @llvm.hexagon.V6.vconv.sf.qf32.128B(<32 x i32> %2)
  %4 = tail call <32 x i32> @llvm.hexagon.V6.vsub.sf.128B(<32 x i32> %3, <32 x i32> %0)
  ret <32 x i32> %4
}

; The convert instruction before vsub should remain as it is.
define dso_local <32 x i32> @sub1_qf16(i32 noundef %input1, i32 noundef %input2, i32 noundef %size) local_unnamed_addr #0 {
; CHECK-LABEL: sub1_qf16
; CHECK: [[V3:v[0-9]+]].qf16 = vadd([[V1:v[0-9]+]].hf
; CHECK: [[V4:v[0-9]+]].hf = [[V3]].qf16
; CHECK: qf16 = vsub([[V1]].hf,[[V4]].hf)
entry:
  %0 = tail call <32 x i32> @llvm.hexagon.V6.lvsplath.128B(i32 %input1)
  %1 = tail call <32 x i32> @llvm.hexagon.V6.lvsplath.128B(i32 %input2)
  %2 = tail call <32 x i32> @llvm.hexagon.V6.vadd.hf.128B(<32 x i32> %0, <32 x i32> %1)
  %3 = tail call <32 x i32> @llvm.hexagon.V6.vconv.hf.qf16.128B(<32 x i32> %2)
  %4 = tail call <32 x i32> @llvm.hexagon.V6.vsub.hf.128B(<32 x i32> %0, <32 x i32> %3)
  ret <32 x i32> %4
}

; The convert instr before vsub can be removed and vsub opcode to be changed to take in qf16 type as op1.
define dso_local <32 x i32> @sub2_qf16(i32 noundef %input1, i32 noundef %input2, i32 noundef %size) local_unnamed_addr #0 {
; CHECK-LABEL: sub2_qf16
; CHECK: [[V3:v[0-9]+]].qf16 = vadd([[V1:v[0-9]+]].hf
; CHECK: qf16 = vsub([[V3]].qf16,[[V1]].hf)
entry:
  %0 = tail call <32 x i32> @llvm.hexagon.V6.lvsplath.128B(i32 %input1)
  %1 = tail call <32 x i32> @llvm.hexagon.V6.lvsplath.128B(i32 %input2)
  %2 = tail call <32 x i32> @llvm.hexagon.V6.vadd.hf.128B(<32 x i32> %0, <32 x i32> %1)
  %3 = tail call <32 x i32> @llvm.hexagon.V6.vconv.hf.qf16.128B(<32 x i32> %2)
  %4 = tail call <32 x i32> @llvm.hexagon.V6.vsub.hf.128B(<32 x i32> %3, <32 x i32> %0)
  ret <32 x i32> %4
}

; The convert instr before vadd can be removed.
define dso_local <32 x i32> @add1_qf32(i32 noundef %input1, i32 noundef %input2, i32 noundef %size) local_unnamed_addr #0 {
; CHECK-LABEL: add1_qf32
; CHECK: [[V3:v[0-9]+]].qf32 = vadd([[V1:v[0-9]+]].sf
; CHECK: qf32 = vadd([[V3]].qf32,[[V1]].sf)
entry:
  %0 = tail call <32 x i32> @llvm.hexagon.V6.lvsplatw.128B(i32 %input1)
  %1 = tail call <32 x i32> @llvm.hexagon.V6.lvsplatw.128B(i32 %input2)
  %2 = tail call <32 x i32> @llvm.hexagon.V6.vadd.sf.128B(<32 x i32> %0, <32 x i32> %1)
  %3 = tail call <32 x i32> @llvm.hexagon.V6.vconv.sf.qf32.128B(<32 x i32> %2)
  %4 = tail call <32 x i32> @llvm.hexagon.V6.vadd.sf.128B(<32 x i32> %3, <32 x i32> %0)
  ret <32 x i32> %4
}

; The convert instr before vsub can be removed and ops to last vadd can be interchanged
define dso_local <32 x i32> @add2_qf32(i32 noundef %input1, i32 noundef %input2, i32 noundef %size) local_unnamed_addr #0 {
; CHECK-LABEL: add2_qf32
; CHECK: [[V3:v[0-9]+]].qf32 = vadd([[V1:v[0-9]+]].sf
; CHECK: qf32 = vadd([[V3]].qf32,[[V1]].sf)
entry:
  %0 = tail call <32 x i32> @llvm.hexagon.V6.lvsplatw.128B(i32 %input1)
  %1 = tail call <32 x i32> @llvm.hexagon.V6.lvsplatw.128B(i32 %input2)
  %2 = tail call <32 x i32> @llvm.hexagon.V6.vadd.sf.128B(<32 x i32> %0, <32 x i32> %1)
  %3 = tail call <32 x i32> @llvm.hexagon.V6.vconv.sf.qf32.128B(<32 x i32> %2)
  %4 = tail call <32 x i32> @llvm.hexagon.V6.vadd.sf.128B(<32 x i32> %0, <32 x i32> %3)
  ret <32 x i32> %4
}

; The convert instr before vadd can be removed.
define dso_local <32 x i32> @add1_qf16(i32 noundef %input1, i32 noundef %input2, i32 noundef %size) local_unnamed_addr #0 {
; CHECK-LABEL: add1_qf16
; CHECK: [[V3:v[0-9]+]].qf16 = vadd([[V1:v[0-9]+]].hf
; CHECK: qf16 = vadd([[V3]].qf16,[[V1]].hf)
entry:
  %0 = tail call <32 x i32> @llvm.hexagon.V6.lvsplath.128B(i32 %input1)
  %1 = tail call <32 x i32> @llvm.hexagon.V6.lvsplath.128B(i32 %input2)
  %2 = tail call <32 x i32> @llvm.hexagon.V6.vadd.hf.128B(<32 x i32> %0, <32 x i32> %1)
  %3 = tail call <32 x i32> @llvm.hexagon.V6.vconv.hf.qf16.128B(<32 x i32> %2)
  %4 = tail call <32 x i32> @llvm.hexagon.V6.vadd.hf.128B(<32 x i32> %3, <32 x i32> %0)
  ret <32 x i32> %4
}

; The convert instr before vsub can be removed and ops to last vadd can be interchanged
define dso_local <32 x i32> @add2_qf16(i32 noundef %input1, i32 noundef %input2, i32 noundef %size) local_unnamed_addr #0 {
; CHECK-LABEL: add2_qf16
; CHECK: [[V3:v[0-9]+]].qf16 = vadd([[V1:v[0-9]+]].hf
; CHECK: qf16 = vadd([[V3]].qf16,[[V1]].hf)
entry:
  %0 = tail call <32 x i32> @llvm.hexagon.V6.lvsplath.128B(i32 %input1)
  %1 = tail call <32 x i32> @llvm.hexagon.V6.lvsplath.128B(i32 %input2)
  %2 = tail call <32 x i32> @llvm.hexagon.V6.vadd.hf.128B(<32 x i32> %0, <32 x i32> %1)
  %3 = tail call <32 x i32> @llvm.hexagon.V6.vconv.hf.qf16.128B(<32 x i32> %2)
  %4 = tail call <32 x i32> @llvm.hexagon.V6.vadd.hf.128B(<32 x i32> %0, <32 x i32> %3)
  ret <32 x i32> %4
}

declare <32 x i32> @llvm.hexagon.V6.vsub.sf.128B(<32 x i32>, <32 x i32>) #1
declare <32 x i32> @llvm.hexagon.V6.vsub.hf.128B(<32 x i32>, <32 x i32>) #1
declare <32 x i32> @llvm.hexagon.V6.lvsplatw.128B(i32) #1
declare <32 x i32> @llvm.hexagon.V6.lvsplath.128B(i32) #1
declare <32 x i32> @llvm.hexagon.V6.vconv.sf.qf32.128B(<32 x i32>) #1
declare <32 x i32> @llvm.hexagon.V6.vconv.hf.qf16.128B(<32 x i32>) #1
declare <32 x i32> @llvm.hexagon.V6.vadd.sf.128B(<32 x i32>, <32 x i32>) #1
declare <32 x i32> @llvm.hexagon.V6.vadd.hf.128B(<32 x i32>, <32 x i32>) #1

attributes #0 = { nounwind "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="hexagonv79" "target-features"="+hvx-length128b,+hvxv79,+v79,-long-calls,-small-data" }
attributes #1 = { nocallback nofree nosync nounwind willreturn memory(none) }
