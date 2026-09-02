; For v81, tests if correct vsub instructions are generated under different conditions

; RUN:   llc -mtriple=hexagon-unknown-elf -mhvx -mcpu=hexagonv81 -mattr=+hvxv81,+hvx-length128b,+hvx-qfloat -enable-xqf-gen=true \
; RUN: -enable-rem-conv=true -hexagon-qfloat-mode=ieee < %s | FileCheck %s

; The convert instruction before vsub should be removed and vsub opcode changed to take in qf32 as op2
define dso_local <32 x i32> @sub1_qf32(i32 noundef %input1, i32 noundef %input2, i32 noundef %size) local_unnamed_addr #0 {
; CHECK-LABEL: sub1_qf32
; CHECK: [[V3:v[0-9]+]].qf32 = vadd([[V1:v[0-9]+]].sf
; CHECK: qf32 = vsub([[V1]].sf,[[V3]].qf32)
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

; The convert instruction before vsub should be removed and vsub opcode changed to take in qf16 as op2
define dso_local <32 x i32> @sub1_qf16(i32 noundef %input1, i32 noundef %input2, i32 noundef %size) local_unnamed_addr #0 {
; CHECK-LABEL: sub1_qf16
; CHECK: [[V3:v[0-9]+]].qf16 = vadd([[V1:v[0-9]+]].hf
; CHECK: qf16 = vsub([[V1]].hf,[[V3]].qf16)
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

; The convert instr before vadd can be removed and ops to last vadd can be interchanged
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
entry:                                                                                                                                                                                                               %0 = tail call <32 x i32> @llvm.hexagon.V6.lvsplath.128B(i32 %input1)
  %1 = tail call <32 x i32> @llvm.hexagon.V6.lvsplath.128B(i32 %input2)                                                                                                                                              %2 = tail call <32 x i32> @llvm.hexagon.V6.vadd.hf.128B(<32 x i32> %0, <32 x i32> %1)
  %3 = tail call <32 x i32> @llvm.hexagon.V6.vconv.hf.qf16.128B(<32 x i32> %2)                                                                                                                                       %4 = tail call <32 x i32> @llvm.hexagon.V6.vadd.hf.128B(<32 x i32> %3, <32 x i32> %0)
  ret <32 x i32> %4                                                                                                                                                                                                }

; The convert instr before vadd can be removed and ops to last vadd can be interchanged
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

; The convert instr before vmul can be removed and ops to last vadd can be interchanged
define dso_local <32 x i32> @mpy2_qf32(i32 noundef %input1, i32 noundef %input2, i32 noundef %size) local_unnamed_addr #0 {
; CHECK-LABEL: mpy2_qf32
; CHECK: qf32 = vadd([[V1:v[0-9]+]].sf
; CHECK: qf32 = vmpy(v{{[0-9]+}}.qf32,v{{[0-9]+}}.qf32)
entry:
  %0 = tail call <32 x i32> @llvm.hexagon.V6.lvsplatw.128B(i32 %input1)
  %1 = tail call <32 x i32> @llvm.hexagon.V6.lvsplatw.128B(i32 %input2)
  %2 = tail call <32 x i32> @llvm.hexagon.V6.vadd.sf.128B(<32 x i32> %0, <32 x i32> %1)
  %3 = tail call <32 x i32> @llvm.hexagon.V6.vconv.sf.qf32.128B(<32 x i32> %2)
  %4 = tail call <32 x i32> @llvm.hexagon.V6.vmpy.qf32.sf.128B(<32 x i32> %0, <32 x i32> %3)
  ret <32 x i32> %4
}

; The convert instr before vmul can be removed.
define dso_local <32 x i32> @mpy1_qf16(i32 noundef %input1, i32 noundef %input2, i32 noundef %size) local_unnamed_addr #0 {
; CHECK-LABEL: mpy1_qf16
; CHECK: [[V3:v[0-9]+]].qf16 = vadd([[V1:v[0-9]+]].hf
; CHECK: qf32 = vmpy([[V3]].qf16,[[V1]].hf)
entry:                                                                                                                                                                                                               %0 = tail call <32 x i32> @llvm.hexagon.V6.lvsplath.128B(i32 %input1)
  %1 = tail call <32 x i32> @llvm.hexagon.V6.lvsplath.128B(i32 %input2)                                                                                                                                              %2 = tail call <32 x i32> @llvm.hexagon.V6.vadd.hf.128B(<32 x i32> %0, <32 x i32> %1)
  %3 = tail call <32 x i32> @llvm.hexagon.V6.vconv.hf.qf16.128B(<32 x i32> %2)                                                                                                                                       %4 = tail call <32 x i32> @llvm.hexagon.V6.vmpy.qf16.hf.128B(<32 x i32> %3, <32 x i32> %0)
  ret <32 x i32> %4                                                                                                                                                                                                }

; The convert instr before vmul can be removed and ops to last vadd can be interchanged
define dso_local <32 x i32> @mpy2_qf16(i32 noundef %input1, i32 noundef %input2, i32 noundef %size) local_unnamed_addr #0 {
; CHECK-LABEL: mpy2_qf16
; CHECK: [[V3:v[0-9]+]].qf16 = vadd([[V1:v[0-9]+]].hf
; CHECK: qf32 = vmpy([[V3]].qf16,[[V1]].hf)
entry:
  %0 = tail call <32 x i32> @llvm.hexagon.V6.lvsplath.128B(i32 %input1)
  %1 = tail call <32 x i32> @llvm.hexagon.V6.lvsplath.128B(i32 %input2)
  %2 = tail call <32 x i32> @llvm.hexagon.V6.vadd.hf.128B(<32 x i32> %0, <32 x i32> %1)
  %3 = tail call <32 x i32> @llvm.hexagon.V6.vconv.hf.qf16.128B(<32 x i32> %2)
  %4 = tail call <32 x i32> @llvm.hexagon.V6.vmpy.qf16.hf.128B(<32 x i32> %0, <32 x i32> %3)
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

attributes #0 = { nounwind "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-features"="-long-calls,-small-data" }
attributes #1 = { nocallback nofree nosync nounwind willreturn memory(none) }
