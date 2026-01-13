; Tests if the flag to disable qfp optimizer pass works or not.

; RUN: llc -march=hexagon -mcpu=hexagonv69 -mattr=+hvxv69,+hvx-length128b \
; RUN: < %s -o -| FileCheck %s --check-prefix=ENABLED
; RUN: llc -march=hexagon -mcpu=hexagonv69 -mattr=+hvxv69,+hvx-length128b \
; RUN: -disable-qfp-opt < %s -o -| FileCheck %s --check-prefix=DISABLED

define dso_local <32 x i32> @conv1_qf32(<32 x i32> noundef %input1, <32 x i32> noundef %input2) local_unnamed_addr {
entry:
; DISABLED: [[V2:v[0-9]+]].qf32 = vadd(v0.sf,v1.sf)
; DISABLED: [[V3:v[0-9]+]].sf = [[V2]].qf32
; DISABLED: qf32 = vadd(v0.sf,[[V3]].sf)
; ENABLED: [[V4:v[0-9]+]].qf32 = vadd(v0.sf,v1.sf)
; ENABLED: qf32 = vadd([[V4]].qf32,v0.sf)
  %0 = tail call <32 x i32> @llvm.hexagon.V6.vadd.sf.128B(<32 x i32> %input1, <32 x i32> %input2)
  %1 = tail call <32 x i32> @llvm.hexagon.V6.vconv.sf.qf32.128B(<32 x i32> %0)
  %2 = tail call <32 x i32> @llvm.hexagon.V6.vadd.sf.128B(<32 x i32> %input1, <32 x i32> %1)
  ret <32 x i32> %2
}
