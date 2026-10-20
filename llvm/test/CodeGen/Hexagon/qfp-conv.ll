; RUN: llc -mtriple=hexagon -mattr=+hvxv68,+hvx,+hvx-length128b < %s | FileCheck %s

; Test that the Qfloat optimization pass doesn't crash due to an invalid
; instructions.

; CHECK: v{{[0-9]+}}.hf = v{{[0-9]:[0-9]}}.qf32

define void @test(
    <32 x i32>* %optr,
    <64 x i32> %in64,
    <32 x i32> %va,
    <32 x i32> %vb
) local_unnamed_addr #0 {
entry:
  br label %for.body

for.body:
  %optr.068 = phi <32 x i32>* [ %optr, %entry ], [ %incdec.ptr6, %for.body ]
  %0 = tail call <32 x i32> @llvm.hexagon.V6.vconv.hf.qf32.128B(<64 x i32> %in64) #2
  %1 = tail call <32 x i32> @llvm.hexagon.V6.vdealh.128B(<32 x i32> %0) #2
  %2 = tail call <128 x i1> @llvm.hexagon.V6.vgth.128B(<32 x i32> %va, <32 x i32> %1) #2
  %3 = tail call <32 x i32> @llvm.hexagon.V6.vmux.128B(<128 x i1> %2, <32 x i32> %va, <32 x i32> %vb) #2
  %4 = tail call <32 x i32> @llvm.hexagon.V6.vaddhsat.128B(<32 x i32> %3, <32 x i32> %vb) #2
  %5 = tail call <32 x i32> @llvm.hexagon.V6.vpackhub.sat.128B(<32 x i32> %va, <32 x i32> %4) #2
  store <32 x i32> %5, <32 x i32>* %optr.068, align 1
  %incdec.ptr6 = getelementptr inbounds <32 x i32>, <32 x i32>* %optr.068, i32 1
  br label %for.body
}

declare <32 x i32> @llvm.hexagon.V6.vdealh.128B(<32 x i32>) #1
declare <32 x i32> @llvm.hexagon.V6.vconv.hf.qf32.128B(<64 x i32>) #1
declare <32 x i32> @llvm.hexagon.V6.vaddhsat.128B(<32 x i32>, <32 x i32>) #1
declare <32 x i32> @llvm.hexagon.V6.vpackhub.sat.128B(<32 x i32>, <32 x i32>) #1
declare <128 x i1> @llvm.hexagon.V6.vgth.128B(<32 x i32>, <32 x i32>) #1
declare <32 x i32> @llvm.hexagon.V6.vmux.128B(<128 x i1>, <32 x i32>, <32 x i32>) #1
