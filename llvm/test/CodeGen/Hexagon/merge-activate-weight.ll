; RUN: llc -mtriple=hexagon -mcpu=hexagonv68 -mattr=+hmx < %s | FileCheck %s

; This test was checking for a copy of an operand that is defined between the
; activation and weight instructions when HexagonMergeActivateWeight was run
; only after register allocator. A copy is not created anymore because
; HexagonMergeActivateWeight now run before register allocator as well.

; CHECK: [[REG1:r(([0-9]+))]] = or
; CHECK: activation.hf = mxmem([[REG1]],r{{[0-9]+}})
; CHECK: weight.hf = mxmem

define dllexport i32 @conv2d_hmx_kernel1(i8* noalias, i8* noalias) local_unnamed_addr #1 {
entry:
  tail call void @llvm.hexagon.M8.mxclracc.hf()
  %2 = ptrtoint i8* %0 to i32
  %3 = or i32 %2, 1152
  tail call void @llvm.hexagon.M8.mxmem.blk.sm.act.hf(i32 %3, i32 3960)
  %4 = ptrtoint i8* %1 to i32
  tail call void @llvm.hexagon.M8.mxmem.wei.hf(i32 %4, i32 25600)
  ret i32 0
}

declare void @llvm.hexagon.M8.mxclracc.hf() #2
declare void @llvm.hexagon.M8.mxmem.blk.sm.act.hf(i32, i32) #2
declare void @llvm.hexagon.M8.mxmem.wei.hf(i32, i32) #2
