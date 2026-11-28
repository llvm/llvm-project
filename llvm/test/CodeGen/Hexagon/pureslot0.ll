; RUN: llc -mtriple=hexagon -mcpu=hexagonv73 -mattr=+hmx < %s | FileCheck %s

; Test that we packetize HMX instructions correctly. In this case, the
; mxmem2_bias must be in SLOT0, so it can't be packetized with another
; HMX instructions.

; CHECK: mxclracc.hf
; CHECK: bias = mxmem2(r0)
; CHECK: {
; CHECK: mxmem(r1,r2) = cvt


source_filename = "conv2d_hmx_kernel0"
target datalayout = "e-m:e-p:32:32:32-a:0-n16:32-i64:64:64-i32:32:32-i16:16:16-i1:8:8-f32:32:32-f64:64:64-v32:32:32-v64:64:64-v512:512:512-v1024:1024:1024-v2048:2048:2048"
target triple = "hexagon"

define dllexport i32 @conv2d_hmx_kernel0(ptr noalias %0, ptr noalias %1) local_unnamed_addr #0 {
entry:
  tail call void @llvm.hexagon.M8.mxclracc.hf()
  tail call void @llvm.hexagon.M8.mxmem2.bias(ptr %0)
  %2 = ptrtoint ptr %1 to i32
  %3 = or i32 %2, 1152
  tail call void @llvm.hexagon.M8.mxmem(ptr %1, i32 %3)
  ret i32 0
}

; Function Attrs: nounwind
declare void @llvm.hexagon.M8.mxclracc.hf() #0

; Function Attrs: nounwind
declare void @llvm.hexagon.M8.mxmem2.bias(ptr) #0

; Function Attrs: nounwind
declare void @llvm.hexagon.M8.mxmem(ptr, i32) #0

attributes #0 = { nounwind }
