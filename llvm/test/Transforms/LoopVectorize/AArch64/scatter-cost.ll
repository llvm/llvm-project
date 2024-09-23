; RUN: opt -mtriple aarch64 -mcpu=neoverse-v2 -passes="print<cost-model>" -disable-output < %s 2>&1 | FileCheck %s -check-prefix=CHECK-V2
; RUN: opt -mtriple aarch64 -mattr=+sve2  -passes="print<cost-model>" -disable-output < %s 2>&1 | FileCheck %s -check-prefix=CHECK-GENERIC
; CHECK-V2: Cost Model: Found an estimated cost of 52 for instruction: call void @llvm.masked.scatter.nxv4f32
; CHECK-GENERIC: Cost Model: Found an estimated cost of 80 for instruction: call void @llvm.masked.scatter.nxv4f32

define void @masked_scatter_nxv8f32_i64(<vscale x 4 x float> %data, <vscale x 4 x ptr> %b, <vscale x 4 x i64> %V) #0 {
  call void @llvm.masked.scatter.nxv4f32.nxv4p0(<vscale x 4 x float> %data, <vscale x 4 x ptr> %b, i32 4, <vscale x 4 x i1> shufflevector (<vscale x 4 x i1> insertelement (<vscale x 4 x i1> poison, i1 true, i64 0), <vscale x 4 x i1> poison, <vscale x 4 x i32> zeroinitializer))
  ret void
}

