; RUN: llc -march=hexagon < %s | FileCheck %s

target datalayout = "e-m:e-p:32:32:32-a:0-n16:32-i64:64:64-i32:32:32-i16:16:16-i1:8:8-f32:32:32-f64:64:64-v32:32:32-v64:64:64-v512:512:512-v1024:1024:1024-v2048:2048:2048"
target triple = "hexagon"

; Check that there are two pairs of adjacent activation/weight instructions.

; CHECK-LABEL: f0:
; CHECK: activation.hf
; CHECK-NEXT: weight.hf
; CHECK-LABEL: LBB0_2:
; CHECK: activation.hf
; CHECK-NEXT: weight.hf
define dllexport void @f0(i32 %a0, i32 %a1) local_unnamed_addr #0 {
b0:
  %v0 = icmp slt i32 %a0, %a1
  br i1 %v0, label %b1, label %b2

b1:                                               ; preds = %b1, %b0
  %v1 = add i32 %a0, 1
  tail call void @f1()
  tail call void @llvm.hexagon.M8.mxmems.blk.sm.act.hf(i32 %v1, i32 %a1)
  tail call void @llvm.hexagon.M8.mxmem.wei.hf(i32 %a0, i32 %a1)
  br label %b3

b2:
  %v2 = add i32 %a1, 4
  tail call void @llvm.hexagon.M8.mxmems.blk.sm.act.hf(i32 %a0, i32 %v2)
  tail call void @llvm.hexagon.M8.mxmem.wei.hf(i32 %a0, i32 %a1)
  br label %b3

b3:
  ret void
}

declare void @f1() #0

; Function Attrs: nounwind writeonly
declare void @llvm.hexagon.M8.mxmem.wei.hf(i32, i32) #1

; Function Attrs: nounwind writeonly
declare void @llvm.hexagon.M8.mxmems.blk.sm.act.hf(i32, i32) #1

attributes #0 = { "target-features"="+hmx" }
attributes #1 = { nounwind writeonly convergent nomerge noduplicate }
