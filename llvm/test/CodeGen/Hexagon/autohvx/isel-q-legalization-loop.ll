; RUN: llc -march=hexagon < %s | FileCheck %s
; REQUIRES: asserts

; Check that this doesn't crash.
; CHECK: vand

target datalayout = "e-m:e-p:32:32:32-a:0-n16:32-i64:64:64-i32:32:32-i16:16:16-i1:8:8-f32:32:32-f64:64:64-v32:32:32-v64:64:64-v512:512:512-v1024:1024:1024-v2048:2048:2048"
target triple = "hexagon"

%s.0 = type { [4 x <32 x i32>] }

declare <32 x i32> @llvm.hexagon.V6.vmux.128B(<128 x i1>, <32 x i32>, <32 x i32>) #0
declare <128 x i1> @llvm.hexagon.V6.vandvrt.128B(<32 x i32>, i32) #0
declare <64 x i32> @llvm.hexagon.V6.vdealvdd.128B(<32 x i32>, <32 x i32>, i32) #0
declare <32 x i32> @llvm.hexagon.V6.hi.128B(<64 x i32>) #0

; Function Attrs: nounwind
define void @f0() local_unnamed_addr #1 {
b0:
  %v0 = tail call <128 x i1> @llvm.hexagon.V6.vandvrt.128B(<32 x i32> undef, i32 16843009)
  %v1 = getelementptr inbounds %s.0, %s.0* null, i32 0, i32 0, i32 3
  br label %b1

b1:                                               ; preds = %b1, %b0
  %v2 = phi i32 [ 0, %b0 ], [ %v11, %b1 ]
  %v3 = and i32 %v2, 1
  %v4 = icmp eq i32 %v3, 0
  %v5 = select i1 %v4, <128 x i1> zeroinitializer, <128 x i1> %v0
  %v6 = tail call <32 x i32> @llvm.hexagon.V6.vmux.128B(<128 x i1> %v5, <32 x i32> undef, <32 x i32> undef)
  %v7 = tail call <64 x i32> @llvm.hexagon.V6.vdealvdd.128B(<32 x i32> undef, <32 x i32> %v6, i32 -32)
  %v8 = tail call <32 x i32> @llvm.hexagon.V6.hi.128B(<64 x i32> %v7)
  %v9 = tail call <64 x i32> @llvm.hexagon.V6.vdealvdd.128B(<32 x i32> undef, <32 x i32> %v8, i32 -32)
  %v10 = tail call <32 x i32> @llvm.hexagon.V6.hi.128B(<64 x i32> %v9)
  store <32 x i32> %v10, <32 x i32>* %v1, align 128
  %v11 = add nuw nsw i32 %v2, 1
  br label %b1
}

attributes #0 = { nounwind readnone }
attributes #1 = { nounwind "target-cpu"="hexagonv66" "target-features"="+hvx,+hvx-length128b" }
