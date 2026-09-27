; RUN: llc -mtriple=hexagon -O3 -stop-after=finalize-isel -o - < %s | FileCheck %s

; CHECK: V6_vgathermh_pseudo {{.*}} :: (volatile load store (s1024)
; CHECK: V6_vgathermw_pseudo {{.*}} :: (volatile load store (s1024)
; CHECK: V6_vgathermhw_pseudo {{.*}} :: (volatile load store (s2048)
; CHECK: V6_vgathermhq_pseudo {{.*}} :: (volatile load store (s1024)
; CHECK: V6_vgathermwq_pseudo {{.*}} :: (volatile load store (s1024)
; CHECK: V6_vgathermhwq_pseudo {{.*}} :: (volatile load store (s2048)
; CHECK: V6_vgather_vscatter_mh_pseudo {{.*}} :: (volatile load store (s1024)

target triple = "hexagon"

define dso_local void @test_vgather_memvt_128B(ptr %p, i32 %Rb, i32 %mu, <32 x i32> %Vv, <64 x i32> %Vvv, <32 x i32> %Qs) local_unnamed_addr #0 {
entry:
  %Qp = tail call <128 x i1> @llvm.hexagon.V6.vandvrt.128B(<32 x i32> %Qs, i32 -1)
  call void @llvm.hexagon.V6.vgathermh.128B(ptr %p, i32 %Rb, i32 %mu, <32 x i32> %Vv)
  call void @llvm.hexagon.V6.vgathermw.128B(ptr %p, i32 %Rb, i32 %mu, <32 x i32> %Vv)
  call void @llvm.hexagon.V6.vgathermhw.128B(ptr %p, i32 %Rb, i32 %mu, <64 x i32> %Vvv)
  call void @llvm.hexagon.V6.vgathermhq.128B(ptr %p, <128 x i1> %Qp, i32 %Rb, i32 %mu, <32 x i32> %Vv)
  call void @llvm.hexagon.V6.vgathermwq.128B(ptr %p, <128 x i1> %Qp, i32 %Rb, i32 %mu, <32 x i32> %Vv)
  call void @llvm.hexagon.V6.vgathermhwq.128B(ptr %p, <128 x i1> %Qp, i32 %Rb, i32 %mu, <64 x i32> %Vvv)
  call void @llvm.hexagon.V6.vgather.vscattermh.128B(ptr %p, i32 %Rb, i32 %mu, <32 x i32> %Vv)
  ret void
}

declare void @llvm.hexagon.V6.vgathermh.128B(ptr, i32, i32, <32 x i32>) #1
declare void @llvm.hexagon.V6.vgathermw.128B(ptr, i32, i32, <32 x i32>) #1
declare void @llvm.hexagon.V6.vgathermhw.128B(ptr, i32, i32, <64 x i32>) #1
declare void @llvm.hexagon.V6.vgathermhq.128B(ptr, <128 x i1>, i32, i32, <32 x i32>) #1
declare void @llvm.hexagon.V6.vgathermwq.128B(ptr, <128 x i1>, i32, i32, <32 x i32>) #1
declare void @llvm.hexagon.V6.vgathermhwq.128B(ptr, <128 x i1>, i32, i32, <64 x i32>) #1
declare void @llvm.hexagon.V6.vgather.vscattermh.128B(ptr, i32, i32, <32 x i32>) #1
declare <128 x i1> @llvm.hexagon.V6.vandvrt.128B(<32 x i32>, i32) #2

attributes #0 = { nounwind "target-cpu"="hexagonv75" "target-features"="+hvx-length128b,+hvxv75,-long-calls" }
