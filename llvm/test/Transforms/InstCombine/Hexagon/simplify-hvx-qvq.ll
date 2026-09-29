; RUN: opt -passes=instcombine -S < %s | FileCheck %s

; Simplify the Q -> V -> Q sequence, i.e. (vandvrt (vandqrt q b) m) -> q
; when every byte in (b & m) is non-zero.

target datalayout = "e-m:e-p:32:32:32-a:0-n16:32-i64:64:64-i32:32:32-i16:16:16-i1:8:8-f32:32:32-f64:64:64-v32:32:32-v64:64:64-v512:512:512-v1024:1024:1024-v2048:2048:2048"
target triple = "hexagon"


; CHECK-LABEL: define {{.*}} @f0(<32 x i32> %a0, <32 x i32> %a1, <32 x i32> %a2)
; CHECK: %[[V0:v[0-9]+]] = call <128 x i1> @llvm.hexagon.V6.vgtw.128B(<32 x i32> %a0, <32 x i32> %a1)
; CHECK: %[[V1:v[0-9]+]] = call <128 x i1> @llvm.hexagon.V6.veqb.or.128B(<128 x i1> %[[V0]], <32 x i32> %a1, <32 x i32> %a2)
; CHECK: %[[V2:v[0-9]+]] = call <128 x i1> @llvm.hexagon.V6.pred.and.128B(<128 x i1> %[[V0]], <128 x i1> %[[V1]])
; CHECK: %[[V3:v[0-9]+]] = call <128 x i1> @llvm.hexagon.V6.pred.xor.128B(<128 x i1> %[[V1]], <128 x i1> %[[V2]])
; CHECK: %[[V4:v[0-9]+]] = call <32 x i32> @llvm.hexagon.V6.vmux.128B(<128 x i1> %[[V3]], <32 x i32> %a1, <32 x i32> %a2)
; CHECK: ret <32 x i32> %[[V4]]
define inreg <32 x i32> @f0(<32 x i32> %a0, <32 x i32> %a1, <32 x i32> %a2) #0 {
b0:
  %v0 = call <128 x i1> @llvm.hexagon.V6.vgtw.128B(<32 x i32> %a0, <32 x i32> %a1)
  %v1 = call <32 x i32> @llvm.hexagon.V6.vandqrt.128B(<128 x i1> %v0, i32 -1)
  %v2 = call <128 x i1> @llvm.hexagon.V6.vandvrt.128B(<32 x i32> %v1, i32 -1)
  %v3 = call <128 x i1> @llvm.hexagon.V6.veqb.or.128B(<128 x i1> %v2, <32 x i32> %a1, <32 x i32> %a2)
  %v4 = call <32 x i32> @llvm.hexagon.V6.vandqrt.128B(<128 x i1> %v3, i32 -1)
  %v5 = call <128 x i1> @llvm.hexagon.V6.vandvrt.128B(<32 x i32> %v1, i32 -1)
  %v6 = call <128 x i1> @llvm.hexagon.V6.vandvrt.128B(<32 x i32> %v4, i32 -1)
  %v7 = call <128 x i1> @llvm.hexagon.V6.pred.and.128B(<128 x i1> %v5, <128 x i1> %v6)
  %v8 = call <32 x i32> @llvm.hexagon.V6.vandqrt.128B(<128 x i1> %v7, i32 -1)
  %v9 = call <128 x i1> @llvm.hexagon.V6.vandvrt.128B(<32 x i32> %v4, i32 -1)
  %v10 = call <128 x i1> @llvm.hexagon.V6.vandvrt.128B(<32 x i32> %v8, i32 -1)
  %v11 = call <128 x i1> @llvm.hexagon.V6.pred.xor.128B(<128 x i1> %v9, <128 x i1> %v10)
  %v12 = call <32 x i32> @llvm.hexagon.V6.vandqrt.128B(<128 x i1> %v11, i32 -1)
  %v13 = call <128 x i1> @llvm.hexagon.V6.vandvrt.128B(<32 x i32> %v12, i32 -1)
  %v14 = call <32 x i32> @llvm.hexagon.V6.vmux.128B(<128 x i1> %v13, <32 x i32> %a1, <32 x i32> %a2)
  ret <32 x i32> %v14
}

; Bytes = 0x08040201, Mask = 0x0C060309, Common = 0x08040201: all bytes in
; the common bits are non-zero, expect simplification.
; CHECK-LABEL: define {{.*}} @f1(<32 x i32> %a0, <32 x i32> %a1, <32 x i32> %a2)
; CHECK: %[[V0:v[0-9]+]] = call <128 x i1> @llvm.hexagon.V6.vgtw.128B(<32 x i32> %a0, <32 x i32> %a1)
; CHECK: %[[V1:v[0-9]+]] = call <32 x i32> @llvm.hexagon.V6.vmux.128B(<128 x i1> %[[V0]], <32 x i32> %a1, <32 x i32> %a2)
; CHECK:  ret <32 x i32> %[[V1]]
define inreg <32 x i32> @f1(<32 x i32> %a0, <32 x i32> %a1, <32 x i32> %a2) #0 {
b0:
  %v0 = call <128 x i1> @llvm.hexagon.V6.vgtw.128B(<32 x i32> %a0, <32 x i32> %a1)
  %v1 = call <32 x i32> @llvm.hexagon.V6.vandqrt.128B(<128 x i1> %v0, i32 134480385)
  %v2 = call <128 x i1> @llvm.hexagon.V6.vandvrt.128B(<32 x i32> %v1, i32 201720585)
  %v3 = call <32 x i32> @llvm.hexagon.V6.vmux.128B(<128 x i1> %v2, <32 x i32> %a1, <32 x i32> %a2)
  ret <32 x i32> %v3
}

; Bytes = 0x08040201, Mask = 0x0C060309, Common = 0x08040200: there is a
; zero byte in the common bits, so vandqrt->vandvrt cannot be simplified.
; CHECK-LABEL: define {{.*}} @f2(<32 x i32> %a0, <32 x i32> %a1, <32 x i32> %a2)
; CHECK: %[[V0:v[0-9]+]] = call <128 x i1> @llvm.hexagon.V6.vgtw.128B(<32 x i32> %a0, <32 x i32> %a1)
; CHECK: %[[V1:v[0-9]+]] = call <32 x i32> @llvm.hexagon.V6.vandqrt.128B(<128 x i1> %[[V0]], i32 134480385)
; CHECK: %[[V2:v[0-9]+]] = call <128 x i1> @llvm.hexagon.V6.vandvrt.128B(<32 x i32> %[[V1]], i32 201720584)
; CHECK: %[[V3:v[0-9]+]] = call <32 x i32> @llvm.hexagon.V6.vmux.128B(<128 x i1> %[[V2]], <32 x i32> %a1, <32 x i32> %a2)
; CHECK: ret <32 x i32> %[[V3]]
define inreg <32 x i32> @f2(<32 x i32> %a0, <32 x i32> %a1, <32 x i32> %a2) #0 {
b0:
  %v0 = call <128 x i1> @llvm.hexagon.V6.vgtw.128B(<32 x i32> %a0, <32 x i32> %a1)
  %v1 = call <32 x i32> @llvm.hexagon.V6.vandqrt.128B(<128 x i1> %v0, i32 134480385)
  %v2 = call <128 x i1> @llvm.hexagon.V6.vandvrt.128B(<32 x i32> %v1, i32 201720584)
  %v3 = call <32 x i32> @llvm.hexagon.V6.vmux.128B(<128 x i1> %v2, <32 x i32> %a1, <32 x i32> %a2)
  ret <32 x i32> %v3
}

declare <128 x i1> @llvm.hexagon.V6.vgtw.128B(<32 x i32>, <32 x i32>) #1
declare <32 x i32> @llvm.hexagon.V6.vandqrt.128B(<128 x i1>, i32) #1
declare <128 x i1> @llvm.hexagon.V6.veqb.or.128B(<128 x i1>, <32 x i32>, <32 x i32>) #1
declare <128 x i1> @llvm.hexagon.V6.vandvrt.128B(<32 x i32>, i32) #1
declare <128 x i1> @llvm.hexagon.V6.pred.and.128B(<128 x i1>, <128 x i1>) #1
declare <128 x i1> @llvm.hexagon.V6.pred.xor.128B(<128 x i1>, <128 x i1>) #1
declare <32 x i32> @llvm.hexagon.V6.vmux.128B(<128 x i1>, <32 x i32>, <32 x i32>) #1

attributes #0 = { noinline nounwind "target-cpu"="hexagonv65" "target-features"="+hvx-length128b,+hvxv65,+v65,-long-calls" }
attributes #1 = { nounwind readnone }

!llvm.module.flags = !{!0}

!0 = !{i32 1, !"wchar_size", i32 4}
