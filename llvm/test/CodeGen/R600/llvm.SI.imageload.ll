;RUN: llc < %s -march=amdgcn -mcpu=verde -verify-machineinstrs | FileCheck %s
;RUN: llc < %s -march=amdgcn -mcpu=tonga -verify-machineinstrs | FileCheck %s

;CHECK-DAG: image_load {{v\[[0-9]+:[0-9]+\]}}, 15, 0, 0, -1
;CHECK-DAG: image_load_mip {{v\[[0-9]+:[0-9]+\]}}, 3, 0, 0, 0
;CHECK-DAG: image_load_mip {{v[0-9]+}}, 2, 0, 0, 0
;CHECK-DAG: image_load_mip {{v[0-9]+}}, 1, 0, 0, 0
;CHECK-DAG: image_load_mip {{v[0-9]+}}, 4, 0, 0, 0
;CHECK-DAG: image_load_mip {{v[0-9]+}}, 8, 0, 0, 0
;CHECK-DAG: image_load_mip {{v\[[0-9]+:[0-9]+\]}}, 5, 0, 0, 0
;CHECK-DAG: image_load_mip {{v\[[0-9]+:[0-9]+\]}}, 12, 0, 0, -1
;CHECK-DAG: image_load_mip {{v\[[0-9]+:[0-9]+\]}}, 7, 0, 0, 0
;CHECK-DAG: image_load_mip {{v[0-9]+}}, 8, 0, 0, -1

define void @test(i32 %a1, i32 %a2, i32 %a3, i32 %a4) {
   %v1 = insertelement <4 x i32> undef, i32 %a1, i32 0
   %v2 = insertelement <4 x i32> undef, i32 %a1, i32 1
   %v3 = insertelement <4 x i32> undef, i32 %a1, i32 2
   %v4 = insertelement <4 x i32> undef, i32 %a1, i32 3
   %v5 = insertelement <4 x i32> undef, i32 %a2, i32 0
   %v6 = insertelement <4 x i32> undef, i32 %a2, i32 1
   %v10 = insertelement <4 x i32> undef, i32 %a3, i32 1
   %v11 = insertelement <4 x i32> undef, i32 %a3, i32 2
   %v15 = insertelement <4 x i32> undef, i32 %a4, i32 2
   %v16 = insertelement <4 x i32> undef, i32 %a4, i32 3
   %res1 = call <4 x i32> @llvm.SI.imageload.(<4 x i32> %v1,
      <32 x i8> undef, i32 1)
   %res2 = call <4 x i32> @llvm.SI.imageload.(<4 x i32> %v2,
      <32 x i8> undef, i32 2)
   %res3 = call <4 x i32> @llvm.SI.imageload.(<4 x i32> %v3,
      <32 x i8> undef, i32 3)
   %res4 = call <4 x i32> @llvm.SI.imageload.(<4 x i32> %v4,
      <32 x i8> undef, i32 4)
   %res5 = call <4 x i32> @llvm.SI.imageload.(<4 x i32> %v5,
      <32 x i8> undef, i32 5)
   %res6 = call <4 x i32> @llvm.SI.imageload.(<4 x i32> %v6,
      <32 x i8> undef, i32 6)
   %res10 = call <4 x i32> @llvm.SI.imageload.(<4 x i32> %v10,
      <32 x i8> undef, i32 10)
   %res11 = call <4 x i32> @llvm.SI.imageload.(<4 x i32> %v11,
      <32 x i8> undef, i32 11)
   %res15 = call <4 x i32> @llvm.SI.imageload.(<4 x i32> %v15,
      <32 x i8> undef, i32 15)
   %res16 = call <4 x i32> @llvm.SI.imageload.(<4 x i32> %v16,
      <32 x i8> undef, i32 16)
   %e1 = extractelement <4 x i32> %res1, i32 0
   %e2 = extractelement <4 x i32> %res2, i32 1
   %e3 = extractelement <4 x i32> %res3, i32 2
   %e4 = extractelement <4 x i32> %res4, i32 3
   %t0 = extractelement <4 x i32> %res5, i32 0
   %t1 = extractelement <4 x i32> %res5, i32 1
   %e5 = add i32 %t0, %t1
   %t2 = extractelement <4 x i32> %res6, i32 0
   %t3 = extractelement <4 x i32> %res6, i32 2
   %e6 = add i32 %t2, %t3
   %t10 = extractelement <4 x i32> %res10, i32 2
   %t11 = extractelement <4 x i32> %res10, i32 3
   %e10 = add i32 %t10, %t11
   %t12 = extractelement <4 x i32> %res11, i32 0
   %t13 = extractelement <4 x i32> %res11, i32 1
   %t14 = extractelement <4 x i32> %res11, i32 2
   %t15 = add i32 %t12, %t13
   %e11 = add i32 %t14, %t15
   %t28 = extractelement <4 x i32> %res15, i32 0
   %t29 = extractelement <4 x i32> %res15, i32 1
   %t30 = extractelement <4 x i32> %res15, i32 2
   %t31 = extractelement <4 x i32> %res15, i32 3
   %t32 = add i32 %t28, %t29
   %t33 = add i32 %t30, %t31
   %e15 = add i32 %t32, %t33
   %e16 = extractelement <4 x i32> %res16, i32 3
   %s1 = add i32 %e1, %e2
   %s2 = add i32 %s1, %e3
   %s3 = add i32 %s2, %e4
   %s4 = add i32 %s3, %e5
   %s5 = add i32 %s4, %e6
   %s9 = add i32 %s5, %e10
   %s10 = add i32 %s9, %e11
   %s14 = add i32 %s10, %e15
   %s15 = add i32 %s14, %e16
   %s16 = bitcast i32 %s15 to float
   call void @llvm.SI.export(i32 15, i32 0, i32 1, i32 12, i32 0, float %s16, float %s16, float %s16, float %s16)
   ret void
}

; Test that ccordinates are stored in vgprs and not sgprs
; CHECK: vgpr_coords
; CHECK: image_load_mip {{v\[[0-9]+:[0-9]+\]}}, 15, 0, 0, 0, 0, 0, 0, 0, {{v\[[0-9]+:[0-9]+\]}}
define void @vgpr_coords(float addrspace(2)* addrspace(2)* inreg, <16 x i8> addrspace(2)* inreg, <32 x i8> addrspace(2)* inreg, i32 inreg, <2 x i32>, <2 x i32>, <2 x i32>, <3 x i32>, <2 x i32>, <2 x i32>, <2 x i32>, float, float, float, float, float, float, float, float, float) #0 {
main_body:
  %20 = getelementptr float addrspace(2)* addrspace(2)* %0, i32 0
  %21 = load float addrspace(2)* addrspace(2)* %20, !tbaa !2
  %22 = getelementptr float addrspace(2)* %21, i32 0
  %23 = load float addrspace(2)* %22, !tbaa !2, !invariant.load !1
  %24 = getelementptr float addrspace(2)* %21, i32 1
  %25 = load float addrspace(2)* %24, !tbaa !2, !invariant.load !1
  %26 = getelementptr float addrspace(2)* %21, i32 4
  %27 = load float addrspace(2)* %26, !tbaa !2, !invariant.load !1
  %28 = getelementptr <32 x i8> addrspace(2)* %2, i32 0
  %29 = load <32 x i8> addrspace(2)* %28, !tbaa !2
  %30 = bitcast float %27 to i32
  %31 = bitcast float %23 to i32
  %32 = bitcast float %25 to i32
  %33 = insertelement <4 x i32> undef, i32 %31, i32 0
  %34 = insertelement <4 x i32> %33, i32 %32, i32 1
  %35 = insertelement <4 x i32> %34, i32 %30, i32 2
  %36 = insertelement <4 x i32> %35, i32 undef, i32 3
  %37 = call <4 x i32> @llvm.SI.imageload.v4i32(<4 x i32> %36, <32 x i8> %29, i32 2)
  %38 = extractelement <4 x i32> %37, i32 0
  %39 = extractelement <4 x i32> %37, i32 1
  %40 = extractelement <4 x i32> %37, i32 2
  %41 = extractelement <4 x i32> %37, i32 3
  %42 = bitcast i32 %38 to float
  %43 = bitcast i32 %39 to float
  %44 = bitcast i32 %40 to float
  %45 = bitcast i32 %41 to float
  call void @llvm.SI.export(i32 15, i32 1, i32 1, i32 0, i32 0, float %42, float %43, float %44, float %45)
  ret void
}

declare <4 x i32> @llvm.SI.imageload.(<4 x i32>, <32 x i8>, i32) readnone
; Function Attrs: nounwind readnone
declare <4 x i32> @llvm.SI.imageload.v4i32(<4 x i32>, <32 x i8>, i32) #1

declare void @llvm.SI.export(i32, i32, i32, i32, i32, float, float, float, float)

attributes #0 = { "ShaderType"="0" }
attributes #1 = { nounwind readnone }

!0 = !{!"const", null}
!1 = !{}
!2 = !{!0, !0, i64 0, i32 1}
