;RUN: llc < %s -march=amdgcn -mcpu=verde -verify-machineinstrs | FileCheck %s
;RUN: llc < %s -march=amdgcn -mcpu=tonga -verify-machineinstrs | FileCheck %s

;CHECK-DAG: image_sample_d {{v\[[0-9]+:[0-9]+\]}}, 15
;CHECK-DAG: image_sample_d {{v\[[0-9]+:[0-9]+\]}}, 3
;CHECK-DAG: image_sample_d {{v[0-9]+}}, 2
;CHECK-DAG: image_sample_d {{v[0-9]+}}, 1
;CHECK-DAG: image_sample_d {{v[0-9]+}}, 4
;CHECK-DAG: image_sample_d {{v[0-9]+}}, 8
;CHECK-DAG: image_sample_c_d {{v\[[0-9]+:[0-9]+\]}}, 5
;CHECK-DAG: image_sample_c_d {{v\[[0-9]+:[0-9]+\]}}, 9
;CHECK-DAG: image_sample_c_d {{v\[[0-9]+:[0-9]+\]}}, 6
;CHECK-DAG: image_sample_d {{v\[[0-9]+:[0-9]+\]}}, 10
;CHECK-DAG: image_sample_d {{v\[[0-9]+:[0-9]+\]}}, 12
;CHECK-DAG: image_sample_c_d {{v\[[0-9]+:[0-9]+\]}}, 7
;CHECK-DAG: image_sample_c_d {{v\[[0-9]+:[0-9]+\]}}, 11
;CHECK-DAG: image_sample_c_d {{v\[[0-9]+:[0-9]+\]}}, 13
;CHECK-DAG: image_sample_d {{v\[[0-9]+:[0-9]+\]}}, 14
;CHECK-DAG: image_sample_d {{v[0-9]+}}, 8

define void @test(i32 %a1, i32 %a2, i32 %a3, i32 %a4) #0 {
   %v1 = insertelement <4 x i32> undef, i32 %a1, i32 0
   %v2 = insertelement <4 x i32> undef, i32 %a1, i32 1
   %v3 = insertelement <4 x i32> undef, i32 %a1, i32 2
   %v4 = insertelement <4 x i32> undef, i32 %a1, i32 3
   %v5 = insertelement <4 x i32> undef, i32 %a2, i32 0
   %v6 = insertelement <4 x i32> undef, i32 %a2, i32 1
   %v7 = insertelement <4 x i32> undef, i32 %a2, i32 2
   %v8 = insertelement <4 x i32> undef, i32 %a2, i32 3
   %v9 = insertelement <4 x i32> undef, i32 %a3, i32 0
   %v10 = insertelement <4 x i32> undef, i32 %a3, i32 1
   %v11 = insertelement <4 x i32> undef, i32 %a3, i32 2
   %v12 = insertelement <4 x i32> undef, i32 %a3, i32 3
   %v13 = insertelement <4 x i32> undef, i32 %a4, i32 0
   %v14 = insertelement <4 x i32> undef, i32 %a4, i32 1
   %v15 = insertelement <4 x i32> undef, i32 %a4, i32 2
   %v16 = insertelement <4 x i32> undef, i32 %a4, i32 3
   %res1 = call <4 x float> @llvm.SI.sampled.(<4 x i32> %v1,
      <32 x i8> undef, <16 x i8> undef, i32 1)
   %res2 = call <4 x float> @llvm.SI.sampled.(<4 x i32> %v2,
      <32 x i8> undef, <16 x i8> undef, i32 2)
   %res3 = call <4 x float> @llvm.SI.sampled.(<4 x i32> %v3,
      <32 x i8> undef, <16 x i8> undef, i32 3)
   %res4 = call <4 x float> @llvm.SI.sampled.(<4 x i32> %v4,
      <32 x i8> undef, <16 x i8> undef, i32 4)
   %res5 = call <4 x float> @llvm.SI.sampled.(<4 x i32> %v5,
      <32 x i8> undef, <16 x i8> undef, i32 5)
   %res6 = call <4 x float> @llvm.SI.sampled.(<4 x i32> %v6,
      <32 x i8> undef, <16 x i8> undef, i32 6)
   %res7 = call <4 x float> @llvm.SI.sampled.(<4 x i32> %v7,
      <32 x i8> undef, <16 x i8> undef, i32 7)
   %res8 = call <4 x float> @llvm.SI.sampled.(<4 x i32> %v8,
      <32 x i8> undef, <16 x i8> undef, i32 8)
   %res9 = call <4 x float> @llvm.SI.sampled.(<4 x i32> %v9,
      <32 x i8> undef, <16 x i8> undef, i32 9)
   %res10 = call <4 x float> @llvm.SI.sampled.(<4 x i32> %v10,
      <32 x i8> undef, <16 x i8> undef, i32 10)
   %res11 = call <4 x float> @llvm.SI.sampled.(<4 x i32> %v11,
      <32 x i8> undef, <16 x i8> undef, i32 11)
   %res12 = call <4 x float> @llvm.SI.sampled.(<4 x i32> %v12,
      <32 x i8> undef, <16 x i8> undef, i32 12)
   %res13 = call <4 x float> @llvm.SI.sampled.(<4 x i32> %v13,
      <32 x i8> undef, <16 x i8> undef, i32 13)
   %res14 = call <4 x float> @llvm.SI.sampled.(<4 x i32> %v14,
      <32 x i8> undef, <16 x i8> undef, i32 14)
   %res15 = call <4 x float> @llvm.SI.sampled.(<4 x i32> %v15,
      <32 x i8> undef, <16 x i8> undef, i32 15)
   %res16 = call <4 x float> @llvm.SI.sampled.(<4 x i32> %v16,
      <32 x i8> undef, <16 x i8> undef, i32 16)
   %e1 = extractelement <4 x float> %res1, i32 0
   %e2 = extractelement <4 x float> %res2, i32 1
   %e3 = extractelement <4 x float> %res3, i32 2
   %e4 = extractelement <4 x float> %res4, i32 3
   %t0 = extractelement <4 x float> %res5, i32 0
   %t1 = extractelement <4 x float> %res5, i32 1
   %e5 = fadd float %t0, %t1
   %t2 = extractelement <4 x float> %res6, i32 0
   %t3 = extractelement <4 x float> %res6, i32 2
   %e6 = fadd float %t2, %t3
   %t4 = extractelement <4 x float> %res7, i32 0
   %t5 = extractelement <4 x float> %res7, i32 3
   %e7 = fadd float %t4, %t5
   %t6 = extractelement <4 x float> %res8, i32 1
   %t7 = extractelement <4 x float> %res8, i32 2
   %e8 = fadd float %t6, %t7
   %t8 = extractelement <4 x float> %res9, i32 1
   %t9 = extractelement <4 x float> %res9, i32 3
   %e9 = fadd float %t8, %t9
   %t10 = extractelement <4 x float> %res10, i32 2
   %t11 = extractelement <4 x float> %res10, i32 3
   %e10 = fadd float %t10, %t11
   %t12 = extractelement <4 x float> %res11, i32 0
   %t13 = extractelement <4 x float> %res11, i32 1
   %t14 = extractelement <4 x float> %res11, i32 2
   %t15 = fadd float %t12, %t13
   %e11 = fadd float %t14, %t15
   %t16 = extractelement <4 x float> %res12, i32 0
   %t17 = extractelement <4 x float> %res12, i32 1
   %t18 = extractelement <4 x float> %res12, i32 3
   %t19 = fadd float %t16, %t17
   %e12 = fadd float %t18, %t19
   %t20 = extractelement <4 x float> %res13, i32 0
   %t21 = extractelement <4 x float> %res13, i32 2
   %t22 = extractelement <4 x float> %res13, i32 3
   %t23 = fadd float %t20, %t21
   %e13 = fadd float %t22, %t23
   %t24 = extractelement <4 x float> %res14, i32 1
   %t25 = extractelement <4 x float> %res14, i32 2
   %t26 = extractelement <4 x float> %res14, i32 3
   %t27 = fadd float %t24, %t25
   %e14 = fadd float %t26, %t27
   %t28 = extractelement <4 x float> %res15, i32 0
   %t29 = extractelement <4 x float> %res15, i32 1
   %t30 = extractelement <4 x float> %res15, i32 2
   %t31 = extractelement <4 x float> %res15, i32 3
   %t32 = fadd float %t28, %t29
   %t33 = fadd float %t30, %t31
   %e15 = fadd float %t32, %t33
   %e16 = extractelement <4 x float> %res16, i32 3
   %s1 = fadd float %e1, %e2
   %s2 = fadd float %s1, %e3
   %s3 = fadd float %s2, %e4
   %s4 = fadd float %s3, %e5
   %s5 = fadd float %s4, %e6
   %s6 = fadd float %s5, %e7
   %s7 = fadd float %s6, %e8
   %s8 = fadd float %s7, %e9
   %s9 = fadd float %s8, %e10
   %s10 = fadd float %s9, %e11
   %s11 = fadd float %s10, %e12
   %s12 = fadd float %s11, %e13
   %s13 = fadd float %s12, %e14
   %s14 = fadd float %s13, %e15
   %s15 = fadd float %s14, %e16
   call void @llvm.SI.export(i32 15, i32 0, i32 1, i32 12, i32 0, float %s15, float %s15, float %s15, float %s15)
   ret void
}

declare <4 x float> @llvm.SI.sampled.(<4 x i32>, <32 x i8>, <16 x i8>, i32) readnone

declare void @llvm.SI.export(i32, i32, i32, i32, i32, float, float, float, float)

attributes #0 = { "ShaderType"="0" }
