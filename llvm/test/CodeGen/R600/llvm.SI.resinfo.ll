; RUN: llc -march=amdgcn -mcpu=verde -verify-machineinstrs < %s | FileCheck %s
; RUN: llc -march=amdgcn -mcpu=tonga -verify-machineinstrs < %s | FileCheck %s

; CHECK-DAG: image_get_resinfo {{v\[[0-9]+:[0-9]+\]}}, 15, 0, 0, -1
; CHECK-DAG: image_get_resinfo {{v\[[0-9]+:[0-9]+\]}}, 3, 0, 0, 0
; CHECK-DAG: image_get_resinfo {{v[0-9]+}}, 2, 0, 0, 0
; CHECK-DAG: image_get_resinfo {{v[0-9]+}}, 1, 0, 0, 0
; CHECK-DAG: image_get_resinfo {{v[0-9]+}}, 4, 0, 0, 0
; CHECK-DAG: image_get_resinfo {{v[0-9]+}}, 8, 0, 0, 0
; CHECK-DAG: image_get_resinfo {{v\[[0-9]+:[0-9]+\]}}, 5, 0, 0, 0
; CHECK-DAG: image_get_resinfo {{v\[[0-9]+:[0-9]+\]}}, 9, 0, 0, 0
; CHECK-DAG: image_get_resinfo {{v\[[0-9]+:[0-9]+\]}}, 6, 0, 0, 0
; CHECK-DAG: image_get_resinfo {{v\[[0-9]+:[0-9]+\]}}, 10, 0, 0, -1
; CHECK-DAG: image_get_resinfo {{v\[[0-9]+:[0-9]+\]}}, 12, 0, 0, -1
; CHECK-DAG: image_get_resinfo {{v\[[0-9]+:[0-9]+\]}}, 7, 0, 0, 0
; CHECK-DAG: image_get_resinfo {{v\[[0-9]+:[0-9]+\]}}, 11, 0, 0, 0
; CHECK-DAG: image_get_resinfo {{v\[[0-9]+:[0-9]+\]}}, 13, 0, 0, 0
; CHECK-DAG: image_get_resinfo {{v\[[0-9]+:[0-9]+\]}}, 14, 0, 0, 0
; CHECK-DAG: image_get_resinfo {{v[0-9]+}}, 8, 0, 0, -1

define void @test(i32 %a1, i32 %a2, i32 %a3, i32 %a4, i32 %a5, i32 %a6, i32 %a7, i32 %a8,
		  i32 %a9, i32 %a10, i32 %a11, i32 %a12, i32 %a13, i32 %a14, i32 %a15, i32 %a16) {
   %res1 = call <4 x i32> @llvm.SI.resinfo(i32 %a1, <32 x i8> undef, i32 1)
   %res2 = call <4 x i32> @llvm.SI.resinfo(i32 %a2, <32 x i8> undef, i32 2)
   %res3 = call <4 x i32> @llvm.SI.resinfo(i32 %a3, <32 x i8> undef, i32 3)
   %res4 = call <4 x i32> @llvm.SI.resinfo(i32 %a4, <32 x i8> undef, i32 4)
   %res5 = call <4 x i32> @llvm.SI.resinfo(i32 %a5, <32 x i8> undef, i32 5)
   %res6 = call <4 x i32> @llvm.SI.resinfo(i32 %a6, <32 x i8> undef, i32 6)
   %res7 = call <4 x i32> @llvm.SI.resinfo(i32 %a7, <32 x i8> undef, i32 7)
   %res8 = call <4 x i32> @llvm.SI.resinfo(i32 %a8, <32 x i8> undef, i32 8)
   %res9 = call <4 x i32> @llvm.SI.resinfo(i32 %a9, <32 x i8> undef, i32 9)
   %res10 = call <4 x i32> @llvm.SI.resinfo(i32 %a10, <32 x i8> undef, i32 10)
   %res11 = call <4 x i32> @llvm.SI.resinfo(i32 %a11, <32 x i8> undef, i32 11)
   %res12 = call <4 x i32> @llvm.SI.resinfo(i32 %a12, <32 x i8> undef, i32 12)
   %res13 = call <4 x i32> @llvm.SI.resinfo(i32 %a13, <32 x i8> undef, i32 13)
   %res14 = call <4 x i32> @llvm.SI.resinfo(i32 %a14, <32 x i8> undef, i32 14)
   %res15 = call <4 x i32> @llvm.SI.resinfo(i32 %a15, <32 x i8> undef, i32 15)
   %res16 = call <4 x i32> @llvm.SI.resinfo(i32 %a16, <32 x i8> undef, i32 16)
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
   %t4 = extractelement <4 x i32> %res7, i32 0
   %t5 = extractelement <4 x i32> %res7, i32 3
   %e7 = add i32 %t4, %t5
   %t6 = extractelement <4 x i32> %res8, i32 1
   %t7 = extractelement <4 x i32> %res8, i32 2
   %e8 = add i32 %t6, %t7
   %t8 = extractelement <4 x i32> %res9, i32 1
   %t9 = extractelement <4 x i32> %res9, i32 3
   %e9 = add i32 %t8, %t9
   %t10 = extractelement <4 x i32> %res10, i32 2
   %t11 = extractelement <4 x i32> %res10, i32 3
   %e10 = add i32 %t10, %t11
   %t12 = extractelement <4 x i32> %res11, i32 0
   %t13 = extractelement <4 x i32> %res11, i32 1
   %t14 = extractelement <4 x i32> %res11, i32 2
   %t15 = add i32 %t12, %t13
   %e11 = add i32 %t14, %t15
   %t16 = extractelement <4 x i32> %res12, i32 0
   %t17 = extractelement <4 x i32> %res12, i32 1
   %t18 = extractelement <4 x i32> %res12, i32 3
   %t19 = add i32 %t16, %t17
   %e12 = add i32 %t18, %t19
   %t20 = extractelement <4 x i32> %res13, i32 0
   %t21 = extractelement <4 x i32> %res13, i32 2
   %t22 = extractelement <4 x i32> %res13, i32 3
   %t23 = add i32 %t20, %t21
   %e13 = add i32 %t22, %t23
   %t24 = extractelement <4 x i32> %res14, i32 1
   %t25 = extractelement <4 x i32> %res14, i32 2
   %t26 = extractelement <4 x i32> %res14, i32 3
   %t27 = add i32 %t24, %t25
   %e14 = add i32 %t26, %t27
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
   %s6 = add i32 %s5, %e7
   %s7 = add i32 %s6, %e8
   %s8 = add i32 %s7, %e9
   %s9 = add i32 %s8, %e10
   %s10 = add i32 %s9, %e11
   %s11 = add i32 %s10, %e12
   %s12 = add i32 %s11, %e13
   %s13 = add i32 %s12, %e14
   %s14 = add i32 %s13, %e15
   %s15 = add i32 %s14, %e16
   %s16 = bitcast i32 %s15 to float
   call void @llvm.SI.export(i32 15, i32 0, i32 1, i32 12, i32 0, float %s16, float %s16, float %s16, float %s16)
   ret void
}

declare <4 x i32> @llvm.SI.resinfo(i32, <32 x i8>, i32) readnone

declare void @llvm.SI.export(i32, i32, i32, i32, i32, float, float, float, float)
