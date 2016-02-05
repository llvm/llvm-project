; RUN: llc < %s -march=aarch64 -aarch64-neon-syntax=apple -aarch64-stp-suppress=false -verify-machineinstrs -asm-verbose=false | FileCheck %s

; CHECK-LABEL: test_strd_sturd:
; CHECK-NEXT: stp d0, d1, [x0, #-8]
; CHECK-NEXT: ret
define void @test_strd_sturd(float* %ptr, <2 x float> %v1, <2 x float> %v2) #0 {
  %tmp1 = bitcast float* %ptr to <2 x float>*
  store <2 x float> %v2, <2 x float>* %tmp1, align 16
  %add.ptr = getelementptr inbounds float, float* %ptr, i64 -2
  %tmp = bitcast float* %add.ptr to <2 x float>*
  store <2 x float> %v1, <2 x float>* %tmp, align 16
  ret void
}

; CHECK-LABEL: test_sturd_strd:
; CHECK-NEXT: stp d0, d1, [x0, #-8]
; CHECK-NEXT: ret
define void @test_sturd_strd(float* %ptr, <2 x float> %v1, <2 x float> %v2) #0 {
  %add.ptr = getelementptr inbounds float, float* %ptr, i64 -2
  %tmp = bitcast float* %add.ptr to <2 x float>*
  store <2 x float> %v1, <2 x float>* %tmp, align 16
  %tmp1 = bitcast float* %ptr to <2 x float>*
  store <2 x float> %v2, <2 x float>* %tmp1, align 16
  ret void
}

; CHECK-LABEL: test_strq_sturq:
; CHECK-NEXT: stp q0, q1, [x0, #-16]
; CHECK-NEXT: ret
define void @test_strq_sturq(double* %ptr, <2 x double> %v1, <2 x double> %v2) #0 {
  %tmp1 = bitcast double* %ptr to <2 x double>*
  store <2 x double> %v2, <2 x double>* %tmp1, align 16
  %add.ptr = getelementptr inbounds double, double* %ptr, i64 -2
  %tmp = bitcast double* %add.ptr to <2 x double>*
  store <2 x double> %v1, <2 x double>* %tmp, align 16
  ret void
}

; CHECK-LABEL: test_sturq_strq:
; CHECK-NEXT: stp q0, q1, [x0, #-16]
; CHECK-NEXT: ret
define void @test_sturq_strq(double* %ptr, <2 x double> %v1, <2 x double> %v2) #0 {
  %add.ptr = getelementptr inbounds double, double* %ptr, i64 -2
  %tmp = bitcast double* %add.ptr to <2 x double>*
  store <2 x double> %v1, <2 x double>* %tmp, align 16
  %tmp1 = bitcast double* %ptr to <2 x double>*
  store <2 x double> %v2, <2 x double>* %tmp1, align 16
  ret void
}

; CHECK-LABEL: test_ldrx_ldurx:
; CHECK-NEXT: ldp [[V0:x[0-9]+]], [[V1:x[0-9]+]], [x0, #-8]
; CHECK-NEXT: add x0, [[V0]], [[V1]]
; CHECK-NEXT: ret
define i64 @test_ldrx_ldurx(i64* %p) #0 {
  %tmp = load i64, i64* %p, align 4
  %add.ptr = getelementptr inbounds i64, i64* %p, i64 -1
  %tmp1 = load i64, i64* %add.ptr, align 4
  %add = add nsw i64 %tmp1, %tmp
  ret i64 %add
}

; CHECK-LABEL: test_ldurx_ldrx:
; CHECK-NEXT: ldp [[V0:x[0-9]+]], [[V1:x[0-9]+]], [x0, #-8]
; CHECK-NEXT: add x0, [[V0]], [[V1]]
; CHECK-NEXT: ret
define i64 @test_ldurx_ldrx(i64* %p) #0 {
  %add.ptr = getelementptr inbounds i64, i64* %p, i64 -1
  %tmp1 = load i64, i64* %add.ptr, align 4
  %tmp = load i64, i64* %p, align 4
  %add = add nsw i64 %tmp1, %tmp
  ret i64 %add
}

; CHECK-LABEL: test_ldrsw_ldursw:
; CHECK-NEXT: ldpsw [[V0:x[0-9]+]], [[V1:x[0-9]+]], [x0, #-4]
; CHECK-NEXT: add x0, [[V0]], [[V1]]
; CHECK-NEXT: ret
define i64 @test_ldrsw_ldursw(i32* %p) #0 {
  %tmp = load i32, i32* %p, align 4
  %add.ptr = getelementptr inbounds i32, i32* %p, i64 -1
  %tmp1 = load i32, i32* %add.ptr, align 4
  %sexttmp = sext i32 %tmp to i64
  %sexttmp1 = sext i32 %tmp1 to i64
  %add = add nsw i64 %sexttmp1, %sexttmp
  ret i64 %add
}

; Also make sure we only match valid offsets.
; CHECK-LABEL: test_ldrq_ldruq_invalidoffset:
; CHECK-NEXT: ldr q[[V0:[0-9]+]], [x0]
; CHECK-NEXT: ldur q[[V1:[0-9]+]], [x0, #24]
; CHECK-NEXT: add.2d v0, v[[V0]], v[[V1]]
; CHECK-NEXT: ret
define <2 x i64> @test_ldrq_ldruq_invalidoffset(i64* %p) nounwind {
  %a1 = bitcast i64* %p to <2 x i64>*
  %tmp1 = load <2 x i64>, < 2 x i64>* %a1, align 8
  %add.ptr2 = getelementptr inbounds i64, i64* %p, i64 3
  %a2 = bitcast i64* %add.ptr2 to <2 x i64>*
  %tmp2 = load <2 x i64>, <2 x i64>* %a2, align 8
  %add = add nsw <2 x i64> %tmp1, %tmp2
  ret <2 x i64> %add
}

attributes #0 = { nounwind }
