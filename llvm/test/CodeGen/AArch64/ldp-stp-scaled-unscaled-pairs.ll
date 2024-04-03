; RUN: llc < %s -mtriple=aarch64 -aarch64-neon-syntax=apple -aarch64-enable-stp-suppress=false -verify-machineinstrs -asm-verbose=false | FileCheck %s

; CHECK-LABEL: test_strd_sturd:
; CHECK-NEXT: stp d0, d1, [x0, #-8]
; CHECK-NEXT: ret
define void @test_strd_sturd(ptr %ptr, <2 x float> %v1, <2 x float> %v2) #0 {
  store <2 x float> %v2, ptr %ptr, align 16
  %add.ptr = getelementptr inbounds float, ptr %ptr, i64 -2
  store <2 x float> %v1, ptr %add.ptr, align 16
  ret void
}

; CHECK-LABEL: test_sturd_strd:
; CHECK-NEXT: stp d0, d1, [x0, #-8]
; CHECK-NEXT: ret
define void @test_sturd_strd(ptr %ptr, <2 x float> %v1, <2 x float> %v2) #0 {
  %add.ptr = getelementptr inbounds float, ptr %ptr, i64 -2
  store <2 x float> %v1, ptr %add.ptr, align 16
  store <2 x float> %v2, ptr %ptr, align 16
  ret void
}

; CHECK-LABEL: test_strq_sturq:
; CHECK-NEXT: stp q0, q1, [x0, #-16]
; CHECK-NEXT: ret
define void @test_strq_sturq(ptr %ptr, <2 x double> %v1, <2 x double> %v2) #0 {
  store <2 x double> %v2, ptr %ptr, align 16
  %add.ptr = getelementptr inbounds double, ptr %ptr, i64 -2
  store <2 x double> %v1, ptr %add.ptr, align 16
  ret void
}

; CHECK-LABEL: test_sturq_strq:
; CHECK-NEXT: stp q0, q1, [x0, #-16]
; CHECK-NEXT: ret
define void @test_sturq_strq(ptr %ptr, <2 x double> %v1, <2 x double> %v2) #0 {
  %add.ptr = getelementptr inbounds double, ptr %ptr, i64 -2
  store <2 x double> %v1, ptr %add.ptr, align 16
  store <2 x double> %v2, ptr %ptr, align 16
  ret void
}

; CHECK-LABEL: test_ldrx_ldurx:
; CHECK-NEXT: ldp [[V0:x[0-9]+]], [[V1:x[0-9]+]], [x0, #-8]
; CHECK-NEXT: add x0, [[V0]], [[V1]]
; CHECK-NEXT: ret
define i64 @test_ldrx_ldurx(ptr %p) #0 {
  %tmp = load i64, ptr %p, align 4
  %add.ptr = getelementptr inbounds i64, ptr %p, i64 -1
  %tmp1 = load i64, ptr %add.ptr, align 4
  %add = add nsw i64 %tmp1, %tmp
  ret i64 %add
}

; CHECK-LABEL: test_ldurx_ldrx:
; CHECK-NEXT: ldp [[V0:x[0-9]+]], [[V1:x[0-9]+]], [x0, #-8]
; CHECK-NEXT: add x0, [[V0]], [[V1]]
; CHECK-NEXT: ret
define i64 @test_ldurx_ldrx(ptr %p) #0 {
  %add.ptr = getelementptr inbounds i64, ptr %p, i64 -1
  %tmp1 = load i64, ptr %add.ptr, align 4
  %tmp = load i64, ptr %p, align 4
  %add = add nsw i64 %tmp1, %tmp
  ret i64 %add
}

; CHECK-LABEL: test_ldrsw_ldursw:
; CHECK-NEXT: ldpsw [[V0:x[0-9]+]], [[V1:x[0-9]+]], [x0, #-4]
; CHECK-NEXT: add x0, [[V0]], [[V1]]
; CHECK-NEXT: ret
define i64 @test_ldrsw_ldursw(ptr %p) #0 {
  %tmp = load i32, ptr %p, align 4
  %add.ptr = getelementptr inbounds i32, ptr %p, i64 -1
  %tmp1 = load i32, ptr %add.ptr, align 4
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
define <2 x i64> @test_ldrq_ldruq_invalidoffset(ptr %p) #0 {
  %tmp1 = load <2 x i64>, < 2 x i64>* %p, align 8
  %add.ptr2 = getelementptr inbounds i64, ptr %p, i64 3
  %tmp2 = load <2 x i64>, ptr %add.ptr2, align 8
  %add = add nsw <2 x i64> %tmp1, %tmp2
  ret <2 x i64> %add
}

; Pair an unscaled store with a scaled store where the scaled store has a
; non-zero offset.  This should not hit an assert.
; CHECK-LABEL: test_stur_str_no_assert
; CHECK: stp xzr, xzr, [sp, #16]
; CHECK: ret
define void @test_stur_str_no_assert() #0 {
entry:
  %a1 = alloca i64, align 4
  %a2 = alloca [12 x i8], align 4
  %C = getelementptr inbounds [12 x i8], ptr %a2, i64 0, i64 4
  store i64 0, ptr %C, align 4
  call void @llvm.memset.p0.i64(ptr align 8 %a1, i8 0, i64 8, i1 false)
  ret void
}

declare void @llvm.memset.p0.i64(ptr nocapture, i8, i64, i1)


attributes #0 = { nounwind }
