; REQUIRES: asserts
; RUN: llc < %s -mtriple=arm64-linux-gnu -mcpu=cortex-a57 -verify-misched -debug-only=misched -o - 2>&1 > /dev/null | FileCheck %s

; Test ldr clustering.
; CHECK: ********** MI Scheduling **********
; CHECK-LABEL: ldr_int:BB#0
; CHECK: Cluster loads SU(1) - SU(2)
; CHECK: SU(1):   %vreg{{[0-9]+}}<def> = LDRWui
; CHECK: SU(2):   %vreg{{[0-9]+}}<def> = LDRWui
define i32 @ldr_int(i32* %a) nounwind {
  %p1 = getelementptr inbounds i32, i32* %a, i32 1
  %tmp1 = load i32, i32* %p1, align 2
  %p2 = getelementptr inbounds i32, i32* %a, i32 2
  %tmp2 = load i32, i32* %p2, align 2
  %tmp3 = add i32 %tmp1, %tmp2
  ret i32 %tmp3
}

; Test ldpsw clustering
; CHECK: ********** MI Scheduling **********
; CHECK-LABEL: ldp_sext_int:BB#0
; CHECK: Cluster loads SU(1) - SU(2)
; CHECK: SU(1):   %vreg{{[0-9]+}}<def> = LDRSWui
; CHECK: SU(2):   %vreg{{[0-9]+}}<def> = LDRSWui
define i64 @ldp_sext_int(i32* %p) nounwind {
  %tmp = load i32, i32* %p, align 4
  %add.ptr = getelementptr inbounds i32, i32* %p, i64 1
  %tmp1 = load i32, i32* %add.ptr, align 4
  %sexttmp = sext i32 %tmp to i64
  %sexttmp1 = sext i32 %tmp1 to i64
  %add = add nsw i64 %sexttmp1, %sexttmp
  ret i64 %add
}

; Test ldur clustering.
; CHECK: ********** MI Scheduling **********
; CHECK-LABEL: ldur_int:BB#0
; CHECK: Cluster loads SU(2) - SU(1)
; CHECK: SU(1):   %vreg{{[0-9]+}}<def> = LDURWi
; CHECK: SU(2):   %vreg{{[0-9]+}}<def> = LDURWi
define i32 @ldur_int(i32* %a) nounwind {
  %p1 = getelementptr inbounds i32, i32* %a, i32 -1
  %tmp1 = load i32, i32* %p1, align 2
  %p2 = getelementptr inbounds i32, i32* %a, i32 -2
  %tmp2 = load i32, i32* %p2, align 2
  %tmp3 = add i32 %tmp1, %tmp2
  ret i32 %tmp3
}

; Test sext + zext clustering.
; CHECK: ********** MI Scheduling **********
; CHECK-LABEL: ldp_half_sext_zext_int:BB#0
; CHECK: Cluster loads SU(3) - SU(4)
; CHECK: SU(3):   %vreg{{[0-9]+}}<def> = LDRSWui
; CHECK: SU(4):   %vreg{{[0-9]+}}:sub_32<def,read-undef> = LDRWui
define i64 @ldp_half_sext_zext_int(i64* %q, i32* %p) nounwind {
  %tmp0 = load i64, i64* %q, align 4
  %tmp = load i32, i32* %p, align 4
  %add.ptr = getelementptr inbounds i32, i32* %p, i64 1
  %tmp1 = load i32, i32* %add.ptr, align 4
  %sexttmp = sext i32 %tmp to i64
  %sexttmp1 = zext i32 %tmp1 to i64
  %add = add nsw i64 %sexttmp1, %sexttmp
  %add1 = add nsw i64 %add, %tmp0
  ret i64 %add1
}

; Test zext + sext clustering.
; CHECK: ********** MI Scheduling **********
; CHECK-LABEL: ldp_half_zext_sext_int:BB#0
; CHECK: Cluster loads SU(3) - SU(4)
; CHECK: SU(3):   %vreg{{[0-9]+}}:sub_32<def,read-undef> = LDRWui
; CHECK: SU(4):   %vreg{{[0-9]+}}<def> = LDRSWui
define i64 @ldp_half_zext_sext_int(i64* %q, i32* %p) nounwind {
  %tmp0 = load i64, i64* %q, align 4
  %tmp = load i32, i32* %p, align 4
  %add.ptr = getelementptr inbounds i32, i32* %p, i64 1
  %tmp1 = load i32, i32* %add.ptr, align 4
  %sexttmp = zext i32 %tmp to i64
  %sexttmp1 = sext i32 %tmp1 to i64
  %add = add nsw i64 %sexttmp1, %sexttmp
  %add1 = add nsw i64 %add, %tmp0
  ret i64 %add1
}

; Verify we don't cluster volatile loads.
; CHECK: ********** MI Scheduling **********
; CHECK-LABEL: ldr_int_volatile:BB#0
; CHECK-NOT: Cluster loads
; CHECK: SU(1):   %vreg{{[0-9]+}}<def> = LDRWui
; CHECK: SU(2):   %vreg{{[0-9]+}}<def> = LDRWui
define i32 @ldr_int_volatile(i32* %a) nounwind {
  %p1 = getelementptr inbounds i32, i32* %a, i32 1
  %tmp1 = load volatile i32, i32* %p1, align 2
  %p2 = getelementptr inbounds i32, i32* %a, i32 2
  %tmp2 = load volatile i32, i32* %p2, align 2
  %tmp3 = add i32 %tmp1, %tmp2
  ret i32 %tmp3
}
