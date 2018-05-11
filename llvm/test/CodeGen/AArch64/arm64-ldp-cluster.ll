; REQUIRES: asserts
; RUN: llc < %s -mtriple=arm64-linux-gnu -mcpu=cortex-a57 -verify-misched -debug-only=machine-scheduler -o - 2>&1 > /dev/null | FileCheck %s
; RUN: llc < %s -mtriple=arm64-linux-gnu -mcpu=exynos-m1 -verify-misched -debug-only=machine-scheduler -o - 2>&1 > /dev/null | FileCheck --check-prefix=EXYNOSM1 %s
; RUN: llc < %s -mtriple=arm64-linux-gnu -mcpu=exynos-m3 -verify-misched -debug-only=machine-scheduler -o - 2>&1 > /dev/null | FileCheck %s

; Test ldr clustering.
; CHECK: ********** MI Scheduling **********
; CHECK-LABEL: ldr_int:%bb.0
; CHECK: Cluster ld/st SU(1) - SU(2)
; CHECK: SU(1):   %{{[0-9]+}}:gpr32 = LDRWui
; CHECK: SU(2):   %{{[0-9]+}}:gpr32 = LDRWui
; EXYNOSM1: ********** MI Scheduling **********
; EXYNOSM1-LABEL: ldr_int:%bb.0
; EXYNOSM1: Cluster ld/st SU(1) - SU(2)
; EXYNOSM1: SU(1):   %{{[0-9]+}}:gpr32 = LDRWui
; EXYNOSM1: SU(2):   %{{[0-9]+}}:gpr32 = LDRWui
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
; CHECK-LABEL: ldp_sext_int:%bb.0
; CHECK: Cluster ld/st SU(1) - SU(2)
; CHECK: SU(1):   %{{[0-9]+}}:gpr64 = LDRSWui
; CHECK: SU(2):   %{{[0-9]+}}:gpr64 = LDRSWui
; EXYNOSM1: ********** MI Scheduling **********
; EXYNOSM1-LABEL: ldp_sext_int:%bb.0
; EXYNOSM1: Cluster ld/st SU(1) - SU(2)
; EXYNOSM1: SU(1):   %{{[0-9]+}}:gpr64 = LDRSWui
; EXYNOSM1: SU(2):   %{{[0-9]+}}:gpr64 = LDRSWui
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
; CHECK-LABEL: ldur_int:%bb.0
; CHECK: Cluster ld/st SU(2) - SU(1)
; CHECK: SU(1):   %{{[0-9]+}}:gpr32 = LDURWi
; CHECK: SU(2):   %{{[0-9]+}}:gpr32 = LDURWi
; EXYNOSM1: ********** MI Scheduling **********
; EXYNOSM1-LABEL: ldur_int:%bb.0
; EXYNOSM1: Cluster ld/st SU(2) - SU(1)
; EXYNOSM1: SU(1):   %{{[0-9]+}}:gpr32 = LDURWi
; EXYNOSM1: SU(2):   %{{[0-9]+}}:gpr32 = LDURWi
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
; CHECK-LABEL: ldp_half_sext_zext_int:%bb.0
; CHECK: Cluster ld/st SU(3) - SU(4)
; CHECK: SU(3):   %{{[0-9]+}}:gpr64 = LDRSWui
; CHECK: SU(4):   undef %{{[0-9]+}}.sub_32:gpr64 = LDRWui
; EXYNOSM1: ********** MI Scheduling **********
; EXYNOSM1-LABEL: ldp_half_sext_zext_int:%bb.0
; EXYNOSM1: Cluster ld/st SU(3) - SU(4)
; EXYNOSM1: SU(3):   %{{[0-9]+}}:gpr64 = LDRSWui
; EXYNOSM1: SU(4):   undef %{{[0-9]+}}.sub_32:gpr64 = LDRWui
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
; CHECK-LABEL: ldp_half_zext_sext_int:%bb.0
; CHECK: Cluster ld/st SU(3) - SU(4)
; CHECK: SU(3):   undef %{{[0-9]+}}.sub_32:gpr64 = LDRWui
; CHECK: SU(4):   %{{[0-9]+}}:gpr64 = LDRSWui
; EXYNOSM1: ********** MI Scheduling **********
; EXYNOSM1-LABEL: ldp_half_zext_sext_int:%bb.0
; EXYNOSM1: Cluster ld/st SU(3) - SU(4)
; EXYNOSM1: SU(3):   undef %{{[0-9]+}}.sub_32:gpr64 = LDRWui
; EXYNOSM1: SU(4):   %{{[0-9]+}}:gpr64 = LDRSWui
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
; CHECK-LABEL: ldr_int_volatile:%bb.0
; CHECK-NOT: Cluster ld/st
; CHECK: SU(1):   %{{[0-9]+}}:gpr32 = LDRWui
; CHECK: SU(2):   %{{[0-9]+}}:gpr32 = LDRWui
; EXYNOSM1: ********** MI Scheduling **********
; EXYNOSM1-LABEL: ldr_int_volatile:%bb.0
; EXYNOSM1-NOT: Cluster ld/st
; EXYNOSM1: SU(1):   %{{[0-9]+}}:gpr32 = LDRWui
; EXYNOSM1: SU(2):   %{{[0-9]+}}:gpr32 = LDRWui
define i32 @ldr_int_volatile(i32* %a) nounwind {
  %p1 = getelementptr inbounds i32, i32* %a, i32 1
  %tmp1 = load volatile i32, i32* %p1, align 2
  %p2 = getelementptr inbounds i32, i32* %a, i32 2
  %tmp2 = load volatile i32, i32* %p2, align 2
  %tmp3 = add i32 %tmp1, %tmp2
  ret i32 %tmp3
}

; Test ldq clustering (no clustering for Exynos).
; CHECK: ********** MI Scheduling **********
; CHECK-LABEL: ldq_cluster:%bb.0
; CHECK: Cluster ld/st SU(1) - SU(3)
; CHECK: SU(1):   %{{[0-9]+}}:fpr128 = LDRQui
; CHECK: SU(3):   %{{[0-9]+}}:fpr128 = LDRQui
; EXYNOSM1: ********** MI Scheduling **********
; EXYNOSM1-LABEL: ldq_cluster:%bb.0
; EXYNOSM1-NOT: Cluster ld/st
define <2 x i64> @ldq_cluster(i64* %p) {
  %a1 = bitcast i64* %p to <2 x i64>*
  %tmp1 = load <2 x i64>, < 2 x i64>* %a1, align 8
  %add.ptr2 = getelementptr inbounds i64, i64* %p, i64 2
  %a2 = bitcast i64* %add.ptr2 to <2 x i64>*
  %tmp2 = add nsw <2 x i64> %tmp1, %tmp1
  %tmp3 = load <2 x i64>, <2 x i64>* %a2, align 8
  %res  = mul nsw <2 x i64> %tmp2, %tmp3
  ret <2 x i64> %res
}
