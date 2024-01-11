; REQUIRES: asserts
; RUN: llc < %s -mtriple=arm64-linux-gnu -mcpu=cortex-a57 -verify-misched -debug-only=machine-scheduler -o - 2>&1 > /dev/null | FileCheck %s
; RUN: llc < %s -mtriple=arm64-linux-gnu -mcpu=exynos-m3 -verify-misched -debug-only=machine-scheduler -o - 2>&1 > /dev/null | FileCheck %s

; Test ldr clustering.
; CHECK: ********** MI Scheduling **********
; CHECK-LABEL: ldr_int:%bb.0
; CHECK: Cluster ld/st SU(1) - SU(2)
; CHECK: SU(1):   %{{[0-9]+}}:gpr32 = LDRWui
; CHECK: SU(2):   %{{[0-9]+}}:gpr32 = LDRWui
define i32 @ldr_int(ptr %a) nounwind {
  %p1 = getelementptr inbounds i32, ptr %a, i32 1
  %tmp1 = load i32, ptr %p1, align 2
  %p2 = getelementptr inbounds i32, ptr %a, i32 2
  %tmp2 = load i32, ptr %p2, align 2
  %tmp3 = add i32 %tmp1, %tmp2
  ret i32 %tmp3
}

; Test ldpsw clustering
; CHECK: ********** MI Scheduling **********
; CHECK-LABEL: ldp_sext_int:%bb.0
; CHECK: Cluster ld/st SU(1) - SU(2)
; CHECK: SU(1):   %{{[0-9]+}}:gpr64 = LDRSWui
; CHECK: SU(2):   %{{[0-9]+}}:gpr64 = LDRSWui
define i64 @ldp_sext_int(ptr %p) nounwind {
  %tmp = load i32, ptr %p, align 4
  %add.ptr = getelementptr inbounds i32, ptr %p, i64 1
  %tmp1 = load i32, ptr %add.ptr, align 4
  %sexttmp = sext i32 %tmp to i64
  %sexttmp1 = sext i32 %tmp1 to i64
  %add = add nsw i64 %sexttmp1, %sexttmp
  ret i64 %add
}

; Test ldur clustering.
; CHECK: ********** MI Scheduling **********
; CHECK-LABEL: ldur_int:%bb.0
; CHECK: Cluster ld/st SU(1) - SU(2)
; CHECK: SU(1):   %{{[0-9]+}}:gpr32 = LDURWi
; CHECK: SU(2):   %{{[0-9]+}}:gpr32 = LDURWi
define i32 @ldur_int(ptr %a) nounwind {
  %p1 = getelementptr inbounds i32, ptr %a, i32 -1
  %tmp1 = load i32, ptr %p1, align 2
  %p2 = getelementptr inbounds i32, ptr %a, i32 -2
  %tmp2 = load i32, ptr %p2, align 2
  %tmp3 = add i32 %tmp1, %tmp2
  ret i32 %tmp3
}

; Test sext + zext clustering.
; CHECK: ********** MI Scheduling **********
; CHECK-LABEL: ldp_half_sext_zext_int:%bb.0
; CHECK: Cluster ld/st SU(3) - SU(4)
; CHECK: SU(3):   %{{[0-9]+}}:gpr64 = LDRSWui
; CHECK: SU(4):   undef %{{[0-9]+}}.sub_32:gpr64 = LDRWui
define i64 @ldp_half_sext_zext_int(ptr %q, ptr %p) nounwind {
  %tmp0 = load i64, ptr %q, align 4
  %tmp = load i32, ptr %p, align 4
  %add.ptr = getelementptr inbounds i32, ptr %p, i64 1
  %tmp1 = load i32, ptr %add.ptr, align 4
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
define i64 @ldp_half_zext_sext_int(ptr %q, ptr %p) nounwind {
  %tmp0 = load i64, ptr %q, align 4
  %tmp = load i32, ptr %p, align 4
  %add.ptr = getelementptr inbounds i32, ptr %p, i64 1
  %tmp1 = load i32, ptr %add.ptr, align 4
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
define i32 @ldr_int_volatile(ptr %a) nounwind {
  %p1 = getelementptr inbounds i32, ptr %a, i32 1
  %tmp1 = load volatile i32, ptr %p1, align 2
  %p2 = getelementptr inbounds i32, ptr %a, i32 2
  %tmp2 = load volatile i32, ptr %p2, align 2
  %tmp3 = add i32 %tmp1, %tmp2
  ret i32 %tmp3
}

; Test ldq clustering (no clustering for Exynos).
; CHECK: ********** MI Scheduling **********
; CHECK-LABEL: ldq_cluster:%bb.0
; CHECK: Cluster ld/st SU(1) - SU(3)
; CHECK: SU(1):   %{{[0-9]+}}:fpr128 = LDRQui
; CHECK: SU(3):   %{{[0-9]+}}:fpr128 = LDRQui
define <2 x i64> @ldq_cluster(ptr %p) {
  %tmp1 = load <2 x i64>, < 2 x i64>* %p, align 8
  %add.ptr2 = getelementptr inbounds i64, ptr %p, i64 2
  %tmp2 = add nsw <2 x i64> %tmp1, %tmp1
  %tmp3 = load <2 x i64>, ptr %add.ptr2, align 8
  %res  = mul nsw <2 x i64> %tmp2, %tmp3
  ret <2 x i64> %res
}

; Test LDURQi / LDRQui clustering
;
; CHECK: ********** MI Scheduling **********
; CHECK: LDURQi_LDRQui:%bb.1 vector_body
;
; CHECK: Cluster ld/st SU(0) - SU(4)
; CHECK: Cluster ld/st SU(1) - SU(5)
;
; CHECK: SU(0): %{{[0-9]+}}:fpr128 = LDURQi
; CHECK: SU(1): %{{[0-9]+}}:fpr128 = LDURQi
; CHECK: SU(4): %{{[0-9]+}}:fpr128 = LDRQui
; CHECK: SU(5): %{{[0-9]+}}:fpr128 = LDRQui
;
define void @LDURQi_LDRQui(ptr nocapture readonly %arg) {
entry:
  br label %vector_body
vector_body:
  %phi1 = phi ptr [ null, %entry ], [ %r63, %vector_body ]
  %phi2 = phi ptr [ %arg, %entry ], [ %r62, %vector_body ]
  %phi3 = phi i32 [ 0, %entry ], [ %r61, %vector_body ]
  %r51 = getelementptr i8, ptr %phi1, i64 -16
  %r52 = load <2 x double>, ptr %r51, align 8
  %r53 = getelementptr i8, ptr %phi2, i64 -16
  %r54 = load <2 x double>, ptr %r53, align 8
  %r55 = fmul fast <2 x double> %r54, <double 3.0, double 4.0>
  %r56 = fsub fast <2 x double> %r52, %r55
  store <2 x double> %r56, ptr %r51, align 1
  %r57 = load <2 x double>, ptr %phi1, align 8
  %r58 = load <2 x double>, ptr %phi2, align 8
  %r59 = fmul fast <2 x double> %r58,<double 3.0, double 4.0>
  %r60 = fsub fast <2 x double> %r57, %r59
  store <2 x double> %r60, ptr %phi1, align 1
  %r61 = add i32 %phi3, 4
  %r62 = getelementptr i8, ptr %phi2, i64 32
  %r63 = getelementptr i8, ptr %phi1, i64 32
  %r.not = icmp eq i32 %r61, 0
  br i1 %r.not, label %exit, label %vector_body
exit:
  ret void
}
