; RUN: llc -mtriple=hexagon --filetype=obj < %s -o - | llvm-objdump -d - | FileCheck %s

@g0 = common global double 0.000000e+00, align 8
@g1 = common global double 0.000000e+00, align 8

; CHECK-LABEL: <f0>:
; CHECK: r{{[0-9]}}:{{[0-9]}} += vrcmpys(r{{[0-9]}}:{{[0-9]}},r{{[0-9]}}:{{[0-9]}}):<<1:sat:raw:lo
define double @f0(i32 %a0, i32 %a1) {
b0:
  %v0 = load double, ptr @g0, align 8, !tbaa !0
  %v1 = fptosi double %v0 to i64
  %v2 = load double, ptr @g1, align 8, !tbaa !0
  %v3 = fptosi double %v2 to i64
  %v4 = tail call i64 @llvm.hexagon.M2.vrcmpys.acc.s1(i64 %v1, i64 %v3, i32 %a0)
  %v5 = sitofp i64 %v4 to double
  ret double %v5
}

; Function Attrs: nounwind readnone
declare i64 @llvm.hexagon.M2.vrcmpys.acc.s1(i64, i64, i32) #0

; CHECK-LABEL: <f1>:
; CHECK: r{{[0-9]}}:{{[0-9]}} += vrcmpys(r{{[0-9]}}:{{[0-9]}},r{{[0-9]}}:{{[0-9]}}):<<1:sat:raw:hi
define double @f1(i32 %a0, i32 %a1) {
b0:
  %v0 = load double, ptr @g0, align 8, !tbaa !0
  %v1 = fptosi double %v0 to i64
  %v2 = load double, ptr @g1, align 8, !tbaa !0
  %v3 = fptosi double %v2 to i64
  %v4 = tail call i64 @llvm.hexagon.M2.vrcmpys.acc.s1(i64 %v1, i64 %v3, i32 %a1)
  %v5 = sitofp i64 %v4 to double
  ret double %v5
}

; CHECK-LABEL: <f2>:
; CHECK: r{{[0-9]}}:{{[0-9]}} = vrcmpys(r{{[0-9]}}:{{[0-9]}},r{{[0-9]}}:{{[0-9]}}):<<1:sat:raw:lo
define double @f2(i32 %a0, i32 %a1) {
b0:
  %v0 = load double, ptr @g1, align 8, !tbaa !0
  %v1 = fptosi double %v0 to i64
  %v2 = tail call i64 @llvm.hexagon.M2.vrcmpys.s1(i64 %v1, i32 %a0)
  %v3 = sitofp i64 %v2 to double
  ret double %v3
}

; Function Attrs: nounwind readnone
declare i64 @llvm.hexagon.M2.vrcmpys.s1(i64, i32) #0

; CHECK-LABEL: <f3>:
; CHECK: r{{[0-9]}}:{{[0-9]}} = vrcmpys(r{{[0-9]}}:{{[0-9]}},r{{[0-9]}}:{{[0-9]}}):<<1:sat:raw:hi
define double @f3(i32 %a0, i32 %a1) {
b0:
  %v0 = load double, ptr @g1, align 8, !tbaa !0
  %v1 = fptosi double %v0 to i64
  %v2 = tail call i64 @llvm.hexagon.M2.vrcmpys.s1(i64 %v1, i32 %a1)
  %v3 = sitofp i64 %v2 to double
  ret double %v3
}

; CHECK-LABEL: <f4>:
; CHECK: e9a4c2e0 { r0 = vrcmpys(r5:4,r3:2):<<1:rnd:sat:raw:lo }
; CHECK: e9a4c2c0 { r0 = vrcmpys(r5:4,r3:2):<<1:rnd:sat:raw:hi }
define void @f4() {
b0:
  call void asm sideeffect "r0=vrcmpys(r5:4,r2):<<1:rnd:sat", ""(), !srcloc !4
  call void asm sideeffect "r0=vrcmpys(r5:4,r3):<<1:rnd:sat", ""(), !srcloc !5
  ret void
}

attributes #0 = { nounwind readnone }

!0 = !{!1, !1, i64 0}
!1 = !{!"double", !2, i64 0}
!2 = !{!"omnipotent char", !3, i64 0}
!3 = !{!"Simple C/C++ TBAA"}
!4 = !{i32 25}
!5 = !{i32 71}
