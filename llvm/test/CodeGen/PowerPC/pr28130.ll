; RUN: llc -verify-machineinstrs -O0 < %s | FileCheck %s
target triple = "powerpc64le-unknown-linux-gnu"

%StructA = type { double, double, double, double, double, double, double, double }

define void @Test(ptr %tmp) unnamed_addr #0 align 2 {
; CHECK-LABEL: Test:
; CHECK: lxvd2x
; CHECK-NEXT: xxswapd
; CHECK: lxvd2x
; CHECK-NEXT: xxswapd
; CHECK: lxvd2x
; CHECK-NEXT: xxswapd
; CHECK: lxvd2x
; CHECK-NEXT: xxswapd
; CHECK: xxswapd [[OUTPUT:[0-9]+]]
; CHECK-NEXT: stxvd2x [[OUTPUT]]
bb:
  %tmp5 = getelementptr inbounds %StructA, ptr %tmp, i64 0, i32 2
  %tmp9 = getelementptr inbounds %StructA, ptr %tmp, i64 0, i32 4
  %tmp11 = getelementptr inbounds %StructA, ptr %tmp, i64 0, i32 5
  %tmp13 = getelementptr inbounds %StructA, ptr %tmp, i64 0, i32 6
  %tmp15 = getelementptr inbounds %StructA, ptr %tmp, i64 0, i32 7
  %tmp18 = load double, ptr %tmp, align 16
  %tmp19 = load double, ptr %tmp11, align 8
  %tmp20 = load double, ptr %tmp9, align 16
  %tmp21 = fsub double 1.210000e+04, %tmp20
  %tmp22 = fmul double %tmp18, %tmp21
  %tmp23 = fadd double %tmp20, %tmp22
  %tmp24 = load double, ptr %tmp13, align 16
  %tmp25 = fsub double 1.000000e+02, %tmp24
  %tmp26 = fmul double %tmp18, %tmp25
  %tmp27 = fadd double %tmp24, %tmp26
  %tmp28 = load double, ptr %tmp15, align 8
  %tmp29 = insertelement <2 x double> undef, double %tmp19, i32 0
  %tmp30 = insertelement <2 x double> %tmp29, double %tmp28, i32 1
  %tmp31 = fsub <2 x double> <double 1.100000e+04, double 1.100000e+02>, %tmp30
  %tmp32 = insertelement <2 x double> undef, double %tmp18, i32 0
  %tmp33 = insertelement <2 x double> %tmp32, double %tmp18, i32 1
  %tmp34 = fmul <2 x double> %tmp33, %tmp31
  %tmp35 = fadd <2 x double> %tmp30, %tmp34
  %tmp37 = load <2 x double>, ptr %tmp5, align 16
  %tmp38 = fsub <2 x double> <double 1.000000e+00, double 1.000000e+04>, %tmp37
  %tmp39 = fmul <2 x double> %tmp33, %tmp38
  %tmp40 = fadd <2 x double> %tmp37, %tmp39
  %tmp41 = fsub <2 x double> <double 1.000000e+00, double 1.000000e+04>, %tmp40
  %tmp42 = fmul <2 x double> %tmp33, %tmp41
  %tmp43 = fadd <2 x double> %tmp40, %tmp42
  %tmp44 = fsub <2 x double> <double 1.200000e+04, double 1.200000e+02>, %tmp35
  %tmp45 = fmul <2 x double> %tmp33, %tmp44
  %tmp46 = fadd <2 x double> %tmp35, %tmp45
  %tmp48 = fsub double 1.440000e+04, %tmp23
  %tmp49 = fmul double %tmp18, %tmp48
  %tmp50 = fadd double %tmp23, %tmp49
  store double %tmp50, ptr %tmp9, align 16
  %tmp51 = fsub double 1.000000e+02, %tmp27
  %tmp52 = fmul double %tmp18, %tmp51
  %tmp53 = fadd double %tmp27, %tmp52
  store double %tmp53, ptr %tmp13, align 16
  %tmp54 = extractelement <2 x double> %tmp46, i32 1
  store double %tmp54, ptr %tmp15, align 8
  store <2 x double> %tmp43, ptr %tmp5, align 16
  ret void
}

attributes #0 = { nounwind "disable-tail-calls"="false" "less-precise-fpmad"="false" "frame-pointer"="all" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="pwr8" "target-features"="+altivec,+bpermd,+crypto,+direct-move,+extdiv,+power8-vector,+vsx" "unsafe-fp-math"="false" "use-soft-float"="false" }
