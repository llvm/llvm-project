; RUN: llc -march=hexagon -filetype=obj < %s -o - | llvm-objdump -d - | FileCheck %s

; CHECK-LABEL: f0:
; CHECK: r{{[1-9]}}:{{[0-9]}} = abs(r{{[1-9]}}:{{[0-9]}})
define double @f0(double %a0) #0 {
b0:
  %v0 = alloca double, align 8
  store double %a0, double* %v0, align 8
  %v1 = load double, double* %v0, align 8
  %v2 = fptosi double %v1 to i64
  %v3 = call i64 @llvm.hexagon.A2.absp(i64 %v2)
  %v4 = sitofp i64 %v3 to double
  ret double %v4
}

declare i64 @llvm.hexagon.A2.absp(i64) #1

; CHECK-LABEL: f1:
; CHECK: r{{[1-9]}}:{{[0-9]}} = neg(r{{[1-9]}}:{{[0-9]}})
define double @f1(double %a0) #0 {
b0:
  %v0 = alloca double, align 8
  store double %a0, double* %v0, align 8
  %v1 = load double, double* %v0, align 8
  %v2 = fptosi double %v1 to i64
  %v3 = call i64 @llvm.hexagon.A2.negp(i64 %v2)
  %v4 = sitofp i64 %v3 to double
  ret double %v4
}

declare i64 @llvm.hexagon.A2.negp(i64) #1

; CHECK-LABEL: f2:
; CHECK: r{{[1-9]}}:{{[0-9]}} = not(r{{[1-9]}}:{{[0-9]}})
define double @f2(double %a0) #0 {
b0:
  %v0 = alloca double, align 8
  store double %a0, double* %v0, align 8
  %v1 = load double, double* %v0, align 8
  %v2 = fptosi double %v1 to i64
  %v3 = call i64 @llvm.hexagon.A2.notp(i64 %v2)
  %v4 = sitofp i64 %v3 to double
  ret double %v4
}

declare i64 @llvm.hexagon.A2.notp(i64) #1

; CHECK-LABEL: f3:
; CHECK: r{{[1-9]}}:{{[0-9]}} = interleave(r{{[1-9]}}:{{[0-9]}})
define double @f3(double %a0) #0 {
b0:
  %v0 = alloca double, align 8
  store double %a0, double* %v0, align 8
  %v1 = load double, double* %v0, align 8
  %v2 = fptosi double %v1 to i64
  %v3 = call i64 @llvm.hexagon.S2.interleave(i64 %v2)
  %v4 = sitofp i64 %v3 to double
  ret double %v4
}

declare i64 @llvm.hexagon.S2.interleave(i64) #1

; CHECK-LABEL: f4:
; CHECK: r{{[1-9]}}:{{[0-9]}} = deinterleave(r{{[1-9]}}:{{[0-9]}})
define double @f4(double %a0) #0 {
b0:
  %v0 = alloca double, align 8
  store double %a0, double* %v0, align 8
  %v1 = load double, double* %v0, align 8
  %v2 = fptosi double %v1 to i64
  %v3 = call i64 @llvm.hexagon.S2.deinterleave(i64 %v2)
  %v4 = sitofp i64 %v3 to double
  ret double %v4
}

declare i64 @llvm.hexagon.S2.deinterleave(i64) #1

; CHECK-LABEL: f5:
; CHECK: r{{[1-9]}}:{{[0-9]}} = vconj(r{{[1-9]}}:{{[0-9]}}):sat
define double @f5(double %a0) #0 {
b0:
  %v0 = alloca double, align 8
  store double %a0, double* %v0, align 8
  %v1 = load double, double* %v0, align 8
  %v2 = fptosi double %v1 to i64
  %v3 = call i64 @llvm.hexagon.A2.vconj(i64 %v2)
  %v4 = sitofp i64 %v3 to double
  ret double %v4
}

declare i64 @llvm.hexagon.A2.vconj(i64) #1

; CHECK-LABEL: f6:
; CHECK: r{{[1-9]}}:{{[0-9]}} = vsathb(r{{[1-9]}}:{{[0-9]}})
define double @f6(double %a0) #0 {
b0:
  %v0 = alloca double, align 8
  store double %a0, double* %v0, align 8
  %v1 = load double, double* %v0, align 8
  %v2 = fptosi double %v1 to i64
  %v3 = call i64 @llvm.hexagon.S2.vsathb.nopack(i64 %v2)
  %v4 = sitofp i64 %v3 to double
  ret double %v4
}

declare i64 @llvm.hexagon.S2.vsathb.nopack(i64) #1

; CHECK-LABEL: f7:
; CHECK: r{{[1-9]}}:{{[0-9]}} = vsathub(r{{[1-9]}}:{{[0-9]}})
define double @f7(double %a0) #0 {
b0:
  %v0 = alloca double, align 8
  store double %a0, double* %v0, align 8
  %v1 = load double, double* %v0, align 8
  %v2 = fptosi double %v1 to i64
  %v3 = call i64 @llvm.hexagon.S2.vsathub.nopack(i64 %v2)
  %v4 = sitofp i64 %v3 to double
  ret double %v4
}

declare i64 @llvm.hexagon.S2.vsathub.nopack(i64) #1

; CHECK-LABEL: f8:
; CHECK: r{{[1-9]}}:{{[0-9]}} = vsatwh(r{{[1-9]}}:{{[0-9]}})
define double @f8(double %a0) #0 {
b0:
  %v0 = alloca double, align 8
  store double %a0, double* %v0, align 8
  %v1 = load double, double* %v0, align 8
  %v2 = fptosi double %v1 to i64
  %v3 = call i64 @llvm.hexagon.S2.vsatwh.nopack(i64 %v2)
  %v4 = sitofp i64 %v3 to double
  ret double %v4
}

declare i64 @llvm.hexagon.S2.vsatwh.nopack(i64) #1

; CHECK-LABEL: f9:
; CHECK: r{{[1-9]}}:{{[0-9]}} = vsatwuh(r{{[1-9]}}:{{[0-9]}})
define double @f9(double %a0) #0 {
b0:
  %v0 = alloca double, align 8
  store double %a0, double* %v0, align 8
  %v1 = load double, double* %v0, align 8
  %v2 = fptosi double %v1 to i64
  %v3 = call i64 @llvm.hexagon.S2.vsatwuh.nopack(i64 %v2)
  %v4 = sitofp i64 %v3 to double
  ret double %v4
}

declare i64 @llvm.hexagon.S2.vsatwuh.nopack(i64) #1

; CHECK-LABEL: f10:
; CHECK: r{{[1-9]}}:{{[0-9]}} = asr(r{{[1-9]}}:{{[0-9]}},#1)
define double @f10(double %a0) #0 {
b0:
  %v0 = alloca double, align 8
  store double %a0, double* %v0, align 8
  %v1 = load double, double* %v0, align 8
  %v2 = fptosi double %v1 to i64
  %v3 = call i64 @llvm.hexagon.S2.asr.i.p(i64 %v2, i32 1)
  %v4 = sitofp i64 %v3 to double
  ret double %v4
}

declare i64 @llvm.hexagon.S2.asr.i.p(i64, i32) #1

; CHECK-LABEL: f11:
; CHECK: r{{[1-9]}}:{{[0-9]}} = lsr(r{{[1-9]}}:{{[0-9]}},#1)
define double @f11(double %a0) #0 {
b0:
  %v0 = alloca double, align 8
  store double %a0, double* %v0, align 8
  %v1 = load double, double* %v0, align 8
  %v2 = fptosi double %v1 to i64
  %v3 = call i64 @llvm.hexagon.S2.lsr.i.p(i64 %v2, i32 1)
  %v4 = sitofp i64 %v3 to double
  ret double %v4
}

declare i64 @llvm.hexagon.S2.lsr.i.p(i64, i32) #1

; CHECK-LABEL: f12:
; CHECK: r{{[1-9]}}:{{[0-9]}} = asl(r{{[1-9]}}:{{[0-9]}},#1)
define double @f12(double %a0) #0 {
b0:
  %v0 = alloca double, align 8
  store double %a0, double* %v0, align 8
  %v1 = load double, double* %v0, align 8
  %v2 = fptosi double %v1 to i64
  %v3 = call i64 @llvm.hexagon.S2.asl.i.p(i64 %v2, i32 1)
  %v4 = sitofp i64 %v3 to double
  ret double %v4
}

declare i64 @llvm.hexagon.S2.asl.i.p(i64, i32) #1

; CHECK-LABEL: f13:
; CHECK: r{{[1-9]}}:{{[0-9]}} = vabsh(r{{[1-9]}}:{{[0-9]}})
define double @f13(double %a0) #0 {
b0:
  %v0 = alloca double, align 8
  store double %a0, double* %v0, align 8
  %v1 = load double, double* %v0, align 8
  %v2 = fptosi double %v1 to i64
  %v3 = call i64 @llvm.hexagon.A2.vabsh(i64 %v2)
  %v4 = sitofp i64 %v3 to double
  ret double %v4
}

declare i64 @llvm.hexagon.A2.vabsh(i64) #1

; CHECK-LABEL: f14:
; CHECK: r{{[1-9]}}:{{[0-9]}} = vabsh(r{{[1-9]}}:{{[0-9]}}):sat
define double @f14(double %a0) #0 {
b0:
  %v0 = alloca double, align 8
  store double %a0, double* %v0, align 8
  %v1 = load double, double* %v0, align 8
  %v2 = fptosi double %v1 to i64
  %v3 = call i64 @llvm.hexagon.A2.vabshsat(i64 %v2)
  %v4 = sitofp i64 %v3 to double
  ret double %v4
}

declare i64 @llvm.hexagon.A2.vabshsat(i64) #1

; CHECK-LABEL: f15:
; CHECK: r{{[0-9]}}:{{[0-9]}} = vasrh(r{{[1-9]}}:{{[0-9]}},#1)
define double @f15(double %a0) #0 {
b0:
  %v0 = alloca double, align 8
  store double %a0, double* %v0, align 8
  %v1 = load double, double* %v0, align 8
  %v2 = fptosi double %v1 to i64
  %v3 = call i64 @llvm.hexagon.S2.asr.i.vh(i64 %v2, i32 1)
  %v4 = sitofp i64 %v3 to double
  ret double %v4
}

declare i64 @llvm.hexagon.S2.asr.i.vh(i64, i32) #1

; CHECK-LABEL: f16:
; CHECK: r{{[1-9]}}:{{[0-9]}} = vlsrh(r{{[1-9]}}:{{[0-9]}},#1)
define double @f16(double %a0) #0 {
b0:
  %v0 = alloca double, align 8
  store double %a0, double* %v0, align 8
  %v1 = load double, double* %v0, align 8
  %v2 = fptosi double %v1 to i64
  %v3 = call i64 @llvm.hexagon.S2.lsr.i.vh(i64 %v2, i32 1)
  %v4 = sitofp i64 %v3 to double
  ret double %v4
}

declare i64 @llvm.hexagon.S2.lsr.i.vh(i64, i32) #1

; CHECK-LABEL: f17:
; CHECK: r{{[1-9]}}:{{[0-9]}} = vaslh(r{{[1-9]}}:{{[0-9]}},#1)
define double @f17(double %a0) #0 {
b0:
  %v0 = alloca double, align 8
  store double %a0, double* %v0, align 8
  %v1 = load double, double* %v0, align 8
  %v2 = fptosi double %v1 to i64
  %v3 = call i64 @llvm.hexagon.S2.asl.i.vh(i64 %v2, i32 1)
  %v4 = sitofp i64 %v3 to double
  ret double %v4
}

declare i64 @llvm.hexagon.S2.asl.i.vh(i64, i32) #1

; CHECK-LABEL: f18:
; CHECK: r{{[1-9]}}:{{[0-9]}} = vabsw(r{{[1-9]}}:{{[0-9]}})
define double @f18(double %a0) #0 {
b0:
  %v0 = alloca double, align 8
  store double %a0, double* %v0, align 8
  %v1 = load double, double* %v0, align 8
  %v2 = fptosi double %v1 to i64
  %v3 = call i64 @llvm.hexagon.A2.vabsw(i64 %v2)
  %v4 = sitofp i64 %v3 to double
  ret double %v4
}

declare i64 @llvm.hexagon.A2.vabsw(i64) #1

; CHECK-LABEL: f19:
; CHECK: r{{[1-9]}}:{{[0-9]}} = vabsw(r{{[1-9]}}:{{[0-9]}}):sat
define double @f19(double %a0) #0 {
b0:
  %v0 = alloca double, align 8
  store double %a0, double* %v0, align 8
  %v1 = load double, double* %v0, align 8
  %v2 = fptosi double %v1 to i64
  %v3 = call i64 @llvm.hexagon.A2.vabswsat(i64 %v2)
  %v4 = sitofp i64 %v3 to double
  ret double %v4
}

declare i64 @llvm.hexagon.A2.vabswsat(i64) #1

; CHECK-LABEL: f20:
; CHECK: r{{[1-9]}}:{{[0-9]}} = vasrw(r{{[1-9]}}:{{[0-9]}},#1)
define double @f20(double %a0) #0 {
b0:
  %v0 = alloca double, align 8
  store double %a0, double* %v0, align 8
  %v1 = load double, double* %v0, align 8
  %v2 = fptosi double %v1 to i64
  %v3 = call i64 @llvm.hexagon.S2.asr.i.vw(i64 %v2, i32 1)
  %v4 = sitofp i64 %v3 to double
  ret double %v4
}

declare i64 @llvm.hexagon.S2.asr.i.vw(i64, i32) #1

; CHECK-LABEL: f21:
; CHECK: r{{[1-9]}}:{{[0-9]}} = vlsrw(r{{[1-9]}}:{{[0-9]}},#1)
define double @f21(double %a0) #0 {
b0:
  %v0 = alloca double, align 8
  store double %a0, double* %v0, align 8
  %v1 = load double, double* %v0, align 8
  %v2 = fptosi double %v1 to i64
  %v3 = call i64 @llvm.hexagon.S2.lsr.i.vw(i64 %v2, i32 1)
  %v4 = sitofp i64 %v3 to double
  ret double %v4
}

declare i64 @llvm.hexagon.S2.lsr.i.vw(i64, i32) #1

; CHECK-LABEL: f22:
; CHECK: r{{[1-9]}}:{{[0-9]}} = vaslw(r{{[1-9]}}:{{[0-9]}},#1)
define double @f22(double %a0) #0 {
b0:
  %v0 = alloca double, align 8
  store double %a0, double* %v0, align 8
  %v1 = load double, double* %v0, align 8
  %v2 = fptosi double %v1 to i64
  %v3 = call i64 @llvm.hexagon.S2.asl.i.vw(i64 %v2, i32 1)
  %v4 = sitofp i64 %v3 to double
  ret double %v4
}

declare i64 @llvm.hexagon.S2.asl.i.vw(i64, i32) #1

; CHECK-LABEL: f23:
; CHECK: r{{[1-9]}}:{{[0-9]}} = brev(r{{[1-9]}}:{{[0-9]}})
define double @f23(double %a0) #0 {
b0:
  %v0 = alloca double, align 8
  store double %a0, double* %v0, align 8
  %v1 = load double, double* %v0, align 8
  %v2 = fptosi double %v1 to i64
  %v3 = call i64 @llvm.hexagon.S2.brevp(i64 %v2)
  %v4 = sitofp i64 %v3 to double
  ret double %v4
}

declare i64 @llvm.hexagon.S2.brevp(i64) #1

attributes #0 = { nounwind }
attributes #1 = { nounwind readnone }
