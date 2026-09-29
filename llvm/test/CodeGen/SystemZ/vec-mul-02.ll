; Test vector multiply-and-add.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z13 | FileCheck %s

declare <2 x double> @llvm.fma.v2f64(<2 x double>, <2 x double>, <2 x double>)

; Test a v16i8 multiply-and-add.
define <16 x i8> @f1(<16 x i8> %dummy, <16 x i8> %val1, <16 x i8> %val2,
                     <16 x i8> %val3) {
; CHECK-LABEL: f1:
; CHECK: vmalb %v24, %v26, %v28, %v30
; CHECK: br %r14
  %mul = mul <16 x i8> %val1, %val2
  %ret = add <16 x i8> %mul, %val3
  ret <16 x i8> %ret
}

; Test a v8i16 multiply-and-add.
define <8 x i16> @f2(<8 x i16> %dummy, <8 x i16> %val1, <8 x i16> %val2,
                     <8 x i16> %val3) {
; CHECK-LABEL: f2:
; CHECK: vmalhw %v24, %v26, %v28, %v30
; CHECK: br %r14
  %mul = mul <8 x i16> %val1, %val2
  %ret = add <8 x i16> %mul, %val3
  ret <8 x i16> %ret
}

; Test a v4i32 multiply-and-add.
define <4 x i32> @f3(<4 x i32> %dummy, <4 x i32> %val1, <4 x i32> %val2,
                     <4 x i32> %val3) {
; CHECK-LABEL: f3:
; CHECK: vmalf %v24, %v26, %v28, %v30
; CHECK: br %r14
  %mul = mul <4 x i32> %val1, %val2
  %ret = add <4 x i32> %mul, %val3
  ret <4 x i32> %ret
}

; Test a v2f64 multiply-and-add.
define <2 x double> @f4(<2 x double> %dummy, <2 x double> %val1,
                        <2 x double> %val2, <2 x double> %val3) {
; CHECK-LABEL: f4:
; CHECK: vfmadb %v24, %v26, %v28, %v30
; CHECK: br %r14
  %ret = call <2 x double> @llvm.fma.v2f64 (<2 x double> %val1,
                                            <2 x double> %val2,
                                            <2 x double> %val3)
  ret <2 x double> %ret
}

; Test a v2f64 multiply-and-subtract.
define <2 x double> @f5(<2 x double> %dummy, <2 x double> %val1,
                        <2 x double> %val2, <2 x double> %val3) {
; CHECK-LABEL: f5:
; CHECK: vfmsdb %v24, %v26, %v28, %v30
; CHECK: br %r14
  %negval3 = fneg <2 x double> %val3
  %ret = call <2 x double> @llvm.fma.v2f64 (<2 x double> %val1,
                                            <2 x double> %val2,
                                            <2 x double> %negval3)
  ret <2 x double> %ret
}
