; Test vector insertion of memory values.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z13 | FileCheck %s

; Test v16i8 insertion into the first element.
define <16 x i8> @f1(<16 x i8> %val, ptr %ptr) {
; CHECK-LABEL: f1:
; CHECK: vleb %v24, 0(%r2), 0
; CHECK: br %r14
  %element = load i8, ptr %ptr
  %ret = insertelement <16 x i8> %val, i8 %element, i32 0
  ret <16 x i8> %ret
}

; Test v16i8 insertion into the last element.
define <16 x i8> @f2(<16 x i8> %val, ptr %ptr) {
; CHECK-LABEL: f2:
; CHECK: vleb %v24, 0(%r2), 15
; CHECK: br %r14
  %element = load i8, ptr %ptr
  %ret = insertelement <16 x i8> %val, i8 %element, i32 15
  ret <16 x i8> %ret
}

; Test v16i8 insertion with the highest in-range offset.
define <16 x i8> @f3(<16 x i8> %val, ptr %base) {
; CHECK-LABEL: f3:
; CHECK: vleb %v24, 4095(%r2), 10
; CHECK: br %r14
  %ptr = getelementptr i8, ptr %base, i32 4095
  %element = load i8, ptr %ptr
  %ret = insertelement <16 x i8> %val, i8 %element, i32 10
  ret <16 x i8> %ret
}

; Test v16i8 insertion with the first ouf-of-range offset.
define <16 x i8> @f4(<16 x i8> %val, ptr %base) {
; CHECK-LABEL: f4:
; CHECK: aghi %r2, 4096
; CHECK: vleb %v24, 0(%r2), 5
; CHECK: br %r14
  %ptr = getelementptr i8, ptr %base, i32 4096
  %element = load i8, ptr %ptr
  %ret = insertelement <16 x i8> %val, i8 %element, i32 5
  ret <16 x i8> %ret
}

; Test v16i8 insertion into a variable element.
define <16 x i8> @f5(<16 x i8> %val, ptr %ptr, i32 %index) {
; CHECK-LABEL: f5:
; CHECK-NOT: vleb
; CHECK: br %r14
  %element = load i8, ptr %ptr
  %ret = insertelement <16 x i8> %val, i8 %element, i32 %index
  ret <16 x i8> %ret
}

; Test v8i16 insertion into the first element.
define <8 x i16> @f6(<8 x i16> %val, ptr %ptr) {
; CHECK-LABEL: f6:
; CHECK: vleh %v24, 0(%r2), 0
; CHECK: br %r14
  %element = load i16, ptr %ptr
  %ret = insertelement <8 x i16> %val, i16 %element, i32 0
  ret <8 x i16> %ret
}

; Test v8i16 insertion into the last element.
define <8 x i16> @f7(<8 x i16> %val, ptr %ptr) {
; CHECK-LABEL: f7:
; CHECK: vleh %v24, 0(%r2), 7
; CHECK: br %r14
  %element = load i16, ptr %ptr
  %ret = insertelement <8 x i16> %val, i16 %element, i32 7
  ret <8 x i16> %ret
}

; Test v8i16 insertion with the highest in-range offset.
define <8 x i16> @f8(<8 x i16> %val, ptr %base) {
; CHECK-LABEL: f8:
; CHECK: vleh %v24, 4094(%r2), 5
; CHECK: br %r14
  %ptr = getelementptr i16, ptr %base, i32 2047
  %element = load i16, ptr %ptr
  %ret = insertelement <8 x i16> %val, i16 %element, i32 5
  ret <8 x i16> %ret
}

; Test v8i16 insertion with the first ouf-of-range offset.
define <8 x i16> @f9(<8 x i16> %val, ptr %base) {
; CHECK-LABEL: f9:
; CHECK: aghi %r2, 4096
; CHECK: vleh %v24, 0(%r2), 1
; CHECK: br %r14
  %ptr = getelementptr i16, ptr %base, i32 2048
  %element = load i16, ptr %ptr
  %ret = insertelement <8 x i16> %val, i16 %element, i32 1
  ret <8 x i16> %ret
}

; Test v8i16 insertion into a variable element.
define <8 x i16> @f10(<8 x i16> %val, ptr %ptr, i32 %index) {
; CHECK-LABEL: f10:
; CHECK-NOT: vleh
; CHECK: br %r14
  %element = load i16, ptr %ptr
  %ret = insertelement <8 x i16> %val, i16 %element, i32 %index
  ret <8 x i16> %ret
}

; Test v4i32 insertion into the first element.
define <4 x i32> @f11(<4 x i32> %val, ptr %ptr) {
; CHECK-LABEL: f11:
; CHECK: vlef %v24, 0(%r2), 0
; CHECK: br %r14
  %element = load i32, ptr %ptr
  %ret = insertelement <4 x i32> %val, i32 %element, i32 0
  ret <4 x i32> %ret
}

; Test v4i32 insertion into the last element.
define <4 x i32> @f12(<4 x i32> %val, ptr %ptr) {
; CHECK-LABEL: f12:
; CHECK: vlef %v24, 0(%r2), 3
; CHECK: br %r14
  %element = load i32, ptr %ptr
  %ret = insertelement <4 x i32> %val, i32 %element, i32 3
  ret <4 x i32> %ret
}

; Test v4i32 insertion with the highest in-range offset.
define <4 x i32> @f13(<4 x i32> %val, ptr %base) {
; CHECK-LABEL: f13:
; CHECK: vlef %v24, 4092(%r2), 2
; CHECK: br %r14
  %ptr = getelementptr i32, ptr %base, i32 1023
  %element = load i32, ptr %ptr
  %ret = insertelement <4 x i32> %val, i32 %element, i32 2
  ret <4 x i32> %ret
}

; Test v4i32 insertion with the first ouf-of-range offset.
define <4 x i32> @f14(<4 x i32> %val, ptr %base) {
; CHECK-LABEL: f14:
; CHECK: aghi %r2, 4096
; CHECK: vlef %v24, 0(%r2), 1
; CHECK: br %r14
  %ptr = getelementptr i32, ptr %base, i32 1024
  %element = load i32, ptr %ptr
  %ret = insertelement <4 x i32> %val, i32 %element, i32 1
  ret <4 x i32> %ret
}

; Test v4i32 insertion into a variable element.
define <4 x i32> @f15(<4 x i32> %val, ptr %ptr, i32 %index) {
; CHECK-LABEL: f15:
; CHECK-NOT: vlef
; CHECK: br %r14
  %element = load i32, ptr %ptr
  %ret = insertelement <4 x i32> %val, i32 %element, i32 %index
  ret <4 x i32> %ret
}

; Test v2i64 insertion into the first element.
define <2 x i64> @f16(<2 x i64> %val, ptr %ptr) {
; CHECK-LABEL: f16:
; CHECK: vleg %v24, 0(%r2), 0
; CHECK: br %r14
  %element = load i64, ptr %ptr
  %ret = insertelement <2 x i64> %val, i64 %element, i32 0
  ret <2 x i64> %ret
}

; Test v2i64 insertion into the last element.
define <2 x i64> @f17(<2 x i64> %val, ptr %ptr) {
; CHECK-LABEL: f17:
; CHECK: vleg %v24, 0(%r2), 1
; CHECK: br %r14
  %element = load i64, ptr %ptr
  %ret = insertelement <2 x i64> %val, i64 %element, i32 1
  ret <2 x i64> %ret
}

; Test v2i64 insertion with the highest in-range offset.
define <2 x i64> @f18(<2 x i64> %val, ptr %base) {
; CHECK-LABEL: f18:
; CHECK: vleg %v24, 4088(%r2), 1
; CHECK: br %r14
  %ptr = getelementptr i64, ptr %base, i32 511
  %element = load i64, ptr %ptr
  %ret = insertelement <2 x i64> %val, i64 %element, i32 1
  ret <2 x i64> %ret
}

; Test v2i64 insertion with the first ouf-of-range offset.
define <2 x i64> @f19(<2 x i64> %val, ptr %base) {
; CHECK-LABEL: f19:
; CHECK: aghi %r2, 4096
; CHECK: vleg %v24, 0(%r2), 0
; CHECK: br %r14
  %ptr = getelementptr i64, ptr %base, i32 512
  %element = load i64, ptr %ptr
  %ret = insertelement <2 x i64> %val, i64 %element, i32 0
  ret <2 x i64> %ret
}

; Test v2i64 insertion into a variable element.
define <2 x i64> @f20(<2 x i64> %val, ptr %ptr, i32 %index) {
; CHECK-LABEL: f20:
; CHECK-NOT: vleg
; CHECK: br %r14
  %element = load i64, ptr %ptr
  %ret = insertelement <2 x i64> %val, i64 %element, i32 %index
  ret <2 x i64> %ret
}

; Test v4f32 insertion into the first element.
define <4 x float> @f21(<4 x float> %val, ptr %ptr) {
; CHECK-LABEL: f21:
; CHECK: vlef %v24, 0(%r2), 0
; CHECK: br %r14
  %element = load float, ptr %ptr
  %ret = insertelement <4 x float> %val, float %element, i32 0
  ret <4 x float> %ret
}

; Test v4f32 insertion into the last element.
define <4 x float> @f22(<4 x float> %val, ptr %ptr) {
; CHECK-LABEL: f22:
; CHECK: vlef %v24, 0(%r2), 3
; CHECK: br %r14
  %element = load float, ptr %ptr
  %ret = insertelement <4 x float> %val, float %element, i32 3
  ret <4 x float> %ret
}

; Test v4f32 insertion with the highest in-range offset.
define <4 x float> @f23(<4 x float> %val, ptr %base) {
; CHECK-LABEL: f23:
; CHECK: vlef %v24, 4092(%r2), 2
; CHECK: br %r14
  %ptr = getelementptr float, ptr %base, i32 1023
  %element = load float, ptr %ptr
  %ret = insertelement <4 x float> %val, float %element, i32 2
  ret <4 x float> %ret
}

; Test v4f32 insertion with the first ouf-of-range offset.
define <4 x float> @f24(<4 x float> %val, ptr %base) {
; CHECK-LABEL: f24:
; CHECK: aghi %r2, 4096
; CHECK: vlef %v24, 0(%r2), 1
; CHECK: br %r14
  %ptr = getelementptr float, ptr %base, i32 1024
  %element = load float, ptr %ptr
  %ret = insertelement <4 x float> %val, float %element, i32 1
  ret <4 x float> %ret
}

; Test v4f32 insertion into a variable element.
define <4 x float> @f25(<4 x float> %val, ptr %ptr, i32 %index) {
; CHECK-LABEL: f25:
; CHECK-NOT: vlef
; CHECK: br %r14
  %element = load float, ptr %ptr
  %ret = insertelement <4 x float> %val, float %element, i32 %index
  ret <4 x float> %ret
}

; Test v2f64 insertion into the first element.
define <2 x double> @f26(<2 x double> %val, ptr %ptr) {
; CHECK-LABEL: f26:
; CHECK: vleg %v24, 0(%r2), 0
; CHECK: br %r14
  %element = load double, ptr %ptr
  %ret = insertelement <2 x double> %val, double %element, i32 0
  ret <2 x double> %ret
}

; Test v2f64 insertion into the last element.
define <2 x double> @f27(<2 x double> %val, ptr %ptr) {
; CHECK-LABEL: f27:
; CHECK: vleg %v24, 0(%r2), 1
; CHECK: br %r14
  %element = load double, ptr %ptr
  %ret = insertelement <2 x double> %val, double %element, i32 1
  ret <2 x double> %ret
}

; Test v2f64 insertion with the highest in-range offset.
define <2 x double> @f28(<2 x double> %val, ptr %base) {
; CHECK-LABEL: f28:
; CHECK: vleg %v24, 4088(%r2), 1
; CHECK: br %r14
  %ptr = getelementptr double, ptr %base, i32 511
  %element = load double, ptr %ptr
  %ret = insertelement <2 x double> %val, double %element, i32 1
  ret <2 x double> %ret
}

; Test v2f64 insertion with the first ouf-of-range offset.
define <2 x double> @f29(<2 x double> %val, ptr %base) {
; CHECK-LABEL: f29:
; CHECK: aghi %r2, 4096
; CHECK: vleg %v24, 0(%r2), 0
; CHECK: br %r14
  %ptr = getelementptr double, ptr %base, i32 512
  %element = load double, ptr %ptr
  %ret = insertelement <2 x double> %val, double %element, i32 0
  ret <2 x double> %ret
}

; Test v2f64 insertion into a variable element.
define <2 x double> @f30(<2 x double> %val, ptr %ptr, i32 %index) {
; CHECK-LABEL: f30:
; CHECK-NOT: vleg
; CHECK: br %r14
  %element = load double, ptr %ptr
  %ret = insertelement <2 x double> %val, double %element, i32 %index
  ret <2 x double> %ret
}

; Test a v4i32 gather of the first element.
define <4 x i32> @f31(<4 x i32> %val, <4 x i32> %index, i64 %base) {
; CHECK-LABEL: f31:
; CHECK: vgef %v24, 0(%v26,%r2), 0
; CHECK: br %r14
  %elem = extractelement <4 x i32> %index, i32 0
  %ext = zext i32 %elem to i64
  %add = add i64 %base, %ext
  %ptr = inttoptr i64 %add to ptr
  %element = load i32, ptr %ptr
  %ret = insertelement <4 x i32> %val, i32 %element, i32 0
  ret <4 x i32> %ret
}

; Test a v4i32 gather of the last element.
define <4 x i32> @f32(<4 x i32> %val, <4 x i32> %index, i64 %base) {
; CHECK-LABEL: f32:
; CHECK: vgef %v24, 0(%v26,%r2), 3
; CHECK: br %r14
  %elem = extractelement <4 x i32> %index, i32 3
  %ext = zext i32 %elem to i64
  %add = add i64 %base, %ext
  %ptr = inttoptr i64 %add to ptr
  %element = load i32, ptr %ptr
  %ret = insertelement <4 x i32> %val, i32 %element, i32 3
  ret <4 x i32> %ret
}

; Test a v4i32 gather with the highest in-range offset.
define <4 x i32> @f33(<4 x i32> %val, <4 x i32> %index, i64 %base) {
; CHECK-LABEL: f33:
; CHECK: vgef %v24, 4095(%v26,%r2), 1
; CHECK: br %r14
  %elem = extractelement <4 x i32> %index, i32 1
  %ext = zext i32 %elem to i64
  %add1 = add i64 %base, %ext
  %add2 = add i64 %add1, 4095
  %ptr = inttoptr i64 %add2 to ptr
  %element = load i32, ptr %ptr
  %ret = insertelement <4 x i32> %val, i32 %element, i32 1
  ret <4 x i32> %ret
}

; Test a v2i64 gather of the first element.
define <2 x i64> @f34(<2 x i64> %val, <2 x i64> %index, i64 %base) {
; CHECK-LABEL: f34:
; CHECK: vgeg %v24, 0(%v26,%r2), 0
; CHECK: br %r14
  %elem = extractelement <2 x i64> %index, i32 0
  %add = add i64 %base, %elem
  %ptr = inttoptr i64 %add to ptr
  %element = load i64, ptr %ptr
  %ret = insertelement <2 x i64> %val, i64 %element, i32 0
  ret <2 x i64> %ret
}

; Test a v2i64 gather of the last element.
define <2 x i64> @f35(<2 x i64> %val, <2 x i64> %index, i64 %base) {
; CHECK-LABEL: f35:
; CHECK: vgeg %v24, 0(%v26,%r2), 1
; CHECK: br %r14
  %elem = extractelement <2 x i64> %index, i32 1
  %add = add i64 %base, %elem
  %ptr = inttoptr i64 %add to ptr
  %element = load i64, ptr %ptr
  %ret = insertelement <2 x i64> %val, i64 %element, i32 1
  ret <2 x i64> %ret
}

; Test a v4f32 gather of the first element.
define <4 x float> @f36(<4 x float> %val, <4 x i32> %index, i64 %base) {
; CHECK-LABEL: f36:
; CHECK: vgef %v24, 0(%v26,%r2), 0
; CHECK: br %r14
  %elem = extractelement <4 x i32> %index, i32 0
  %ext = zext i32 %elem to i64
  %add = add i64 %base, %ext
  %ptr = inttoptr i64 %add to ptr
  %element = load float, ptr %ptr
  %ret = insertelement <4 x float> %val, float %element, i32 0
  ret <4 x float> %ret
}

; Test a v4f32 gather of the last element.
define <4 x float> @f37(<4 x float> %val, <4 x i32> %index, i64 %base) {
; CHECK-LABEL: f37:
; CHECK: vgef %v24, 0(%v26,%r2), 3
; CHECK: br %r14
  %elem = extractelement <4 x i32> %index, i32 3
  %ext = zext i32 %elem to i64
  %add = add i64 %base, %ext
  %ptr = inttoptr i64 %add to ptr
  %element = load float, ptr %ptr
  %ret = insertelement <4 x float> %val, float %element, i32 3
  ret <4 x float> %ret
}

; Test a v2f64 gather of the first element.
define <2 x double> @f38(<2 x double> %val, <2 x i64> %index, i64 %base) {
; CHECK-LABEL: f38:
; CHECK: vgeg %v24, 0(%v26,%r2), 0
; CHECK: br %r14
  %elem = extractelement <2 x i64> %index, i32 0
  %add = add i64 %base, %elem
  %ptr = inttoptr i64 %add to ptr
  %element = load double, ptr %ptr
  %ret = insertelement <2 x double> %val, double %element, i32 0
  ret <2 x double> %ret
}

; Test a v2f64 gather of the last element.
define <2 x double> @f39(<2 x double> %val, <2 x i64> %index, i64 %base) {
; CHECK-LABEL: f39:
; CHECK: vgeg %v24, 0(%v26,%r2), 1
; CHECK: br %r14
  %elem = extractelement <2 x i64> %index, i32 1
  %add = add i64 %base, %elem
  %ptr = inttoptr i64 %add to ptr
  %element = load double, ptr %ptr
  %ret = insertelement <2 x double> %val, double %element, i32 1
  ret <2 x double> %ret
}

; Test a v4i32 gather where the load is chained.
define void @f40(<4 x i32> %val, <4 x i32> %index, i64 %base, ptr %res) {
; CHECK-LABEL: f40:
; CHECK: vgef %v24, 0(%v26,%r2), 1
; CHECK: vst %v24, 0(%r3)
; CHECK: br %r14
  %elem = extractelement <4 x i32> %index, i32 1
  %ext = zext i32 %elem to i64
  %add = add i64 %base, %ext
  %ptr = inttoptr i64 %add to ptr
  %element = load i32, ptr %ptr
  %ret = insertelement <4 x i32> %val, i32 %element, i32 1
  store <4 x i32> %ret, ptr %res
  ret void
}

; Test a v2i64 gather where the load is chained.
define void @f41(<2 x i64> %val, <2 x i64> %index, i64 %base, ptr %res) {
; CHECK-LABEL: f41:
; CHECK: vgeg %v24, 0(%v26,%r2), 1
; CHECK: vst %v24, 0(%r3)
; CHECK: br %r14
  %elem = extractelement <2 x i64> %index, i32 1
  %add = add i64 %base, %elem
  %ptr = inttoptr i64 %add to ptr
  %element = load i64, ptr %ptr
  %ret = insertelement <2 x i64> %val, i64 %element, i32 1
  store <2 x i64> %ret, ptr %res
  ret void
}

