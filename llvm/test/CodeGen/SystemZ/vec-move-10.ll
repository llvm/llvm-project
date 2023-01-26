; Test vector extraction to memory.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z13 | FileCheck %s

; Test v16i8 extraction from the first element.
define void @f1(<16 x i8> %val, ptr %ptr) {
; CHECK-LABEL: f1:
; CHECK: vsteb %v24, 0(%r2), 0
; CHECK: br %r14
  %element = extractelement <16 x i8> %val, i32 0
  store i8 %element, ptr %ptr
  ret void
}

; Test v16i8 extraction from the last element.
define void @f2(<16 x i8> %val, ptr %ptr) {
; CHECK-LABEL: f2:
; CHECK: vsteb %v24, 0(%r2), 15
; CHECK: br %r14
  %element = extractelement <16 x i8> %val, i32 15
  store i8 %element, ptr %ptr
  ret void
}

; Test v16i8 extraction of an invalid element.  This must compile,
; but we don't care what it does.
define void @f3(<16 x i8> %val, ptr %ptr) {
; CHECK-LABEL: f3:
; CHECK-NOT: vsteb %v24, 0(%r2), 16
; CHECK: br %r14
  %element = extractelement <16 x i8> %val, i32 16
  store i8 %element, ptr %ptr
  ret void
}

; Test v16i8 extraction with the highest in-range offset.
define void @f4(<16 x i8> %val, ptr %base) {
; CHECK-LABEL: f4:
; CHECK: vsteb %v24, 4095(%r2), 10
; CHECK: br %r14
  %ptr = getelementptr i8, ptr %base, i32 4095
  %element = extractelement <16 x i8> %val, i32 10
  store i8 %element, ptr %ptr
  ret void
}

; Test v16i8 extraction with the first ouf-of-range offset.
define void @f5(<16 x i8> %val, ptr %base) {
; CHECK-LABEL: f5:
; CHECK: aghi %r2, 4096
; CHECK: vsteb %v24, 0(%r2), 5
; CHECK: br %r14
  %ptr = getelementptr i8, ptr %base, i32 4096
  %element = extractelement <16 x i8> %val, i32 5
  store i8 %element, ptr %ptr
  ret void
}

; Test v16i8 extraction from a variable element.
define void @f6(<16 x i8> %val, ptr %ptr, i32 %index) {
; CHECK-LABEL: f6:
; CHECK-NOT: vsteb
; CHECK: br %r14
  %element = extractelement <16 x i8> %val, i32 %index
  store i8 %element, ptr %ptr
  ret void
}

; Test v8i16 extraction from the first element.
define void @f7(<8 x i16> %val, ptr %ptr) {
; CHECK-LABEL: f7:
; CHECK: vsteh %v24, 0(%r2), 0
; CHECK: br %r14
  %element = extractelement <8 x i16> %val, i32 0
  store i16 %element, ptr %ptr
  ret void
}

; Test v8i16 extraction from the last element.
define void @f8(<8 x i16> %val, ptr %ptr) {
; CHECK-LABEL: f8:
; CHECK: vsteh %v24, 0(%r2), 7
; CHECK: br %r14
  %element = extractelement <8 x i16> %val, i32 7
  store i16 %element, ptr %ptr
  ret void
}

; Test v8i16 extraction of an invalid element.  This must compile,
; but we don't care what it does.
define void @f9(<8 x i16> %val, ptr %ptr) {
; CHECK-LABEL: f9:
; CHECK-NOT: vsteh %v24, 0(%r2), 8
; CHECK: br %r14
  %element = extractelement <8 x i16> %val, i32 8
  store i16 %element, ptr %ptr
  ret void
}

; Test v8i16 extraction with the highest in-range offset.
define void @f10(<8 x i16> %val, ptr %base) {
; CHECK-LABEL: f10:
; CHECK: vsteh %v24, 4094(%r2), 5
; CHECK: br %r14
  %ptr = getelementptr i16, ptr %base, i32 2047
  %element = extractelement <8 x i16> %val, i32 5
  store i16 %element, ptr %ptr
  ret void
}

; Test v8i16 extraction with the first ouf-of-range offset.
define void @f11(<8 x i16> %val, ptr %base) {
; CHECK-LABEL: f11:
; CHECK: aghi %r2, 4096
; CHECK: vsteh %v24, 0(%r2), 1
; CHECK: br %r14
  %ptr = getelementptr i16, ptr %base, i32 2048
  %element = extractelement <8 x i16> %val, i32 1
  store i16 %element, ptr %ptr
  ret void
}

; Test v8i16 extraction from a variable element.
define void @f12(<8 x i16> %val, ptr %ptr, i32 %index) {
; CHECK-LABEL: f12:
; CHECK-NOT: vsteh
; CHECK: br %r14
  %element = extractelement <8 x i16> %val, i32 %index
  store i16 %element, ptr %ptr
  ret void
}

; Test v4i32 extraction from the first element.
define void @f13(<4 x i32> %val, ptr %ptr) {
; CHECK-LABEL: f13:
; CHECK: vstef %v24, 0(%r2), 0
; CHECK: br %r14
  %element = extractelement <4 x i32> %val, i32 0
  store i32 %element, ptr %ptr
  ret void
}

; Test v4i32 extraction from the last element.
define void @f14(<4 x i32> %val, ptr %ptr) {
; CHECK-LABEL: f14:
; CHECK: vstef %v24, 0(%r2), 3
; CHECK: br %r14
  %element = extractelement <4 x i32> %val, i32 3
  store i32 %element, ptr %ptr
  ret void
}

; Test v4i32 extraction of an invalid element.  This must compile,
; but we don't care what it does.
define void @f15(<4 x i32> %val, ptr %ptr) {
; CHECK-LABEL: f15:
; CHECK-NOT: vstef %v24, 0(%r2), 4
; CHECK: br %r14
  %element = extractelement <4 x i32> %val, i32 4
  store i32 %element, ptr %ptr
  ret void
}

; Test v4i32 extraction with the highest in-range offset.
define void @f16(<4 x i32> %val, ptr %base) {
; CHECK-LABEL: f16:
; CHECK: vstef %v24, 4092(%r2), 2
; CHECK: br %r14
  %ptr = getelementptr i32, ptr %base, i32 1023
  %element = extractelement <4 x i32> %val, i32 2
  store i32 %element, ptr %ptr
  ret void
}

; Test v4i32 extraction with the first ouf-of-range offset.
define void @f17(<4 x i32> %val, ptr %base) {
; CHECK-LABEL: f17:
; CHECK: aghi %r2, 4096
; CHECK: vstef %v24, 0(%r2), 1
; CHECK: br %r14
  %ptr = getelementptr i32, ptr %base, i32 1024
  %element = extractelement <4 x i32> %val, i32 1
  store i32 %element, ptr %ptr
  ret void
}

; Test v4i32 extraction from a variable element.
define void @f18(<4 x i32> %val, ptr %ptr, i32 %index) {
; CHECK-LABEL: f18:
; CHECK-NOT: vstef
; CHECK: br %r14
  %element = extractelement <4 x i32> %val, i32 %index
  store i32 %element, ptr %ptr
  ret void
}

; Test v2i64 extraction from the first element.
define void @f19(<2 x i64> %val, ptr %ptr) {
; CHECK-LABEL: f19:
; CHECK: vsteg %v24, 0(%r2), 0
; CHECK: br %r14
  %element = extractelement <2 x i64> %val, i32 0
  store i64 %element, ptr %ptr
  ret void
}

; Test v2i64 extraction from the last element.
define void @f20(<2 x i64> %val, ptr %ptr) {
; CHECK-LABEL: f20:
; CHECK: vsteg %v24, 0(%r2), 1
; CHECK: br %r14
  %element = extractelement <2 x i64> %val, i32 1
  store i64 %element, ptr %ptr
  ret void
}

; Test v2i64 extraction of an invalid element.  This must compile,
; but we don't care what it does.
define void @f21(<2 x i64> %val, ptr %ptr) {
; CHECK-LABEL: f21:
; CHECK-NOT: vsteg %v24, 0(%r2), 2
; CHECK: br %r14
  %element = extractelement <2 x i64> %val, i32 2
  store i64 %element, ptr %ptr
  ret void
}

; Test v2i64 extraction with the highest in-range offset.
define void @f22(<2 x i64> %val, ptr %base) {
; CHECK-LABEL: f22:
; CHECK: vsteg %v24, 4088(%r2), 1
; CHECK: br %r14
  %ptr = getelementptr i64, ptr %base, i32 511
  %element = extractelement <2 x i64> %val, i32 1
  store i64 %element, ptr %ptr
  ret void
}

; Test v2i64 extraction with the first ouf-of-range offset.
define void @f23(<2 x i64> %val, ptr %base) {
; CHECK-LABEL: f23:
; CHECK: aghi %r2, 4096
; CHECK: vsteg %v24, 0(%r2), 0
; CHECK: br %r14
  %ptr = getelementptr i64, ptr %base, i32 512
  %element = extractelement <2 x i64> %val, i32 0
  store i64 %element, ptr %ptr
  ret void
}

; Test v2i64 extraction from a variable element.
define void @f24(<2 x i64> %val, ptr %ptr, i32 %index) {
; CHECK-LABEL: f24:
; CHECK-NOT: vsteg
; CHECK: br %r14
  %element = extractelement <2 x i64> %val, i32 %index
  store i64 %element, ptr %ptr
  ret void
}

; Test v4f32 extraction from the first element.
define void @f25(<4 x float> %val, ptr %ptr) {
; CHECK-LABEL: f25:
; CHECK: vstef %v24, 0(%r2), 0
; CHECK: br %r14
  %element = extractelement <4 x float> %val, i32 0
  store float %element, ptr %ptr
  ret void
}

; Test v4f32 extraction from the last element.
define void @f26(<4 x float> %val, ptr %ptr) {
; CHECK-LABEL: f26:
; CHECK: vstef %v24, 0(%r2), 3
; CHECK: br %r14
  %element = extractelement <4 x float> %val, i32 3
  store float %element, ptr %ptr
  ret void
}

; Test v4f32 extraction of an invalid element.  This must compile,
; but we don't care what it does.
define void @f27(<4 x float> %val, ptr %ptr) {
; CHECK-LABEL: f27:
; CHECK-NOT: vstef %v24, 0(%r2), 4
; CHECK: br %r14
  %element = extractelement <4 x float> %val, i32 4
  store float %element, ptr %ptr
  ret void
}

; Test v4f32 extraction with the highest in-range offset.
define void @f28(<4 x float> %val, ptr %base) {
; CHECK-LABEL: f28:
; CHECK: vstef %v24, 4092(%r2), 2
; CHECK: br %r14
  %ptr = getelementptr float, ptr %base, i32 1023
  %element = extractelement <4 x float> %val, i32 2
  store float %element, ptr %ptr
  ret void
}

; Test v4f32 extraction with the first ouf-of-range offset.
define void @f29(<4 x float> %val, ptr %base) {
; CHECK-LABEL: f29:
; CHECK: aghi %r2, 4096
; CHECK: vstef %v24, 0(%r2), 1
; CHECK: br %r14
  %ptr = getelementptr float, ptr %base, i32 1024
  %element = extractelement <4 x float> %val, i32 1
  store float %element, ptr %ptr
  ret void
}

; Test v4f32 extraction from a variable element.
define void @f30(<4 x float> %val, ptr %ptr, i32 %index) {
; CHECK-LABEL: f30:
; CHECK-NOT: vstef
; CHECK: br %r14
  %element = extractelement <4 x float> %val, i32 %index
  store float %element, ptr %ptr
  ret void
}

; Test v2f64 extraction from the first element.
define void @f32(<2 x double> %val, ptr %ptr) {
; CHECK-LABEL: f32:
; CHECK: vsteg %v24, 0(%r2), 0
; CHECK: br %r14
  %element = extractelement <2 x double> %val, i32 0
  store double %element, ptr %ptr
  ret void
}

; Test v2f64 extraction from the last element.
define void @f33(<2 x double> %val, ptr %ptr) {
; CHECK-LABEL: f33:
; CHECK: vsteg %v24, 0(%r2), 1
; CHECK: br %r14
  %element = extractelement <2 x double> %val, i32 1
  store double %element, ptr %ptr
  ret void
}

; Test v2f64 extraction with the highest in-range offset.
define void @f34(<2 x double> %val, ptr %base) {
; CHECK-LABEL: f34:
; CHECK: vsteg %v24, 4088(%r2), 1
; CHECK: br %r14
  %ptr = getelementptr double, ptr %base, i32 511
  %element = extractelement <2 x double> %val, i32 1
  store double %element, ptr %ptr
  ret void
}

; Test v2f64 extraction with the first ouf-of-range offset.
define void @f35(<2 x double> %val, ptr %base) {
; CHECK-LABEL: f35:
; CHECK: aghi %r2, 4096
; CHECK: vsteg %v24, 0(%r2), 0
; CHECK: br %r14
  %ptr = getelementptr double, ptr %base, i32 512
  %element = extractelement <2 x double> %val, i32 0
  store double %element, ptr %ptr
  ret void
}

; Test v2f64 extraction from a variable element.
define void @f36(<2 x double> %val, ptr %ptr, i32 %index) {
; CHECK-LABEL: f36:
; CHECK-NOT: vsteg
; CHECK: br %r14
  %element = extractelement <2 x double> %val, i32 %index
  store double %element, ptr %ptr
  ret void
}

; Test a v4i32 scatter of the first element.
define void @f37(<4 x i32> %val, <4 x i32> %index, i64 %base) {
; CHECK-LABEL: f37:
; CHECK: vscef %v24, 0(%v26,%r2), 0
; CHECK: br %r14
  %elem = extractelement <4 x i32> %index, i32 0
  %ext = zext i32 %elem to i64
  %add = add i64 %base, %ext
  %ptr = inttoptr i64 %add to ptr
  %element = extractelement <4 x i32> %val, i32 0
  store i32 %element, ptr %ptr
  ret void
}

; Test a v4i32 scatter of the last element.
define void @f38(<4 x i32> %val, <4 x i32> %index, i64 %base) {
; CHECK-LABEL: f38:
; CHECK: vscef %v24, 0(%v26,%r2), 3
; CHECK: br %r14
  %elem = extractelement <4 x i32> %index, i32 3
  %ext = zext i32 %elem to i64
  %add = add i64 %base, %ext
  %ptr = inttoptr i64 %add to ptr
  %element = extractelement <4 x i32> %val, i32 3
  store i32 %element, ptr %ptr
  ret void
}

; Test a v4i32 scatter with the highest in-range offset.
define void @f39(<4 x i32> %val, <4 x i32> %index, i64 %base) {
; CHECK-LABEL: f39:
; CHECK: vscef %v24, 4095(%v26,%r2), 1
; CHECK: br %r14
  %elem = extractelement <4 x i32> %index, i32 1
  %ext = zext i32 %elem to i64
  %add1 = add i64 %base, %ext
  %add2 = add i64 %add1, 4095
  %ptr = inttoptr i64 %add2 to ptr
  %element = extractelement <4 x i32> %val, i32 1
  store i32 %element, ptr %ptr
  ret void
}

; Test a v2i64 scatter of the first element.
define void @f40(<2 x i64> %val, <2 x i64> %index, i64 %base) {
; CHECK-LABEL: f40:
; CHECK: vsceg %v24, 0(%v26,%r2), 0
; CHECK: br %r14
  %elem = extractelement <2 x i64> %index, i32 0
  %add = add i64 %base, %elem
  %ptr = inttoptr i64 %add to ptr
  %element = extractelement <2 x i64> %val, i32 0
  store i64 %element, ptr %ptr
  ret void
}

; Test a v2i64 scatter of the last element.
define void @f41(<2 x i64> %val, <2 x i64> %index, i64 %base) {
; CHECK-LABEL: f41:
; CHECK: vsceg %v24, 0(%v26,%r2), 1
; CHECK: br %r14
  %elem = extractelement <2 x i64> %index, i32 1
  %add = add i64 %base, %elem
  %ptr = inttoptr i64 %add to ptr
  %element = extractelement <2 x i64> %val, i32 1
  store i64 %element, ptr %ptr
  ret void
}

; Test a v4f32 scatter of the first element.
define void @f42(<4 x float> %val, <4 x i32> %index, i64 %base) {
; CHECK-LABEL: f42:
; CHECK: vscef %v24, 0(%v26,%r2), 0
; CHECK: br %r14
  %elem = extractelement <4 x i32> %index, i32 0
  %ext = zext i32 %elem to i64
  %add = add i64 %base, %ext
  %ptr = inttoptr i64 %add to ptr
  %element = extractelement <4 x float> %val, i32 0
  store float %element, ptr %ptr
  ret void
}

; Test a v4f32 scatter of the last element.
define void @f43(<4 x float> %val, <4 x i32> %index, i64 %base) {
; CHECK-LABEL: f43:
; CHECK: vscef %v24, 0(%v26,%r2), 3
; CHECK: br %r14
  %elem = extractelement <4 x i32> %index, i32 3
  %ext = zext i32 %elem to i64
  %add = add i64 %base, %ext
  %ptr = inttoptr i64 %add to ptr
  %element = extractelement <4 x float> %val, i32 3
  store float %element, ptr %ptr
  ret void
}

; Test a v2f64 scatter of the first element.
define void @f44(<2 x double> %val, <2 x i64> %index, i64 %base) {
; CHECK-LABEL: f44:
; CHECK: vsceg %v24, 0(%v26,%r2), 0
; CHECK: br %r14
  %elem = extractelement <2 x i64> %index, i32 0
  %add = add i64 %base, %elem
  %ptr = inttoptr i64 %add to ptr
  %element = extractelement <2 x double> %val, i32 0
  store double %element, ptr %ptr
  ret void
}

; Test a v2f64 scatter of the last element.
define void @f45(<2 x double> %val, <2 x i64> %index, i64 %base) {
; CHECK-LABEL: f45:
; CHECK: vsceg %v24, 0(%v26,%r2), 1
; CHECK: br %r14
  %elem = extractelement <2 x i64> %index, i32 1
  %add = add i64 %base, %elem
  %ptr = inttoptr i64 %add to ptr
  %element = extractelement <2 x double> %val, i32 1
  store double %element, ptr %ptr
  ret void
}
