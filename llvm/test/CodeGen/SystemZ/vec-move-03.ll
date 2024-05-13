; Test vector stores.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z13 | FileCheck %s

; Test v16i8 stores.
define void @f1(<16 x i8> %val, ptr %ptr) {
; CHECK-LABEL: f1:
; CHECK: vst %v24, 0(%r2), 3
; CHECK: br %r14
  store <16 x i8> %val, ptr %ptr
  ret void
}

; Test v8i16 stores.
define void @f2(<8 x i16> %val, ptr %ptr) {
; CHECK-LABEL: f2:
; CHECK: vst %v24, 0(%r2), 3
; CHECK: br %r14
  store <8 x i16> %val, ptr %ptr
  ret void
}

; Test v4i32 stores.
define void @f3(<4 x i32> %val, ptr %ptr) {
; CHECK-LABEL: f3:
; CHECK: vst %v24, 0(%r2), 3
; CHECK: br %r14
  store <4 x i32> %val, ptr %ptr
  ret void
}

; Test v2i64 stores.
define void @f4(<2 x i64> %val, ptr %ptr) {
; CHECK-LABEL: f4:
; CHECK: vst %v24, 0(%r2), 3
; CHECK: br %r14
  store <2 x i64> %val, ptr %ptr
  ret void
}

; Test v4f32 stores.
define void @f5(<4 x float> %val, ptr %ptr) {
; CHECK-LABEL: f5:
; CHECK: vst %v24, 0(%r2), 3
; CHECK: br %r14
  store <4 x float> %val, ptr %ptr
  ret void
}

; Test v2f64 stores.
define void @f6(<2 x double> %val, ptr %ptr) {
; CHECK-LABEL: f6:
; CHECK: vst %v24, 0(%r2), 3
; CHECK: br %r14
  store <2 x double> %val, ptr %ptr
  ret void
}

; Test the highest aligned in-range offset.
define void @f7(<16 x i8> %val, ptr %base) {
; CHECK-LABEL: f7:
; CHECK: vst %v24, 4080(%r2), 3
; CHECK: br %r14
  %ptr = getelementptr <16 x i8>, ptr %base, i64 255
  store <16 x i8> %val, ptr %ptr
  ret void
}

; Test the highest unaligned in-range offset.
define void @f8(<16 x i8> %val, ptr %base) {
; CHECK-LABEL: f8:
; CHECK: vst %v24, 4095(%r2)
; CHECK: br %r14
  %addr = getelementptr i8, ptr %base, i64 4095
  store <16 x i8> %val, ptr %addr, align 1
  ret void
}

; Test the next offset up, which requires separate address logic,
define void @f9(<16 x i8> %val, ptr %base) {
; CHECK-LABEL: f9:
; CHECK: aghi %r2, 4096
; CHECK: vst %v24, 0(%r2), 3
; CHECK: br %r14
  %ptr = getelementptr <16 x i8>, ptr %base, i64 256
  store <16 x i8> %val, ptr %ptr
  ret void
}

; Test negative offsets, which also require separate address logic,
define void @f10(<16 x i8> %val, ptr %base) {
; CHECK-LABEL: f10:
; CHECK: aghi %r2, -16
; CHECK: vst %v24, 0(%r2), 3
; CHECK: br %r14
  %ptr = getelementptr <16 x i8>, ptr %base, i64 -1
  store <16 x i8> %val, ptr %ptr
  ret void
}

; Check that indexes are allowed.
define void @f11(<16 x i8> %val, ptr %base, i64 %index) {
; CHECK-LABEL: f11:
; CHECK: vst %v24, 0(%r3,%r2)
; CHECK: br %r14
  %addr = getelementptr i8, ptr %base, i64 %index
  store <16 x i8> %val, ptr %addr, align 1
  ret void
}

; Test v2i8 stores.
define void @f12(<2 x i8> %val, ptr %ptr) {
; CHECK-LABEL: f12:
; CHECK: vsteh %v24, 0(%r2), 0
; CHECK: br %r14
  store <2 x i8> %val, ptr %ptr
  ret void
}

; Test v4i8 stores.
define void @f13(<4 x i8> %val, ptr %ptr) {
; CHECK-LABEL: f13:
; CHECK: vstef %v24, 0(%r2)
; CHECK: br %r14
  store <4 x i8> %val, ptr %ptr
  ret void
}

; Test v8i8 stores.
define void @f14(<8 x i8> %val, ptr %ptr) {
; CHECK-LABEL: f14:
; CHECK: vsteg %v24, 0(%r2)
; CHECK: br %r14
  store <8 x i8> %val, ptr %ptr
  ret void
}

; Test v2i16 stores.
define void @f15(<2 x i16> %val, ptr %ptr) {
; CHECK-LABEL: f15:
; CHECK: vstef %v24, 0(%r2), 0
; CHECK: br %r14
  store <2 x i16> %val, ptr %ptr
  ret void
}

; Test v4i16 stores.
define void @f16(<4 x i16> %val, ptr %ptr) {
; CHECK-LABEL: f16:
; CHECK: vsteg %v24, 0(%r2)
; CHECK: br %r14
  store <4 x i16> %val, ptr %ptr
  ret void
}

; Test v2i32 stores.
define void @f17(<2 x i32> %val, ptr %ptr) {
; CHECK-LABEL: f17:
; CHECK: vsteg %v24, 0(%r2), 0
; CHECK: br %r14
  store <2 x i32> %val, ptr %ptr
  ret void
}

; Test v2f32 stores.
define void @f18(<2 x float> %val, ptr %ptr) {
; CHECK-LABEL: f18:
; CHECK: vsteg %v24, 0(%r2), 0
; CHECK: br %r14
  store <2 x float> %val, ptr %ptr
  ret void
}

; Test quadword-aligned stores.
define void @f19(<16 x i8> %val, ptr %ptr) {
; CHECK-LABEL: f19:
; CHECK: vst %v24, 0(%r2), 4
; CHECK: br %r14
  store <16 x i8> %val, ptr %ptr, align 16
  ret void
}

; Test that the alignment hint for VST is emitted also when CFG optimizer
; replaces two VSTs with just one that then carries two memoperands.
define void @f20() {
; CHECK-LABEL: f20:
; CHECK: vst %v0, 0(%r1), 3
; CHECK-NOT: vst
entry:
  switch i32 undef, label %exit [
    i32 1, label %bb1
    i32 2, label %bb2
  ]

bb1:
  %C1 = call ptr @foo()
  %I1 = insertelement <2 x ptr> poison, ptr %C1, i64 0
  %S1 = shufflevector <2 x ptr> %I1, <2 x ptr> poison, <2 x i32> zeroinitializer
  store <2 x ptr> %S1, ptr undef, align 8
  br label %exit

bb2:
  %C2 = call ptr @foo()
  %I2 = insertelement <2 x ptr> poison, ptr %C2, i64 0
  %S2 = shufflevector <2 x ptr> %I2, <2 x ptr> poison, <2 x i32> zeroinitializer
  store <2 x ptr> %S2, ptr undef, align 8
  br label %exit

exit:
  ret void
}

declare ptr @foo()
