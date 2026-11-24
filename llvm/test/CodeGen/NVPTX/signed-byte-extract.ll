; RUN: llc < %s -mtriple=nvptx64 -mcpu=sm_80 -O0 -combiner-disabled | FileCheck %s

; Store the i16 result: returning it lets ABI lowering bypass the failing DAG.

; CHECK-LABEL: extract32_24(
; CHECK: bfe.s32 {{.*}}, 24, 8;
; CHECK: st.global.b16
; CHECK: ret;
define void @extract32_24(i32 %x, ptr addrspace(1) %out) {
  %s = lshr i32 %x, 24
  %b = trunc i32 %s to i8
  %r = sext i8 %b to i16
  store i16 %r, ptr addrspace(1) %out, align 2
  ret void
}

; CHECK-LABEL: extract32_25(
; CHECK-NOT: bfe.s32
; CHECK: bfe.u32 {{.*}}, 25, 8;
; CHECK: cvt.s8.s32
; CHECK: st.global.b16
; CHECK: ret;
define void @extract32_25(i32 %x, ptr addrspace(1) %out) {
  %s = lshr i32 %x, 25
  %b = trunc i32 %s to i8
  %r = sext i8 %b to i16
  store i16 %r, ptr addrspace(1) %out, align 2
  ret void
}

; CHECK-LABEL: extract32_27(
; CHECK-NOT: bfe.s32
; CHECK: bfe.u32 {{.*}}, 27, 8;
; CHECK: cvt.s8.s32
; CHECK: st.global.b16
; CHECK: ret;
define void @extract32_27(i32 %x, ptr addrspace(1) %out) {
  %s = lshr i32 %x, 27
  %b = trunc i32 %s to i8
  %r = sext i8 %b to i16
  store i16 %r, ptr addrspace(1) %out, align 2
  ret void
}

; CHECK-LABEL: extract32_31(
; CHECK-NOT: bfe.s32
; CHECK: bfe.u32 {{.*}}, 31, 8;
; CHECK: cvt.s8.s32
; CHECK: st.global.b16
; CHECK: ret;
define void @extract32_31(i32 %x, ptr addrspace(1) %out) {
  %s = lshr i32 %x, 31
  %b = trunc i32 %s to i8
  %r = sext i8 %b to i16
  store i16 %r, ptr addrspace(1) %out, align 2
  ret void
}

; CHECK-LABEL: extract64_56(
; CHECK: bfe.s64 {{.*}}, 56, 8;
; CHECK: st.global.b16
; CHECK: ret;
define void @extract64_56(i64 %x, ptr addrspace(1) %out) {
  %s = lshr i64 %x, 56
  %b = trunc i64 %s to i8
  %r = sext i8 %b to i16
  store i16 %r, ptr addrspace(1) %out, align 2
  ret void
}

; CHECK-LABEL: extract64_57(
; CHECK-NOT: bfe.s64
; CHECK: bfe.u64 {{.*}}, 57, 8;
; CHECK: cvt.s8.s64
; CHECK: st.global.b16
; CHECK: ret;
define void @extract64_57(i64 %x, ptr addrspace(1) %out) {
  %s = lshr i64 %x, 57
  %b = trunc i64 %s to i8
  %r = sext i8 %b to i16
  store i16 %r, ptr addrspace(1) %out, align 2
  ret void
}

; CHECK-LABEL: extract64_60(
; CHECK-NOT: bfe.s64
; CHECK: bfe.u64 {{.*}}, 60, 8;
; CHECK: cvt.s8.s64
; CHECK: st.global.b16
; CHECK: ret;
define void @extract64_60(i64 %x, ptr addrspace(1) %out) {
  %s = lshr i64 %x, 60
  %b = trunc i64 %s to i8
  %r = sext i8 %b to i16
  store i16 %r, ptr addrspace(1) %out, align 2
  ret void
}

; CHECK-LABEL: extract64_63(
; CHECK-NOT: bfe.s64
; CHECK: bfe.u64 {{.*}}, 63, 8;
; CHECK: cvt.s8.s64
; CHECK: st.global.b16
; CHECK: ret;
define void @extract64_63(i64 %x, ptr addrspace(1) %out) {
  %s = lshr i64 %x, 63
  %b = trunc i64 %s to i8
  %r = sext i8 %b to i16
  store i16 %r, ptr addrspace(1) %out, align 2
  ret void
}

; Full-width results fold to logical shifts when the byte extends past the
; source width. Keep signed BFE for the last complete byte.

; CHECK-LABEL: extract32_24_full(
; CHECK: bfe.s32 {{.*}}, 24, 8;
; CHECK: ret;
define i32 @extract32_24_full(i32 %x) {
  %s = lshr i32 %x, 24
  %b = trunc i32 %s to i8
  %r = sext i8 %b to i32
  ret i32 %r
}

; CHECK-LABEL: extract32_25_full(
; CHECK: shr.u32 {{.*}}, 25;
; CHECK: ret;
define i32 @extract32_25_full(i32 %x) {
  %s = lshr i32 %x, 25
  %b = trunc i32 %s to i8
  %r = sext i8 %b to i32
  ret i32 %r
}

; CHECK-LABEL: extract32_31_full(
; CHECK: shr.u32 {{.*}}, 31;
; CHECK: ret;
define i32 @extract32_31_full(i32 %x) {
  %s = lshr i32 %x, 31
  %b = trunc i32 %s to i8
  %r = sext i8 %b to i32
  ret i32 %r
}

; CHECK-LABEL: extract64_56_full(
; CHECK: bfe.s64 {{.*}}, 56, 8;
; CHECK: ret;
define i64 @extract64_56_full(i64 %x) {
  %s = lshr i64 %x, 56
  %b = trunc i64 %s to i8
  %r = sext i8 %b to i64
  ret i64 %r
}

; CHECK-LABEL: extract64_57_full(
; CHECK: shr.u64 {{.*}}, 57;
; CHECK: ret;
define i64 @extract64_57_full(i64 %x) {
  %s = lshr i64 %x, 57
  %b = trunc i64 %s to i8
  %r = sext i8 %b to i64
  ret i64 %r
}

; CHECK-LABEL: extract64_63_full(
; CHECK: shr.u64 {{.*}}, 63;
; CHECK: ret;
define i64 @extract64_63_full(i64 %x) {
  %s = lshr i64 %x, 63
  %b = trunc i64 %s to i8
  %r = sext i8 %b to i64
  ret i64 %r
}
