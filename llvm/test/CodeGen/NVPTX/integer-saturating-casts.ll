; RUN: llc < %s -mtriple=nvptx64 -mcpu=sm_50 | FileCheck %s
; RUN: llc < %s -mtriple=nvptx -mcpu=sm_50 | FileCheck %s
; RUN: %if ptxas-sm_80 %{ llc < %s -mtriple=nvptx64 -mcpu=sm_80 | %ptxas-verify -arch=sm_80 %}

define void @s16_to_s8(i16 %x, ptr %out) {
; CHECK-LABEL: s16_to_s8(
; CHECK: cvt.sat.s8.s16
; CHECK: ret;
  %lo = call i16 @llvm.smax.i16(i16 %x, i16 -128)
  %hi = call i16 @llvm.smin.i16(i16 %lo, i16 127)
  %r = trunc i16 %hi to i8
  store i8 %r, ptr %out
  ret void
}

define void @s16_to_u8(i16 %x, ptr %out) {
; CHECK-LABEL: s16_to_u8(
; CHECK: cvt.sat.u8.s16
; CHECK: ret;
  %lo = call i16 @llvm.smax.i16(i16 %x, i16 0)
  %hi = call i16 @llvm.smin.i16(i16 %lo, i16 255)
  %r = trunc i16 %hi to i8
  store i8 %r, ptr %out
  ret void
}

define void @u16_to_s8(i16 %x, ptr %out) {
; CHECK-LABEL: u16_to_s8(
; CHECK: cvt.sat.s8.u16
; CHECK: ret;
  %hi = call i16 @llvm.umin.i16(i16 %x, i16 127)
  %r = trunc i16 %hi to i8
  store i8 %r, ptr %out
  ret void
}

define void @u16_to_u8(i16 %x, ptr %out) {
; CHECK-LABEL: u16_to_u8(
; CHECK: cvt.sat.u8.u16
; CHECK: ret;
  %hi = call i16 @llvm.umin.i16(i16 %x, i16 255)
  %r = trunc i16 %hi to i8
  store i8 %r, ptr %out
  ret void
}

define void @s32_to_s8(i32 %x, ptr %out) {
; CHECK-LABEL: s32_to_s8(
; CHECK: cvt.sat.s8.s32
; CHECK: ret;
  %lo = call i32 @llvm.smax.i32(i32 %x, i32 -128)
  %hi = call i32 @llvm.smin.i32(i32 %lo, i32 127)
  %r = trunc i32 %hi to i8
  store i8 %r, ptr %out
  ret void
}

define void @s32_to_u8(i32 %x, ptr %out) {
; CHECK-LABEL: s32_to_u8(
; CHECK: cvt.sat.u8.s32
; CHECK: ret;
  %lo = call i32 @llvm.smax.i32(i32 %x, i32 0)
  %hi = call i32 @llvm.smin.i32(i32 %lo, i32 255)
  %r = trunc i32 %hi to i8
  store i8 %r, ptr %out
  ret void
}

define void @u32_to_s8(i32 %x, ptr %out) {
; CHECK-LABEL: u32_to_s8(
; CHECK: cvt.sat.s8.u32
; CHECK: ret;
  %hi = call i32 @llvm.umin.i32(i32 %x, i32 127)
  %r = trunc i32 %hi to i8
  store i8 %r, ptr %out
  ret void
}

define void @u32_to_u8(i32 %x, ptr %out) {
; CHECK-LABEL: u32_to_u8(
; CHECK: cvt.sat.u8.u32
; CHECK: ret;
  %hi = call i32 @llvm.umin.i32(i32 %x, i32 255)
  %r = trunc i32 %hi to i8
  store i8 %r, ptr %out
  ret void
}

define void @s64_to_s8(i64 %x, ptr %out) {
; CHECK-LABEL: s64_to_s8(
; CHECK: cvt.sat.s8.s64
; CHECK: ret;
  %lo = call i64 @llvm.smax.i64(i64 %x, i64 -128)
  %hi = call i64 @llvm.smin.i64(i64 %lo, i64 127)
  %r = trunc i64 %hi to i8
  store i8 %r, ptr %out
  ret void
}

define void @s64_to_u8(i64 %x, ptr %out) {
; CHECK-LABEL: s64_to_u8(
; CHECK: cvt.sat.u8.s64
; CHECK: ret;
  %lo = call i64 @llvm.smax.i64(i64 %x, i64 0)
  %hi = call i64 @llvm.smin.i64(i64 %lo, i64 255)
  %r = trunc i64 %hi to i8
  store i8 %r, ptr %out
  ret void
}

define void @u64_to_s8(i64 %x, ptr %out) {
; CHECK-LABEL: u64_to_s8(
; CHECK: cvt.sat.s8.u64
; CHECK: ret;
  %hi = call i64 @llvm.umin.i64(i64 %x, i64 127)
  %r = trunc i64 %hi to i8
  store i8 %r, ptr %out
  ret void
}

define void @u64_to_u8(i64 %x, ptr %out) {
; CHECK-LABEL: u64_to_u8(
; CHECK: cvt.sat.u8.u64
; CHECK: ret;
  %hi = call i64 @llvm.umin.i64(i64 %x, i64 255)
  %r = trunc i64 %hi to i8
  store i8 %r, ptr %out
  ret void
}

define void @s32_to_s16(i32 %x, ptr %out) {
; CHECK-LABEL: s32_to_s16(
; CHECK: cvt.sat.s16.s32
; CHECK: ret;
  %lo = call i32 @llvm.smax.i32(i32 %x, i32 -32768)
  %hi = call i32 @llvm.smin.i32(i32 %lo, i32 32767)
  %r = trunc i32 %hi to i16
  store i16 %r, ptr %out
  ret void
}

define void @s32_to_u16(i32 %x, ptr %out) {
; CHECK-LABEL: s32_to_u16(
; CHECK: cvt.sat.u16.s32
; CHECK: ret;
  %lo = call i32 @llvm.smax.i32(i32 %x, i32 0)
  %hi = call i32 @llvm.smin.i32(i32 %lo, i32 65535)
  %r = trunc i32 %hi to i16
  store i16 %r, ptr %out
  ret void
}

define void @u32_to_s16(i32 %x, ptr %out) {
; CHECK-LABEL: u32_to_s16(
; CHECK: cvt.sat.s16.u32
; CHECK: ret;
  %hi = call i32 @llvm.umin.i32(i32 %x, i32 32767)
  %r = trunc i32 %hi to i16
  store i16 %r, ptr %out
  ret void
}

define void @u32_to_u16(i32 %x, ptr %out) {
; CHECK-LABEL: u32_to_u16(
; CHECK: cvt.sat.u16.u32
; CHECK: ret;
  %hi = call i32 @llvm.umin.i32(i32 %x, i32 65535)
  %r = trunc i32 %hi to i16
  store i16 %r, ptr %out
  ret void
}

define void @s64_to_s16(i64 %x, ptr %out) {
; CHECK-LABEL: s64_to_s16(
; CHECK: cvt.sat.s16.s64
; CHECK: ret;
  %lo = call i64 @llvm.smax.i64(i64 %x, i64 -32768)
  %hi = call i64 @llvm.smin.i64(i64 %lo, i64 32767)
  %r = trunc i64 %hi to i16
  store i16 %r, ptr %out
  ret void
}

define void @s64_to_u16(i64 %x, ptr %out) {
; CHECK-LABEL: s64_to_u16(
; CHECK: cvt.sat.u16.s64
; CHECK: ret;
  %lo = call i64 @llvm.smax.i64(i64 %x, i64 0)
  %hi = call i64 @llvm.smin.i64(i64 %lo, i64 65535)
  %r = trunc i64 %hi to i16
  store i16 %r, ptr %out
  ret void
}

define void @u64_to_s16(i64 %x, ptr %out) {
; CHECK-LABEL: u64_to_s16(
; CHECK: cvt.sat.s16.u64
; CHECK: ret;
  %hi = call i64 @llvm.umin.i64(i64 %x, i64 32767)
  %r = trunc i64 %hi to i16
  store i16 %r, ptr %out
  ret void
}

define void @u64_to_u16(i64 %x, ptr %out) {
; CHECK-LABEL: u64_to_u16(
; CHECK: cvt.sat.u16.u64
; CHECK: ret;
  %hi = call i64 @llvm.umin.i64(i64 %x, i64 65535)
  %r = trunc i64 %hi to i16
  store i16 %r, ptr %out
  ret void
}

define void @s64_to_s32(i64 %x, ptr %out) {
; CHECK-LABEL: s64_to_s32(
; CHECK: cvt.sat.s32.s64
; CHECK: ret;
  %lo = call i64 @llvm.smax.i64(i64 %x, i64 -2147483648)
  %hi = call i64 @llvm.smin.i64(i64 %lo, i64 2147483647)
  %r = trunc i64 %hi to i32
  store i32 %r, ptr %out
  ret void
}

define void @s64_to_u32(i64 %x, ptr %out) {
; CHECK-LABEL: s64_to_u32(
; CHECK: cvt.sat.u32.s64
; CHECK: ret;
  %lo = call i64 @llvm.smax.i64(i64 %x, i64 0)
  %hi = call i64 @llvm.smin.i64(i64 %lo, i64 4294967295)
  %r = trunc i64 %hi to i32
  store i32 %r, ptr %out
  ret void
}

define void @u64_to_s32(i64 %x, ptr %out) {
; CHECK-LABEL: u64_to_s32(
; CHECK: cvt.sat.s32.u64
; CHECK: ret;
  %hi = call i64 @llvm.umin.i64(i64 %x, i64 2147483647)
  %r = trunc i64 %hi to i32
  store i32 %r, ptr %out
  ret void
}

define void @u64_to_u32(i64 %x, ptr %out) {
; CHECK-LABEL: u64_to_u32(
; CHECK: cvt.sat.u32.u64
; CHECK: ret;
  %hi = call i64 @llvm.umin.i64(i64 %x, i64 4294967295)
  %r = trunc i64 %hi to i32
  store i32 %r, ptr %out
  ret void
}

define i32 @s16_to_u32(i16 %x) {
; CHECK-LABEL: s16_to_u32(
; CHECK: cvt.sat.u32.s16
; CHECK: ret;
  %lo = call i16 @llvm.smax.i16(i16 %x, i16 0)
  %r = zext i16 %lo to i32
  ret i32 %r
}

define i64 @s16_to_u64(i16 %x) {
; CHECK-LABEL: s16_to_u64(
; CHECK: cvt.sat.u64.s16
; CHECK: ret;
  %lo = call i16 @llvm.smax.i16(i16 %x, i16 0)
  %r = zext i16 %lo to i64
  ret i64 %r
}

define i64 @s32_to_u64(i32 %x) {
; CHECK-LABEL: s32_to_u64(
; CHECK: cvt.sat.u64.s32
; CHECK: ret;
  %lo = call i32 @llvm.smax.i32(i32 %x, i32 0)
  %r = zext i32 %lo to i64
  ret i64 %r
}

define void @s32_to_s16_reversed(i32 %x, ptr %out) {
; CHECK-LABEL: s32_to_s16_reversed(
; CHECK: cvt.sat.s16.s32
; CHECK: ret;
  %hi = call i32 @llvm.smin.i32(i32 %x, i32 32767)
  %lo = call i32 @llvm.smax.i32(i32 %hi, i32 -32768)
  %r = trunc i32 %lo to i16
  store i16 %r, ptr %out
  ret void
}

define void @s32_to_u8_reversed(i32 %x, ptr %out) {
; CHECK-LABEL: s32_to_u8_reversed(
; CHECK: cvt.sat.u8.s32
; CHECK: ret;
  %hi = call i32 @llvm.smin.i32(i32 %x, i32 255)
  %lo = call i32 @llvm.smax.i32(i32 %hi, i32 0)
  %r = trunc i32 %lo to i8
  store i8 %r, ptr %out
  ret void
}

define void @s32_to_u8_umin(i32 %x, ptr %out) {
; CHECK-LABEL: s32_to_u8_umin(
; CHECK: cvt.sat.u8.s32
; CHECK: ret;
  %lo = call i32 @llvm.smax.i32(i32 %x, i32 0)
  %hi = call i32 @llvm.umin.i32(i32 %lo, i32 255)
  %r = trunc i32 %hi to i8
  store i8 %r, ptr %out
  ret void
}

define void @s32_to_s16_non_saturating_bounds(i32 %x, ptr %out) {
; CHECK-LABEL: s32_to_s16_non_saturating_bounds(
; CHECK: max.s32
; CHECK: min.s32
; CHECK-NOT: cvt.sat.s16.s32
; CHECK: ret;
  %lo = call i32 @llvm.smax.i32(i32 %x, i32 -32767)
  %hi = call i32 @llvm.smin.i32(i32 %lo, i32 32766)
  %r = trunc i32 %hi to i16
  store i16 %r, ptr %out
  ret void
}

define void @v2s32_to_v2s16(<2 x i32> %x, ptr %out) {
; CHECK-LABEL: v2s32_to_v2s16(
; CHECK: cvt.sat.s16.s32
; CHECK: cvt.sat.s16.s32
; CHECK: ret;
  %lo = call <2 x i32> @llvm.smax.v2i32(<2 x i32> %x, <2 x i32> splat (i32 -32768))
  %hi = call <2 x i32> @llvm.smin.v2i32(<2 x i32> %lo, <2 x i32> splat (i32 32767))
  %r = trunc <2 x i32> %hi to <2 x i16>
  store <2 x i16> %r, ptr %out
  ret void
}

; Predicate results must not be combined into unsupported saturating DAG nodes.
define i1 @u32_to_u1(i32 %x) {
; CHECK-LABEL: u32_to_u1(
; CHECK-NOT: cvt.sat
; CHECK: ret;
  %hi = call i32 @llvm.umin.i32(i32 %x, i32 1)
  %r = trunc i32 %hi to i1
  ret i1 %r
}

define i1 @s64_to_s1(i64 %x) {
; CHECK-LABEL: s64_to_s1(
; CHECK-NOT: cvt.sat
; CHECK: ret;
  %lo = call i64 @llvm.smax.i64(i64 %x, i64 -1)
  %hi = call i64 @llvm.smin.i64(i64 %lo, i64 0)
  %r = trunc i64 %hi to i1
  ret i1 %r
}

; Return values can retain the clamp in a wider register after legalization.
define i8 @s32_to_s8_return(i32 %x) {
; CHECK-LABEL: s32_to_s8_return(
; CHECK: cvt.sat.s8.s32
; CHECK: ret;
  %lo = call i32 @llvm.smax.i32(i32 %x, i32 -128)
  %hi = call i32 @llvm.smin.i32(i32 %lo, i32 127)
  %r = trunc i32 %hi to i8
  ret i8 %r
}
