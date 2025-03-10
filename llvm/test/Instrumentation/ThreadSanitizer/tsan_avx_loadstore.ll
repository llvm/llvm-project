; RUN: opt < %s -passes='function(tsan),module(tsan-module)' -S | FileCheck %s

; target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-unknown-linux-gnu"

define <8 x i32> @read_8_int(ptr %a) sanitize_thread {
entry:
  %tmp1 = load <8 x i32>, ptr %a, align 4
  ret <8 x i32> %tmp1
}
; CHECK: define <8 x i32> @read_8_int(
; CHECK: call void @__tsan_unaligned_read32(ptr %a)

define <4 x i32> @read_4_int(ptr %a) sanitize_thread {
entry:
  %tmp1 = load <4 x i32>, ptr %a, align 4
  ret <4 x i32> %tmp1
}
; CHECK: define <4 x i32> @read_4_int(
; CHECK: call void @__tsan_unaligned_read16(ptr %a)

define <8 x double> @read_8_double_unaligned(ptr %a) sanitize_thread {
entry:
  %tmp1 = load <8 x double>, ptr %a, align 4
  ret <8 x double> %tmp1
}
; CHECK: define <8 x double> @read_8_double_unaligned(
; CHECK: call void @__tsan_unaligned_read64(ptr %a)

define <4 x double> @read_4_double_unaligned(ptr %a) sanitize_thread {
entry:
  %tmp1 = load <4 x double>, ptr %a, align 4
  ret <4 x double> %tmp1
}
; CHECK: define <4 x double> @read_4_double_unaligned(
; CHECK: call void @__tsan_unaligned_read32(ptr %a)

define <2 x double> @read_2_double_unaligned(ptr %a) sanitize_thread {
entry:
  %tmp1 = load <2 x double>, ptr %a, align 4
  ret <2 x double> %tmp1
}
; CHECK: define <2 x double> @read_2_double_unaligned(
; CHECK: call void @__tsan_unaligned_read16(ptr %a)

define <8 x double> @read_8_double(ptr %a) sanitize_thread {
entry:
  %tmp1 = load <8 x double>, ptr %a, align 8
  ret <8 x double> %tmp1
}
; CHECK: define <8 x double> @read_8_double(
; CHECK: call void @__tsan_read64(ptr %a)

define <4 x double> @read_4_double(ptr %a) sanitize_thread {
entry:
  %tmp1 = load <4 x double>, ptr %a, align 8
  ret <4 x double> %tmp1
}
; CHECK: define <4 x double> @read_4_double(
; CHECK: call void @__tsan_read32(ptr %a)

define <2 x double> @read_2_double(ptr %a) sanitize_thread {
entry:
  %tmp1 = load <2 x double>, ptr %a, align 8
  ret <2 x double> %tmp1
}
; CHECK: define <2 x double> @read_2_double(
; CHECK: call void @__tsan_read16(ptr %a)

define void @write_8_double(ptr %a) sanitize_thread {
entry:
  store <8 x double> <double 1.0, double 1.0, double 1.0, double 1.0, double 1.0, double 1.0, double 1.0, double 1.0>, ptr %a
  ret void
}
; CHECK: define void @write_8_double(
; CHECK: call void @__tsan_write64(ptr %a)

define void @write_4_double(ptr %a) sanitize_thread {
entry:
  store <4 x double> <double 1.0, double 1.0, double 1.0, double 1.0>, ptr %a
  ret void
}
; CHECK: define void @write_4_double(
; CHECK: call void @__tsan_write32(ptr %a)

define void @write_8_float(ptr %a) sanitize_thread {
entry:
  store <8 x float> <float 1.0, float 1.0, float 1.0, float 1.0, float 1.0, float 1.0, float 1.0, float 1.0>, ptr %a
  ret void
}
; CHECK: define void @write_8_float(
; CHECK: call void @__tsan_write32(ptr %a)

define void @write_4_float(ptr %a) sanitize_thread {
entry:
  store <4 x float> <float 1.0, float 1.0, float 1.0, float 1.0>, ptr %a
  ret void
}
; CHECK: define void @write_4_float(
; CHECK: call void @__tsan_write16(ptr %a)
