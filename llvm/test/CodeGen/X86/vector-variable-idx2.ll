; RUN: llc < %s -mattr=+sse4.1

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-apple-darwin11.0.0"

define i64 @__builtin_ia32_vec_ext_v2di(<2 x i64> %a, i32 %i) nounwind {
  %1 = alloca <2 x i64>, align 16
  %2 = alloca i32, align 4
  store <2 x i64> %a, ptr %1, align 16
  store i32 %i, ptr %2, align 4
  %3 = load <2 x i64>, ptr %1, align 16
  %4 = load i32, ptr %2, align 4
  %5 = extractelement <2 x i64> %3, i32 %4
  ret i64 %5
}

define <2 x i64> @__builtin_ia32_vec_int_v2di(<2 x i64> %a, i32 %i) nounwind {
  %1 = alloca <2 x i64>, align 16
  %2 = alloca i32, align 4
  store <2 x i64> %a, ptr %1, align 16
  store i32 %i, ptr %2, align 4
  %3 = load <2 x i64>, ptr %1, align 16
  %4 = load i32, ptr %2, align 4
  %5 = insertelement <2 x i64> %3, i64 1, i32 %4
  ret <2 x i64> %5
}
