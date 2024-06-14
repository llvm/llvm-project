; RUN: llc < %s -O0
; PR8211
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-unknown-linux-gnu"

module asm "\09.ident\09\22GCC: (GNU) 4.5.2 20100914 (prerelease) LLVM: 114628\22"

%"int[]" = type [4 x i32]
%0 = type { %"int[]" }
%float = type float
%"float[]" = type [4 x float]
%int = type i32
%"long unsigned int" = type i64

define void @swizzle(ptr %a, ptr %b, ptr %c) nounwind {
entry:
  %a_addr = alloca ptr
  %b_addr = alloca ptr
  %c_addr = alloca ptr
  %"alloca point" = bitcast i32 0 to i32
  store ptr %a, ptr %a_addr
  store ptr %b, ptr %b_addr
  store ptr %c, ptr %c_addr
  %0 = load ptr, ptr %a_addr, align 64
  %1 = load ptr, ptr %b_addr, align 64
  %2 = load ptr, ptr %c_addr, align 64
  %"ssa point" = bitcast i32 0 to i32
  br label %"2"

"2":                                              ; preds = %entry
  %3 = load <4 x float>, ptr %1, align 16
  %4 = load double, ptr %0
  %5 = insertelement <2 x double> undef, double %4, i32 0
  %6 = insertelement <2 x double> %5, double undef, i32 1
  %7 = bitcast <2 x double> %6 to <4 x float>
  %8 = shufflevector <4 x float> %3, <4 x float> %7, <4 x i32> <i32 4, i32 5, i32 2, i32 3>
  store <4 x float> %8, ptr %1, align 16
  %9 = getelementptr i8, ptr %0, i64 8
  %10 = load <4 x float>, ptr %2, align 16
  %11 = load double, ptr %9
  %12 = insertelement <2 x double> undef, double %11, i32 0
  %13 = insertelement <2 x double> %12, double undef, i32 1
  %14 = bitcast <2 x double> %13 to <4 x float>
  %15 = shufflevector <4 x float> %10, <4 x float> %14, <4 x i32> <i32 4, i32 5, i32 2, i32 3>
  store <4 x float> %15, ptr %2, align 16
  br label %return

return:                                           ; preds = %"2"
  ret void
}
