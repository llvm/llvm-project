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

define void @swizzle(i8* %a, %0* %b, %0* %c) nounwind {
entry:
  %a_addr = alloca i8*
  %b_addr = alloca %0*
  %c_addr = alloca %0*
  %"alloca point" = bitcast i32 0 to i32
  store i8* %a, i8** %a_addr
  store %0* %b, %0** %b_addr
  store %0* %c, %0** %c_addr
  %0 = load i8*, i8** %a_addr, align 64
  %1 = load %0*, %0** %b_addr, align 64
  %2 = load %0*, %0** %c_addr, align 64
  %"ssa point" = bitcast i32 0 to i32
  br label %"2"

"2":                                              ; preds = %entry
  %3 = bitcast i8* %0 to <2 x i32>*
  %4 = getelementptr inbounds %0, %0* %1, i32 0, i32 0
  %5 = bitcast %"int[]"* %4 to <4 x float>*
  %6 = load <4 x float>, <4 x float>* %5, align 16
  %7 = bitcast <2 x i32>* %3 to <2 x float>*
  %8 = bitcast <2 x float>* %7 to double*
  %9 = load double, double* %8
  %10 = insertelement <2 x double> undef, double %9, i32 0
  %11 = insertelement <2 x double> %10, double undef, i32 1
  %12 = bitcast <2 x double> %11 to <4 x float>
  %13 = shufflevector <4 x float> %6, <4 x float> %12, <4 x i32> <i32 4, i32 5, i32 2, i32 3>
  %14 = getelementptr inbounds %0, %0* %1, i32 0, i32 0
  %15 = bitcast %"int[]"* %14 to <4 x float>*
  store <4 x float> %13, <4 x float>* %15, align 16
  %16 = bitcast i8* %0 to <2 x i32>*
  %17 = bitcast <2 x i32>* %16 to i8*
  %18 = getelementptr i8, i8* %17, i64 8
  %19 = bitcast i8* %18 to <2 x i32>*
  %20 = getelementptr inbounds %0, %0* %2, i32 0, i32 0
  %21 = bitcast %"int[]"* %20 to <4 x float>*
  %22 = load <4 x float>, <4 x float>* %21, align 16
  %23 = bitcast <2 x i32>* %19 to <2 x float>*
  %24 = bitcast <2 x float>* %23 to double*
  %25 = load double, double* %24
  %26 = insertelement <2 x double> undef, double %25, i32 0
  %27 = insertelement <2 x double> %26, double undef, i32 1
  %28 = bitcast <2 x double> %27 to <4 x float>
  %29 = shufflevector <4 x float> %22, <4 x float> %28, <4 x i32> <i32 4, i32 5, i32 2, i32 3>
  %30 = getelementptr inbounds %0, %0* %2, i32 0, i32 0
  %31 = bitcast %"int[]"* %30 to <4 x float>*
  store <4 x float> %29, <4 x float>* %31, align 16
  br label %return

return:                                           ; preds = %"2"
  ret void
}
