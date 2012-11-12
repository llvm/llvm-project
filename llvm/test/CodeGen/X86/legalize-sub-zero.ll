; RUN: llc < %s -mtriple=i686-pc-win32

;target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-f80:128:128-v64:64:64-v128:128:128-a0:0:64-f80:32:32-n8:16:32"
;target triple = "i686-pc-win32"

define void @test() nounwind {
  %1 = fdiv <3 x double> zeroinitializer, undef
  %2 = fdiv <2 x double> zeroinitializer, undef
  %3 = shufflevector <2 x double> %2, <2 x double> undef, <3 x i32> <i32 0, i32
1, i32 undef>
  %4 = insertelement <3 x double> %3, double undef, i32 2
  %5 = bitcast <3 x double> %1 to <3 x i64>
  %6 = bitcast <3 x double> %4 to <3 x i64>
  %7 = sub <3 x i64> %5, %6
  %8 = shufflevector <3 x i64> %7, <3 x i64> undef, <2 x i32> <i32 0, i32 1>
  %9 = xor <2 x i64> %8, zeroinitializer
  %10 = add nsw <2 x i64> %9, zeroinitializer
  %11 = shufflevector <2 x i64> %10, <2 x i64> undef, <3 x i32> <i32 0, i32 1,
i32 undef>
  %12 = insertelement <3 x i64> %11, i64 0, i32 2
  %13 = shufflevector <3 x i64> %12, <3 x i64> undef, <4 x i32> <i32 0, i32 1,
i32 2, i32 3>
  %14 = shufflevector <4 x i64> %13, <4 x i64> undef, <2 x i32> <i32 0, i32 1>
  %15 = bitcast <2 x i64> %14 to <4 x i32>
  %16 = shufflevector <4 x i32> %15, <4 x i32> undef, <4 x i32> <i32 0, i32 2,
i32 0, i32 2>
  %17 = bitcast <4 x i32> %16 to <2 x i64>
  %18 = shufflevector <2 x i64> %17, <2 x i64> undef, <2 x i32> <i32 0, i32 2>
  %19 = bitcast <2 x i64> %18 to <4 x i32>
  %20 = shufflevector <4 x i32> %19, <4 x i32> undef, <3 x i32> <i32 0, i32 1,
i32 2>
  %21 = or <3 x i32> %20, zeroinitializer
  store <3 x i32> %21, <3 x i32> addrspace(1)* undef, align 16
  ret void
}
