; RUN: llc < %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-apple-darwin11.0.0"

%0 = type { i8, [3 x i8] }
%struct.anon = type { float, x86_fp80 }

define i32 @main() nounwind ssp {
entry:
  %retval = alloca i32, align 4
  %F = alloca %struct.anon, align 16
  %K = alloca %0, align 4
  store i32 0, i32* %retval
  %0 = bitcast %0* %K to i32*
  %1 = load i32, i32* %0, align 4
  %2 = and i32 %1, -121
  %3 = or i32 %2, 32
  store i32 %3, i32* %0, align 4
  %4 = bitcast %0* %K to i32*
  %5 = load i32, i32* %4, align 4
  %6 = lshr i32 %5, 3
  %bf.clear = and i32 %6, 15
  %conv = sitofp i32 %bf.clear to float
  %f = getelementptr inbounds %struct.anon, %struct.anon* %F, i32 0, i32 0
  %tmp = load float, float* %f, align 4
  %sub = fsub float %tmp, %conv
  store float %sub, float* %f, align 4
  %ld = getelementptr inbounds %struct.anon, %struct.anon* %F, i32 0, i32 1
  %tmp1 = load x86_fp80, x86_fp80* %ld, align 16
  %7 = bitcast %0* %K to i32*
  %8 = load i32, i32* %7, align 4
  %9 = lshr i32 %8, 7
  %bf.clear2 = and i32 %9, 1
  %conv3 = uitofp i32 %bf.clear2 to x86_fp80
  %sub4 = fsub x86_fp80 %conv3, %tmp1
  %conv5 = fptoui x86_fp80 %sub4 to i32
  %bf.value = and i32 %conv5, 1
  %10 = bitcast %0* %K to i32*
  %11 = and i32 %bf.value, 1
  %12 = shl i32 %11, 7
  %13 = load i32, i32* %10, align 4
  %14 = and i32 %13, -129
  %15 = or i32 %14, %12
  store i32 %15, i32* %10, align 4
  %call = call i32 (...) @iequals(i32 1841, i32 %bf.value, i32 0)
  %16 = load i32, i32* %retval
  ret i32 %16
}

declare i32 @iequals(...)
