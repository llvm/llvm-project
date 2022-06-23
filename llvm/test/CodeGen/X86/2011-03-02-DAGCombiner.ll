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
  store i32 0, ptr %retval
  %0 = load i32, ptr %K, align 4
  %1 = and i32 %0, -121
  %2 = or i32 %1, 32
  store i32 %2, ptr %K, align 4
  %3 = load i32, ptr %K, align 4
  %4 = lshr i32 %3, 3
  %bf.clear = and i32 %4, 15
  %conv = sitofp i32 %bf.clear to float
  %tmp = load float, ptr %F, align 4
  %sub = fsub float %tmp, %conv
  store float %sub, ptr %F, align 4
  %ld = getelementptr inbounds %struct.anon, ptr %F, i32 0, i32 1
  %tmp1 = load x86_fp80, ptr %ld, align 16
  %5 = load i32, ptr %K, align 4
  %6 = lshr i32 %5, 7
  %bf.clear2 = and i32 %6, 1
  %conv3 = uitofp i32 %bf.clear2 to x86_fp80
  %sub4 = fsub x86_fp80 %conv3, %tmp1
  %conv5 = fptoui x86_fp80 %sub4 to i32
  %bf.value = and i32 %conv5, 1
  %7 = and i32 %bf.value, 1
  %8 = shl i32 %7, 7
  %9 = load i32, ptr %K, align 4
  %10 = and i32 %9, -129
  %11 = or i32 %10, %8
  store i32 %11, ptr %K, align 4
  %call = call i32 (...) @iequals(i32 1841, i32 %bf.value, i32 0)
  %12 = load i32, ptr %retval
  ret i32 %12
}

declare i32 @iequals(...)
