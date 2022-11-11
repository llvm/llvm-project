; RUN: llc -verify-machineinstrs -O0 -mtriple=powerpc64-unknown-linux-gnu -mcpu=pwr7 < %s
target datalayout = "E-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-f128:128:128-v128:128:128-n32:64"
target triple = "powerpc64-unknown-linux-gnu"

; Function Attrs: nounwind
define internal ptr @func_65(ptr %p_66) #0 {
entry:
  br i1 undef, label %for.body, label %for.end731

for.body:                                         ; preds = %entry
  %0 = load i32, ptr undef, align 4
  %or31 = or i32 %0, 319143828
  store i32 %or31, ptr undef, align 4
  %cmp32 = icmp eq i32 319143828, %or31
  %conv33 = zext i1 %cmp32 to i32
  %conv34 = sext i32 %conv33 to i64
  %call35 = call i64 @safe_mod_func_uint64_t_u_u(i64 %conv34, i64 -10)
  unreachable

for.end731:                                       ; preds = %entry
  ret ptr undef
}

; Function Attrs: nounwind
declare i64 @safe_mod_func_uint64_t_u_u(i64, i64) #0

attributes #0 = { nounwind "less-precise-fpmad"="false" "frame-pointer"="all" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "unsafe-fp-math"="false" "use-soft-float"="false" }
