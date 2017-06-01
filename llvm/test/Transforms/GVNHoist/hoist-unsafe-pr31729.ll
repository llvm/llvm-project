; RUN: opt -gvn-hoist -S < %s | FileCheck %s

; Check that urem is not hoisted.
; CHECK-LABEL: @main
; CHECK: urem
; CHECK: urem
; CHECK: urem

@g_x_s = global i32 -470211272, align 4
@g_z_s = global i32 2007237709, align 4
@g_x_u = global i32 282475249, align 4
@g_z_u = global i32 984943658, align 4
@g_m = global i32 16807, align 4
@res = common global i32 0, align 4

; Function Attrs:
define i64 @func() #0 {
entry:
  ret i64 1
}

; Function Attrs:
define i32 @main() {
entry:
  %0 = load volatile i32, i32* @g_x_s, align 4
  %1 = load volatile i32, i32* @g_z_s, align 4
  %2 = load volatile i32, i32* @g_x_u, align 4
  %3 = load volatile i32, i32* @g_z_u, align 4
  %4 = load volatile i32, i32* @g_m, align 4
  %call = call i64 @func() #4
  %conv = sext i32 %1 to i64
  %cmp = icmp ne i64 %call, %conv
  br i1 %cmp, label %if.end, label %lor.lhs.false

lor.lhs.false:
  %div = udiv i32 %4, %1
  %rem = urem i32 %0, %div
  %cmp2 = icmp eq i32 %rem, 0
  br i1 %cmp2, label %if.end, label %if.then

if.then:
  br label %cleanup

if.end:
  %call4 = call i64 @func() #4
  %conv5 = zext i32 %3 to i64
  %cmp6 = icmp ne i64 %call4, %conv5
  br i1 %cmp6, label %if.end14, label %lor.lhs.false8

lor.lhs.false8:
  %div9 = udiv i32 %4, %3
  %rem10 = urem i32 %0, %div9
  %cmp11 = icmp eq i32 %rem10, 0
  br i1 %cmp11, label %if.end14, label %if.then13

if.then13:
  br label %cleanup

if.end14:
  %call15 = call i64 @func() #4
  %cmp17 = icmp ne i64 %call15, %conv
  br i1 %cmp17, label %if.end25, label %lor.lhs.false19

lor.lhs.false19:
  %div20 = udiv i32 %4, %1
  %rem21 = urem i32 %0, %div20
  %cmp22 = icmp eq i32 %rem21, 0
  br i1 %cmp22, label %if.end25, label %if.then24

if.then24:
  br label %cleanup

if.end25:
  br label %cleanup

cleanup:
  %retval.0 = phi i32 [ 0, %if.end25 ], [ 1, %if.then24 ], [ 1, %if.then13 ], [ 1, %if.then ]
  ret i32 %retval.0
}

attributes #0 = { minsize noinline nounwind optsize uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
