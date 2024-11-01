; RUN: opt < %s -passes='print<branch-prob>' -disable-output 2>&1 | FileCheck %s

; In this test, the else clause is taken about 90% of the time. This was not
; reflected in the probability computation because the weight is larger than
; the branch weight cap (about 2 billion).
;
; CHECK: edge for.body -> if.then probability is 0x0cccba45 / 0x80000000 = 10.00%
; CHECK: edge for.body -> if.else probability is 0x733345bb / 0x80000000 = 90.00% [HOT edge]

@y = common global i64 0, align 8
@x = common global i64 0, align 8
@.str = private unnamed_addr constant [17 x i8] c"x = %lu\0Ay = %lu\0A\00", align 1

; Function Attrs: inlinehint nounwind uwtable
define i32 @main() #0 {
entry:
  %retval = alloca i32, align 4
  %i = alloca i64, align 8
  store i32 0, ptr %retval
  store i64 0, ptr @y, align 8
  store i64 0, ptr @x, align 8
  call void @srand(i32 422304) #3
  store i64 0, ptr %i, align 8
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %0 = load i64, ptr %i, align 8
  %cmp = icmp ult i64 %0, 13000000000
  br i1 %cmp, label %for.body, label %for.end, !prof !1

for.body:                                         ; preds = %for.cond
  %call = call i32 @rand() #3
  %conv = sitofp i32 %call to double
  %mul = fmul double %conv, 1.000000e+02
  %div = fdiv double %mul, 0x41E0000000000000
  %cmp1 = fcmp ogt double %div, 9.000000e+01
  br i1 %cmp1, label %if.then, label %if.else, !prof !2

if.then:                                          ; preds = %for.body
  %1 = load i64, ptr @x, align 8
  %inc = add i64 %1, 1
  store i64 %inc, ptr @x, align 8
  br label %if.end

if.else:                                          ; preds = %for.body
  %2 = load i64, ptr @y, align 8
  %inc3 = add i64 %2, 1
  store i64 %inc3, ptr @y, align 8
  br label %if.end

if.end:                                           ; preds = %if.else, %if.then
  br label %for.inc

for.inc:                                          ; preds = %if.end
  %3 = load i64, ptr %i, align 8
  %inc4 = add i64 %3, 1
  store i64 %inc4, ptr %i, align 8
  br label %for.cond

for.end:                                          ; preds = %for.cond
  %4 = load i64, ptr @x, align 8
  %5 = load i64, ptr @y, align 8
  %call5 = call i32 (ptr, ...) @printf(ptr @.str, i64 %4, i64 %5)
  ret i32 0
}

; Function Attrs: nounwind
declare void @srand(i32) #1

; Function Attrs: nounwind
declare i32 @rand() #1

declare i32 @printf(ptr, ...) #2

attributes #0 = { inlinehint nounwind uwtable "less-precise-fpmad"="false" "frame-pointer"="all" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+sse,+sse2" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind "less-precise-fpmad"="false" "frame-pointer"="all" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+sse,+sse2" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { "less-precise-fpmad"="false" "frame-pointer"="all" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+sse,+sse2" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #3 = { nounwind }

!llvm.ident = !{!0}

!0 = !{!"clang version 3.7.0 (trunk 236218) (llvm/trunk 236235)"}
!1 = !{!"branch_weights", i32 -1044967295, i32 1}
!2 = !{!"branch_weights", i32 433323762, i32 -394957723}
