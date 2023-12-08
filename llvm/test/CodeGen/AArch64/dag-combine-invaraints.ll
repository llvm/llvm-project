; RUN: llc -mtriple=arm64-apple-darwin8.0 -relocation-model=pic -O1 < %s | FileCheck %s

@.str2 = private unnamed_addr constant [9 x i8] c"_%d____\0A\00", align 1

; Function Attrs: nounwind ssp
define i32 @main(i32 %argc, ptr %argv) #0 {
main_:
  %tmp = alloca i32, align 4
  %i32T = alloca i32, align 4
  %i32F = alloca i32, align 4
  %i32X = alloca i32, align 4
  store i32 %argc, ptr %tmp
  store i32 15, ptr %i32T, align 4
  store i32 5, ptr %i32F, align 4
  %tmp6 = load i32, ptr %tmp, align 4
  %tmp7 = icmp ne i32 %tmp6, 0
  %tmp8 = xor i1 %tmp7, true
  %tmp9 = load i32, ptr %i32T, align 4
  %tmp10 = load i32, ptr %i32F, align 4
  %DHSelect = select i1 %tmp8, i32 %tmp9, i32 %tmp10
  store i32 %DHSelect, ptr %i32X, align 4
  %tmp15 = load i32, ptr %i32X, align 4
  %tmp17 = call i32 (ptr, ...) @printf(ptr @.str2, i32 %tmp15)
  ret i32 0

; CHECK: main:
; CHECK-DAG: mov {{.*}}, #15
; CHECK-DAG: mov {{.*}}, #5
; CHECK: csel
}


declare i32 @printf(ptr, ...) #1

attributes #0 = { nounwind ssp "less-precise-fpmad"="false" "frame-pointer"="all" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { "less-precise-fpmad"="false" "frame-pointer"="all" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
