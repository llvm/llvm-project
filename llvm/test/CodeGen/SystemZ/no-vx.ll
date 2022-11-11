; RUN: llc < %s -mtriple=s390x-linux-gnu -o - | FileCheck %s

; This test verifies that passing the "-vector" feature disables *any*
; use of vector instructions, even if +vector-enhancements-1 if given.

; Function Attrs: noinline nounwind optnone
define dso_local signext i32 @foo(ptr %0, ptr %1) #0 {
; CHECK-LABEL: @foo
; CHECK-NOT: vl
; CHECK-NOT: vst
; CHECK: br %r14
entry:
  %arg1.addr = alloca fp128, align 8
  %arg2.addr = alloca fp128, align 8
  %indirect-arg-temp = alloca fp128, align 8
  %indirect-arg-temp1 = alloca fp128, align 8
  %arg1 = load fp128, ptr %0, align 8
  %arg2 = load fp128, ptr %1, align 8
  store fp128 %arg1, ptr %arg1.addr, align 8
  store fp128 %arg2, ptr %arg2.addr, align 8
  %2 = load fp128, ptr %arg1.addr, align 8
  %3 = load fp128, ptr %arg2.addr, align 8
  store fp128 %2, ptr %indirect-arg-temp, align 8
  store fp128 %3, ptr %indirect-arg-temp1, align 8
  %call = call signext i32 @bar(i32 signext 2, ptr %indirect-arg-temp, ptr %indirect-arg-temp1)
  ret i32 %call
}

declare dso_local signext i32 @bar(i32 signext, ptr, ptr) #1

attributes #0 = { noinline nounwind optnone "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="z14" "target-features"="+transactional-execution,+vector-enhancements-1,-vector" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="z14" "target-features"="+transactional-execution,+vector-enhancements-1,-vector" "unsafe-fp-math"="false" "use-soft-float"="false" }

