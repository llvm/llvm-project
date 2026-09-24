; RUN: not --crash llc -mtriple=x86_64-unknown-unknown -x86-indirect-branch-tracking < %s 2>&1 | FileCheck %s

; NOTRACK tail call is not implemented, so nocf_check just disables tail call.
define void @NoCfCheckTail(ptr %p) #1 {
; CHECK: failed to perform tail call elimination on a call site marked musttail
  %f = load ptr, ptr %p, align 8
  musttail call void %f(ptr null) #2
  ret void
}

attributes #0 = { nocf_check noinline nounwind optnone uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" "use-soft-float"="false" }
attributes #1 = { noinline nounwind optnone uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" "use-soft-float"="false" }
attributes #2 = { nocf_check }

!llvm.module.flags = !{!0}

!0 = !{i32 8, !"cf-protection-branch", i32 1}
