; Test that llvm-reduce can remove uninteresting attributes.
;
; RUN: rm -rf %t
; RUN: llvm-reduce --test %python --test-arg %p/Inputs/remove-attributes.py %s -o %t
; RUN: cat %t | FileCheck  %s

define void @a() #0 {
  ret void
}
define void @b() #1 {
  ret void
}

; CHECK: attributes #0 = { "use-soft-float"="false" }
attributes #0 = { norecurse noreturn nounwind readnone uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="none" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "patchable-function-entry"="2" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { norecurse }
