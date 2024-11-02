; RUN: opt -S < %s | FileCheck %s -strict-whitespace

; CHECK: {{^}}; Function Attrs: nounwind ssp memory(none) uwtable{{$}}
; CHECK-NEXT: define void @test1() #0
define void @test1() #0 {
  ret void
}

attributes #0 = { nounwind ssp "less-precise-fpmad"="false" uwtable "frame-pointer"="all" readnone "no-infs-fp-math"="false" "no-nans-fp-math"="false" "unsafe-fp-math"="false" "use-soft-float"="false" }
