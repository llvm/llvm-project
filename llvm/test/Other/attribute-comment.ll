; RUN: opt -S < %s | FileCheck %s -strict-whitespace

; CHECK: {{^}}; Function Attrs: nounwind readnone ssp uwtable{{$}}
; CHECK-NEXT: define void @test1() #0
define void @test1() #0 {
  ret void
}

attributes #0 = { nounwind ssp uwtable "frame-pointer"="all" readnone "use-soft-float"="false" }
