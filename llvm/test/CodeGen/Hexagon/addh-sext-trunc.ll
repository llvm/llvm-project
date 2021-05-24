; RUN: llc -march=hexagon < %s | FileCheck %s
; CHECK: r{{[0-9]+}} = add(r{{[0-9]+}}.{{L|l}},r{{[0-9]+}}.{{H|h}})

target datalayout = "e-p:32:32:32-i64:64:64-i32:32:32-i16:16:16-i1:32:32-f64:64:64-f32:32:32-v64:64:64-v32:32:32-a0:0-n16:32"
target triple = "hexagon-unknown-none"


define i32 @foo(i16 %a, i32 %b) #0 {
  %and = and i16 %a, -4
  %conv3 = sext i16 %and to i32
  %add13 = mul i32 %b, 65536
  %sext = add i32 %add13, 262144
  %phitmp = ashr exact i32 %sext, 16
  ret i32 %phitmp
}


attributes #0 = { nounwind readonly "less-precise-fpmad"="false" "frame-pointer"="non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "unsafe-fp-math"="false" "use-soft-float"="false" }

!0 = !{!"short", !1}
!1 = !{!"omnipotent char", !2}
!2 = !{!"Simple C/C++ TBAA"}
!3 = !{!"any pointer", !1}
