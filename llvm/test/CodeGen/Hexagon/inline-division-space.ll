; Test for checking division is inlined or not in case of Os.
; RUN: llc -O2 -march=hexagon   < %s | FileCheck  %s

; Function Attrs: optsize
define dso_local i32 @testInt(i32 %a, i32 %b) local_unnamed_addr #0 {
entry:
;CHECK: call __hexagon_divsi3
  %div = sdiv i32 %a, %b
  %conv = sitofp i32 %div to float
  %conv1 = fptosi float %conv to i32
  ret i32 %conv1
}

; Function Attrs: optsize
define dso_local float @testFloat(float %a, float %b) local_unnamed_addr #0 {
entry:
;CHECK: call __hexagon_divsf3
  %div = fdiv float %a, %b
  ret float %div
}

; Function Attrs: optsize
define dso_local double @testDouble(double %a, double %b) local_unnamed_addr #0 {
entry:
;CHECK: call __hexagon_divdf3
  %div = fdiv double %a, %b
  ret double %div
}

attributes #0 = { optsize }
