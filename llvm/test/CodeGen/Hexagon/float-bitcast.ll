; RUN: llc -march=hexagon < %s | FileCheck %s
; All of these should be no-ops. Check this with -O0, to make sure
; that no register copies are generated at any time.

; CHECK-LABEL: f0:
; CHECK-NOT: r{{[0-9]+}} = r{{[0-9]+}}
; CHECK: jumpr r31
define float @f0(i32 %a0) #0 {
b0:
  %v0 = bitcast i32 %a0 to float
  ret float %v0
}

; CHECK-LABEL: f1:
; CHECK-NOT: r{{[0-9]+}} = r{{[0-9]+}}
; CHECK: jumpr r31
define i32 @f1(float %a0) #0 {
b0:
  %v0 = bitcast float %a0 to i32
  ret i32 %v0
}

; CHECK-LABEL: f2:
; CHECK-NOT: r{{[0-9:]*}} = r{{[0-9:]*}}
; CHECK: jumpr r31
define double @f2(i64 %a0) #0 {
b0:
  %v0 = bitcast i64 %a0 to double
  ret double %v0
}

; CHECK-LABEL: f3:
; CHECK-NOT: r{{[0-9:]*}} = r{{[0-9:]*}}
; CHECK: jumpr r31
define i64 @f3(double %a0) #0 {
b0:
  %v0 = bitcast double %a0 to i64
  ret i64 %v0
}

attributes #0 = { nounwind "target-cpu"="hexagonv55" }
