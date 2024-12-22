; RUN: llc -march=hexagon < %s | FileCheck %s
; CHECK: r{{[0-9]+}}:{{[0-9]+}} = add(r{{[0-9]+}}:{{[0-9+]}},r{{[0-9]+}}:{{[0-9]+}}):raw:{{..}}

define i64 @f0(i32 %a0, i64 %a1) #0 {
b0:
  %v0 = sext i32 %a0 to i64
  %v1 = add nsw i64 %v0, %a1
  ret i64 %v1
}

attributes #0 = { nounwind readnone }
