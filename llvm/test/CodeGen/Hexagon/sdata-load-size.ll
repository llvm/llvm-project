; RUN: llc -march=hexagon -hexagon-small-data-threshold=8 < %s | FileCheck %s
; CHECK: = memd(gp+#g0)
; If an object will be placed in .sdata, do not shrink any references to it.
; In this case, g0 must be loaded via memd.

target triple = "hexagon"

@g0 = common global i64 0, align 8

define i32 @f0() #0 {
entry:
  %v0 = load i64, i64* @g0, align 8
  %v1 = trunc i64 %v0 to i8
  %v2 = zext i8 %v1 to i32
  ret i32 %v2
}

attributes #0 = { nounwind "target-cpu"="hexagonv60" "target-features"="+small-data" }

