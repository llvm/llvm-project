; RUN: llc -mtriple=hexagon < %s | FileCheck %s
; Check for memd(base + #offset), instead of memd(base + reg<<#c).
; CHECK: memd(r{{[0-9]+}}+#

define void @fred(i32 %p, i64 %v) #0 {
  %t0 = add i32 %p, 4
  %t1 = inttoptr i32 %t0 to ptr
  store i64 %v, ptr %t1
  ret void
}

attributes #0 = { nounwind "target-cpu"="hexagonv60" }
