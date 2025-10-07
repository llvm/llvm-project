; RUN: llc -mtriple=hexagon < %s | FileCheck %s

; CHECK-LABEL: enabled:
; CHECK: memw({{.*}}) += #1
define void @enabled(ptr %p) #0 {
  %v0 = load i32, ptr %p
  %v1 = add i32 %v0, 1
  store i32 %v1, ptr %p
  ret void
}

; CHECK-LABEL: disabled:
; CHECK-NOT: memw({{.*}}) += #1
define void @disabled(ptr %p) #1 {
  %v0 = load i32, ptr %p
  %v1 = add i32 %v0, 1
  store i32 %v1, ptr %p
  ret void
}

attributes #0 = { nounwind }
attributes #1 = { nounwind "target-features"="-memops" }

