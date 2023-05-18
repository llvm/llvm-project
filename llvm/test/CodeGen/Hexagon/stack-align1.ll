; RUN: llc -O0 -march=hexagon < %s | FileCheck %s
; CHECK: and(r29,#-32)
; CHECK-DAG: add(r29,#0)
; CHECK-DAG: add(r29,#28)

target triple = "hexagon-unknown-unknown"

; Function Attrs: nounwind uwtable
define void @foo() #0 {
entry:
  %x = alloca i32, align 4
  %y = alloca i32, align 32
  call void @bar(ptr %x, ptr %y)
  ret void
}

declare void @bar(ptr, ptr) #0

attributes #0 = { nounwind }
