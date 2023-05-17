; RUN: llc -O0 -march=hexagon < %s | FileCheck %s
; CHECK: and(r29,#-128)
; CHECK-DAG: add(r29,#0)
; CHECK-DAG: add(r29,#64)
; CHECK-DAG: add(r29,#96)
; CHECK-DAG: add(r29,#124)

target triple = "hexagon-unknown-unknown"

; Function Attrs: nounwind uwtable
define void @foo() #0 {
entry:
  %x = alloca i32, align 4
  %y = alloca i32, align 32
  %z = alloca i32, align 64
  %w = alloca i32, align 128
  call void @bar(ptr %x, ptr %y, ptr %z, ptr %w)
  ret void
}

declare void @bar(ptr, ptr, ptr, ptr) #0

attributes #0 = { nounwind }
