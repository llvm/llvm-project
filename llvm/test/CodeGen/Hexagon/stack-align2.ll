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
  %0 = bitcast i32* %x to i8*
  %1 = bitcast i32* %y to i8*
  %2 = bitcast i32* %z to i8*
  %3 = bitcast i32* %w to i8*
  call void @bar(i8* %0, i8* %1, i8* %2, i8* %3)
  ret void
}

declare void @bar(i8*, i8*, i8*, i8*) #0

attributes #0 = { nounwind }
