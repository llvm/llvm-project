; RUN: llc -O0 -march=hexagon < %s | FileCheck %s
; CHECK: sub(r29,r[[REG:[0-9]+]])
; CHECK: r29 = r[[REG]]

target triple = "hexagon-unknown-unknown"

; Function Attrs: nounwind uwtable
define void @foo(i32 %n) #0 {
entry:
  %x = alloca i32, i32 %n
  %0 = bitcast i32* %x to i8*
  call void @bar(i8* %0)
  ret void
}

declare void @bar(i8*) #0

attributes #0 = { nounwind }
