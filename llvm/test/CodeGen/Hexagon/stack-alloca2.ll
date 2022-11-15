; RUN: llc -O0 -march=hexagon < %s | FileCheck %s
; CHECK: r[[AP:[0-9]+]] = and(r30,#-32)
; CHECK: sub(r29,r[[SP:[0-9]+]])
; CHECK: r29 = r[[SP]]
; CHECK: r1 = r[[AP]]
; CHECK: r1 = add(r1,#-32)

target triple = "hexagon-unknown-unknown"

; Function Attrs: nounwind uwtable
define void @foo(i32 %n) #0 {
entry:
  %x = alloca i32, i32 %n
  %y = alloca i32, align 32
  %0 = bitcast i32* %x to i8*
  %1 = bitcast i32* %y to i8*
  call void @bar(i8* %0, i8* %1)
  ret void
}

declare void @bar(i8*, i8* %y) #0

attributes #0 = { nounwind }
