; RUN: llc -march=hexagon < %s | FileCheck %s
; CHECK: r0 = add(r0,r1)

; Allow simple structures to be returned by value.

%s = type { i32, i32 }

declare %s @foo() #0

define i32 @fred() #0 {
  %t0 = call %s @foo()
  %x = extractvalue %s %t0, 0
  %y = extractvalue %s %t0, 1
  %r = add i32 %x, %y
  ret i32 %r
}

attributes #0 = { nounwind }
