; RUN: llc -debug < %s -o /dev/null 2>&1 | FileCheck %s

; Verify that the backend correctly overrides the optimization level
; of optnone functions during instruction selection.

define float @foo(float %x) #0 {
entry:
  %add = fadd fast float %x, %x
  %add1 = fadd fast float %add, %x
  ret float %add1
}

; CHECK-NOT: Changing optimization level for Function foo
; CHECK-NOT: Restoring optimization level for Function foo

; Function Attrs: noinline optnone
define float @fooWithOptnone(float %x) #1 {
entry:
  %add = fadd fast float %x, %x
  %add1 = fadd fast float %add, %x
  ret float %add1
}

; CHECK: Changing optimization level for Function fooWithOptnone
; CHECK-NEXT: Before: -O2 ; After: -O0

; CHECK: Restoring optimization level for Function fooWithOptnone
; CHECK-NEXT: Before: -O0 ; After: -O2

attributes #0 = { "unsafe-fp-math"="true" }
attributes #1 = { noinline optnone "unsafe-fp-math"="true" }
