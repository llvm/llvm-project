; RUN: llc -O2 -mtriple=i686-- < %s | FileCheck %s

define ptr @fooOptnone(ptr %p, ptr %q, ptr %z) #0 {
entry:
  %r = load i32, ptr %p
  %s = load i32, ptr %q
  %y = load ptr, ptr %z

  %t0 = add i32 %r, %s
  %t1 = add i32 %t0, 1
  %t2 = getelementptr i32, ptr %y, i32 1
  %t3 = getelementptr i32, ptr %t2, i32 %t1

  ret ptr %t3

; 'optnone' should use fast-isel which will not produce 'lea'.
; CHECK-LABEL: fooOptnone:
; CHECK-NOT:   lea
; CHECK:       ret
}

define ptr @fooNormal(ptr %p, ptr %q, ptr %z) #1 {
entry:
  %r = load i32, ptr %p
  %s = load i32, ptr %q
  %y = load ptr, ptr %z

  %t0 = add i32 %r, %s
  %t1 = add i32 %t0, 1
  %t2 = getelementptr i32, ptr %y, i32 1
  %t3 = getelementptr i32, ptr %t2, i32 %t1

  ret ptr %t3

; Normal ISel will produce 'lea'.
; CHECK-LABEL: fooNormal:
; CHECK:       lea
; CHECK:       ret
}

attributes #0 = { nounwind optnone noinline }
attributes #1 = { nounwind }
