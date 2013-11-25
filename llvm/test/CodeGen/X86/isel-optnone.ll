; RUN: llc -O2 -march=x86 < %s | FileCheck %s

define i32* @fooOptnone(i32* %p, i32* %q, i32** %z) #0 {
entry:
  %r = load i32* %p
  %s = load i32* %q
  %y = load i32** %z

  %t0 = add i32 %r, %s
  %t1 = add i32 %t0, 1
  %t2 = getelementptr i32* %y, i32 1
  %t3 = getelementptr i32* %t2, i32 %t1

  ret i32* %t3

; 'optnone' should use fast-isel which will not produce 'lea'.
; CHECK-LABEL: fooOptnone:
; CHECK-NOT:   lea
; CHECK:       ret
}

define i32* @fooNormal(i32* %p, i32* %q, i32** %z) #1 {
entry:
  %r = load i32* %p
  %s = load i32* %q
  %y = load i32** %z

  %t0 = add i32 %r, %s
  %t1 = add i32 %t0, 1
  %t2 = getelementptr i32* %y, i32 1
  %t3 = getelementptr i32* %t2, i32 %t1

  ret i32* %t3

; Normal ISel will produce 'lea'.
; CHECK-LABEL: fooNormal:
; CHECK:       lea
; CHECK:       ret
}

attributes #0 = { nounwind optnone noinline }
attributes #1 = { nounwind }
