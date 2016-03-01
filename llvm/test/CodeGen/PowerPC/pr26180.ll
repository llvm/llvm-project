; RUN: llc -mcpu=generic -mtriple=powerpc64le-unknown-unknown -O0 < %s | FileCheck %s --check-prefix=GENERIC
; RUN: llc -mcpu=ppc -mtriple=powerpc64le-unknown-unknown -O0 < %s | FileCheck %s

define i32 @bad(double %x) {
  %1 = fptoui double %x to i32
  ret i32 %1
}

; CHECK: fctidz 1, 1
; CHECK: stfd 1, [[OFF:.*]](1)
; CHECK: lwz {{[0-9]*}}, [[OFF]](1)
; GENERIC: fctiwuz 1, 1
; GENERIC: stfd 1, [[OFF:.*]](1)
; GENERIC: lwz {{[0-9]*}}, [[OFF]](1)
