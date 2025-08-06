; This tests that llc accepts all valid Xtensa CPUs

; RUN: llc < %s --mtriple=xtensa --mcpu=esp8266 2>&1 | FileCheck %s
; RUN: llc < %s --mtriple=xtensa --mcpu=esp32 2>&1 | FileCheck %s
; RUN: llc < %s --mtriple=xtensa --mcpu=generic 2>&1 | FileCheck %s

; CHECK-NOT: {{.*}}  is not a recognized processor for this target
; INVALID: {{.*}}  is not a recognized processor for this target

define i32 @f(i32 %z) {
	ret i32 0
}
