; RUN: llc < %s -mtriple=avr | FileCheck %s

; The bug can be found here:
; https://github.com/rust-lang/rust/issues/98167
;
; In this test, `extractvalue` + `call` generate a copy with overlapping
; registers (`$r25r24 = COPY $r24r23`) that used to be expanded incorrectly.

define void @main() {
; CHECK-LABEL: main:
; CHECK: rcall foo
; CHECK-NEXT: mov r25, r24
; CHECK-NEXT: mov r24, r23
; CHECK-NEXT: rcall bar
  %1 = call { i8, i16 } @foo()
  %2 = extractvalue { i8, i16 } %1, 1
  call void @bar(i16 %2)
  ret void
}

declare { i8, i16 } @foo()
declare void @bar(i16 %0)
