; RUN: llc < %s -mtriple=avr | FileCheck %s

; This test ensures that the backend can lower returns of struct values.
; It does not check how these are lowered.
;
; In the past, this code used to return an error
;
; Assertion `InVals.size() == Ins.size() && "LowerFormalArguments didn't emit the correct number of values!"' failed.
;
; This feature was first implemented in r325474.

declare i8 @do_something(i8 %val)

; CHECK-LABEL: main
define { i1, i8 } @main(i8) #2 {
entry:
  %1 = call zeroext i8 @do_something(i8 zeroext %0)
  %2 = insertvalue { i1, i8 } { i1 true, i8 undef }, i8 %1, 1
  ret { i1, i8 } %2
}

