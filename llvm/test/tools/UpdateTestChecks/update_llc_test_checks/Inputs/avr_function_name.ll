; Check that we accept functions with '$' in the name.

; RUN: llc -mtriple=avr < %s | FileCheck %s

define hidden i8 @"_Z54bar$ompvariant$bar"() {
entry:
  ret i8 2
}
