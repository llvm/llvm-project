; RUN: llc --mtriple=mips-mti-linux-gnu < %s -debug 2>&1 | FileCheck %s --check-prefixes=CHECK,MIPS
; RUN: llc --mtriple=mips-mti-linux-gnu < %s -mattr=+micromips -debug 2>&1 | FileCheck %s --check-prefixes=CHECK,MM

; REQUIRES: asserts

; Test that the correct mul instruction is selected upfront.

; CHECK-LABEL: Instruction selection ends:
; MIPS: t{{[0-9]+}}: i32,i32 = MUL t{{[0-9]+}}, t{{[0-9]+}}
; MM: t{{[0-9]+}}: i32,i32 = MUL_MM t{{[0-9]+}}, t{{[0-9]+}}

define i32 @mul(i32 %a, i32 %b) {
entry:
  %0 = mul i32 %a, %b
  ret i32 %0
}
