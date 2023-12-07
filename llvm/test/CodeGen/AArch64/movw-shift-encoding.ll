; RUN: llc -mtriple=aarch64-linux-gnu < %s -show-mc-encoding -code-model=large | FileCheck %s

@var = dso_local global i32 0

; CodeGen should ensure that the correct shift bits are set, because the linker
; isn't going to!

define dso_local ptr @get_var() {
  ret ptr @var

; CHECK: movz    x0, #:abs_g0_nc:var // encoding: [0bAAA00000,A,0b100AAAAA,0xd2]
; CHECK: movk    x0, #:abs_g1_nc:var // encoding: [0bAAA00000,A,0b101AAAAA,0xf2]
; CHECK: movk    x0, #:abs_g2_nc:var // encoding: [0bAAA00000,A,0b110AAAAA,0xf2]
; CHECK: movk    x0, #:abs_g3:var    // encoding: [0bAAA00000,A,0b111AAAAA,0xf2]
}
