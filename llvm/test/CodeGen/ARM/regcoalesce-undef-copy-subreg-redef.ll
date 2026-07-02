; RUN: llc -mtriple=armv8a-unknown-linux -verify-machineinstrs < %s -o /dev/null

; Regression test for issue #202263. Lowering the REG_SEQUENCE that builds the
; <2 x i64> keeps a COPY of an undefined value which feeds a later partial
; subregister redef. When RegisterCoalescer eliminates that copy, the redef
; must be turned into a read-undef def. Otherwise the coalescer asserts with
; "Instruction is reading nonexistent value" on targets that do not track
; subregister liveness (armv8a A-profile has no MVE).

target triple = "armv8a-unknown-linux"

define void @init(i64 %x, i1 %c, ptr %p) {
entry:
  %v0 = insertelement <2 x i64> poison, i64 %x, i64 1
  br i1 %c, label %exit, label %body

body:
  %v1 = insertelement <2 x i64> %v0, i64 1, i64 0
  store <2 x i64> %v1, ptr %p, align 8
  br label %exit

exit:
  ret void
}
