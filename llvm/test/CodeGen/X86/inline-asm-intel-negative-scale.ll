; RUN: not llc -mtriple=x86_64-unknown-linux-gnu %s -o /dev/null 2>&1 | FileCheck %s

; Verify that inline asm with inteldialect correctly rejects a negative scale
; factor in an address expression (e.g. [rax - 8 * rdx]) instead of silently
; inverting the sign and producing [rax + 8*rdx].

; CHECK: error: Scale can't be negative

define i64 @test_neg_scale(i64 %0) {
  %result = call i64 asm sideeffect alignstack inteldialect
    "xor rax, rax\0Alea rax, [rax - 8 * rdx]",
    "=&{ax},{dx},~{dirflag},~{fpsr},~{flags},~{memory}"(i64 %0)
  ret i64 %result
}
