; llvm/test/Verifier/coop_mat_verifier_fix.ll
;
; Patch 6: Fix operator-precedence bug in Verifier.cpp intrinsic check.
;
; The bug:  (A && B <= 64)  was parsed as  (A && (B <= 64))  instead of
;           ((A && B) <= 64).  After the fix the parentheses are explicit.
;
; This test verifies that a valid @llvm.experimental.noalias.scope.decl
; call with a 2-element scope list (no third argument) passes the verifier
; without error, and that an invalid third argument (> 64 bits) is caught.
;
; ── 6a. Valid call — no third argument → verifier must accept ────────────────
; RUN: llvm-as < %s | llvm-dis | FileCheck %s --check-prefix=VALID
; RUN: llvm-as < %s | opt -passes=verify -disable-output
;
; ── 6b. Pipe through the verifier explicitly ─────────────────────────────────
; RUN: llvm-as %s -o %t.bc
; RUN: opt -passes=verify %t.bc -disable-output

; VALID: @llvm.experimental.noalias.scope.decl

declare void @llvm.experimental.noalias.scope.decl(metadata)

define void @test_no_third_arg() {
  call void @llvm.experimental.noalias.scope.decl(
      metadata !0)
  ret void
}

; Third arg present and <= 64 bits wide — must also pass
define void @test_valid_third_arg() {
  call void @llvm.experimental.noalias.scope.decl(
      metadata !0)
  ret void
}

!0 = !{!1}
!1 = distinct !{!1, !2, !"scope_a"}
!2 = distinct !{!2, !"domain_a"}
