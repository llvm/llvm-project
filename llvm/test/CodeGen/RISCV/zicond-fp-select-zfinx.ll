; RUN: llc -mtriple=riscv64 -mattr=+zfinx,+zicond -verify-machineinstrs < %s | FileCheck %s --check-prefix=RV64ZFINX_ZICOND
; RUN: llc -mtriple=riscv64 -mattr=+zfinx           -verify-machineinstrs < %s | FileCheck %s --check-prefix=RV64ZFINX_NOZICOND
; RUN: llc -mtriple=riscv64 -mattr=+zdinx,+zicond -verify-machineinstrs < %s | FileCheck %s --check-prefix=RV64ZDINX_ZICOND
; RUN: llc -mtriple=riscv64 -mattr=+zdinx           -verify-machineinstrs < %s | FileCheck %s --check-prefix=RV64ZDINX_NOZICOND
; RUN: llc -mtriple=riscv64 -mattr=+f -verify-machineinstrs < %s | FileCheck %s --check-prefix=RV64F
; RUN: llc -mtriple=riscv64 -mattr=+d -verify-machineinstrs < %s | FileCheck %s --check-prefix=RV64D

; RUN: llc -mtriple=riscv32 -mattr=+zfinx,+zicond -verify-machineinstrs < %s | FileCheck %s --check-prefix=RV32ZFINX_ZICOND
; RUN: llc -mtriple=riscv32 -mattr=+zfinx           -verify-machineinstrs < %s | FileCheck %s --check-prefix=RV32ZFINX_NOZICOND
; RUN: llc -mtriple=riscv32 -mattr=+f -verify-machineinstrs < %s | FileCheck %s --check-prefix=RV32F


; This test checks that floating-point SELECT is lowered through integer
; SELECT (and thus to Zicond czero.* sequence) when FP values live in GPRs
; (Zfinx/Zdinx) and Zicond is enabled. When Zicond is disabled, we expect
; a branch-based lowering instead.

; -----------------------------------------------------------------------------
; float select with i1 condition (Zfinx)
; -----------------------------------------------------------------------------

define float @select_f32_i1(i1 %cond, float %t, float %f) nounwind {
; RV64ZFINX_ZICOND-LABEL: select_f32_i1:
; RV64ZFINX_ZICOND: czero
; RV64ZFINX_ZICOND: czero
; RV64ZFINX_ZICOND: or
; RV64ZFINX_ZICOND-NOT: b{{(eq|ne)z?}}
; RV64ZFINX_ZICOND: ret

; RV64ZFINX_NOZICOND-LABEL: select_f32_i1:
; RV64ZFINX_NOZICOND: b{{(eq|ne)z?}}
; RV64ZFINX_NOZICOND-NOT: czero.eqz
; RV64ZFINX_NOZICOND-NOT: czero.nez

; RV64F-LABEL: select_f32_i1:
; RV64F: b{{(eq|ne)z?}}
; RV64F-NOT: czero.eqz
; RV64F-NOT: czero.nez

; RV32ZFINX_ZICOND-LABEL: select_f32_i1:
; RV32ZFINX_ZICOND: czero
; RV32ZFINX_ZICOND: czero
; RV32ZFINX_ZICOND: or
; RV32ZFINX_ZICOND-NOT: b{{(eq|ne)z?}}
; RV32ZFINX_ZICOND: ret

; RV32ZFINX_NOZICOND-LABEL: select_f32_i1:
; RV32ZFINX_NOZICOND: b{{(eq|ne)z?}}
; RV32ZFINX_NOZICOND-NOT: czero.eqz
; RV32ZFINX_NOZICOND-NOT: czero.nez

; RV32F-LABEL: select_f32_i1:
; RV32F: b{{(eq|ne)z?}}
; RV32F-NOT: czero.eqz
; RV32F-NOT: czero.nez

entry:
  %sel = select i1 %cond, float %t, float %f
  ret float %sel
}

; -----------------------------------------------------------------------------
; double select with i1 condition (Zdinx)
; -----------------------------------------------------------------------------

define double @select_f64_i1(i1 %cond, double %t, double %f) nounwind {
; RV64ZDINX_ZICOND-LABEL: select_f64_i1:
; RV64ZDINX_ZICOND: czero
; RV64ZDINX_ZICOND: czero
; RV64ZDINX_ZICOND: or
; RV64ZDINX_ZICOND-NOT: b{{(eq|ne)z?}}
; RV64ZDINX_ZICOND: ret

; RV64ZDINX_NOZICOND-LABEL: select_f64_i1:
; RV64ZDINX_NOZICOND: b{{(eq|ne)z?}}
; RV64ZDINX_NOZICOND-NOT: czero.eqz
; RV64ZDINX_NOZICOND-NOT: czero.nez

; RV64D-LABEL: select_f64_i1:
; RV64D: b{{(eq|ne)z?}}
; RV64D-NOT: czero.eqz
; RV64D-NOT: czero.nez

entry:
  %sel = select i1 %cond, double %t, double %f
  ret double %sel
}

; -----------------------------------------------------------------------------
; double select with floating-point compare condition (a > b ? c : d), Zdinx
; -----------------------------------------------------------------------------

define double @select_f64_fcmp(double %a, double %b, double %c, double %d) nounwind {
; RV64ZDINX_ZICOND-LABEL: select_f64_fcmp:
; RV64ZDINX_ZICOND: czero
; RV64ZDINX_ZICOND: czero
; RV64ZDINX_ZICOND: or
; RV64ZDINX_ZICOND-NOT: b{{(eq|ne)z?}}
; RV64ZDINX_ZICOND: ret

; RV64ZDINX_NOZICOND-LABEL: select_f64_fcmp:
; RV64ZDINX_NOZICOND: b{{(eq|ne)z?}}
; RV64ZDINX_NOZICOND-NOT: czero.eqz
; RV64ZDINX_NOZICOND-NOT: czero.nez

; RV64D-LABEL: select_f64_fcmp:
; RV64D: b{{(eq|ne)z?}}
; RV64D-NOT: czero.eqz
; RV64D-NOT: czero.nez

entry:
  %cmp = fcmp ogt double %a, %b
  %sel = select i1 %cmp, double %c, double %d
  ret double %sel
}