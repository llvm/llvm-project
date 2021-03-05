; RUN: llc < %s -mtriple=powerpc-unknown-aix-xcoff -verify-machineinstrs \
; RUN:     -mcpu=pwr7 -mattr=+altivec 2>&1 | \
; RUN:   FileCheck --check-prefix=DFLTWRN %s

; RUN: llc < %s -mtriple=powerpc64-unknown-aix-xcoff -verify-machineinstrs \
; RUN:     -mcpu=pwr7 -mattr=+altivec 2>&1 | \
; RUN:   FileCheck --check-prefix=DFLTWRN %s
define dso_local void @vec_warn() {
entry:
  call void asm sideeffect "", "~{v23}"()
  ret void
}

; DFLTWRN: warning: vector registers 20 to 32 are reserved in the default AIX AltiVec ABI and cannot be used
