; RUN: llc -mtriple=x86_64 -relocation-model=pic < %s | FileCheck %s

;; Test that we use the local alias for dso_local globals in inline assembly.

@mxcsr0 = dso_local global i32 0
@mxcsr1 = dso_preemptable global i32 1

define <2 x double> @foo(<2 x double> %a, <2 x double> %b) {
; CHECK-LABEL: foo:
; CHECK:        movq mxcsr1@GOTPCREL(%rip), %rax
; CHECK:        #APP
; CHECK-NEXT:   ldmxcsr .Lmxcsr0$local(%rip)
; CHECK-NEXT:   addpd %xmm1, %xmm0
; CHECK-NEXT:   ldmxcsr (%rax)
; CHECK-NEXT:   #NO_APP
entry:
  %0 = call <2 x double> asm sideeffect "ldmxcsr $2; addpd $1, $0; ldmxcsr $3",
         "=x,x,*m,*m,0,~{dirflag},~{fpsr},~{flags}"(
           <2 x double> %b, i32* nonnull @mxcsr0, i32* nonnull @mxcsr1, <2 x double> %a)
  ret <2 x double> %0
}
