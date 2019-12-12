; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s
;
; Test the packed stack layout.

; Test spill/restore of an FPR and a GPR.
define void @f1() #0 {
; CHECK-LABEL: f1:
; CHECK: stmg %r12, %r15, 128(%r15)
; CHECK-NEXT: .cfi_offset %r12, -32
; CHECK-NEXT: .cfi_offset %r15, -8
; CHECK-NEXT: std %f8, 120(%r15)          # 8-byte Folded Spill
; CHECK-NEXT: .cfi_offset %f8, -40
; CHECK-NEXT: #APP
; CHECK-NEXT: #NO_APP
; CHECK-NEXT: ld %f8, 120(%r15)          # 8-byte Folded Reload
; CHECK-NEXT: lmg %r12, %r15, 128(%r15)
; CHECK-NEXT: br %r14
  call void asm sideeffect "", "~{f8},~{r12}"() nounwind
  ret void
}

; Test spill/restore with anyregcc, including an FP argument register.
define anyregcc void @f2() #0 {
; CHECK-LABEL: f2:
; CHECK: stmg %r3, %r15, 56(%r15)
; CHECK-NEXT: .cfi_offset %r3, -104
; CHECK-NEXT: .cfi_offset %r15, -8
; CHECK-NEXT: std %f0, 48(%r15)          # 8-byte Folded Spill
; CHECK-NEXT: std %f1, 40(%r15)          # 8-byte Folded Spill
; CHECK-NEXT: .cfi_offset %f0, -112
; CHECK-NEXT: .cfi_offset %f1, -120
; CHECK-NEXT: #APP
; CHECK-NEXT: #NO_APP
; CHECK-NEXT: ld %f0, 48(%r15)          # 8-byte Folded Reload
; CHECK-NEXT: ld %f1, 40(%r15)          # 8-byte Folded Reload
; CHECK-NEXT: lmg %r3, %r15, 56(%r15)
; CHECK-NEXT: br %r14
  call void asm sideeffect "", "~{f0},~{f1},~{r3}"() nounwind
  ret void
}

; Test spill/restore in local area with incoming stack arguments.
define i64 @f3(i64 %a, i64 %b, i64 %c, i64 %d, i64 %e, i64 %f,
               double %A, double %B, double %C, double %D, double %E) #0 {
; CHECK-LABEL: f3:
; CHECK: std %f8, 152(%r15)          # 8-byte Folded Spill
; CHECK-NEXT: .cfi_offset %f8, -8
; CHECK-NEXT: ld %f0, 168(%r15)
; CHECK-NEXT: cgdbr %r2, 5, %f0
; CHECK-NEXT: ag %r2, 160(%r15)
; CHECK-NEXT: #APP
; CHECK-NEXT: #NO_APP
; CHECK-NEXT: ld %f8, 152(%r15)          # 8-byte Folded Reload
; CHECK-NEXT: br %r14
  call void asm sideeffect "", "~{f8}"() nounwind
  %Ei = fptosi double %E to i64
  %S = add i64 %f, %Ei
  ret i64 %S
}

; Test spill/restore in local area with outgoing stack arguments.
define i64 @f4() #0 {
; CHECK-LABEL: f4:
; CHECK: stmg    %r6, %r15, 80(%r15)
; CHECK-NEXT: .cfi_offset %r6, -80
; CHECK-NEXT: .cfi_offset %r14, -16
; CHECK-NEXT: .cfi_offset %r15, -8
; CHECK-NEXT: aghi    %r15, -104
; CHECK-NEXT: .cfi_def_cfa_offset 264
; CHECK-NEXT: std     %f8, 176(%r15)          # 8-byte Folded Spill
; CHECK-NEXT: .cfi_offset %f8, -88
; CHECK-NEXT: #APP
; CHECK-NEXT: #NO_APP
; CHECK-NEXT: llihh   %r0, 16404
; CHECK-NEXT: stg     %r0, 168(%r15)
; CHECK: mvghi   160(%r15), 6
; CHECK-NEXT: brasl   %r14, f3@PLT
; CHECK-NEXT: ld      %f8, 176(%r15)          # 8-byte Folded Reload
; CHECK-NEXT: lmg     %r6, %r15, 184(%r15)
; CHECK-NEXT: br      %r14
  call void asm sideeffect "", "~{f8}"() nounwind
  %C = call i64 @f3 (i64 1, i64 2, i64 3, i64 4, i64 5, i64 6,
                     double 1.0, double 2.0, double 3.0, double 4.0, double 5.0)
  ret i64 %C
}

attributes #0 = { "packed-stack"="true" }
