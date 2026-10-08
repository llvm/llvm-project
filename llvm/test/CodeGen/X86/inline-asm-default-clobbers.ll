; RUN: llc < %s -mtriple=i686 -stop-after=finalize-isel -verify-machineinstrs | FileCheck %s

; CHECK-LABEL: name: foo
; CHECK: INLINEASM &"", sideeffect attdialect, clobber, implicit-def dead early-clobber $df, clobber, implicit-def dead early-clobber $fpsw, clobber, implicit-def dead early-clobber $eflags
define void @foo() {
entry:
  call void asm sideeffect "", "~{dirflag},~{fpsr},~{flags}"()
  ret void
}

;; The flag output reads EFLAGS, which only the clobber defines.
; CHECK-LABEL: name: flag_output
; CHECK: INLINEASM &"", maystore attdialect, regdef:GR32, def %{{[0-9]+}}, clobber, implicit-def dead early-clobber $df, clobber, implicit-def dead early-clobber $fpsw, clobber, implicit-def early-clobber $eflags
; CHECK-NEXT: SETCCr 4, implicit $eflags
define i8 @flag_output() {
entry:
  %r = call i8 asm "", "={@ccz},~{dirflag},~{fpsr},~{flags}"()
  ret i8 %r
}
