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

;; The output def covers the read of $ax, so the overlapping clobber is dead.
; CHECK-LABEL: name: covered_output
; CHECK: INLINEASM &"", sideeffect attdialect, regdef, implicit-def $ax, clobber, implicit-def dead early-clobber $eax
define i16 @covered_output() {
entry:
  %r = call i16 asm sideeffect "", "={ax},~{eax}"()
  ret i16 %r
}

;; The clobber is a subregister of the output def.
; CHECK-LABEL: name: covered_output_super
; CHECK: INLINEASM &"", sideeffect attdialect, regdef, implicit-def $eax, clobber, implicit-def dead early-clobber $ax
define i32 @covered_output_super() {
entry:
  %r = call i32 asm sideeffect "", "={eax},~{ax}"()
  ret i32 %r
}

;; Each clobber is checked against its own covering output def.
; CHECK-LABEL: name: multiple_covered_outputs
; CHECK: INLINEASM &"", sideeffect maystore attdialect, regdef, implicit-def $ax, regdef, implicit-def $dx, regdef:GR32, def %{{[0-9]+}}, clobber, implicit-def dead early-clobber $eax, clobber, implicit-def dead early-clobber $edx, clobber, implicit-def early-clobber $eflags
define { i16, i16, i8 } @multiple_covered_outputs() {
entry:
  %r = call { i16, i16, i8 } asm sideeffect "", "={ax},={dx},={@ccz},~{eax},~{edx},~{flags}"()
  ret { i16, i16, i8 } %r
}

;; The clobber is the same register as the output def, so it is not dead.
; CHECK-LABEL: name: same_output_clobber
; CHECK: INLINEASM &"", sideeffect attdialect, regdef, implicit-def $eax, clobber, implicit-def early-clobber $eax
define i32 @same_output_clobber() {
entry:
  %r = call i32 asm sideeffect "", "={eax},~{eax}"()
  ret i32 %r
}
