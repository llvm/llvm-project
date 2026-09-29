; RUN: llc < %s -mtriple=i686 -stop-after=finalize-isel | FileCheck %s

; CHECK: INLINEASM &"", sideeffect attdialect, clobber, implicit-def dead early-clobber $df, clobber, implicit-def dead early-clobber $fpsw, clobber, implicit-def dead early-clobber $eflags
define void @foo() {
entry:
  call void asm sideeffect "", "~{dirflag},~{fpsr},~{flags}"()
  ret void
}
