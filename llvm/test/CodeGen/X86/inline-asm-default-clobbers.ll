; RUN: llc < %s -mtriple=i686 -stop-after=finalize-isel | FileCheck %s

; CHECK: INLINEASM &"", sideeffect attdialect, clobber, implicit-def $df, clobber, implicit-def $fpsw, clobber, implicit-def $eflags
define void @foo() {
entry:
  call void asm sideeffect "", "~{dirflag},~{fpsr},~{flags}"()
  ret void
}
