; RUN: llc < %s -mtriple=i686 -stop-after=finalize-isel | FileCheck %s

; CHECK: INLINEASM &"", 1 /* sideeffect attdialect */, 12 /* clobber */, implicit-def early-clobber $df, 12 /* clobber */, implicit-def early-clobber $fpsw, 12 /* clobber */, implicit-def early-clobber $eflags
define void @foo() {
entry:
  call void asm sideeffect "", "~{dirflag},~{fpsr},~{flags}"()
  ret void
}
