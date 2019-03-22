; RUN: llc < %s -mtriple=i686 -stop-after=expand-isel-pseudos | FileCheck %s

; CHECK: INLINEASM &"", 1, 12, implicit-def early-clobber $df, 12, implicit-def early-clobber $fpsw, 12, implicit-def early-clobber $eflags
define void @foo() {
entry:
  call void asm sideeffect "", "~{dirflag},~{fpsr},~{flags}"()
  ret void
}
