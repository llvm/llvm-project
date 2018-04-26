; RUN: not llc < %s 2>&1 | FileCheck %s
; REQUIRES: default_triple

define void @test() {
  call void asm sideeffect ".macro FOO\0A.endm", "~{dirflag},~{fpsr},~{flags}"() #1
  call void asm sideeffect ".macro FOO\0A.endm", "~{dirflag},~{fpsr},~{flags}"() #1
; CHECK: error: macro 'FOO' is already defined
  ret void
}
