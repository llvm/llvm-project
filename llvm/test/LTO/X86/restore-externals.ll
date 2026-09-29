; Check that "internalizedfn" is re-externalized prior to CodeGen when
; setShouldRestoreGlobalsLinkage is enabled.
;
; RUN: llvm-as < %s > %t1
; RUN: llvm-lto -exported-symbol=preservedfn -restore-linkage -filetype=asm -o - %t1 | FileCheck %s
;
; CHECK: .globl internalizedfn

target triple = "x86_64-unknown-linux-gnu"

declare void @f()

define void @internalizedfn() noinline {
entry:
  call void @f()
  ret void
}

define void @preservedfn() {
entry:
  call void @internalizedfn()
  ret void
}

