; RUN: llc -mtriple=mipsel-linux-gnu -relocation-model=pic -filetype=asm < %s | \
; RUN:     llvm-mc -triple=mipsel-linux-gnu --position-independent -filetype=obj | \
; RUN:     llvm-objdump -d - | FileCheck %s
; RUN: llc -mtriple=mipsel-linux-gnu -relocation-model=pic -filetype=obj < %s | \
; RUN:     llvm-objdump -d - | FileCheck %s

define void @foo() {
  call void asm sideeffect "\09.cprestore 512", "~{$1}"()
  ret void
}

; CHECK: sw $gp, 512($sp)
