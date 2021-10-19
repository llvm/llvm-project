; RUN: llc -mtriple=arm64-apple-ios %s -filetype=obj -o - | llvm-objdump --unwind-info - | FileCheck %s

; Swift asynchronous context is incompatible with the compact unwind encoding
; that currently exists and assumes callee-saved registers are right next to FP
; in a particular order. This isn't a problem now because C++ exceptions aren't
; allowed to unwind through Swift code, but at least make sure the compact info
; says to use DWARF correctly.

; CHECK: compact encoding: 0x03000000
define void @foo(i8* swiftasync %in) "frame-pointer"="all" {
  call void asm sideeffect "", "~{x23}"()
  ret void
}
