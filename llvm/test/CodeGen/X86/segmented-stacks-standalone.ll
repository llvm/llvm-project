; RUN: llc < %s -mcpu=generic -mtriple=i686-linux -verify-machineinstrs | FileCheck %s
; RUN: llc < %s -mcpu=generic -mtriple=x86_64-linux  -verify-machineinstrs | FileCheck %s

; This test is standalone because segmented-stacks.ll generates
; object-files with both .note.GNU-split-stack (for the split-stack
; functions) and .note.GNU-no-split-stack sections (for the
; non-split-stack functions). But a split-stack function without a
; stack frame should have a .note.GNU-split-stack section regardless
; of any other contents of the compilation unit.

define void @test_nostack() #0 {
	ret void
}

attributes #0 = { "split-stack" }

; CHECK: .section ".note.GNU-split-stack","",@progbits
; CHECK: .section ".note.GNU-no-split-stack","",@progbits
