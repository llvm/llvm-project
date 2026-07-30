; REQUIRES: default_triple

; Often LLVM tools use "-" to indicate that output should be written to stdout
; instead of a file. This behaviour is implemented by the raw_fd_ostream class.
; This test verifies that when doing so multiple times we don't try to access a
; closed STDOUT_FILENO. The exact options used in this test are unimportant, as
; long as they write to stdout using raw_fd_ostream.
; RUN: llc %s -o=- -pass-remarks-output=- -filetype=asm | FileCheck %s
; foobar should appear as a function somewhere in the assembly file.
; CHECK: foobar
; !Analysis appears at the start of pass-remarks-output.
; CHECK: !Analysis

define void @foobar() {
  ret void
}
