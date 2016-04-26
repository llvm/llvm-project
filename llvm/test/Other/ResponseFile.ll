; Test that we can recurse, at least a little bit.  The -time-passes flag here
; is a hack to make sure that neither echo nor the shell expands the response
; file for us.  Tokenization with quotes is tested in unittests.
; On Windows, paths contain \ characters, which are escape characters in
; GNU-style response files.  So replace \ with \\ to make the tests work there.
; RUN: echo %s | sed -e 's:\\:\\\\:g' > %t.list1
; RUN: echo "-time-passes @%t.list1" | sed -e 's:\\:\\\\:g' > %t.list2
; RUN: llvm-as @%t.list2 -o %t.bc
; RUN: llvm-nm %t.bc 2>&1 | FileCheck %s

; When the response file begins with UTF8 BOM sequence, we shall remove them.
; Neither command below should return a "Could not open input file" error.
; RUN: llvm-as @%S/Inputs/utf8-response > /dev/null
; RUN: llvm-as @%S/Inputs/utf8-bom-response > /dev/null

; CHECK: T foobar

define void @foobar() {
  ret void
}
