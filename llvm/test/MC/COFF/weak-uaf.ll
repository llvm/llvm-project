; RUN: llc --compile-twice -mtriple=x86_64-pc-win32 -filetype=obj < %s

; UAF when re-using the MCObjectWriter. does not leak into the output,
; but should be detectable with --compile-twice under ASAN or so.

define weak void @foo() nounwind {
  ret void
}

define weak void @bar() nounwind {
  ret void
}
