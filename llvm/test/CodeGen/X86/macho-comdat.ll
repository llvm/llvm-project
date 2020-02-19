; RUN: not --crash llc -mtriple x86_64-apple-darwin < %s 2> %t
; RUN: FileCheck < %t %s

$f = comdat any
@v = global i32 0, comdat($f)
; CHECK: LLVM ERROR: MachO doesn't support COMDATs, 'f' cannot be lowered.
