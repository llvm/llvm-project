; RUN: not llc -mtriple=x86_64-unknown-windows-msvc -mattr=+egpr -o /dev/null %s 2>&1 | FileCheck %s
; CHECK: LLVM ERROR: EGPR (R16-R31) requires V3 unwind info on Windows x64

; EGPR enabled without V3 unwind (default V1) should produce a fatal error.

define dso_local void @func() {
entry:
  ret void
}
