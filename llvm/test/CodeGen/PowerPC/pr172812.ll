; RUN: not opt %s 2>&1 | FileCheck %s

; CHECK: LLVM ERROR: The only supported target OS's are AIX and ELF-based OS's
target triple = "powerpc-apple-darwin7.2"

define void @_Z4testv() {
entry:
  ret void
}
