; RUN: not llc --mtriple=loongarch64 -emulated-tls -mattr=+d \
; RUN:     -relocation-model=pic < %s 2>&1 | FileCheck %s

; CHECK: LLVM ERROR: the emulated TLS is prohibited

@x = thread_local global i8 7, align 2

define ptr @get_x() nounwind {
entry:
  ret ptr @x
}
