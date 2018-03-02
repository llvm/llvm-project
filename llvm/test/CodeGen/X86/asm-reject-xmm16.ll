; RUN: not llc -o /dev/null %s 2>&1 | FileCheck %s
target triple = "x86_64--"

; CHECK: error: couldn't allocate output register for constraint '{xmm16}'
define i64 @blup() {
  %v = tail call i64 asm "", "={xmm16},0"(i64 0)
  ret i64 %v
}
