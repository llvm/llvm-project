; RUN: not llc -o /dev/null %s 2>&1 | FileCheck %s
target triple = "x86_64--"

; CHECK: error: couldn't allocate output register for constraint '{ax}'
define i128 @blup() {
  %v = tail call i128 asm "", "={ax},0"(i128 0)
  ret i128 %v
}
