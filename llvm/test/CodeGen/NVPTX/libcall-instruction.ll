; RUN: not llc < %s -march=nvptx 2>&1 | FileCheck %s
; used to panic on failed assertion and now fails with an "Undefined external symbol"

; CHECK: LLVM ERROR: Undefined external symbol "__umodti3"
define hidden i128 @remainder(i128, i128) {
  %3 = urem i128 %0, %1
  ret i128 %3
}
