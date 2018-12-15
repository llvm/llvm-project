; RUN: not llc < %s -march=nvptx 2>&1 | FileCheck %s
; used to panic on failed assetion and now fails with a "Cannot select"

; CHECK: LLVM ERROR: Cannot select: {{t28|0x[0-9a-f]+}}: i32 = ExternalSymbol'__umodti3'
define hidden i128 @remainder(i128, i128) {
  %3 = urem i128 %0, %1
  ret i128 %3
}
