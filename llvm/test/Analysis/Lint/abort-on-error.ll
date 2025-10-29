; RUN: not opt -passes='lint<abort-on-error>' -disable-output %s 2>&1 | FileCheck %s

; CHECK: Undefined behavior: Division by zero
; CHECK-NEXT:   %b = sdiv i32 %a, 0
; CHECK-NEXT: LLVM ERROR: linter found errors, aborting. (enabled by abort-on-error)

define i32 @sdiv_by_zero(i32 %a) {
  %b = sdiv i32 %a, 0
  ret i32 %b
}
