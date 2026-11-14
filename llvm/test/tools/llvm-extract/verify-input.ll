; RUN: not llvm-extract -S --func=e %s -o /dev/null 2>&1 | FileCheck -check-prefix=ERR %s
; RUN: llvm-extract -S -disable-verify --func=e < %s | FileCheck %s

; ERR: The unwind destination does not have an exception handling instruction!
; ERR: error: input module is broken!

; CHECK: declare void @c()
define void @c() {
  ret void
}

; CHECK: define void @e() {
; CHECK-NEXT: invoke void @c()
; CHECK-NEXT: to label %L unwind label %L

; CHECK: L:
; CHECK-NEXT: ret void
define void @e() {
  invoke void @c()
  to label %L unwind label %L
L:
  ret void
}

