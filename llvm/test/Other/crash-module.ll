; REQUIRES: asserts

; RUN: not --crash opt -passes=crash-module %s 2> %t
; RUN: FileCheck --input-file=%t %s

; CHECK:      Stack dump:
; CHECK-NEXT: 0. Program arguments:
; CHECK-NEXT: 1. Running pass 'CrashingModulePass' on module

define void @foo() {
  ret void
}
