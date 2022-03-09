; REQUIRES: asserts

; RUN: not --crash opt -passes=crash-cgscc %s 2> %t
; RUN: FileCheck --input-file=%t %s

; CHECK:      Stack dump:
; CHECK-NEXT: 0.    Program arguments:
; CHECK-NEXT: 1.    Running pass 'ModuleToPostOrderCGSCCPassAdaptor' on module
; CHECK-NEXT: 2.    Running pass 'CrashingCGSCCPass' on CGSCC '(foo)'.

define void @foo() {
  ret void
}
