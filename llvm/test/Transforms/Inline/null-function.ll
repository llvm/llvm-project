; RUN: opt -print-before=always-inline -always-inline -enable-new-pm=0 < %s -o /dev/null 2>&1 | FileCheck %s

define i32 @main() #0 {
entry:
  ret i32 0
}

; CHECK: *** IR Dump Before Inliner for always_inline functions ***
; CHECK: Printing <null> Function
