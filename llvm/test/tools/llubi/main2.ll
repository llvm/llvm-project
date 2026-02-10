; RUN: llubi --verbose < %s 2>&1 | FileCheck %s

define i32 @main() {
  ret i32 0
}

; CHECK: Entering function: main
; CHECK:   ret i32 0
; CHECK: Exiting function: main
