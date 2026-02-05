; RUN: llubi --verbose < %s 2>&1 | FileCheck %s

define i32 @main(i32 %argc, ptr %argv) {
  ret i32 0
}

; CHECK: Entering function: main
; CHECK:   i32 %argc = i32 1
; CHECK:   ptr %argv = ptr 16
; CHECK:   ret i32 0
; CHECK: Exiting function: main
