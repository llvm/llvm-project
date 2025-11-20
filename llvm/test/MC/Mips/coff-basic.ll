; RUN: llc -mtriple mipsel-windows -filetype=obj < %s | obj2yaml | FileCheck %s

define i32 @foo() {
  ret i32 0
}

; CHECK: Machine:         IMAGE_FILE_MACHINE_R4000
