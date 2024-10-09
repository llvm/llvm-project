; RUN: llc -mtriple mipsel-windows-msvc -filetype=obj < %s | obj2yaml | FileCheck %s
; RUN: llc -mtriple mipsel-windows-gnu -filetype=obj < %s | obj2yaml | FileCheck %s

define i32 @foo() {
  ret i32 0
}

; CHECK: Machine:         IMAGE_FILE_MACHINE_R4000
