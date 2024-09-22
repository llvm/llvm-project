; REQUIRES: asserts
; RUN: not --crash llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx900 -filetype=null %s 2>&1 | FileCheck %s
; RUN: not --crash llc -O0 -mtriple=amdgcn-amd-amdhsa -mcpu=gfx900 -filetype=null %s 2>&1 | FileCheck %s

; CHECK: function must have been generated already

define internal i32 @func() {
  ret i32 0
}
