; REQUIRES: asserts
; RUN: llc -mtriple=amdgpu9.00-amd-amdhsa < %s | FileCheck %s
; RUN: llc -O0 -mtriple=amdgpu9.00-amd-amdhsa < %s | FileCheck %s

; CHECK-NOT: func

define internal i32 @func() {
  ret i32 0
}
