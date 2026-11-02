; RUN: not llc -mtriple=amdgpu9.0a-amd-amdhsa -amdgpu-lds-contention-multiplier=0 -filetype=null %s 2>&1 | FileCheck --check-prefix=CHECK-ZERO %s
; RUN: not llc -mtriple=amdgpu9.0a-amd-amdhsa -amdgpu-lds-contention-multiplier=-1 -filetype=null %s 2>&1 | FileCheck --check-prefix=CHECK-NEG %s

; Verify that zero value is rejected by the parser
; CHECK-ZERO: '0' value must be greater than 0!

; Verify that negative values are rejected by the command line parser
; CHECK-NEG: '-1' value invalid for uint argument!

define void @test() {
  ret void
}
