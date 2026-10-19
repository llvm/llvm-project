; RUN: not llc -mtriple=agx -mcpu=agx-g13g < %s 2>&1 | FileCheck %s
; RUN: not llc -mtriple=agx -mcpu=agx-g17p < %s 2>&1 | FileCheck %s --check-prefix=APPLE9

; CHECK: error: {{.*}}unsupported call to 'external' (AGX has no calls)
; APPLE9: error: no AGX instruction encoding for {{.*}} on agx-g17p (the Apple9 encoding is not yet supported)

declare void @external()

define agx_kernel void @calls() {
  call void @external()
  ret void
}
