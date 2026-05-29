; AIR system-value intrinsic calls are converted to extra kernel parameters
; (<3 x i32> for grid positions, i32 for simdlane). The pass also emits
; !air.kernel / !air.version / !air.language_version metadata.
; RUN: llc -mtriple=air %s -o - | FileCheck %s

; CHECK-LABEL: define void @kernel
; CHECK-SAME: <3 x i32> %tid
; CHECK: extractelement <3 x i32> %tid

; CHECK: !air.kernel = !{
; CHECK: !air.version = !{
; CHECK: !air.language_version = !{

declare { i32, i32, i32 } @air.thread_position_in_grid.v3i32()

define void @kernel(ptr addrspace(1) %p) {
entry:
  %t = call { i32, i32, i32 } @air.thread_position_in_grid.v3i32()
  %tx = extractvalue { i32, i32, i32 } %t, 0
  store i32 %tx, ptr addrspace(1) %p
  ret void
}
