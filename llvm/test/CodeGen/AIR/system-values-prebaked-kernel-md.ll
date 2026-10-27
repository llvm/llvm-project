; AIRSystemValues replaces each kernel that reads a system value with a new
; function that takes the value as a parameter. Metadata naming the old
; function must follow it: a pre-baked !air.kernel entry used to be left
; pointing at the erased function (!{null}), which dropped the kernel and made
; -filetype=obj emit bare bitcode instead of a .metallib.
; RUN: llc -mtriple=air %s -o - | FileCheck %s
; RUN: llc -mtriple=air -filetype=obj %s -o %t.metallib
; RUN: od -An -c -N 4 %t.metallib | FileCheck %s --check-prefix=OBJ

; CHECK: define void @kernel(ptr addrspace(1) {{.*}}%p, <3 x i32> %tid
; CHECK: !air.kernel = !{![[K:[0-9]+]]}
; CHECK: ![[K]] = !{ptr @kernel}
; OBJ: M T L B

declare { i32, i32, i32 } @air.thread_position_in_grid.v3i32()

define void @kernel(ptr addrspace(1) %p) {
entry:
  %t = call { i32, i32, i32 } @air.thread_position_in_grid.v3i32()
  %tx = extractvalue { i32, i32, i32 } %t, 0
  store i32 %tx, ptr addrspace(1) %p
  ret void
}

!air.kernel = !{!0}
!0 = !{ptr @kernel}
