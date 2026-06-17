; Malformed spv.cloned_funcs metadata referencing an argument index past the
; end of the parameter list must be diagnosed, not cause an out-of-bounds write.

; RUN: not --crash llc -O0 -mtriple=spirv64-unknown-unknown %s -o - 2>&1 | FileCheck %s

; CHECK: invalid argument index in function type metadata

define i32 @foo(i32 %x) {
  ret i32 %x
}

; Index 5 is out of range for the single-parameter function.
!spv.cloned_funcs = !{!0}
!0 = !{!"foo", !1}
!1 = !{i32 5, <4 x i32> zeroinitializer}
