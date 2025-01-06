; RUN: llc -O0 -mtriple=spirv32-unknown-unknown %s -o - | FileCheck %s --check-prefix=CHECK
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv32-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; Checks SPIR-V blocks are correctly reordered so that dominators shows up
; before others in the binary layout.

define void @main() {
; CHECK: OpLabel
; CHECK: OpBranch %[[#l1:]]

; CHECK: %[[#l1]] = OpLabel
; CHECK:            OpBranch %[[#l2:]]

; CHECK: %[[#l2]] = OpLabel
; CHECK:            OpBranch %[[#end:]]

; CHECK: %[[#end]] = OpLabel
; CHECK:             OpReturn
entry:
  br label %l1

l2:
  br label %end

l1:
  br label %l2

end:
  ret void
}
