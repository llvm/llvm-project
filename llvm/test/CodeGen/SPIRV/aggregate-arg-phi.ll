; An aggregate argument feeding a PHI: CodeGenPrepare merges exit.pre into exit and
; adds PHI edges, so it must not see the i32 placeholder the argument becomes.
; RUN: llc -O2 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s
; RUN: llc -O2 -enable-new-pm -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O2 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

%S = type { i32, float }

; CHECK: %[[#A:]] = OpFunctionParameter
; CHECK: %[[#B:]] = OpFunctionParameter
; CHECK: OpPhi %[[#]] %[[#B]] %[[#]] %[[#A]] %[[#]]
define %S @f(%S %a, %S %b, i32 %k) {
entry:
  switch i32 %k, label %other [ i32 0, label %p1
                                i32 1, label %p2 ]
p1:
  br label %exit.pre
p2:
  br label %exit.pre
exit.pre:
  br label %exit
other:
  br label %exit
exit:
  %r = phi %S [ %a, %exit.pre ], [ %b, %other ]
  ret %S %r
}
