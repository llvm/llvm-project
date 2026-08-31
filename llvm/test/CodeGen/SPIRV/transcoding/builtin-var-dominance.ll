; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv32-unknown-unknown %s -o - | FileCheck %s
; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv32-unknown-unknown %s -o - -filetype=obj | spirv-val %}
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

;; Verify correct translation when the same builtin is called from multiple
;; non-entry blocks. The fix is validated by -verify-machineinstrs that fails
;; if the OpVariable's VReg definition does not dominate all its uses in MIR.

; CHECK-DAG: OpDecorate %[[#VarID:]] BuiltIn LocalInvocationId
; CHECK-DAG: %[[#VarID]] = OpVariable %[[#]] Input

define spir_kernel void @test(ptr addrspace(1) %out, i32 %n) {
entry:
  %cmp = icmp sgt i32 %n, 0
  br i1 %cmp, label %then, label %else

then:
  %id0 = call spir_func i32 @_Z12get_local_idj(i32 0)
  store i32 %id0, ptr addrspace(1) %out, align 4
  br label %exit

else:
  %id1 = call spir_func i32 @_Z12get_local_idj(i32 0)
  %neg = sub i32 0, %id1
  store i32 %neg, ptr addrspace(1) %out, align 4
  br label %exit

exit:
  ret void
}

declare spir_func i32 @_Z12get_local_idj(i32)
