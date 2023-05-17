; RUN: llc -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s

; CHECK-DAG: OpCapability Addresses

; CHECK-DAG: OpCapability Linkage
define spir_func void @func_export(i32 addrspace(1)* nocapture %a) {
entry:
; CHECK-DAG: OpCapability Int64
  %call = tail call spir_func i64 @_Z13get_global_idj(i32 0)
  %cmp = icmp eq i64 %call, 0
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  store i32 1, i32 addrspace(1)* %a, align 4
  br label %if.end

if.end:                                           ; preds = %if.then, %entry
  ret void
}

declare spir_func i64 @_Z13get_global_idj(i32)

; CHECK-DAG: OpCapability Kernel
; CHECK-NOT: OpCapability Shader
; CHECK-NOT: OpCapability Float64
define spir_kernel void @func_kernel(i32 addrspace(1)* %a) {
entry:
  tail call spir_func void @func_import(i32 addrspace(1)* %a)
  ret void
}

declare spir_func void @func_import(i32 addrspace(1)*)
