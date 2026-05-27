; RUN: llc -O0 -mtriple=spirv32-unknown-unknown %s -o - | FileCheck %s --implicit-check-not=ae_kernel --implicit-check-not=ae_func
; RUN: llc -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s --implicit-check-not=ae_kernel --implicit-check-not=ae_func
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; available_externally definitions must not be emitted: they have no
; MachineFunction, so the module analysis and AsmPrinter would otherwise
; assert.

; CHECK: OpName %[[#used:]] "used_kernel"
; CHECK: %[[#used]] = OpFunction

define available_externally spir_kernel void @ae_kernel(ptr addrspace(1) %out) {
entry:
  store i32 1, ptr addrspace(1) %out, align 4
  ret void
}

define available_externally spir_func i32 @ae_func(i32 %x) {
entry:
  %r = add i32 %x, 1
  ret i32 %r
}

define spir_kernel void @used_kernel(ptr addrspace(1) %out, i32 %n) {
entry:
  store i32 %n, ptr addrspace(1) %out, align 4
  ret void
}
