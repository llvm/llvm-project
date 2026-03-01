; RUN: llc -O0 -mtriple=spirv64-unknown-unknown < %s | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown -filetype=obj < %s  | spirv-val %}

target triple = "spirv64-amd-amdhsa"

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.end.p0(ptr captures(none)) addrspace(4) #0

define spir_kernel void @foo() addrspace(4) {
; CHECK: %4 = OpFunction %2 None %3              ; -- Begin function foo
; CHECK: %5 = OpLabel
; CHECK: OpReturn
  call addrspace(4) void @llvm.lifetime.start.p0(ptr poison)
  call addrspace(4) void @llvm.lifetime.end.p0(ptr poison)
  ret void
}

attributes #0 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
