; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; Ensure that these calls do not represent any code and don't cause a crash.
; CHECK: OpFunction
; CHECK-NEXT: OpFunctionParameter
; CHECK-NEXT: OpLabel
; CHECK-NEXT: OpReturn
; CHECK-NEXT: OpFunctionEnd

define spir_kernel void @foo(ptr %p) {
entry:
  call void @llvm.trap()
  call void @llvm.debugtrap()
  call void @llvm.ubsantrap(i8 100)

  %r1 = call ptr @llvm.invariant.start.p0(i64 1024, ptr %p)
  call void @llvm.invariant.end.p0(ptr %r1, i64 1024, ptr %p)

  call void @llvm.instrprof.increment(ptr %p, i64 0, i32 1, i32 0)
  call void @llvm.instrprof.increment.step(ptr %p, i64 0, i32 1, i32 0, i64 1)
  call void @llvm.instrprof.value.profile(ptr %p, i64 0, i64 0, i32 1, i32 0)

  ret void
}
