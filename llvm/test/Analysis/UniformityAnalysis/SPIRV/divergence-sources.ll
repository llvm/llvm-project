; RUN: opt -mtriple=spirv64-unknown-unknown -passes='print<uniformity>' -disable-output %s 2>&1 | FileCheck %s

@cross = external addrspace(1) global i32
@input = external addrspace(7) global i32
@output = external addrspace(8) global i32
@private = external addrspace(10) global i32
@__spirv_BuiltInLocalInvocationId = external addrspace(1) global <3 x i64>
@decorated = external addrspace(1) global i32, !spirv.Decorations !0

; CHECK-LABEL: for function 'kernel_args'
; CHECK-NOT: DIVERGENT
define spir_kernel void @kernel_args(i32 %a) {
  ret void
}

; CHECK-LABEL: for function 'func_args'
; CHECK: DIVERGENT: i32 %a
define spir_func void @func_args(i32 %a) {
  ret void
}

; CHECK-LABEL: for function 'sources'
; CHECK: DIVERGENT: %lid = call
; CHECK: DIVERGENT: %tid = call
; CHECK: DIVERGENT: %rmw = atomicrmw
; CHECK: DIVERGENT: %cx = cmpxchg
; CHECK: DIVERGENT: %ld.func = load
; CHECK: DIVERGENT: %ld.gen = load
; CHECK: DIVERGENT: %ld.priv = load
; CHECK: DIVERGENT: %ld.in = load
; CHECK: DIVERGENT: %ld.out = load
; CHECK: DIVERGENT: %ld.builtin = load
; CHECK: DIVERGENT: %ld.builtin.y = load
; CHECK: DIVERGENT: %ld.decorated = load
; CHECK: DIVERGENT: %ld.atomic = load atomic
; CHECK-NOT: DIVERGENT: %ld.cross
; CHECK-NOT: DIVERGENT: %ld.arg
define spir_kernel void @sources(ptr addrspace(1) %p, ptr addrspace(4) %g) {
  %lid = call spir_func i64 @_Z12get_local_idj(i32 0)
  %tid = call i32 @llvm.spv.thread.id.i32(i32 0)
  %rmw = atomicrmw add ptr addrspace(1) %p, i32 1 seq_cst
  %cx = cmpxchg ptr addrspace(1) %p, i32 0, i32 1 seq_cst seq_cst
  %alloca = alloca i32
  %ld.func = load i32, ptr %alloca
  %ld.gen = load i32, ptr addrspace(4) %g
  %ld.priv = load i32, ptr addrspace(10) @private
  %ld.in = load i32, ptr addrspace(7) @input
  %ld.out = load i32, ptr addrspace(8) @output
  %ld.builtin = load <3 x i64>, ptr addrspace(1) @__spirv_BuiltInLocalInvocationId
  %ld.builtin.y = load i64, ptr addrspace(1) getelementptr inbounds (i8, ptr addrspace(1) @__spirv_BuiltInLocalInvocationId, i64 8)
  %ld.decorated = load i32, ptr addrspace(1) @decorated
  %ld.atomic = load atomic i32, ptr addrspace(1) %p monotonic, align 4
  %ld.cross = load i32, ptr addrspace(1) @cross
  %ld.arg = load i32, ptr addrspace(1) %p
  ret void
}

declare spir_func i64 @_Z12get_local_idj(i32)

!0 = !{!1}
!1 = !{i32 11, i32 28}
