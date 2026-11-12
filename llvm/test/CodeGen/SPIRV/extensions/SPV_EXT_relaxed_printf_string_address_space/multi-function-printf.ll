; RUN: llc -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s
; RUN: llc -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_EXT_relaxed_printf_string_address_space %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; The format string is in the constant address space, so no extension is
; required. The pointer type of the format string used to be looked up in
; whichever function was processed last instead of the calling function,
; spuriously requiring the extension when an unrelated function's colliding
; virtual register had a non-UniformConstant pointer type. The loads in @last
; only exist to populate its vreg-to-type map with non-UniformConstant types
; at the vreg numbers that collide with the format string's vreg in @kern.

; CHECK-NOT: OpExtension "SPV_EXT_relaxed_printf_string_address_space"
; CHECK: OpExtInstImport "OpenCL.std"

@.str = private unnamed_addr addrspace(2) constant [6 x i8] c"hello\00", align 1

declare i32 @printf(ptr addrspace(2), ...)

define spir_kernel void @kern() {
entry:
  %r = call i32 (ptr addrspace(2), ...) @printf(ptr addrspace(2) @.str)
  ret void
}

define void @last(ptr addrspace(3) %p) {
entry:
  %g1 = getelementptr inbounds i32, ptr addrspace(3) %p, i64 1
  %v1 = load i32, ptr addrspace(3) %g1, align 4
  %g2 = getelementptr inbounds i32, ptr addrspace(3) %p, i64 2
  %v2 = load i32, ptr addrspace(3) %g2, align 4
  %g3 = getelementptr inbounds i32, ptr addrspace(3) %p, i64 3
  %v3 = load i32, ptr addrspace(3) %g3, align 4
  %g4 = getelementptr inbounds i32, ptr addrspace(3) %p, i64 4
  %v4 = load i32, ptr addrspace(3) %g4, align 4
  %g5 = getelementptr inbounds i32, ptr addrspace(3) %p, i64 5
  %v5 = load i32, ptr addrspace(3) %g5, align 4
  %s1 = add i32 %v1, %v2
  %s2 = add i32 %s1, %v3
  %s3 = add i32 %s2, %v4
  %s4 = add i32 %s3, %v5
  store i32 %s4, ptr addrspace(3) %p, align 4
  ret void
}
