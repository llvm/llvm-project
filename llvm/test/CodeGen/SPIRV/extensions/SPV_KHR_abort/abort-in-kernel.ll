; RUN: llc -O0 -mtriple=spirv32-unknown-unknown --spirv-ext=+SPV_KHR_abort %s -o - | FileCheck %s
; RUN: llc -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_KHR_abort %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_KHR_abort %s -o - -filetype=obj | spirv-val %}

;; Kernel entry point invokes a helper that aborts. Models the device library
;; __assert_fail pattern: the kernel calls __assert_fail_internal which calls
;; __spirv_AbortKHR. The kernel's call site is followed by `unreachable`, which
;; must be preserved (the abort is in the callee, not the caller).

; CHECK-DAG: OpCapability AbortKHR
; CHECK-DAG: OpExtension "SPV_KHR_abort"
; CHECK-DAG: OpEntryPoint Kernel %{{[0-9]+}} "test_kernel"

;; Helper function: abort lowered to OpAbortKHR.
; CHECK:     OpFunction
; CHECK:     OpAbortKHR
; CHECK-NEXT: OpFunctionEnd

;; Kernel: conditional branch, then a function call followed by OpUnreachable
;; in the assert.fail block (the unreachable after the call to the abort
;; helper is preserved because the helper, not the kernel, contains the
;; OpAbortKHR).
; CHECK:     OpFunction
; CHECK-DAG: OpBranchConditional
; CHECK-DAG: OpReturn
; CHECK-DAG: OpFunctionCall
; CHECK-DAG: OpUnreachable
; CHECK:     OpFunctionEnd

declare spir_func void @_Z16__spirv_AbortKHRj(i32) #0
declare spir_func i64 @_Z13get_global_idj(i32) #1

; Models __assert_fail from device libraries.
define spir_func void @__assert_fail_internal(i32 %msg) #2 {
entry:
  call spir_func void @_Z16__spirv_AbortKHRj(i32 %msg)
  unreachable
}

; Kernel entry point with conditional assert.
define spir_kernel void @test_kernel(ptr addrspace(1) %in, i32 %N) {
entry:
  %gid = call spir_func i64 @_Z13get_global_idj(i32 0)
  %gid32 = trunc i64 %gid to i32
  %cmp = icmp slt i32 %gid32, %N
  br i1 %cmp, label %ok, label %assert.fail

ok:
  ret void

assert.fail:
  call spir_func void @__assert_fail_internal(i32 42)
  unreachable
}

attributes #0 = { noreturn }
attributes #1 = { nounwind }
attributes #2 = { noinline noreturn nounwind }
