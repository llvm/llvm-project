; RUN: llc -O0 -mtriple=spirv32-unknown-unknown --spirv-ext=+SPV_KHR_abort %s -o - | FileCheck %s
; RUN: llc -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_KHR_abort %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_KHR_abort %s -o - -filetype=obj | spirv-val %}

;; Kernel function with assert-like trap pattern. Models the real-world HIP
;; assert() use case: kernel calls a helper which calls llvm.trap.

; CHECK-DAG: OpCapability AbortKHR
; CHECK-DAG: OpExtension "SPV_KHR_abort"
; CHECK-DAG: OpEntryPoint Kernel %{{[0-9]+}} "test_kernel"

;; __assert_fail_internal: trap → OpAbortKHR.
; CHECK:      OpFunction
; CHECK:      OpAbortKHR
; CHECK-NEXT: OpFunctionEnd

;; test_kernel: conditional branch + return + function call + unreachable.
; CHECK:     OpFunction
; CHECK-DAG: OpBranchConditional
; CHECK-DAG: OpReturn
; CHECK-DAG: OpFunctionCall
; CHECK-DAG: OpUnreachable
; CHECK:     OpFunctionEnd

declare spir_func i64 @_Z13get_global_idj(i32) #2
declare void @llvm.trap() #3

; Models __assert_fail from device libraries.
define spir_func void @__assert_fail_internal() #0 {
entry:
  call void @llvm.trap()
  unreachable
}

; Kernel entry point with conditional assert.
define spir_kernel void @test_kernel(ptr addrspace(1) %in, i32 %N) #1 {
entry:
  %gid = call spir_func i64 @_Z13get_global_idj(i32 0)
  %gid32 = trunc i64 %gid to i32
  %cmp = icmp slt i32 %gid32, %N
  br i1 %cmp, label %ok, label %assert.fail

ok:
  ret void

assert.fail:
  call spir_func void @__assert_fail_internal()
  unreachable
}

attributes #0 = { noinline noreturn nounwind }
attributes #1 = { nounwind }
attributes #2 = { nounwind }
attributes #3 = { cold noreturn nounwind }
