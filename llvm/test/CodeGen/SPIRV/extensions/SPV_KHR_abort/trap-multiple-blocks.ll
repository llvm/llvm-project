; RUN: llc -O0 -mtriple=spirv32-unknown-unknown --spirv-ext=+SPV_KHR_abort %s -o - | FileCheck %s
; RUN: llc -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_KHR_abort %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_KHR_abort %s -o - -filetype=obj | spirv-val %}

;; Multiple basic blocks: some trap, some don't.

; CHECK-DAG: OpCapability AbortKHR
; CHECK-DAG: OpExtension "SPV_KHR_abort"

; CHECK:     OpFunction
; CHECK-DAG: OpBranchConditional
; CHECK-DAG: OpBranchConditional
; CHECK-DAG: OpReturnValue
; CHECK-DAG: OpAbortKHR
; CHECK-DAG: OpAbortKHR
; CHECK:     OpFunctionEnd
; CHECK-NOT: OpUnreachable

define spir_func i32 @multi_trap(i32 %x) {
entry:
  %cmp1 = icmp sgt i32 %x, 0
  br i1 %cmp1, label %work, label %err1

work:
  %result = mul i32 %x, 42
  %cmp2 = icmp slt i32 %result, 1000
  br i1 %cmp2, label %ret, label %err2

ret:
  ret i32 %result

err1:
  call void @llvm.trap()
  unreachable

err2:
  call void @llvm.trap()
  unreachable
}

declare void @llvm.trap() #0

attributes #0 = { cold noreturn nounwind }
