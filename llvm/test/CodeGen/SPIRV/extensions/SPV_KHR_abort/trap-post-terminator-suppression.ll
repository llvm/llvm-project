; RUN: llc -O0 -mtriple=spirv32-unknown-unknown --spirv-ext=+SPV_KHR_abort %s -o - | FileCheck %s
; RUN: llc -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_KHR_abort %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_KHR_abort %s -o - -filetype=obj | spirv-val %}

;; Edge case: instructions after llvm.trap in the same basic block must be
;; suppressed because OpAbortKHR is a SPIR-V block terminator.

; CHECK-DAG: OpCapability AbortKHR
; CHECK-DAG: OpExtension "SPV_KHR_abort"

;; trap then unreachable.
; CHECK:      OpFunction
; CHECK:      OpAbortKHR
; CHECK-NEXT: OpFunctionEnd

;; trap then ret void.
; CHECK:      OpFunction
; CHECK:      OpAbortKHR
; CHECK-NEXT: OpFunctionEnd

;; trap then lifetime.end + ret void.
; CHECK:      OpFunction
; CHECK:      OpAbortKHR
; CHECK-NEXT: OpFunctionEnd

; CHECK-NOT: OpReturn{{[[:space:]]+}}OpFunctionEnd
; CHECK-NOT: OpUnreachable

declare void @llvm.trap() #0
declare void @llvm.lifetime.start.p0(i64 immarg, ptr captures(none)) #1
declare void @llvm.lifetime.end.p0(i64 immarg, ptr captures(none)) #1

define spir_func void @trap_then_unreachable() {
entry:
  call void @llvm.trap()
  unreachable
}

define spir_func void @trap_then_ret() {
entry:
  call void @llvm.trap()
  ret void
}

define spir_func void @trap_then_lifetime_ret() {
entry:
  %buf = alloca i8, align 1
  call void @llvm.lifetime.start.p0(i64 1, ptr %buf)
  call void @llvm.trap()
  call void @llvm.lifetime.end.p0(i64 1, ptr %buf)
  ret void
}

attributes #0 = { cold noreturn nounwind }
attributes #1 = { argmemonly nounwind willreturn }
