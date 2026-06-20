; RUN: llc -O0 -mtriple=spirv32-unknown-unknown --spirv-ext=+SPV_KHR_abort %s -o - | FileCheck %s
; RUN: llc -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_KHR_abort %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_KHR_abort %s -o - -filetype=obj | spirv-val %}

;; Edge case: instructions after the __spirv_AbortKHR call in the same basic
;; block.
;;
;; In real code (e.g. device libraries' __assert_fail), the pattern is:
;;   call void @__spirv_AbortKHR(i32 %msg)
;;   ; ... possibly lifetime.end intrinsics ...
;;   ret void            ; or unreachable
;;
;; OpAbortKHR is itself a SPIR-V block terminator, so all subsequent
;; instructions in the same BB must be suppressed to produce valid SPIR-V.
;; This test verifies that no instructions appear after OpAbortKHR in any
;; function.

; CHECK-DAG: OpCapability AbortKHR
; CHECK-DAG: OpExtension "SPV_KHR_abort"

;; abort then unreachable.
; CHECK:      OpFunction
; CHECK:      OpAbortKHR
; CHECK-NEXT: OpFunctionEnd

;; abort then ret void.
; CHECK:      OpFunction
; CHECK:      OpAbortKHR
; CHECK-NEXT: OpFunctionEnd

;; abort then lifetime.end + ret void.
; CHECK:      OpFunction
; CHECK:      OpAbortKHR
; CHECK-NEXT: OpFunctionEnd

; CHECK-NOT: OpReturn{{[[:space:]]+}}OpFunctionEnd
; CHECK-NOT: OpUnreachable

declare spir_func void @_Z16__spirv_AbortKHRj(i32) #0
declare void @llvm.lifetime.start.p0(i64 immarg, ptr captures(none)) #1
declare void @llvm.lifetime.end.p0(i64 immarg, ptr captures(none)) #1

; Pattern 1: abort + unreachable.
define spir_func void @abort_then_unreachable(i32 %msg) {
entry:
  call spir_func void @_Z16__spirv_AbortKHRj(i32 %msg)
  unreachable
}

; Pattern 2: abort + ret void.
define spir_func void @abort_then_ret(i32 %msg) {
entry:
  call spir_func void @_Z16__spirv_AbortKHRj(i32 %msg)
  ret void
}

; Pattern 3: abort + lifetime.end + ret void.
define spir_func void @abort_then_lifetime_ret(i32 %msg) {
entry:
  %buf = alloca i8, align 1
  call void @llvm.lifetime.start.p0(i64 1, ptr %buf)
  call spir_func void @_Z16__spirv_AbortKHRj(i32 %msg)
  call void @llvm.lifetime.end.p0(i64 1, ptr %buf)
  ret void
}

attributes #0 = { noreturn }
attributes #1 = { argmemonly nounwind willreturn }
