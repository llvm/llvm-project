; RUN: llvm-as %s -o - -f | llvm-dis | FileCheck %s
; RUN: llvm-as %s -o - -f | verify-uselistorder

declare aarch64_outline_atomics_preservemost i32 @helper_scalar(i32, ptr)
; CHECK: declare aarch64_outline_atomics_preservemost i32 @helper_scalar(i32, ptr)

declare aarch64_outline_atomics_preservemost i128 @helper_cas16(i128, i128, ptr)
; CHECK: declare aarch64_outline_atomics_preservemost i128 @helper_cas16(i128, i128, ptr)

define i32 @call_helper_scalar(i32 %v, ptr %p) {
; CHECK-LABEL: define i32 @call_helper_scalar(
; CHECK: %r = call aarch64_outline_atomics_preservemost i32 @helper_scalar(i32 %v, ptr %p)
  %r = call aarch64_outline_atomics_preservemost i32 @helper_scalar(i32 %v, ptr %p)
  ret i32 %r
}

define i128 @call_helper_cas16(i128 %e, i128 %d, ptr %p) {
; CHECK-LABEL: define i128 @call_helper_cas16(
; CHECK: %r = call aarch64_outline_atomics_preservemost i128 @helper_cas16(i128 %e, i128 %d, ptr %p)
  %r = call aarch64_outline_atomics_preservemost i128 @helper_cas16(i128 %e, i128 %d, ptr %p)
  ret i128 %r
}
