; RUN: not llvm-as %s -o /dev/null 2>&1 | FileCheck %s

%swift_error = type {i64, i8}

; CHECK: swifterror value can only be loaded and stored from, or as a swifterror argument!
; CHECK: ptr %error_ptr_ref
; CHECK: %t = getelementptr inbounds ptr, ptr %error_ptr_ref, i64 1
define float @foo(ptr swifterror %error_ptr_ref) {
  %t = getelementptr inbounds ptr, ptr %error_ptr_ref, i64 1
  ret float 1.0
}

; CHECK: swifterror argument for call has mismatched alloca
; CHECK: %error_ptr_ref = alloca ptr
; CHECK: %call = call float @foo(ptr swifterror %error_ptr_ref)
define float @caller(ptr %error_ref) {
entry:
  %error_ptr_ref = alloca ptr
  store ptr null, ptr %error_ptr_ref
  %call = call float @foo(ptr swifterror %error_ptr_ref)
  ret float 1.0
}

; CHECK: swifterror alloca must have pointer type
define void @swifterror_alloca_invalid_type() {
  %a = alloca swifterror i128
  ret void
}

; CHECK: swifterror alloca must not be array allocation
define void @swifterror_alloca_array() {
  %a = alloca swifterror ptr, i64 2
  ret void
}

; CHECK: Cannot have multiple 'swifterror' parameters!
declare void @a(ptr swifterror %a, ptr swifterror %b)

; CHECK: Attribute 'swifterror' applied to incompatible type!
declare void @b(i32 swifterror %a)
