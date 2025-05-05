; RUN: not llvm-as %s -o /dev/null 2>&1 | FileCheck %s --implicit-check-not=swifterror

%swift_error = type {i64, i8}

; CHECK: swifterror value can only be loaded and stored from, or as a swifterror argument!
; CHECK: ptr %error_ptr_ref
; CHECK: %t = getelementptr inbounds ptr, ptr %error_ptr_ref, i64 1
define float @foo(ptr swifterror %error_ptr_ref) {
  %t = getelementptr inbounds ptr, ptr %error_ptr_ref, i64 1
  ret float 1.0
}

; CHECK: swifterror value can only be loaded and stored from, or as a swifterror argument!
; CHECK: %ptr0 = alloca swifterror ptr, align 8
; CHECK: %t = getelementptr inbounds ptr, ptr %err, i64 1
; CHECK: swifterror value can only be loaded and stored from, or as a swifterror argument!
; CHECK: %ptr1 = alloca swifterror ptr, align 8
; CHECK: %t = getelementptr inbounds ptr, ptr %err, i64 1
define float @phi(i1 %a) {
entry:
  %ptr0 = alloca swifterror ptr, align 8
  %ptr1 = alloca swifterror ptr, align 8
  %ptr2 = alloca ptr, align 8
  br i1 %a, label %body, label %body2

body:
  %err = phi ptr [ %ptr0, %entry ], [ %ptr1, %body ]
  %t = getelementptr inbounds ptr, ptr %err, i64 1
  br label %body

; CHECK: swifterror argument for call has mismatched alloca
; CHECK: %ptr2 = alloca ptr, align 8
; CHECK: %call = call float @foo(ptr swifterror %err_bad)
body2:
  %err_bad = phi ptr [ %ptr0, %entry ], [ %ptr2, %body2 ]
  %call = call float @foo(ptr swifterror %err_bad)
  br label %body2

end:
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
; CHECK: %a = alloca swifterror i128, align 8
define void @swifterror_alloca_invalid_type() {
  %a = alloca swifterror i128
  ret void
}

; CHECK: swifterror alloca must not be array allocation
; CHECK: %a = alloca swifterror ptr, i64 2, align 8
define void @swifterror_alloca_array() {
  %a = alloca swifterror ptr, i64 2
  ret void
}

; CHECK: Cannot have multiple 'swifterror' parameters!
declare void @a(ptr swifterror %a, ptr swifterror %b)

; CHECK: Attribute 'swifterror' applied to incompatible type!
declare void @b(i32 swifterror %a)
