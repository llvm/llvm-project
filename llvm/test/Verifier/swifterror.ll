; RUN: not llvm-as %s -o /dev/null 2>&1 | FileCheck %s

%swift_error = type {i64, i8}

; CHECK: swifterror value can only be loaded and stored from, or as a swifterror argument!
; CHECK: %swift_error** %error_ptr_ref
; CHECK: %t = getelementptr inbounds %swift_error*, %swift_error** %error_ptr_ref, i64 1
define float @foo(%swift_error** swifterror %error_ptr_ref) {
  %t = getelementptr inbounds %swift_error*, %swift_error** %error_ptr_ref, i64 1
  ret float 1.0
}

; CHECK: swifterror argument for call has mismatched alloca
; CHECK: %error_ptr_ref = alloca %swift_error*
; CHECK: %call = call float @foo(%swift_error** swifterror %error_ptr_ref)
define float @caller(i8* %error_ref) {
entry:
  %error_ptr_ref = alloca %swift_error*
  store %swift_error* null, %swift_error** %error_ptr_ref
  %call = call float @foo(%swift_error** swifterror %error_ptr_ref)
  ret float 1.0
}

; CHECK: Cannot have multiple 'swifterror' parameters!
declare void @a(i32** swifterror %a, i32** swifterror %b)

; CHECK: Attribute 'swifterror' only applies to parameters with pointer type!
declare void @b(i32 swifterror %a)

; CHECK: Attribute 'swifterror' only applies to parameters with pointer to pointer type!
declare void @c(i32* swifterror %a)
