; RUN: llvm-reduce --abort-on-invalid-reduction --delta-passes=instructions --test FileCheck --test-arg --check-prefixes=CHECK,INTERESTING --test-arg %s --test-arg --input-file %s -o %t
; RUN: FileCheck -check-prefixes=CHECK,RESULT %s < %t

; Make sure verifier errors aren't produced from trying to delete
; swifterror instructions.

%swift_error = type { i64, i8 }

declare float @foo(ptr swifterror %error_ptr_ref)

; CHECK-LABEL: define float @caller(
; INTERESTING: call float @foo(

; RESULT: %error_ptr_ref = alloca swifterror ptr, align 8
; RESULT-NEXT: %call = call float @foo(ptr swifterror %error_ptr_ref)
; RESULT-NEXT: ret float
define float @caller(ptr %error_ref) {
entry:
  %error_ptr_ref = alloca swifterror ptr
  store ptr null, ptr %error_ptr_ref
  %call = call float @foo(ptr swifterror %error_ptr_ref)
  %error_from_foo = load ptr, ptr %error_ptr_ref
  %had_error_from_foo = icmp ne ptr %error_from_foo, null
  %v1 = getelementptr inbounds %swift_error, ptr %error_from_foo, i64 0, i32 1
  %t = load i8, ptr %v1
  store i8 %t, ptr %error_ref
  ret float 1.0
}
