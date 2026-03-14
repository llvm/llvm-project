; RUN: not llvm-as %s -o /dev/null 2>&1 | FileCheck %s

%opaque.ty = type opaque

; CHECK: Attribute 'byref' does not support unsized types!
; CHECK-NEXT: ptr @byref_unsized
define void @byref_unsized(ptr byref(%opaque.ty)) {
  ret void
}

; CHECK: Attributes 'byval', 'inalloca', 'preallocated', 'inreg', 'nest', 'byref', and 'sret' are incompatible!
; CHECK-NEXT: ptr @byref_byval
define void @byref_byval(ptr byref(i32) byval(i32)) {
  ret void
}

; CHECK: Attributes 'byval', 'inalloca', 'preallocated', 'inreg', 'nest', 'byref', and 'sret' are incompatible!
; CHECK-NEXT: ptr @byref_inalloca
define void @byref_inalloca(ptr byref(i32) inalloca(i32)) {
  ret void
}

; CHECK: Attributes 'byval', 'inalloca', 'preallocated', 'inreg', 'nest', 'byref', and 'sret' are incompatible!
; CHECK-NEXT: ptr @byref_preallocated
define void @byref_preallocated(ptr byref(i32) preallocated(i32)) {
  ret void
}

; CHECK: Attributes 'byval', 'inalloca', 'preallocated', 'inreg', 'nest', 'byref', and 'sret' are incompatible!
; CHECK-NEXT: ptr @byref_sret
define void @byref_sret(ptr byref(i32) sret(i32)) {
  ret void
}

; CHECK: Attributes 'byval', 'inalloca', 'preallocated', 'inreg', 'nest', 'byref', and 'sret' are incompatible!
; CHECK-NEXT: ptr @byref_inreg
define void @byref_inreg(ptr byref(i32) inreg) {
  ret void
}

; CHECK: Attributes 'byval', 'inalloca', 'preallocated', 'inreg', 'nest', 'byref', and 'sret' are incompatible!
; CHECK-NEXT: ptr @byref_nest
define void @byref_nest(ptr byref(i32) nest) {
  ret void
}

; CHECK: Attribute 'byref(i32)' applied to incompatible type!
; CHECK-NEXT: ptr @byref_non_pointer
define void @byref_non_pointer(i32 byref(i32)) {
  ret void
}

define void @byref_callee(ptr byref([64 x i8])) {
  ret void
}

define void @no_byref_callee(ptr) {
  ret void
}

; CHECK: cannot guarantee tail call due to mismatched ABI impacting function attributes
; CHECK-NEXT: musttail call void @byref_callee(ptr byref([64 x i8]) %ptr)
; CHECK-NEXT: ptr %ptr
define void @musttail_byref_caller(ptr %ptr) {
  musttail call void @byref_callee(ptr byref([64 x i8]) %ptr)
  ret void
}

; CHECK: cannot guarantee tail call due to mismatched ABI impacting function attributes
; CHECK-NEXT: musttail call void @byref_callee(ptr %ptr)
; CHECK-NEXT: ptr %ptr
define void @musttail_byref_callee(ptr byref([64 x i8]) %ptr) {
  musttail call void @byref_callee(ptr %ptr)
  ret void
}

define void @byref_callee_align32(ptr byref([64 x i8]) align 32) {
  ret void
}

; CHECK: cannot guarantee tail call due to mismatched ABI impacting function attributes
; CHECK-NEXT: musttail call void @byref_callee_align32(ptr byref([64 x i8]) align 32 %ptr)
; CHECK-NEXT: ptr %ptr
define void @musttail_byref_caller_mismatched_align(ptr byref([64 x i8]) align 16 %ptr) {
  musttail call void @byref_callee_align32(ptr byref([64 x i8]) align 32 %ptr)
  ret void
}
