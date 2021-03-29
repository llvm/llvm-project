; RUN: not llvm-as %s -o /dev/null 2>&1 | FileCheck %s

; CHECK: Attribute 'byref' type does not match parameter!
; CHECK-NEXT: void (i32*)* @byref_mismatched_pointee_type0
define void @byref_mismatched_pointee_type0(i32* byref(i8)) {
  ret void
}

; CHECK: Attribute 'byref' type does not match parameter!
; CHECK-NEXT: void (i8*)* @byref_mismatched_pointee_type1
define void @byref_mismatched_pointee_type1(i8* byref(i32)) {
  ret void
}

%opaque.ty = type opaque

; CHECK: Attributes 'byval', 'byref', 'inalloca', and 'preallocated' do not support unsized types!
; CHECK-NEXT: void (%opaque.ty*)* @byref_unsized
define void @byref_unsized(%opaque.ty* byref(%opaque.ty)) {
  ret void
}

; CHECK: Attributes 'byval', 'inalloca', 'preallocated', 'inreg', 'nest', 'byref', and 'sret' are incompatible!
; CHECK-NEXT: void (i32*)* @byref_byval
define void @byref_byval(i32* byref(i32) byval(i32)) {
  ret void
}

; CHECK: Attributes 'byval', 'inalloca', 'preallocated', 'inreg', 'nest', 'byref', and 'sret' are incompatible!
; CHECK-NEXT: void (i32*)* @byref_inalloca
define void @byref_inalloca(i32* byref(i32) inalloca(i32)) {
  ret void
}

; CHECK: Attributes 'byval', 'inalloca', 'preallocated', 'inreg', 'nest', 'byref', and 'sret' are incompatible!
; CHECK-NEXT: void (i32*)* @byref_preallocated
define void @byref_preallocated(i32* byref(i32) preallocated(i32)) {
  ret void
}

; CHECK: Attributes 'byval', 'inalloca', 'preallocated', 'inreg', 'nest', 'byref', and 'sret' are incompatible!
; CHECK-NEXT: void (i32*)* @byref_sret
define void @byref_sret(i32* byref(i32) sret(i32)) {
  ret void
}

; CHECK: Attributes 'byval', 'inalloca', 'preallocated', 'inreg', 'nest', 'byref', and 'sret' are incompatible!
; CHECK-NEXT: void (i32*)* @byref_inreg
define void @byref_inreg(i32* byref(i32) inreg) {
  ret void
}

; CHECK: Attributes 'byval', 'inalloca', 'preallocated', 'inreg', 'nest', 'byref', and 'sret' are incompatible!
; CHECK-NEXT: void (i32*)* @byref_nest
define void @byref_nest(i32* byref(i32) nest) {
  ret void
}

; CHECK: Wrong types for attribute: nest noalias nocapture nonnull readnone readonly byref(i32) byval(i32) inalloca(i32) preallocated(i32) sret(i32) align 1 dereferenceable(1) dereferenceable_or_null(1)
; CHECK-NEXT: void (i32)* @byref_non_pointer
define void @byref_non_pointer(i32 byref(i32)) {
  ret void
}

define void @byref_callee([64 x i8]* byref([64 x i8])) {
  ret void
}

define void @no_byref_callee(i8*) {
  ret void
}

; CHECK: cannot guarantee tail call due to mismatched ABI impacting function attributes
; CHECK-NEXT: musttail call void @byref_callee([64 x i8]* byref([64 x i8]) %cast)
; CHECK-NEXT: i8* %ptr
define void @musttail_byref_caller(i8* %ptr) {
  %cast = bitcast i8* %ptr to [64 x i8]*
  musttail call void @byref_callee([64 x i8]* byref([64 x i8]) %cast)
  ret void
}

; CHECK: cannot guarantee tail call due to mismatched ABI impacting function attributes
; CHECK-NEXT: musttail call void @byref_callee([64 x i8]* %ptr)
; CHECK-NEXT: [64 x i8]* %ptr
define void @musttail_byref_callee([64 x i8]* byref([64 x i8]) %ptr) {
  musttail call void @byref_callee([64 x i8]* %ptr)
  ret void
}

define void @byref_callee_align32(i8* byref([64 x i8]) align 32) {
  ret void
}

; CHECK: cannot guarantee tail call due to mismatched ABI impacting function attributes
; CHECK-NEXT: musttail call void @byref_callee_align32(i8* byref([64 x i8]) align 32 %ptr)
; CHECK-NEXT: i8* %ptr
define void @musttail_byref_caller_mismatched_align(i8* byref([64 x i8]) align 16 %ptr) {
  musttail call void @byref_callee_align32(i8* byref([64 x i8]) align 32 %ptr)
  ret void
}
