; RUN: not opt -passes=verify < %s 2>&1 | FileCheck %s

; Ensure that a pointer return type is required when capturing through the return value.

; CHECK-NOT: ptr @valid_ret_ty
declare ptr @valid_ret_ty(ptr captures(ret: address, provenance))

; CHECK: Attribute 'captures(ret: ...)' requires a pointer return type
; CHECK-NEXT: ptr @invalid_void
declare void @invalid_void(ptr captures(ret: address, provenance))

; CHECK: Attribute 'captures(ret: ...)' requires a pointer return type
; CHECK-NEXT: ptr @invalid_struct
declare { ptr, i64 } @invalid_struct(ptr captures(ret: address))

; CHECK: Attribute 'captures(ret: ...)' requires a pointer return type
; CHECK-NEXT: ptr @invalid_vector
declare <2 x ptr> @invalid_vector(ptr captures(address_is_null, ret: address, provenance))
