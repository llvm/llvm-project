; RUN: not llvm-as -disable-output %s 2>&1 | FileCheck %s

; CHECK: 'nofpclass(nan)' applied to incompatible type!
; CHECK-NEXT: ptr @nofpclass_int_return
define nofpclass(nan) i32 @nofpclass_int_return(i32 %arg) {
  ret i32 %arg
}

; CHECK: 'nofpclass(nan)' applied to incompatible type!
; CHECK-NEXT: ptr @nofpclass_int_param
define i32 @nofpclass_int_param(i32 nofpclass(nan) %arg) {
  ret i32 %arg
}

; CHECK: 'nofpclass(zero)' applied to incompatible type!
; CHECK-NEXT: ptr @nofpclass_int_ret_decl
declare nofpclass(zero) i32 @nofpclass_int_ret_decl()

; CHECK: 'nofpclass(inf)' applied to incompatible type!
; CHECK-NEXT: ptr @nofpclass_int_arg_decl
declare i32 @nofpclass_int_arg_decl(i32 nofpclass(inf))


; CHECK: 'nofpclass(nan)' applied to incompatible type!
; CHECK-NEXT: ptr @nofpclass_vector_int
; CHECK-NEXT: 'nofpclass(zero)' applied to incompatible type!
; CHECK-NEXT: ptr @nofpclass_vector_int
define nofpclass(nan) <4 x i32> @nofpclass_vector_int(<4 x i32> nofpclass(zero) %arg) {
  ret <4 x i32> %arg
}

; CHECK: 'nofpclass(nan)' applied to incompatible type!
; CHECK-NEXT: ptr @nofpclass_array_int
; CHECK-NEXT: 'nofpclass(zero)' applied to incompatible type!
; CHECK-NEXT: ptr @nofpclass_array_int
define nofpclass(nan) [4 x i32] @nofpclass_array_int([4 x i32] nofpclass(zero) %arg) {
  ret [4 x i32] %arg
}

; CHECK: 'nofpclass(nan)' applied to incompatible type!
; CHECK-NEXT: ptr @nofpclass_vector_array_int
; CHECK-NEXT: 'nofpclass(zero)' applied to incompatible type!
; CHECK-NEXT: ptr @nofpclass_vector_array_int
define nofpclass(nan) [4 x <8 x i32>] @nofpclass_vector_array_int([4 x <8 x i32>] nofpclass(zero) %arg) {
  ret [4 x <8 x i32>] %arg
}

%opaque = type opaque

; CHECK: 'nofpclass(nan)' applied to incompatible type!
; CHECK-NEXT: ptr @nofpclass_opaque_type
; CHECK-NEXT: 'nofpclass(zero)' applied to incompatible type!
; CHECK-NEXT: ptr @nofpclass_opaque_type
define nofpclass(nan) %opaque @nofpclass_opaque_type(%opaque nofpclass(zero) %arg) {
  ret %opaque %arg
}

%struct = type { i32, float }

; CHECK: 'nofpclass(nan)' applied to incompatible type!
; CHECK-NEXT: ptr @nofpclass_struct
; CHECK-NEXT: 'nofpclass(zero)' applied to incompatible type!
; CHECK-NEXT: ptr @nofpclass_struct
define nofpclass(nan) %struct @nofpclass_struct(%struct nofpclass(zero) %arg) {
  ret %struct %arg
}
