; RUN: llvm-as < %s | llvm-dis | FileCheck %s

@g = global i64 0

; CHECK: @ce.ptr.to.byte = global b64 bytecast (ptr @g to b64)
@ce.ptr.to.byte = global b64 bytecast (ptr @g to b64)

define b64 @int_to_byte(i64 %i) {
  ; CHECK: %b = bytecast i64 %i to b64
  %b = bytecast i64 %i to b64
  ret b64 %b
}

define i64 @byte_to_int(b64 %b) {
  ; CHECK: %i = bytecast b64 %b to i64
  %i = bytecast b64 %b to i64
  ret i64 %i
}

define double @byte_to_fp(b64 %b) {
  ; CHECK: %d = bytecast b64 %b to double
  %d = bytecast b64 %b to double
  ret double %d
}

define b64 @fp_to_byte(double %d) {
  ; CHECK: %b = bytecast double %d to b64
  %b = bytecast double %d to b64
  ret b64 %b
}

define ptr @byte_to_ptr(b64 %b) {
  ; CHECK: %p = bytecast b64 %b to ptr
  %p = bytecast b64 %b to ptr
  ret ptr %p
}

define b64 @ptr_to_byte(ptr %p) {
  ; CHECK: %b = bytecast ptr %p to b64
  %b = bytecast ptr %p to b64
  ret b64 %b
}

define ptr addrspace(1) @byte_to_ptr_as1(b64 %b) {
  ; CHECK: %p = bytecast b64 %b to ptr addrspace(1)
  %p = bytecast b64 %b to ptr addrspace(1)
  ret ptr addrspace(1) %p
}

define <4 x i8> @byte_vector_to_int_vector(<4 x b8> %b) {
  ; CHECK: %i = bytecast <4 x b8> %b to <4 x i8>
  %i = bytecast <4 x b8> %b to <4 x i8>
  ret <4 x i8> %i
}

define i64 @byte_vector_to_scalar(<2 x b32> %b) {
  ; CHECK: %i = bytecast <2 x b32> %b to i64
  %i = bytecast <2 x b32> %b to i64
  ret i64 %i
}

define <2 x b32> @scalar_to_byte_vector(i64 %i) {
  ; CHECK: %b = bytecast i64 %i to <2 x b32>
  %b = bytecast i64 %i to <2 x b32>
  ret <2 x b32> %b
}

define b8 @byte_to_byte(b8 %b) {
  ; CHECK: %b2 = bytecast b8 %b to b8
  %b2 = bytecast b8 %b to b8
  ret b8 %b2
}
