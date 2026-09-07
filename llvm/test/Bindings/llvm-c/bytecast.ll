; RUN: llvm-as < %s | llvm-dis > %t.orig
; RUN: llvm-as < %s | llvm-c-test --echo > %t.echo
; RUN: diff -w %t.orig %t.echo

define b64 @int_to_byte(i64 %i) {
  %b = bytecast i64 %i to b64
  ret b64 %b
}

define i64 @byte_to_int(b64 %b) {
  %i = bytecast b64 %b to i64
  ret i64 %i
}

define double @byte_to_fp(b64 %b) {
  %d = bytecast b64 %b to double
  ret double %d
}

define b64 @fp_to_byte(double %d) {
  %b = bytecast double %d to b64
  ret b64 %b
}

define ptr @byte_to_ptr(b64 %b) {
  %p = bytecast b64 %b to ptr
  ret ptr %p
}

define b64 @ptr_to_byte(ptr %p) {
  %b = bytecast ptr %p to b64
  ret b64 %b
}

define ptr addrspace(1) @byte_to_ptr_as1(b64 %b) {
  %p = bytecast b64 %b to ptr addrspace(1)
  ret ptr addrspace(1) %p
}

define <4 x i8> @byte_vector_to_int_vector(<4 x b8> %b) {
  %i = bytecast <4 x b8> %b to <4 x i8>
  ret <4 x i8> %i
}

define i64 @byte_vector_to_scalar(<2 x b32> %b) {
  %i = bytecast <2 x b32> %b to i64
  ret i64 %i
}

define <2 x b32> @scalar_to_byte_vector(i64 %i) {
  %b = bytecast i64 %i to <2 x b32>
  ret <2 x b32> %b
}
