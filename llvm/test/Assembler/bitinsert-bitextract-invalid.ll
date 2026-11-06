; RUN: split-file %s %t
; RUN: not llvm-as -disable-output %t/bitextract-invalid-1.ll 2>&1 | FileCheck %s --check-prefix=CHECK-BITEXTRACT-1
; RUN: not llvm-as -disable-output %t/bitinsert-invalid-1.ll 2>&1 | FileCheck %s --check-prefix=CHECK-BITINSERT-1
; RUN: not llvm-as -disable-output %t/bitextract-invalid-aggregate.ll 2>&1 | FileCheck %s --check-prefix=CHECK-BITEXTRACT-AGGREGATE
; RUN: not llvm-as -disable-output %t/bitinsert-invalid-aggregate.ll 2>&1 | FileCheck %s --check-prefix=CHECK-BITINSERT-AGGREGATE
; RUN: not llvm-as -disable-output %t/bitextract-invalid-vec-src.ll 2>&1 | FileCheck %s --check-prefix=CHECK-BITEXTRACT-VEC-SRC
; RUN: not llvm-as -disable-output %t/bitinsert-invalid-vec-base.ll 2>&1 | FileCheck %s --check-prefix=CHECK-BITINSERT-VEC-BASE
; RUN: not llvm-as -disable-output %t/bitextract-invalid-vec-offset-ty.ll 2>&1 | FileCheck %s --check-prefix=CHECK-BITEXTRACT-VEC-OFFSET-TY
; RUN: not llvm-as -disable-output %t/bitinsert-invalid-vec-offset-ty.ll 2>&1 | FileCheck %s --check-prefix=CHECK-BITINSERT-VEC-OFFSET-TY
; RUN: not llvm-as -disable-output %t/bitextract-invalid-scalar-src.ll 2>&1 | FileCheck %s --check-prefix=CHECK-BITEXTRACT-SHAPE
; RUN: not llvm-as -disable-output %t/bitextract-invalid-scalar-offset.ll 2>&1 | FileCheck %s --check-prefix=CHECK-BITEXTRACT-SHAPE
; RUN: not llvm-as -disable-output %t/bitextract-invalid-num-elts.ll 2>&1 | FileCheck %s --check-prefix=CHECK-BITEXTRACT-SHAPE
; RUN: not llvm-as -disable-output %t/bitextract-invalid-offset-num-elts.ll 2>&1 | FileCheck %s --check-prefix=CHECK-BITEXTRACT-SHAPE
; RUN: not llvm-as -disable-output %t/bitextract-invalid-scalable.ll 2>&1 | FileCheck %s --check-prefix=CHECK-BITEXTRACT-SHAPE
; RUN: not llvm-as -disable-output %t/bitinsert-invalid-scalar-val.ll 2>&1 | FileCheck %s --check-prefix=CHECK-BITINSERT-SHAPE
; RUN: not llvm-as -disable-output %t/bitinsert-invalid-scalar-offset.ll 2>&1 | FileCheck %s --check-prefix=CHECK-BITINSERT-SHAPE
; RUN: not llvm-as -disable-output %t/bitinsert-invalid-num-elts.ll 2>&1 | FileCheck %s --check-prefix=CHECK-BITINSERT-SHAPE
; RUN: not llvm-as -disable-output %t/bitinsert-invalid-offset-num-elts.ll 2>&1 | FileCheck %s --check-prefix=CHECK-BITINSERT-SHAPE
; RUN: not llvm-as -disable-output %t/bitinsert-invalid-scalable.ll 2>&1 | FileCheck %s --check-prefix=CHECK-BITINSERT-SHAPE

; CHECK-BITEXTRACT-1: error: bitextract source must be a byte type or a vector of byte type
; CHECK-BITINSERT-1: error: bitinsert base must be a byte type or a vector of byte type
; CHECK-BITEXTRACT-AGGREGATE: error: bitextract result must be an integer, floating-point, pointer, or byte type, or a vector of those types
; CHECK-BITINSERT-AGGREGATE: error: bitinsert value must be an integer, floating-point, pointer, or byte type, or a vector of those types
; CHECK-BITEXTRACT-VEC-SRC: error: bitextract source must be a byte type or a vector of byte type
; CHECK-BITINSERT-VEC-BASE: error: bitinsert base must be a byte type or a vector of byte type
; CHECK-BITEXTRACT-VEC-OFFSET-TY: error: bitextract offset must be i32 or a vector of i32
; CHECK-BITINSERT-VEC-OFFSET-TY: error: bitinsert offset must be i32 or a vector of i32
; CHECK-BITEXTRACT-SHAPE: error: bitextract operands must all be scalars or all be vectors with the same number of elements
; CHECK-BITINSERT-SHAPE: error: bitinsert operands must all be scalars or all be vectors with the same number of elements

;--- bitextract-invalid-1.ll
; CHECK: bitextract source must be a byte type
define i8 @invalid(i32 %src) {
  %r = bitextract i8, i32 %src, i32 0
  ret i8 %r
}

;--- bitinsert-invalid-1.ll
; CHECK: bitinsert base must be a byte type
define i32 @invalid(i32 %base, i8 %val) {
  %r = bitinsert i32 %base, i8 %val, i32 0
  ret i32 %r
}

;--- bitextract-invalid-aggregate.ll
; CHECK: bitextract result must be an integer, floating-point, pointer, or byte type
define void @test_extract_struct(b32 %src) {
  %res = bitextract { i8, i8 }, b32 %src, i32 0
  ret void
}

;--- bitinsert-invalid-aggregate.ll
; CHECK: bitinsert value must be an integer, floating-point, pointer, or byte type
define void @test_insert_array(b32 %base, [2 x i8] %val) {
  %res = bitinsert b32 %base, [2 x i8] %val, i32 0
  ret void
}

;--- bitextract-invalid-vec-src.ll
define <2 x i8> @invalid(<2 x i32> %src) {
  %r = bitextract <2 x i8>, <2 x i32> %src, <2 x i32> zeroinitializer
  ret <2 x i8> %r
}

;--- bitinsert-invalid-vec-base.ll
define <2 x i32> @invalid(<2 x i32> %base, <2 x i8> %val) {
  %r = bitinsert <2 x i32> %base, <2 x i8> %val, <2 x i32> zeroinitializer
  ret <2 x i32> %r
}

;--- bitextract-invalid-vec-offset-ty.ll
define <2 x i8> @invalid(<2 x b32> %src) {
  %r = bitextract <2 x i8>, <2 x b32> %src, <2 x i64> zeroinitializer
  ret <2 x i8> %r
}

;--- bitinsert-invalid-vec-offset-ty.ll
define <2 x b32> @invalid(<2 x b32> %base, <2 x i8> %val) {
  %r = bitinsert <2 x b32> %base, <2 x i8> %val, <2 x i64> zeroinitializer
  ret <2 x b32> %r
}

;--- bitextract-invalid-scalar-src.ll
define <2 x i8> @invalid(b32 %src) {
  %r = bitextract <2 x i8>, b32 %src, <2 x i32> zeroinitializer
  ret <2 x i8> %r
}

;--- bitextract-invalid-scalar-offset.ll
define <2 x i8> @invalid(<2 x b32> %src) {
  %r = bitextract <2 x i8>, <2 x b32> %src, i32 0
  ret <2 x i8> %r
}

;--- bitextract-invalid-num-elts.ll
define <2 x i8> @invalid(<4 x b32> %src) {
  %r = bitextract <2 x i8>, <4 x b32> %src, <2 x i32> zeroinitializer
  ret <2 x i8> %r
}

;--- bitextract-invalid-offset-num-elts.ll
define <2 x i8> @invalid(<2 x b32> %src) {
  %r = bitextract <2 x i8>, <2 x b32> %src, <4 x i32> zeroinitializer
  ret <2 x i8> %r
}

;--- bitextract-invalid-scalable.ll
define <vscale x 2 x i8> @invalid(<2 x b32> %src) {
  %r = bitextract <vscale x 2 x i8>, <2 x b32> %src, <vscale x 2 x i32> zeroinitializer
  ret <vscale x 2 x i8> %r
}

;--- bitinsert-invalid-scalar-val.ll
define <2 x b32> @invalid(<2 x b32> %base, i8 %val) {
  %r = bitinsert <2 x b32> %base, i8 %val, <2 x i32> zeroinitializer
  ret <2 x b32> %r
}

;--- bitinsert-invalid-scalar-offset.ll
define <2 x b32> @invalid(<2 x b32> %base, <2 x i8> %val) {
  %r = bitinsert <2 x b32> %base, <2 x i8> %val, i32 0
  ret <2 x b32> %r
}

;--- bitinsert-invalid-num-elts.ll
define <2 x b32> @invalid(<2 x b32> %base, <4 x i8> %val) {
  %r = bitinsert <2 x b32> %base, <4 x i8> %val, <2 x i32> zeroinitializer
  ret <2 x b32> %r
}

;--- bitinsert-invalid-offset-num-elts.ll
define <2 x b32> @invalid(<2 x b32> %base, <2 x i8> %val) {
  %r = bitinsert <2 x b32> %base, <2 x i8> %val, <4 x i32> zeroinitializer
  ret <2 x b32> %r
}

;--- bitinsert-invalid-scalable.ll
define <2 x b32> @invalid(<2 x b32> %base, <vscale x 2 x i8> %val) {
  %r = bitinsert <2 x b32> %base, <vscale x 2 x i8> %val, <2 x i32> zeroinitializer
  ret <2 x b32> %r
}
