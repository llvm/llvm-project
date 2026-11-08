; RUN: split-file %s %t
; RUN: not llvm-as -disable-output %t/bitextract-invalid-1.ll 2>&1 | FileCheck %t/bitextract-invalid-1.ll
; RUN: not llvm-as -disable-output %t/bitinsert-invalid-1.ll 2>&1 | FileCheck %t/bitinsert-invalid-1.ll
; RUN: not llvm-as -disable-output %t/bitextract-invalid-aggregate.ll 2>&1 | FileCheck %t/bitextract-invalid-aggregate.ll
; RUN: not llvm-as -disable-output %t/bitinsert-invalid-aggregate.ll 2>&1 | FileCheck %t/bitinsert-invalid-aggregate.ll
; RUN: not llvm-as -disable-output %t/bitextract-invalid-vec-src.ll 2>&1 | FileCheck %t/bitextract-invalid-vec-src.ll
; RUN: not llvm-as -disable-output %t/bitinsert-invalid-vec-base.ll 2>&1 | FileCheck %t/bitinsert-invalid-vec-base.ll
; RUN: not llvm-as -disable-output %t/bitextract-invalid-vec-offset-ty.ll 2>&1 | FileCheck %t/bitextract-invalid-vec-offset-ty.ll
; RUN: not llvm-as -disable-output %t/bitinsert-invalid-vec-offset-ty.ll 2>&1 | FileCheck %t/bitinsert-invalid-vec-offset-ty.ll
; RUN: not llvm-as -disable-output %t/bitextract-invalid-scalar-src.ll 2>&1 | FileCheck %t/bitextract-invalid-scalar-src.ll
; RUN: not llvm-as -disable-output %t/bitextract-invalid-scalar-offset.ll 2>&1 | FileCheck %t/bitextract-invalid-scalar-offset.ll
; RUN: not llvm-as -disable-output %t/bitextract-invalid-num-elts.ll 2>&1 | FileCheck %t/bitextract-invalid-num-elts.ll
; RUN: not llvm-as -disable-output %t/bitextract-invalid-offset-num-elts.ll 2>&1 | FileCheck %t/bitextract-invalid-offset-num-elts.ll
; RUN: not llvm-as -disable-output %t/bitextract-invalid-scalable.ll 2>&1 | FileCheck %t/bitextract-invalid-scalable.ll
; RUN: not llvm-as -disable-output %t/bitinsert-invalid-scalar-val.ll 2>&1 | FileCheck %t/bitinsert-invalid-scalar-val.ll
; RUN: not llvm-as -disable-output %t/bitinsert-invalid-scalar-offset.ll 2>&1 | FileCheck %t/bitinsert-invalid-scalar-offset.ll
; RUN: not llvm-as -disable-output %t/bitinsert-invalid-num-elts.ll 2>&1 | FileCheck %t/bitinsert-invalid-num-elts.ll
; RUN: not llvm-as -disable-output %t/bitinsert-invalid-offset-num-elts.ll 2>&1 | FileCheck %t/bitinsert-invalid-offset-num-elts.ll
; RUN: not llvm-as -disable-output %t/bitinsert-invalid-scalable.ll 2>&1 | FileCheck %t/bitinsert-invalid-scalable.ll

;--- bitextract-invalid-1.ll
; CHECK: error: bitextract source must be a byte type or a vector of byte type
define i8 @invalid(i32 %src) {
  %r = bitextract i8, i32 %src, i32 0
  ret i8 %r
}

;--- bitinsert-invalid-1.ll
; CHECK: error: bitinsert base must be a byte type or a vector of byte type
define i32 @invalid(i32 %base, i8 %val) {
  %r = bitinsert i32 %base, i8 %val, i32 0
  ret i32 %r
}

;--- bitextract-invalid-aggregate.ll
; CHECK: error: bitextract result must be an integer, floating-point, pointer, or byte type, or a vector of those types
define void @test_extract_struct(b32 %src) {
  %res = bitextract { i8, i8 }, b32 %src, i32 0
  ret void
}

;--- bitinsert-invalid-aggregate.ll
; CHECK: error: bitinsert value must be an integer, floating-point, pointer, or byte type, or a vector of those types
define void @test_insert_array(b32 %base, [2 x i8] %val) {
  %res = bitinsert b32 %base, [2 x i8] %val, i32 0
  ret void
}

;--- bitextract-invalid-vec-src.ll
; CHECK: error: bitextract source must be a byte type or a vector of byte type
define <2 x i8> @invalid(<2 x i32> %src) {
  %r = bitextract <2 x i8>, <2 x i32> %src, <2 x i32> zeroinitializer
  ret <2 x i8> %r
}

;--- bitinsert-invalid-vec-base.ll
; CHECK: error: bitinsert base must be a byte type or a vector of byte type
define <2 x i32> @invalid(<2 x i32> %base, <2 x i8> %val) {
  %r = bitinsert <2 x i32> %base, <2 x i8> %val, <2 x i32> zeroinitializer
  ret <2 x i32> %r
}

;--- bitextract-invalid-vec-offset-ty.ll
; CHECK: error: bitextract offset must be i32 or a vector of i32
define <2 x i8> @invalid(<2 x b32> %src) {
  %r = bitextract <2 x i8>, <2 x b32> %src, <2 x i64> zeroinitializer
  ret <2 x i8> %r
}

;--- bitinsert-invalid-vec-offset-ty.ll
; CHECK: error: bitinsert offset must be i32 or a vector of i32
define <2 x b32> @invalid(<2 x b32> %base, <2 x i8> %val) {
  %r = bitinsert <2 x b32> %base, <2 x i8> %val, <2 x i64> zeroinitializer
  ret <2 x b32> %r
}

;--- bitextract-invalid-scalar-src.ll
; CHECK: error: bitextract operands must all be scalars or all be vectors with the same number of elements
define <2 x i8> @invalid(b32 %src) {
  %r = bitextract <2 x i8>, b32 %src, <2 x i32> zeroinitializer
  ret <2 x i8> %r
}

;--- bitextract-invalid-scalar-offset.ll
; CHECK: error: bitextract operands must all be scalars or all be vectors with the same number of elements
define <2 x i8> @invalid(<2 x b32> %src) {
  %r = bitextract <2 x i8>, <2 x b32> %src, i32 0
  ret <2 x i8> %r
}

;--- bitextract-invalid-num-elts.ll
; CHECK: error: bitextract operands must all be scalars or all be vectors with the same number of elements
define <2 x i8> @invalid(<4 x b32> %src) {
  %r = bitextract <2 x i8>, <4 x b32> %src, <2 x i32> zeroinitializer
  ret <2 x i8> %r
}

;--- bitextract-invalid-offset-num-elts.ll
; CHECK: error: bitextract operands must all be scalars or all be vectors with the same number of elements
define <2 x i8> @invalid(<2 x b32> %src) {
  %r = bitextract <2 x i8>, <2 x b32> %src, <4 x i32> zeroinitializer
  ret <2 x i8> %r
}

;--- bitextract-invalid-scalable.ll
; CHECK: error: bitextract operands must all be scalars or all be vectors with the same number of elements
define <vscale x 2 x i8> @invalid(<2 x b32> %src) {
  %r = bitextract <vscale x 2 x i8>, <2 x b32> %src, <vscale x 2 x i32> zeroinitializer
  ret <vscale x 2 x i8> %r
}

;--- bitinsert-invalid-scalar-val.ll
; CHECK: error: bitinsert operands must all be scalars or all be vectors with the same number of elements
define <2 x b32> @invalid(<2 x b32> %base, i8 %val) {
  %r = bitinsert <2 x b32> %base, i8 %val, <2 x i32> zeroinitializer
  ret <2 x b32> %r
}

;--- bitinsert-invalid-scalar-offset.ll
; CHECK: error: bitinsert operands must all be scalars or all be vectors with the same number of elements
define <2 x b32> @invalid(<2 x b32> %base, <2 x i8> %val) {
  %r = bitinsert <2 x b32> %base, <2 x i8> %val, i32 0
  ret <2 x b32> %r
}

;--- bitinsert-invalid-num-elts.ll
; CHECK: error: bitinsert operands must all be scalars or all be vectors with the same number of elements
define <2 x b32> @invalid(<2 x b32> %base, <4 x i8> %val) {
  %r = bitinsert <2 x b32> %base, <4 x i8> %val, <2 x i32> zeroinitializer
  ret <2 x b32> %r
}

;--- bitinsert-invalid-offset-num-elts.ll
; CHECK: error: bitinsert operands must all be scalars or all be vectors with the same number of elements
define <2 x b32> @invalid(<2 x b32> %base, <2 x i8> %val) {
  %r = bitinsert <2 x b32> %base, <2 x i8> %val, <4 x i32> zeroinitializer
  ret <2 x b32> %r
}

;--- bitinsert-invalid-scalable.ll
; CHECK: error: bitinsert operands must all be scalars or all be vectors with the same number of elements
define <2 x b32> @invalid(<2 x b32> %base, <vscale x 2 x i8> %val) {
  %r = bitinsert <2 x b32> %base, <vscale x 2 x i8> %val, <2 x i32> zeroinitializer
  ret <2 x b32> %r
}
