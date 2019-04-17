; RUN: opt -S -consthoist < %s | FileCheck %s
; RUN: opt -S -passes='consthoist' < %s | FileCheck %s

target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.9.0"

; Check if the materialization of the constant and the cast instruction are
; inserted in the correct order.
define i32 @cast_inst_test() {
; CHECK-LABEL:  @cast_inst_test
; CHECK:        %const = bitcast i64 4646526064 to i64
; CHECK:        %1 = inttoptr i64 %const to i32*
; CHECK:        %v0 = load i32, i32* %1, align 16
; CHECK:        %const_mat = add i64 %const, 16
; CHECK-NEXT:   %2 = inttoptr i64 %const_mat to i32*
; CHECK-NEXT:   %v1 = load i32, i32* %2, align 16
; CHECK:        %const_mat1 = add i64 %const, 32
; CHECK-NEXT:   %3 = inttoptr i64 %const_mat1 to i32*
; CHECK-NEXT:   %v2 = load i32, i32* %3, align 16
  %a0 = inttoptr i64 4646526064 to i32*
  %v0 = load i32, i32* %a0, align 16
  %a1 = inttoptr i64 4646526080 to i32*
  %v1 = load i32, i32* %a1, align 16
  %a2 = inttoptr i64 4646526096 to i32*
  %v2 = load i32, i32* %a2, align 16
  %r0 = add i32 %v0, %v1
  %r1 = add i32 %r0, %v2
  ret i32 %r1
}

