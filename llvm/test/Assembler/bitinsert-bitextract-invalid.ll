; RUN: split-file %s %t
; RUN: not llvm-as -disable-output %t/bitextract-invalid-1.ll 2>&1 | FileCheck %s --check-prefix=CHECK-BITEXTRACT-1
; RUN: not llvm-as -disable-output %t/bitinsert-invalid-1.ll 2>&1 | FileCheck %s --check-prefix=CHECK-BITINSERT-1
; RUN: not llvm-as -disable-output %t/bitextract-invalid-aggregate.ll 2>&1 | FileCheck %s --check-prefix=CHECK-BITEXTRACT-AGGREGATE
; RUN: not llvm-as -disable-output %t/bitinsert-invalid-aggregate.ll 2>&1 | FileCheck %s --check-prefix=CHECK-BITINSERT-AGGREGATE

; CHECK-BITEXTRACT-1: error: bitextract source must be a byte type
; CHECK-BITINSERT-1: error: bitinsert base must be a byte type
; CHECK-BITEXTRACT-AGGREGATE: error: bitextract result must be an integer, floating-point, pointer, or byte type
; CHECK-BITINSERT-AGGREGATE: error: bitinsert value must be an integer, floating-point, pointer, or byte type

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
