; RUN: opt -S -passes=lowertypetests -lowertypetests-summary-action=import -lowertypetests-read-summary=%S/Inputs/import.yaml %s | FileCheck %s

target datalayout = "e-p:64:64"
target triple = "x86_64-unknown-linux"

declare i1 @llvm.type.test(ptr %ptr, metadata %bitset) nounwind readnone

; CHECK: define i1 @bytearray7(ptr [[p:%.*]])
define i1 @bytearray7(ptr %p) {
  ; CHECK-NEXT: [[pi:%.*]] = ptrtoint ptr [[p]] to i64
  ; CHECK-NEXT: [[sub:%.*]] = sub i64 [[pi]], ptrtoint (ptr @__typeid_bytearray7_global_addr to i64)
  ; CHECK-NEXT: [[lshr:%.*]] = lshr i64 [[sub]], zext (i8 ptrtoint (ptr @__typeid_bytearray7_align to i8) to i64)
  ; CHECK-NEXT: [[shl:%.*]] = shl i64 [[sub]], zext (i8 sub (i8 64, i8 ptrtoint (ptr @__typeid_bytearray7_align to i8)) to i64)
  ; CHECK-NEXT: [[or:%.*]] = or i64 [[lshr]], [[shl]]
  ; CHECK-NEXT: [[ule:%.*]] = icmp ule i64 [[or]], ptrtoint (ptr @__typeid_bytearray7_size_m1 to i64)
  ; CHECK-NEXT: br i1 [[ule]], label %[[t1:.*]], label %[[f:.*]]

  ; CHECK: [[t1]]:
  ; CHECK-NEXT: [[gep:%.*]] = getelementptr i8, ptr @__typeid_bytearray7_byte_array, i64 [[or]]
  ; CHECK-NEXT: [[load:%.*]] = load i8, ptr [[gep]]
  ; CHECK-NEXT: [[and:%.*]] = and i8 [[load]], ptrtoint (ptr @__typeid_bytearray7_bit_mask to i8)
  ; CHECK-NEXT: [[ne:%.*]] = icmp ne i8 [[and]], 0
  ; CHECK-NEXT: br i1 [[ne]], label %[[t:.*]], label %[[f:.*]]

  ; CHECK: [[t]]:
  ; CHECK-NEXT: ret i1 true

  ; CHECK: [[f]]:
  ; CHECK-NEXT: ret i1 false
  %x = call i1 @llvm.type.test(ptr %p, metadata !"bytearray7")
  br i1 %x, label %t, label %f

t:
  ret i1 true

f:
  ret i1 false
}
