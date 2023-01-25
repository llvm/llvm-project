; RUN: opt -mtriple=x86_64-unknown-linux -S -passes=lowertypetests -lowertypetests-summary-action=import -lowertypetests-read-summary=%S/Inputs/import.yaml %s | FileCheck --check-prefixes=CHECK,X86 %s
; RUN: opt -mtriple=aarch64-unknown-linux -S -passes=lowertypetests -lowertypetests-summary-action=import -lowertypetests-read-summary=%S/Inputs/import.yaml %s | FileCheck --check-prefixes=CHECK,ARM %s

target datalayout = "e-p:64:64"

declare i1 @llvm.type.test(ptr %ptr, metadata %bitset) nounwind readnone

; CHECK-DAG: @__typeid_single_global_addr = external hidden global [0 x i8]
; CHECK-DAG: @__typeid_inline6_global_addr = external hidden global [0 x i8]
; X86-DAG: @__typeid_inline6_align = external hidden global [0 x i8], !absolute_symbol !0
; X86-DAG: @__typeid_inline6_size_m1 = external hidden global [0 x i8], !absolute_symbol !1
; X86-DAG: @__typeid_inline6_inline_bits = external hidden global [0 x i8], !absolute_symbol !2
; CHECK-DAG: @__typeid_inline5_global_addr = external hidden global [0 x i8]
; X86-DAG: @__typeid_inline5_align = external hidden global [0 x i8], !absolute_symbol !0
; X86-DAG: @__typeid_inline5_size_m1 = external hidden global [0 x i8], !absolute_symbol !3
; X86-DAG: @__typeid_inline5_inline_bits = external hidden global [0 x i8], !absolute_symbol !4
; CHECK-DAG: @__typeid_bytearray32_global_addr = external hidden global [0 x i8]
; X86-DAG: @__typeid_bytearray32_align = external hidden global [0 x i8], !absolute_symbol !0
; X86-DAG: @__typeid_bytearray32_size_m1 = external hidden global [0 x i8], !absolute_symbol !4
; CHECK-DAG: @__typeid_bytearray32_byte_array = external hidden global [0 x i8]
; X86-DAG: @__typeid_bytearray32_bit_mask = external hidden global [0 x i8], !absolute_symbol !0
; CHECK-DAG: @__typeid_bytearray7_global_addr = external hidden global [0 x i8]
; X86-DAG: @__typeid_bytearray7_align = external hidden global [0 x i8], !absolute_symbol !0
; X86-DAG: @__typeid_bytearray7_size_m1 = external hidden global [0 x i8], !absolute_symbol !5
; CHECK-DAG: @__typeid_bytearray7_byte_array = external hidden global [0 x i8]
; X86-DAG: @__typeid_bytearray7_bit_mask = external hidden global [0 x i8], !absolute_symbol !0
; CHECK-DAG: @__typeid_allones32_global_addr = external hidden global [0 x i8]
; X86-DAG: @__typeid_allones32_align = external hidden global [0 x i8], !absolute_symbol !0
; X86-DAG: @__typeid_allones32_size_m1 = external hidden global [0 x i8], !absolute_symbol !4
; CHECK-DAG: @__typeid_allones7_global_addr = external hidden global [0 x i8]
; X86-DAG: @__typeid_allones7_align = external hidden global [0 x i8], !absolute_symbol !0
; X86-DAG: @__typeid_allones7_size_m1 = external hidden global [0 x i8], !absolute_symbol !5

; CHECK: define i1 @allones7(ptr [[p:%.*]])
define i1 @allones7(ptr %p) {
  ; CHECK-NEXT: [[pi:%.*]] = ptrtoint ptr [[p]] to i64
  ; CHECK-NEXT: [[sub:%.*]] = sub i64 [[pi]], ptrtoint (ptr @__typeid_allones7_global_addr to i64)
  ; X86-NEXT: [[lshr:%.*]] = lshr i64 [[sub]], zext (i8 ptrtoint (ptr @__typeid_allones7_align to i8) to i64)
  ; X86-NEXT: [[shl:%.*]] = shl i64 [[sub]], zext (i8 sub (i8 64, i8 ptrtoint (ptr @__typeid_allones7_align to i8)) to i64)
  ; ARM-NEXT: [[lshr:%.*]] = lshr i64 [[sub]], 1
  ; ARM-NEXT: [[shl:%.*]] = shl i64 [[sub]], 63
  ; CHECK-NEXT: [[or:%.*]] = or i64 [[lshr]], [[shl]]
  ; X86-NEXT: [[ule:%.*]] = icmp ule i64 [[or]], ptrtoint (ptr @__typeid_allones7_size_m1 to i64)
  ; ARM-NEXT: [[ule:%.*]] = icmp ule i64 [[or]], 42
  ; CHECK-NEXT: ret i1 [[ule]]
  %x = call i1 @llvm.type.test(ptr %p, metadata !"allones7")
  ret i1 %x
}

; CHECK: define i1 @allones32(ptr [[p:%.*]])
define i1 @allones32(ptr %p) {
  ; CHECK-NEXT: [[pi:%.*]] = ptrtoint ptr [[p]] to i64
  ; CHECK-NEXT: [[sub:%.*]] = sub i64 [[pi]], ptrtoint (ptr @__typeid_allones32_global_addr to i64)
  ; X86-NEXT: [[lshr:%.*]] = lshr i64 [[sub]], zext (i8 ptrtoint (ptr @__typeid_allones32_align to i8) to i64)
  ; X86-NEXT: [[shl:%.*]] = shl i64 [[sub]], zext (i8 sub (i8 64, i8 ptrtoint (ptr @__typeid_allones32_align to i8)) to i64)
  ; ARM-NEXT: [[lshr:%.*]] = lshr i64 [[sub]], 2
  ; ARM-NEXT: [[shl:%.*]] = shl i64 [[sub]], 62
  ; CHECK-NEXT: [[or:%.*]] = or i64 [[lshr]], [[shl]]
  ; X86-NEXT: [[ule:%.*]] = icmp ule i64 [[or]], ptrtoint (ptr @__typeid_allones32_size_m1 to i64)
  ; ARM-NEXT: [[ule:%.*]] = icmp ule i64 [[or]], 12345
  ; CHECK-NEXT: ret i1 [[ule]]
  %x = call i1 @llvm.type.test(ptr %p, metadata !"allones32")
  ret i1 %x
}

; CHECK: define i1 @bytearray7(ptr [[p:%.*]])
define i1 @bytearray7(ptr %p) {
  ; CHECK-NEXT: [[pi:%.*]] = ptrtoint ptr [[p]] to i64
  ; CHECK-NEXT: [[sub:%.*]] = sub i64 [[pi]], ptrtoint (ptr @__typeid_bytearray7_global_addr to i64)
  ; X86-NEXT: [[lshr:%.*]] = lshr i64 [[sub]], zext (i8 ptrtoint (ptr @__typeid_bytearray7_align to i8) to i64)
  ; X86-NEXT: [[shl:%.*]] = shl i64 [[sub]], zext (i8 sub (i8 64, i8 ptrtoint (ptr @__typeid_bytearray7_align to i8)) to i64)
  ; ARM-NEXT: [[lshr:%.*]] = lshr i64 [[sub]], 3
  ; ARM-NEXT: [[shl:%.*]] = shl i64 [[sub]], 61
  ; CHECK-NEXT: [[or:%.*]] = or i64 [[lshr]], [[shl]]
  ; X86-NEXT: [[ule:%.*]] = icmp ule i64 [[or]], ptrtoint (ptr @__typeid_bytearray7_size_m1 to i64)
  ; ARM-NEXT: [[ule:%.*]] = icmp ule i64 [[or]], 43
  ; CHECK-NEXT: br i1 [[ule]], label %[[t:.*]], label %[[f:.*]]

  ; CHECK: [[t]]:
  ; CHECK-NEXT: [[gep:%.*]] = getelementptr i8, ptr @__typeid_bytearray7_byte_array, i64 [[or]]
  ; CHECK-NEXT: [[load:%.*]] = load i8, ptr [[gep]]
  ; X86-NEXT: [[and:%.*]] = and i8 [[load]], ptrtoint (ptr @__typeid_bytearray7_bit_mask to i8)
  ; ARM-NEXT: [[and:%.*]] = and i8 [[load]], ptrtoint (ptr inttoptr (i64 64 to ptr) to i8)
  ; CHECK-NEXT: [[ne:%.*]] = icmp ne i8 [[and]], 0
  ; CHECK-NEXT: br label %[[f]]

  ; CHECK: [[f]]:
  ; CHECK-NEXT: [[phi:%.*]] = phi i1 [ false, %0 ], [ [[ne]], %[[t]] ]
  ; CHECK-NEXT: ret i1 [[phi]]
  %x = call i1 @llvm.type.test(ptr %p, metadata !"bytearray7")
  ret i1 %x
}

; CHECK: define i1 @bytearray32(ptr [[p:%.*]])
define i1 @bytearray32(ptr %p) {
  ; CHECK-NEXT: [[pi:%.*]] = ptrtoint ptr [[p]] to i64
  ; CHECK-NEXT: [[sub:%.*]] = sub i64 [[pi]], ptrtoint (ptr @__typeid_bytearray32_global_addr to i64)
  ; X86-NEXT: [[lshr:%.*]] = lshr i64 [[sub]], zext (i8 ptrtoint (ptr @__typeid_bytearray32_align to i8) to i64)
  ; X86-NEXT: [[shl:%.*]] = shl i64 [[sub]], zext (i8 sub (i8 64, i8 ptrtoint (ptr @__typeid_bytearray32_align to i8)) to i64)
  ; ARM-NEXT: [[lshr:%.*]] = lshr i64 [[sub]], 4
  ; ARM-NEXT: [[shl:%.*]] = shl i64 [[sub]], 60
  ; CHECK-NEXT: [[or:%.*]] = or i64 [[lshr]], [[shl]]
  ; X86-NEXT: [[ule:%.*]] = icmp ule i64 [[or]], ptrtoint (ptr @__typeid_bytearray32_size_m1 to i64)
  ; ARM-NEXT: [[ule:%.*]] = icmp ule i64 [[or]], 12346
  ; CHECK-NEXT: br i1 [[ule]], label %[[t:.*]], label %[[f:.*]]

  ; CHECK: [[t]]:
  ; CHECK-NEXT: [[gep:%.*]] = getelementptr i8, ptr @__typeid_bytearray32_byte_array, i64 [[or]]
  ; CHECK-NEXT: [[load:%.*]] = load i8, ptr [[gep]]
  ; X86-NEXT: [[and:%.*]] = and i8 [[load]], ptrtoint (ptr @__typeid_bytearray32_bit_mask to i8)
  ; ARM-NEXT: [[and:%.*]] = and i8 [[load]], ptrtoint (ptr inttoptr (i64 128 to ptr) to i8)
  ; CHECK-NEXT: [[ne:%.*]] = icmp ne i8 [[and]], 0
  ; CHECK-NEXT: br label %[[f]]

  ; CHECK: [[f]]:
  ; CHECK-NEXT: [[phi:%.*]] = phi i1 [ false, %0 ], [ [[ne]], %[[t]] ]
  ; CHECK-NEXT: ret i1 [[phi]]
  %x = call i1 @llvm.type.test(ptr %p, metadata !"bytearray32")
  ret i1 %x
}

; CHECK: define i1 @inline5(ptr [[p:%.*]])
define i1 @inline5(ptr %p) {
  ; CHECK-NEXT: [[pi:%.*]] = ptrtoint ptr [[p]] to i64
  ; CHECK-NEXT: [[sub:%.*]] = sub i64 [[pi]], ptrtoint (ptr @__typeid_inline5_global_addr to i64)
  ; X86-NEXT: [[lshr:%.*]] = lshr i64 [[sub]], zext (i8 ptrtoint (ptr @__typeid_inline5_align to i8) to i64)
  ; X86-NEXT: [[shl:%.*]] = shl i64 [[sub]], zext (i8 sub (i8 64, i8 ptrtoint (ptr @__typeid_inline5_align to i8)) to i64)
  ; ARM-NEXT: [[lshr:%.*]] = lshr i64 [[sub]], 5
  ; ARM-NEXT: [[shl:%.*]] = shl i64 [[sub]], 59
  ; CHECK-NEXT: [[or:%.*]] = or i64 [[lshr]], [[shl]]
  ; X86-NEXT: [[ule:%.*]] = icmp ule i64 [[or]], ptrtoint (ptr @__typeid_inline5_size_m1 to i64)
  ; ARM-NEXT: [[ule:%.*]] = icmp ule i64 [[or]], 31
  ; CHECK-NEXT: br i1 [[ule]], label %[[t:.*]], label %[[f:.*]]

  ; CHECK: [[t]]:
  ; CHECK-NEXT: [[trunc:%.*]] = trunc i64 [[or]] to i32
  ; CHECK-NEXT: [[and:%.*]] = and i32 [[trunc]], 31
  ; CHECK-NEXT: [[shl2:%.*]] = shl i32 1, [[and]]
  ; X86-NEXT: [[and2:%.*]] = and i32 ptrtoint (ptr @__typeid_inline5_inline_bits to i32), [[shl2]]
  ; ARM-NEXT: [[and2:%.*]] = and i32 123, [[shl2]]
  ; CHECK-NEXT: [[ne:%.*]] = icmp ne i32 [[and2]], 0
  ; CHECK-NEXT: br label %[[f]]

  ; CHECK: [[f]]:
  ; CHECK-NEXT: [[phi:%.*]] = phi i1 [ false, %0 ], [ [[ne]], %[[t]] ]
  ; CHECK-NEXT: ret i1 [[phi]]
  %x = call i1 @llvm.type.test(ptr %p, metadata !"inline5")
  ret i1 %x
}

; CHECK: define i1 @inline6(ptr [[p:%.*]])
define i1 @inline6(ptr %p) {
  ; CHECK-NEXT: [[pi:%.*]] = ptrtoint ptr [[p]] to i64
  ; CHECK-NEXT: [[sub:%.*]] = sub i64 [[pi]], ptrtoint (ptr @__typeid_inline6_global_addr to i64)
  ; X86-NEXT: [[lshr:%.*]] = lshr i64 [[sub]], zext (i8 ptrtoint (ptr @__typeid_inline6_align to i8) to i64)
  ; X86-NEXT: [[shl:%.*]] = shl i64 [[sub]], zext (i8 sub (i8 64, i8 ptrtoint (ptr @__typeid_inline6_align to i8)) to i64)
  ; ARM-NEXT: [[lshr:%.*]] = lshr i64 [[sub]], 6
  ; ARM-NEXT: [[shl:%.*]] = shl i64 [[sub]], 58
  ; CHECK-NEXT: [[or:%.*]] = or i64 [[lshr]], [[shl]]
  ; X86-NEXT: [[ule:%.*]] = icmp ule i64 [[or]], ptrtoint (ptr @__typeid_inline6_size_m1 to i64)
  ; ARM-NEXT: [[ule:%.*]] = icmp ule i64 [[or]], 63
  ; CHECK-NEXT: br i1 [[ule]], label %[[t:.*]], label %[[f:.*]]

  ; CHECK: [[t]]:
  ; CHECK-NEXT: [[and:%.*]] = and i64 [[or]], 63
  ; CHECK-NEXT: [[shl2:%.*]] = shl i64 1, [[and]]
  ; X86-NEXT: [[and2:%.*]] = and i64 ptrtoint (ptr @__typeid_inline6_inline_bits to i64), [[shl2]]
  ; ARM-NEXT: [[and2:%.*]] = and i64 1000000000000, [[shl2]]
  ; CHECK-NEXT: [[ne:%.*]] = icmp ne i64 [[and2]], 0
  ; CHECK-NEXT: br label %[[f]]

  ; CHECK: [[f]]:
  ; CHECK-NEXT: [[phi:%.*]] = phi i1 [ false, %0 ], [ [[ne]], %[[t]] ]
  ; CHECK-NEXT: ret i1 [[phi]]
  %x = call i1 @llvm.type.test(ptr %p, metadata !"inline6")
  ret i1 %x
}

; CHECK: define i1 @single(ptr [[p:%.*]])
define i1 @single(ptr %p) {
  ; CHECK-NEXT: [[pi:%.*]] = ptrtoint ptr [[p]] to i64
  ; CHECK-NEXT: [[eq:%.*]] = icmp eq i64 [[pi]], ptrtoint (ptr @__typeid_single_global_addr to i64)
  ; CHECK-NEXT: ret i1 [[eq]]
  %x = call i1 @llvm.type.test(ptr %p, metadata !"single")
  ret i1 %x
}

; X86: !0 = !{i64 0, i64 256}
; X86: !1 = !{i64 0, i64 64}
; X86: !2 = !{i64 -1, i64 -1}
; X86: !3 = !{i64 0, i64 32}
; X86: !4 = !{i64 0, i64 4294967296}
; X86: !5 = !{i64 0, i64 128}
