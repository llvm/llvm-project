; RUN: opt < %s -passes=asan -asan-preserve-topbits=4 -S -mtriple=x86_64-unknown-linux-gnu | FileCheck %s --check-prefixes=CHECK,CHECK-64
; RUN: opt < %s -passes=asan -asan-preserve-topbits=4 -S -mtriple=arm-none-eabi | FileCheck %s --check-prefixes=CHECK,CHECK-32

define void @test(ptr %p) sanitize_address {
; CHECK-LABEL: define void @test(
; CHECK-SAME: ptr [[P:%.*]])
entry:
; CHECK-NEXT: entry:
; CHECK-64-NEXT:   [[ADDR:%[0-9]+]] = ptrtoint ptr [[P]] to i64
; CHECK-64-NEXT:   [[TOP:%[0-9]+]] = and i64 [[ADDR]], -1152921504606846976
; CHECK-64-NEXT:   [[BOTTOM:%[0-9]+]] = and i64 [[ADDR]], 1152921504606846975
; CHECK-64-NEXT:   [[BOTTOM_SHR:%[0-9]+]] = lshr i64 [[BOTTOM]], 3
; CHECK-64-NEXT:   [[SHADOW:%[0-9]+]] = or i64 [[TOP]], [[BOTTOM_SHR]]
; CHECK-64-NEXT:   [[SHADOW_ADDR:%[0-9]+]] = add i64 [[SHADOW]], 2147450880

; CHECK-32-NEXT:   [[ADDR:%[0-9]+]] = ptrtoint ptr [[P]] to i32
; CHECK-32-NEXT:   [[TOP:%[0-9]+]] = and i32 [[ADDR]], -268435456
; CHECK-32-NEXT:   [[BOTTOM:%[0-9]+]] = and i32 [[ADDR]], 268435455
; CHECK-32-NEXT:   [[BOTTOM_SHR:%[0-9]+]] = lshr i32 [[BOTTOM]], 3
; CHECK-32-NEXT:   [[SHADOW:%[0-9]+]] = or i32 [[TOP]], [[BOTTOM_SHR]]
; CHECK-32-NEXT:   [[SHADOW_ADDR:%[0-9]+]] = add i32 [[SHADOW]], 536870912

; CHECK-NEXT:   [[SHADOW_PTR:%[0-9]+]] = inttoptr {{i(64|32)}} [[SHADOW_ADDR]] to ptr
; CHECK-NEXT:   [[SHADOW_BYTE:%[0-9]+]] = load i8, ptr [[SHADOW_PTR]], align 1
; CHECK-NEXT:   [[IS_POISONED:%[0-9]+]] = icmp ne i8 [[SHADOW_BYTE]], 0
; CHECK-NEXT:   br i1 [[IS_POISONED]], label %[[L_REPORT:[0-9]+]], label %[[L_STORE:[0-9]+]]

  store i32 0, ptr %p, align 4
  ret void
}
