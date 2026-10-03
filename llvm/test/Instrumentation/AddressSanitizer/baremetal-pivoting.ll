; RUN: opt < %s -passes=asan -asan-memory-pivot=0x20010000 -asan-mapping-offset=0x20004000 -asan-preserve-topbits=4 -S -mtriple=arm-none-eabi | FileCheck %s --check-prefixes=CHECK,CHECK-32-MASK
; RUN: opt < %s -passes=asan -asan-memory-pivot=0x20010000 -asan-mapping-offset=0x20004000 -S -mtriple=arm-none-eabi | FileCheck %s --check-prefixes=CHECK,CHECK-32-NOMASK
; RUN: opt < %s -passes=asan -asan-memory-pivot=0x1000000000010000 -asan-mapping-offset=0x2000000000004000 -asan-preserve-topbits=16 -S -mtriple=aarch64-none-elf | FileCheck %s --check-prefixes=CHECK,CHECK-64-MASK

; RUN: not --crash opt < %s -passes=asan -asan-memory-pivot=0x20010000 -S -mtriple=arm-none-eabi 2>&1 | FileCheck %s --check-prefix=ERR-MISSING-BASE
; RUN: not --crash opt < %s -passes=asan -asan-preserve-topbits=32 -S -mtriple=arm-none-eabi 2>&1 | FileCheck %s --check-prefix=ERR-INVALID-TOPBITS

; ERR-MISSING-BASE: LLVM ERROR: ASan memory pivoting (-asan-memory-pivot) requires an explicit static shadow base (-asan-mapping-offset) and is incompatible with zero or dynamic shadow mapping.
; ERR-INVALID-TOPBITS: LLVM ERROR: -asan-preserve-topbits must be strictly smaller than the target pointer bit width.

define void @test(ptr %p) sanitize_address {
; CHECK-LABEL: define void @test(
; CHECK-SAME: ptr [[P:%.*]])
entry:
; CHECK-NEXT: entry:
; --- 32-bit RP2350 / Pico SDK layout ---
; Memory Pivot: 0x20010000 (masked with 0x0FFFFFFF -> 0x00010000 = 65536)
; Shadow Base:  0x20004000 (decimal 536887296)
; Region Mask (top 4 bits stripped): 0x0FFFFFFF (decimal 268435455)
; CHECK-32-MASK-NEXT:   [[ADDR:%[0-9]+]] = ptrtoint ptr [[P]] to i32
; CHECK-32-MASK-NEXT:   [[MASKED:%[0-9]+]] = and i32 [[ADDR]], 268435455
; CHECK-32-MASK-NEXT:   [[DELTA:%[0-9]+]] = sub i32 [[MASKED]], 65536
; CHECK-32-MASK-NEXT:   [[SHIFTED:%[0-9]+]] = ashr i32 [[DELTA]], 3
; CHECK-32-MASK-NEXT:   [[SHADOW_ADDR:%[0-9]+]] = add i32 536887296, [[SHIFTED]]

; CHECK-32-NOMASK-NEXT: [[ADDR:%[0-9]+]] = ptrtoint ptr [[P]] to i32
; CHECK-32-NOMASK-NEXT: [[DELTA:%[0-9]+]] = sub i32 [[ADDR]], 536936448
; CHECK-32-NOMASK-NEXT: [[SHIFTED:%[0-9]+]] = ashr i32 [[DELTA]], 3
; CHECK-32-NOMASK-NEXT: [[SHADOW_ADDR:%[0-9]+]] = add i32 536887296, [[SHIFTED]]

; --- 64-bit layout ---
; Memory Pivot: 0x1000000000010000 (masked with 0x0000FFFFFFFFFFFF -> 0x00010000 = 65536)
; Shadow Base:  0x2000000000004000 (decimal 2305843009213710336)
; Region Mask (top 16 bits stripped): 0x0000FFFFFFFFFFFF (decimal 281474976710655)
; CHECK-64-MASK-NEXT:   [[ADDR:%[0-9]+]] = ptrtoint ptr [[P]] to i64
; CHECK-64-MASK-NEXT:   [[MASKED:%[0-9]+]] = and i64 [[ADDR]], 281474976710655
; CHECK-64-MASK-NEXT:   [[DELTA:%[0-9]+]] = sub i64 [[MASKED]], 65536
; CHECK-64-MASK-NEXT:   [[SHIFTED:%[0-9]+]] = ashr i64 [[DELTA]], 3
; CHECK-64-MASK-NEXT:   [[SHADOW_ADDR:%[0-9]+]] = add i64 2305843009213710336, [[SHIFTED]]

; CHECK-NEXT:   [[SHADOW_PTR:%[0-9]+]] = inttoptr {{i(64|32)}} [[SHADOW_ADDR]] to ptr
; CHECK-NEXT:   [[SHADOW_BYTE:%[0-9]+]] = load i8, ptr [[SHADOW_PTR]], align 1
; CHECK-NEXT:   [[IS_POISONED:%[0-9]+]] = icmp ne i8 [[SHADOW_BYTE]], 0
; CHECK-NEXT:   br i1 [[IS_POISONED]], label %[[L_REPORT:[0-9]+]], label %[[L_STORE:[0-9]+]]

  store i32 0, ptr %p, align 4
  ret void
}
