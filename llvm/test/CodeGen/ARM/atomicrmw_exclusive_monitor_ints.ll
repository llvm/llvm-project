; Test the instruction sequences produced by atomicrmw instructions. In
; particular, ensure there are no stores/spills inserted between the exclusive
; load and stores, which would invalidate the exclusive monitor.

; RUN: llc -mtriple=armv8-unknown-none-eabi -O0 -o - %s | FileCheck %s --check-prefix=COMMON --check-prefix=EXPAND32 --check-prefix=EXPAND64
; RUN: llc -mtriple=armv6-unknown-none-eabi -O0 -o - %s | FileCheck %s --check-prefix=COMMON --check-prefix=EXPAND32 --check-prefix=EXPAND64
; RUN: llc -mtriple=thumbv7-unknown-none-eabi -O0 -o - %s | FileCheck %s --check-prefix=COMMON --check-prefix=EXPAND32 --check-prefix=EXPAND64
; RUN: llc -mtriple=thumbv6-unknown-none-eabi -O0 -o - %s | FileCheck %s --check-prefix=COMMON --check-prefix=THUMB1
; RUN: llc -mtriple=thumbv8m.base-unknown-none-eabi -O0 -o - %s | FileCheck %s --check-prefix=COMMON --check-prefix=EXPAND32 --check-prefix=BASELINE64

@atomic_i8 = external global i8
@atomic_i16 = external global i16
@atomic_i32 = external global i32
@atomic_i64 = external global i64

define i8 @test_xchg_i8() {
; COMMON-LABEL: test_xchg_i8:
; EXPAND32: ldrexb
; EXPAND32-NOT: str
; EXPAND32: strexb
; THUMB1: bl __sync_lock_test_and_set_1
entry:
  %0 = atomicrmw xchg i8* @atomic_i8, i8 1 monotonic
  ret i8 %0
}
define i8 @test_add_i8() {
; COMMON-LABEL: test_add_i8:
; EXPAND32: ldrexb
; EXPAND32-NOT: str
; EXPAND32: strexb
; THUMB1: bl __sync_fetch_and_add_1
entry:
  %0 = atomicrmw add i8* @atomic_i8, i8 1 monotonic
  ret i8 %0
}
define i8 @test_sub_i8() {
; COMMON-LABEL: test_sub_i8:
; EXPAND32: ldrexb
; EXPAND32-NOT: str
; EXPAND32: strexb
; THUMB1: bl __sync_fetch_and_sub_1
entry:
  %0 = atomicrmw sub i8* @atomic_i8, i8 1 monotonic
  ret i8 %0
}
define i8 @test_and_i8() {
; COMMON-LABEL: test_and_i8:
; EXPAND32: ldrexb
; EXPAND32-NOT: str
; EXPAND32: strexb
; THUMB1: bl __sync_fetch_and_and_1
entry:
  %0 = atomicrmw and i8* @atomic_i8, i8 1 monotonic
  ret i8 %0
}
define i8 @test_nand_i8() {
; COMMON-LABEL: test_nand_i8:
; EXPAND32: ldrexb
; EXPAND32-NOT: str
; EXPAND32: strexb
; THUMB1: bl __sync_fetch_and_nand_1
entry:
  %0 = atomicrmw nand i8* @atomic_i8, i8 1 monotonic
  ret i8 %0
}
define i8 @test_or_i8() {
; COMMON-LABEL: test_or_i8:
; EXPAND32: ldrexb
; EXPAND32-NOT: str
; EXPAND32: strexb
; THUMB1: bl __sync_fetch_and_or_1
entry:
  %0 = atomicrmw or i8* @atomic_i8, i8 1 monotonic
  ret i8 %0
}
define i8 @test_xor_i8() {
; COMMON-LABEL: test_xor_i8:
; EXPAND32: ldrexb
; EXPAND32-NOT: str
; EXPAND32: strexb
; THUMB1: bl __sync_fetch_and_xor_1
entry:
  %0 = atomicrmw xor i8* @atomic_i8, i8 1 monotonic
  ret i8 %0
}
define i8 @test_max_i8() {
; COMMON-LABEL: test_max_i8:
; EXPAND32: ldrexb
; EXPAND32-NOT: str
; EXPAND32: strexb
; THUMB1: bl __sync_fetch_and_max_1
entry:
  %0 = atomicrmw max i8* @atomic_i8, i8 1 monotonic
  ret i8 %0
}
define i8 @test_min_i8() {
; COMMON-LABEL: test_min_i8:
; EXPAND32: ldrexb
; EXPAND32-NOT: str
; EXPAND32: strexb
; THUMB1: bl __sync_fetch_and_min_1
entry:
  %0 = atomicrmw min i8* @atomic_i8, i8 1 monotonic
  ret i8 %0
}
define i8 @test_umax_i8() {
; COMMON-LABEL: test_umax_i8:
; EXPAND32: ldrexb
; EXPAND32-NOT: str
; EXPAND32: strexb
; THUMB1: bl __sync_fetch_and_umax_1
entry:
  %0 = atomicrmw umax i8* @atomic_i8, i8 1 monotonic
  ret i8 %0
}
define i8 @test_umin_i8() {
; COMMON-LABEL: test_umin_i8:
; EXPAND32: ldrexb
; EXPAND32-NOT: str
; EXPAND32: strexb
; THUMB1: bl __sync_fetch_and_umin_1
entry:
  %0 = atomicrmw umin i8* @atomic_i8, i8 1 monotonic
  ret i8 %0
}


define i16 @test_xchg_i16() {
; COMMON-LABEL: test_xchg_i16:
; EXPAND32: ldrexh
; EXPAND32-NOT: str
; EXPAND32: strexh
; THUMB1: bl __sync_lock_test_and_set_2
entry:
  %0 = atomicrmw xchg i16* @atomic_i16, i16 1 monotonic
  ret i16 %0
}
define i16 @test_add_i16() {
; COMMON-LABEL: test_add_i16:
; EXPAND32: ldrexh
; EXPAND32-NOT: str
; EXPAND32: strexh
; THUMB1: bl __sync_fetch_and_add_2
entry:
  %0 = atomicrmw add i16* @atomic_i16, i16 1 monotonic
  ret i16 %0
}
define i16 @test_sub_i16() {
; COMMON-LABEL: test_sub_i16:
; EXPAND32: ldrexh
; EXPAND32-NOT: str
; EXPAND32: strexh
; THUMB1: bl __sync_fetch_and_sub_2
entry:
  %0 = atomicrmw sub i16* @atomic_i16, i16 1 monotonic
  ret i16 %0
}
define i16 @test_and_i16() {
; COMMON-LABEL: test_and_i16:
; EXPAND32: ldrexh
; EXPAND32-NOT: str
; EXPAND32: strexh
; THUMB1: bl __sync_fetch_and_and_2
entry:
  %0 = atomicrmw and i16* @atomic_i16, i16 1 monotonic
  ret i16 %0
}
define i16 @test_nand_i16() {
; COMMON-LABEL: test_nand_i16:
; EXPAND32: ldrexh
; EXPAND32-NOT: str
; EXPAND32: strexh
; THUMB1: bl __sync_fetch_and_nand_2
entry:
  %0 = atomicrmw nand i16* @atomic_i16, i16 1 monotonic
  ret i16 %0
}
define i16 @test_or_i16() {
; COMMON-LABEL: test_or_i16:
; EXPAND32: ldrexh
; EXPAND32-NOT: str
; EXPAND32: strexh
; THUMB1: bl __sync_fetch_and_or_2
entry:
  %0 = atomicrmw or i16* @atomic_i16, i16 1 monotonic
  ret i16 %0
}
define i16 @test_xor_i16() {
; COMMON-LABEL: test_xor_i16:
; EXPAND32: ldrexh
; EXPAND32-NOT: str
; EXPAND32: strexh
; THUMB1: bl __sync_fetch_and_xor_2
entry:
  %0 = atomicrmw xor i16* @atomic_i16, i16 1 monotonic
  ret i16 %0
}
define i16 @test_max_i16() {
; COMMON-LABEL: test_max_i16:
; EXPAND32: ldrexh
; EXPAND32-NOT: str
; EXPAND32: strexh
; THUMB1: bl __sync_fetch_and_max_2
entry:
  %0 = atomicrmw max i16* @atomic_i16, i16 1 monotonic
  ret i16 %0
}
define i16 @test_min_i16() {
; COMMON-LABEL: test_min_i16:
; EXPAND32: ldrexh
; EXPAND32-NOT: str
; EXPAND32: strexh
; THUMB1: bl __sync_fetch_and_min_2
entry:
  %0 = atomicrmw min i16* @atomic_i16, i16 1 monotonic
  ret i16 %0
}
define i16 @test_umax_i16() {
; COMMON-LABEL: test_umax_i16:
; EXPAND32: ldrexh
; EXPAND32-NOT: str
; EXPAND32: strexh
; THUMB1: bl __sync_fetch_and_umax_2
entry:
  %0 = atomicrmw umax i16* @atomic_i16, i16 1 monotonic
  ret i16 %0
}
define i16 @test_umin_i16() {
; COMMON-LABEL: test_umin_i16:
; EXPAND32: ldrexh
; EXPAND32-NOT: str
; EXPAND32: strexh
; THUMB1: bl __sync_fetch_and_umin_2
entry:
  %0 = atomicrmw umin i16* @atomic_i16, i16 1 monotonic
  ret i16 %0
}


define i32 @test_xchg_i32() {
; COMMON-LABEL: test_xchg_i32:
; EXPAND32: ldrex
; EXPAND32-NOT: str
; EXPAND32: strex
; THUMB1: bl __sync_lock_test_and_set_4
entry:
  %0 = atomicrmw xchg i32* @atomic_i32, i32 1 monotonic
  ret i32 %0
}
define i32 @test_add_i32() {
; COMMON-LABEL: test_add_i32:
; EXPAND32: ldrex
; EXPAND32-NOT: str
; EXPAND32: strex
; THUMB1: bl __sync_fetch_and_add_4
entry:
  %0 = atomicrmw add i32* @atomic_i32, i32 1 monotonic
  ret i32 %0
}
define i32 @test_sub_i32() {
; COMMON-LABEL: test_sub_i32:
; EXPAND32: ldrex
; EXPAND32-NOT: str
; EXPAND32: strex
; THUMB1: bl __sync_fetch_and_sub_4
entry:
  %0 = atomicrmw sub i32* @atomic_i32, i32 1 monotonic
  ret i32 %0
}
define i32 @test_and_i32() {
; COMMON-LABEL: test_and_i32:
; EXPAND32: ldrex
; EXPAND32-NOT: str
; EXPAND32: strex
; THUMB1: bl __sync_fetch_and_and_4
entry:
  %0 = atomicrmw and i32* @atomic_i32, i32 1 monotonic
  ret i32 %0
}
define i32 @test_nand_i32() {
; COMMON-LABEL: test_nand_i32:
; EXPAND32: ldrex
; EXPAND32-NOT: str
; EXPAND32: strex
; THUMB1: bl __sync_fetch_and_nand_4
entry:
  %0 = atomicrmw nand i32* @atomic_i32, i32 1 monotonic
  ret i32 %0
}
define i32 @test_or_i32() {
; COMMON-LABEL: test_or_i32:
; EXPAND32: ldrex
; EXPAND32-NOT: str
; EXPAND32: strex
; THUMB1: bl __sync_fetch_and_or_4
entry:
  %0 = atomicrmw or i32* @atomic_i32, i32 1 monotonic
  ret i32 %0
}
define i32 @test_xor_i32() {
; COMMON-LABEL: test_xor_i32:
; EXPAND32: ldrex
; EXPAND32-NOT: str
; EXPAND32: strex
; THUMB1: bl __sync_fetch_and_xor_4
entry:
  %0 = atomicrmw xor i32* @atomic_i32, i32 1 monotonic
  ret i32 %0
}
define i32 @test_max_i32() {
; COMMON-LABEL: test_max_i32:
; EXPAND32: ldrex
; EXPAND32-NOT: str
; EXPAND32: strex
; THUMB1: bl __sync_fetch_and_max_4
entry:
  %0 = atomicrmw max i32* @atomic_i32, i32 1 monotonic
  ret i32 %0
}
define i32 @test_min_i32() {
; COMMON-LABEL: test_min_i32:
; EXPAND32: ldrex
; EXPAND32-NOT: str
; EXPAND32: strex

; THUMB1: bl __sync_fetch_and_min_4
entry:
  %0 = atomicrmw min i32* @atomic_i32, i32 1 monotonic
  ret i32 %0
}
define i32 @test_umax_i32() {
; COMMON-LABEL: test_umax_i32:
; EXPAND32: ldrex
; EXPAND32-NOT: str
; EXPAND32: strex
; THUMB1: bl __sync_fetch_and_umax_4
entry:
  %0 = atomicrmw umax i32* @atomic_i32, i32 1 monotonic
  ret i32 %0
}
define i32 @test_umin_i32() {
; COMMON-LABEL: test_umin_i32:
; EXPAND32: ldrex
; EXPAND32-NOT: str
; EXPAND32: strex
; THUMB1: bl __sync_fetch_and_umin_4
entry:
  %0 = atomicrmw umin i32* @atomic_i32, i32 1 monotonic
  ret i32 %0
}

define i64 @test_xchg_i64() {
; COMMON-LABEL: test_xchg_i64:
; EXPAND64: ldrexd
; EXPAND64-NOT: str
; EXPAND64: strexd
; THUMB1: bl __sync_lock_test_and_set_8
; BASELINE64: bl __sync_val_compare_and_swap_8
entry:
  %0 = atomicrmw xchg i64* @atomic_i64, i64 1 monotonic
  ret i64 %0
}
define i64 @test_add_i64() {
; COMMON-LABEL: test_add_i64:
; EXPAND64: ldrexd
; EXPAND64-NOT: str
; EXPAND64: strexd
; THUMB1: bl __sync_fetch_and_add_8
; BASELINE64: bl __sync_val_compare_and_swap_8
entry:
  %0 = atomicrmw add i64* @atomic_i64, i64 1 monotonic
  ret i64 %0
}
define i64 @test_sub_i64() {
; COMMON-LABEL: test_sub_i64:
; EXPAND64: ldrexd
; EXPAND64-NOT: str
; EXPAND64: strexd
; THUMB1: bl __sync_fetch_and_sub_8
; BASELINE64: bl __sync_val_compare_and_swap_8
entry:
  %0 = atomicrmw sub i64* @atomic_i64, i64 1 monotonic
  ret i64 %0
}
define i64 @test_and_i64() {
; COMMON-LABEL: test_and_i64:
; EXPAND64: ldrexd
; EXPAND64-NOT: str
; EXPAND64: strexd
; THUMB1: bl __sync_fetch_and_and_8
; BASELINE64: bl __sync_val_compare_and_swap_8
entry:
  %0 = atomicrmw and i64* @atomic_i64, i64 1 monotonic
  ret i64 %0
}
define i64 @test_nand_i64() {
; COMMON-LABEL: test_nand_i64:
; EXPAND64: ldrexd
; EXPAND64-NOT: str
; EXPAND64: strexd
; THUMB1: bl __sync_fetch_and_nand_8
; BASELINE64: bl __sync_val_compare_and_swap_8
entry:
  %0 = atomicrmw nand i64* @atomic_i64, i64 1 monotonic
  ret i64 %0
}
define i64 @test_or_i64() {
; COMMON-LABEL: test_or_i64:
; EXPAND64: ldrexd
; EXPAND64-NOT: str
; EXPAND64: strexd
; THUMB1: bl __sync_fetch_and_or_8
; BASELINE64: bl __sync_val_compare_and_swap_8
entry:
  %0 = atomicrmw or i64* @atomic_i64, i64 1 monotonic
  ret i64 %0
}
define i64 @test_xor_i64() {
; COMMON-LABEL: test_xor_i64:
; EXPAND64: ldrexd
; EXPAND64-NOT: str
; EXPAND64: strexd
; THUMB1: bl __sync_fetch_and_xor_8
; BASELINE64: bl __sync_val_compare_and_swap_8
entry:
  %0 = atomicrmw xor i64* @atomic_i64, i64 1 monotonic
  ret i64 %0
}

define i64 @test_max_i64() {
; COMMON-LABEL: test_max_i64:
; EXPAND64: ldrexd
; EXPAND64-NOT: str
; EXPAND64: strexd
; THUMB1: bl __sync_fetch_and_max_8
; BASELINE64: bl __sync_val_compare_and_swap_8
entry:
  %0 = atomicrmw max i64* @atomic_i64, i64 1 monotonic
  ret i64 %0
}
define i64 @test_min_i64() {
; COMMON-LABEL: test_min_i64:
; EXPAND64: ldrexd
; EXPAND64-NOT: str
; EXPAND64: strexd
; THUMB1: bl __sync_fetch_and_min_8
; BASELINE64: bl __sync_val_compare_and_swap_8
entry:
  %0 = atomicrmw min i64* @atomic_i64, i64 1 monotonic
  ret i64 %0
}
define i64 @test_umax_i64() {
; COMMON-LABEL: test_umax_i64:
; EXPAND64: ldrexd
; EXPAND64-NOT: str
; EXPAND64: strexd
; THUMB1: bl __sync_fetch_and_umax_8
; BASELINE64: bl __sync_val_compare_and_swap_8
entry:
  %0 = atomicrmw umax i64* @atomic_i64, i64 1 monotonic
  ret i64 %0
}
define i64 @test_umin_i64() {
; COMMON-LABEL: test_umin_i64:
; EXPAND64: ldrexd
; EXPAND64-NOT: str
; EXPAND64: strexd
; THUMB1: bl __sync_fetch_and_umin_8
; BASELINE64: bl __sync_val_compare_and_swap_8
entry:
  %0 = atomicrmw umin i64* @atomic_i64, i64 1 monotonic
  ret i64 %0
}
