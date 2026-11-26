; RUN: opt -S -mtriple=nvptx64-nvidia-cuda -passes=infer-address-spaces %s | FileCheck %s

; Discover and clone a ptrtoint/xor/inttoptr round trip that has folded to
; nested constant expressions. The 4-byte swizzle is inside NVPTX's preserved
; generic-to-shared pointer mask.

target triple = "nvptx64-nvidia-cuda"

@shared = addrspace(3) global [16 x i8] zeroinitializer, align 1

; The same constant expression can also be the initializer of a global. The
; pass only rewrites the use in the function and leaves the initializer alone.
@initialized = global ptr inttoptr (
    i64 xor (
        i64 ptrtoint (ptr addrspacecast (ptr addrspace(3) @shared to ptr) to i64),
        i64 4)
    to ptr), align 8
; CHECK: @initialized = global ptr inttoptr (i64 xor (i64 ptrtoint (ptr addrspacecast (ptr addrspace(3) @shared to ptr) to i64), i64 4) to ptr), align 8

; CHECK-LABEL: define i8 @constexpr_xor_round_trip(
; CHECK-NOT: load i8, ptr getelementptr
; CHECK: load i8, ptr addrspace(3) getelementptr
; CHECK-SAME: ptr addrspace(3) addrspacecast (ptr inttoptr (i64 xor (i64 ptrtoint (ptr addrspacecast (ptr addrspace(3) @shared to ptr) to i64), i64 4) to ptr) to ptr addrspace(3))
define i8 @constexpr_xor_round_trip() {
  %value = load i8, ptr getelementptr (
      i8,
      ptr inttoptr (
          i64 xor (
              i64 ptrtoint (
                  ptr addrspacecast (ptr addrspace(3) @shared to ptr) to i64),
              i64 4)
          to ptr),
      i64 1), align 1
  ret i8 %value
}

; A constant-expression round trip is still rejected if it changes a bit
; outside NVPTX's preserved generic-to-shared mask.
; CHECK-LABEL: define i8 @constexpr_xor_outside_preserved_mask(
; CHECK-NOT: load i8, ptr addrspace(3)
; CHECK: load i8, ptr getelementptr
define i8 @constexpr_xor_outside_preserved_mask() {
  %value = load i8, ptr getelementptr (
      i8,
      ptr inttoptr (
          i64 xor (
              i64 ptrtoint (
                  ptr addrspacecast (ptr addrspace(3) @shared to ptr) to i64),
              i64 4096)
          to ptr),
      i64 1), align 1
  ret i8 %value
}

; CHECK-LABEL: define i8 @constexpr_round_trip_also_global_initializer(
; CHECK: load i8, ptr addrspace(3) getelementptr
define i8 @constexpr_round_trip_also_global_initializer() {
  %value = load i8, ptr getelementptr (
      i8,
      ptr inttoptr (
          i64 xor (
              i64 ptrtoint (
                  ptr addrspacecast (ptr addrspace(3) @shared to ptr) to i64),
              i64 4)
          to ptr),
      i64 1), align 1
  ret i8 %value
}
