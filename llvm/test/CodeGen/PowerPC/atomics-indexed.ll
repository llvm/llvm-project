; RUN: llc < %s -mtriple=powerpc-unknown-linux-gnu -verify-machineinstrs -ppc-asm-full-reg-names | FileCheck %s --check-prefix=CHECK --check-prefix=PPC32
; FIXME: -verify-machineinstrs currently fail on ppc64 (mismatched register/instruction).
; This is already checked for in Atomics-64.ll
; RUN: llc < %s -mtriple=powerpc64-unknown-linux-gnu -ppc-asm-full-reg-names | FileCheck %s --check-prefix=CHECK --check-prefix=PPC64

; In this file, we check that atomic load/store can make use of the indexed
; versions of the instructions.

; Indexed version of loads
define i8 @load_x_i8_seq_cst([100000 x i8]* %mem) {
; CHECK-LABEL: load_x_i8_seq_cst
; CHECK: sync
; CHECK: lbzx [[VAL:r[0-9]+]]
; CHECK-PPC32: lwsync
; CHECK-PPC64: cmpw [[CR:cr[0-9]+]], [[VAL]], [[VAL]]
; CHECK-PPC64: bne- [[CR]], .+4
; CHECK-PPC64: isync
  %ptr = getelementptr inbounds [100000 x i8], [100000 x i8]* %mem, i64 0, i64 90000
  %val = load atomic i8, i8* %ptr seq_cst, align 1
  ret i8 %val
}
define i16 @load_x_i16_acquire([100000 x i16]* %mem) {
; CHECK-LABEL: load_x_i16_acquire
; CHECK: lhzx [[VAL:r[0-9]+]]
; CHECK-PPC32: lwsync
; CHECK-PPC64: cmpw [[CR:cr[0-9]+]], [[VAL]], [[VAL]]
; CHECK-PPC64: bne- [[CR]], .+4
; CHECK-PPC64: isync
  %ptr = getelementptr inbounds [100000 x i16], [100000 x i16]* %mem, i64 0, i64 90000
  %val = load atomic i16, i16* %ptr acquire, align 2
  ret i16 %val
}
define i32 @load_x_i32_monotonic([100000 x i32]* %mem) {
; CHECK-LABEL: load_x_i32_monotonic
; CHECK: lwzx
; CHECK-NOT: sync
  %ptr = getelementptr inbounds [100000 x i32], [100000 x i32]* %mem, i64 0, i64 90000
  %val = load atomic i32, i32* %ptr monotonic, align 4
  ret i32 %val
}
define i64 @load_x_i64_unordered([100000 x i64]* %mem) {
; CHECK-LABEL: load_x_i64_unordered
; PPC32: __sync_
; PPC64-NOT: __sync_
; PPC64: ldx
; CHECK-NOT: sync
  %ptr = getelementptr inbounds [100000 x i64], [100000 x i64]* %mem, i64 0, i64 90000
  %val = load atomic i64, i64* %ptr unordered, align 8
  ret i64 %val
}

; Indexed version of stores
define void @store_x_i8_seq_cst([100000 x i8]* %mem) {
; CHECK-LABEL: store_x_i8_seq_cst
; CHECK: sync
; CHECK: stbx
  %ptr = getelementptr inbounds [100000 x i8], [100000 x i8]* %mem, i64 0, i64 90000
  store atomic i8 42, i8* %ptr seq_cst, align 1
  ret void
}
define void @store_x_i16_release([100000 x i16]* %mem) {
; CHECK-LABEL: store_x_i16_release
; CHECK: lwsync
; CHECK: sthx
  %ptr = getelementptr inbounds [100000 x i16], [100000 x i16]* %mem, i64 0, i64 90000
  store atomic i16 42, i16* %ptr release, align 2
  ret void
}
define void @store_x_i32_monotonic([100000 x i32]* %mem) {
; CHECK-LABEL: store_x_i32_monotonic
; CHECK-NOT: sync
; CHECK: stwx
  %ptr = getelementptr inbounds [100000 x i32], [100000 x i32]* %mem, i64 0, i64 90000
  store atomic i32 42, i32* %ptr monotonic, align 4
  ret void
}
define void @store_x_i64_unordered([100000 x i64]* %mem) {
; CHECK-LABEL: store_x_i64_unordered
; CHECK-NOT: sync
; PPC32: __sync_
; PPC64-NOT: __sync_
; PPC64: stdx
  %ptr = getelementptr inbounds [100000 x i64], [100000 x i64]* %mem, i64 0, i64 90000
  store atomic i64 42, i64* %ptr unordered, align 8
  ret void
}
