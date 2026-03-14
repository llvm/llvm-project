; RUN: llc -mtriple=aarch64-- -O0 -fast-isel -fast-isel-abort=4 -verify-machineinstrs < %s | FileCheck %s
; RUN: llc -mtriple=aarch64-- -O0 -fast-isel=0 -global-isel=false -verify-machineinstrs < %s | FileCheck %s

; Note that checking SelectionDAG output isn't strictly necessary, but they
; currently match, so we might as well check both!  Feel free to remove SDAG.

; CHECK-LABEL: atomic_store_monotonic_8:
; CHECK-NEXT: // %bb.0:
; CHECK-NEXT:  strb  w1, [x0]
; CHECK-NEXT:  ret
define void @atomic_store_monotonic_8(ptr %p, i8 %val) #0 {
  store atomic i8 %val, ptr %p monotonic, align 1
  ret void
}

; CHECK-LABEL: atomic_store_monotonic_8_off:
; CHECK-NEXT: // %bb.0:
; CHECK-NEXT:  strb w1, [x0, #1]
; CHECK-NEXT:  ret
define void @atomic_store_monotonic_8_off(ptr %p, i8 %val) #0 {
  %tmp0 = getelementptr i8, ptr %p, i32 1
  store atomic i8 %val, ptr %tmp0 monotonic, align 1
  ret void
}

; CHECK-LABEL: atomic_store_monotonic_16:
; CHECK-NEXT: // %bb.0:
; CHECK-NEXT:  strh  w1, [x0]
; CHECK-NEXT:  ret
define void @atomic_store_monotonic_16(ptr %p, i16 %val) #0 {
  store atomic i16 %val, ptr %p monotonic, align 2
  ret void
}

; CHECK-LABEL: atomic_store_monotonic_16_off:
; CHECK-NEXT: // %bb.0:
; CHECK-NEXT:  strh w1, [x0, #2]
; CHECK-NEXT:  ret
define void @atomic_store_monotonic_16_off(ptr %p, i16 %val) #0 {
  %tmp0 = getelementptr i16, ptr %p, i32 1
  store atomic i16 %val, ptr %tmp0 monotonic, align 2
  ret void
}

; CHECK-LABEL: atomic_store_monotonic_32:
; CHECK-NEXT: // %bb.0:
; CHECK-NEXT:  str  w1, [x0]
; CHECK-NEXT:  ret
define void @atomic_store_monotonic_32(ptr %p, i32 %val) #0 {
  store atomic i32 %val, ptr %p monotonic, align 4
  ret void
}

; CHECK-LABEL: atomic_store_monotonic_32_off:
; CHECK-NEXT: // %bb.0:
; CHECK-NEXT:  str w1, [x0, #4]
; CHECK-NEXT:  ret
define void @atomic_store_monotonic_32_off(ptr %p, i32 %val) #0 {
  %tmp0 = getelementptr i32, ptr %p, i32 1
  store atomic i32 %val, ptr %tmp0 monotonic, align 4
  ret void
}

; CHECK-LABEL: atomic_store_monotonic_64:
; CHECK-NEXT: // %bb.0:
; CHECK-NEXT:  str  x1, [x0]
; CHECK-NEXT:  ret
define void @atomic_store_monotonic_64(ptr %p, i64 %val) #0 {
  store atomic i64 %val, ptr %p monotonic, align 8
  ret void
}

; CHECK-LABEL: atomic_store_monotonic_64_off:
; CHECK-NEXT: // %bb.0:
; CHECK-NEXT:  str x1, [x0, #8]
; CHECK-NEXT:  ret
define void @atomic_store_monotonic_64_off(ptr %p, i64 %val) #0 {
  %tmp0 = getelementptr i64, ptr %p, i32 1
  store atomic i64 %val, ptr %tmp0 monotonic, align 8
  ret void
}

; CHECK-LABEL: atomic_store_release_8:
; CHECK-NEXT: // %bb.0:
; CHECK-NEXT:  stlrb w1, [x0]
; CHECK-NEXT:  ret
define void @atomic_store_release_8(ptr %p, i8 %val) #0 {
  store atomic i8 %val, ptr %p release, align 1
  ret void
}

; CHECK-LABEL: atomic_store_release_8_off:
; CHECK-NEXT: // %bb.0:
; CHECK-NEXT:  add [[REG0:x[0-9]+]], x0, #1
; CHECK-NEXT:  stlrb w1, [[[REG0]]]
; CHECK-NEXT:  ret
define void @atomic_store_release_8_off(ptr %p, i8 %val) #0 {
  %tmp0 = getelementptr i8, ptr %p, i32 1
  store atomic i8 %val, ptr %tmp0 release, align 1
  ret void
}

; CHECK-LABEL: atomic_store_release_16:
; CHECK-NEXT: // %bb.0:
; CHECK-NEXT:  stlrh w1, [x0]
; CHECK-NEXT:  ret
define void @atomic_store_release_16(ptr %p, i16 %val) #0 {
  store atomic i16 %val, ptr %p release, align 2
  ret void
}

; CHECK-LABEL: atomic_store_release_16_off:
; CHECK-NEXT: // %bb.0:
; CHECK-NEXT:  add [[REG0:x[0-9]+]], x0, #2
; CHECK-NEXT:  stlrh w1, [[[REG0]]]
; CHECK-NEXT:  ret
define void @atomic_store_release_16_off(ptr %p, i16 %val) #0 {
  %tmp0 = getelementptr i16, ptr %p, i32 1
  store atomic i16 %val, ptr %tmp0 release, align 2
  ret void
}

; CHECK-LABEL: atomic_store_release_32:
; CHECK-NEXT: // %bb.0:
; CHECK-NEXT:  stlr w1, [x0]
; CHECK-NEXT:  ret
define void @atomic_store_release_32(ptr %p, i32 %val) #0 {
  store atomic i32 %val, ptr %p release, align 4
  ret void
}

; CHECK-LABEL: atomic_store_release_32_off:
; CHECK-NEXT: // %bb.0:
; CHECK-NEXT:  add [[REG0:x[0-9]+]], x0, #4
; CHECK-NEXT:  stlr w1, [[[REG0]]]
; CHECK-NEXT:  ret
define void @atomic_store_release_32_off(ptr %p, i32 %val) #0 {
  %tmp0 = getelementptr i32, ptr %p, i32 1
  store atomic i32 %val, ptr %tmp0 release, align 4
  ret void
}

; CHECK-LABEL: atomic_store_release_64:
; CHECK-NEXT: // %bb.0:
; CHECK-NEXT:  stlr x1, [x0]
; CHECK-NEXT:  ret
define void @atomic_store_release_64(ptr %p, i64 %val) #0 {
  store atomic i64 %val, ptr %p release, align 8
  ret void
}

; CHECK-LABEL: atomic_store_release_64_off:
; CHECK-NEXT: // %bb.0:
; CHECK-NEXT:  add [[REG0:x[0-9]+]], x0, #8
; CHECK-NEXT:  stlr x1, [[[REG0]]]
; CHECK-NEXT:  ret
define void @atomic_store_release_64_off(ptr %p, i64 %val) #0 {
  %tmp0 = getelementptr i64, ptr %p, i32 1
  store atomic i64 %val, ptr %tmp0 release, align 8
  ret void
}


; CHECK-LABEL: atomic_store_seq_cst_8:
; CHECK-NEXT: // %bb.0:
; CHECK-NEXT:  stlrb w1, [x0]
; CHECK-NEXT:  ret
define void @atomic_store_seq_cst_8(ptr %p, i8 %val) #0 {
  store atomic i8 %val, ptr %p seq_cst, align 1
  ret void
}

; CHECK-LABEL: atomic_store_seq_cst_8_off:
; CHECK-NEXT: // %bb.0:
; CHECK-NEXT:  add [[REG0:x[0-9]+]], x0, #1
; CHECK-NEXT:  stlrb w1, [[[REG0]]]
; CHECK-NEXT:  ret
define void @atomic_store_seq_cst_8_off(ptr %p, i8 %val) #0 {
  %tmp0 = getelementptr i8, ptr %p, i32 1
  store atomic i8 %val, ptr %tmp0 seq_cst, align 1
  ret void
}

; CHECK-LABEL: atomic_store_seq_cst_16:
; CHECK-NEXT: // %bb.0:
; CHECK-NEXT:  stlrh w1, [x0]
; CHECK-NEXT:  ret
define void @atomic_store_seq_cst_16(ptr %p, i16 %val) #0 {
  store atomic i16 %val, ptr %p seq_cst, align 2
  ret void
}

; CHECK-LABEL: atomic_store_seq_cst_16_off:
; CHECK-NEXT: // %bb.0:
; CHECK-NEXT:  add [[REG0:x[0-9]+]], x0, #2
; CHECK-NEXT:  stlrh w1, [[[REG0]]]
; CHECK-NEXT:  ret
define void @atomic_store_seq_cst_16_off(ptr %p, i16 %val) #0 {
  %tmp0 = getelementptr i16, ptr %p, i32 1
  store atomic i16 %val, ptr %tmp0 seq_cst, align 2
  ret void
}

; CHECK-LABEL: atomic_store_seq_cst_32:
; CHECK-NEXT: // %bb.0:
; CHECK-NEXT:  stlr w1, [x0]
; CHECK-NEXT:  ret
define void @atomic_store_seq_cst_32(ptr %p, i32 %val) #0 {
  store atomic i32 %val, ptr %p seq_cst, align 4
  ret void
}

; CHECK-LABEL: atomic_store_seq_cst_32_off:
; CHECK-NEXT: // %bb.0:
; CHECK-NEXT:  add [[REG0:x[0-9]+]], x0, #4
; CHECK-NEXT:  stlr w1, [[[REG0]]]
; CHECK-NEXT:  ret
define void @atomic_store_seq_cst_32_off(ptr %p, i32 %val) #0 {
  %tmp0 = getelementptr i32, ptr %p, i32 1
  store atomic i32 %val, ptr %tmp0 seq_cst, align 4
  ret void
}

; CHECK-LABEL: atomic_store_seq_cst_64:
; CHECK-NEXT: // %bb.0:
; CHECK-NEXT:  stlr x1, [x0]
; CHECK-NEXT:  ret
define void @atomic_store_seq_cst_64(ptr %p, i64 %val) #0 {
  store atomic i64 %val, ptr %p seq_cst, align 8
  ret void
}

; CHECK-LABEL: atomic_store_seq_cst_64_off:
; CHECK-NEXT: // %bb.0:
; CHECK-NEXT:  add [[REG0:x[0-9]+]], x0, #8
; CHECK-NEXT:  stlr x1, [[[REG0]]]
; CHECK-NEXT:  ret
define void @atomic_store_seq_cst_64_off(ptr %p, i64 %val) #0 {
  %tmp0 = getelementptr i64, ptr %p, i32 1
  store atomic i64 %val, ptr %tmp0 seq_cst, align 8
  ret void
}

attributes #0 = { nounwind }
