; RUN: llc -mtriple=aarch64-- -O0 -fast-isel -fast-isel-abort=4 -verify-machineinstrs < %s | FileCheck %s
; RUN: llc -mtriple=aarch64-- -O0 -fast-isel=0 -global-isel=false -verify-machineinstrs < %s | FileCheck %s

; Note that checking SelectionDAG output isn't strictly necessary, but they
; currently match, so we might as well check both!  Feel free to remove SDAG.

; CHECK-LABEL: atomic_store_monotonic_8:
; CHECK-NEXT: // %bb.0:
; CHECK-NEXT:  strb  w1, [x0]
; CHECK-NEXT:  ret
define void @atomic_store_monotonic_8(i8* %p, i8 %val) #0 {
  store atomic i8 %val, i8* %p monotonic, align 1
  ret void
}

; CHECK-LABEL: atomic_store_monotonic_8_off:
; CHECK-NEXT: // %bb.0:
; CHECK-NEXT:  strb w1, [x0, #1]
; CHECK-NEXT:  ret
define void @atomic_store_monotonic_8_off(i8* %p, i8 %val) #0 {
  %tmp0 = getelementptr i8, i8* %p, i32 1
  store atomic i8 %val, i8* %tmp0 monotonic, align 1
  ret void
}

; CHECK-LABEL: atomic_store_monotonic_16:
; CHECK-NEXT: // %bb.0:
; CHECK-NEXT:  strh  w1, [x0]
; CHECK-NEXT:  ret
define void @atomic_store_monotonic_16(i16* %p, i16 %val) #0 {
  store atomic i16 %val, i16* %p monotonic, align 2
  ret void
}

; CHECK-LABEL: atomic_store_monotonic_16_off:
; CHECK-NEXT: // %bb.0:
; CHECK-NEXT:  strh w1, [x0, #2]
; CHECK-NEXT:  ret
define void @atomic_store_monotonic_16_off(i16* %p, i16 %val) #0 {
  %tmp0 = getelementptr i16, i16* %p, i32 1
  store atomic i16 %val, i16* %tmp0 monotonic, align 2
  ret void
}

; CHECK-LABEL: atomic_store_monotonic_32:
; CHECK-NEXT: // %bb.0:
; CHECK-NEXT:  str  w1, [x0]
; CHECK-NEXT:  ret
define void @atomic_store_monotonic_32(i32* %p, i32 %val) #0 {
  store atomic i32 %val, i32* %p monotonic, align 4
  ret void
}

; CHECK-LABEL: atomic_store_monotonic_32_off:
; CHECK-NEXT: // %bb.0:
; CHECK-NEXT:  str w1, [x0, #4]
; CHECK-NEXT:  ret
define void @atomic_store_monotonic_32_off(i32* %p, i32 %val) #0 {
  %tmp0 = getelementptr i32, i32* %p, i32 1
  store atomic i32 %val, i32* %tmp0 monotonic, align 4
  ret void
}

; CHECK-LABEL: atomic_store_monotonic_64:
; CHECK-NEXT: // %bb.0:
; CHECK-NEXT:  str  x1, [x0]
; CHECK-NEXT:  ret
define void @atomic_store_monotonic_64(i64* %p, i64 %val) #0 {
  store atomic i64 %val, i64* %p monotonic, align 8
  ret void
}

; CHECK-LABEL: atomic_store_monotonic_64_off:
; CHECK-NEXT: // %bb.0:
; CHECK-NEXT:  str x1, [x0, #8]
; CHECK-NEXT:  ret
define void @atomic_store_monotonic_64_off(i64* %p, i64 %val) #0 {
  %tmp0 = getelementptr i64, i64* %p, i32 1
  store atomic i64 %val, i64* %tmp0 monotonic, align 8
  ret void
}

; CHECK-LABEL: atomic_store_release_8:
; CHECK-NEXT: // %bb.0:
; CHECK-NEXT:  stlrb w1, [x0]
; CHECK-NEXT:  ret
define void @atomic_store_release_8(i8* %p, i8 %val) #0 {
  store atomic i8 %val, i8* %p release, align 1
  ret void
}

; CHECK-LABEL: atomic_store_release_8_off:
; CHECK-NEXT: // %bb.0:
; CHECK-NEXT:  add x0, x0, #1
; CHECK-NEXT:  stlrb w1, [x0]
; CHECK-NEXT:  ret
define void @atomic_store_release_8_off(i8* %p, i8 %val) #0 {
  %tmp0 = getelementptr i8, i8* %p, i32 1
  store atomic i8 %val, i8* %tmp0 release, align 1
  ret void
}

; CHECK-LABEL: atomic_store_release_16:
; CHECK-NEXT: // %bb.0:
; CHECK-NEXT:  stlrh w1, [x0]
; CHECK-NEXT:  ret
define void @atomic_store_release_16(i16* %p, i16 %val) #0 {
  store atomic i16 %val, i16* %p release, align 2
  ret void
}

; CHECK-LABEL: atomic_store_release_16_off:
; CHECK-NEXT: // %bb.0:
; CHECK-NEXT:  add x0, x0, #2
; CHECK-NEXT:  stlrh w1, [x0]
; CHECK-NEXT:  ret
define void @atomic_store_release_16_off(i16* %p, i16 %val) #0 {
  %tmp0 = getelementptr i16, i16* %p, i32 1
  store atomic i16 %val, i16* %tmp0 release, align 2
  ret void
}

; CHECK-LABEL: atomic_store_release_32:
; CHECK-NEXT: // %bb.0:
; CHECK-NEXT:  stlr w1, [x0]
; CHECK-NEXT:  ret
define void @atomic_store_release_32(i32* %p, i32 %val) #0 {
  store atomic i32 %val, i32* %p release, align 4
  ret void
}

; CHECK-LABEL: atomic_store_release_32_off:
; CHECK-NEXT: // %bb.0:
; CHECK-NEXT:  add x0, x0, #4
; CHECK-NEXT:  stlr w1, [x0]
; CHECK-NEXT:  ret
define void @atomic_store_release_32_off(i32* %p, i32 %val) #0 {
  %tmp0 = getelementptr i32, i32* %p, i32 1
  store atomic i32 %val, i32* %tmp0 release, align 4
  ret void
}

; CHECK-LABEL: atomic_store_release_64:
; CHECK-NEXT: // %bb.0:
; CHECK-NEXT:  stlr x1, [x0]
; CHECK-NEXT:  ret
define void @atomic_store_release_64(i64* %p, i64 %val) #0 {
  store atomic i64 %val, i64* %p release, align 8
  ret void
}

; CHECK-LABEL: atomic_store_release_64_off:
; CHECK-NEXT: // %bb.0:
; CHECK-NEXT:  add x0, x0, #8
; CHECK-NEXT:  stlr x1, [x0]
; CHECK-NEXT:  ret
define void @atomic_store_release_64_off(i64* %p, i64 %val) #0 {
  %tmp0 = getelementptr i64, i64* %p, i32 1
  store atomic i64 %val, i64* %tmp0 release, align 8
  ret void
}


; CHECK-LABEL: atomic_store_seq_cst_8:
; CHECK-NEXT: // %bb.0:
; CHECK-NEXT:  stlrb w1, [x0]
; CHECK-NEXT:  ret
define void @atomic_store_seq_cst_8(i8* %p, i8 %val) #0 {
  store atomic i8 %val, i8* %p seq_cst, align 1
  ret void
}

; CHECK-LABEL: atomic_store_seq_cst_8_off:
; CHECK-NEXT: // %bb.0:
; CHECK-NEXT:  add x0, x0, #1
; CHECK-NEXT:  stlrb w1, [x0]
; CHECK-NEXT:  ret
define void @atomic_store_seq_cst_8_off(i8* %p, i8 %val) #0 {
  %tmp0 = getelementptr i8, i8* %p, i32 1
  store atomic i8 %val, i8* %tmp0 seq_cst, align 1
  ret void
}

; CHECK-LABEL: atomic_store_seq_cst_16:
; CHECK-NEXT: // %bb.0:
; CHECK-NEXT:  stlrh w1, [x0]
; CHECK-NEXT:  ret
define void @atomic_store_seq_cst_16(i16* %p, i16 %val) #0 {
  store atomic i16 %val, i16* %p seq_cst, align 2
  ret void
}

; CHECK-LABEL: atomic_store_seq_cst_16_off:
; CHECK-NEXT: // %bb.0:
; CHECK-NEXT:  add x0, x0, #2
; CHECK-NEXT:  stlrh w1, [x0]
; CHECK-NEXT:  ret
define void @atomic_store_seq_cst_16_off(i16* %p, i16 %val) #0 {
  %tmp0 = getelementptr i16, i16* %p, i32 1
  store atomic i16 %val, i16* %tmp0 seq_cst, align 2
  ret void
}

; CHECK-LABEL: atomic_store_seq_cst_32:
; CHECK-NEXT: // %bb.0:
; CHECK-NEXT:  stlr w1, [x0]
; CHECK-NEXT:  ret
define void @atomic_store_seq_cst_32(i32* %p, i32 %val) #0 {
  store atomic i32 %val, i32* %p seq_cst, align 4
  ret void
}

; CHECK-LABEL: atomic_store_seq_cst_32_off:
; CHECK-NEXT: // %bb.0:
; CHECK-NEXT:  add x0, x0, #4
; CHECK-NEXT:  stlr w1, [x0]
; CHECK-NEXT:  ret
define void @atomic_store_seq_cst_32_off(i32* %p, i32 %val) #0 {
  %tmp0 = getelementptr i32, i32* %p, i32 1
  store atomic i32 %val, i32* %tmp0 seq_cst, align 4
  ret void
}

; CHECK-LABEL: atomic_store_seq_cst_64:
; CHECK-NEXT: // %bb.0:
; CHECK-NEXT:  stlr x1, [x0]
; CHECK-NEXT:  ret
define void @atomic_store_seq_cst_64(i64* %p, i64 %val) #0 {
  store atomic i64 %val, i64* %p seq_cst, align 8
  ret void
}

; CHECK-LABEL: atomic_store_seq_cst_64_off:
; CHECK-NEXT: // %bb.0:
; CHECK-NEXT:  add x0, x0, #8
; CHECK-NEXT:  stlr x1, [x0]
; CHECK-NEXT:  ret
define void @atomic_store_seq_cst_64_off(i64* %p, i64 %val) #0 {
  %tmp0 = getelementptr i64, i64* %p, i32 1
  store atomic i64 %val, i64* %tmp0 seq_cst, align 8
  ret void
}

attributes #0 = { nounwind }
