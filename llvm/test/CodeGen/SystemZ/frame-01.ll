; Test the allocation of frames in cases where we do not need to save
; registers in the prologue.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s

declare void @foo(ptr)

; The CFA offset is 160 (the caller-allocated part of the frame) + 168.
define void @f1(i64 %x) {
; CHECK-LABEL: f1:
; CHECK: aghi %r15, -168
; CHECK: .cfi_def_cfa_offset 328
; CHECK: stg %r2, 160(%r15)
; CHECK: aghi %r15, 168
; CHECK: br %r14
  %y = alloca i64, align 8
  store volatile i64 %x, ptr %y
  ret void
}

; Check frames of size 32760, which is the largest size that can be both
; allocated and freed using AGHI.  This size is big enough to require
; two emergency spill slots at 160(%r15), for instructions with unsigned
; 12-bit offsets that end up being out of range.  Fill the remaining
; 32760 - 176 bytes by allocating (32760 - 176) / 8 = 4073 doublewords.
define void @f2(i64 %x) {
; CHECK-LABEL: f2:
; CHECK: aghi %r15, -32760
; CHECK: .cfi_def_cfa_offset 32920
; CHECK: stg %r2, 176(%r15)
; CHECK: aghi %r15, 32760
; CHECK: br %r14
  %y = alloca [4073 x i64], align 8
  store volatile i64 %x, ptr %y
  ret void
}

; Allocate one more doubleword.  This is the one frame size that we can
; allocate using AGHI but must free using AGFI.
define void @f3(i64 %x) {
; CHECK-LABEL: f3:
; CHECK: aghi %r15, -32768
; CHECK: .cfi_def_cfa_offset 32928
; CHECK: stg %r2, 176(%r15)
; CHECK: agfi %r15, 32768
; CHECK: br %r14
  %y = alloca [4074 x i64], align 8
  store volatile i64 %x, ptr %y
  ret void
}

; Allocate another doubleword on top of that.  The allocation and free
; must both use AGFI.
define void @f4(i64 %x) {
; CHECK-LABEL: f4:
; CHECK: agfi %r15, -32776
; CHECK: .cfi_def_cfa_offset 32936
; CHECK: stg %r2, 176(%r15)
; CHECK: agfi %r15, 32776
; CHECK: br %r14
  %y = alloca [4075 x i64], align 8
  store volatile i64 %x, ptr %y
  ret void
}

; The largest size that can be both allocated and freed using AGFI.
; At this point the frame is too big to represent properly in the CFI.
define void @f5(i64 %x) {
; CHECK-LABEL: f5:
; CHECK: agfi %r15, -2147483640
; CHECK: stg %r2, 176(%r15)
; CHECK: agfi %r15, 2147483640
; CHECK: br %r14
  %y = alloca [268435433 x i64], align 8
  store volatile i64 %x, ptr %y
  ret void
}

; The only frame size that can be allocated using a single AGFI but which
; must be freed using two instructions.
define void @f6(i64 %x) {
; CHECK-LABEL: f6:
; CHECK: agfi %r15, -2147483648
; CHECK: stg %r2, 176(%r15)
; CHECK: agfi %r15, 2147483640
; CHECK: aghi %r15, 8
; CHECK: br %r14
  %y = alloca [268435434 x i64], align 8
  store volatile i64 %x, ptr %y
  ret void
}

; The smallest frame size that needs two instructions to both allocate
; and free the frame.
define void @f7(i64 %x) {
; CHECK-LABEL: f7:
; CHECK: agfi %r15, -2147483648
; CHECK: aghi %r15, -8
; CHECK: stg %r2, 176(%r15)
; CHECK: agfi %r15, 2147483640
; CHECK: aghi %r15, 16
; CHECK: br %r14
  %y = alloca [268435435 x i64], align 8
  store volatile i64 %x, ptr %y
  ret void
}

; Make sure that LA can be rematerialized.
define void @f8() {
; CHECK-LABEL: f8:
; CHECK: la %r2, 164(%r15)
; CHECK: brasl %r14, foo@PLT
; CHECK: la %r2, 164(%r15)
; CHECK: brasl %r14, foo@PLT
; CHECK: br %r14
  %ptr = alloca i32
  call void @foo(ptr %ptr)
  call void @foo(ptr %ptr)
  ret void
}
