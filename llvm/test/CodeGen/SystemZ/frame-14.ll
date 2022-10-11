; Test the handling of base + displacement addresses for large frames,
; in cases where both 12-bit and 20-bit displacements are allowed.
; The tests here assume z10 register pressure, without the high words
; being available.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z10 | \
; RUN:   FileCheck -check-prefix=CHECK-NOFP %s
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z10 -frame-pointer=all | \
; RUN:   FileCheck -check-prefix=CHECK-FP %s
;
; This file tests what happens when a displacement is converted from
; being relative to the start of a frame object to being relative to
; the frame itself.  In some cases the test is only possible if two
; objects are allocated.
;
; Rather than rely on a particular order for those objects, the tests
; instead allocate two objects of the same size and apply the test to
; both of them.  For consistency, all tests follow this model, even if
; one object would actually be enough.

; First check the highest offset that is in range of the 12-bit form.
;
; The last in-range doubleword offset is 4088.  Since the frame has two
; emergency spill slots at 160(%r15), the amount that we need to allocate
; in order to put another object at offset 4088 is 4088 - 176 = 3912 bytes.
define void @f1() {
; CHECK-NOFP-LABEL: f1:
; CHECK-NOFP: mvi 4095(%r15), 42
; CHECK-NOFP: br %r14
;
; CHECK-FP-LABEL: f1:
; CHECK-FP: mvi 4095(%r11), 42
; CHECK-FP: br %r14
  %region1 = alloca [3912 x i8], align 8
  %region2 = alloca [3912 x i8], align 8
  %ptr1 = getelementptr inbounds [3912 x i8], ptr %region1, i64 0, i64 7
  %ptr2 = getelementptr inbounds [3912 x i8], ptr %region2, i64 0, i64 7
  store volatile i8 42, ptr %ptr1
  store volatile i8 42, ptr %ptr2
  ret void
}

; Test the first offset that is out-of-range of the 12-bit form.
define void @f2() {
; CHECK-NOFP-LABEL: f2:
; CHECK-NOFP: mviy 4096(%r15), 42
; CHECK-NOFP: br %r14
;
; CHECK-FP-LABEL: f2:
; CHECK-FP: mviy 4096(%r11), 42
; CHECK-FP: br %r14
  %region1 = alloca [3912 x i8], align 8
  %region2 = alloca [3912 x i8], align 8
  %ptr1 = getelementptr inbounds [3912 x i8], ptr %region1, i64 0, i64 8
  %ptr2 = getelementptr inbounds [3912 x i8], ptr %region2, i64 0, i64 8
  store volatile i8 42, ptr %ptr1
  store volatile i8 42, ptr %ptr2
  ret void
}

; Test the last offset that is in range of the 20-bit form.
;
; The last in-range doubleword offset is 524280, so by the same reasoning
; as above, we need to allocate objects of 524280 - 176 = 524104 bytes.
define void @f3() {
; CHECK-NOFP-LABEL: f3:
; CHECK-NOFP: mviy 524287(%r15), 42
; CHECK-NOFP: br %r14
;
; CHECK-FP-LABEL: f3:
; CHECK-FP: mviy 524287(%r11), 42
; CHECK-FP: br %r14
  %region1 = alloca [524104 x i8], align 8
  %region2 = alloca [524104 x i8], align 8
  %ptr1 = getelementptr inbounds [524104 x i8], ptr %region1, i64 0, i64 7
  %ptr2 = getelementptr inbounds [524104 x i8], ptr %region2, i64 0, i64 7
  store volatile i8 42, ptr %ptr1
  store volatile i8 42, ptr %ptr2
  ret void
}

; Test the first out-of-range offset.  We can't use an index register here,
; and the offset is also out of LAY's range, so expect a constant load
; followed by an addition.
define void @f4() {
; CHECK-NOFP-LABEL: f4:
; CHECK-NOFP: llilh %r1, 8
; CHECK-NOFP: la %r1, 0(%r1,%r15)
; CHECK-NOFP: mvi 0(%r1), 42
; CHECK-NOFP: br %r14
;
; CHECK-FP-LABEL: f4:
; CHECK-FP: llilh %r1, 8
; CHECK-FP: la %r1, 0(%r1,%r11)
; CHECK-FP: mvi 0(%r1), 42
; CHECK-FP: br %r14
  %region1 = alloca [524104 x i8], align 8
  %region2 = alloca [524104 x i8], align 8
  %ptr1 = getelementptr inbounds [524104 x i8], ptr %region1, i64 0, i64 8
  %ptr2 = getelementptr inbounds [524104 x i8], ptr %region2, i64 0, i64 8
  store volatile i8 42, ptr %ptr1
  store volatile i8 42, ptr %ptr2
  ret void
}

; Add 4095 to the previous offset, to test the other end of the MVI range.
; The instruction will actually be STCY before frame lowering.
define void @f5() {
; CHECK-NOFP-LABEL: f5:
; CHECK-NOFP: llilh %r1, 8
; CHECK-NOFP: la %r1, 0(%r1,%r15)
; CHECK-NOFP: mvi 4095(%r1), 42
; CHECK-NOFP: br %r14
;
; CHECK-FP-LABEL: f5:
; CHECK-FP: llilh %r1, 8
; CHECK-FP: la %r1, 0(%r1,%r11)
; CHECK-FP: mvi 4095(%r1), 42
; CHECK-FP: br %r14
  %region1 = alloca [524104 x i8], align 8
  %region2 = alloca [524104 x i8], align 8
  %ptr1 = getelementptr inbounds [524104 x i8], ptr %region1, i64 0, i64 4103
  %ptr2 = getelementptr inbounds [524104 x i8], ptr %region2, i64 0, i64 4103
  store volatile i8 42, ptr %ptr1
  store volatile i8 42, ptr %ptr2
  ret void
}

; Test the next offset after that, which uses MVIY instead of MVI.
define void @f6() {
; CHECK-NOFP-LABEL: f6:
; CHECK-NOFP: llilh %r1, 8
; CHECK-NOFP: la %r1, 0(%r1,%r15)
; CHECK-NOFP: mviy 4096(%r1), 42
; CHECK-NOFP: br %r14
;
; CHECK-FP-LABEL: f6:
; CHECK-FP: llilh %r1, 8
; CHECK-FP: la %r1, 0(%r1,%r11)
; CHECK-FP: mviy 4096(%r1), 42
; CHECK-FP: br %r14
  %region1 = alloca [524104 x i8], align 8
  %region2 = alloca [524104 x i8], align 8
  %ptr1 = getelementptr inbounds [524104 x i8], ptr %region1, i64 0, i64 4104
  %ptr2 = getelementptr inbounds [524104 x i8], ptr %region2, i64 0, i64 4104
  store volatile i8 42, ptr %ptr1
  store volatile i8 42, ptr %ptr2
  ret void
}

; Now try an offset of 524287 from the start of the object, with the
; object being at offset 1048576 (1 << 20).  The backend prefers to create
; anchors 0x10000 bytes apart, so that the high part can be loaded using
; LLILH while still using MVI in more cases than 0x40000 anchors would.
define void @f7() {
; CHECK-NOFP-LABEL: f7:
; CHECK-NOFP: llilh %r1, 23
; CHECK-NOFP: la %r1, 0(%r1,%r15)
; CHECK-NOFP: mviy 65535(%r1), 42
; CHECK-NOFP: br %r14
;
; CHECK-FP-LABEL: f7:
; CHECK-FP: llilh %r1, 23
; CHECK-FP: la %r1, 0(%r1,%r11)
; CHECK-FP: mviy 65535(%r1), 42
; CHECK-FP: br %r14
  %region1 = alloca [1048400 x i8], align 8
  %region2 = alloca [1048400 x i8], align 8
  %ptr1 = getelementptr inbounds [1048400 x i8], ptr %region1, i64 0, i64 524287
  %ptr2 = getelementptr inbounds [1048400 x i8], ptr %region2, i64 0, i64 524287
  store volatile i8 42, ptr %ptr1
  store volatile i8 42, ptr %ptr2
  ret void
}

; Keep the object-relative offset the same but bump the size of the
; objects by one doubleword.
define void @f8() {
; CHECK-NOFP-LABEL: f8:
; CHECK-NOFP: llilh %r1, 24
; CHECK-NOFP: la %r1, 0(%r1,%r15)
; CHECK-NOFP: mvi 7(%r1), 42
; CHECK-NOFP: br %r14
;
; CHECK-FP-LABEL: f8:
; CHECK-FP: llilh %r1, 24
; CHECK-FP: la %r1, 0(%r1,%r11)
; CHECK-FP: mvi 7(%r1), 42
; CHECK-FP: br %r14
  %region1 = alloca [1048408 x i8], align 8
  %region2 = alloca [1048408 x i8], align 8
  %ptr1 = getelementptr inbounds [1048408 x i8], ptr %region1, i64 0, i64 524287
  %ptr2 = getelementptr inbounds [1048408 x i8], ptr %region2, i64 0, i64 524287
  store volatile i8 42, ptr %ptr1
  store volatile i8 42, ptr %ptr2
  ret void
}

; Check a case where the original displacement is out of range.  The backend
; should force separate address logic from the outset.  We don't yet do any
; kind of anchor optimization, so there should be no offset on the MVI itself.
;
; Before frame lowering this is an LA followed by the AGFI seen below.
; The LA then gets lowered into the LLILH/LA form.  The exact sequence
; isn't that important though.
define void @f9() {
; CHECK-NOFP-LABEL: f9:
; CHECK-NOFP: llilh [[R1:%r[1-5]]], 16
; CHECK-NOFP: la [[R2:%r[1-5]]], 8([[R1]],%r15)
; CHECK-NOFP: agfi [[R2]], 524288
; CHECK-NOFP: mvi 0([[R2]]), 42
; CHECK-NOFP: br %r14
;
; CHECK-FP-LABEL: f9:
; CHECK-FP: llilh [[R1:%r[1-5]]], 16
; CHECK-FP: la [[R2:%r[1-5]]], 8([[R1]],%r11)
; CHECK-FP: agfi [[R2]], 524288
; CHECK-FP: mvi 0([[R2]]), 42
; CHECK-FP: br %r14
  %region1 = alloca [1048408 x i8], align 8
  %region2 = alloca [1048408 x i8], align 8
  %ptr1 = getelementptr inbounds [1048408 x i8], ptr %region1, i64 0, i64 524288
  %ptr2 = getelementptr inbounds [1048408 x i8], ptr %region2, i64 0, i64 524288
  store volatile i8 42, ptr %ptr1
  store volatile i8 42, ptr %ptr2
  ret void
}

; Repeat f4 in a case that needs the emergency spill slots (because all
; call-clobbered registers are live and no call-saved ones have been
; allocated).
define void @f10(ptr %vptr) {
; CHECK-NOFP-LABEL: f10:
; CHECK-NOFP: stg [[REGISTER:%r[1-9][0-4]?]], [[OFFSET:160|168]](%r15)
; CHECK-NOFP: llilh [[REGISTER]], 8
; CHECK-NOFP: la [[REGISTER]], 0([[REGISTER]],%r15)
; CHECK-NOFP: mvi 0([[REGISTER]]), 42
; CHECK-NOFP: lg [[REGISTER]], [[OFFSET]](%r15)
; CHECK-NOFP: br %r14
;
; CHECK-FP-LABEL: f10:
; CHECK-FP: stg [[REGISTER:%r[1-9][0-4]?]], [[OFFSET:160|168]](%r11)
; CHECK-FP: llilh [[REGISTER]], 8
; CHECK-FP: la [[REGISTER]], 0([[REGISTER]],%r11)
; CHECK-FP: mvi 0([[REGISTER]]), 42
; CHECK-FP: lg [[REGISTER]], [[OFFSET]](%r11)
; CHECK-FP: br %r14
  %i0 = load volatile i32, ptr %vptr
  %i1 = load volatile i32, ptr %vptr
  %i3 = load volatile i32, ptr %vptr
  %i4 = load volatile i32, ptr %vptr
  %i5 = load volatile i32, ptr %vptr
  %region1 = alloca [524104 x i8], align 8
  %region2 = alloca [524104 x i8], align 8
  %ptr1 = getelementptr inbounds [524104 x i8], ptr %region1, i64 0, i64 8
  %ptr2 = getelementptr inbounds [524104 x i8], ptr %region2, i64 0, i64 8
  store volatile i8 42, ptr %ptr1
  store volatile i8 42, ptr %ptr2
  store volatile i32 %i0, ptr %vptr
  store volatile i32 %i1, ptr %vptr
  store volatile i32 %i3, ptr %vptr
  store volatile i32 %i4, ptr %vptr
  store volatile i32 %i5, ptr %vptr
  ret void
}

; And again with maximum register pressure.  The only spill slots that the
; NOFP case needs are the emergency ones, so the offsets are the same as for f4.
; The FP case needs to spill an extra register and is too dependent on
; register allocation heuristics for a stable test.
define void @f11(ptr %vptr) {
; CHECK-NOFP-LABEL: f11:
; CHECK-NOFP: stmg %r6, %r15,
; CHECK-NOFP: stg [[REGISTER:%r[1-9][0-4]?]], [[OFFSET:160|168]](%r15)
; CHECK-NOFP: llilh [[REGISTER]], 8
; CHECK-NOFP: la [[REGISTER]], 0([[REGISTER]],%r15)
; CHECK-NOFP: mvi 0([[REGISTER]]), 42
; CHECK-NOFP: lg [[REGISTER]], [[OFFSET]](%r15)
; CHECK-NOFP: lmg %r6, %r15,
; CHECK-NOFP: br %r14
  %i0 = load volatile i32, ptr %vptr
  %i1 = load volatile i32, ptr %vptr
  %i3 = load volatile i32, ptr %vptr
  %i4 = load volatile i32, ptr %vptr
  %i5 = load volatile i32, ptr %vptr
  %i6 = load volatile i32, ptr %vptr
  %i7 = load volatile i32, ptr %vptr
  %i8 = load volatile i32, ptr %vptr
  %i9 = load volatile i32, ptr %vptr
  %i10 = load volatile i32, ptr %vptr
  %i11 = load volatile i32, ptr %vptr
  %i12 = load volatile i32, ptr %vptr
  %i13 = load volatile i32, ptr %vptr
  %i14 = load volatile i32, ptr %vptr
  %region1 = alloca [524104 x i8], align 8
  %region2 = alloca [524104 x i8], align 8
  %ptr1 = getelementptr inbounds [524104 x i8], ptr %region1, i64 0, i64 8
  %ptr2 = getelementptr inbounds [524104 x i8], ptr %region2, i64 0, i64 8
  store volatile i8 42, ptr %ptr1
  store volatile i8 42, ptr %ptr2
  store volatile i32 %i0, ptr %vptr
  store volatile i32 %i1, ptr %vptr
  store volatile i32 %i3, ptr %vptr
  store volatile i32 %i4, ptr %vptr
  store volatile i32 %i5, ptr %vptr
  store volatile i32 %i6, ptr %vptr
  store volatile i32 %i7, ptr %vptr
  store volatile i32 %i8, ptr %vptr
  store volatile i32 %i9, ptr %vptr
  store volatile i32 %i10, ptr %vptr
  store volatile i32 %i11, ptr %vptr
  store volatile i32 %i12, ptr %vptr
  store volatile i32 %i13, ptr %vptr
  store volatile i32 %i14, ptr %vptr
  ret void
}
