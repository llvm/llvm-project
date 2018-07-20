; Test the handling of base + index + 12-bit displacement addresses for
; large frames, in cases where no 20-bit form exists.  The tests here
; assume z10 register pressure, without the high words being available.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z10 | \
; RUN:   FileCheck -check-prefix=CHECK-NOFP %s
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z10 -disable-fp-elim | \
; RUN:   FileCheck -check-prefix=CHECK-FP %s

declare void @foo(float *%ptr1, float *%ptr2)

; This file tests what happens when a displacement is converted from
; being relative to the start of a frame object to being relative to
; the frame itself.  In some cases the test is only possible if two
; objects are allocated.
;
; Rather than rely on a particular order for those objects, the tests
; instead allocate two objects of the same size and apply the test to
; both of them.  For consistency, all tests follow this model, even if
; one object would actually be enough.

; First check the highest in-range offset after conversion, which is 4092
; for word-addressing instructions like LDEB.
;
; The last in-range doubleword offset is 4088.  Since the frame has two
; emergency spill slots at 160(%r15), the amount that we need to allocate
; in order to put another object at offset 4088 is (4088 - 176) / 4 = 978
; words.
define void @f1(double *%dst) {
; CHECK-NOFP-LABEL: f1:
; CHECK-NOFP: ldeb {{%f[0-7]}}, 4092(%r15)
; CHECK-NOFP: br %r14
;
; CHECK-FP-LABEL: f1:
; CHECK-FP: ldeb {{%f[0-7]}}, 4092(%r11)
; CHECK-FP: br %r14
  %region1 = alloca [978 x float], align 8
  %region2 = alloca [978 x float], align 8
  %start1 = getelementptr inbounds [978 x float], [978 x float]* %region1, i64 0, i64 0
  %start2 = getelementptr inbounds [978 x float], [978 x float]* %region2, i64 0, i64 0
  call void @foo(float *%start1, float *%start2)
  %ptr1 = getelementptr inbounds [978 x float], [978 x float]* %region1, i64 0, i64 1
  %ptr2 = getelementptr inbounds [978 x float], [978 x float]* %region2, i64 0, i64 1
  %float1 = load float, float *%ptr1
  %float2 = load float, float *%ptr2
  %double1 = fpext float %float1 to double
  %double2 = fpext float %float2 to double
  store volatile double %double1, double *%dst
  store volatile double %double2, double *%dst
  ret void
}

; Test the first out-of-range offset.
define void @f2(double *%dst) {
; CHECK-NOFP-LABEL: f2:
; CHECK-NOFP: lghi %r1, 4096
; CHECK-NOFP: ldeb {{%f[0-7]}}, 0(%r1,%r15)
; CHECK-NOFP: br %r14
;
; CHECK-FP-LABEL: f2:
; CHECK-FP: lghi %r1, 4096
; CHECK-FP: ldeb {{%f[0-7]}}, 0(%r1,%r11)
; CHECK-FP: br %r14
  %region1 = alloca [978 x float], align 8
  %region2 = alloca [978 x float], align 8
  %start1 = getelementptr inbounds [978 x float], [978 x float]* %region1, i64 0, i64 0
  %start2 = getelementptr inbounds [978 x float], [978 x float]* %region2, i64 0, i64 0
  call void @foo(float *%start1, float *%start2)
  %ptr1 = getelementptr inbounds [978 x float], [978 x float]* %region1, i64 0, i64 2
  %ptr2 = getelementptr inbounds [978 x float], [978 x float]* %region2, i64 0, i64 2
  %float1 = load float, float *%ptr1
  %float2 = load float, float *%ptr2
  %double1 = fpext float %float1 to double
  %double2 = fpext float %float2 to double
  store volatile double %double1, double *%dst
  store volatile double %double2, double *%dst
  ret void
}

; Test the next offset after that.
define void @f3(double *%dst) {
; CHECK-NOFP-LABEL: f3:
; CHECK-NOFP: lghi %r1, 4096
; CHECK-NOFP: ldeb {{%f[0-7]}}, 4(%r1,%r15)
; CHECK-NOFP: br %r14
;
; CHECK-FP-LABEL: f3:
; CHECK-FP: lghi %r1, 4096
; CHECK-FP: ldeb {{%f[0-7]}}, 4(%r1,%r11)
; CHECK-FP: br %r14
  %region1 = alloca [978 x float], align 8
  %region2 = alloca [978 x float], align 8
  %start1 = getelementptr inbounds [978 x float], [978 x float]* %region1, i64 0, i64 0
  %start2 = getelementptr inbounds [978 x float], [978 x float]* %region2, i64 0, i64 0
  call void @foo(float *%start1, float *%start2)
  %ptr1 = getelementptr inbounds [978 x float], [978 x float]* %region1, i64 0, i64 3
  %ptr2 = getelementptr inbounds [978 x float], [978 x float]* %region2, i64 0, i64 3
  %float1 = load float, float *%ptr1
  %float2 = load float, float *%ptr2
  %double1 = fpext float %float1 to double
  %double2 = fpext float %float2 to double
  store volatile double %double1, double *%dst
  store volatile double %double2, double *%dst
  ret void
}

; Add 4096 bytes (1024 words) to the size of each object and repeat.
define void @f4(double *%dst) {
; CHECK-NOFP-LABEL: f4:
; CHECK-NOFP: lghi %r1, 4096
; CHECK-NOFP: ldeb {{%f[0-7]}}, 4092(%r1,%r15)
; CHECK-NOFP: br %r14
;
; CHECK-FP-LABEL: f4:
; CHECK-FP: lghi %r1, 4096
; CHECK-FP: ldeb {{%f[0-7]}}, 4092(%r1,%r11)
; CHECK-FP: br %r14
  %region1 = alloca [2002 x float], align 8
  %region2 = alloca [2002 x float], align 8
  %start1 = getelementptr inbounds [2002 x float], [2002 x float]* %region1, i64 0, i64 0
  %start2 = getelementptr inbounds [2002 x float], [2002 x float]* %region2, i64 0, i64 0
  call void @foo(float *%start1, float *%start2)
  %ptr1 = getelementptr inbounds [2002 x float], [2002 x float]* %region1, i64 0, i64 1
  %ptr2 = getelementptr inbounds [2002 x float], [2002 x float]* %region2, i64 0, i64 1
  %float1 = load float, float *%ptr1
  %float2 = load float, float *%ptr2
  %double1 = fpext float %float1 to double
  %double2 = fpext float %float2 to double
  store volatile double %double1, double *%dst
  store volatile double %double2, double *%dst
  ret void
}

; ...as above.
define void @f5(double *%dst) {
; CHECK-NOFP-LABEL: f5:
; CHECK-NOFP: lghi %r1, 8192
; CHECK-NOFP: ldeb {{%f[0-7]}}, 0(%r1,%r15)
; CHECK-NOFP: br %r14
;
; CHECK-FP-LABEL: f5:
; CHECK-FP: lghi %r1, 8192
; CHECK-FP: ldeb {{%f[0-7]}}, 0(%r1,%r11)
; CHECK-FP: br %r14
  %region1 = alloca [2002 x float], align 8
  %region2 = alloca [2002 x float], align 8
  %start1 = getelementptr inbounds [2002 x float], [2002 x float]* %region1, i64 0, i64 0
  %start2 = getelementptr inbounds [2002 x float], [2002 x float]* %region2, i64 0, i64 0
  call void @foo(float *%start1, float *%start2)
  %ptr1 = getelementptr inbounds [2002 x float], [2002 x float]* %region1, i64 0, i64 2
  %ptr2 = getelementptr inbounds [2002 x float], [2002 x float]* %region2, i64 0, i64 2
  %float1 = load float, float *%ptr1
  %float2 = load float, float *%ptr2
  %double1 = fpext float %float1 to double
  %double2 = fpext float %float2 to double
  store volatile double %double1, double *%dst
  store volatile double %double2, double *%dst
  ret void
}

; ...as above.
define void @f6(double *%dst) {
; CHECK-NOFP-LABEL: f6:
; CHECK-NOFP: lghi %r1, 8192
; CHECK-NOFP: ldeb {{%f[0-7]}}, 4(%r1,%r15)
; CHECK-NOFP: br %r14
;
; CHECK-FP-LABEL: f6:
; CHECK-FP: lghi %r1, 8192
; CHECK-FP: ldeb {{%f[0-7]}}, 4(%r1,%r11)
; CHECK-FP: br %r14
  %region1 = alloca [2002 x float], align 8
  %region2 = alloca [2002 x float], align 8
  %start1 = getelementptr inbounds [2002 x float], [2002 x float]* %region1, i64 0, i64 0
  %start2 = getelementptr inbounds [2002 x float], [2002 x float]* %region2, i64 0, i64 0
  call void @foo(float *%start1, float *%start2)
  %ptr1 = getelementptr inbounds [2002 x float], [2002 x float]* %region1, i64 0, i64 3
  %ptr2 = getelementptr inbounds [2002 x float], [2002 x float]* %region2, i64 0, i64 3
  %float1 = load float, float *%ptr1
  %float2 = load float, float *%ptr2
  %double1 = fpext float %float1 to double
  %double2 = fpext float %float2 to double
  store volatile double %double1, double *%dst
  store volatile double %double2, double *%dst
  ret void
}

; Now try an offset of 4092 from the start of the object, with the object
; being at offset 8192.  This time we need objects of (8192 - 168) / 4 = 2004
; words.
define void @f7(double *%dst) {
; CHECK-NOFP-LABEL: f7:
; CHECK-NOFP: lghi %r1, 8192
; CHECK-NOFP: ldeb {{%f[0-7]}}, 4092(%r1,%r15)
; CHECK-NOFP: br %r14
;
; CHECK-FP-LABEL: f7:
; CHECK-FP: lghi %r1, 8192
; CHECK-FP: ldeb {{%f[0-7]}}, 4092(%r1,%r11)
; CHECK-FP: br %r14
  %region1 = alloca [2004 x float], align 8
  %region2 = alloca [2004 x float], align 8
  %start1 = getelementptr inbounds [2004 x float], [2004 x float]* %region1, i64 0, i64 0
  %start2 = getelementptr inbounds [2004 x float], [2004 x float]* %region2, i64 0, i64 0
  call void @foo(float *%start1, float *%start2)
  %ptr1 = getelementptr inbounds [2004 x float], [2004 x float]* %region1, i64 0, i64 1023
  %ptr2 = getelementptr inbounds [2004 x float], [2004 x float]* %region2, i64 0, i64 1023
  %float1 = load float, float *%ptr1
  %float2 = load float, float *%ptr2
  %double1 = fpext float %float1 to double
  %double2 = fpext float %float2 to double
  store volatile double %double1, double *%dst
  store volatile double %double2, double *%dst
  ret void
}

; Keep the object-relative offset the same but bump the size of the
; objects by one doubleword.
define void @f8(double *%dst) {
; CHECK-NOFP-LABEL: f8:
; CHECK-NOFP: lghi %r1, 12288
; CHECK-NOFP: ldeb {{%f[0-7]}}, 4(%r1,%r15)
; CHECK-NOFP: br %r14
;
; CHECK-FP-LABEL: f8:
; CHECK-FP: lghi %r1, 12288
; CHECK-FP: ldeb {{%f[0-7]}}, 4(%r1,%r11)
; CHECK-FP: br %r14
  %region1 = alloca [2006 x float], align 8
  %region2 = alloca [2006 x float], align 8
  %start1 = getelementptr inbounds [2006 x float], [2006 x float]* %region1, i64 0, i64 0
  %start2 = getelementptr inbounds [2006 x float], [2006 x float]* %region2, i64 0, i64 0
  call void @foo(float *%start1, float *%start2)
  %ptr1 = getelementptr inbounds [2006 x float], [2006 x float]* %region1, i64 0, i64 1023
  %ptr2 = getelementptr inbounds [2006 x float], [2006 x float]* %region2, i64 0, i64 1023
  %float1 = load float, float *%ptr1
  %float2 = load float, float *%ptr2
  %double1 = fpext float %float1 to double
  %double2 = fpext float %float2 to double
  store volatile double %double1, double *%dst
  store volatile double %double2, double *%dst
  ret void
}

; Check a case where the original displacement is out of range.  The backend
; should force an LAY from the outset.  We don't yet do any kind of anchor
; optimization, so there should be no offset on the LDEB itself.
define void @f9(double *%dst) {
; CHECK-NOFP-LABEL: f9:
; CHECK-NOFP: lay %r1, 12296(%r15)
; CHECK-NOFP: ldeb {{%f[0-7]}}, 0(%r1)
; CHECK-NOFP: br %r14
;
; CHECK-FP-LABEL: f9:
; CHECK-FP: lay %r1, 12296(%r11)
; CHECK-FP: ldeb {{%f[0-7]}}, 0(%r1)
; CHECK-FP: br %r14
  %region1 = alloca [2006 x float], align 8
  %region2 = alloca [2006 x float], align 8
  %start1 = getelementptr inbounds [2006 x float], [2006 x float]* %region1, i64 0, i64 0
  %start2 = getelementptr inbounds [2006 x float], [2006 x float]* %region2, i64 0, i64 0
  call void @foo(float *%start1, float *%start2)
  %ptr1 = getelementptr inbounds [2006 x float], [2006 x float]* %region1, i64 0, i64 1024
  %ptr2 = getelementptr inbounds [2006 x float], [2006 x float]* %region2, i64 0, i64 1024
  %float1 = load float, float *%ptr1
  %float2 = load float, float *%ptr2
  %double1 = fpext float %float1 to double
  %double2 = fpext float %float2 to double
  store volatile double %double1, double *%dst
  store volatile double %double2, double *%dst
  ret void
}

; Repeat f2 in a case that needs the emergency spill slots, because all
; call-clobbered and allocated call-saved registers are live.  Note that
; %vptr and %dst are copied to call-saved registers, freeing up %r2 and
; %r3 during the main test.
define void @f10(i32 *%vptr, double *%dst) {
; CHECK-NOFP-LABEL: f10:
; CHECK-NOFP: stg [[REGISTER:%r[1-9][0-4]?]], [[OFFSET:160|168]](%r15)
; CHECK-NOFP: lghi [[REGISTER]], 4096
; CHECK-NOFP: ldeb {{%f[0-7]}}, 0([[REGISTER]],%r15)
; CHECK-NOFP: lg [[REGISTER]], [[OFFSET]](%r15)
; CHECK-NOFP: br %r14
;
; CHECK-FP-LABEL: f10:
; CHECK-FP: stg [[REGISTER:%r[1-9][0-4]?]], [[OFFSET:160|168]](%r11)
; CHECK-FP: lghi [[REGISTER]], 4096
; CHECK-FP: ldeb {{%f[0-7]}}, 0([[REGISTER]],%r11)
; CHECK-FP: lg [[REGISTER]], [[OFFSET]](%r11)
; CHECK-FP: br %r14
  %region1 = alloca [978 x float], align 8
  %region2 = alloca [978 x float], align 8
  %start1 = getelementptr inbounds [978 x float], [978 x float]* %region1, i64 0, i64 0
  %start2 = getelementptr inbounds [978 x float], [978 x float]* %region2, i64 0, i64 0
  call void @foo(float *%start1, float *%start2)
  %ptr1 = getelementptr inbounds [978 x float], [978 x float]* %region1, i64 0, i64 2
  %ptr2 = getelementptr inbounds [978 x float], [978 x float]* %region2, i64 0, i64 2
  %i0 = load volatile i32, i32 *%vptr
  %i1 = load volatile i32, i32 *%vptr
  %i2 = load volatile i32, i32 *%vptr
  %i3 = load volatile i32, i32 *%vptr
  %i4 = load volatile i32, i32 *%vptr
  %i5 = load volatile i32, i32 *%vptr
  %i14 = load volatile i32, i32 *%vptr
  %float1 = load float, float *%ptr1
  %float2 = load float, float *%ptr2
  %double1 = fpext float %float1 to double
  %double2 = fpext float %float2 to double
  store volatile double %double1, double *%dst
  store volatile double %double2, double *%dst
  store volatile i32 %i0, i32 *%vptr
  store volatile i32 %i1, i32 *%vptr
  store volatile i32 %i2, i32 *%vptr
  store volatile i32 %i3, i32 *%vptr
  store volatile i32 %i4, i32 *%vptr
  store volatile i32 %i5, i32 *%vptr
  store volatile i32 %i14, i32 *%vptr
  ret void
}

; Repeat f2 in a case where the index register is already occupied.
define void @f11(double *%dst, i64 %index) {
; CHECK-NOFP-LABEL: f11:
; CHECK-NOFP: lgr [[REGISTER:%r[1-9][0-5]?]], %r3
; CHECK-NOFP: lay %r1, 4096(%r15)
; CHECK-NOFP: ldeb {{%f[0-7]}}, 0([[REGISTER]],%r1)
; CHECK-NOFP: br %r14
;
; CHECK-FP-LABEL: f11:
; CHECK-FP: lgr [[REGISTER:%r[1-9][0-5]?]], %r3
; CHECK-FP: lay %r1, 4096(%r11)
; CHECK-FP: ldeb {{%f[0-7]}}, 0([[REGISTER]],%r1)
; CHECK-FP: br %r14
  %region1 = alloca [978 x float], align 8
  %region2 = alloca [978 x float], align 8
  %start1 = getelementptr inbounds [978 x float], [978 x float]* %region1, i64 0, i64 0
  %start2 = getelementptr inbounds [978 x float], [978 x float]* %region2, i64 0, i64 0
  call void @foo(float *%start1, float *%start2)
  %elem1 = getelementptr inbounds [978 x float], [978 x float]* %region1, i64 0, i64 2
  %elem2 = getelementptr inbounds [978 x float], [978 x float]* %region2, i64 0, i64 2
  %base1 = ptrtoint float *%elem1 to i64
  %base2 = ptrtoint float *%elem2 to i64
  %addr1 = add i64 %base1, %index
  %addr2 = add i64 %base2, %index
  %ptr1 = inttoptr i64 %addr1 to float *
  %ptr2 = inttoptr i64 %addr2 to float *
  %float1 = load float, float *%ptr1
  %float2 = load float, float *%ptr2
  %double1 = fpext float %float1 to double
  %double2 = fpext float %float2 to double
  store volatile double %double1, double *%dst
  store volatile double %double2, double *%dst
  ret void
}
