; Test spilling using MVC.  The tests here assume z10 register pressure,
; without the high words being available.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z10 -verify-machineinstrs | FileCheck %s

declare void @foo()

@g0 = dso_local global i32 0
@g1 = dso_local global i32 1
@g2 = dso_local global i32 2
@g3 = dso_local global i32 3
@g4 = dso_local global i32 4
@g5 = dso_local global i32 5
@g6 = dso_local global i32 6
@g7 = dso_local global i32 7
@g8 = dso_local global i32 8
@g9 = dso_local global i32 9

@h0 = dso_local global i64 0
@h1 = dso_local global i64 1
@h2 = dso_local global i64 2
@h3 = dso_local global i64 3
@h4 = dso_local global i64 4
@h5 = dso_local global i64 5
@h6 = dso_local global i64 6
@h7 = dso_local global i64 7
@h8 = dso_local global i64 8
@h9 = dso_local global i64 9

; This function shouldn't spill anything
define dso_local void @f1(i32 *%ptr0) {
; CHECK-LABEL: f1:
; CHECK: stmg
; CHECK: aghi %r15, -160
; CHECK-NOT: %r15
; CHECK: brasl %r14, foo@PLT
; CHECK-NOT: %r15
; CHECK: lmg
; CHECK: br %r14
  %ptr1 = getelementptr i32, i32 *%ptr0, i32 2
  %ptr2 = getelementptr i32, i32 *%ptr0, i32 4
  %ptr3 = getelementptr i32, i32 *%ptr0, i32 6
  %ptr4 = getelementptr i32, i32 *%ptr0, i32 8
  %ptr5 = getelementptr i32, i32 *%ptr0, i32 10
  %ptr6 = getelementptr i32, i32 *%ptr0, i32 12

  %val0 = load i32, i32 *%ptr0
  %val1 = load i32, i32 *%ptr1
  %val2 = load i32, i32 *%ptr2
  %val3 = load i32, i32 *%ptr3
  %val4 = load i32, i32 *%ptr4
  %val5 = load i32, i32 *%ptr5
  %val6 = load i32, i32 *%ptr6

  call void @foo()

  store i32 %val0, i32 *%ptr0
  store i32 %val1, i32 *%ptr1
  store i32 %val2, i32 *%ptr2
  store i32 %val3, i32 *%ptr3
  store i32 %val4, i32 *%ptr4
  store i32 %val5, i32 *%ptr5
  store i32 %val6, i32 *%ptr6

  ret void
}

; Test a case where at least one i32 load and at least one i32 store
; need spills.
define dso_local void @f2(i32 *%ptr0) {
; CHECK-LABEL: f2:
; CHECK: mvc [[OFFSET1:16[04]]](4,%r15), [[OFFSET2:[0-9]+]]({{%r[0-9]+}})
; CHECK: brasl %r14, foo@PLT
; CHECK: mvc [[OFFSET2]](4,{{%r[0-9]+}}), [[OFFSET1]](%r15)
; CHECK: br %r14
  %ptr1 = getelementptr i32, i32 *%ptr0, i64 2
  %ptr2 = getelementptr i32, i32 *%ptr0, i64 4
  %ptr3 = getelementptr i32, i32 *%ptr0, i64 6
  %ptr4 = getelementptr i32, i32 *%ptr0, i64 8
  %ptr5 = getelementptr i32, i32 *%ptr0, i64 10
  %ptr6 = getelementptr i32, i32 *%ptr0, i64 12
  %ptr7 = getelementptr i32, i32 *%ptr0, i64 14
  %ptr8 = getelementptr i32, i32 *%ptr0, i64 16

  %val0 = load i32, i32 *%ptr0
  %val1 = load i32, i32 *%ptr1
  %val2 = load i32, i32 *%ptr2
  %val3 = load i32, i32 *%ptr3
  %val4 = load i32, i32 *%ptr4
  %val5 = load i32, i32 *%ptr5
  %val6 = load i32, i32 *%ptr6
  %val7 = load i32, i32 *%ptr7
  %val8 = load i32, i32 *%ptr8

  call void @foo()

  store i32 %val0, i32 *%ptr0
  store i32 %val1, i32 *%ptr1
  store i32 %val2, i32 *%ptr2
  store i32 %val3, i32 *%ptr3
  store i32 %val4, i32 *%ptr4
  store i32 %val5, i32 *%ptr5
  store i32 %val6, i32 *%ptr6
  store i32 %val7, i32 *%ptr7
  store i32 %val8, i32 *%ptr8

  ret void
}

; Test a case where at least one i64 load and at least one i64 store
; need spills.
define dso_local void @f3(i64 *%ptr0) {
; CHECK-LABEL: f3:
; CHECK: mvc 160(8,%r15), [[OFFSET:[0-9]+]]({{%r[0-9]+}})
; CHECK: brasl %r14, foo@PLT
; CHECK: mvc [[OFFSET]](8,{{%r[0-9]+}}), 160(%r15)
; CHECK: br %r14
  %ptr1 = getelementptr i64, i64 *%ptr0, i64 2
  %ptr2 = getelementptr i64, i64 *%ptr0, i64 4
  %ptr3 = getelementptr i64, i64 *%ptr0, i64 6
  %ptr4 = getelementptr i64, i64 *%ptr0, i64 8
  %ptr5 = getelementptr i64, i64 *%ptr0, i64 10
  %ptr6 = getelementptr i64, i64 *%ptr0, i64 12
  %ptr7 = getelementptr i64, i64 *%ptr0, i64 14
  %ptr8 = getelementptr i64, i64 *%ptr0, i64 16

  %val0 = load i64, i64 *%ptr0
  %val1 = load i64, i64 *%ptr1
  %val2 = load i64, i64 *%ptr2
  %val3 = load i64, i64 *%ptr3
  %val4 = load i64, i64 *%ptr4
  %val5 = load i64, i64 *%ptr5
  %val6 = load i64, i64 *%ptr6
  %val7 = load i64, i64 *%ptr7
  %val8 = load i64, i64 *%ptr8

  call void @foo()

  store i64 %val0, i64 *%ptr0
  store i64 %val1, i64 *%ptr1
  store i64 %val2, i64 *%ptr2
  store i64 %val3, i64 *%ptr3
  store i64 %val4, i64 *%ptr4
  store i64 %val5, i64 *%ptr5
  store i64 %val6, i64 *%ptr6
  store i64 %val7, i64 *%ptr7
  store i64 %val8, i64 *%ptr8

  ret void
}


; Test a case where at least at least one f32 load and at least one f32 store
; need spills.  The 8 call-saved FPRs could be used for 8 of the %vals
; (and are at the time of writing), but it would really be better to use
; MVC for all 10.
define dso_local void @f4(float *%ptr0) {
; CHECK-LABEL: f4:
; CHECK: mvc [[OFFSET1:16[04]]](4,%r15), [[OFFSET2:[0-9]+]]({{%r[0-9]+}})
; CHECK: brasl %r14, foo@PLT
; CHECK: mvc [[OFFSET2]](4,{{%r[0-9]+}}), [[OFFSET1]](%r15)
; CHECK: br %r14
  %ptr1 = getelementptr float, float *%ptr0, i64 2
  %ptr2 = getelementptr float, float *%ptr0, i64 4
  %ptr3 = getelementptr float, float *%ptr0, i64 6
  %ptr4 = getelementptr float, float *%ptr0, i64 8
  %ptr5 = getelementptr float, float *%ptr0, i64 10
  %ptr6 = getelementptr float, float *%ptr0, i64 12
  %ptr7 = getelementptr float, float *%ptr0, i64 14
  %ptr8 = getelementptr float, float *%ptr0, i64 16
  %ptr9 = getelementptr float, float *%ptr0, i64 18

  %val0 = load float, float *%ptr0
  %val1 = load float, float *%ptr1
  %val2 = load float, float *%ptr2
  %val3 = load float, float *%ptr3
  %val4 = load float, float *%ptr4
  %val5 = load float, float *%ptr5
  %val6 = load float, float *%ptr6
  %val7 = load float, float *%ptr7
  %val8 = load float, float *%ptr8
  %val9 = load float, float *%ptr9

  call void @foo()

  store float %val0, float *%ptr0
  store float %val1, float *%ptr1
  store float %val2, float *%ptr2
  store float %val3, float *%ptr3
  store float %val4, float *%ptr4
  store float %val5, float *%ptr5
  store float %val6, float *%ptr6
  store float %val7, float *%ptr7
  store float %val8, float *%ptr8
  store float %val9, float *%ptr9

  ret void
}

; Similarly for f64.
define dso_local void @f5(double *%ptr0) {
; CHECK-LABEL: f5:
; CHECK: mvc 160(8,%r15), [[OFFSET:[0-9]+]]({{%r[0-9]+}})
; CHECK: brasl %r14, foo@PLT
; CHECK: mvc [[OFFSET]](8,{{%r[0-9]+}}), 160(%r15)
; CHECK: br %r14
  %ptr1 = getelementptr double, double *%ptr0, i64 2
  %ptr2 = getelementptr double, double *%ptr0, i64 4
  %ptr3 = getelementptr double, double *%ptr0, i64 6
  %ptr4 = getelementptr double, double *%ptr0, i64 8
  %ptr5 = getelementptr double, double *%ptr0, i64 10
  %ptr6 = getelementptr double, double *%ptr0, i64 12
  %ptr7 = getelementptr double, double *%ptr0, i64 14
  %ptr8 = getelementptr double, double *%ptr0, i64 16
  %ptr9 = getelementptr double, double *%ptr0, i64 18

  %val0 = load double, double *%ptr0
  %val1 = load double, double *%ptr1
  %val2 = load double, double *%ptr2
  %val3 = load double, double *%ptr3
  %val4 = load double, double *%ptr4
  %val5 = load double, double *%ptr5
  %val6 = load double, double *%ptr6
  %val7 = load double, double *%ptr7
  %val8 = load double, double *%ptr8
  %val9 = load double, double *%ptr9

  call void @foo()

  store double %val0, double *%ptr0
  store double %val1, double *%ptr1
  store double %val2, double *%ptr2
  store double %val3, double *%ptr3
  store double %val4, double *%ptr4
  store double %val5, double *%ptr5
  store double %val6, double *%ptr6
  store double %val7, double *%ptr7
  store double %val8, double *%ptr8
  store double %val9, double *%ptr9

  ret void
}

; Repeat f2 with atomic accesses.  We shouldn't use MVC here.
define dso_local void @f6(i32 *%ptr0) {
; CHECK-LABEL: f6:
; CHECK-NOT: mvc
; CHECK: br %r14
  %ptr1 = getelementptr i32, i32 *%ptr0, i64 2
  %ptr2 = getelementptr i32, i32 *%ptr0, i64 4
  %ptr3 = getelementptr i32, i32 *%ptr0, i64 6
  %ptr4 = getelementptr i32, i32 *%ptr0, i64 8
  %ptr5 = getelementptr i32, i32 *%ptr0, i64 10
  %ptr6 = getelementptr i32, i32 *%ptr0, i64 12
  %ptr7 = getelementptr i32, i32 *%ptr0, i64 14
  %ptr8 = getelementptr i32, i32 *%ptr0, i64 16

  %val0 = load atomic i32, i32 *%ptr0 unordered, align 4
  %val1 = load atomic i32, i32 *%ptr1 unordered, align 4
  %val2 = load atomic i32, i32 *%ptr2 unordered, align 4
  %val3 = load atomic i32, i32 *%ptr3 unordered, align 4
  %val4 = load atomic i32, i32 *%ptr4 unordered, align 4
  %val5 = load atomic i32, i32 *%ptr5 unordered, align 4
  %val6 = load atomic i32, i32 *%ptr6 unordered, align 4
  %val7 = load atomic i32, i32 *%ptr7 unordered, align 4
  %val8 = load atomic i32, i32 *%ptr8 unordered, align 4

  call void @foo()

  store atomic i32 %val0, i32 *%ptr0 unordered, align 4
  store atomic i32 %val1, i32 *%ptr1 unordered, align 4
  store atomic i32 %val2, i32 *%ptr2 unordered, align 4
  store atomic i32 %val3, i32 *%ptr3 unordered, align 4
  store atomic i32 %val4, i32 *%ptr4 unordered, align 4
  store atomic i32 %val5, i32 *%ptr5 unordered, align 4
  store atomic i32 %val6, i32 *%ptr6 unordered, align 4
  store atomic i32 %val7, i32 *%ptr7 unordered, align 4
  store atomic i32 %val8, i32 *%ptr8 unordered, align 4

  ret void
}

; ...likewise volatile accesses.
define dso_local void @f7(i32 *%ptr0) {
; CHECK-LABEL: f7:
; CHECK-NOT: mvc
; CHECK: br %r14
  %ptr1 = getelementptr i32, i32 *%ptr0, i64 2
  %ptr2 = getelementptr i32, i32 *%ptr0, i64 4
  %ptr3 = getelementptr i32, i32 *%ptr0, i64 6
  %ptr4 = getelementptr i32, i32 *%ptr0, i64 8
  %ptr5 = getelementptr i32, i32 *%ptr0, i64 10
  %ptr6 = getelementptr i32, i32 *%ptr0, i64 12
  %ptr7 = getelementptr i32, i32 *%ptr0, i64 14
  %ptr8 = getelementptr i32, i32 *%ptr0, i64 16

  %val0 = load volatile i32, i32 *%ptr0
  %val1 = load volatile i32, i32 *%ptr1
  %val2 = load volatile i32, i32 *%ptr2
  %val3 = load volatile i32, i32 *%ptr3
  %val4 = load volatile i32, i32 *%ptr4
  %val5 = load volatile i32, i32 *%ptr5
  %val6 = load volatile i32, i32 *%ptr6
  %val7 = load volatile i32, i32 *%ptr7
  %val8 = load volatile i32, i32 *%ptr8

  call void @foo()

  store volatile i32 %val0, i32 *%ptr0
  store volatile i32 %val1, i32 *%ptr1
  store volatile i32 %val2, i32 *%ptr2
  store volatile i32 %val3, i32 *%ptr3
  store volatile i32 %val4, i32 *%ptr4
  store volatile i32 %val5, i32 *%ptr5
  store volatile i32 %val6, i32 *%ptr6
  store volatile i32 %val7, i32 *%ptr7
  store volatile i32 %val8, i32 *%ptr8

  ret void
}

; Check that LRL and STRL are not converted.
define dso_local void @f8() {
; CHECK-LABEL: f8:
; CHECK-NOT: mvc
; CHECK: br %r14
  %val0 = load i32, i32 *@g0
  %val1 = load i32, i32 *@g1
  %val2 = load i32, i32 *@g2
  %val3 = load i32, i32 *@g3
  %val4 = load i32, i32 *@g4
  %val5 = load i32, i32 *@g5
  %val6 = load i32, i32 *@g6
  %val7 = load i32, i32 *@g7
  %val8 = load i32, i32 *@g8
  %val9 = load i32, i32 *@g9

  call void @foo()

  store i32 %val0, i32 *@g0
  store i32 %val1, i32 *@g1
  store i32 %val2, i32 *@g2
  store i32 %val3, i32 *@g3
  store i32 %val4, i32 *@g4
  store i32 %val5, i32 *@g5
  store i32 %val6, i32 *@g6
  store i32 %val7, i32 *@g7
  store i32 %val8, i32 *@g8
  store i32 %val9, i32 *@g9

  ret void
}

; Likewise LGRL and STGRL.
define dso_local void @f9() {
; CHECK-LABEL: f9:
; CHECK-NOT: mvc
; CHECK: br %r14
  %val0 = load i64, i64 *@h0
  %val1 = load i64, i64 *@h1
  %val2 = load i64, i64 *@h2
  %val3 = load i64, i64 *@h3
  %val4 = load i64, i64 *@h4
  %val5 = load i64, i64 *@h5
  %val6 = load i64, i64 *@h6
  %val7 = load i64, i64 *@h7
  %val8 = load i64, i64 *@h8
  %val9 = load i64, i64 *@h9

  call void @foo()

  store i64 %val0, i64 *@h0
  store i64 %val1, i64 *@h1
  store i64 %val2, i64 *@h2
  store i64 %val3, i64 *@h3
  store i64 %val4, i64 *@h4
  store i64 %val5, i64 *@h5
  store i64 %val6, i64 *@h6
  store i64 %val7, i64 *@h7
  store i64 %val8, i64 *@h8
  store i64 %val9, i64 *@h9

  ret void
}

; This showed a problem with the way stack coloring updated instructions.
; The copy from %val9 to %newval8 can be done using an MVC, which then
; has two frame index operands.  Stack coloring chose a valid renumbering
; [FI0, FI1] -> [FI1, FI2], but applied it in the form FI0 -> FI1 -> FI2,
; so that both operands ended up being the same.
define dso_local void @f10() {
; CHECK-LABEL: f10:
; CHECK: lgrl [[REG:%r[0-9]+]], h9
; CHECK: stg [[REG]], [[VAL9:[0-9]+]](%r15)
; CHECK: brasl %r14, foo@PLT
; CHECK: brasl %r14, foo@PLT
; CHECK: mvc [[NEWVAL8:[0-9]+]](8,%r15), [[VAL9]](%r15)
; CHECK: brasl %r14, foo@PLT
; CHECK: lg [[REG:%r[0-9]+]], [[NEWVAL8]](%r15)
; CHECK: stgrl [[REG]], h8
; CHECK: br %r14
entry:
  %val8 = load volatile i64, i64 *@h8
  %val0 = load volatile i64, i64 *@h0
  %val1 = load volatile i64, i64 *@h1
  %val2 = load volatile i64, i64 *@h2
  %val3 = load volatile i64, i64 *@h3
  %val4 = load volatile i64, i64 *@h4
  %val5 = load volatile i64, i64 *@h5
  %val6 = load volatile i64, i64 *@h6
  %val7 = load volatile i64, i64 *@h7
  %val9 = load volatile i64, i64 *@h9

  call void @foo()

  store volatile i64 %val0, i64 *@h0
  store volatile i64 %val1, i64 *@h1
  store volatile i64 %val2, i64 *@h2
  store volatile i64 %val3, i64 *@h3
  store volatile i64 %val4, i64 *@h4
  store volatile i64 %val5, i64 *@h5
  store volatile i64 %val6, i64 *@h6
  store volatile i64 %val7, i64 *@h7

  %check = load volatile i64, i64 *@h0
  %cond = icmp eq i64 %check, 0
  br i1 %cond, label %skip, label %fallthru

fallthru:
  call void @foo()

  store volatile i64 %val0, i64 *@h0
  store volatile i64 %val1, i64 *@h1
  store volatile i64 %val2, i64 *@h2
  store volatile i64 %val3, i64 *@h3
  store volatile i64 %val4, i64 *@h4
  store volatile i64 %val5, i64 *@h5
  store volatile i64 %val6, i64 *@h6
  store volatile i64 %val7, i64 *@h7
  store volatile i64 %val8, i64 *@h8
  br label %skip

skip:
  %newval8 = phi i64 [ %val8, %entry ], [ %val9, %fallthru ]
  call void @foo()

  store volatile i64 %val0, i64 *@h0
  store volatile i64 %val1, i64 *@h1
  store volatile i64 %val2, i64 *@h2
  store volatile i64 %val3, i64 *@h3
  store volatile i64 %val4, i64 *@h4
  store volatile i64 %val5, i64 *@h5
  store volatile i64 %val6, i64 *@h6
  store volatile i64 %val7, i64 *@h7
  store volatile i64 %newval8, i64 *@h8
  store volatile i64 %val9, i64 *@h9

  ret void
}

; This used to generate a no-op MVC.  It is very sensitive to spill heuristics.
define dso_local void @f11() {
; CHECK-LABEL: f11:
; CHECK-NOT: mvc [[OFFSET:[0-9]+]](8,%r15), [[OFFSET]](%r15)
; CHECK: br %r14
entry:
  %val0 = load volatile i64, i64 *@h0
  %val1 = load volatile i64, i64 *@h1
  %val2 = load volatile i64, i64 *@h2
  %val3 = load volatile i64, i64 *@h3
  %val4 = load volatile i64, i64 *@h4
  %val5 = load volatile i64, i64 *@h5
  %val6 = load volatile i64, i64 *@h6
  %val7 = load volatile i64, i64 *@h7

  %altval0 = load volatile i64, i64 *@h0
  %altval1 = load volatile i64, i64 *@h1

  call void @foo()

  store volatile i64 %val0, i64 *@h0
  store volatile i64 %val1, i64 *@h1
  store volatile i64 %val2, i64 *@h2
  store volatile i64 %val3, i64 *@h3
  store volatile i64 %val4, i64 *@h4
  store volatile i64 %val5, i64 *@h5
  store volatile i64 %val6, i64 *@h6
  store volatile i64 %val7, i64 *@h7

  %check = load volatile i64, i64 *@h0
  %cond = icmp eq i64 %check, 0
  br i1 %cond, label %a1, label %b1

a1:
  call void @foo()
  br label %join1

b1:
  call void @foo()
  br label %join1

join1:
  %newval0 = phi i64 [ %val0, %a1 ], [ %altval0, %b1 ]

  call void @foo()

  store volatile i64 %val1, i64 *@h1
  store volatile i64 %val2, i64 *@h2
  store volatile i64 %val3, i64 *@h3
  store volatile i64 %val4, i64 *@h4
  store volatile i64 %val5, i64 *@h5
  store volatile i64 %val6, i64 *@h6
  store volatile i64 %val7, i64 *@h7
  br i1 %cond, label %a2, label %b2

a2:
  call void @foo()
  br label %join2

b2:
  call void @foo()
  br label %join2

join2:
  %newval1 = phi i64 [ %val1, %a2 ], [ %altval1, %b2 ]

  call void @foo()

  store volatile i64 %val2, i64 *@h2
  store volatile i64 %val3, i64 *@h3
  store volatile i64 %val4, i64 *@h4
  store volatile i64 %val5, i64 *@h5
  store volatile i64 %val6, i64 *@h6
  store volatile i64 %val7, i64 *@h7

  call void @foo()

  store volatile i64 %newval0, i64 *@h0
  store volatile i64 %newval1, i64 *@h1
  store volatile i64 %val2, i64 *@h2
  store volatile i64 %val3, i64 *@h3
  store volatile i64 %val4, i64 *@h4
  store volatile i64 %val5, i64 *@h5
  store volatile i64 %val6, i64 *@h6
  store volatile i64 %val7, i64 *@h7

  ret void
}
