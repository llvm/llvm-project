; RUN: llc -ppc-asm-full-reg-names -verify-machineinstrs -mtriple=powerpc64le-unknown-linux-gnu -mcpu=pwr9 < %s | FileCheck %s

; test_no_prep:
; unsigned long test_no_prep(char *p, int count) {
;   unsigned long i=0, res=0;
;   int DISP1 = 4001;
;   int DISP2 = 4002;
;   int DISP3 = 4003;
;   int DISP4 = 4004;
;   for (; i < count ; i++) {
;     unsigned long x1 = *(unsigned long *)(p + i + DISP1);
;     unsigned long x2 = *(unsigned long *)(p + i + DISP2);
;     unsigned long x3 = *(unsigned long *)(p + i + DISP3);
;     unsigned long x4 = *(unsigned long *)(p + i + DISP4);
;     res += x1*x2*x3*x4;
;   }
;   return res + count;
; }

define i64 @test_no_prep(i8* %0, i32 signext %1) {
; CHECK-LABEL: test_no_prep:
; CHECK:         addi r3, r3, 4004
; CHECK:       .LBB0_2: #
; CHECK-NEXT:    ldx r9, r3, r6
; CHECK-NEXT:    ldx r10, r3, r7
; CHECK-NEXT:    mulld r9, r10, r9
; CHECK-NEXT:    ldx r11, r3, r8
; CHECK-NEXT:    mulld r9, r9, r11
; CHECK-NEXT:    ld r12, 0(r3)
; CHECK-NEXT:    addi r3, r3, 1
; CHECK-NEXT:    maddld r5, r9, r12, r5
; CHECK-NEXT:    bdnz .LBB0_2
  %3 = sext i32 %1 to i64
  %4 = icmp eq i32 %1, 0
  br i1 %4, label %27, label %5

5:                                                ; preds = %2, %5
  %6 = phi i64 [ %25, %5 ], [ 0, %2 ]
  %7 = phi i64 [ %24, %5 ], [ 0, %2 ]
  %8 = getelementptr inbounds i8, i8* %0, i64 %6
  %9 = getelementptr inbounds i8, i8* %8, i64 4001
  %10 = bitcast i8* %9 to i64*
  %11 = load i64, i64* %10, align 8
  %12 = getelementptr inbounds i8, i8* %8, i64 4002
  %13 = bitcast i8* %12 to i64*
  %14 = load i64, i64* %13, align 8
  %15 = getelementptr inbounds i8, i8* %8, i64 4003
  %16 = bitcast i8* %15 to i64*
  %17 = load i64, i64* %16, align 8
  %18 = getelementptr inbounds i8, i8* %8, i64 4004
  %19 = bitcast i8* %18 to i64*
  %20 = load i64, i64* %19, align 8
  %21 = mul i64 %14, %11
  %22 = mul i64 %21, %17
  %23 = mul i64 %22, %20
  %24 = add i64 %23, %7
  %25 = add nuw i64 %6, 1
  %26 = icmp ult i64 %25, %3
  br i1 %26, label %5, label %27

27:                                               ; preds = %5, %2
  %28 = phi i64 [ 0, %2 ], [ %24, %5 ]
  %29 = add i64 %28, %3
  ret i64 %29
}

; test_ds_prep:
; unsigned long test_ds_prep(char *p, int count) {
;   unsigned long i=0, res=0;
;   int DISP1 = 4001;
;   int DISP2 = 4002;
;   int DISP3 = 4003;
;   int DISP4 = 4006;
;   for (; i < count ; i++) {
;     unsigned long x1 = *(unsigned long *)(p + i + DISP1);
;     unsigned long x2 = *(unsigned long *)(p + i + DISP2);
;     unsigned long x3 = *(unsigned long *)(p + i + DISP3);
;     unsigned long x4 = *(unsigned long *)(p + i + DISP4);
;     res += x1*x2*x3*x4;
;   }
;   return res + count;
; }

define i64 @test_ds_prep(i8* %0, i32 signext %1) {
; CHECK-LABEL: test_ds_prep:
; CHECK:         addi r6, r3, 4002
; CHECK:       .LBB1_2: #
; CHECK-NEXT:    ldx r9, r6, r7
; CHECK-NEXT:    ld r10, 0(r6)
; CHECK-NEXT:    mulld r9, r10, r9
; CHECK-NEXT:    ldx r11, r6, r5
; CHECK-NEXT:    mulld r9, r9, r11
; CHECK-NEXT:    addi r8, r6, 1
; CHECK-NEXT:    ld r6, 4(r6)
; CHECK-NEXT:    maddld r3, r9, r6, r3
; CHECK-NEXT:    mr r6, r8
; CHECK-NEXT:    bdnz .LBB1_2
  %3 = sext i32 %1 to i64
  %4 = icmp eq i32 %1, 0
  br i1 %4, label %27, label %5

5:                                                ; preds = %2, %5
  %6 = phi i64 [ %25, %5 ], [ 0, %2 ]
  %7 = phi i64 [ %24, %5 ], [ 0, %2 ]
  %8 = getelementptr inbounds i8, i8* %0, i64 %6
  %9 = getelementptr inbounds i8, i8* %8, i64 4001
  %10 = bitcast i8* %9 to i64*
  %11 = load i64, i64* %10, align 8
  %12 = getelementptr inbounds i8, i8* %8, i64 4002
  %13 = bitcast i8* %12 to i64*
  %14 = load i64, i64* %13, align 8
  %15 = getelementptr inbounds i8, i8* %8, i64 4003
  %16 = bitcast i8* %15 to i64*
  %17 = load i64, i64* %16, align 8
  %18 = getelementptr inbounds i8, i8* %8, i64 4006
  %19 = bitcast i8* %18 to i64*
  %20 = load i64, i64* %19, align 8
  %21 = mul i64 %14, %11
  %22 = mul i64 %21, %17
  %23 = mul i64 %22, %20
  %24 = add i64 %23, %7
  %25 = add nuw i64 %6, 1
  %26 = icmp ult i64 %25, %3
  br i1 %26, label %5, label %27

27:                                               ; preds = %5, %2
  %28 = phi i64 [ 0, %2 ], [ %24, %5 ]
  %29 = add i64 %28, %3
  ret i64 %29
}

; test_max_number_reminder:
; unsigned long test_max_number_reminder(char *p, int count) {
;  unsigned long i=0, res=0;
;  int DISP1 = 4001;
;  int DISP2 = 4002;
;  int DISP3 = 4003;
;  int DISP4 = 4005;
;  int DISP5 = 4006;
;  int DISP6 = 4007;
;  int DISP7 = 4014;
;  int DISP8 = 4010;
;  int DISP9 = 4011;
;  for (; i < count ; i++) {
;    unsigned long x1 = *(unsigned long *)(p + i + DISP1);
;    unsigned long x2 = *(unsigned long *)(p + i + DISP2);
;    unsigned long x3 = *(unsigned long *)(p + i + DISP3);
;    unsigned long x4 = *(unsigned long *)(p + i + DISP4);
;    unsigned long x5 = *(unsigned long *)(p + i + DISP5);
;    unsigned long x6 = *(unsigned long *)(p + i + DISP6);
;    unsigned long x7 = *(unsigned long *)(p + i + DISP7);
;    unsigned long x8 = *(unsigned long *)(p + i + DISP8);
;    unsigned long x9 = *(unsigned long *)(p + i + DISP9);
;    res += x1*x2*x3*x4*x5*x6*x7*x8*x9;
;  }
;  return res + count;
;}

define i64 @test_max_number_reminder(i8* %0, i32 signext %1) {
; CHECK-LABEL: test_max_number_reminder:
; CHECK:         addi r9, r3, 4002
; CHECK:       .LBB2_2: #
; CHECK-NEXT:    ldx r12, r9, r6
; CHECK-NEXT:    ld r0, 0(r9)
; CHECK-NEXT:    mulld r12, r0, r12
; CHECK-NEXT:    addi r11, r9, 1
; CHECK-NEXT:    ldx r30, r9, r7
; CHECK-NEXT:    ld r29, 4(r9)
; CHECK-NEXT:    ldx r28, r9, r8
; CHECK-NEXT:    ld r27, 12(r9)
; CHECK-NEXT:    ld r26, 8(r9)
; CHECK-NEXT:    ldx r25, r9, r10
; CHECK-NEXT:    ldx r9, r9, r5
; CHECK-NEXT:    mulld r9, r12, r9
; CHECK-NEXT:    mulld r9, r9, r30
; CHECK-NEXT:    mulld r9, r9, r29
; CHECK-NEXT:    mulld r9, r9, r28
; CHECK-NEXT:    mulld r9, r9, r27
; CHECK-NEXT:    mulld r9, r9, r26
; CHECK-NEXT:    maddld r3, r9, r25, r3
; CHECK-NEXT:    mr r9, r11
; CHECK-NEXT:    bdnz .LBB2_2
  %3 = sext i32 %1 to i64
  %4 = icmp eq i32 %1, 0
  br i1 %4, label %47, label %5

5:                                                ; preds = %2, %5
  %6 = phi i64 [ %45, %5 ], [ 0, %2 ]
  %7 = phi i64 [ %44, %5 ], [ 0, %2 ]
  %8 = getelementptr inbounds i8, i8* %0, i64 %6
  %9 = getelementptr inbounds i8, i8* %8, i64 4001
  %10 = bitcast i8* %9 to i64*
  %11 = load i64, i64* %10, align 8
  %12 = getelementptr inbounds i8, i8* %8, i64 4002
  %13 = bitcast i8* %12 to i64*
  %14 = load i64, i64* %13, align 8
  %15 = getelementptr inbounds i8, i8* %8, i64 4003
  %16 = bitcast i8* %15 to i64*
  %17 = load i64, i64* %16, align 8
  %18 = getelementptr inbounds i8, i8* %8, i64 4005
  %19 = bitcast i8* %18 to i64*
  %20 = load i64, i64* %19, align 8
  %21 = getelementptr inbounds i8, i8* %8, i64 4006
  %22 = bitcast i8* %21 to i64*
  %23 = load i64, i64* %22, align 8
  %24 = getelementptr inbounds i8, i8* %8, i64 4007
  %25 = bitcast i8* %24 to i64*
  %26 = load i64, i64* %25, align 8
  %27 = getelementptr inbounds i8, i8* %8, i64 4014
  %28 = bitcast i8* %27 to i64*
  %29 = load i64, i64* %28, align 8
  %30 = getelementptr inbounds i8, i8* %8, i64 4010
  %31 = bitcast i8* %30 to i64*
  %32 = load i64, i64* %31, align 8
  %33 = getelementptr inbounds i8, i8* %8, i64 4011
  %34 = bitcast i8* %33 to i64*
  %35 = load i64, i64* %34, align 8
  %36 = mul i64 %14, %11
  %37 = mul i64 %36, %17
  %38 = mul i64 %37, %20
  %39 = mul i64 %38, %23
  %40 = mul i64 %39, %26
  %41 = mul i64 %40, %29
  %42 = mul i64 %41, %32
  %43 = mul i64 %42, %35
  %44 = add i64 %43, %7
  %45 = add nuw i64 %6, 1
  %46 = icmp ult i64 %45, %3
  br i1 %46, label %5, label %47

47:                                               ; preds = %5, %2
  %48 = phi i64 [ 0, %2 ], [ %44, %5 ]
  %49 = add i64 %48, %3
  ret i64 %49
}

; test_update_ds_prep_interact:
; unsigned long test_update_ds_prep_interact(char *p, int count) {
;   unsigned long i=0, res=0;
;   int DISP1 = 4001;
;   int DISP2 = 4002;
;   int DISP3 = 4003;
;   int DISP4 = 4006;
;   for (; i < count ; i++) {
;     unsigned long x1 = *(unsigned long *)(p + 4 * i + DISP1);
;     unsigned long x2 = *(unsigned long *)(p + 4 * i + DISP2);
;     unsigned long x3 = *(unsigned long *)(p + 4 * i + DISP3);
;     unsigned long x4 = *(unsigned long *)(p + 4 * i + DISP4);
;     res += x1*x2*x3*x4;
;   }
;   return res + count;
; }

define dso_local i64 @test_update_ds_prep_interact(i8* %0, i32 signext %1) {
; CHECK-LABEL: test_update_ds_prep_interact:
; CHECK:         addi r3, r3, 3998
; CHECK:       .LBB3_2: #
; CHECK-NEXT:    ldu r8, 4(r3)
; CHECK-NEXT:    ldx r9, r3, r7
; CHECK-NEXT:    mulld r8, r8, r9
; CHECK-NEXT:    ldx r10, r3, r6
; CHECK-NEXT:    mulld r8, r8, r10
; CHECK-NEXT:    ld r11, 4(r3)
; CHECK-NEXT:    maddld r5, r8, r11, r5
; CHECK-NEXT:    bdnz .LBB3_2
  %3 = sext i32 %1 to i64
  %4 = icmp eq i32 %1, 0
  br i1 %4, label %28, label %5

5:                                                ; preds = %2, %5
  %6 = phi i64 [ %26, %5 ], [ 0, %2 ]
  %7 = phi i64 [ %25, %5 ], [ 0, %2 ]
  %8 = shl i64 %6, 2
  %9 = getelementptr inbounds i8, i8* %0, i64 %8
  %10 = getelementptr inbounds i8, i8* %9, i64 4001
  %11 = bitcast i8* %10 to i64*
  %12 = load i64, i64* %11, align 8
  %13 = getelementptr inbounds i8, i8* %9, i64 4002
  %14 = bitcast i8* %13 to i64*
  %15 = load i64, i64* %14, align 8
  %16 = getelementptr inbounds i8, i8* %9, i64 4003
  %17 = bitcast i8* %16 to i64*
  %18 = load i64, i64* %17, align 8
  %19 = getelementptr inbounds i8, i8* %9, i64 4006
  %20 = bitcast i8* %19 to i64*
  %21 = load i64, i64* %20, align 8
  %22 = mul i64 %15, %12
  %23 = mul i64 %22, %18
  %24 = mul i64 %23, %21
  %25 = add i64 %24, %7
  %26 = add nuw i64 %6, 1
  %27 = icmp ult i64 %26, %3
  br i1 %27, label %5, label %28

28:                                               ; preds = %5, %2
  %29 = phi i64 [ 0, %2 ], [ %25, %5 ]
  %30 = add i64 %29, %3
  ret i64 %30
}

; test_update_ds_prep_nointeract:
; unsigned long test_update_ds_prep_nointeract(char *p, int count) {
;   unsigned long i=0, res=0;
;   int DISP1 = 4001;
;   int DISP2 = 4002;
;   int DISP3 = 4003;
;   int DISP4 = 4007;
;   for (; i < count ; i++) {
;     char x1 = *(p + i + DISP1);
;     unsigned long x2 = *(unsigned long *)(p + i + DISP2);
;     unsigned long x3 = *(unsigned long *)(p + i + DISP3);
;     unsigned long x4 = *(unsigned long *)(p + i + DISP4);
;     res += (unsigned long)x1*x2*x3*x4;
;   }
;   return res + count;
; }

define i64 @test_update_ds_prep_nointeract(i8* %0, i32 signext %1) {
; CHECK-LABEL: test_update_ds_prep_nointeract:
; CHECK:         addi r5, r3, 4000
; CHECK:         addi r3, r3, 4003
; CHECK:       .LBB4_2: #
; CHECK-NEXT:    lbzu r8, 1(r5)
; CHECK-NEXT:    ldx r9, r3, r7
; CHECK-NEXT:    ld r10, 0(r3)
; CHECK-NEXT:    ld r11, 4(r3)
; CHECK-NEXT:    addi r3, r3, 1
; CHECK-NEXT:    mulld r8, r9, r8
; CHECK-NEXT:    mulld r8, r8, r10
; CHECK-NEXT:    maddld r6, r8, r11, r6
; CHECK-NEXT:    bdnz .LBB4_2
  %3 = sext i32 %1 to i64
  %4 = icmp eq i32 %1, 0
  br i1 %4, label %27, label %5

5:                                                ; preds = %2, %5
  %6 = phi i64 [ %25, %5 ], [ 0, %2 ]
  %7 = phi i64 [ %24, %5 ], [ 0, %2 ]
  %8 = getelementptr inbounds i8, i8* %0, i64 %6
  %9 = getelementptr inbounds i8, i8* %8, i64 4001
  %10 = load i8, i8* %9, align 1
  %11 = getelementptr inbounds i8, i8* %8, i64 4002
  %12 = bitcast i8* %11 to i64*
  %13 = load i64, i64* %12, align 8
  %14 = getelementptr inbounds i8, i8* %8, i64 4003
  %15 = bitcast i8* %14 to i64*
  %16 = load i64, i64* %15, align 8
  %17 = getelementptr inbounds i8, i8* %8, i64 4007
  %18 = bitcast i8* %17 to i64*
  %19 = load i64, i64* %18, align 8
  %20 = zext i8 %10 to i64
  %21 = mul i64 %13, %20
  %22 = mul i64 %21, %16
  %23 = mul i64 %22, %19
  %24 = add i64 %23, %7
  %25 = add nuw i64 %6, 1
  %26 = icmp ult i64 %25, %3
  br i1 %26, label %5, label %27

27:                                               ; preds = %5, %2
  %28 = phi i64 [ 0, %2 ], [ %24, %5 ]
  %29 = add i64 %28, %3
  ret i64 %29
}

; test_ds_multiple_chains:
; unsigned long test_ds_multiple_chains(char *p, char *q, int count) {
;   unsigned long i=0, res=0;
;   int DISP1 = 4001;
;   int DISP2 = 4010;
;   int DISP3 = 4005;
;   int DISP4 = 4009;
;   for (; i < count ; i++) {
;     unsigned long x1 = *(unsigned long *)(p + i + DISP1);
;     unsigned long x2 = *(unsigned long *)(p + i + DISP2);
;     unsigned long x3 = *(unsigned long *)(p + i + DISP3);
;     unsigned long x4 = *(unsigned long *)(p + i + DISP4);
;     unsigned long x5 = *(unsigned long *)(q + i + DISP1);
;     unsigned long x6 = *(unsigned long *)(q + i + DISP2);
;     unsigned long x7 = *(unsigned long *)(q + i + DISP3);
;     unsigned long x8 = *(unsigned long *)(q + i + DISP4);
;     res += x1*x2*x3*x4*x5*x6*x7*x8;
;   }
;   return res + count;
; }

define dso_local i64 @test_ds_multiple_chains(i8* %0, i8* %1, i32 signext %2) {
; CHECK-LABEL: test_ds_multiple_chains:
; CHECK:         addi r3, r3, 4001
; CHECK:         addi r4, r4, 4001
; CHECK:       .LBB5_2: #
; CHECK-NEXT:    ld r8, 0(r3)
; CHECK-NEXT:    ldx r9, r3, r7
; CHECK-NEXT:    mulld r8, r9, r8
; CHECK-NEXT:    ld r9, 4(r3)
; CHECK-NEXT:    mulld r8, r8, r9
; CHECK-NEXT:    ld r10, 8(r3)
; CHECK-NEXT:    addi r3, r3, 1
; CHECK-NEXT:    mulld r8, r8, r10
; CHECK-NEXT:    ld r11, 0(r4)
; CHECK-NEXT:    mulld r8, r8, r11
; CHECK-NEXT:    ldx r12, r4, r7
; CHECK-NEXT:    mulld r8, r8, r12
; CHECK-NEXT:    ld r0, 4(r4)
; CHECK-NEXT:    mulld r8, r8, r0
; CHECK-NEXT:    ld r30, 8(r4)
; CHECK-NEXT:    addi r4, r4, 1
; CHECK-NEXT:    maddld r6, r8, r30, r6
; CHECK-NEXT:    bdnz .LBB5_2
  %4 = sext i32 %2 to i64
  %5 = icmp eq i32 %2, 0
  br i1 %5, label %45, label %6

6:                                                ; preds = %3, %6
  %7 = phi i64 [ %43, %6 ], [ 0, %3 ]
  %8 = phi i64 [ %42, %6 ], [ 0, %3 ]
  %9 = getelementptr inbounds i8, i8* %0, i64 %7
  %10 = getelementptr inbounds i8, i8* %9, i64 4001
  %11 = bitcast i8* %10 to i64*
  %12 = load i64, i64* %11, align 8
  %13 = getelementptr inbounds i8, i8* %9, i64 4010
  %14 = bitcast i8* %13 to i64*
  %15 = load i64, i64* %14, align 8
  %16 = getelementptr inbounds i8, i8* %9, i64 4005
  %17 = bitcast i8* %16 to i64*
  %18 = load i64, i64* %17, align 8
  %19 = getelementptr inbounds i8, i8* %9, i64 4009
  %20 = bitcast i8* %19 to i64*
  %21 = load i64, i64* %20, align 8
  %22 = getelementptr inbounds i8, i8* %1, i64 %7
  %23 = getelementptr inbounds i8, i8* %22, i64 4001
  %24 = bitcast i8* %23 to i64*
  %25 = load i64, i64* %24, align 8
  %26 = getelementptr inbounds i8, i8* %22, i64 4010
  %27 = bitcast i8* %26 to i64*
  %28 = load i64, i64* %27, align 8
  %29 = getelementptr inbounds i8, i8* %22, i64 4005
  %30 = bitcast i8* %29 to i64*
  %31 = load i64, i64* %30, align 8
  %32 = getelementptr inbounds i8, i8* %22, i64 4009
  %33 = bitcast i8* %32 to i64*
  %34 = load i64, i64* %33, align 8
  %35 = mul i64 %15, %12
  %36 = mul i64 %35, %18
  %37 = mul i64 %36, %21
  %38 = mul i64 %37, %25
  %39 = mul i64 %38, %28
  %40 = mul i64 %39, %31
  %41 = mul i64 %40, %34
  %42 = add i64 %41, %8
  %43 = add nuw i64 %7, 1
  %44 = icmp ult i64 %43, %4
  br i1 %44, label %6, label %45

45:                                               ; preds = %6, %3
  %46 = phi i64 [ 0, %3 ], [ %42, %6 ]
  %47 = add i64 %46, %4
  ret i64 %47
}

; test_ds_cross_basic_blocks:
;extern char *arr;
;unsigned long foo(char *p, int count)
;{
;  unsigned long i=0, res=0;
;  int DISP1 = 4000;
;  int DISP2 = 4001;
;  int DISP3 = 4002;
;  int DISP4 = 4003;
;  int DISP5 = 4005;
;  int DISP6 = 4009;
;  unsigned long x1, x2, x3, x4, x5, x6;
;  x1=x2=x3=x4=x5=x6=1;
;  for (; i < count ; i++) {
;    if (arr[i] % 3 == 1) {
;      x1 += *(unsigned long *)(p + i + DISP1);
;      x2 += *(unsigned long *)(p + i + DISP2);
;    }
;    else if (arr[i] % 3 == 2) {
;      x3 += *(unsigned long *)(p + i + DISP3);
;      x4 += *(unsigned long *)(p + i + DISP5);
;    }
;    else {
;      x5 += *(unsigned long *)(p + i + DISP4);
;      x6 += *(unsigned long *)(p + i + DISP6);
;    }
;    res += x1*x2*x3*x4*x5*x6;
;  }
;  return res;
;}

@arr = external local_unnamed_addr global i8*, align 8

define i64 @test_ds_cross_basic_blocks(i8* %0, i32 signext %1) {
; CHECK-LABEL: test_ds_cross_basic_blocks:
; CHECK:         addi r6, r3, 4009
; CHECK:       .LBB6_2: #
; CHECK-NEXT:    ldx r0, r6, r8
; CHECK-NEXT:    add r28, r0, r28
; CHECK-NEXT:    ld r0, -8(r6)
; CHECK-NEXT:    add r29, r0, r29
; CHECK-NEXT:  .LBB6_3: #
; CHECK-NEXT:    mulld r0, r29, r28
; CHECK-NEXT:    mulld r0, r0, r30
; CHECK-NEXT:    mulld r0, r0, r12
; CHECK-NEXT:    mulld r0, r0, r11
; CHECK-NEXT:    maddld r3, r0, r7, r3
; CHECK-NEXT:    addi r6, r6, 1
; CHECK-NEXT:    bdz .LBB6_9
; CHECK-NEXT:  .LBB6_4: #
; CHECK-NEXT:    lbzu r0, 1(r5)
; CHECK-NEXT:    clrldi r27, r0, 32
; CHECK-NEXT:    mulld r27, r27, r4
; CHECK-NEXT:    rldicl r27, r27, 31, 33
; CHECK-NEXT:    slwi r26, r27, 1
; CHECK-NEXT:    add r27, r27, r26
; CHECK-NEXT:    subf r0, r27, r0
; CHECK-NEXT:    cmplwi r0, 1
; CHECK-NEXT:    beq cr0, .LBB6_2
; CHECK-NEXT:  # %bb.5: #
; CHECK-NEXT:    clrlwi r0, r0, 24
; CHECK-NEXT:    cmplwi r0, 2
; CHECK-NEXT:    bne cr0, .LBB6_7
; CHECK-NEXT:  # %bb.6: #
; CHECK-NEXT:    ldx r0, r6, r9
; CHECK-NEXT:    add r30, r0, r30
; CHECK-NEXT:    ld r0, -4(r6)
; CHECK-NEXT:    add r12, r0, r12
; CHECK-NEXT:    b .LBB6_3
; CHECK-NEXT:    .p2align 4
; CHECK-NEXT:  .LBB6_7: #
; CHECK-NEXT:    ldx r0, r6, r10
; CHECK-NEXT:    add r11, r0, r11
; CHECK-NEXT:    ld r0, 0(r6)
; CHECK-NEXT:    add r7, r0, r7
  %3 = sext i32 %1 to i64
  %4 = icmp eq i32 %1, 0
  br i1 %4, label %66, label %5

5:                                                ; preds = %2
  %6 = load i8*, i8** @arr, align 8
  br label %7

7:                                                ; preds = %5, %51
  %8 = phi i64 [ 1, %5 ], [ %57, %51 ]
  %9 = phi i64 [ 1, %5 ], [ %56, %51 ]
  %10 = phi i64 [ 1, %5 ], [ %55, %51 ]
  %11 = phi i64 [ 1, %5 ], [ %54, %51 ]
  %12 = phi i64 [ 1, %5 ], [ %53, %51 ]
  %13 = phi i64 [ 1, %5 ], [ %52, %51 ]
  %14 = phi i64 [ 0, %5 ], [ %64, %51 ]
  %15 = phi i64 [ 0, %5 ], [ %63, %51 ]
  %16 = getelementptr inbounds i8, i8* %6, i64 %14
  %17 = load i8, i8* %16, align 1
  %18 = urem i8 %17, 3
  %19 = icmp eq i8 %18, 1
  br i1 %19, label %20, label %30

20:                                               ; preds = %7
  %21 = getelementptr inbounds i8, i8* %0, i64 %14
  %22 = getelementptr inbounds i8, i8* %21, i64 4000
  %23 = bitcast i8* %22 to i64*
  %24 = load i64, i64* %23, align 8
  %25 = add i64 %24, %13
  %26 = getelementptr inbounds i8, i8* %21, i64 4001
  %27 = bitcast i8* %26 to i64*
  %28 = load i64, i64* %27, align 8
  %29 = add i64 %28, %12
  br label %51

30:                                               ; preds = %7
  %31 = icmp eq i8 %18, 2
  %32 = getelementptr inbounds i8, i8* %0, i64 %14
  br i1 %31, label %33, label %42

33:                                               ; preds = %30
  %34 = getelementptr inbounds i8, i8* %32, i64 4002
  %35 = bitcast i8* %34 to i64*
  %36 = load i64, i64* %35, align 8
  %37 = add i64 %36, %11
  %38 = getelementptr inbounds i8, i8* %32, i64 4005
  %39 = bitcast i8* %38 to i64*
  %40 = load i64, i64* %39, align 8
  %41 = add i64 %40, %10
  br label %51

42:                                               ; preds = %30
  %43 = getelementptr inbounds i8, i8* %32, i64 4003
  %44 = bitcast i8* %43 to i64*
  %45 = load i64, i64* %44, align 8
  %46 = add i64 %45, %9
  %47 = getelementptr inbounds i8, i8* %32, i64 4009
  %48 = bitcast i8* %47 to i64*
  %49 = load i64, i64* %48, align 8
  %50 = add i64 %49, %8
  br label %51

51:                                               ; preds = %33, %42, %20
  %52 = phi i64 [ %25, %20 ], [ %13, %33 ], [ %13, %42 ]
  %53 = phi i64 [ %29, %20 ], [ %12, %33 ], [ %12, %42 ]
  %54 = phi i64 [ %11, %20 ], [ %37, %33 ], [ %11, %42 ]
  %55 = phi i64 [ %10, %20 ], [ %41, %33 ], [ %10, %42 ]
  %56 = phi i64 [ %9, %20 ], [ %9, %33 ], [ %46, %42 ]
  %57 = phi i64 [ %8, %20 ], [ %8, %33 ], [ %50, %42 ]
  %58 = mul i64 %53, %52
  %59 = mul i64 %58, %54
  %60 = mul i64 %59, %55
  %61 = mul i64 %60, %56
  %62 = mul i64 %61, %57
  %63 = add i64 %62, %15
  %64 = add nuw i64 %14, 1
  %65 = icmp ult i64 %64, %3
  br i1 %65, label %7, label %66

66:                                               ; preds = %51, %2
  %67 = phi i64 [ 0, %2 ], [ %63, %51 ]
  ret i64 %67
}

; test_ds_float:
;float test_ds_float(char *p, int count) {
;  int i=0 ;
;  float res=0;
;  int DISP1 = 4001;
;  int DISP2 = 4002;
;  int DISP3 = 4022;
;  int DISP4 = 4062;
;  for (; i < count ; i++) {
;    float x1 = *(float *)(p + i + DISP1);
;    float x2 = *(float *)(p + i + DISP2);
;    float x3 = *(float *)(p + i + DISP3);
;    float x4 = *(float *)(p + i + DISP4);
;    res += x1*x2*x3*x4;
;  }
;  return res;
;}

define float @test_ds_float(i8* %0, i32 signext %1) {
; CHECK-LABEL: test_ds_float:
; CHECK:         addi r3, r3, 4002
; CHECK:       .LBB7_2: #
; CHECK-NEXT:    lfsx f0, r3, r4
; CHECK-NEXT:    lfs f2, 0(r3)
; CHECK-NEXT:    xsmulsp f0, f0, f2
; CHECK-NEXT:    lfs f3, 20(r3)
; CHECK-NEXT:    xsmulsp f0, f0, f3
; CHECK-NEXT:    lfs f4, 60(r3)
; CHECK-NEXT:    addi r3, r3, 1
; CHECK-NEXT:    xsmulsp f0, f0, f4
; CHECK-NEXT:    xsaddsp f1, f1, f0
; CHECK-NEXT:    bdnz .LBB7_2
  %3 = icmp sgt i32 %1, 0
  br i1 %3, label %4, label %28

4:                                                ; preds = %2
  %5 = zext i32 %1 to i64
  br label %6

6:                                                ; preds = %6, %4
  %7 = phi i64 [ 0, %4 ], [ %26, %6 ]
  %8 = phi float [ 0.000000e+00, %4 ], [ %25, %6 ]
  %9 = getelementptr inbounds i8, i8* %0, i64 %7
  %10 = getelementptr inbounds i8, i8* %9, i64 4001
  %11 = bitcast i8* %10 to float*
  %12 = load float, float* %11, align 4
  %13 = getelementptr inbounds i8, i8* %9, i64 4002
  %14 = bitcast i8* %13 to float*
  %15 = load float, float* %14, align 4
  %16 = getelementptr inbounds i8, i8* %9, i64 4022
  %17 = bitcast i8* %16 to float*
  %18 = load float, float* %17, align 4
  %19 = getelementptr inbounds i8, i8* %9, i64 4062
  %20 = bitcast i8* %19 to float*
  %21 = load float, float* %20, align 4
  %22 = fmul float %12, %15
  %23 = fmul float %22, %18
  %24 = fmul float %23, %21
  %25 = fadd float %8, %24
  %26 = add nuw nsw i64 %7, 1
  %27 = icmp eq i64 %26, %5
  br i1 %27, label %28, label %6

28:                                               ; preds = %6, %2
  %29 = phi float [ 0.000000e+00, %2 ], [ %25, %6 ]
  ret float %29
}

; test_ds_combine_float_int:
;float test_ds_combine_float_int(char *p, int count) {
;  int i=0 ;
;  float res=0;
;  int DISP1 = 4001;
;  int DISP2 = 4002;
;  int DISP3 = 4022;
;  int DISP4 = 4062;
;  for (; i < count ; i++) {
;    float x1 = *(float *)(p + i + DISP1);
;    unsigned long x2 = *(unsigned long*)(p + i + DISP2);
;    float x3 = *(float *)(p + i + DISP3);
;    float x4 = *(float *)(p + i + DISP4);
;    res += x1*x2*x3*x4;
;  }
;  return res;
;}

define float @test_ds_combine_float_int(i8* %0, i32 signext %1) {
; CHECK-LABEL: test_ds_combine_float_int:
; CHECK:         addi r3, r3, 4002
; CHECK:       .LBB8_2: #
; CHECK-NEXT:    lfd f4, 0(r3)
; CHECK-NEXT:    lfsx f0, r3, r4
; CHECK-NEXT:    xscvuxdsp f4, f4
; CHECK-NEXT:    lfs f2, 20(r3)
; CHECK-NEXT:    xsmulsp f0, f0, f4
; CHECK-NEXT:    xsmulsp f0, f2, f0
; CHECK-NEXT:    lfs f3, 60(r3)
; CHECK-NEXT:    addi r3, r3, 1
; CHECK-NEXT:    xsmulsp f0, f3, f0
; CHECK-NEXT:    xsaddsp f1, f1, f0
; CHECK-NEXT:    bdnz .LBB8_2
  %3 = icmp sgt i32 %1, 0
  br i1 %3, label %4, label %29

4:                                                ; preds = %2
  %5 = zext i32 %1 to i64
  br label %6

6:                                                ; preds = %6, %4
  %7 = phi i64 [ 0, %4 ], [ %27, %6 ]
  %8 = phi float [ 0.000000e+00, %4 ], [ %26, %6 ]
  %9 = getelementptr inbounds i8, i8* %0, i64 %7
  %10 = getelementptr inbounds i8, i8* %9, i64 4001
  %11 = bitcast i8* %10 to float*
  %12 = load float, float* %11, align 4
  %13 = getelementptr inbounds i8, i8* %9, i64 4002
  %14 = bitcast i8* %13 to i64*
  %15 = load i64, i64* %14, align 8
  %16 = getelementptr inbounds i8, i8* %9, i64 4022
  %17 = bitcast i8* %16 to float*
  %18 = load float, float* %17, align 4
  %19 = getelementptr inbounds i8, i8* %9, i64 4062
  %20 = bitcast i8* %19 to float*
  %21 = load float, float* %20, align 4
  %22 = uitofp i64 %15 to float
  %23 = fmul float %12, %22
  %24 = fmul float %18, %23
  %25 = fmul float %21, %24
  %26 = fadd float %8, %25
  %27 = add nuw nsw i64 %7, 1
  %28 = icmp eq i64 %27, %5
  br i1 %28, label %29, label %6

29:                                               ; preds = %6, %2
  %30 = phi float [ 0.000000e+00, %2 ], [ %26, %6 ]
  ret float %30
}

; test_ds_lwa_prep:
; long long test_ds_lwa_prep(char *p, int count) {
;   long long i=0, res=0;
;   int DISP1 = 4001;
;   int DISP2 = 4002;
;   int DISP3 = 4006;
;   int DISP4 = 4010;
;   for (; i < count ; i++) {
;     long long x1 = *(int *)(p + i + DISP1);
;     long long x2 = *(int *)(p + i + DISP2);
;     long long x3 = *(int *)(p + i + DISP3);
;     long long x4 = *(int *)(p + i + DISP4);
;     res += x1*x2*x3*x4;
;   }
;   return res + count;
; }

define i64 @test_ds_lwa_prep(i8* %0, i32 signext %1) {
; CHECK-LABEL: test_ds_lwa_prep:
; CHECK:         addi r5, r3, 2
; CHECK:         li r6, -1
; CHECK:       .LBB9_2: #
; CHECK-NEXT:    lwax r7, r5, r6
; CHECK-NEXT:    lwa r8, 0(r5)
; CHECK-NEXT:    lwa r9, 4(r5)
; CHECK-NEXT:    lwa r10, 8(r5)
; CHECK-NEXT:    addi r5, r5, 1
; CHECK-NEXT:    mulld r7, r8, r7
; CHECK-NEXT:    mulld r7, r7, r9
; CHECK-NEXT:    maddld r3, r7, r10, r3
; CHECK-NEXT:    bdnz .LBB9_2

  %3 = sext i32 %1 to i64
  %4 = icmp sgt i32 %1, 0
  br i1 %4, label %5, label %31

5:                                                ; preds = %2, %5
  %6 = phi i64 [ %29, %5 ], [ 0, %2 ]
  %7 = phi i64 [ %28, %5 ], [ 0, %2 ]
  %8 = getelementptr inbounds i8, i8* %0, i64 %6
  %9 = getelementptr inbounds i8, i8* %8, i64 1
  %10 = bitcast i8* %9 to i32*
  %11 = load i32, i32* %10, align 4
  %12 = sext i32 %11 to i64
  %13 = getelementptr inbounds i8, i8* %8, i64 2
  %14 = bitcast i8* %13 to i32*
  %15 = load i32, i32* %14, align 4
  %16 = sext i32 %15 to i64
  %17 = getelementptr inbounds i8, i8* %8, i64 6
  %18 = bitcast i8* %17 to i32*
  %19 = load i32, i32* %18, align 4
  %20 = sext i32 %19 to i64
  %21 = getelementptr inbounds i8, i8* %8, i64 10
  %22 = bitcast i8* %21 to i32*
  %23 = load i32, i32* %22, align 4
  %24 = sext i32 %23 to i64
  %25 = mul nsw i64 %16, %12
  %26 = mul nsw i64 %25, %20
  %27 = mul nsw i64 %26, %24
  %28 = add nsw i64 %27, %7
  %29 = add nuw nsw i64 %6, 1
  %30 = icmp eq i64 %29, %3
  br i1 %30, label %31, label %5

31:                                               ; preds = %5, %2
  %32 = phi i64 [ 0, %2 ], [ %28, %5 ]
  %33 = add nsw i64 %32, %3
  ret i64 %33
}

