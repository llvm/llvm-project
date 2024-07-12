; RUN: llc < %s -mtriple=powerpc64le-unknown-linux-gnu -mcpu=pwr9 -verify-machineinstrs | FileCheck %s
; RUN: llc < %s -mtriple=powerpc64-unknown-linux-gnu -mcpu=pwr9 -verify-machineinstrs | FileCheck %s
; RUN: llc < %s -mtriple=powerpc64le-unknown-linux-gnu -mcpu=pwr8 -verify-machineinstrs | FileCheck %s
; RUN: llc < %s -mtriple=powerpc64-unknown-linux-gnu -mcpu=pwr8 -verify-machineinstrs | FileCheck %s

; Verify pre-inc preparation pass doesn't prepare pre-inc for i64 load/store
; when the stride doesn't conform LDU/STDU DS-form requirement.

@result = local_unnamed_addr global i64 0, align 8

define i64 @test_preinc_i64_ld(ptr nocapture readonly, i64) local_unnamed_addr {
  %3 = icmp eq i64 %1, 0
  br i1 %3, label %4, label %6

; <label>:4:                                      ; preds = %2
  %5 = load i64, ptr @result, align 8
  br label %13

; <label>:6:                                      ; preds = %2
  %7 = getelementptr inbounds i8, ptr %0, i64 -50000
  %8 = getelementptr inbounds i8, ptr %0, i64 -61024
  %9 = getelementptr inbounds i8, ptr %0, i64 -62048
  %10 = getelementptr inbounds i8, ptr %0, i64 -64096
  %11 = load i64, ptr @result, align 8
  br label %15

; <label>:12:                                     ; preds = %15
  store i64 %33, ptr @result, align 8
  br label %13

; <label>:13:                                     ; preds = %12, %4
  %14 = phi i64 [ %5, %4 ], [ %33, %12 ]
  ret i64 %14

; <label>:15:                                     ; preds = %15, %6
  %16 = phi i64 [ %11, %6 ], [ %33, %15 ]
  %17 = phi i64 [ 0, %6 ], [ %34, %15 ]
  %18 = getelementptr inbounds i8, ptr %7, i64 %17
  %19 = bitcast ptr %18 to ptr
  %20 = load i64, ptr %19, align 8
  %21 = getelementptr inbounds i8, ptr %8, i64 %17
  %22 = bitcast ptr %21 to ptr
  %23 = load i64, ptr %22, align 8
  %24 = getelementptr inbounds i8, ptr %9, i64 %17
  %25 = bitcast ptr %24 to ptr
  %26 = load i64, ptr %25, align 8
  %27 = getelementptr inbounds i8, ptr %10, i64 %17
  %28 = bitcast ptr %27 to ptr
  %29 = load i64, ptr %28, align 8
  %30 = mul i64 %23, %20
  %31 = mul i64 %30, %26
  %32 = mul i64 %31, %29
  %33 = mul i64 %32, %16
  %34 = add nuw i64 %17, 1
  %35 = icmp eq i64 %34, %1
  br i1 %35, label %12, label %15
}

; CHECK-LABEL: test_preinc_i64_ld
; CHECK-NOT: addi {{[0-9]+}}, {{[0-9]+}}, -11023
; CHECK-NOT: addi {{[0-9]+}}, {{[0-9]+}}, -12047
; CHECK-NOT: addi {{[0-9]+}}, {{[0-9]+}}, -14095
; CHECK-DAG: ld {{[0-9]+}}, 14096([[REG1:[0-9]+]])
; CHECK-DAG: ld {{[0-9]+}},  3072([[REG1]])
; CHECK-DAG: ld {{[0-9]+}},  2048([[REG1]])
; CHECK-DAG: ld {{[0-9]+}},  0([[REG1]])
; CHECK: blr

define i64 @test_preinc_i64_ldst(ptr nocapture, i64, i64) local_unnamed_addr {
  %4 = icmp eq i64 %1, 0
  br i1 %4, label %5, label %7

; <label>:5:                                      ; preds = %3
  %6 = load i64, ptr @result, align 8
  br label %16

; <label>:7:                                      ; preds = %3
  %8 = add i64 %2, 1
  %9 = getelementptr inbounds i8, ptr %0, i64 -1024
  %10 = add i64 %2, 2
  %11 = getelementptr inbounds i8, ptr %0, i64 -2048
  %12 = getelementptr inbounds i8, ptr %0, i64 -3072
  %13 = getelementptr inbounds i8, ptr %0, i64 -4096
  %14 = load i64, ptr @result, align 8
  br label %18

; <label>:15:                                     ; preds = %18
  store i64 %32, ptr @result, align 8
  br label %16

; <label>:16:                                     ; preds = %15, %5
  %17 = phi i64 [ %6, %5 ], [ %32, %15 ]
  ret i64 %17

; <label>:18:                                     ; preds = %18, %7
  %19 = phi i64 [ %14, %7 ], [ %32, %18 ]
  %20 = phi i64 [ 0, %7 ], [ %33, %18 ]
  %21 = getelementptr inbounds i8, ptr %9, i64 %20
  %22 = bitcast ptr %21 to ptr
  store i64 %8, ptr %22, align 8
  %23 = getelementptr inbounds i8, ptr %11, i64 %20
  %24 = bitcast ptr %23 to ptr
  store i64 %10, ptr %24, align 8
  %25 = getelementptr inbounds i8, ptr %12, i64 %20
  %26 = bitcast ptr %25 to ptr
  %27 = load i64, ptr %26, align 8
  %28 = getelementptr inbounds i8, ptr %13, i64 %20
  %29 = bitcast ptr %28 to ptr
  %30 = load i64, ptr %29, align 8
  %31 = mul i64 %30, %27
  %32 = mul i64 %31, %19
  %33 = add nuw i64 %20, 1
  %34 = icmp eq i64 %33, %1
  br i1 %34, label %15, label %18
}

; CHECK-LABEL: test_preinc_i64_ldst
; CHECK-NOT: addi {{[0-9]+}}, {{[0-9]+}}, -3071
; CHECK-NOT: addi {{[0-9]+}}, {{[0-9]+}}, -2047
; CHECK-NOT: addi {{[0-9]+}}, {{[0-9]+}}, -1023
; CHECK-DAG: ld  {{[0-9]+}}, -1024([[REG2:[0-9]+]])
; CHECK-DAG: ld  {{[0-9]+}}, -2048([[REG2]])
; CHECK-DAG: std {{[0-9]+}},  1024([[REG2]])
; CHECK-DAG: std {{[0-9]+}},  0([[REG2]])
