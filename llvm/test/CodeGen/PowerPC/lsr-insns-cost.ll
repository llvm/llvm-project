; RUN: llc -ppc-asm-full-reg-names -verify-machineinstrs < %s \
; RUN:   -mtriple=powerpc64le-unknown-linux-gnu | FileCheck %s -check-prefix=INST
; RUN: llc -ppc-asm-full-reg-names -verify-machineinstrs -ppc-lsr-no-insns-cost=true \
; RUN:   < %s -mtriple=powerpc64le-unknown-linux-gnu | FileCheck %s -check-prefix=REG

; void test(unsigned *a, unsigned *b, unsigned *c)
; {
;   for (unsigned long i = 0; i < 1024; i++)
;       c[i] = a[i] + b[i];
; }
;
; compile with -fno-unroll-loops

define void @lsr-insts-cost(ptr %0, ptr %1, ptr %2) {
; INST-LABEL: lsr-insts-cost
; INST:       .LBB0_4: # =>This Inner Loop Header: Depth=1
; INST-NEXT:    lxvd2x vs34, r3, r6
; INST-NEXT:    lxvd2x vs35, r4, r6
; INST-NEXT:    vadduwm v2, v3, v2
; INST-NEXT:    stxvd2x vs34, r5, r6
; INST-NEXT:    addi r6, r6, 16
; INST-NEXT:    bdnz .LBB0_4
;
; REG-LABEL: lsr-insts-cost
; REG:       .LBB0_4: # =>This Inner Loop Header: Depth=1
; REG-NEXT:    lxvd2x vs34, 0, r3
; REG-NEXT:    lxvd2x vs35, 0, r4
; REG-NEXT:    addi r4, r4, 16
; REG-NEXT:    addi r3, r3, 16
; REG-NEXT:    vadduwm v2, v3, v2
; REG-NEXT:    stxvd2x vs34, 0, r5
; REG-NEXT:    addi r5, r5, 16
; REG-NEXT:    bdnz .LBB0_4
  %4 = getelementptr i32, ptr %2, i64 1024
  %5 = getelementptr i32, ptr %0, i64 1024
  %6 = getelementptr i32, ptr %1, i64 1024
  %7 = icmp ugt ptr %5, %2
  %8 = icmp ugt ptr %4, %0
  %9 = and i1 %7, %8
  %10 = icmp ugt ptr %6, %2
  %11 = icmp ugt ptr %4, %1
  %12 = and i1 %10, %11
  %13 = or i1 %9, %12
  br i1 %13, label %28, label %14

14:                                               ; preds = %3, %14
  %15 = phi i64 [ %25, %14 ], [ 0, %3 ]
  %16 = getelementptr inbounds i32, ptr %0, i64 %15
  %17 = bitcast ptr %16 to ptr
  %18 = load <4 x i32>, ptr %17, align 4
  %19 = getelementptr inbounds i32, ptr %1, i64 %15
  %20 = bitcast ptr %19 to ptr
  %21 = load <4 x i32>, ptr %20, align 4
  %22 = add <4 x i32> %21, %18
  %23 = getelementptr inbounds i32, ptr %2, i64 %15
  %24 = bitcast ptr %23 to ptr
  store <4 x i32> %22, ptr %24, align 4
  %25 = add i64 %15, 4
  %26 = icmp eq i64 %25, 1024
  br i1 %26, label %27, label %14

27:                                               ; preds = %14, %28
  ret void

28:                                               ; preds = %3, %28
  %29 = phi i64 [ %36, %28 ], [ 0, %3 ]
  %30 = getelementptr inbounds i32, ptr %0, i64 %29
  %31 = load i32, ptr %30, align 4
  %32 = getelementptr inbounds i32, ptr %1, i64 %29
  %33 = load i32, ptr %32, align 4
  %34 = add i32 %33, %31
  %35 = getelementptr inbounds i32, ptr %2, i64 %29
  store i32 %34, ptr %35, align 4
  %36 = add nuw nsw i64 %29, 1
  %37 = icmp eq i64 %36, 1024
  br i1 %37, label %27, label %28
}
