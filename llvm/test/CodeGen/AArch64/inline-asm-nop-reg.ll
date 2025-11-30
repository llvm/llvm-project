; RUN: llc -O1 -mtriple=aarch64-linux-gnu %s -o - 2>&1 | FileCheck %s

; This test checks that the code containing "nop" inline assembler instruction
; with 16/32/64-bit FP "v0" register, will be successfully compiled
; and generated code will contain one optimized nop-instruction
; per each function.
;
; IR for this test was generated from the following source code:
;
; #define _FP16 _Float16
; #define _FP32 float
; #define _FP64 double
;
; #define FOO(BITS) \
; int foo##BITS(void) { \
;   register _FP##BITS a0 asm("v0"); \
;   for (int i = 0; i < 2; ++i) { \
;     __asm__ volatile("nop" : [a0] "+w"(a0)::); \
;   } \
;   return 0; \
; }
;
; FOO(16)
; FOO(32)
; FOO(64)


; test nop_fp16_reg
; CHECK-LABEL: foo16:
; CHECK: nop
; CHECK-NOT: nop
define dso_local i32 @foo16() #0 {
  %1 = alloca half, align 2
  %2 = alloca i32, align 4
  store i32 0, ptr %2, align 4
  br label %3

3:                                                ; preds = %9, %0
  %4 = load i32, ptr %2, align 4
  %5 = icmp slt i32 %4, 2
  br i1 %5, label %6, label %12

6:                                                ; preds = %3
  %7 = load half, ptr %1, align 2
  %8 = call half asm sideeffect "nop", "={v0},{v0}"(half %7) #1, !srcloc !6
  store half %8, ptr %1, align 2
  br label %9

9:                                                ; preds = %6
  %10 = load i32, ptr %2, align 4
  %11 = add nsw i32 %10, 1
  store i32 %11, ptr %2, align 4
  br label %3, !llvm.loop !7

12:                                               ; preds = %3
  ret i32 0
}

; test nop_fp32_reg
; CHECK-LABEL: foo32:
; CHECK: nop
; CHECK-NOT: nop
define dso_local i32 @foo32() #0 {
  %1 = alloca float, align 4
  %2 = alloca i32, align 4
  store i32 0, ptr %2, align 4
  br label %3

3:                                                ; preds = %9, %0
  %4 = load i32, ptr %2, align 4
  %5 = icmp slt i32 %4, 2
  br i1 %5, label %6, label %12

6:                                                ; preds = %3
  %7 = load float, ptr %1, align 4
  %8 = call float asm sideeffect "nop", "={v0},{v0}"(float %7) #1, !srcloc !9
  store float %8, ptr %1, align 4
  br label %9

9:                                                ; preds = %6
  %10 = load i32, ptr %2, align 4
  %11 = add nsw i32 %10, 1
  store i32 %11, ptr %2, align 4
  br label %3, !llvm.loop !10

12:                                               ; preds = %3
  ret i32 0
}

; test nop_fp64_reg
; CHECK-LABEL: foo64:
; CHECK: nop
; CHECK-NOT: nop
define dso_local i32 @foo64() #0 {
  %1 = alloca double, align 8
  %2 = alloca i32, align 4
  store i32 0, ptr %2, align 4
  br label %3

3:                                                ; preds = %9, %0
  %4 = load i32, ptr %2, align 4
  %5 = icmp slt i32 %4, 2
  br i1 %5, label %6, label %12

6:                                                ; preds = %3
  %7 = load double, ptr %1, align 8
  %8 = call double asm sideeffect "nop", "={v0},{v0}"(double %7) #1, !srcloc !11
  store double %8, ptr %1, align 8
  br label %9

9:                                                ; preds = %6
  %10 = load i32, ptr %2, align 4
  %11 = add nsw i32 %10, 1
  store i32 %11, ptr %2, align 4
  br label %3, !llvm.loop !12

12:                                               ; preds = %3
  ret i32 0
}

!llvm.module.flags = !{!0, !1, !2, !3, !4}
!llvm.ident = !{!5}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 8, !"PIC Level", i32 2}
!2 = !{i32 7, !"PIE Level", i32 2}
!3 = !{i32 7, !"uwtable", i32 2}
!4 = !{i32 7, !"frame-pointer", i32 1}
!5 = !{!"clang version 22.0.0git"}
!6 = !{i64 2147502427}
!7 = distinct !{!7, !8}
!8 = !{!"llvm.loop.mustprogress"}
!9 = !{i64 2147502622}
!10 = distinct !{!10, !8}
!11 = !{i64 2147502814}
!12 = distinct !{!12, !8}
