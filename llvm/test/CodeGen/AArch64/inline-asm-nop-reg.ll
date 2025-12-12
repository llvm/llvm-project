; RUN: llc -O1 -mtriple=aarch64-linux-gnu %s -o - 2>&1 | FileCheck %s

; This test contains "nop" inline assembler instruction with 16..128-bit
; affectd FP "v0" register. It checks that the IR will be successfully
; compiled and generated code will contain nop-instruction and proper register
; name placed in each function.
;
; IR for this test was generated from the following source code:
;
; #define _FP16 _Float16
; #define _FP32 float
; #define _FP64 double
; #define _FP128 long double
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
; FOO(128)


; test nop_fp16_reg
; CHECK-LABEL: foo16:
; CHECK: nop
; CHECK: str h0
define dso_local i32 @foo16() #0 {
entry:
  %a0 = alloca half, align 2
  %i = alloca i32, align 4
  store i32 0, ptr %i, align 4
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %0 = load i32, ptr %i, align 4
  %cmp = icmp slt i32 %0, 2
  br i1 %cmp, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  %1 = load half, ptr %a0, align 2
  %2 = call half asm sideeffect "nop", "={v0},{v0}"(half %1) #1, !srcloc !6
  store half %2, ptr %a0, align 2
  br label %for.inc

for.inc:                                          ; preds = %for.body
  %3 = load i32, ptr %i, align 4
  %inc = add nsw i32 %3, 1
  store i32 %inc, ptr %i, align 4
  br label %for.cond, !llvm.loop !7

for.end:                                          ; preds = %for.cond
  ret i32 0
}

; test nop_fp32_reg
; CHECK-LABEL: foo32:
; CHECK: nop
; CHECK: str s0
define dso_local i32 @foo32() #0 {
entry:
  %a0 = alloca float, align 4
  %i = alloca i32, align 4
  store i32 0, ptr %i, align 4
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %0 = load i32, ptr %i, align 4
  %cmp = icmp slt i32 %0, 2
  br i1 %cmp, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  %1 = load float, ptr %a0, align 4
  %2 = call float asm sideeffect "nop", "={v0},{v0}"(float %1) #1, !srcloc !9
  store float %2, ptr %a0, align 4
  br label %for.inc

for.inc:                                          ; preds = %for.body
  %3 = load i32, ptr %i, align 4
  %inc = add nsw i32 %3, 1
  store i32 %inc, ptr %i, align 4
  br label %for.cond, !llvm.loop !10

for.end:                                          ; preds = %for.cond
  ret i32 0
}

; test nop_fp64_reg
; CHECK-LABEL: foo64:
; CHECK: nop
; CHECK: str d0
define dso_local i32 @foo64() #0 {
entry:
  %a0 = alloca double, align 8
  %i = alloca i32, align 4
  store i32 0, ptr %i, align 4
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %0 = load i32, ptr %i, align 4
  %cmp = icmp slt i32 %0, 2
  br i1 %cmp, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  %1 = load double, ptr %a0, align 8
  %2 = call double asm sideeffect "nop", "={v0},{v0}"(double %1) #1, !srcloc !11
  store double %2, ptr %a0, align 8
  br label %for.inc

for.inc:                                          ; preds = %for.body
  %3 = load i32, ptr %i, align 4
  %inc = add nsw i32 %3, 1
  store i32 %inc, ptr %i, align 4
  br label %for.cond, !llvm.loop !12

for.end:                                          ; preds = %for.cond
  ret i32 0
}

; test nop_fp128_reg
; CHECK-LABEL: foo128:
; CHECK: nop
; CHECK: str q0
define dso_local i32 @foo128() #0 {
entry:
  %a0 = alloca fp128, align 16
  %i = alloca i32, align 4
  store i32 0, ptr %i, align 4
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %0 = load i32, ptr %i, align 4
  %cmp = icmp slt i32 %0, 2
  br i1 %cmp, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  %1 = load fp128, ptr %a0, align 16
  %2 = call fp128 asm sideeffect "nop", "={v0},{v0}"(fp128 %1) #1, !srcloc !13
  store fp128 %2, ptr %a0, align 16
  br label %for.inc

for.inc:                                          ; preds = %for.body
  %3 = load i32, ptr %i, align 4
  %inc = add nsw i32 %3, 1
  store i32 %inc, ptr %i, align 4
  br label %for.cond, !llvm.loop !14

for.end:                                          ; preds = %for.cond
  ret i32 0
}

!llvm.module.flags = !{!0, !1, !2, !3, !4}
!llvm.ident = !{!5}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 8, !"PIC Level", i32 2}
!2 = !{i32 7, !"PIE Level", i32 2}
!3 = !{i32 7, !"uwtable", i32 2}
!4 = !{i32 7, !"frame-pointer", i32 4}
!5 = !{!"clang version 22.0.0git (https://github.com/llvm/llvm-project.git)"}
!6 = !{i64 2147502487}
!7 = distinct !{!7, !8}
!8 = !{!"llvm.loop.mustprogress"}
!9 = !{i64 2147502682}
!10 = distinct !{!10, !8}
!11 = !{i64 2147502874}
!12 = distinct !{!12, !8}
!13 = !{i64 2147503067}
!14 = distinct !{!14, !8}