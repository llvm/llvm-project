; RUN: llc -O1 -mtriple=aarch64-linux-gnu %s -o - 2>&1 | FileCheck %s

; This test contains "nop" inline assembler instruction with 16..128-bit
; affectd "v0" register. It checks that the IR will be successfully
; compiled and generated code will contain nop-instruction and proper register
; name placed in each function.
;
; IR for this test was generated from the following source code:
;
; #include <stdint.h>
;
; #define _FP16 _Float16
; #define _FP32 float
; #define _FP64 double
; #define _FP128 long double
;
; #define _INT16 int16_t
; #define _INT32 int32_t
; #define _INT64 int64_t
; #define _INT128 __int128
;
; #define FOO(TYPE) \
; void foo_##TYPE(void) { \
;   register _##TYPE a0 asm("v0"); \
;   __asm__ volatile("nop" : [a0] "+w"(a0)::); \
; }
;
; FOO(FP16)
; FOO(FP32)
; FOO(FP64)
; FOO(FP128)
;
; FOO(INT16)
; FOO(INT32)
; FOO(INT64)
; FOO(INT128)

; test nop_FP16_reg
; CHECK-LABEL: foo_FP16:
; CHECK: ldr h0
; CHECK: nop
; CHECK: str h0
define dso_local void @foo_FP16() #0 {
entry:
  %a0 = alloca half, align 2
  %0 = load half, ptr %a0, align 2
  %1 = call half asm sideeffect "nop", "={v0},{v0}"(half %0) #1, !srcloc !6
  store half %1, ptr %a0, align 2
  ret void
}

; test nop_FP32_reg
; CHECK-LABEL: foo_FP32:
; CHECK: ldr s0
; CHECK: nop
; CHECK: str s0
define dso_local void @foo_FP32() #0 {
entry:
  %a0 = alloca float, align 4
  %0 = load float, ptr %a0, align 4
  %1 = call float asm sideeffect "nop", "={v0},{v0}"(float %0) #1, !srcloc !7
  store float %1, ptr %a0, align 4
  ret void
}

; test nop_FP64_reg
; CHECK-LABEL: foo_FP64:
; CHECK: ldr d0
; CHECK: nop
; CHECK: str d0
define dso_local void @foo_FP64() #0 {
entry:
  %a0 = alloca double, align 8
  %0 = load double, ptr %a0, align 8
  %1 = call double asm sideeffect "nop", "={v0},{v0}"(double %0) #1, !srcloc !8
  store double %1, ptr %a0, align 8
  ret void
}

; test nop_FP128_reg
; CHECK-LABEL: foo_FP128:
; CHECK: ldr q0
; CHECK: nop
; CHECK: str q0
define dso_local void @foo_FP128() #0 {
entry:
  %a0 = alloca fp128, align 16
  %0 = load fp128, ptr %a0, align 16
  %1 = call fp128 asm sideeffect "nop", "={v0},{v0}"(fp128 %0) #1, !srcloc !9
  store fp128 %1, ptr %a0, align 16
  ret void
}

; test nop_INT16_reg
; CHECK-LABEL: foo_INT16:
; CHECK: ldr h0
; CHECK: nop
; CHECK: str h0
define dso_local void @foo_INT16() #0 {
entry:
  %a0 = alloca i16, align 2
  %0 = load i16, ptr %a0, align 2
  %1 = call i16 asm sideeffect "nop", "={v0},{v0}"(i16 %0) #1, !srcloc !10
  store i16 %1, ptr %a0, align 2
  ret void
}

; test nop_INT32_reg
; CHECK-LABEL: foo_INT32:
; CHECK: ldr s0
; CHECK: nop
; CHECK: str s0
define dso_local void @foo_INT32() #0 {
entry:
  %a0 = alloca i32, align 4
  %0 = load i32, ptr %a0, align 4
  %1 = call i32 asm sideeffect "nop", "={v0},{v0}"(i32 %0) #1, !srcloc !11
  store i32 %1, ptr %a0, align 4
  ret void
}

; test nop_INT64_reg
; CHECK-LABEL: foo_INT64:
; CHECK: ldr d0
; CHECK: nop
; CHECK: str d0
define dso_local void @foo_INT64() #0 {
entry:
  %a0 = alloca i64, align 8
  %0 = load i64, ptr %a0, align 8
  %1 = call i64 asm sideeffect "nop", "={v0},{v0}"(i64 %0) #1, !srcloc !12
  store i64 %1, ptr %a0, align 8
  ret void
}

; test nop_INT128_reg
; CHECK-LABEL: foo_INT128:
; CHECK: ldr q0
; CHECK: nop
; CHECK: str q0
define dso_local void @foo_INT128() #0 {
entry:
  %a0 = alloca i128, align 16
  %0 = load i128, ptr %a0, align 16
  %1 = call i128 asm sideeffect "nop", "={v0},{v0}"(i128 %0) #1, !srcloc !13
  store i128 %1, ptr %a0, align 16
  ret void
}

!llvm.module.flags = !{!0, !1, !2, !3, !4}
!llvm.ident = !{!5}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 8, !"PIC Level", i32 2}
!2 = !{i32 7, !"PIE Level", i32 2}
!3 = !{i32 7, !"uwtable", i32 2}
!4 = !{i32 7, !"frame-pointer", i32 4}
!5 = !{!"clang version 22.0.0git (https://github.com/llvm/llvm-project.git)"}
!6 = !{i64 2147598584}
!7 = !{i64 2147598730}
!8 = !{i64 2147598873}
!9 = !{i64 2147599017}
!10 = !{i64 2147599170}
!11 = !{i64 2147599319}
!12 = !{i64 2147599468}
!13 = !{i64 2147599617}
