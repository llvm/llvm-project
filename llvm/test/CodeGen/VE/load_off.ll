; RUN: llc < %s -mtriple=ve-unknown-unknown | FileCheck %s

@bufi8 = global [3 x i8] zeroinitializer, align 1
@bufi16 = global [3 x i16] zeroinitializer, align 2
@bufi32 = global [3 x i32] zeroinitializer, align 4
@bufi64 = global [3 x i64] zeroinitializer, align 8
@bufi128 = global [3 x i128] zeroinitializer, align 16
@buff32 = global [3 x float] zeroinitializer, align 4
@buff64 = global [3 x double] zeroinitializer, align 8
@buff128 = global [3 x fp128] zeroinitializer, align 16

; Function Attrs: noinline nounwind optnone
define signext i8 @loadi8s() {
; CHECK-LABEL: loadi8s:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    lea %s0, bufi8@lo
; CHECK-NEXT:    and %s0, %s0, (32)0
; CHECK-NEXT:    lea.sl %s0, bufi8@hi(, %s0)
; CHECK-NEXT:    ld1b.sx %s0, 2(, %s0)
; CHECK-NEXT:    or %s11, 0, %s9
entry:
  %0 = load i8, i8* getelementptr inbounds ([3 x i8], [3 x i8]* @bufi8, i64 0, i64 2), align 1
  ret i8 %0
}

; Function Attrs: noinline nounwind optnone
define signext i16 @loadi16s() {
; CHECK-LABEL: loadi16s:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    lea %s0, bufi16@lo
; CHECK-NEXT:    and %s0, %s0, (32)0
; CHECK-NEXT:    lea.sl %s0, bufi16@hi(, %s0)
; CHECK-NEXT:    ld2b.sx %s0, 4(, %s0)
; CHECK-NEXT:    or %s11, 0, %s9
entry:
  %0 = load i16, i16* getelementptr inbounds ([3 x i16], [3 x i16]* @bufi16, i64 0, i64 2), align 2
  ret i16 %0
}

; Function Attrs: noinline nounwind optnone
define signext i32 @loadi32s() {
; CHECK-LABEL: loadi32s:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    lea %s0, bufi32@lo
; CHECK-NEXT:    and %s0, %s0, (32)0
; CHECK-NEXT:    lea.sl %s0, bufi32@hi(, %s0)
; CHECK-NEXT:    ldl.sx %s0, 8(, %s0)
; CHECK-NEXT:    or %s11, 0, %s9
entry:
  %0 = load i32, i32* getelementptr inbounds ([3 x i32], [3 x i32]* @bufi32, i64 0, i64 2), align 4
  ret i32 %0
}

; Function Attrs: noinline nounwind optnone
define i64 @loadi64s() {
; CHECK-LABEL: loadi64s:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    lea %s0, bufi64@lo
; CHECK-NEXT:    and %s0, %s0, (32)0
; CHECK-NEXT:    lea.sl %s0, bufi64@hi(, %s0)
; CHECK-NEXT:    ld %s0, 16(, %s0)
; CHECK-NEXT:    or %s11, 0, %s9
entry:
  %0 = load i64, i64* getelementptr inbounds ([3 x i64], [3 x i64]* @bufi64, i64 0, i64 2), align 8
  ret i64 %0
}

; Function Attrs: noinline nounwind optnone
define i128 @loadi128s() {
; CHECK-LABEL: loadi128s:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    lea %s0, bufi128@lo
; CHECK-NEXT:    and %s0, %s0, (32)0
; CHECK-NEXT:    lea.sl %s1, bufi128@hi(, %s0)
; CHECK-NEXT:    ld %s0, 32(, %s1)
; CHECK-NEXT:    ld %s1, 40(, %s1)
; CHECK-NEXT:    or %s11, 0, %s9
entry:
  %0 = load i128, i128* getelementptr inbounds ([3 x i128], [3 x i128]* @bufi128, i64 0, i64 2), align 16
  ret i128 %0
}

; Function Attrs: noinline nounwind optnone
define zeroext i8 @loadi8z() {
; CHECK-LABEL: loadi8z:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    lea %s0, bufi8@lo
; CHECK-NEXT:    and %s0, %s0, (32)0
; CHECK-NEXT:    lea.sl %s0, bufi8@hi(, %s0)
; CHECK-NEXT:    ld1b.zx %s0, 2(, %s0)
; CHECK-NEXT:    or %s11, 0, %s9
entry:
  %0 = load i8, i8* getelementptr inbounds ([3 x i8], [3 x i8]* @bufi8, i64 0, i64 2), align 1
  ret i8 %0
}

; Function Attrs: noinline nounwind optnone
define zeroext i16 @loadi16z() {
; CHECK-LABEL: loadi16z:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    lea %s0, bufi16@lo
; CHECK-NEXT:    and %s0, %s0, (32)0
; CHECK-NEXT:    lea.sl %s0, bufi16@hi(, %s0)
; CHECK-NEXT:    ld2b.zx %s0, 4(, %s0)
; CHECK-NEXT:    or %s11, 0, %s9
entry:
  %0 = load i16, i16* getelementptr inbounds ([3 x i16], [3 x i16]* @bufi16, i64 0, i64 2), align 2
  ret i16 %0
}

; Function Attrs: noinline nounwind optnone
define zeroext i32 @loadi32z() {
; CHECK-LABEL: loadi32z:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    lea %s0, bufi32@lo
; CHECK-NEXT:    and %s0, %s0, (32)0
; CHECK-NEXT:    lea.sl %s0, bufi32@hi(, %s0)
; CHECK-NEXT:    ldl.zx %s0, 8(, %s0)
; CHECK-NEXT:    or %s11, 0, %s9
entry:
  %0 = load i32, i32* getelementptr inbounds ([3 x i32], [3 x i32]* @bufi32, i64 0, i64 2), align 4
  ret i32 %0
}

; Function Attrs: noinline nounwind optnone
define i64 @loadi64z() {
; CHECK-LABEL: loadi64z:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    lea %s0, bufi64@lo
; CHECK-NEXT:    and %s0, %s0, (32)0
; CHECK-NEXT:    lea.sl %s0, bufi64@hi(, %s0)
; CHECK-NEXT:    ld %s0, 16(, %s0)
; CHECK-NEXT:    or %s11, 0, %s9
entry:
  %0 = load i64, i64* getelementptr inbounds ([3 x i64], [3 x i64]* @bufi64, i64 0, i64 2), align 8
  ret i64 %0
}

; Function Attrs: noinline nounwind optnone
define i128 @loadi128z() {
; CHECK-LABEL: loadi128z:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    lea %s0, bufi128@lo
; CHECK-NEXT:    and %s0, %s0, (32)0
; CHECK-NEXT:    lea.sl %s1, bufi128@hi(, %s0)
; CHECK-NEXT:    ld %s0, 32(, %s1)
; CHECK-NEXT:    ld %s1, 40(, %s1)
; CHECK-NEXT:    or %s11, 0, %s9
entry:
  %0 = load i128, i128* getelementptr inbounds ([3 x i128], [3 x i128]* @bufi128, i64 0, i64 2), align 16
  ret i128 %0
}

; Function Attrs: noinline nounwind optnone
define float @loadf32() {
; CHECK-LABEL: loadf32:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    lea %s0, buff32@lo
; CHECK-NEXT:    and %s0, %s0, (32)0
; CHECK-NEXT:    lea.sl %s0, buff32@hi(, %s0)
; CHECK-NEXT:    ldu %s0, 8(, %s0)
; CHECK-NEXT:    or %s11, 0, %s9
entry:
  %0 = load float, float* getelementptr inbounds ([3 x float], [3 x float]* @buff32, i64 0, i64 2), align 4
  ret float %0
}

; Function Attrs: noinline nounwind optnone
define double @loadf64() {
; CHECK-LABEL: loadf64:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    lea %s0, buff64@lo
; CHECK-NEXT:    and %s0, %s0, (32)0
; CHECK-NEXT:    lea.sl %s0, buff64@hi(, %s0)
; CHECK-NEXT:    ld %s0, 16(, %s0)
; CHECK-NEXT:    or %s11, 0, %s9
entry:
  %0 = load double, double* getelementptr inbounds ([3 x double], [3 x double]* @buff64, i64 0, i64 2), align 8
  ret double %0
}

; Function Attrs: noinline nounwind optnone
define fp128 @loadf128() {
; CHECK-LABEL: loadf128:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    lea %s0, buff128@lo
; CHECK-NEXT:    and %s0, %s0, (32)0
; CHECK-NEXT:    lea.sl %s2, buff128@hi(, %s0)
; CHECK-NEXT:    ld %s0, 40(, %s2)
; CHECK-NEXT:    ld %s1, 32(, %s2)
; CHECK-NEXT:    or %s11, 0, %s9
entry:
  %0 = load fp128, fp128* getelementptr inbounds ([3 x fp128], [3 x fp128]* @buff128, i64 0, i64 2), align 16
  ret fp128 %0
}
