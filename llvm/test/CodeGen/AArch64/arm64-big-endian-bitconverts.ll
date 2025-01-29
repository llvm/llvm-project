; RUN: llc -mtriple aarch64_be < %s -aarch64-enable-ldst-opt=false -O1 -o - | FileCheck %s
; RUN: llc -mtriple aarch64_be < %s -aarch64-enable-ldst-opt=false -O0 -fast-isel=true -o - | FileCheck %s

; CHECK-LABEL: test_i64_f64:
define void @test_i64_f64(ptr %p, ptr %q) {
; CHECK: ldr
; CHECK: str
    %1 = load double, ptr %p
    %2 = fadd double %1, %1
    %3 = bitcast double %2 to i64
    %4 = add i64 %3, %3
    store i64 %4, ptr %q
    ret void
}

; CHECK-LABEL: test_i64_v1i64:
define void @test_i64_v1i64(ptr %p, ptr %q) {
; CHECK: ldr
; CHECK: str
    %1 = load <1 x i64>, ptr %p
    %2 = add <1 x i64> %1, %1
    %3 = bitcast <1 x i64> %2 to i64
    %4 = add i64 %3, %3
    store i64 %4, ptr %q
    ret void
}

; CHECK-LABEL: test_i64_v2f32:
define void @test_i64_v2f32(ptr %p, ptr %q) {
; CHECK: ld1 { v{{[0-9]+}}.2s }
; CHECK: rev64 v{{[0-9]+}}.2s
; CHECK: str
    %1 = load volatile <2 x float>, ptr %p
    %2 = fadd <2 x float> %1, %1
    %3 = bitcast <2 x float> %2 to i64
    %4 = add i64 %3, %3
    store i64 %4, ptr %q
    ret void
}

; CHECK-LABEL: test_i64_v2i32:
define void @test_i64_v2i32(ptr %p, ptr %q) {
; CHECK: ld1 { v{{[0-9]+}}.2s }
; CHECK: rev64 v{{[0-9]+}}.2s
; CHECK: str
    %1 = load volatile <2 x i32>, ptr %p
    %2 = add <2 x i32> %1, %1
    %3 = bitcast <2 x i32> %2 to i64
    %4 = add i64 %3, %3
    store i64 %4, ptr %q
    ret void
}

; CHECK-LABEL: test_i64_v4f16:
define void @test_i64_v4f16(ptr %p, ptr %q) {
; CHECK: ld1 { v{{[0-9]+}}.4h }
; CHECK-NOT: rev
; CHECK: fadd
; CHECK: rev64 v{{[0-9]+}}.4h
; CHECK: str
    %1 = load <4 x half>, ptr %p
    %2 = fadd <4 x half> %1, %1
    %3 = bitcast <4 x half> %2 to i64
    %4 = add i64 %3, %3
    store i64 %4, ptr %q
    ret void
}

; CHECK-LABEL: test_i64_v4i16:
define void @test_i64_v4i16(ptr %p, ptr %q) {
; CHECK: ld1 { v{{[0-9]+}}.4h }
; CHECK: rev64 v{{[0-9]+}}.4h
; CHECK: str
    %1 = load <4 x i16>, ptr %p
    %2 = add <4 x i16> %1, %1
    %3 = bitcast <4 x i16> %2 to i64
    %4 = add i64 %3, %3
    store i64 %4, ptr %q
    ret void
}

; CHECK-LABEL: test_i64_v8i8:
define void @test_i64_v8i8(ptr %p, ptr %q) {
; CHECK: ld1 { v{{[0-9]+}}.8b }
; CHECK: rev64 v{{[0-9]+}}.8b
; CHECK: str
    %1 = load <8 x i8>, ptr %p
    %2 = add <8 x i8> %1, %1
    %3 = bitcast <8 x i8> %2 to i64
    %4 = add i64 %3, %3
    store i64 %4, ptr %q
    ret void
}

; CHECK-LABEL: test_f64_i64:
define void @test_f64_i64(ptr %p, ptr %q) {
; CHECK: ldr
; CHECK: str
    %1 = load i64, ptr %p
    %2 = add i64 %1, %1
    %3 = bitcast i64 %2 to double
    %4 = fadd double %3, %3
    store double %4, ptr %q
    ret void
}

; CHECK-LABEL: test_f64_v1i64:
define void @test_f64_v1i64(ptr %p, ptr %q) {
; CHECK: ldr
; CHECK: str
    %1 = load <1 x i64>, ptr %p
    %2 = add <1 x i64> %1, %1
    %3 = bitcast <1 x i64> %2 to double
    %4 = fadd double %3, %3
    store double %4, ptr %q
    ret void
}

; CHECK-LABEL: test_f64_v2f32:
define void @test_f64_v2f32(ptr %p, ptr %q) {
; CHECK: ld1 { v{{[0-9]+}}.2s }
; CHECK: rev64 v{{[0-9]+}}.2s
; CHECK: str
    %1 = load volatile <2 x float>, ptr %p
    %2 = fadd <2 x float> %1, %1
    %3 = bitcast <2 x float> %2 to double
    %4 = fadd double %3, %3
    store double %4, ptr %q
    ret void
}

; CHECK-LABEL: test_f64_v2i32:
define void @test_f64_v2i32(ptr %p, ptr %q) {
; CHECK: ld1 { v{{[0-9]+}}.2s }
; CHECK: rev64 v{{[0-9]+}}.2s
; CHECK: str
    %1 = load volatile <2 x i32>, ptr %p
    %2 = add <2 x i32> %1, %1
    %3 = bitcast <2 x i32> %2 to double
    %4 = fadd double %3, %3
    store double %4, ptr %q
    ret void
}

; CHECK-LABEL: test_f64_v4i16:
define void @test_f64_v4i16(ptr %p, ptr %q) {
; CHECK: ld1 { v{{[0-9]+}}.4h }
; CHECK: rev64 v{{[0-9]+}}.4h
; CHECK: str
    %1 = load <4 x i16>, ptr %p
    %2 = add <4 x i16> %1, %1
    %3 = bitcast <4 x i16> %2 to double
    %4 = fadd double %3, %3
    store double %4, ptr %q
    ret void
}

; CHECK-LABEL: test_f64_v4f16:
define void @test_f64_v4f16(ptr %p, ptr %q) {
; CHECK: ld1 { v{{[0-9]+}}.4h }
; CHECK-NOT: rev
; CHECK: fadd
; CHECK: rev64 v{{[0-9]+}}.4h
; CHECK: fadd
; CHECK: str
    %1 = load <4 x half>, ptr %p
    %2 = fadd <4 x half> %1, %1
    %3 = bitcast <4 x half> %2 to double
    %4 = fadd double %3, %3
    store double %4, ptr %q
    ret void
}

; CHECK-LABEL: test_f64_v8i8:
define void @test_f64_v8i8(ptr %p, ptr %q) {
; CHECK: ld1 { v{{[0-9]+}}.8b }
; CHECK: rev64 v{{[0-9]+}}.8b
; CHECK: str
    %1 = load <8 x i8>, ptr %p
    %2 = add <8 x i8> %1, %1
    %3 = bitcast <8 x i8> %2 to double
    %4 = fadd double %3, %3
    store double %4, ptr %q
    ret void
}

; CHECK-LABEL: test_v1i64_i64:
define void @test_v1i64_i64(ptr %p, ptr %q) {
; CHECK: ldr
; CHECK: str
    %1 = load i64, ptr %p
    %2 = add i64 %1, %1
    %3 = bitcast i64 %2 to <1 x i64>
    %4 = add <1 x i64> %3, %3
    store <1 x i64> %4, ptr %q
    ret void
}

; CHECK-LABEL: test_v1i64_f64:
define void @test_v1i64_f64(ptr %p, ptr %q) {
; CHECK: ldr
; CHECK: str
    %1 = load double, ptr %p
    %2 = fadd double %1, %1
    %3 = bitcast double %2 to <1 x i64>
    %4 = add <1 x i64> %3, %3
    store <1 x i64> %4, ptr %q
    ret void
}

; CHECK-LABEL: test_v1i64_v2f32:
define void @test_v1i64_v2f32(ptr %p, ptr %q) {
; CHECK: ld1 { v{{[0-9]+}}.2s }
; CHECK: rev64 v{{[0-9]+}}.2s
; CHECK: str
    %1 = load volatile <2 x float>, ptr %p
    %2 = fadd <2 x float> %1, %1
    %3 = bitcast <2 x float> %2 to <1 x i64>
    %4 = add <1 x i64> %3, %3
    store <1 x i64> %4, ptr %q
    ret void
}

; CHECK-LABEL: test_v1i64_v2i32:
define void @test_v1i64_v2i32(ptr %p, ptr %q) {
; CHECK: ld1 { v{{[0-9]+}}.2s }
; CHECK: rev64 v{{[0-9]+}}.2s
; CHECK: str
    %1 = load volatile <2 x i32>, ptr %p
    %2 = add <2 x i32> %1, %1
    %3 = bitcast <2 x i32> %2 to <1 x i64>
    %4 = add <1 x i64> %3, %3
    store <1 x i64> %4, ptr %q
    ret void
}

; CHECK-LABEL: test_v1i64_v4f16:
define void @test_v1i64_v4f16(ptr %p, ptr %q) {
; CHECK: ld1 { v{{[0-9]+}}.4h }
; CHECK-NOT: rev
; CHECK: fadd
; CHECK: rev64 v{{[0-9]+}}.4h
; CHECK: str
    %1 = load <4 x half>, ptr %p
    %2 = fadd <4 x half> %1, %1
    %3 = bitcast <4 x half> %2 to <1 x i64>
    %4 = add <1 x i64> %3, %3
    store <1 x i64> %4, ptr %q
    ret void
}

; CHECK-LABEL: test_v1i64_v4i16:
define void @test_v1i64_v4i16(ptr %p, ptr %q) {
; CHECK: ld1 { v{{[0-9]+}}.4h }
; CHECK: rev64 v{{[0-9]+}}.4h
; CHECK: str
    %1 = load <4 x i16>, ptr %p
    %2 = add <4 x i16> %1, %1
    %3 = bitcast <4 x i16> %2 to <1 x i64>
    %4 = add <1 x i64> %3, %3
    store <1 x i64> %4, ptr %q
    ret void
}

; CHECK-LABEL: test_v1i64_v8i8:
define void @test_v1i64_v8i8(ptr %p, ptr %q) {
; CHECK: ld1 { v{{[0-9]+}}.8b }
; CHECK: rev64 v{{[0-9]+}}.8b
; CHECK: str
    %1 = load <8 x i8>, ptr %p
    %2 = add <8 x i8> %1, %1
    %3 = bitcast <8 x i8> %2 to <1 x i64>
    %4 = add <1 x i64> %3, %3
    store <1 x i64> %4, ptr %q
    ret void
}

; CHECK-LABEL: test_v2f32_i64:
define void @test_v2f32_i64(ptr %p, ptr %q) {
; CHECK: ldr
; CHECK: rev64 v{{[0-9]+}}.2s
; CHECK: st1 { v{{[0-9]+}}.2s }
    %1 = load i64, ptr %p
    %2 = add i64 %1, %1
    %3 = bitcast i64 %2 to <2 x float>
    %4 = fadd <2 x float> %3, %3
    store <2 x float> %4, ptr %q
    ret void
}

; CHECK-LABEL: test_v2f32_f64:
define void @test_v2f32_f64(ptr %p, ptr %q) {
; CHECK: ldr
; CHECK: rev64 v{{[0-9]+}}.2s
; CHECK: st1 { v{{[0-9]+}}.2s }
    %1 = load double, ptr %p
    %2 = fadd double %1, %1
    %3 = bitcast double %2 to <2 x float>
    %4 = fadd <2 x float> %3, %3
    store <2 x float> %4, ptr %q
    ret void
}

; CHECK-LABEL: test_v2f32_v1i64:
define void @test_v2f32_v1i64(ptr %p, ptr %q) {
; CHECK: ldr
; CHECK: rev64 v{{[0-9]+}}.2s
; CHECK: st1 { v{{[0-9]+}}.2s }
    %1 = load <1 x i64>, ptr %p
    %2 = add <1 x i64> %1, %1
    %3 = bitcast <1 x i64> %2 to <2 x float>
    %4 = fadd <2 x float> %3, %3
    store <2 x float> %4, ptr %q
    ret void
}

; CHECK-LABEL: test_v2f32_v2i32:
define void @test_v2f32_v2i32(ptr %p, ptr %q) {
; CHECK: ld1 { v{{[0-9]+}}.2s }
; CHECK: st1 { v{{[0-9]+}}.2s }
    %1 = load volatile <2 x i32>, ptr %p
    %2 = add <2 x i32> %1, %1
    %3 = bitcast <2 x i32> %2 to <2 x float>
    %4 = fadd <2 x float> %3, %3
    store <2 x float> %4, ptr %q
    ret void
}

; CHECK-LABEL: test_v2f32_v4i16:
define void @test_v2f32_v4i16(ptr %p, ptr %q) {
; CHECK: ld1 { v{{[0-9]+}}.4h }
; CHECK: rev32 v{{[0-9]+}}.4h
; CHECK: st1 { v{{[0-9]+}}.2s }
    %1 = load <4 x i16>, ptr %p
    %2 = add <4 x i16> %1, %1
    %3 = bitcast <4 x i16> %2 to <2 x float>
    %4 = fadd <2 x float> %3, %3
    store <2 x float> %4, ptr %q
    ret void
}

; CHECK-LABEL: test_v2f32_v4f16:
define void @test_v2f32_v4f16(ptr %p, ptr %q) {
; CHECK: ld1 { v{{[0-9]+}}.4h }
; CHECK-NOT: rev
; CHECK: fadd
; CHECK: rev32 v{{[0-9]+}}.4h
; CHECK: st1 { v{{[0-9]+}}.2s }
    %1 = load <4 x half>, ptr %p
    %2 = fadd <4 x half> %1, %1
    %3 = bitcast <4 x half> %2 to <2 x float>
    %4 = fadd <2 x float> %3, %3
    store <2 x float> %4, ptr %q
    ret void
}

; CHECK-LABEL: test_v2f32_v8i8:
define void @test_v2f32_v8i8(ptr %p, ptr %q) {
; CHECK: ld1 { v{{[0-9]+}}.8b }
; CHECK: rev32 v{{[0-9]+}}.8b
; CHECK: st1 { v{{[0-9]+}}.2s }
    %1 = load <8 x i8>, ptr %p
    %2 = add <8 x i8> %1, %1
    %3 = bitcast <8 x i8> %2 to <2 x float>
    %4 = fadd <2 x float> %3, %3
    store <2 x float> %4, ptr %q
    ret void
}

; CHECK-LABEL: test_v2i32_i64:
define void @test_v2i32_i64(ptr %p, ptr %q) {
; CHECK: ldr
; CHECK: rev64 v{{[0-9]+}}.2s
; CHECK: st1 { v{{[0-9]+}}.2s }
    %1 = load i64, ptr %p
    %2 = add i64 %1, %1
    %3 = bitcast i64 %2 to <2 x i32>
    %4 = add <2 x i32> %3, %3
    store <2 x i32> %4, ptr %q
    ret void
}

; CHECK-LABEL: test_v2i32_f64:
define void @test_v2i32_f64(ptr %p, ptr %q) {
; CHECK: ldr
; CHECK: rev64 v{{[0-9]+}}.2s
; CHECK: st1 { v{{[0-9]+}}.2s }
    %1 = load double, ptr %p
    %2 = fadd double %1, %1
    %3 = bitcast double %2 to <2 x i32>
    %4 = add <2 x i32> %3, %3
    store <2 x i32> %4, ptr %q
    ret void
}

; CHECK-LABEL: test_v2i32_v1i64:
define void @test_v2i32_v1i64(ptr %p, ptr %q) {
; CHECK: ldr
; CHECK: rev64 v{{[0-9]+}}.2s
; CHECK: st1 { v{{[0-9]+}}.2s }
    %1 = load <1 x i64>, ptr %p
    %2 = add <1 x i64> %1, %1
    %3 = bitcast <1 x i64> %2 to <2 x i32>
    %4 = add <2 x i32> %3, %3
    store <2 x i32> %4, ptr %q
    ret void
}

; CHECK-LABEL: test_v2i32_v2f32:
define void @test_v2i32_v2f32(ptr %p, ptr %q) {
; CHECK: ld1 { v{{[0-9]+}}.2s }
; CHECK: st1 { v{{[0-9]+}}.2s }
    %1 = load volatile <2 x float>, ptr %p
    %2 = fadd <2 x float> %1, %1
    %3 = bitcast <2 x float> %2 to <2 x i32>
    %4 = add <2 x i32> %3, %3
    store <2 x i32> %4, ptr %q
    ret void
}

; CHECK-LABEL: test_v2i32_v4i16:
define void @test_v2i32_v4i16(ptr %p, ptr %q) {
; CHECK: ld1 { v{{[0-9]+}}.4h }
; CHECK: rev32 v{{[0-9]+}}.4h
; CHECK: st1 { v{{[0-9]+}}.2s }
    %1 = load <4 x i16>, ptr %p
    %2 = add <4 x i16> %1, %1
    %3 = bitcast <4 x i16> %2 to <2 x i32>
    %4 = add <2 x i32> %3, %3
    store <2 x i32> %4, ptr %q
    ret void
}

; CHECK-LABEL: test_v2i32_v8i8:
define void @test_v2i32_v8i8(ptr %p, ptr %q) {
; CHECK: ld1 { v{{[0-9]+}}.8b }
; CHECK: rev32 v{{[0-9]+}}.8b
; CHECK: st1 { v{{[0-9]+}}.2s }
    %1 = load <8 x i8>, ptr %p
    %2 = add <8 x i8> %1, %1
    %3 = bitcast <8 x i8> %2 to <2 x i32>
    %4 = add <2 x i32> %3, %3
    store <2 x i32> %4, ptr %q
    ret void
}

; CHECK-LABEL: test_v4i16_i64:
define void @test_v4i16_i64(ptr %p, ptr %q) {
; CHECK: ldr
; CHECK: rev64 v{{[0-9]+}}.4h
; CHECK: st1 { v{{[0-9]+}}.4h }
    %1 = load i64, ptr %p
    %2 = add i64 %1, %1
    %3 = bitcast i64 %2 to <4 x i16>
    %4 = add <4 x i16> %3, %3
    store <4 x i16> %4, ptr %q
    ret void
}

; CHECK-LABEL: test_v4i16_f64:
define void @test_v4i16_f64(ptr %p, ptr %q) {
; CHECK: ldr
; CHECK: rev64 v{{[0-9]+}}.4h
; CHECK: st1 { v{{[0-9]+}}.4h }
    %1 = load double, ptr %p
    %2 = fadd double %1, %1
    %3 = bitcast double %2 to <4 x i16>
    %4 = add <4 x i16> %3, %3
    store <4 x i16> %4, ptr %q
    ret void
}

; CHECK-LABEL: test_v4i16_v1i64:
define void @test_v4i16_v1i64(ptr %p, ptr %q) {
; CHECK: ldr
; CHECK: rev64 v{{[0-9]+}}.4h
; CHECK: st1 { v{{[0-9]+}}.4h }
    %1 = load <1 x i64>, ptr %p
    %2 = add <1 x i64> %1, %1
    %3 = bitcast <1 x i64> %2 to <4 x i16>
    %4 = add <4 x i16> %3, %3
    store <4 x i16> %4, ptr %q
    ret void
}

; CHECK-LABEL: test_v4i16_v2f32:
define void @test_v4i16_v2f32(ptr %p, ptr %q) {
; CHECK: ld1 { v{{[0-9]+}}.2s }
; CHECK: rev32 v{{[0-9]+}}.4h
; CHECK: st1 { v{{[0-9]+}}.4h }
    %1 = load volatile <2 x float>, ptr %p
    %2 = fadd <2 x float> %1, %1
    %3 = bitcast <2 x float> %2 to <4 x i16>
    %4 = add <4 x i16> %3, %3
    store <4 x i16> %4, ptr %q
    ret void
}

; CHECK-LABEL: test_v4i16_v2i32:
define void @test_v4i16_v2i32(ptr %p, ptr %q) {
; CHECK: ld1 { v{{[0-9]+}}.2s }
; CHECK: rev32 v{{[0-9]+}}.4h
; CHECK: st1 { v{{[0-9]+}}.4h }
    %1 = load volatile <2 x i32>, ptr %p
    %2 = add <2 x i32> %1, %1
    %3 = bitcast <2 x i32> %2 to <4 x i16>
    %4 = add <4 x i16> %3, %3
    store <4 x i16> %4, ptr %q
    ret void
}

; CHECK-LABEL: test_v4i16_v4f16:
define void @test_v4i16_v4f16(ptr %p, ptr %q) {
; CHECK: ld1 { v{{[0-9]+}}.4h }
; CHECK-NOT: rev
; CHECK: st1 { v{{[0-9]+}}.4h }
    %1 = load <4 x half>, ptr %p
    %2 = fadd <4 x half> %1, %1
    %3 = bitcast <4 x half> %2 to <4 x i16>
    %4 = add <4 x i16> %3, %3
    store <4 x i16> %4, ptr %q
    ret void
}

; CHECK-LABEL: test_v4i16_v8i8:
define void @test_v4i16_v8i8(ptr %p, ptr %q) {
; CHECK: ld1 { v{{[0-9]+}}.8b }
; CHECK: rev16 v{{[0-9]+}}.8b
; CHECK: st1 { v{{[0-9]+}}.4h }
    %1 = load <8 x i8>, ptr %p
    %2 = add <8 x i8> %1, %1
    %3 = bitcast <8 x i8> %2 to <4 x i16>
    %4 = add <4 x i16> %3, %3
    store <4 x i16> %4, ptr %q
    ret void
}

; CHECK-LABEL: test_v4f16_i64:
define void @test_v4f16_i64(ptr %p, ptr %q) {
; CHECK: ldr
; CHECK: rev64 v{{[0-9]+}}.4h
; CHECK: fadd
; CHECK-NOT: rev
; CHECK: st1 { v{{[0-9]+}}.4h }
    %1 = load i64, ptr %p
    %2 = add i64 %1, %1
    %3 = bitcast i64 %2 to <4 x half>
    %4 = fadd <4 x half> %3, %3
    store <4 x half> %4, ptr %q
    ret void
}

; CHECK-LABEL: test_v4f16_f64:
define void @test_v4f16_f64(ptr %p, ptr %q) {
; CHECK: ldr
; CHECK: rev64 v{{[0-9]+}}.4h
; CHECK: fadd
; CHECK-NOT: rev
; CHECK: st1 { v{{[0-9]+}}.4h }
    %1 = load double, ptr %p
    %2 = fadd double %1, %1
    %3 = bitcast double %2 to <4 x half>
    %4 = fadd <4 x half> %3, %3
    store <4 x half> %4, ptr %q
    ret void
}

; CHECK-LABEL: test_v4f16_v1i64:
define void @test_v4f16_v1i64(ptr %p, ptr %q) {
; CHECK: ldr
; CHECK: rev64 v{{[0-9]+}}.4h
; CHECK: fadd
; CHECK-NOT: rev
; CHECK: st1 { v{{[0-9]+}}.4h }
    %1 = load <1 x i64>, ptr %p
    %2 = add <1 x i64> %1, %1
    %3 = bitcast <1 x i64> %2 to <4 x half>
    %4 = fadd <4 x half> %3, %3
    store <4 x half> %4, ptr %q
    ret void
}

; CHECK-LABEL: test_v4f16_v2f32:
define void @test_v4f16_v2f32(ptr %p, ptr %q) {
; CHECK: ld1 { v{{[0-9]+}}.2s }
; CHECK: rev32 v{{[0-9]+}}.4h
; CHECK: fadd
; CHECK-NOT: rev
; CHECK: st1 { v{{[0-9]+}}.4h }
    %1 = load volatile <2 x float>, ptr %p
    %2 = fadd <2 x float> %1, %1
    %3 = bitcast <2 x float> %2 to <4 x half>
    %4 = fadd <4 x half> %3, %3
    store <4 x half> %4, ptr %q
    ret void
}

; CHECK-LABEL: test_v4f16_v2i32:
define void @test_v4f16_v2i32(ptr %p, ptr %q) {
; CHECK: ld1 { v{{[0-9]+}}.2s }
; CHECK: rev32 v{{[0-9]+}}.4h
; CHECK: fadd
; CHECK-NOT: rev
; CHECK: st1 { v{{[0-9]+}}.4h }
    %1 = load volatile <2 x i32>, ptr %p
    %2 = add <2 x i32> %1, %1
    %3 = bitcast <2 x i32> %2 to <4 x half>
    %4 = fadd <4 x half> %3, %3
    store <4 x half> %4, ptr %q
    ret void
}

; CHECK-LABEL: test_v4f16_v4i16:
define void @test_v4f16_v4i16(ptr %p, ptr %q) {
; CHECK: ld1 { v{{[0-9]+}}.4h }
; CHECK-NOT: rev
; CHECK: st1 { v{{[0-9]+}}.4h }
    %1 = load <4 x i16>, ptr %p
    %2 = add <4 x i16> %1, %1
    %3 = bitcast <4 x i16> %2 to <4 x half>
    %4 = fadd <4 x half> %3, %3
    store <4 x half> %4, ptr %q
    ret void
}

; CHECK-LABEL: test_v4f16_v8i8:
define void @test_v4f16_v8i8(ptr %p, ptr %q) {
; CHECK: ld1 { v{{[0-9]+}}.8b }
; CHECK: rev16 v{{[0-9]+}}.8b
; CHECK: fadd
; CHECK-NOT: rev
; CHECK: st1 { v{{[0-9]+}}.4h }
    %1 = load <8 x i8>, ptr %p
    %2 = add <8 x i8> %1, %1
    %3 = bitcast <8 x i8> %2 to <4 x half>
    %4 = fadd <4 x half> %3, %3
    store <4 x half> %4, ptr %q
    ret void
}

; CHECK-LABEL: test_v8i8_i64:
define void @test_v8i8_i64(ptr %p, ptr %q) {
; CHECK: ldr
; CHECK: rev64 v{{[0-9]+}}.8b
; CHECK: st1 { v{{[0-9]+}}.8b }
    %1 = load i64, ptr %p
    %2 = add i64 %1, %1
    %3 = bitcast i64 %2 to <8 x i8>
    %4 = add <8 x i8> %3, %3
    store <8 x i8> %4, ptr %q
    ret void
}

; CHECK-LABEL: test_v8i8_f64:
define void @test_v8i8_f64(ptr %p, ptr %q) {
; CHECK: ldr
; CHECK: rev64 v{{[0-9]+}}.8b
; CHECK: st1 { v{{[0-9]+}}.8b }
    %1 = load double, ptr %p
    %2 = fadd double %1, %1
    %3 = bitcast double %2 to <8 x i8>
    %4 = add <8 x i8> %3, %3
    store <8 x i8> %4, ptr %q
    ret void
}

; CHECK-LABEL: test_v8i8_v1i64:
define void @test_v8i8_v1i64(ptr %p, ptr %q) {
; CHECK: ldr
; CHECK: rev64 v{{[0-9]+}}.8b
; CHECK: st1 { v{{[0-9]+}}.8b }
    %1 = load <1 x i64>, ptr %p
    %2 = add <1 x i64> %1, %1
    %3 = bitcast <1 x i64> %2 to <8 x i8>
    %4 = add <8 x i8> %3, %3
    store <8 x i8> %4, ptr %q
    ret void
}

; CHECK-LABEL: test_v8i8_v2f32:
define void @test_v8i8_v2f32(ptr %p, ptr %q) {
; CHECK: ld1 { v{{[0-9]+}}.2s }
; CHECK: rev32 v{{[0-9]+}}.8b
; CHECK: st1 { v{{[0-9]+}}.8b }
    %1 = load volatile <2 x float>, ptr %p
    %2 = fadd <2 x float> %1, %1
    %3 = bitcast <2 x float> %2 to <8 x i8>
    %4 = add <8 x i8> %3, %3
    store <8 x i8> %4, ptr %q
    ret void
}

; CHECK-LABEL: test_v8i8_v2i32:
define void @test_v8i8_v2i32(ptr %p, ptr %q) {
; CHECK: ld1 { v{{[0-9]+}}.2s }
; CHECK: rev32 v{{[0-9]+}}.8b
; CHECK: st1 { v{{[0-9]+}}.8b }
    %1 = load volatile <2 x i32>, ptr %p
    %2 = add <2 x i32> %1, %1
    %3 = bitcast <2 x i32> %2 to <8 x i8>
    %4 = add <8 x i8> %3, %3
    store <8 x i8> %4, ptr %q
    ret void
}

; CHECK-LABEL: test_v8i8_v4i16:
define void @test_v8i8_v4i16(ptr %p, ptr %q) {
; CHECK: ld1 { v{{[0-9]+}}.4h }
; CHECK: rev16 v{{[0-9]+}}.8b
; CHECK: st1 { v{{[0-9]+}}.8b }
    %1 = load <4 x i16>, ptr %p
    %2 = add <4 x i16> %1, %1
    %3 = bitcast <4 x i16> %2 to <8 x i8>
    %4 = add <8 x i8> %3, %3
    store <8 x i8> %4, ptr %q
    ret void
}

; CHECK-LABEL: test_f128_v2f64:
define void @test_f128_v2f64(ptr %p, ptr %q) {
; CHECK: ld1 { v{{[0-9]+}}.2d }
; CHECK: ext
; CHECK: str
    %1 = load volatile <2 x double>, ptr %p
    %2 = fadd <2 x double> %1, %1
    %3 = bitcast <2 x double> %2 to fp128
    %4 = fadd fp128 %3, %3
    store fp128 %4, ptr %q
    ret void
}

; CHECK-LABEL: test_f128_v2i64:
define void @test_f128_v2i64(ptr %p, ptr %q) {
; CHECK: ld1 { v{{[0-9]+}}.2d }
; CHECK: ext
; CHECK: str
    %1 = load volatile <2 x i64>, ptr %p
    %2 = add <2 x i64> %1, %1
    %3 = bitcast <2 x i64> %2 to fp128
    %4 = fadd fp128 %3, %3
    store fp128 %4, ptr %q
    ret void
}

; CHECK-LABEL: test_f128_v4f32:
define void @test_f128_v4f32(ptr %p, ptr %q) {
; CHECK: ld1 { v{{[0-9]+}}.4s }
; CHECK-NOT: rev
; CHECK: fadd
; CHECK: rev64 v{{[0-9]+}}.4s
; CHECK: ext
; CHECK: str q
    %1 = load <4 x float>, ptr %p
    %2 = fadd <4 x float> %1, %1
    %3 = bitcast <4 x float> %2 to fp128
    %4 = fadd fp128 %3, %3
    store fp128 %4, ptr %q
    ret void
}

; CHECK-LABEL: test_f128_v4i32:
define void @test_f128_v4i32(ptr %p, ptr %q) {
; CHECK: ld1 { v{{[0-9]+}}.4s }
; CHECK: rev64 v{{[0-9]+}}.4s
; CHECK: ext
; CHECK: str
    %1 = load <4 x i32>, ptr %p
    %2 = add <4 x i32> %1, %1
    %3 = bitcast <4 x i32> %2 to fp128
    %4 = fadd fp128 %3, %3
    store fp128 %4, ptr %q
    ret void
}

; CHECK-LABEL: test_f128_v8i16:
define void @test_f128_v8i16(ptr %p, ptr %q) {
; CHECK: ld1 { v{{[0-9]+}}.8h }
; CHECK: rev64 v{{[0-9]+}}.8h
; CHECK: ext
; CHECK: str
    %1 = load <8 x i16>, ptr %p
    %2 = add <8 x i16> %1, %1
    %3 = bitcast <8 x i16> %2 to fp128
    %4 = fadd fp128 %3, %3
    store fp128 %4, ptr %q
    ret void
}

; CHECK-LABEL: test_f128_v16i8:
define void @test_f128_v16i8(ptr %p, ptr %q) {
; CHECK: ld1 { v{{[0-9]+}}.16b }
; CHECK: ext
; CHECK: str q
    %1 = load <16 x i8>, ptr %p
    %2 = add <16 x i8> %1, %1
    %3 = bitcast <16 x i8> %2 to fp128
    %4 = fadd fp128 %3, %3
    store fp128 %4, ptr %q
    ret void
}

; CHECK-LABEL: test_v2f64_f128:
define void @test_v2f64_f128(ptr %p, ptr %q) {
; CHECK: ldr
; CHECK: ext
; CHECK: st1 { v{{[0-9]+}}.2d }
    %1 = load fp128, ptr %p
    %2 = fadd fp128 %1, %1
    %3 = bitcast fp128 %2 to <2 x double>
    %4 = fadd <2 x double> %3, %3
    store <2 x double> %4, ptr %q
    ret void
}

; CHECK-LABEL: test_v2f64_v2i64:
define void @test_v2f64_v2i64(ptr %p, ptr %q) {
; CHECK: ld1 { v{{[0-9]+}}.2d }
; CHECK: st1 { v{{[0-9]+}}.2d }
    %1 = load volatile <2 x i64>, ptr %p
    %2 = add <2 x i64> %1, %1
    %3 = bitcast <2 x i64> %2 to <2 x double>
    %4 = fadd <2 x double> %3, %3
    store <2 x double> %4, ptr %q
    ret void
}

; CHECK-LABEL: test_v2f64_v4f32:
define void @test_v2f64_v4f32(ptr %p, ptr %q) {
; CHECK: ld1 { v{{[0-9]+}}.4s }
; CHECK-NOT: rev
; CHECK: fadd
; CHECK: rev64 v{{[0-9]+}}.4s
; CHECK: st1 { v{{[0-9]+}}.2d }
    %1 = load <4 x float>, ptr %p
    %2 = fadd <4 x float> %1, %1
    %3 = bitcast <4 x float> %2 to <2 x double>
    %4 = fadd <2 x double> %3, %3
    store <2 x double> %4, ptr %q
    ret void
}

; CHECK-LABEL: test_v2f64_v4i32:
define void @test_v2f64_v4i32(ptr %p, ptr %q) {
; CHECK: ld1 { v{{[0-9]+}}.4s }
; CHECK: rev64 v{{[0-9]+}}.4s
; CHECK: st1 { v{{[0-9]+}}.2d }
    %1 = load <4 x i32>, ptr %p
    %2 = add <4 x i32> %1, %1
    %3 = bitcast <4 x i32> %2 to <2 x double>
    %4 = fadd <2 x double> %3, %3
    store <2 x double> %4, ptr %q
    ret void
}

; CHECK-LABEL: test_v2f64_v8i16:
define void @test_v2f64_v8i16(ptr %p, ptr %q) {
; CHECK: ld1 { v{{[0-9]+}}.8h }
; CHECK: rev64 v{{[0-9]+}}.8h
; CHECK: st1 { v{{[0-9]+}}.2d }
    %1 = load <8 x i16>, ptr %p
    %2 = add <8 x i16> %1, %1
    %3 = bitcast <8 x i16> %2 to <2 x double>
    %4 = fadd <2 x double> %3, %3
    store <2 x double> %4, ptr %q
    ret void
}

; CHECK-LABEL: test_v2f64_v16i8:
define void @test_v2f64_v16i8(ptr %p, ptr %q) {
; CHECK: ld1 { v{{[0-9]+}}.16b }
; CHECK: rev64 v{{[0-9]+}}.16b
; CHECK: st1 { v{{[0-9]+}}.2d }
    %1 = load <16 x i8>, ptr %p
    %2 = add <16 x i8> %1, %1
    %3 = bitcast <16 x i8> %2 to <2 x double>
    %4 = fadd <2 x double> %3, %3
    store <2 x double> %4, ptr %q
    ret void
}

; CHECK-LABEL: test_v2i64_f128:
define void @test_v2i64_f128(ptr %p, ptr %q) {
; CHECK: ldr
; CHECK: ext
; CHECK: st1 { v{{[0-9]+}}.2d }
    %1 = load fp128, ptr %p
    %2 = fadd fp128 %1, %1
    %3 = bitcast fp128 %2 to <2 x i64>
    %4 = add <2 x i64> %3, %3
    store <2 x i64> %4, ptr %q
    ret void
}

; CHECK-LABEL: test_v2i64_v2f64:
define void @test_v2i64_v2f64(ptr %p, ptr %q) {
; CHECK: ld1 { v{{[0-9]+}}.2d }
; CHECK: st1 { v{{[0-9]+}}.2d }
    %1 = load volatile <2 x double>, ptr %p
    %2 = fadd <2 x double> %1, %1
    %3 = bitcast <2 x double> %2 to <2 x i64>
    %4 = add <2 x i64> %3, %3
    store <2 x i64> %4, ptr %q
    ret void
}

; CHECK-LABEL: test_v2i64_v4f32:
define void @test_v2i64_v4f32(ptr %p, ptr %q) {
; CHECK: ld1 { v{{[0-9]+}}.4s }
; CHECK-NOT: rev
; CHECK: fadd
; CHECK: rev64 v{{[0-9]+}}.4s
; CHECK: add
; CHECK: st1 { v{{[0-9]+}}.2d }
    %1 = load <4 x float>, ptr %p
    %2 = fadd <4 x float> %1, %1
    %3 = bitcast <4 x float> %2 to <2 x i64>
    %4 = add <2 x i64> %3, %3
    store <2 x i64> %4, ptr %q
    ret void
}

; CHECK-LABEL: test_v2i64_v4i32:
define void @test_v2i64_v4i32(ptr %p, ptr %q) {
; CHECK: ld1 { v{{[0-9]+}}.4s }
; CHECK: rev64 v{{[0-9]+}}.4s
; CHECK: st1 { v{{[0-9]+}}.2d }
    %1 = load <4 x i32>, ptr %p
    %2 = add <4 x i32> %1, %1
    %3 = bitcast <4 x i32> %2 to <2 x i64>
    %4 = add <2 x i64> %3, %3
    store <2 x i64> %4, ptr %q
    ret void
}

; CHECK-LABEL: test_v2i64_v8i16:
define void @test_v2i64_v8i16(ptr %p, ptr %q) {
; CHECK: ld1 { v{{[0-9]+}}.8h }
; CHECK: rev64 v{{[0-9]+}}.8h
; CHECK: st1 { v{{[0-9]+}}.2d }
    %1 = load <8 x i16>, ptr %p
    %2 = add <8 x i16> %1, %1
    %3 = bitcast <8 x i16> %2 to <2 x i64>
    %4 = add <2 x i64> %3, %3
    store <2 x i64> %4, ptr %q
    ret void
}

; CHECK-LABEL: test_v2i64_v16i8:
define void @test_v2i64_v16i8(ptr %p, ptr %q) {
; CHECK: ld1 { v{{[0-9]+}}.16b }
; CHECK: rev64 v{{[0-9]+}}.16b
; CHECK: st1 { v{{[0-9]+}}.2d }
    %1 = load <16 x i8>, ptr %p
    %2 = add <16 x i8> %1, %1
    %3 = bitcast <16 x i8> %2 to <2 x i64>
    %4 = add <2 x i64> %3, %3
    store <2 x i64> %4, ptr %q
    ret void
}

; CHECK-LABEL: test_v4f32_f128:
define void @test_v4f32_f128(ptr %p, ptr %q) {
; CHECK: ldr q
; CHECK: rev64 v{{[0-9]+}}.4s
; CHECK: ext
; CHECK-NOT: rev
; CHECK: st1 { v{{[0-9]+}}.4s }
    %1 = load fp128, ptr %p
    %2 = fadd fp128 %1, %1
    %3 = bitcast fp128 %2 to <4 x float>
    %4 = fadd <4 x float> %3, %3
    store <4 x float> %4, ptr %q
    ret void
}

; CHECK-LABEL: test_v4f32_v2f64:
define void @test_v4f32_v2f64(ptr %p, ptr %q) {
; CHECK: ld1 { v{{[0-9]+}}.2d }
; CHECK: rev64 v{{[0-9]+}}.4s
; CHECK-NOT: rev
; CHECK: st1 { v{{[0-9]+}}.4s }
    %1 = load volatile <2 x double>, ptr %p
    %2 = fadd <2 x double> %1, %1
    %3 = bitcast <2 x double> %2 to <4 x float>
    %4 = fadd <4 x float> %3, %3
    store <4 x float> %4, ptr %q
    ret void
}

; CHECK-LABEL: test_v4f32_v2i64:
define void @test_v4f32_v2i64(ptr %p, ptr %q) {
; CHECK: ld1 { v{{[0-9]+}}.2d }
; CHECK: rev64 v{{[0-9]+}}.4s
; CHECK: fadd
; CHECK-NOT: rev
; CHECK: st1 { v{{[0-9]+}}.4s }
    %1 = load volatile <2 x i64>, ptr %p
    %2 = add <2 x i64> %1, %1
    %3 = bitcast <2 x i64> %2 to <4 x float>
    %4 = fadd <4 x float> %3, %3
    store <4 x float> %4, ptr %q
    ret void
}

; CHECK-LABEL: test_v4f32_v4i32:
define void @test_v4f32_v4i32(ptr %p, ptr %q) {
; CHECK: ld1 { v{{[0-9]+}}.4s }
; CHECK-NOT: rev
; CHECK: st1 { v{{[0-9]+}}.4s }
    %1 = load <4 x i32>, ptr %p
    %2 = add <4 x i32> %1, %1
    %3 = bitcast <4 x i32> %2 to <4 x float>
    %4 = fadd <4 x float> %3, %3
    store <4 x float> %4, ptr %q
    ret void
}

; CHECK-LABEL: test_v4f32_v8i16:
define void @test_v4f32_v8i16(ptr %p, ptr %q) {
; CHECK: ld1 { v{{[0-9]+}}.8h }
; CHECK: rev32 v{{[0-9]+}}.8h
; CHECK-NOT: rev
; CHECK: st1 { v{{[0-9]+}}.4s }
    %1 = load <8 x i16>, ptr %p
    %2 = add <8 x i16> %1, %1
    %3 = bitcast <8 x i16> %2 to <4 x float>
    %4 = fadd <4 x float> %3, %3
    store <4 x float> %4, ptr %q
    ret void
}

; CHECK-LABEL: test_v4f32_v16i8:
define void @test_v4f32_v16i8(ptr %p, ptr %q) {
; CHECK: ld1 { v{{[0-9]+}}.16b }
; CHECK: rev32 v{{[0-9]+}}.16b
; CHECK-NOT: rev
; CHECK: st1 { v{{[0-9]+}}.4s }
    %1 = load <16 x i8>, ptr %p
    %2 = add <16 x i8> %1, %1
    %3 = bitcast <16 x i8> %2 to <4 x float>
    %4 = fadd <4 x float> %3, %3
    store <4 x float> %4, ptr %q
    ret void
}

; CHECK-LABEL: test_v4i32_f128:
define void @test_v4i32_f128(ptr %p, ptr %q) {
; CHECK: ldr
; CHECK: rev64 v{{[0-9]+}}.4s
; CHECK: ext
; CHECK: st1 { v{{[0-9]+}}.4s }
    %1 = load fp128, ptr %p
    %2 = fadd fp128 %1, %1
    %3 = bitcast fp128 %2 to <4 x i32>
    %4 = add <4 x i32> %3, %3
    store <4 x i32> %4, ptr %q
    ret void
}

; CHECK-LABEL: test_v4i32_v2f64:
define void @test_v4i32_v2f64(ptr %p, ptr %q) {
; CHECK: ld1 { v{{[0-9]+}}.2d }
; CHECK: rev64 v{{[0-9]+}}.4s
; CHECK: st1 { v{{[0-9]+}}.4s }
    %1 = load volatile <2 x double>, ptr %p
    %2 = fadd <2 x double> %1, %1
    %3 = bitcast <2 x double> %2 to <4 x i32>
    %4 = add <4 x i32> %3, %3
    store <4 x i32> %4, ptr %q
    ret void
}

; CHECK-LABEL: test_v4i32_v2i64:
define void @test_v4i32_v2i64(ptr %p, ptr %q) {
; CHECK: ld1 { v{{[0-9]+}}.2d }
; CHECK: rev64 v{{[0-9]+}}.4s
; CHECK: st1 { v{{[0-9]+}}.4s }
    %1 = load volatile <2 x i64>, ptr %p
    %2 = add <2 x i64> %1, %1
    %3 = bitcast <2 x i64> %2 to <4 x i32>
    %4 = add <4 x i32> %3, %3
    store <4 x i32> %4, ptr %q
    ret void
}

; CHECK-LABEL: test_v4i32_v4f32:
define void @test_v4i32_v4f32(ptr %p, ptr %q) {
; CHECK: ld1 { v{{[0-9]+}}.4s }
; CHECK-NOT: rev
; CHECK: st1 { v{{[0-9]+}}.4s }
    %1 = load <4 x float>, ptr %p
    %2 = fadd <4 x float> %1, %1
    %3 = bitcast <4 x float> %2 to <4 x i32>
    %4 = add <4 x i32> %3, %3
    store <4 x i32> %4, ptr %q
    ret void
}

; CHECK-LABEL: test_v4i32_v8i16:
define void @test_v4i32_v8i16(ptr %p, ptr %q) {
; CHECK: ld1 { v{{[0-9]+}}.8h }
; CHECK: rev32 v{{[0-9]+}}.8h
; CHECK: st1 { v{{[0-9]+}}.4s }
    %1 = load <8 x i16>, ptr %p
    %2 = add <8 x i16> %1, %1
    %3 = bitcast <8 x i16> %2 to <4 x i32>
    %4 = add <4 x i32> %3, %3
    store <4 x i32> %4, ptr %q
    ret void
}

; CHECK-LABEL: test_v4i32_v16i8:
define void @test_v4i32_v16i8(ptr %p, ptr %q) {
; CHECK: ld1 { v{{[0-9]+}}.16b }
; CHECK: rev32 v{{[0-9]+}}.16b
; CHECK: st1 { v{{[0-9]+}}.4s }
    %1 = load <16 x i8>, ptr %p
    %2 = add <16 x i8> %1, %1
    %3 = bitcast <16 x i8> %2 to <4 x i32>
    %4 = add <4 x i32> %3, %3
    store <4 x i32> %4, ptr %q
    ret void
}

; CHECK-LABEL: test_v8i16_f128:
define void @test_v8i16_f128(ptr %p, ptr %q) {
; CHECK: ldr
; CHECK: rev64 v{{[0-9]+}}.8h
; CHECK: ext
; CHECK: st1 { v{{[0-9]+}}.8h }
    %1 = load fp128, ptr %p
    %2 = fadd fp128 %1, %1
    %3 = bitcast fp128 %2 to <8 x i16>
    %4 = add <8 x i16> %3, %3
    store <8 x i16> %4, ptr %q
    ret void
}

; CHECK-LABEL: test_v8i16_v2f64:
define void @test_v8i16_v2f64(ptr %p, ptr %q) {
; CHECK: ld1 { v{{[0-9]+}}.2d }
; CHECK: rev64 v{{[0-9]+}}.8h
; CHECK: st1 { v{{[0-9]+}}.8h }
    %1 = load volatile <2 x double>, ptr %p
    %2 = fadd <2 x double> %1, %1
    %3 = bitcast <2 x double> %2 to <8 x i16>
    %4 = add <8 x i16> %3, %3
    store <8 x i16> %4, ptr %q
    ret void
}

; CHECK-LABEL: test_v8i16_v2i64:
define void @test_v8i16_v2i64(ptr %p, ptr %q) {
; CHECK: ld1 { v{{[0-9]+}}.2d }
; CHECK: rev64 v{{[0-9]+}}.8h
; CHECK: st1 { v{{[0-9]+}}.8h }
    %1 = load volatile <2 x i64>, ptr %p
    %2 = add <2 x i64> %1, %1
    %3 = bitcast <2 x i64> %2 to <8 x i16>
    %4 = add <8 x i16> %3, %3
    store <8 x i16> %4, ptr %q
    ret void
}

; CHECK-LABEL: test_v8i16_v4f32:
define void @test_v8i16_v4f32(ptr %p, ptr %q) {
; CHECK: ld1 { v{{[0-9]+}}.4s }
; CHECK: rev32 v{{[0-9]+}}.8h
; CHECK-NOT: rev
; CHECK: st1 { v{{[0-9]+}}.8h }
    %1 = load <4 x float>, ptr %p
    %2 = fadd <4 x float> %1, %1
    %3 = bitcast <4 x float> %2 to <8 x i16>
    %4 = add <8 x i16> %3, %3
    store <8 x i16> %4, ptr %q
    ret void
}

; CHECK-LABEL: test_v8i16_v4i32:
define void @test_v8i16_v4i32(ptr %p, ptr %q) {
; CHECK: ld1 { v{{[0-9]+}}.4s }
; CHECK: rev32 v{{[0-9]+}}.8h
; CHECK: st1 { v{{[0-9]+}}.8h }
    %1 = load <4 x i32>, ptr %p
    %2 = add <4 x i32> %1, %1
    %3 = bitcast <4 x i32> %2 to <8 x i16>
    %4 = add <8 x i16> %3, %3
    store <8 x i16> %4, ptr %q
    ret void
}

; CHECK-LABEL: test_v8i16_v8f16:
define void @test_v8i16_v8f16(ptr %p, ptr %q) {
; CHECK: ld1 { v{{[0-9]+}}.8h }
; CHECK-NOT: rev
; CHECK: st1 { v{{[0-9]+}}.8h }
    %1 = load <8 x half>, ptr %p
    %2 = fadd <8 x half> %1, %1
    %3 = bitcast <8 x half> %2 to <8 x i16>
    %4 = add <8 x i16> %3, %3
    store <8 x i16> %4, ptr %q
    ret void
}

; CHECK-LABEL: test_v8i16_v16i8:
define void @test_v8i16_v16i8(ptr %p, ptr %q) {
; CHECK: ld1 { v{{[0-9]+}}.16b }
; CHECK: rev16 v{{[0-9]+}}.16b
; CHECK: st1 { v{{[0-9]+}}.8h }
    %1 = load <16 x i8>, ptr %p
    %2 = add <16 x i8> %1, %1
    %3 = bitcast <16 x i8> %2 to <8 x i16>
    %4 = add <8 x i16> %3, %3
    store <8 x i16> %4, ptr %q
    ret void
}

; CHECK-LABEL: test_v16i8_f128:
define void @test_v16i8_f128(ptr %p, ptr %q) {
; CHECK: ldr q
; CHECK: rev64 v{{[0-9]+}}.16b
; CHECK: ext
; CHECK: st1 { v{{[0-9]+}}.16b }
    %1 = load fp128, ptr %p
    %2 = fadd fp128 %1, %1
    %3 = bitcast fp128 %2 to <16 x i8>
    %4 = add <16 x i8> %3, %3
    store <16 x i8> %4, ptr %q
    ret void
}

; CHECK-LABEL: test_v16i8_v2f64:
define void @test_v16i8_v2f64(ptr %p, ptr %q) {
; CHECK: ld1 { v{{[0-9]+}}.2d }
; CHECK: rev64 v{{[0-9]+}}.16b
; CHECK: st1 { v{{[0-9]+}}.16b }
    %1 = load volatile <2 x double>, ptr %p
    %2 = fadd <2 x double> %1, %1
    %3 = bitcast <2 x double> %2 to <16 x i8>
    %4 = add <16 x i8> %3, %3
    store <16 x i8> %4, ptr %q
    ret void
}

; CHECK-LABEL: test_v16i8_v2i64:
define void @test_v16i8_v2i64(ptr %p, ptr %q) {
; CHECK: ld1 { v{{[0-9]+}}.2d }
; CHECK: rev64 v{{[0-9]+}}.16b
; CHECK: st1 { v{{[0-9]+}}.16b }
    %1 = load volatile <2 x i64>, ptr %p
    %2 = add <2 x i64> %1, %1
    %3 = bitcast <2 x i64> %2 to <16 x i8>
    %4 = add <16 x i8> %3, %3
    store <16 x i8> %4, ptr %q
    ret void
}

; CHECK-LABEL: test_v16i8_v4f32:
define void @test_v16i8_v4f32(ptr %p, ptr %q) {
; CHECK: ld1 { v{{[0-9]+}}.4s }
; CHECK: rev32 v{{[0-9]+}}.16b
; CHECK-NOT: rev
; CHECK: st1 { v{{[0-9]+}}.16b }
    %1 = load <4 x float>, ptr %p
    %2 = fadd <4 x float> %1, %1
    %3 = bitcast <4 x float> %2 to <16 x i8>
    %4 = add <16 x i8> %3, %3
    store <16 x i8> %4, ptr %q
    ret void
}

; CHECK-LABEL: test_v16i8_v4i32:
define void @test_v16i8_v4i32(ptr %p, ptr %q) {
; CHECK: ld1 { v{{[0-9]+}}.4s }
; CHECK: rev32 v{{[0-9]+}}.16b
; CHECK: st1 { v{{[0-9]+}}.16b }
    %1 = load <4 x i32>, ptr %p
    %2 = add <4 x i32> %1, %1
    %3 = bitcast <4 x i32> %2 to <16 x i8>
    %4 = add <16 x i8> %3, %3
    store <16 x i8> %4, ptr %q
    ret void
}

; CHECK-LABEL: test_v16i8_v8f16:
define void @test_v16i8_v8f16(ptr %p, ptr %q) {
; CHECK: ld1 { v{{[0-9]+}}.8h }
; CHECK: rev16 v{{[0-9]+}}.16b
; CHECK-NOT: rev
; CHECK: st1 { v{{[0-9]+}}.16b }
    %1 = load <8 x half>, ptr %p
    %2 = fadd <8 x half> %1, %1
    %3 = bitcast <8 x half> %2 to <16 x i8>
    %4 = add <16 x i8> %3, %3
    store <16 x i8> %4, ptr %q
    ret void
}

; CHECK-LABEL: test_v16i8_v8i16:
define void @test_v16i8_v8i16(ptr %p, ptr %q) {
; CHECK: ld1 { v{{[0-9]+}}.8h }
; CHECK: rev16 v{{[0-9]+}}.16b
; CHECK: st1 { v{{[0-9]+}}.16b }
    %1 = load <8 x i16>, ptr %p
    %2 = add <8 x i16> %1, %1
    %3 = bitcast <8 x i16> %2 to <16 x i8>
    %4 = add <16 x i8> %3, %3
    store <16 x i8> %4, ptr %q
    ret void
}

; CHECK-LABEL: test_v4f16_struct:
%struct.struct1 = type { half, half, half, half }
define %struct.struct1 @test_v4f16_struct(ptr %ret) {
entry:
; CHECK: ld1 { {{v[0-9]+}}.4h }
; CHECK-NOT: rev
  %0 = load volatile <4 x half>, ptr %ret, align 2
  %1 = extractelement <4 x half> %0, i32 0
  %.fca.0.insert = insertvalue %struct.struct1 undef, half %1, 0
  ret %struct.struct1 %.fca.0.insert
}
