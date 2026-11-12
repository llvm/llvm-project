; RUN: llc -mtriple aarch64_be < %s -aarch64-enable-ldst-opt=false -o - | FileCheck %s
; RUN: llc -mtriple aarch64_be < %s -aarch64-enable-ldst-opt=false -fast-isel=true -O0 -o - | FileCheck %s

; Note, we split the functions in to multiple BBs below to isolate the call
; instruction we want to test, from fast-isel failing to select instructions
; after it.

; CHECK-LABEL: test_i64_f64:
declare i64 @test_i64_f64_helper(double %p)
define void @test_i64_f64(ptr %p, ptr %q) {
; CHECK-NOT: rev
    %1 = load double, ptr %p
    %2 = fadd double %1, %1
    %3 = call i64 @test_i64_f64_helper(double %2)
    br label %return_bb
return_bb:
    %4 = add i64 %3, %3
    store i64 %4, ptr %q
    ret void
}

; CHECK-LABEL: test_i64_v1i64:
declare i64 @test_i64_v1i64_helper(<1 x i64> %p)
define void @test_i64_v1i64(ptr %p, ptr %q) {
; CHECK-NOT: rev
    %1 = load <1 x i64>, ptr %p
    %2 = add <1 x i64> %1, %1
    %3 = call i64 @test_i64_v1i64_helper(<1 x i64> %2)
    br label %return_bb
return_bb:
    %4 = add i64 %3, %3
    store i64 %4, ptr %q
    ret void
}

; CHECK-LABEL: test_i64_v2f32:
declare i64 @test_i64_v2f32_helper(<2 x float> %p)
define void @test_i64_v2f32(ptr %p, ptr %q) {
; CHECK: rev64 v{{[0-9]+}}.2s
    %1 = load <2 x float>, ptr %p
    %2 = fadd <2 x float> %1, %1
    %3 = call i64 @test_i64_v2f32_helper(<2 x float> %2)
    br label %return_bb
return_bb:
    %4 = add i64 %3, %3
    store i64 %4, ptr %q
    ret void
}

; CHECK-LABEL: test_i64_v2i32:
declare i64 @test_i64_v2i32_helper(<2 x i32> %p)
define void @test_i64_v2i32(ptr %p, ptr %q) {
; CHECK: rev64 v{{[0-9]+}}.2s
    %1 = load <2 x i32>, ptr %p
    %2 = add <2 x i32> %1, %1
    %3 = call i64 @test_i64_v2i32_helper(<2 x i32> %2)
    br label %return_bb
return_bb:
    %4 = add i64 %3, %3
    store i64 %4, ptr %q
    ret void
}

; CHECK-LABEL: test_i64_v4i16:
declare i64 @test_i64_v4i16_helper(<4 x i16> %p)
define void @test_i64_v4i16(ptr %p, ptr %q) {
; CHECK: rev64 v{{[0-9]+}}.4h
    %1 = load <4 x i16>, ptr %p
    %2 = add <4 x i16> %1, %1
    %3 = call i64 @test_i64_v4i16_helper(<4 x i16> %2)
    br label %return_bb
return_bb:
    %4 = add i64 %3, %3
    store i64 %4, ptr %q
    ret void
}

; CHECK-LABEL: test_i64_v8i8:
declare i64 @test_i64_v8i8_helper(<8 x i8> %p)
define void @test_i64_v8i8(ptr %p, ptr %q) {
; CHECK: rev64 v{{[0-9]+}}.8b
    %1 = load <8 x i8>, ptr %p
    %2 = add <8 x i8> %1, %1
    %3 = call i64 @test_i64_v8i8_helper(<8 x i8> %2)
    br label %return_bb
return_bb:
    %4 = add i64 %3, %3
    store i64 %4, ptr %q
    ret void
}

; CHECK-LABEL: test_f64_i64:
declare double @test_f64_i64_helper(i64 %p)
define void @test_f64_i64(ptr %p, ptr %q) {
; CHECK-NOT: rev
    %1 = load i64, ptr %p
    %2 = add i64 %1, %1
    %3 = call double @test_f64_i64_helper(i64 %2)
    br label %return_bb
return_bb:
    %4 = fadd double %3, %3
    store double %4, ptr %q
    ret void
}

; CHECK-LABEL: test_f64_v1i64:
declare double @test_f64_v1i64_helper(<1 x i64> %p)
define void @test_f64_v1i64(ptr %p, ptr %q) {
; CHECK-NOT: rev
    %1 = load <1 x i64>, ptr %p
    %2 = add <1 x i64> %1, %1
    %3 = call double @test_f64_v1i64_helper(<1 x i64> %2)
    br label %return_bb
return_bb:
    %4 = fadd double %3, %3
    store double %4, ptr %q
    ret void
}

; CHECK-LABEL: test_f64_v2f32:
declare double @test_f64_v2f32_helper(<2 x float> %p)
define void @test_f64_v2f32(ptr %p, ptr %q) {
; CHECK: rev64 v{{[0-9]+}}.2s
    %1 = load <2 x float>, ptr %p
    %2 = fadd <2 x float> %1, %1
    %3 = call double @test_f64_v2f32_helper(<2 x float> %2)
    br label %return_bb
return_bb:
    %4 = fadd double %3, %3
    store double %4, ptr %q
    ret void
}

; CHECK-LABEL: test_f64_v2i32:
declare double @test_f64_v2i32_helper(<2 x i32> %p)
define void @test_f64_v2i32(ptr %p, ptr %q) {
; CHECK: rev64 v{{[0-9]+}}.2s
    %1 = load <2 x i32>, ptr %p
    %2 = add <2 x i32> %1, %1
    %3 = call double @test_f64_v2i32_helper(<2 x i32> %2)
    br label %return_bb
return_bb:
    %4 = fadd double %3, %3
    store double %4, ptr %q
    ret void
}

; CHECK-LABEL: test_f64_v4i16:
declare double @test_f64_v4i16_helper(<4 x i16> %p)
define void @test_f64_v4i16(ptr %p, ptr %q) {
; CHECK: rev64 v{{[0-9]+}}.4h
    %1 = load <4 x i16>, ptr %p
    %2 = add <4 x i16> %1, %1
    %3 = call double @test_f64_v4i16_helper(<4 x i16> %2)
    br label %return_bb
return_bb:
    %4 = fadd double %3, %3
    store double %4, ptr %q
    ret void
}

; CHECK-LABEL: test_f64_v8i8:
declare double @test_f64_v8i8_helper(<8 x i8> %p)
define void @test_f64_v8i8(ptr %p, ptr %q) {
; CHECK: rev64 v{{[0-9]+}}.8b
    %1 = load <8 x i8>, ptr %p
    %2 = add <8 x i8> %1, %1
    %3 = call double @test_f64_v8i8_helper(<8 x i8> %2)
    br label %return_bb
return_bb:
    %4 = fadd double %3, %3
    store double %4, ptr %q
    ret void
}

; CHECK-LABEL: test_v1i64_i64:
declare <1 x i64> @test_v1i64_i64_helper(i64 %p)
define void @test_v1i64_i64(ptr %p, ptr %q) {
; CHECK-NOT: rev
    %1 = load i64, ptr %p
    %2 = add i64 %1, %1
    %3 = call <1 x i64> @test_v1i64_i64_helper(i64 %2)
    br label %return_bb
return_bb:
    %4 = add <1 x i64> %3, %3
    store <1 x i64> %4, ptr %q
    ret void
}

; CHECK-LABEL: test_v1i64_f64:
declare <1 x i64> @test_v1i64_f64_helper(double %p)
define void @test_v1i64_f64(ptr %p, ptr %q) {
; CHECK-NOT: rev
    %1 = load double, ptr %p
    %2 = fadd double %1, %1
    %3 = call <1 x i64> @test_v1i64_f64_helper(double %2)
    br label %return_bb
return_bb:
    %4 = add <1 x i64> %3, %3
    store <1 x i64> %4, ptr %q
    ret void
}

; CHECK-LABEL: test_v1i64_v2f32:
declare <1 x i64> @test_v1i64_v2f32_helper(<2 x float> %p)
define void @test_v1i64_v2f32(ptr %p, ptr %q) {
; CHECK: rev64 v{{[0-9]+}}.2s
    %1 = load <2 x float>, ptr %p
    %2 = fadd <2 x float> %1, %1
    %3 = call <1 x i64> @test_v1i64_v2f32_helper(<2 x float> %2)
    br label %return_bb
return_bb:
    %4 = add <1 x i64> %3, %3
    store <1 x i64> %4, ptr %q
    ret void
}

; CHECK-LABEL: test_v1i64_v2i32:
declare <1 x i64> @test_v1i64_v2i32_helper(<2 x i32> %p)
define void @test_v1i64_v2i32(ptr %p, ptr %q) {
; CHECK: rev64 v{{[0-9]+}}.2s
    %1 = load <2 x i32>, ptr %p
    %2 = add <2 x i32> %1, %1
    %3 = call <1 x i64> @test_v1i64_v2i32_helper(<2 x i32> %2)
    br label %return_bb
return_bb:
    %4 = add <1 x i64> %3, %3
    store <1 x i64> %4, ptr %q
    ret void
}

; CHECK-LABEL: test_v1i64_v4i16:
declare <1 x i64> @test_v1i64_v4i16_helper(<4 x i16> %p)
define void @test_v1i64_v4i16(ptr %p, ptr %q) {
; CHECK: rev64 v{{[0-9]+}}.4h
    %1 = load <4 x i16>, ptr %p
    %2 = add <4 x i16> %1, %1
    %3 = call <1 x i64> @test_v1i64_v4i16_helper(<4 x i16> %2)
    br label %return_bb
return_bb:
    %4 = add <1 x i64> %3, %3
    store <1 x i64> %4, ptr %q
    ret void
}

; CHECK-LABEL: test_v1i64_v8i8:
declare <1 x i64> @test_v1i64_v8i8_helper(<8 x i8> %p)
define void @test_v1i64_v8i8(ptr %p, ptr %q) {
; CHECK: rev64 v{{[0-9]+}}.8b
    %1 = load <8 x i8>, ptr %p
    %2 = add <8 x i8> %1, %1
    %3 = call <1 x i64> @test_v1i64_v8i8_helper(<8 x i8> %2)
    br label %return_bb
return_bb:
    %4 = add <1 x i64> %3, %3
    store <1 x i64> %4, ptr %q
    ret void
}

; CHECK-LABEL: test_v2f32_i64:
declare <2 x float> @test_v2f32_i64_helper(i64 %p)
define void @test_v2f32_i64(ptr %p, ptr %q) {
; CHECK: rev64 v{{[0-9]+}}.2s
    %1 = load i64, ptr %p
    %2 = add i64 %1, %1
    %3 = call <2 x float> @test_v2f32_i64_helper(i64 %2)
    br label %return_bb
return_bb:
    %4 = fadd <2 x float> %3, %3
    store <2 x float> %4, ptr %q
    ret void
}

; CHECK-LABEL: test_v2f32_f64:
declare <2 x float> @test_v2f32_f64_helper(double %p)
define void @test_v2f32_f64(ptr %p, ptr %q) {
; CHECK: rev64 v{{[0-9]+}}.2s
    %1 = load double, ptr %p
    %2 = fadd double %1, %1
    %3 = call <2 x float> @test_v2f32_f64_helper(double %2)
    br label %return_bb
return_bb:
    %4 = fadd <2 x float> %3, %3
    store <2 x float> %4, ptr %q
    ret void
}

; CHECK-LABEL: test_v2f32_v1i64:
declare <2 x float> @test_v2f32_v1i64_helper(<1 x i64> %p)
define void @test_v2f32_v1i64(ptr %p, ptr %q) {
; CHECK: rev64 v{{[0-9]+}}.2s
    %1 = load <1 x i64>, ptr %p
    %2 = add <1 x i64> %1, %1
    %3 = call <2 x float> @test_v2f32_v1i64_helper(<1 x i64> %2)
    br label %return_bb
return_bb:
    %4 = fadd <2 x float> %3, %3
    store <2 x float> %4, ptr %q
    ret void
}

; CHECK-LABEL: test_v2f32_v2i32:
declare <2 x float> @test_v2f32_v2i32_helper(<2 x i32> %p)
define void @test_v2f32_v2i32(ptr %p, ptr %q) {
; CHECK: rev64 v{{[0-9]+}}.2s
; CHECK: rev64 v{{[0-9]+}}.2s
    %1 = load <2 x i32>, ptr %p
    %2 = add <2 x i32> %1, %1
    %3 = call <2 x float> @test_v2f32_v2i32_helper(<2 x i32> %2)
    br label %return_bb
return_bb:
    %4 = fadd <2 x float> %3, %3
    store <2 x float> %4, ptr %q
    ret void
}

; CHECK-LABEL: test_v2f32_v4i16:
declare <2 x float> @test_v2f32_v4i16_helper(<4 x i16> %p)
define void @test_v2f32_v4i16(ptr %p, ptr %q) {
; CHECK: rev64 v{{[0-9]+}}.4h
; CHECK: rev64 v{{[0-9]+}}.2s
    %1 = load <4 x i16>, ptr %p
    %2 = add <4 x i16> %1, %1
    %3 = call <2 x float> @test_v2f32_v4i16_helper(<4 x i16> %2)
    br label %return_bb
return_bb:
    %4 = fadd <2 x float> %3, %3
    store <2 x float> %4, ptr %q
    ret void
}

; CHECK-LABEL: test_v2f32_v8i8:
declare <2 x float> @test_v2f32_v8i8_helper(<8 x i8> %p)
define void @test_v2f32_v8i8(ptr %p, ptr %q) {
; CHECK: rev64 v{{[0-9]+}}.8b
; CHECK: rev64 v{{[0-9]+}}.2s
    %1 = load <8 x i8>, ptr %p
    %2 = add <8 x i8> %1, %1
    %3 = call <2 x float> @test_v2f32_v8i8_helper(<8 x i8> %2)
    br label %return_bb
return_bb:
    %4 = fadd <2 x float> %3, %3
    store <2 x float> %4, ptr %q
    ret void
}

; CHECK-LABEL: test_v2i32_i64:
declare <2 x i32> @test_v2i32_i64_helper(i64 %p)
define void @test_v2i32_i64(ptr %p, ptr %q) {
; CHECK: rev64 v{{[0-9]+}}.2s
    %1 = load i64, ptr %p
    %2 = add i64 %1, %1
    %3 = call <2 x i32> @test_v2i32_i64_helper(i64 %2)
    br label %return_bb
return_bb:
    %4 = add <2 x i32> %3, %3
    store <2 x i32> %4, ptr %q
    ret void
}

; CHECK-LABEL: test_v2i32_f64:
declare <2 x i32> @test_v2i32_f64_helper(double %p)
define void @test_v2i32_f64(ptr %p, ptr %q) {
; CHECK: rev64 v{{[0-9]+}}.2s
    %1 = load double, ptr %p
    %2 = fadd double %1, %1
    %3 = call <2 x i32> @test_v2i32_f64_helper(double %2)
    br label %return_bb
return_bb:
    %4 = add <2 x i32> %3, %3
    store <2 x i32> %4, ptr %q
    ret void
}

; CHECK-LABEL: test_v2i32_v1i64:
declare <2 x i32> @test_v2i32_v1i64_helper(<1 x i64> %p)
define void @test_v2i32_v1i64(ptr %p, ptr %q) {
; CHECK: rev64 v{{[0-9]+}}.2s
    %1 = load <1 x i64>, ptr %p
    %2 = add <1 x i64> %1, %1
    %3 = call <2 x i32> @test_v2i32_v1i64_helper(<1 x i64> %2)
    br label %return_bb
return_bb:
    %4 = add <2 x i32> %3, %3
    store <2 x i32> %4, ptr %q
    ret void
}

; CHECK-LABEL: test_v2i32_v2f32:
declare <2 x i32> @test_v2i32_v2f32_helper(<2 x float> %p)
define void @test_v2i32_v2f32(ptr %p, ptr %q) {
; CHECK: rev64 v{{[0-9]+}}.2s
; CHECK: rev64 v{{[0-9]+}}.2s
    %1 = load <2 x float>, ptr %p
    %2 = fadd <2 x float> %1, %1
    %3 = call <2 x i32> @test_v2i32_v2f32_helper(<2 x float> %2)
    br label %return_bb
return_bb:
    %4 = add <2 x i32> %3, %3
    store <2 x i32> %4, ptr %q
    ret void
}

; CHECK-LABEL: test_v2i32_v4i16:
declare <2 x i32> @test_v2i32_v4i16_helper(<4 x i16> %p)
define void @test_v2i32_v4i16(ptr %p, ptr %q) {
; CHECK: rev64 v{{[0-9]+}}.4h
; CHECK: rev64 v{{[0-9]+}}.2s
    %1 = load <4 x i16>, ptr %p
    %2 = add <4 x i16> %1, %1
    %3 = call <2 x i32> @test_v2i32_v4i16_helper(<4 x i16> %2)
    br label %return_bb
return_bb:
    %4 = add <2 x i32> %3, %3
    store <2 x i32> %4, ptr %q
    ret void
}

; CHECK-LABEL: test_v2i32_v8i8:
declare <2 x i32> @test_v2i32_v8i8_helper(<8 x i8> %p)
define void @test_v2i32_v8i8(ptr %p, ptr %q) {
; CHECK: rev64 v{{[0-9]+}}.8b
; CHECK: rev64 v{{[0-9]+}}.2s
    %1 = load <8 x i8>, ptr %p
    %2 = add <8 x i8> %1, %1
    %3 = call <2 x i32> @test_v2i32_v8i8_helper(<8 x i8> %2)
    br label %return_bb
return_bb:
    %4 = add <2 x i32> %3, %3
    store <2 x i32> %4, ptr %q
    ret void
}

; CHECK-LABEL: test_v4i16_i64:
declare <4 x i16> @test_v4i16_i64_helper(i64 %p)
define void @test_v4i16_i64(ptr %p, ptr %q) {
; CHECK: rev64 v{{[0-9]+}}.4h
    %1 = load i64, ptr %p
    %2 = add i64 %1, %1
    %3 = call <4 x i16> @test_v4i16_i64_helper(i64 %2)
    br label %return_bb
return_bb:
    %4 = add <4 x i16> %3, %3
    store <4 x i16> %4, ptr %q
    ret void
}

; CHECK-LABEL: test_v4i16_f64:
declare <4 x i16> @test_v4i16_f64_helper(double %p)
define void @test_v4i16_f64(ptr %p, ptr %q) {
; CHECK: rev64 v{{[0-9]+}}.4h
    %1 = load double, ptr %p
    %2 = fadd double %1, %1
    %3 = call <4 x i16> @test_v4i16_f64_helper(double %2)
    br label %return_bb
return_bb:
    %4 = add <4 x i16> %3, %3
    store <4 x i16> %4, ptr %q
    ret void
}

; CHECK-LABEL: test_v4i16_v1i64:
declare <4 x i16> @test_v4i16_v1i64_helper(<1 x i64> %p)
define void @test_v4i16_v1i64(ptr %p, ptr %q) {
; CHECK: rev64 v{{[0-9]+}}.4h
    %1 = load <1 x i64>, ptr %p
    %2 = add <1 x i64> %1, %1
    %3 = call <4 x i16> @test_v4i16_v1i64_helper(<1 x i64> %2)
    br label %return_bb
return_bb:
    %4 = add <4 x i16> %3, %3
    store <4 x i16> %4, ptr %q
    ret void
}

; CHECK-LABEL: test_v4i16_v2f32:
declare <4 x i16> @test_v4i16_v2f32_helper(<2 x float> %p)
define void @test_v4i16_v2f32(ptr %p, ptr %q) {
; CHECK: rev64 v{{[0-9]+}}.2s
; CHECK: rev64 v{{[0-9]+}}.4h
    %1 = load <2 x float>, ptr %p
    %2 = fadd <2 x float> %1, %1
    %3 = call <4 x i16> @test_v4i16_v2f32_helper(<2 x float> %2)
    br label %return_bb
return_bb:
    %4 = add <4 x i16> %3, %3
    store <4 x i16> %4, ptr %q
    ret void
}

; CHECK-LABEL: test_v4i16_v2i32:
declare <4 x i16> @test_v4i16_v2i32_helper(<2 x i32> %p)
define void @test_v4i16_v2i32(ptr %p, ptr %q) {
; CHECK: rev64 v{{[0-9]+}}.2s
; CHECK: rev64 v{{[0-9]+}}.4h
    %1 = load <2 x i32>, ptr %p
    %2 = add <2 x i32> %1, %1
    %3 = call <4 x i16> @test_v4i16_v2i32_helper(<2 x i32> %2)
    br label %return_bb
return_bb:
    %4 = add <4 x i16> %3, %3
    store <4 x i16> %4, ptr %q
    ret void
}

; CHECK-LABEL: test_v4i16_v8i8:
declare <4 x i16> @test_v4i16_v8i8_helper(<8 x i8> %p)
define void @test_v4i16_v8i8(ptr %p, ptr %q) {
; CHECK: rev64 v{{[0-9]+}}.8b
; CHECK: rev64 v{{[0-9]+}}.4h
    %1 = load <8 x i8>, ptr %p
    %2 = add <8 x i8> %1, %1
    %3 = call <4 x i16> @test_v4i16_v8i8_helper(<8 x i8> %2)
    br label %return_bb
return_bb:
    %4 = add <4 x i16> %3, %3
    store <4 x i16> %4, ptr %q
    ret void
}

; CHECK-LABEL: test_v8i8_i64:
declare <8 x i8> @test_v8i8_i64_helper(i64 %p)
define void @test_v8i8_i64(ptr %p, ptr %q) {
; CHECK: rev64 v{{[0-9]+}}.8b
    %1 = load i64, ptr %p
    %2 = add i64 %1, %1
    %3 = call <8 x i8> @test_v8i8_i64_helper(i64 %2)
    br label %return_bb
return_bb:
    %4 = add <8 x i8> %3, %3
    store <8 x i8> %4, ptr %q
    ret void
}

; CHECK-LABEL: test_v8i8_f64:
declare <8 x i8> @test_v8i8_f64_helper(double %p)
define void @test_v8i8_f64(ptr %p, ptr %q) {
; CHECK: rev64 v{{[0-9]+}}.8b
    %1 = load double, ptr %p
    %2 = fadd double %1, %1
    %3 = call <8 x i8> @test_v8i8_f64_helper(double %2)
    br label %return_bb
return_bb:
    %4 = add <8 x i8> %3, %3
    store <8 x i8> %4, ptr %q
    ret void
}

; CHECK-LABEL: test_v8i8_v1i64:
declare <8 x i8> @test_v8i8_v1i64_helper(<1 x i64> %p)
define void @test_v8i8_v1i64(ptr %p, ptr %q) {
; CHECK: rev64 v{{[0-9]+}}.8b
    %1 = load <1 x i64>, ptr %p
    %2 = add <1 x i64> %1, %1
    %3 = call <8 x i8> @test_v8i8_v1i64_helper(<1 x i64> %2)
    br label %return_bb
return_bb:
    %4 = add <8 x i8> %3, %3
    store <8 x i8> %4, ptr %q
    ret void
}

; CHECK-LABEL: test_v8i8_v2f32:
declare <8 x i8> @test_v8i8_v2f32_helper(<2 x float> %p)
define void @test_v8i8_v2f32(ptr %p, ptr %q) {
; CHECK: rev64 v{{[0-9]+}}.2s
; CHECK: rev64 v{{[0-9]+}}.8b
    %1 = load <2 x float>, ptr %p
    %2 = fadd <2 x float> %1, %1
    %3 = call <8 x i8> @test_v8i8_v2f32_helper(<2 x float> %2)
    br label %return_bb
return_bb:
    %4 = add <8 x i8> %3, %3
    store <8 x i8> %4, ptr %q
    ret void
}

; CHECK-LABEL: test_v8i8_v2i32:
declare <8 x i8> @test_v8i8_v2i32_helper(<2 x i32> %p)
define void @test_v8i8_v2i32(ptr %p, ptr %q) {
; CHECK: rev64 v{{[0-9]+}}.2s
; CHECK: rev64 v{{[0-9]+}}.8b
    %1 = load <2 x i32>, ptr %p
    %2 = add <2 x i32> %1, %1
    %3 = call <8 x i8> @test_v8i8_v2i32_helper(<2 x i32> %2)
    br label %return_bb
return_bb:
    %4 = add <8 x i8> %3, %3
    store <8 x i8> %4, ptr %q
    ret void
}

; CHECK-LABEL: test_v8i8_v4i16:
declare <8 x i8> @test_v8i8_v4i16_helper(<4 x i16> %p)
define void @test_v8i8_v4i16(ptr %p, ptr %q) {
; CHECK: rev64 v{{[0-9]+}}.4h
; CHECK: rev64 v{{[0-9]+}}.8b
    %1 = load <4 x i16>, ptr %p
    %2 = add <4 x i16> %1, %1
    %3 = call <8 x i8> @test_v8i8_v4i16_helper(<4 x i16> %2)
    br label %return_bb
return_bb:
    %4 = add <8 x i8> %3, %3
    store <8 x i8> %4, ptr %q
    ret void
}

; CHECK-LABEL: test_f128_v2f64:
declare fp128 @test_f128_v2f64_helper(<2 x double> %p)
define void @test_f128_v2f64(ptr %p, ptr %q) {
; CHECK: ext
    %1 = load <2 x double>, ptr %p
    %2 = fadd <2 x double> %1, %1
    %3 = call fp128 @test_f128_v2f64_helper(<2 x double> %2)
    br label %return_bb
return_bb:
    %4 = fadd fp128 %3, %3
    store fp128 %4, ptr %q
    ret void
}

; CHECK-LABEL: test_f128_v2i64:
declare fp128 @test_f128_v2i64_helper(<2 x i64> %p)
define void @test_f128_v2i64(ptr %p, ptr %q) {
; CHECK: ext
    %1 = load <2 x i64>, ptr %p
    %2 = add <2 x i64> %1, %1
    %3 = call fp128 @test_f128_v2i64_helper(<2 x i64> %2)
    br label %return_bb
return_bb:
    %4 = fadd fp128 %3, %3
    store fp128 %4, ptr %q
    ret void
}

; CHECK-LABEL: test_f128_v4f32:
declare fp128 @test_f128_v4f32_helper(<4 x float> %p)
define void @test_f128_v4f32(ptr %p, ptr %q) {
; CHECK: rev64 v{{[0-9]+}}.4s
; CHECK: ext
    %1 = load <4 x float>, ptr %p
    %2 = fadd <4 x float> %1, %1
    %3 = call fp128 @test_f128_v4f32_helper(<4 x float> %2)
    br label %return_bb
return_bb:
    %4 = fadd fp128 %3, %3
    store fp128 %4, ptr %q
    ret void
}

; CHECK-LABEL: test_f128_v4i32:
declare fp128 @test_f128_v4i32_helper(<4 x i32> %p)
define void @test_f128_v4i32(ptr %p, ptr %q) {
; CHECK: rev64 v{{[0-9]+}}.4s
; CHECK: ext
    %1 = load <4 x i32>, ptr %p
    %2 = add <4 x i32> %1, %1
    %3 = call fp128 @test_f128_v4i32_helper(<4 x i32> %2)
    br label %return_bb
return_bb:
    %4 = fadd fp128 %3, %3
    store fp128 %4, ptr %q
    ret void
}

; CHECK-LABEL: test_f128_v8i16:
declare fp128 @test_f128_v8i16_helper(<8 x i16> %p)
define void @test_f128_v8i16(ptr %p, ptr %q) {
; CHECK: rev64 v{{[0-9]+}}.8h
; CHECK: ext
    %1 = load <8 x i16>, ptr %p
    %2 = add <8 x i16> %1, %1
    %3 = call fp128 @test_f128_v8i16_helper(<8 x i16> %2)
    br label %return_bb
return_bb:
    %4 = fadd fp128 %3, %3
    store fp128 %4, ptr %q
    ret void
}

; CHECK-LABEL: test_f128_v16i8:
declare fp128 @test_f128_v16i8_helper(<16 x i8> %p)
define void @test_f128_v16i8(ptr %p, ptr %q) {
; CHECK: rev64 v{{[0-9]+}}.16b
; CHECK: ext
    %1 = load <16 x i8>, ptr %p
    %2 = add <16 x i8> %1, %1
    %3 = call fp128 @test_f128_v16i8_helper(<16 x i8> %2)
    br label %return_bb
return_bb:
    %4 = fadd fp128 %3, %3
    store fp128 %4, ptr %q
    ret void
}

; CHECK-LABEL: test_v2f64_f128:
declare <2 x double> @test_v2f64_f128_helper(fp128 %p)
define void @test_v2f64_f128(ptr %p, ptr %q) {
; CHECK: ext
    %1 = load fp128, ptr %p
    %2 = fadd fp128 %1, %1
    %3 = call <2 x double> @test_v2f64_f128_helper(fp128 %2)
    br label %return_bb
return_bb:
    %4 = fadd <2 x double> %3, %3
    store <2 x double> %4, ptr %q
    ret void
}

; CHECK-LABEL: test_v2f64_v2i64:
declare <2 x double> @test_v2f64_v2i64_helper(<2 x i64> %p)
define void @test_v2f64_v2i64(ptr %p, ptr %q) {
; CHECK: ext
; CHECK: ext
    %1 = load <2 x i64>, ptr %p
    %2 = add <2 x i64> %1, %1
    %3 = call <2 x double> @test_v2f64_v2i64_helper(<2 x i64> %2)
    br label %return_bb
return_bb:
    %4 = fadd <2 x double> %3, %3
    store <2 x double> %4, ptr %q
    ret void
}

; CHECK-LABEL: test_v2f64_v4f32:
declare <2 x double> @test_v2f64_v4f32_helper(<4 x float> %p)
define void @test_v2f64_v4f32(ptr %p, ptr %q) {
; CHECK: rev64 v{{[0-9]+}}.4s
; CHECK: ext
; CHECK: ext
    %1 = load <4 x float>, ptr %p
    %2 = fadd <4 x float> %1, %1
    %3 = call <2 x double> @test_v2f64_v4f32_helper(<4 x float> %2)
    br label %return_bb
return_bb:
    %4 = fadd <2 x double> %3, %3
    store <2 x double> %4, ptr %q
    ret void
}

; CHECK-LABEL: test_v2f64_v4i32:
declare <2 x double> @test_v2f64_v4i32_helper(<4 x i32> %p)
define void @test_v2f64_v4i32(ptr %p, ptr %q) {
; CHECK: rev64 v{{[0-9]+}}.4s
; CHECK: ext
; CHECK: ext
    %1 = load <4 x i32>, ptr %p
    %2 = add <4 x i32> %1, %1
    %3 = call <2 x double> @test_v2f64_v4i32_helper(<4 x i32> %2)
    br label %return_bb
return_bb:
    %4 = fadd <2 x double> %3, %3
    store <2 x double> %4, ptr %q
    ret void
}

; CHECK-LABEL: test_v2f64_v8i16:
declare <2 x double> @test_v2f64_v8i16_helper(<8 x i16> %p)
define void @test_v2f64_v8i16(ptr %p, ptr %q) {
; CHECK: rev64 v{{[0-9]+}}.8h
; CHECK: ext
; CHECK: ext
    %1 = load <8 x i16>, ptr %p
    %2 = add <8 x i16> %1, %1
    %3 = call <2 x double> @test_v2f64_v8i16_helper(<8 x i16> %2)
    br label %return_bb
return_bb:
    %4 = fadd <2 x double> %3, %3
    store <2 x double> %4, ptr %q
    ret void
}

; CHECK-LABEL: test_v2f64_v16i8:
declare <2 x double> @test_v2f64_v16i8_helper(<16 x i8> %p)
define void @test_v2f64_v16i8(ptr %p, ptr %q) {
; CHECK: rev64 v{{[0-9]+}}.16b
; CHECK: ext
; CHECK: ext
    %1 = load <16 x i8>, ptr %p
    %2 = add <16 x i8> %1, %1
    %3 = call <2 x double> @test_v2f64_v16i8_helper(<16 x i8> %2)
    br label %return_bb
return_bb:
    %4 = fadd <2 x double> %3, %3
    store <2 x double> %4, ptr %q
    ret void
}

; CHECK-LABEL: test_v2i64_f128:
declare <2 x i64> @test_v2i64_f128_helper(fp128 %p)
define void @test_v2i64_f128(ptr %p, ptr %q) {
; CHECK: ext
    %1 = load fp128, ptr %p
    %2 = fadd fp128 %1, %1
    %3 = call <2 x i64> @test_v2i64_f128_helper(fp128 %2)
    br label %return_bb
return_bb:
    %4 = add <2 x i64> %3, %3
    store <2 x i64> %4, ptr %q
    ret void
}

; CHECK-LABEL: test_v2i64_v2f64:
declare <2 x i64> @test_v2i64_v2f64_helper(<2 x double> %p)
define void @test_v2i64_v2f64(ptr %p, ptr %q) {
; CHECK: ext
; CHECK: ext
    %1 = load <2 x double>, ptr %p
    %2 = fadd <2 x double> %1, %1
    %3 = call <2 x i64> @test_v2i64_v2f64_helper(<2 x double> %2)
    br label %return_bb
return_bb:
    %4 = add <2 x i64> %3, %3
    store <2 x i64> %4, ptr %q
    ret void
}

; CHECK-LABEL: test_v2i64_v4f32:
declare <2 x i64> @test_v2i64_v4f32_helper(<4 x float> %p)
define void @test_v2i64_v4f32(ptr %p, ptr %q) {
; CHECK: rev64 v{{[0-9]+}}.4s
; CHECK: ext
; CHECK: ext
    %1 = load <4 x float>, ptr %p
    %2 = fadd <4 x float> %1, %1
    %3 = call <2 x i64> @test_v2i64_v4f32_helper(<4 x float> %2)
    br label %return_bb
return_bb:
    %4 = add <2 x i64> %3, %3
    store <2 x i64> %4, ptr %q
    ret void
}

; CHECK-LABEL: test_v2i64_v4i32:
declare <2 x i64> @test_v2i64_v4i32_helper(<4 x i32> %p)
define void @test_v2i64_v4i32(ptr %p, ptr %q) {
; CHECK: rev64 v{{[0-9]+}}.4s
; CHECK: ext
; CHECK: ext
    %1 = load <4 x i32>, ptr %p
    %2 = add <4 x i32> %1, %1
    %3 = call <2 x i64> @test_v2i64_v4i32_helper(<4 x i32> %2)
    br label %return_bb
return_bb:
    %4 = add <2 x i64> %3, %3
    store <2 x i64> %4, ptr %q
    ret void
}

; CHECK-LABEL: test_v2i64_v8i16:
declare <2 x i64> @test_v2i64_v8i16_helper(<8 x i16> %p)
define void @test_v2i64_v8i16(ptr %p, ptr %q) {
; CHECK: rev64 v{{[0-9]+}}.8h
; CHECK: ext
; CHECK: ext
    %1 = load <8 x i16>, ptr %p
    %2 = add <8 x i16> %1, %1
    %3 = call <2 x i64> @test_v2i64_v8i16_helper(<8 x i16> %2)
    br label %return_bb
return_bb:
    %4 = add <2 x i64> %3, %3
    store <2 x i64> %4, ptr %q
    ret void
}

; CHECK-LABEL: test_v2i64_v16i8:
declare <2 x i64> @test_v2i64_v16i8_helper(<16 x i8> %p)
define void @test_v2i64_v16i8(ptr %p, ptr %q) {
; CHECK: rev64 v{{[0-9]+}}.16b
; CHECK: ext
; CHECK: ext
    %1 = load <16 x i8>, ptr %p
    %2 = add <16 x i8> %1, %1
    %3 = call <2 x i64> @test_v2i64_v16i8_helper(<16 x i8> %2)
    br label %return_bb
return_bb:
    %4 = add <2 x i64> %3, %3
    store <2 x i64> %4, ptr %q
    ret void
}

; CHECK-LABEL: test_v4f32_f128:
declare <4 x float> @test_v4f32_f128_helper(fp128 %p)
define void @test_v4f32_f128(ptr %p, ptr %q) {
; CHECK: rev64 v{{[0-9]+}}.4s
; CHECK: ext
    %1 = load fp128, ptr %p
    %2 = fadd fp128 %1, %1
    %3 = call <4 x float> @test_v4f32_f128_helper(fp128 %2)
    br label %return_bb
return_bb:
    %4 = fadd <4 x float> %3, %3
    store <4 x float> %4, ptr %q
    ret void
}

; CHECK-LABEL: test_v4f32_v2f64:
declare <4 x float> @test_v4f32_v2f64_helper(<2 x double> %p)
define void @test_v4f32_v2f64(ptr %p, ptr %q) {
; CHECK: ext
; CHECK: rev64 v{{[0-9]+}}.4s
; CHECK: ext
    %1 = load <2 x double>, ptr %p
    %2 = fadd <2 x double> %1, %1
    %3 = call <4 x float> @test_v4f32_v2f64_helper(<2 x double> %2)
    br label %return_bb
return_bb:
    %4 = fadd <4 x float> %3, %3
    store <4 x float> %4, ptr %q
    ret void
}

; CHECK-LABEL: test_v4f32_v2i64:
declare <4 x float> @test_v4f32_v2i64_helper(<2 x i64> %p)
define void @test_v4f32_v2i64(ptr %p, ptr %q) {
; CHECK: ext
; CHECK: rev64 v{{[0-9]+}}.4s
; CHECK: ext
    %1 = load <2 x i64>, ptr %p
    %2 = add <2 x i64> %1, %1
    %3 = call <4 x float> @test_v4f32_v2i64_helper(<2 x i64> %2)
    br label %return_bb
return_bb:
    %4 = fadd <4 x float> %3, %3
    store <4 x float> %4, ptr %q
    ret void
}

; CHECK-LABEL: test_v4f32_v4i32:
declare <4 x float> @test_v4f32_v4i32_helper(<4 x i32> %p)
define void @test_v4f32_v4i32(ptr %p, ptr %q) {
; CHECK: rev64 v{{[0-9]+}}.4s
; CHECK: ext
; CHECK: rev64 v{{[0-9]+}}.4s
; CHECK: ext
    %1 = load <4 x i32>, ptr %p
    %2 = add <4 x i32> %1, %1
    %3 = call <4 x float> @test_v4f32_v4i32_helper(<4 x i32> %2)
    br label %return_bb
return_bb:
    %4 = fadd <4 x float> %3, %3
    store <4 x float> %4, ptr %q
    ret void
}

; CHECK-LABEL: test_v4f32_v8i16:
declare <4 x float> @test_v4f32_v8i16_helper(<8 x i16> %p)
define void @test_v4f32_v8i16(ptr %p, ptr %q) {
; CHECK: rev64 v{{[0-9]+}}.8h
; CHECK: ext
; CHECK: rev64 v{{[0-9]+}}.4s
; CHECK: ext
    %1 = load <8 x i16>, ptr %p
    %2 = add <8 x i16> %1, %1
    %3 = call <4 x float> @test_v4f32_v8i16_helper(<8 x i16> %2)
    br label %return_bb
return_bb:
    %4 = fadd <4 x float> %3, %3
    store <4 x float> %4, ptr %q
    ret void
}

; CHECK-LABEL: test_v4f32_v16i8:
declare <4 x float> @test_v4f32_v16i8_helper(<16 x i8> %p)
define void @test_v4f32_v16i8(ptr %p, ptr %q) {
; CHECK: rev64 v{{[0-9]+}}.16b
; CHECK: ext
; CHECK: rev64 v{{[0-9]+}}.4s
; CHECK: ext
    %1 = load <16 x i8>, ptr %p
    %2 = add <16 x i8> %1, %1
    %3 = call <4 x float> @test_v4f32_v16i8_helper(<16 x i8> %2)
    br label %return_bb
return_bb:
    %4 = fadd <4 x float> %3, %3
    store <4 x float> %4, ptr %q
    ret void
}

; CHECK-LABEL: test_v4i32_f128:
declare <4 x i32> @test_v4i32_f128_helper(fp128 %p)
define void @test_v4i32_f128(ptr %p, ptr %q) {
; CHECK: rev64 v{{[0-9]+}}.4s
; CHECK: ext
    %1 = load fp128, ptr %p
    %2 = fadd fp128 %1, %1
    %3 = call <4 x i32> @test_v4i32_f128_helper(fp128 %2)
    br label %return_bb
return_bb:
    %4 = add <4 x i32> %3, %3
    store <4 x i32> %4, ptr %q
    ret void
}

; CHECK-LABEL: test_v4i32_v2f64:
declare <4 x i32> @test_v4i32_v2f64_helper(<2 x double> %p)
define void @test_v4i32_v2f64(ptr %p, ptr %q) {
; CHECK: ext
; CHECK: rev64 v{{[0-9]+}}.4s
; CHECK: ext
    %1 = load <2 x double>, ptr %p
    %2 = fadd <2 x double> %1, %1
    %3 = call <4 x i32> @test_v4i32_v2f64_helper(<2 x double> %2)
    br label %return_bb
return_bb:
    %4 = add <4 x i32> %3, %3
    store <4 x i32> %4, ptr %q
    ret void
}

; CHECK-LABEL: test_v4i32_v2i64:
declare <4 x i32> @test_v4i32_v2i64_helper(<2 x i64> %p)
define void @test_v4i32_v2i64(ptr %p, ptr %q) {
; CHECK: ext
; CHECK: rev64 v{{[0-9]+}}.4s
; CHECK: ext
    %1 = load <2 x i64>, ptr %p
    %2 = add <2 x i64> %1, %1
    %3 = call <4 x i32> @test_v4i32_v2i64_helper(<2 x i64> %2)
    br label %return_bb
return_bb:
    %4 = add <4 x i32> %3, %3
    store <4 x i32> %4, ptr %q
    ret void
}

; CHECK-LABEL: test_v4i32_v4f32:
declare <4 x i32> @test_v4i32_v4f32_helper(<4 x float> %p)
define void @test_v4i32_v4f32(ptr %p, ptr %q) {
; CHECK: rev64 v{{[0-9]+}}.4s
; CHECK: ext
; CHECK: rev64 v{{[0-9]+}}.4s
; CHECK: ext
    %1 = load <4 x float>, ptr %p
    %2 = fadd <4 x float> %1, %1
    %3 = call <4 x i32> @test_v4i32_v4f32_helper(<4 x float> %2)
    br label %return_bb
return_bb:
    %4 = add <4 x i32> %3, %3
    store <4 x i32> %4, ptr %q
    ret void
}

; CHECK-LABEL: test_v4i32_v8i16:
declare <4 x i32> @test_v4i32_v8i16_helper(<8 x i16> %p)
define void @test_v4i32_v8i16(ptr %p, ptr %q) {
; CHECK: rev64 v{{[0-9]+}}.8h
; CHECK: ext
; CHECK: rev64 v{{[0-9]+}}.4s
; CHECK: ext
    %1 = load <8 x i16>, ptr %p
    %2 = add <8 x i16> %1, %1
    %3 = call <4 x i32> @test_v4i32_v8i16_helper(<8 x i16> %2)
    br label %return_bb
return_bb:
    %4 = add <4 x i32> %3, %3
    store <4 x i32> %4, ptr %q
    ret void
}

; CHECK-LABEL: test_v4i32_v16i8:
declare <4 x i32> @test_v4i32_v16i8_helper(<16 x i8> %p)
define void @test_v4i32_v16i8(ptr %p, ptr %q) {
; CHECK: rev64 v{{[0-9]+}}.16b
; CHECK: ext
; CHECK: rev64 v{{[0-9]+}}.4s
; CHECK: ext
    %1 = load <16 x i8>, ptr %p
    %2 = add <16 x i8> %1, %1
    %3 = call <4 x i32> @test_v4i32_v16i8_helper(<16 x i8> %2)
    br label %return_bb
return_bb:
    %4 = add <4 x i32> %3, %3
    store <4 x i32> %4, ptr %q
    ret void
}

; CHECK-LABEL: test_v8i16_f128:
declare <8 x i16> @test_v8i16_f128_helper(fp128 %p)
define void @test_v8i16_f128(ptr %p, ptr %q) {
; CHECK: rev64 v{{[0-9]+}}.8h
; CHECK: ext
    %1 = load fp128, ptr %p
    %2 = fadd fp128 %1, %1
    %3 = call <8 x i16> @test_v8i16_f128_helper(fp128 %2)
    br label %return_bb
return_bb:
    %4 = add <8 x i16> %3, %3
    store <8 x i16> %4, ptr %q
    ret void
}

; CHECK-LABEL: test_v8i16_v2f64:
declare <8 x i16> @test_v8i16_v2f64_helper(<2 x double> %p)
define void @test_v8i16_v2f64(ptr %p, ptr %q) {
; CHECK: ext
; CHECK: rev64 v{{[0-9]+}}.8h
; CHECK: ext
    %1 = load <2 x double>, ptr %p
    %2 = fadd <2 x double> %1, %1
    %3 = call <8 x i16> @test_v8i16_v2f64_helper(<2 x double> %2)
    br label %return_bb
return_bb:
    %4 = add <8 x i16> %3, %3
    store <8 x i16> %4, ptr %q
    ret void
}

; CHECK-LABEL: test_v8i16_v2i64:
declare <8 x i16> @test_v8i16_v2i64_helper(<2 x i64> %p)
define void @test_v8i16_v2i64(ptr %p, ptr %q) {
; CHECK: ext
; CHECK: rev64 v{{[0-9]+}}.8h
; CHECK: ext
    %1 = load <2 x i64>, ptr %p
    %2 = add <2 x i64> %1, %1
    %3 = call <8 x i16> @test_v8i16_v2i64_helper(<2 x i64> %2)
    br label %return_bb
return_bb:
    %4 = add <8 x i16> %3, %3
    store <8 x i16> %4, ptr %q
    ret void
}

; CHECK-LABEL: test_v8i16_v4f32:
declare <8 x i16> @test_v8i16_v4f32_helper(<4 x float> %p)
define void @test_v8i16_v4f32(ptr %p, ptr %q) {
; CHECK: rev64 v{{[0-9]+}}.4s
; CHECK: ext
; CHECK: rev64 v{{[0-9]+}}.8h
; CHECK: ext
    %1 = load <4 x float>, ptr %p
    %2 = fadd <4 x float> %1, %1
    %3 = call <8 x i16> @test_v8i16_v4f32_helper(<4 x float> %2)
    br label %return_bb
return_bb:
    %4 = add <8 x i16> %3, %3
    store <8 x i16> %4, ptr %q
    ret void
}

; CHECK-LABEL: test_v8i16_v4i32:
declare <8 x i16> @test_v8i16_v4i32_helper(<4 x i32> %p)
define void @test_v8i16_v4i32(ptr %p, ptr %q) {
; CHECK: rev64 v{{[0-9]+}}.4s
; CHECK: ext
; CHECK: rev64 v{{[0-9]+}}.8h
; CHECK: ext
    %1 = load <4 x i32>, ptr %p
    %2 = add <4 x i32> %1, %1
    %3 = call <8 x i16> @test_v8i16_v4i32_helper(<4 x i32> %2)
    br label %return_bb
return_bb:
    %4 = add <8 x i16> %3, %3
    store <8 x i16> %4, ptr %q
    ret void
}

; CHECK-LABEL: test_v8i16_v16i8:
declare <8 x i16> @test_v8i16_v16i8_helper(<16 x i8> %p)
define void @test_v8i16_v16i8(ptr %p, ptr %q) {
; CHECK: rev64 v{{[0-9]+}}.16b
; CHECK: ext
; CHECK: rev64 v{{[0-9]+}}.8h
; CHECK: ext
    %1 = load <16 x i8>, ptr %p
    %2 = add <16 x i8> %1, %1
    %3 = call <8 x i16> @test_v8i16_v16i8_helper(<16 x i8> %2)
    br label %return_bb
return_bb:
    %4 = add <8 x i16> %3, %3
    store <8 x i16> %4, ptr %q
    ret void
}

; CHECK-LABEL: test_v16i8_f128:
declare <16 x i8> @test_v16i8_f128_helper(fp128 %p)
define void @test_v16i8_f128(ptr %p, ptr %q) {
; CHECK: rev64 v{{[0-9]+}}.16b
; CHECK: ext
    %1 = load fp128, ptr %p
    %2 = fadd fp128 %1, %1
    %3 = call <16 x i8> @test_v16i8_f128_helper(fp128 %2)
    br label %return_bb
return_bb:
    %4 = add <16 x i8> %3, %3
    store <16 x i8> %4, ptr %q
    ret void
}

; CHECK-LABEL: test_v16i8_v2f64:
declare <16 x i8> @test_v16i8_v2f64_helper(<2 x double> %p)
define void @test_v16i8_v2f64(ptr %p, ptr %q) {
; CHECK: ext
; CHECK: rev64 v{{[0-9]+}}.16b
; CHECK: ext
    %1 = load <2 x double>, ptr %p
    %2 = fadd <2 x double> %1, %1
    %3 = call <16 x i8> @test_v16i8_v2f64_helper(<2 x double> %2)
    br label %return_bb
return_bb:
    %4 = add <16 x i8> %3, %3
    store <16 x i8> %4, ptr %q
    ret void
}

; CHECK-LABEL: test_v16i8_v2i64:
declare <16 x i8> @test_v16i8_v2i64_helper(<2 x i64> %p)
define void @test_v16i8_v2i64(ptr %p, ptr %q) {
; CHECK: ext
; CHECK: rev64 v{{[0-9]+}}.16b
; CHECK: ext
    %1 = load <2 x i64>, ptr %p
    %2 = add <2 x i64> %1, %1
    %3 = call <16 x i8> @test_v16i8_v2i64_helper(<2 x i64> %2)
    br label %return_bb
return_bb:
    %4 = add <16 x i8> %3, %3
    store <16 x i8> %4, ptr %q
    ret void
}

; CHECK-LABEL: test_v16i8_v4f32:
declare <16 x i8> @test_v16i8_v4f32_helper(<4 x float> %p)
define void @test_v16i8_v4f32(ptr %p, ptr %q) {
; CHECK: rev64 v{{[0-9]+}}.4s
; CHECK: ext
; CHECK: rev64 v{{[0-9]+}}.16b
; CHECK: ext
    %1 = load <4 x float>, ptr %p
    %2 = fadd <4 x float> %1, %1
    %3 = call <16 x i8> @test_v16i8_v4f32_helper(<4 x float> %2)
    br label %return_bb
return_bb:
    %4 = add <16 x i8> %3, %3
    store <16 x i8> %4, ptr %q
    ret void
}

; CHECK-LABEL: test_v16i8_v4i32:
declare <16 x i8> @test_v16i8_v4i32_helper(<4 x i32> %p)
define void @test_v16i8_v4i32(ptr %p, ptr %q) {
; CHECK: rev64 v{{[0-9]+}}.4s
; CHECK: ext
; CHECK: rev64 v{{[0-9]+}}.16b
; CHECK: ext
    %1 = load <4 x i32>, ptr %p
    %2 = add <4 x i32> %1, %1
    %3 = call <16 x i8> @test_v16i8_v4i32_helper(<4 x i32> %2)
    br label %return_bb
return_bb:
    %4 = add <16 x i8> %3, %3
    store <16 x i8> %4, ptr %q
    ret void
}

; CHECK-LABEL: test_v16i8_v8i16:
declare <16 x i8> @test_v16i8_v8i16_helper(<8 x i16> %p)
define void @test_v16i8_v8i16(ptr %p, ptr %q) {
; CHECK: rev64 v{{[0-9]+}}.8h
; CHECK: ext
; CHECK: rev64 v{{[0-9]+}}.16b
; CHECK: ext
    %1 = load <8 x i16>, ptr %p
    %2 = add <8 x i16> %1, %1
    %3 = call <16 x i8> @test_v16i8_v8i16_helper(<8 x i16> %2)
    br label %return_bb
return_bb:
    %4 = add <16 x i8> %3, %3
    store <16 x i8> %4, ptr %q
    ret void
}
