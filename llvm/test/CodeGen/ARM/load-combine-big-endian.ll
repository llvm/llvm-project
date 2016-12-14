; RUN: llc < %s -mtriple=armeb-unknown | FileCheck %s
; RUN: llc < %s -mtriple=arm64eb-unknown | FileCheck %s --check-prefix=CHECK64

; i8* p; // p is 4 byte aligned
; ((i32) p[0] << 24) | ((i32) p[1] << 16) | ((i32) p[2] << 8) | (i32) p[3]
define i32 @load_i32_by_i8_big_endian(i32*) {
; CHECK-LABEL: load_i32_by_i8_big_endian:
; CHECK: ldr r0, [r0]
; CHECK-NEXT: mov pc, lr

; CHECK64-LABEL: load_i32_by_i8_big_endian:
; CHECK64: ldr		w0, [x0]
; CHECK64-NEXT: ret
  %2 = bitcast i32* %0 to i8*
  %3 = load i8, i8* %2, align 4
  %4 = zext i8 %3 to i32
  %5 = shl nuw nsw i32 %4, 24
  %6 = getelementptr inbounds i8, i8* %2, i32 1
  %7 = load i8, i8* %6, align 1
  %8 = zext i8 %7 to i32
  %9 = shl nuw nsw i32 %8, 16
  %10 = or i32 %9, %5
  %11 = getelementptr inbounds i8, i8* %2, i32 2
  %12 = load i8, i8* %11, align 1
  %13 = zext i8 %12 to i32
  %14 = shl nuw nsw i32 %13, 8
  %15 = or i32 %10, %14
  %16 = getelementptr inbounds i8, i8* %2, i32 3
  %17 = load i8, i8* %16, align 1
  %18 = zext i8 %17 to i32
  %19 = or i32 %15, %18
  ret i32 %19
}

; i8* p; // p is 4 byte aligned
; ((i32) (((i16) p[0] << 8) | (i16) p[1]) << 16) | (i32) (((i16) p[3] << 8) | (i16) p[4])
define i32 @load_i32_by_i16_by_i8_big_endian(i32*) {
; CHECK-LABEL: load_i32_by_i16_by_i8_big_endian:
; CHECK: ldr r0, [r0]
; CHECK-NEXT: mov pc, lr

; CHECK64-LABEL: load_i32_by_i16_by_i8_big_endian:
; CHECK64: ldr		w0, [x0]
; CHECK64-NEXT: ret
  %2 = bitcast i32* %0 to i8*
  %3 = load i8, i8* %2, align 4
  %4 = zext i8 %3 to i16
  %5 = getelementptr inbounds i8, i8* %2, i32 1
  %6 = load i8, i8* %5, align 1
  %7 = zext i8 %6 to i16
  %8 = shl nuw nsw i16 %4, 8
  %9 = or i16 %8, %7
  %10 = getelementptr inbounds i8, i8* %2, i32 2
  %11 = load i8, i8* %10, align 1
  %12 = zext i8 %11 to i16
  %13 = getelementptr inbounds i8, i8* %2, i32 3
  %14 = load i8, i8* %13, align 1
  %15 = zext i8 %14 to i16
  %16 = shl nuw nsw i16 %12, 8
  %17 = or i16 %16, %15
  %18 = zext i16 %9 to i32
  %19 = zext i16 %17 to i32
  %20 = shl nuw nsw i32 %18, 16
  %21 = or i32 %20, %19
  ret i32 %21
}

; i16* p; // p is 4 byte aligned
; ((i32) p[0] << 16) | (i32) p[1]
define i32 @load_i32_by_i16(i32*) {
; CHECK-LABEL: load_i32_by_i16:
; CHECK: ldr r0, [r0]
; CHECK-NEXT: mov pc, lr

; CHECK64-LABEL: load_i32_by_i16:
; CHECK64: ldr		w0, [x0]
; CHECK64-NEXT: ret
  %2 = bitcast i32* %0 to i16*
  %3 = load i16, i16* %2, align 4
  %4 = zext i16 %3 to i32
  %5 = getelementptr inbounds i16, i16* %2, i32 1
  %6 = load i16, i16* %5, align 1
  %7 = zext i16 %6 to i32
  %8 = shl nuw nsw i32 %4, 16
  %9 = or i32 %8, %7
  ret i32 %9
}

; i16* p_16; // p_16 is 4 byte aligned
; i8* p_8 = (i8*) p_16;
; (i32) (p_16[0] << 16) | ((i32) p[2] << 8) | (i32) p[3]
define i32 @load_i32_by_i16_i8(i32*) {
; CHECK-LABEL: load_i32_by_i16_i8:
; CHECK: ldr r0, [r0]
; CHECK-NEXT: mov pc, lr

; CHECK64-LABEL: load_i32_by_i16_i8:
; CHECK64: ldr		w0, [x0]
; CHECK64-NEXT: ret
  %2 = bitcast i32* %0 to i16*
  %3 = bitcast i32* %0 to i8*
  %4 = load i16, i16* %2, align 4
  %5 = zext i16 %4 to i32
  %6 = shl nuw nsw i32 %5, 16
  %7 = getelementptr inbounds i8, i8* %3, i32 2
  %8 = load i8, i8* %7, align 1
  %9 = zext i8 %8 to i32
  %10 = shl nuw nsw i32 %9, 8
  %11 = getelementptr inbounds i8, i8* %3, i32 3
  %12 = load i8, i8* %11, align 1
  %13 = zext i8 %12 to i32
  %14 = or i32 %10, %13
  %15 = or i32 %14, %6
  ret i32 %15
}

; i8* p; // p is 8 byte aligned
; (i64) p[0] | ((i64) p[1] << 8) | ((i64) p[2] << 16) | ((i64) p[3] << 24) | ((i64) p[4] << 32) | ((i64) p[5] << 40) | ((i64) p[6] << 48) | ((i64) p[7] << 56)
define i64 @load_i64_by_i8_bswap(i64*) {
; CHECK-LABEL: load_i64_by_i8_bswap:
; CHECK: ldrb{{.*}}r0
; CHECK: ldrb{{.*}}r0
; CHECK: ldrb{{.*}}r0
; CHECK: ldrb{{.*}}r0
; CHECK: ldrb{{.*}}r0
; CHECK: ldrb{{.*}}r0
; CHECK: ldrb{{.*}}r0
; CHECK: ldrb{{.*}}r0
; CHECK: orr
; CHECK: mov pc, lr

; CHECK64-LABEL: load_i64_by_i8_bswap:
; CHECK64: ldr		x8, [x0]
; CHECK64-NEXT: rev	x0, x8
; CHECK64-NEXT: ret
  %2 = bitcast i64* %0 to i8*
  %3 = load i8, i8* %2, align 8
  %4 = zext i8 %3 to i64
  %5 = getelementptr inbounds i8, i8* %2, i64 1
  %6 = load i8, i8* %5, align 1
  %7 = zext i8 %6 to i64
  %8 = shl nuw nsw i64 %7, 8
  %9 = or i64 %8, %4
  %10 = getelementptr inbounds i8, i8* %2, i64 2
  %11 = load i8, i8* %10, align 1
  %12 = zext i8 %11 to i64
  %13 = shl nuw nsw i64 %12, 16
  %14 = or i64 %9, %13
  %15 = getelementptr inbounds i8, i8* %2, i64 3
  %16 = load i8, i8* %15, align 1
  %17 = zext i8 %16 to i64
  %18 = shl nuw nsw i64 %17, 24
  %19 = or i64 %14, %18
  %20 = getelementptr inbounds i8, i8* %2, i64 4
  %21 = load i8, i8* %20, align 1
  %22 = zext i8 %21 to i64
  %23 = shl nuw nsw i64 %22, 32
  %24 = or i64 %19, %23
  %25 = getelementptr inbounds i8, i8* %2, i64 5
  %26 = load i8, i8* %25, align 1
  %27 = zext i8 %26 to i64
  %28 = shl nuw nsw i64 %27, 40
  %29 = or i64 %24, %28
  %30 = getelementptr inbounds i8, i8* %2, i64 6
  %31 = load i8, i8* %30, align 1
  %32 = zext i8 %31 to i64
  %33 = shl nuw nsw i64 %32, 48
  %34 = or i64 %29, %33
  %35 = getelementptr inbounds i8, i8* %2, i64 7
  %36 = load i8, i8* %35, align 1
  %37 = zext i8 %36 to i64
  %38 = shl nuw i64 %37, 56
  %39 = or i64 %34, %38
  ret i64 %39
}

; i8* p; // p is 8 byte aligned
; ((i64) p[0] << 56) | ((i64) p[1] << 48) | ((i64) p[2] << 40) | ((i64) p[3] << 32) | ((i64) p[4] << 24) | ((i64) p[5] << 16) | ((i64) p[6] << 8) | (i64) p[7]
define i64 @load_i64_by_i8(i64*) {
; CHECK-LABEL: load_i64_by_i8:
; CHECK: ldrb{{.*}}r0
; CHECK: ldrb{{.*}}r0
; CHECK: ldrb{{.*}}r0
; CHECK: ldrb{{.*}}r0
; CHECK: ldrb{{.*}}r0
; CHECK: ldrb{{.*}}r0
; CHECK: ldrb{{.*}}r0
; CHECK: ldrb{{.*}}r0
; CHECK: orr
; CHECK: mov pc, lr

; CHECK64-LABEL: load_i64_by_i8:
; CHECK64: ldr		x0, [x0]
; CHECK64-NEXT: ret
  %2 = bitcast i64* %0 to i8*
  %3 = load i8, i8* %2, align 8
  %4 = zext i8 %3 to i64
  %5 = shl nuw i64 %4, 56
  %6 = getelementptr inbounds i8, i8* %2, i64 1
  %7 = load i8, i8* %6, align 1
  %8 = zext i8 %7 to i64
  %9 = shl nuw nsw i64 %8, 48
  %10 = or i64 %9, %5
  %11 = getelementptr inbounds i8, i8* %2, i64 2
  %12 = load i8, i8* %11, align 1
  %13 = zext i8 %12 to i64
  %14 = shl nuw nsw i64 %13, 40
  %15 = or i64 %10, %14
  %16 = getelementptr inbounds i8, i8* %2, i64 3
  %17 = load i8, i8* %16, align 1
  %18 = zext i8 %17 to i64
  %19 = shl nuw nsw i64 %18, 32
  %20 = or i64 %15, %19
  %21 = getelementptr inbounds i8, i8* %2, i64 4
  %22 = load i8, i8* %21, align 1
  %23 = zext i8 %22 to i64
  %24 = shl nuw nsw i64 %23, 24
  %25 = or i64 %20, %24
  %26 = getelementptr inbounds i8, i8* %2, i64 5
  %27 = load i8, i8* %26, align 1
  %28 = zext i8 %27 to i64
  %29 = shl nuw nsw i64 %28, 16
  %30 = or i64 %25, %29
  %31 = getelementptr inbounds i8, i8* %2, i64 6
  %32 = load i8, i8* %31, align 1
  %33 = zext i8 %32 to i64
  %34 = shl nuw nsw i64 %33, 8
  %35 = or i64 %30, %34
  %36 = getelementptr inbounds i8, i8* %2, i64 7
  %37 = load i8, i8* %36, align 1
  %38 = zext i8 %37 to i64
  %39 = or i64 %35, %38
  ret i64 %39
}
