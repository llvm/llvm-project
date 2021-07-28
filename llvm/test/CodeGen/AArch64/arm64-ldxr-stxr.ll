; RUN: llc < %s -mtriple=arm64-linux-gnu | FileCheck %s
; RUN: llc < %s -global-isel -global-isel-abort=1 -pass-remarks-missed=gisel* -mtriple=arm64-linux-gnu 2>&1 | FileCheck %s --check-prefixes=GISEL,FALLBACK

%0 = type { i64, i64 }

define dso_local i128 @f0(i8* %p) nounwind readonly {
; CHECK-LABEL: f0:
; CHECK: ldxp {{x[0-9]+}}, {{x[0-9]+}}, [x0]
entry:
  %ldrexd = tail call %0 @llvm.aarch64.ldxp(i8* %p)
  %0 = extractvalue %0 %ldrexd, 1
  %1 = extractvalue %0 %ldrexd, 0
  %2 = zext i64 %0 to i128
  %3 = zext i64 %1 to i128
  %shl = shl nuw i128 %2, 64
  %4 = or i128 %shl, %3
  ret i128 %4
}

define dso_local i32 @f1(i8* %ptr, i128 %val) nounwind {
; CHECK-LABEL: f1:
; CHECK: stxp {{w[0-9]+}}, {{x[0-9]+}}, {{x[0-9]+}}, [x0]
entry:
  %tmp4 = trunc i128 %val to i64
  %tmp6 = lshr i128 %val, 64
  %tmp7 = trunc i128 %tmp6 to i64
  %strexd = tail call i32 @llvm.aarch64.stxp(i64 %tmp4, i64 %tmp7, i8* %ptr)
  ret i32 %strexd
}

declare %0 @llvm.aarch64.ldxp(i8*) nounwind
declare i32 @llvm.aarch64.stxp(i64, i64, i8*) nounwind

@var = dso_local global i64 0, align 8

; FALLBACK-NOT: remark:{{.*}}test_load_i8
define dso_local void @test_load_i8(i8* %addr) {
; CHECK-LABEL: test_load_i8:
; CHECK: ldxrb w[[LOADVAL:[0-9]+]], [x0]
; CHECK-NOT: uxtb
; CHECK-NOT: and
; CHECK: str x[[LOADVAL]], [{{x[0-9]+}}, :lo12:var]

; GISEL-LABEL: test_load_i8:
; GISEL: ldxrb w[[LOADVAL:[0-9]+]], [x0]
; GISEL-NOT: uxtb
; GISEL: str x[[LOADVAL]], [{{x[0-9]+}}, :lo12:var]
  %val = call i64 @llvm.aarch64.ldxr.p0i8(i8* %addr)
  %shortval = trunc i64 %val to i8
  %extval = zext i8 %shortval to i64
  store i64 %extval, i64* @var, align 8
  ret void
}

; FALLBACK-NOT: remark:{{.*}}test_load_i16
define dso_local void @test_load_i16(i16* %addr) {
; CHECK-LABEL: test_load_i16:
; CHECK: ldxrh w[[LOADVAL:[0-9]+]], [x0]
; CHECK-NOT: uxth
; CHECK-NOT: and
; CHECK: str x[[LOADVAL]], [{{x[0-9]+}}, :lo12:var]

; GISEL-LABEL: test_load_i16:
; GISEL: ldxrh w[[LOADVAL:[0-9]+]], [x0]
; GISEL-NOT: uxtb
; GISEL: str x[[LOADVAL]], [{{x[0-9]+}}, :lo12:var]
  %val = call i64 @llvm.aarch64.ldxr.p0i16(i16* %addr)
  %shortval = trunc i64 %val to i16
  %extval = zext i16 %shortval to i64
  store i64 %extval, i64* @var, align 8
  ret void
}

; FALLBACK-NOT: remark:{{.*}}test_load_i32
define dso_local void @test_load_i32(i32* %addr) {
; CHECK-LABEL: test_load_i32:
; CHECK: ldxr w[[LOADVAL:[0-9]+]], [x0]
; CHECK-NOT: uxtw
; CHECK-NOT: and
; CHECK: str x[[LOADVAL]], [{{x[0-9]+}}, :lo12:var]

; GISEL-LABEL: test_load_i32:
; GISEL: ldxr w[[LOADVAL:[0-9]+]], [x0]
; GISEL-NOT: uxtb
; GISEL: str x[[LOADVAL]], [{{x[0-9]+}}, :lo12:var]
  %val = call i64 @llvm.aarch64.ldxr.p0i32(i32* %addr)
  %shortval = trunc i64 %val to i32
  %extval = zext i32 %shortval to i64
  store i64 %extval, i64* @var, align 8
  ret void
}

; FALLBACK-NOT: remark:{{.*}}test_load_i64
define dso_local void @test_load_i64(i64* %addr) {
; CHECK-LABEL: test_load_i64:
; CHECK: ldxr x[[LOADVAL:[0-9]+]], [x0]
; CHECK: str x[[LOADVAL]], [{{x[0-9]+}}, :lo12:var]

; GISEL-LABEL: test_load_i64:
; GISEL: ldxr x[[LOADVAL:[0-9]+]], [x0]
; GISEL-NOT: uxtb
; GISEL: str x[[LOADVAL]], [{{x[0-9]+}}, :lo12:var]
  %val = call i64 @llvm.aarch64.ldxr.p0i64(i64* %addr)
  store i64 %val, i64* @var, align 8
  ret void
}


declare i64 @llvm.aarch64.ldxr.p0i8(i8*) nounwind
declare i64 @llvm.aarch64.ldxr.p0i16(i16*) nounwind
declare i64 @llvm.aarch64.ldxr.p0i32(i32*) nounwind
declare i64 @llvm.aarch64.ldxr.p0i64(i64*) nounwind

; FALLBACK-NOT: remark:{{.*}}test_store_i8
define dso_local i32 @test_store_i8(i32, i8 %val, i8* %addr) {
; CHECK-LABEL: test_store_i8:
; CHECK-NOT: uxtb
; CHECK-NOT: and
; CHECK: stxrb w0, w1, [x2]
; GISEL-LABEL: test_store_i8:
; GISEL-NOT: uxtb
; GISEL-NOT: and
; GISEL: stxrb w0, w1, [x2]
  %extval = zext i8 %val to i64
  %res = call i32 @llvm.aarch64.stxr.p0i8(i64 %extval, i8* %addr)
  ret i32 %res
}

; FALLBACK-NOT: remark:{{.*}}test_store_i16
define dso_local i32 @test_store_i16(i32, i16 %val, i16* %addr) {
; CHECK-LABEL: test_store_i16:
; CHECK-NOT: uxth
; CHECK-NOT: and
; CHECK: stxrh w0, w1, [x2]
; GISEL-LABEL: test_store_i16:
; GISEL-NOT: uxth
; GISEL-NOT: and
; GISEL: stxrh w0, w1, [x2]
  %extval = zext i16 %val to i64
  %res = call i32 @llvm.aarch64.stxr.p0i16(i64 %extval, i16* %addr)
  ret i32 %res
}

; FALLBACK-NOT: remark:{{.*}}test_store_i32
define dso_local i32 @test_store_i32(i32, i32 %val, i32* %addr) {
; CHECK-LABEL: test_store_i32:
; CHECK-NOT: uxtw
; CHECK-NOT: and
; CHECK: stxr w0, w1, [x2]
; GISEL-LABEL: test_store_i32:
; GISEL-NOT: uxtw
; GISEL-NOT: and
; GISEL: stxr w0, w1, [x2]
  %extval = zext i32 %val to i64
  %res = call i32 @llvm.aarch64.stxr.p0i32(i64 %extval, i32* %addr)
  ret i32 %res
}

; FALLBACK-NOT: remark:{{.*}}test_store_i64
define dso_local i32 @test_store_i64(i32, i64 %val, i64* %addr) {
; CHECK-LABEL: test_store_i64:
; CHECK: stxr w0, x1, [x2]
; GISEL-LABEL: test_store_i64:
; GISEL: stxr w0, x1, [x2]
  %res = call i32 @llvm.aarch64.stxr.p0i64(i64 %val, i64* %addr)
  ret i32 %res
}

declare i32 @llvm.aarch64.stxr.p0i8(i64, i8*) nounwind
declare i32 @llvm.aarch64.stxr.p0i16(i64, i16*) nounwind
declare i32 @llvm.aarch64.stxr.p0i32(i64, i32*) nounwind
declare i32 @llvm.aarch64.stxr.p0i64(i64, i64*) nounwind

; CHECK: test_clear:
; CHECK: clrex
define dso_local void @test_clear() {
  call void @llvm.aarch64.clrex()
  ret void
}

declare void @llvm.aarch64.clrex() nounwind

define dso_local i128 @test_load_acquire_i128(i8* %p) nounwind readonly {
; CHECK-LABEL: test_load_acquire_i128:
; CHECK: ldaxp {{x[0-9]+}}, {{x[0-9]+}}, [x0]
entry:
  %ldrexd = tail call %0 @llvm.aarch64.ldaxp(i8* %p)
  %0 = extractvalue %0 %ldrexd, 1
  %1 = extractvalue %0 %ldrexd, 0
  %2 = zext i64 %0 to i128
  %3 = zext i64 %1 to i128
  %shl = shl nuw i128 %2, 64
  %4 = or i128 %shl, %3
  ret i128 %4
}

define dso_local i32 @test_store_release_i128(i8* %ptr, i128 %val) nounwind {
; CHECK-LABEL: test_store_release_i128:
; CHECK: stlxp {{w[0-9]+}}, {{x[0-9]+}}, {{x[0-9]+}}, [x0]
entry:
  %tmp4 = trunc i128 %val to i64
  %tmp6 = lshr i128 %val, 64
  %tmp7 = trunc i128 %tmp6 to i64
  %strexd = tail call i32 @llvm.aarch64.stlxp(i64 %tmp4, i64 %tmp7, i8* %ptr)
  ret i32 %strexd
}

declare %0 @llvm.aarch64.ldaxp(i8*) nounwind
declare i32 @llvm.aarch64.stlxp(i64, i64, i8*) nounwind

; FALLBACK-NOT: remark:{{.*}}test_load_acquire_i8
define dso_local void @test_load_acquire_i8(i8* %addr) {
; CHECK-LABEL: test_load_acquire_i8:
; CHECK: ldaxrb w[[LOADVAL:[0-9]+]], [x0]
; CHECK-NOT: uxtb
; CHECK-NOT: and
; CHECK: str x[[LOADVAL]], [{{x[0-9]+}}, :lo12:var]

; GISEL-LABEL: test_load_acquire_i8:
; GISEL: ldaxrb w[[LOADVAL:[0-9]+]], [x0]
; GISEL-DAG: str x[[LOADVAL]], [{{x[0-9]+}}, :lo12:var]
  %val = call i64 @llvm.aarch64.ldaxr.p0i8(i8* %addr)
  %shortval = trunc i64 %val to i8
  %extval = zext i8 %shortval to i64
  store i64 %extval, i64* @var, align 8
  ret void
}

; FALLBACK-NOT: remark:{{.*}}test_load_acquire_i16
define dso_local void @test_load_acquire_i16(i16* %addr) {
; CHECK-LABEL: test_load_acquire_i16:
; CHECK: ldaxrh w[[LOADVAL:[0-9]+]], [x0]
; CHECK-NOT: uxth
; CHECK-NOT: and
; CHECK: str x[[LOADVAL]], [{{x[0-9]+}}, :lo12:var]

; GISEL-LABEL: test_load_acquire_i16:
; GISEL: ldaxrh w[[LOADVAL:[0-9]+]], [x0]
; GISEL: str x[[LOADVAL]], [{{x[0-9]+}}, :lo12:var]
  %val = call i64 @llvm.aarch64.ldaxr.p0i16(i16* %addr)
  %shortval = trunc i64 %val to i16
  %extval = zext i16 %shortval to i64
  store i64 %extval, i64* @var, align 8
  ret void
}

; FALLBACK-NOT: remark:{{.*}}test_load_acquire_i32
define dso_local void @test_load_acquire_i32(i32* %addr) {
; CHECK-LABEL: test_load_acquire_i32:
; CHECK: ldaxr w[[LOADVAL:[0-9]+]], [x0]
; CHECK-NOT: uxtw
; CHECK-NOT: and
; CHECK: str x[[LOADVAL]], [{{x[0-9]+}}, :lo12:var]

; GISEL-LABEL: test_load_acquire_i32:
; GISEL: ldaxr w[[LOADVAL:[0-9]+]], [x0]
; GISEL: str x[[LOADVAL]], [{{x[0-9]+}}, :lo12:var]
  %val = call i64 @llvm.aarch64.ldaxr.p0i32(i32* %addr)
  %shortval = trunc i64 %val to i32
  %extval = zext i32 %shortval to i64
  store i64 %extval, i64* @var, align 8
  ret void
}

; FALLBACK-NOT: remark:{{.*}}test_load_acquire_i64
define dso_local void @test_load_acquire_i64(i64* %addr) {
; CHECK-LABEL: test_load_acquire_i64:
; CHECK: ldaxr x[[LOADVAL:[0-9]+]], [x0]
; CHECK: str x[[LOADVAL]], [{{x[0-9]+}}, :lo12:var]

; GISEL-LABEL: test_load_acquire_i64:
; GISEL: ldaxr x[[LOADVAL:[0-9]+]], [x0]
; GISEL: str x[[LOADVAL]], [{{x[0-9]+}}, :lo12:var]
  %val = call i64 @llvm.aarch64.ldaxr.p0i64(i64* %addr)
  store i64 %val, i64* @var, align 8
  ret void
}


declare i64 @llvm.aarch64.ldaxr.p0i8(i8*) nounwind
declare i64 @llvm.aarch64.ldaxr.p0i16(i16*) nounwind
declare i64 @llvm.aarch64.ldaxr.p0i32(i32*) nounwind
declare i64 @llvm.aarch64.ldaxr.p0i64(i64*) nounwind

; FALLBACK-NOT: remark:{{.*}}test_store_release_i8
define dso_local i32 @test_store_release_i8(i32, i8 %val, i8* %addr) {
; CHECK-LABEL: test_store_release_i8:
; CHECK-NOT: uxtb
; CHECK-NOT: and
; CHECK: stlxrb w0, w1, [x2]
; GISEL-LABEL: test_store_release_i8:
; GISEL-NOT: uxtb
; GISEL-NOT: and
; GISEL: stlxrb w0, w1, [x2]
  %extval = zext i8 %val to i64
  %res = call i32 @llvm.aarch64.stlxr.p0i8(i64 %extval, i8* %addr)
  ret i32 %res
}

; FALLBACK-NOT: remark:{{.*}}test_store_release_i16
define dso_local i32 @test_store_release_i16(i32, i16 %val, i16* %addr) {
; CHECK-LABEL: test_store_release_i16:
; CHECK-NOT: uxth
; CHECK-NOT: and
; CHECK: stlxrh w0, w1, [x2]
; GISEL-LABEL: test_store_release_i16:
; GISEL-NOT: uxth
; GISEL-NOT: and
; GISEL: stlxrh w0, w1, [x2]
  %extval = zext i16 %val to i64
  %res = call i32 @llvm.aarch64.stlxr.p0i16(i64 %extval, i16* %addr)
  ret i32 %res
}

; FALLBACK-NOT: remark:{{.*}}test_store_release_i32
define dso_local i32 @test_store_release_i32(i32, i32 %val, i32* %addr) {
; CHECK-LABEL: test_store_release_i32:
; CHECK-NOT: uxtw
; CHECK-NOT: and
; CHECK: stlxr w0, w1, [x2]
; GISEL-LABEL: test_store_release_i32:
; GISEL-NOT: uxtw
; GISEL-NOT: and
; GISEL: stlxr w0, w1, [x2]
  %extval = zext i32 %val to i64
  %res = call i32 @llvm.aarch64.stlxr.p0i32(i64 %extval, i32* %addr)
  ret i32 %res
}

; FALLBACK-NOT: remark:{{.*}}test_store_release_i64
define dso_local i32 @test_store_release_i64(i32, i64 %val, i64* %addr) {
; CHECK-LABEL: test_store_release_i64:
; CHECK: stlxr w0, x1, [x2]
; GISEL-LABEL: test_store_release_i64:
; GISEL: stlxr w0, x1, [x2]
  %res = call i32 @llvm.aarch64.stlxr.p0i64(i64 %val, i64* %addr)
  ret i32 %res
}

declare i32 @llvm.aarch64.stlxr.p0i8(i64, i8*) nounwind
declare i32 @llvm.aarch64.stlxr.p0i16(i64, i16*) nounwind
declare i32 @llvm.aarch64.stlxr.p0i32(i64, i32*) nounwind
declare i32 @llvm.aarch64.stlxr.p0i64(i64, i64*) nounwind
