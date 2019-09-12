; RUN: llc -mtriple=arm64_32-apple-ios7.0 -o - %s | FileCheck %s

define i8 @test_load_8(i8* %addr) {
; CHECK-LABAL: test_load_8:
; CHECK: ldarb w0, [x0]
  %val = load atomic i8, i8* %addr seq_cst, align 1
  ret i8 %val
}

define i16 @test_load_16(i16* %addr) {
; CHECK-LABAL: test_load_16:
; CHECK: ldarh w0, [x0]
  %val = load atomic i16, i16* %addr acquire, align 2
  ret i16 %val
}

define i32 @test_load_32(i32* %addr) {
; CHECK-LABAL: test_load_32:
; CHECK: ldar w0, [x0]
  %val = load atomic i32, i32* %addr seq_cst, align 4
  ret i32 %val
}

define i64 @test_load_64(i64* %addr) {
; CHECK-LABAL: test_load_64:
; CHECK: ldar x0, [x0]
  %val = load atomic i64, i64* %addr seq_cst, align 8
  ret i64 %val
}

define i8* @test_load_ptr(i8** %addr) {
; CHECK-LABAL: test_load_ptr:
; CHECK: ldar w0, [x0]
  %val = load atomic i8*, i8** %addr seq_cst, align 8
  ret i8* %val
}

define void @test_store_8(i8* %addr) {
; CHECK-LABAL: test_store_8:
; CHECK: stlrb wzr, [x0]
  store atomic i8 0, i8* %addr seq_cst, align 1
  ret void
}

define void @test_store_16(i16* %addr) {
; CHECK-LABAL: test_store_16:
; CHECK: stlrh wzr, [x0]
  store atomic i16 0, i16* %addr seq_cst, align 2
  ret void
}

define void @test_store_32(i32* %addr) {
; CHECK-LABAL: test_store_32:
; CHECK: stlr wzr, [x0]
  store atomic i32 0, i32* %addr seq_cst, align 4
  ret void
}

define void @test_store_64(i64* %addr) {
; CHECK-LABAL: test_store_64:
; CHECK: stlr xzr, [x0]
  store atomic i64 0, i64* %addr seq_cst, align 8
  ret void
}

define void @test_store_ptr(i8** %addr) {
; CHECK-LABAL: test_store_ptr:
; CHECK: stlr wzr, [x0]
  store atomic i8* null, i8** %addr seq_cst, align 8
  ret void
}

declare i64 @llvm.aarch64.ldxr.p0i8(i8* %addr)
declare i64 @llvm.aarch64.ldxr.p0i16(i16* %addr)
declare i64 @llvm.aarch64.ldxr.p0i32(i32* %addr)
declare i64 @llvm.aarch64.ldxr.p0i64(i64* %addr)

define i8 @test_ldxr_8(i8* %addr) {
; CHECK-LABEL: test_ldxr_8:
; CHECK: ldxrb w0, [x0]

  %val = call i64 @llvm.aarch64.ldxr.p0i8(i8* %addr)
  %val8 = trunc i64 %val to i8
  ret i8 %val8
}

define i16 @test_ldxr_16(i16* %addr) {
; CHECK-LABEL: test_ldxr_16:
; CHECK: ldxrh w0, [x0]

  %val = call i64 @llvm.aarch64.ldxr.p0i16(i16* %addr)
  %val16 = trunc i64 %val to i16
  ret i16 %val16
}

define i32 @test_ldxr_32(i32* %addr) {
; CHECK-LABEL: test_ldxr_32:
; CHECK: ldxr w0, [x0]

  %val = call i64 @llvm.aarch64.ldxr.p0i32(i32* %addr)
  %val32 = trunc i64 %val to i32
  ret i32 %val32
}

define i64 @test_ldxr_64(i64* %addr) {
; CHECK-LABEL: test_ldxr_64:
; CHECK: ldxr x0, [x0]

  %val = call i64 @llvm.aarch64.ldxr.p0i64(i64* %addr)
  ret i64 %val
}

declare i64 @llvm.aarch64.ldaxr.p0i8(i8* %addr)
declare i64 @llvm.aarch64.ldaxr.p0i16(i16* %addr)
declare i64 @llvm.aarch64.ldaxr.p0i32(i32* %addr)
declare i64 @llvm.aarch64.ldaxr.p0i64(i64* %addr)

define i8 @test_ldaxr_8(i8* %addr) {
; CHECK-LABEL: test_ldaxr_8:
; CHECK: ldaxrb w0, [x0]

  %val = call i64 @llvm.aarch64.ldaxr.p0i8(i8* %addr)
  %val8 = trunc i64 %val to i8
  ret i8 %val8
}

define i16 @test_ldaxr_16(i16* %addr) {
; CHECK-LABEL: test_ldaxr_16:
; CHECK: ldaxrh w0, [x0]

  %val = call i64 @llvm.aarch64.ldaxr.p0i16(i16* %addr)
  %val16 = trunc i64 %val to i16
  ret i16 %val16
}

define i32 @test_ldaxr_32(i32* %addr) {
; CHECK-LABEL: test_ldaxr_32:
; CHECK: ldaxr w0, [x0]

  %val = call i64 @llvm.aarch64.ldaxr.p0i32(i32* %addr)
  %val32 = trunc i64 %val to i32
  ret i32 %val32
}

define i64 @test_ldaxr_64(i64* %addr) {
; CHECK-LABEL: test_ldaxr_64:
; CHECK: ldaxr x0, [x0]

  %val = call i64 @llvm.aarch64.ldaxr.p0i64(i64* %addr)
  ret i64 %val
}

declare i32 @llvm.aarch64.stxr.p0i8(i64, i8*)
declare i32 @llvm.aarch64.stxr.p0i16(i64, i16*)
declare i32 @llvm.aarch64.stxr.p0i32(i64, i32*)
declare i32 @llvm.aarch64.stxr.p0i64(i64, i64*)

define i32 @test_stxr_8(i8* %addr, i8 %val) {
; CHECK-LABEL: test_stxr_8:
; CHECK: stxrb [[TMP:w[0-9]+]], w1, [x0]
; CHECK: mov w0, [[TMP]]

  %extval = zext i8 %val to i64
  %success = call i32 @llvm.aarch64.stxr.p0i8(i64 %extval, i8* %addr)
  ret i32 %success
}

define i32 @test_stxr_16(i16* %addr, i16 %val) {
; CHECK-LABEL: test_stxr_16:
; CHECK: stxrh [[TMP:w[0-9]+]], w1, [x0]
; CHECK: mov w0, [[TMP]]

  %extval = zext i16 %val to i64
  %success = call i32 @llvm.aarch64.stxr.p0i16(i64 %extval, i16* %addr)
  ret i32 %success
}

define i32 @test_stxr_32(i32* %addr, i32 %val) {
; CHECK-LABEL: test_stxr_32:
; CHECK: stxr [[TMP:w[0-9]+]], w1, [x0]
; CHECK: mov w0, [[TMP]]

  %extval = zext i32 %val to i64
  %success = call i32 @llvm.aarch64.stxr.p0i32(i64 %extval, i32* %addr)
  ret i32 %success
}

define i32 @test_stxr_64(i64* %addr, i64 %val) {
; CHECK-LABEL: test_stxr_64:
; CHECK: stxr [[TMP:w[0-9]+]], x1, [x0]
; CHECK: mov w0, [[TMP]]

  %success = call i32 @llvm.aarch64.stxr.p0i64(i64 %val, i64* %addr)
  ret i32 %success
}

declare i32 @llvm.aarch64.stlxr.p0i8(i64, i8*)
declare i32 @llvm.aarch64.stlxr.p0i16(i64, i16*)
declare i32 @llvm.aarch64.stlxr.p0i32(i64, i32*)
declare i32 @llvm.aarch64.stlxr.p0i64(i64, i64*)

define i32 @test_stlxr_8(i8* %addr, i8 %val) {
; CHECK-LABEL: test_stlxr_8:
; CHECK: stlxrb [[TMP:w[0-9]+]], w1, [x0]
; CHECK: mov w0, [[TMP]]

  %extval = zext i8 %val to i64
  %success = call i32 @llvm.aarch64.stlxr.p0i8(i64 %extval, i8* %addr)
  ret i32 %success
}

define i32 @test_stlxr_16(i16* %addr, i16 %val) {
; CHECK-LABEL: test_stlxr_16:
; CHECK: stlxrh [[TMP:w[0-9]+]], w1, [x0]
; CHECK: mov w0, [[TMP]]

  %extval = zext i16 %val to i64
  %success = call i32 @llvm.aarch64.stlxr.p0i16(i64 %extval, i16* %addr)
  ret i32 %success
}

define i32 @test_stlxr_32(i32* %addr, i32 %val) {
; CHECK-LABEL: test_stlxr_32:
; CHECK: stlxr [[TMP:w[0-9]+]], w1, [x0]
; CHECK: mov w0, [[TMP]]

  %extval = zext i32 %val to i64
  %success = call i32 @llvm.aarch64.stlxr.p0i32(i64 %extval, i32* %addr)
  ret i32 %success
}

define i32 @test_stlxr_64(i64* %addr, i64 %val) {
; CHECK-LABEL: test_stlxr_64:
; CHECK: stlxr [[TMP:w[0-9]+]], x1, [x0]
; CHECK: mov w0, [[TMP]]

  %success = call i32 @llvm.aarch64.stlxr.p0i64(i64 %val, i64* %addr)
  ret i32 %success
}

define {i8*, i1} @test_cmpxchg_ptr(i8** %addr, i8* %cmp, i8* %new) {
; CHECK-LABEL: test_cmpxchg_ptr:
; CHECK: [[LOOP:LBB[0-9]+_[0-9]+]]:
; CHECK:     ldaxr [[OLD:w[0-9]+]], [x0]
; CHECK:     cmp [[OLD]], w1
; CHECK:     b.ne [[DONE:LBB[0-9]+_[0-9]+]]
; CHECK:     stlxr [[SUCCESS:w[0-9]+]], w2, [x0]
; CHECK:     cbnz [[SUCCESS]], [[LOOP]]

; CHECK:     mov w1, #1
; CHECK:     mov w0, [[OLD]]
; CHECK:     ret

; CHECK: [[DONE]]:
; CHECK:     clrex
; CHECK:     mov w1, wzr
; CHECK:     mov w0, [[OLD]]
; CHECK:     ret
  %res = cmpxchg i8** %addr, i8* %cmp, i8* %new acq_rel acquire
  ret {i8*, i1} %res
}
