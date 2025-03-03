; RUN: llc -mtriple=arm64_32-apple-ios7.0 -o - %s | FileCheck %s
; RUN: llc -mtriple=arm64_32-apple-ios7.0 -mattr=+outline-atomics -o - %s | FileCheck %s -check-prefix=OUTLINE-ATOMICS

define i8 @test_load_8(ptr %addr) {
; CHECK-LABEL: test_load_8:
; CHECK: ldarb w0, [x0]
  %val = load atomic i8, ptr %addr seq_cst, align 1
  ret i8 %val
}

define i16 @test_load_16(ptr %addr) {
; CHECK-LABEL: test_load_16:
; CHECK: ldarh w0, [x0]
  %val = load atomic i16, ptr %addr acquire, align 2
  ret i16 %val
}

define i32 @test_load_32(ptr %addr) {
; CHECK-LABEL: test_load_32:
; CHECK: ldar w0, [x0]
  %val = load atomic i32, ptr %addr seq_cst, align 4
  ret i32 %val
}

define i64 @test_load_64(ptr %addr) {
; CHECK-LABEL: test_load_64:
; CHECK: ldar x0, [x0]
  %val = load atomic i64, ptr %addr seq_cst, align 8
  ret i64 %val
}

define ptr @test_load_ptr(ptr %addr) {
; CHECK-LABEL: test_load_ptr:
; CHECK: ldar w0, [x0]
  %val = load atomic ptr, ptr %addr seq_cst, align 8
  ret ptr %val
}

define void @test_store_8(ptr %addr) {
; CHECK-LABEL: test_store_8:
; CHECK: stlrb wzr, [x0]
  store atomic i8 0, ptr %addr seq_cst, align 1
  ret void
}

define void @test_store_16(ptr %addr) {
; CHECK-LABEL: test_store_16:
; CHECK: stlrh wzr, [x0]
  store atomic i16 0, ptr %addr seq_cst, align 2
  ret void
}

define void @test_store_32(ptr %addr) {
; CHECK-LABEL: test_store_32:
; CHECK: stlr wzr, [x0]
  store atomic i32 0, ptr %addr seq_cst, align 4
  ret void
}

define void @test_store_64(ptr %addr) {
; CHECK-LABEL: test_store_64:
; CHECK: stlr xzr, [x0]
  store atomic i64 0, ptr %addr seq_cst, align 8
  ret void
}

define void @test_store_ptr(ptr %addr) {
; CHECK-LABEL: test_store_ptr:
; CHECK: stlr wzr, [x0]
  store atomic ptr null, ptr %addr seq_cst, align 8
  ret void
}

declare i64 @llvm.aarch64.ldxr.p0(ptr %addr)

define i8 @test_ldxr_8(ptr %addr) {
; CHECK-LABEL: test_ldxr_8:
; CHECK: ldxrb w0, [x0]

  %val = call i64 @llvm.aarch64.ldxr.p0(ptr elementtype(i8) %addr)
  %val8 = trunc i64 %val to i8
  ret i8 %val8
}

define i16 @test_ldxr_16(ptr %addr) {
; CHECK-LABEL: test_ldxr_16:
; CHECK: ldxrh w0, [x0]

  %val = call i64 @llvm.aarch64.ldxr.p0(ptr elementtype(i16) %addr)
  %val16 = trunc i64 %val to i16
  ret i16 %val16
}

define i32 @test_ldxr_32(ptr %addr) {
; CHECK-LABEL: test_ldxr_32:
; CHECK: ldxr w0, [x0]

  %val = call i64 @llvm.aarch64.ldxr.p0(ptr elementtype(i32) %addr)
  %val32 = trunc i64 %val to i32
  ret i32 %val32
}

define i64 @test_ldxr_64(ptr %addr) {
; CHECK-LABEL: test_ldxr_64:
; CHECK: ldxr x0, [x0]

  %val = call i64 @llvm.aarch64.ldxr.p0(ptr elementtype(i64) %addr)
  ret i64 %val
}

declare i64 @llvm.aarch64.ldaxr.p0(ptr %addr)

define i8 @test_ldaxr_8(ptr %addr) {
; CHECK-LABEL: test_ldaxr_8:
; CHECK: ldaxrb w0, [x0]

  %val = call i64 @llvm.aarch64.ldaxr.p0(ptr elementtype(i8) %addr)
  %val8 = trunc i64 %val to i8
  ret i8 %val8
}

define i16 @test_ldaxr_16(ptr %addr) {
; CHECK-LABEL: test_ldaxr_16:
; CHECK: ldaxrh w0, [x0]

  %val = call i64 @llvm.aarch64.ldaxr.p0(ptr elementtype(i16) %addr)
  %val16 = trunc i64 %val to i16
  ret i16 %val16
}

define i32 @test_ldaxr_32(ptr %addr) {
; CHECK-LABEL: test_ldaxr_32:
; CHECK: ldaxr w0, [x0]

  %val = call i64 @llvm.aarch64.ldaxr.p0(ptr elementtype(i32) %addr)
  %val32 = trunc i64 %val to i32
  ret i32 %val32
}

define i64 @test_ldaxr_64(ptr %addr) {
; CHECK-LABEL: test_ldaxr_64:
; CHECK: ldaxr x0, [x0]

  %val = call i64 @llvm.aarch64.ldaxr.p0(ptr elementtype(i64) %addr)
  ret i64 %val
}

declare i32 @llvm.aarch64.stxr.p0(i64, ptr)

define i32 @test_stxr_8(ptr %addr, i8 %val) {
; CHECK-LABEL: test_stxr_8:
; CHECK: stxrb [[TMP:w[0-9]+]], w1, [x0]
; CHECK: mov w0, [[TMP]]

  %extval = zext i8 %val to i64
  %success = call i32 @llvm.aarch64.stxr.p0(i64 %extval, ptr elementtype(i8) %addr)
  ret i32 %success
}

define i32 @test_stxr_16(ptr %addr, i16 %val) {
; CHECK-LABEL: test_stxr_16:
; CHECK: stxrh [[TMP:w[0-9]+]], w1, [x0]
; CHECK: mov w0, [[TMP]]

  %extval = zext i16 %val to i64
  %success = call i32 @llvm.aarch64.stxr.p0(i64 %extval, ptr elementtype(i16) %addr)
  ret i32 %success
}

define i32 @test_stxr_32(ptr %addr, i32 %val) {
; CHECK-LABEL: test_stxr_32:
; CHECK: stxr [[TMP:w[0-9]+]], w1, [x0]
; CHECK: mov w0, [[TMP]]

  %extval = zext i32 %val to i64
  %success = call i32 @llvm.aarch64.stxr.p0(i64 %extval, ptr elementtype(i32) %addr)
  ret i32 %success
}

define i32 @test_stxr_64(ptr %addr, i64 %val) {
; CHECK-LABEL: test_stxr_64:
; CHECK: stxr [[TMP:w[0-9]+]], x1, [x0]
; CHECK: mov w0, [[TMP]]

  %success = call i32 @llvm.aarch64.stxr.p0(i64 %val, ptr elementtype(i64) %addr)
  ret i32 %success
}

declare i32 @llvm.aarch64.stlxr.p0(i64, ptr)

define i32 @test_stlxr_8(ptr %addr, i8 %val) {
; CHECK-LABEL: test_stlxr_8:
; CHECK: stlxrb [[TMP:w[0-9]+]], w1, [x0]
; CHECK: mov w0, [[TMP]]

  %extval = zext i8 %val to i64
  %success = call i32 @llvm.aarch64.stlxr.p0(i64 %extval, ptr elementtype(i8) %addr)
  ret i32 %success
}

define i32 @test_stlxr_16(ptr %addr, i16 %val) {
; CHECK-LABEL: test_stlxr_16:
; CHECK: stlxrh [[TMP:w[0-9]+]], w1, [x0]
; CHECK: mov w0, [[TMP]]

  %extval = zext i16 %val to i64
  %success = call i32 @llvm.aarch64.stlxr.p0(i64 %extval, ptr elementtype(i16) %addr)
  ret i32 %success
}

define i32 @test_stlxr_32(ptr %addr, i32 %val) {
; CHECK-LABEL: test_stlxr_32:
; CHECK: stlxr [[TMP:w[0-9]+]], w1, [x0]
; CHECK: mov w0, [[TMP]]

  %extval = zext i32 %val to i64
  %success = call i32 @llvm.aarch64.stlxr.p0(i64 %extval, ptr elementtype(i32) %addr)
  ret i32 %success
}

define i32 @test_stlxr_64(ptr %addr, i64 %val) {
; CHECK-LABEL: test_stlxr_64:
; CHECK: stlxr [[TMP:w[0-9]+]], x1, [x0]
; CHECK: mov w0, [[TMP]]

  %success = call i32 @llvm.aarch64.stlxr.p0(i64 %val, ptr elementtype(i64) %addr)
  ret i32 %success
}

define {ptr, i1} @test_cmpxchg_ptr(ptr %addr, ptr %cmp, ptr %new) {
; OUTLINE-ATOMICS: bl ___aarch64_cas4_acq_rel
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
; CHECK:     mov w1, wzr
; CHECK:     mov w0, [[OLD]]
; CHECK:     clrex
; CHECK:     ret
  %res = cmpxchg ptr %addr, ptr %cmp, ptr %new acq_rel acquire
  ret {ptr, i1} %res
}
