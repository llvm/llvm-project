; Test transactional-execution intrinsics.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=zEC12 | FileCheck %s

declare i32 @llvm.s390.tbegin(i8 *, i32)
declare i32 @llvm.s390.tbegin.nofloat(i8 *, i32)
declare void @llvm.s390.tbeginc(i8 *, i32)
declare i32 @llvm.s390.tend()
declare void @llvm.s390.tabort(i64)
declare void @llvm.s390.ntstg(i64, i64 *)
declare i32 @llvm.s390.etnd()
declare void @llvm.s390.ppa.txassist(i32)

; TBEGIN.
define void @test_tbegin() {
; CHECK-LABEL: test_tbegin:
; CHECK-NOT: stmg
; CHECK: std %f8,
; CHECK: std %f9,
; CHECK: std %f10,
; CHECK: std %f11,
; CHECK: std %f12,
; CHECK: std %f13,
; CHECK: std %f14,
; CHECK: std %f15,
; CHECK: tbegin 0, 65292
; CHECK: ld %f8,
; CHECK: ld %f9,
; CHECK: ld %f10,
; CHECK: ld %f11,
; CHECK: ld %f12,
; CHECK: ld %f13,
; CHECK: ld %f14,
; CHECK: ld %f15,
; CHECK: br %r14
  call i32 @llvm.s390.tbegin(i8 *null, i32 65292)
  ret void
}

; TBEGIN (nofloat).
define void @test_tbegin_nofloat1() {
; CHECK-LABEL: test_tbegin_nofloat1:
; CHECK-NOT: stmg
; CHECK-NOT: std
; CHECK: tbegin 0, 65292
; CHECK: br %r14
  call i32 @llvm.s390.tbegin.nofloat(i8 *null, i32 65292)
  ret void
}

; TBEGIN (nofloat) with integer CC return value.
define i32 @test_tbegin_nofloat2() {
; CHECK-LABEL: test_tbegin_nofloat2:
; CHECK-NOT: stmg
; CHECK-NOT: std
; CHECK: tbegin 0, 65292
; CHECK: ipm %r2
; CHECK: srl %r2, 28
; CHECK: br %r14
  %res = call i32 @llvm.s390.tbegin.nofloat(i8 *null, i32 65292)
  ret i32 %res
}

; TBEGIN (nofloat) with implicit CC check.
define void @test_tbegin_nofloat3(i32 *%ptr) {
; CHECK-LABEL: test_tbegin_nofloat3:
; CHECK-NOT: stmg
; CHECK-NOT: std
; CHECK: tbegin 0, 65292
; CHECK: bnhr %r14
; CHECK: mvhi 0(%r2), 0
; CHECK: br %r14
  %res = call i32 @llvm.s390.tbegin.nofloat(i8 *null, i32 65292)
  %cmp = icmp eq i32 %res, 2
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  store i32 0, i32* %ptr, align 4
  br label %if.end

if.end:                                           ; preds = %if.then, %entry
  ret void
}

; TBEGIN (nofloat) with dual CC use.
define i32 @test_tbegin_nofloat4(i32 %pad, i32 *%ptr) {
; CHECK-LABEL: test_tbegin_nofloat4:
; CHECK-NOT: stmg
; CHECK-NOT: std
; CHECK: tbegin 0, 65292
; CHECK: ipm %r2
; CHECK: srl %r2, 28
; CHECK: ciblh %r2, 2, 0(%r14)
; CHECK: mvhi 0(%r3), 0
; CHECK: br %r14
  %res = call i32 @llvm.s390.tbegin.nofloat(i8 *null, i32 65292)
  %cmp = icmp eq i32 %res, 2
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  store i32 0, i32* %ptr, align 4
  br label %if.end

if.end:                                           ; preds = %if.then, %entry
  ret i32 %res
}

; TBEGIN (nofloat) with register.
define void @test_tbegin_nofloat5(i8 *%ptr) {
; CHECK-LABEL: test_tbegin_nofloat5:
; CHECK-NOT: stmg
; CHECK-NOT: std
; CHECK: tbegin 0(%r2), 65292
; CHECK: br %r14
  call i32 @llvm.s390.tbegin.nofloat(i8 *%ptr, i32 65292)
  ret void
}

; TBEGIN (nofloat) with GRSM 0x0f00.
define void @test_tbegin_nofloat6() {
; CHECK-LABEL: test_tbegin_nofloat6:
; CHECK: stmg %r6, %r15,
; CHECK-NOT: std
; CHECK: tbegin 0, 3840
; CHECK: br %r14
  call i32 @llvm.s390.tbegin.nofloat(i8 *null, i32 3840)
  ret void
}

; TBEGIN (nofloat) with GRSM 0xf100.
define void @test_tbegin_nofloat7() {
; CHECK-LABEL: test_tbegin_nofloat7:
; CHECK: stmg %r8, %r15,
; CHECK-NOT: std
; CHECK: tbegin 0, 61696
; CHECK: br %r14
  call i32 @llvm.s390.tbegin.nofloat(i8 *null, i32 61696)
  ret void
}

; TBEGIN (nofloat) with GRSM 0xfe00 -- stack pointer added automatically.
define void @test_tbegin_nofloat8() {
; CHECK-LABEL: test_tbegin_nofloat8:
; CHECK-NOT: stmg
; CHECK-NOT: std
; CHECK: tbegin 0, 65280
; CHECK: br %r14
  call i32 @llvm.s390.tbegin.nofloat(i8 *null, i32 65024)
  ret void
}

; TBEGIN (nofloat) with GRSM 0xfb00 -- no frame pointer needed.
define void @test_tbegin_nofloat9() {
; CHECK-LABEL: test_tbegin_nofloat9:
; CHECK: stmg %r10, %r15,
; CHECK-NOT: std
; CHECK: tbegin 0, 64256
; CHECK: br %r14
  call i32 @llvm.s390.tbegin.nofloat(i8 *null, i32 64256)
  ret void
}

; TBEGIN (nofloat) with GRSM 0xfb00 -- frame pointer added automatically.
define void @test_tbegin_nofloat10(i64 %n) {
; CHECK-LABEL: test_tbegin_nofloat10:
; CHECK: stmg %r11, %r15,
; CHECK-NOT: std
; CHECK: tbegin 0, 65280
; CHECK: br %r14
  %buf = alloca i8, i64 %n
  call i32 @llvm.s390.tbegin.nofloat(i8 *null, i32 64256)
  ret void
}

; TBEGINC.
define void @test_tbeginc() {
; CHECK-LABEL: test_tbeginc:
; CHECK-NOT: stmg
; CHECK-NOT: std
; CHECK: tbeginc 0, 65288
; CHECK: br %r14
  call void @llvm.s390.tbeginc(i8 *null, i32 65288)
  ret void
}

; TEND with integer CC return value.
define i32 @test_tend1() {
; CHECK-LABEL: test_tend1:
; CHECK: tend
; CHECK: ipm %r2
; CHECK: srl %r2, 28
; CHECK: br %r14
  %res = call i32 @llvm.s390.tend()
  ret i32 %res
}

; TEND with implicit CC check.
define void @test_tend3(i32 *%ptr) {
; CHECK-LABEL: test_tend3:
; CHECK: tend
; CHECK: ber %r14
; CHECK: mvhi 0(%r2), 0
; CHECK: br %r14
  %res = call i32 @llvm.s390.tend()
  %cmp = icmp eq i32 %res, 2
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  store i32 0, i32* %ptr, align 4
  br label %if.end

if.end:                                           ; preds = %if.then, %entry
  ret void
}

; TEND with dual CC use.
define i32 @test_tend2(i32 %pad, i32 *%ptr) {
; CHECK-LABEL: test_tend2:
; CHECK: tend
; CHECK: ipm %r2
; CHECK: srl %r2, 28
; CHECK: ciblh %r2, 2, 0(%r14)
; CHECK: mvhi 0(%r3), 0
; CHECK: br %r14
  %res = call i32 @llvm.s390.tend()
  %cmp = icmp eq i32 %res, 2
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  store i32 0, i32* %ptr, align 4
  br label %if.end

if.end:                                           ; preds = %if.then, %entry
  ret i32 %res
}

; TABORT with register only.
define void @test_tabort1(i64 %val) {
; CHECK-LABEL: test_tabort1:
; CHECK: tabort 0(%r2)
; CHECK: br %r14
  call void @llvm.s390.tabort(i64 %val)
  ret void
}

; TABORT with immediate only.
define void @test_tabort2(i64 %val) {
; CHECK-LABEL: test_tabort2:
; CHECK: tabort 1234
; CHECK: br %r14
  call void @llvm.s390.tabort(i64 1234)
  ret void
}

; TABORT with register + immediate.
define void @test_tabort3(i64 %val) {
; CHECK-LABEL: test_tabort3:
; CHECK: tabort 1234(%r2)
; CHECK: br %r14
  %sum = add i64 %val, 1234
  call void @llvm.s390.tabort(i64 %sum)
  ret void
}

; TABORT with out-of-range immediate.
define void @test_tabort4(i64 %val) {
; CHECK-LABEL: test_tabort4:
; CHECK: tabort 0({{%r[1-5]}})
; CHECK: br %r14
  call void @llvm.s390.tabort(i64 4096)
  ret void
}

; NTSTG with base pointer only.
define void @test_ntstg1(i64 *%ptr, i64 %val) {
; CHECK-LABEL: test_ntstg1:
; CHECK: ntstg %r3, 0(%r2)
; CHECK: br %r14
  call void @llvm.s390.ntstg(i64 %val, i64 *%ptr)
  ret void
}

; NTSTG with base and index.
; Check that VSTL doesn't allow an index.
define void @test_ntstg2(i64 *%base, i64 %index, i64 %val) {
; CHECK-LABEL: test_ntstg2:
; CHECK: sllg [[REG:%r[1-5]]], %r3, 3
; CHECK: ntstg %r4, 0([[REG]],%r2)
; CHECK: br %r14
  %ptr = getelementptr i64, i64 *%base, i64 %index
  call void @llvm.s390.ntstg(i64 %val, i64 *%ptr)
  ret void
}

; NTSTG with the highest in-range displacement.
define void @test_ntstg3(i64 *%base, i64 %val) {
; CHECK-LABEL: test_ntstg3:
; CHECK: ntstg %r3, 524280(%r2)
; CHECK: br %r14
  %ptr = getelementptr i64, i64 *%base, i64 65535
  call void @llvm.s390.ntstg(i64 %val, i64 *%ptr)
  ret void
}

; NTSTG with an out-of-range positive displacement.
define void @test_ntstg4(i64 *%base, i64 %val) {
; CHECK-LABEL: test_ntstg4:
; CHECK: ntstg %r3, 0({{%r[1-5]}})
; CHECK: br %r14
  %ptr = getelementptr i64, i64 *%base, i64 65536
  call void @llvm.s390.ntstg(i64 %val, i64 *%ptr)
  ret void
}

; NTSTG with the lowest in-range displacement.
define void @test_ntstg5(i64 *%base, i64 %val) {
; CHECK-LABEL: test_ntstg5:
; CHECK: ntstg %r3, -524288(%r2)
; CHECK: br %r14
  %ptr = getelementptr i64, i64 *%base, i64 -65536
  call void @llvm.s390.ntstg(i64 %val, i64 *%ptr)
  ret void
}

; NTSTG with an out-of-range negative displacement.
define void @test_ntstg6(i64 *%base, i64 %val) {
; CHECK-LABEL: test_ntstg6:
; CHECK: ntstg %r3, 0({{%r[1-5]}})
; CHECK: br %r14
  %ptr = getelementptr i64, i64 *%base, i64 -65537
  call void @llvm.s390.ntstg(i64 %val, i64 *%ptr)
  ret void
}

; ETND.
define i32 @test_etnd() {
; CHECK-LABEL: test_etnd:
; CHECK: etnd %r2
; CHECK: br %r14
  %res = call i32 @llvm.s390.etnd()
  ret i32 %res
}

; PPA (Transaction-Abort Assist)
define void @test_ppa_txassist(i32 %val) {
; CHECK-LABEL: test_ppa_txassist:
; CHECK: ppa %r2, 0, 1
; CHECK: br %r14
  call void @llvm.s390.ppa.txassist(i32 %val)
  ret void
}

