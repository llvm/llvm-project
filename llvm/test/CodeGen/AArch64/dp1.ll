; RUN: llc -verify-machineinstrs -o - %s -mtriple=aarch64-linux-gnu | FileCheck %s
; RUN: llc -global-isel -global-isel-abort=2 -pass-remarks-missed=gisel* -verify-machineinstrs -o - %s -mtriple=aarch64-linux-gnu | FileCheck %s --check-prefixes=FALLBACK,GISEL

@var32 = global i32 0
@var64 = global i64 0

; FALLBACK-NOT: remark{{.*}}rev_i32
define void @rev_i32() {
; CHECK-LABEL: rev_i32:
; GISEL-LABEL: rev_i32:
    %val0_tmp = load i32, i32* @var32
    %val1_tmp = call i32 @llvm.bswap.i32(i32 %val0_tmp)
; CHECK: rev	{{w[0-9]+}}, {{w[0-9]+}}
; GISEL: rev	{{w[0-9]+}}, {{w[0-9]+}}
    store volatile i32 %val1_tmp, i32* @var32
    ret void
}

; FALLBACK-NOT: remark{{.*}}rev_i64
define void @rev_i64() {
; CHECK-LABEL: rev_i64:
; GISEL-LABEL: rev_i64:
    %val0_tmp = load i64, i64* @var64
    %val1_tmp = call i64 @llvm.bswap.i64(i64 %val0_tmp)
; CHECK: rev	{{x[0-9]+}}, {{x[0-9]+}}
; GISEL: rev	{{x[0-9]+}}, {{x[0-9]+}}
    store volatile i64 %val1_tmp, i64* @var64
    ret void
}

define void @rev32_i64() {
; CHECK-LABEL: rev32_i64:
    %val0_tmp = load i64, i64* @var64
    %val1_tmp = shl i64 %val0_tmp, 32
    %val5_tmp = sub i64 64, 32
    %val2_tmp = lshr i64 %val0_tmp, %val5_tmp
    %val3_tmp = or i64 %val1_tmp, %val2_tmp
    %val4_tmp = call i64 @llvm.bswap.i64(i64 %val3_tmp)
; CHECK: rev32	{{x[0-9]+}}, {{x[0-9]+}}
    store volatile i64 %val4_tmp, i64* @var64
    ret void
}

define void @rev16_i32() {
; CHECK-LABEL: rev16_i32:
    %val0_tmp = load i32, i32* @var32
    %val1_tmp = shl i32 %val0_tmp, 16
    %val2_tmp = lshr i32 %val0_tmp, 16
    %val3_tmp = or i32 %val1_tmp, %val2_tmp
    %val4_tmp = call i32 @llvm.bswap.i32(i32 %val3_tmp)
; CHECK: rev16	{{w[0-9]+}}, {{w[0-9]+}}
    store volatile i32 %val4_tmp, i32* @var32
    ret void
}

define void @clz_zerodef_i32() {
; CHECK-LABEL: clz_zerodef_i32:
    %val0_tmp = load i32, i32* @var32
    %val4_tmp = call i32 @llvm.ctlz.i32(i32 %val0_tmp, i1 0)
; CHECK: clz	{{w[0-9]+}}, {{w[0-9]+}}
    store volatile i32 %val4_tmp, i32* @var32
    ret void
}

define void @clz_zerodef_i64() {
; CHECK-LABEL: clz_zerodef_i64:
    %val0_tmp = load i64, i64* @var64
    %val4_tmp = call i64 @llvm.ctlz.i64(i64 %val0_tmp, i1 0)
; CHECK: clz	{{x[0-9]+}}, {{x[0-9]+}}
    store volatile i64 %val4_tmp, i64* @var64
    ret void
}

define void @clz_zeroundef_i32() {
; CHECK-LABEL: clz_zeroundef_i32:
    %val0_tmp = load i32, i32* @var32
    %val4_tmp = call i32 @llvm.ctlz.i32(i32 %val0_tmp, i1 1)
; CHECK: clz	{{w[0-9]+}}, {{w[0-9]+}}
    store volatile i32 %val4_tmp, i32* @var32
    ret void
}

define void @clz_zeroundef_i64() {
; CHECK-LABEL: clz_zeroundef_i64:
    %val0_tmp = load i64, i64* @var64
    %val4_tmp = call i64 @llvm.ctlz.i64(i64 %val0_tmp, i1 1)
; CHECK: clz	{{x[0-9]+}}, {{x[0-9]+}}
    store volatile i64 %val4_tmp, i64* @var64
    ret void
}

define void @cttz_zerodef_i32() {
; CHECK-LABEL: cttz_zerodef_i32:
    %val0_tmp = load i32, i32* @var32
    %val4_tmp = call i32 @llvm.cttz.i32(i32 %val0_tmp, i1 0)
; CHECK: rbit   [[REVERSED:w[0-9]+]], {{w[0-9]+}}
; CHECK: clz	{{w[0-9]+}}, [[REVERSED]]
    store volatile i32 %val4_tmp, i32* @var32
    ret void
}

define void @cttz_zerodef_i64() {
; CHECK-LABEL: cttz_zerodef_i64:
    %val0_tmp = load i64, i64* @var64
    %val4_tmp = call i64 @llvm.cttz.i64(i64 %val0_tmp, i1 0)
; CHECK: rbit   [[REVERSED:x[0-9]+]], {{x[0-9]+}}
; CHECK: clz	{{x[0-9]+}}, [[REVERSED]]
    store volatile i64 %val4_tmp, i64* @var64
    ret void
}

define void @cttz_zeroundef_i32() {
; CHECK-LABEL: cttz_zeroundef_i32:
    %val0_tmp = load i32, i32* @var32
    %val4_tmp = call i32 @llvm.cttz.i32(i32 %val0_tmp, i1 1)
; CHECK: rbit   [[REVERSED:w[0-9]+]], {{w[0-9]+}}
; CHECK: clz	{{w[0-9]+}}, [[REVERSED]]
    store volatile i32 %val4_tmp, i32* @var32
    ret void
}

define void @cttz_zeroundef_i64() {
; CHECK-LABEL: cttz_zeroundef_i64:
    %val0_tmp = load i64, i64* @var64
    %val4_tmp = call i64 @llvm.cttz.i64(i64 %val0_tmp, i1 1)
; CHECK: rbit   [[REVERSED:x[0-9]+]], {{x[0-9]+}}
; CHECK: clz	{{x[0-9]+}}, [[REVERSED]]
    store volatile i64 %val4_tmp, i64* @var64
    ret void
}

; These two are just compilation tests really: the operation's set to Expand in
; ISelLowering.
define void @ctpop_i32() {
; CHECK-LABEL: ctpop_i32:
    %val0_tmp = load i32, i32* @var32
    %val4_tmp = call i32 @llvm.ctpop.i32(i32 %val0_tmp)
    store volatile i32 %val4_tmp, i32* @var32
    ret void
}

define void @ctpop_i64() {
; CHECK-LABEL: ctpop_i64:
    %val0_tmp = load i64, i64* @var64
    %val4_tmp = call i64 @llvm.ctpop.i64(i64 %val0_tmp)
    store volatile i64 %val4_tmp, i64* @var64
    ret void
}


declare i32 @llvm.bswap.i32(i32)
declare i64 @llvm.bswap.i64(i64)
declare i32  @llvm.ctlz.i32 (i32, i1)
declare i64  @llvm.ctlz.i64 (i64, i1)
declare i32  @llvm.cttz.i32 (i32, i1)
declare i64  @llvm.cttz.i64 (i64, i1)
declare i32  @llvm.ctpop.i32 (i32)
declare i64  @llvm.ctpop.i64 (i64)
