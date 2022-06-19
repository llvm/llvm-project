; RUN: opt < %s  -passes="print<cost-model>" 2>&1 -disable-output -mtriple=i386 -mcpu=corei7-avx | FileCheck %s -check-prefix=CHECK32
; RUN: opt < %s  -passes="print<cost-model>" 2>&1 -disable-output -mtriple=x86_64-apple-macosx10.8.0 -mcpu=corei7-avx | FileCheck %s -check-prefix=CHECK64

; Test vector scalarization costs.
; RUN: llc < %s -march=x86 -mcpu=i386
; RUN: llc < %s -march=x86 -mcpu=yonah

%i4 = type <4 x i32>
%i8 = type <2 x i64>

;;; TEST HANDLING OF VARIOUS VECTOR SIZES

declare %i4 @llvm.bswap.v4i32(%i4)
declare %i8 @llvm.bswap.v2i64(%i8)

declare %i4 @llvm.cttz.v4i32(%i4)
declare %i8 @llvm.cttz.v2i64(%i8)

; CHECK32-LABEL: test_scalarized_intrinsics
; CHECK64-LABEL: test_scalarized_intrinsics
define void @test_scalarized_intrinsics() {
        %r1 = add %i8 undef, undef

; CHECK32: cost of 1 {{.*}}bswap.v4i32
; CHECK64: cost of 1 {{.*}}bswap.v4i32
        %r2 = call %i4 @llvm.bswap.v4i32(%i4 undef)
; CHECK32: cost of 1 {{.*}}bswap.v2i64
; CHECK64: cost of 1 {{.*}}bswap.v2i64
        %r3 = call %i8 @llvm.bswap.v2i64(%i8 undef)

; CHECK32: cost of 14 {{.*}}cttz.v4i32
; CHECK64: cost of 14 {{.*}}cttz.v4i32
        %r4 = call %i4 @llvm.cttz.v4i32(%i4 undef)
; CHECK32: cost of 10 {{.*}}cttz.v2i64
; CHECK64: cost of 10 {{.*}}cttz.v2i64
        %r5 = call %i8 @llvm.cttz.v2i64(%i8 undef)

; CHECK32: ret
; CHECK64: ret
        ret void
}
