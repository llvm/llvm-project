; Verify that fp128 intrinsics only lower to `long double` calls (e.g. sinl) on
; platforms where 128 and `long double` have the same layout. Otherwise, lower
; to f128 versions (e.g. sinf128).
;
; Targets include:
; * aarch64 (long double == f128, should use ld syms)
; * arm (long double == f64, should use f128 syms)
; * s390x (long double == f128, should use ld syms, some hardware support)
; * x86, x64 (80-bit long double, should use ld syms)
; * gnu (has f128 symbols on all platforms so we can use those)
; * musl (no f128 symbols available)
; * Windows and MacOS (no f128 symbols, long double == f64)

; FIXME(#44744): arm32, x86-{32,64} musl targets, MacOS, and Windows don't have
; f128 long double. They should be passing with CHECK-F128 rather than
; CHECK-USELD.

; RUN: %if aarch64-registered-target %{ llc < %s -mtriple=aarch64-unknown-linux-gnu    | FileCheck %s --check-prefixes=CHECK-ALL,CHECK-USELD %}
; RUN: %if aarch64-registered-target %{ llc < %s -mtriple=aarch64-unknown-linux-musl   | FileCheck %s --check-prefixes=CHECK-ALL,CHECK-USELD %}
; RUN: %if aarch64-registered-target %{ llc < %s -mtriple=aarch64-unknown-none         | FileCheck %s --check-prefixes=CHECK-ALL,CHECK-USELD %}
; RUN: %if aarch64-registered-target %{ llc < %s -mtriple=arm64-apple-macosx           | FileCheck %s --check-prefixes=CHECK-ALL,CHECK-USELD %}
; RUN: %if arm-registered-target     %{ llc < %s -mtriple=arm-none-eabi                | FileCheck %s --check-prefixes=CHECK-ALL,CHECK-USELD %}
; RUN: %if arm-registered-target     %{ llc < %s -mtriple=arm-unknown-linux-gnueabi    | FileCheck %s --check-prefixes=CHECK-ALL,CHECK-USELD %}
; RUN: %if powerpc-registered-target %{ llc < %s -mtriple=powerpc-unknown-linux-gnu    | FileCheck %s --check-prefixes=CHECK-ALL,CHECK-F128  %}
; RUN: %if powerpc-registered-target %{ llc < %s -mtriple=powerpc64-unknown-linux-gnu  | FileCheck %s --check-prefixes=CHECK-ALL,CHECK-F128  %}
; RUN: %if powerpc-registered-target %{ llc < %s -mtriple=powerpc64-unknown-linux-musl | FileCheck %s --check-prefixes=CHECK-ALL,CHECK-F128  %}
; RUN: %if riscv-registered-target   %{ llc < %s -mtriple=riscv32-unknown-linux-gnu    | FileCheck %s --check-prefixes=CHECK-ALL,CHECK-USELD %}
; RUN: %if systemz-registered-target %{ llc < %s -mtriple=s390x-unknown-linux-gnu      | FileCheck %s --check-prefixes=CHECK-ALL,CHECK-S390X %}
; RUN: %if x86-registered-target     %{ llc < %s -mtriple=i686-unknown-linux-gnu       | FileCheck %s --check-prefixes=CHECK-ALL,CHECK-F128  %}
; RUN: %if x86-registered-target     %{ llc < %s -mtriple=i686-unknown-linux-musl      | FileCheck %s --check-prefixes=CHECK-ALL,CHECK-USELD %}
; RUN: %if x86-registered-target     %{ llc < %s -mtriple=x86_64-unknown-linux-gnu     | FileCheck %s --check-prefixes=CHECK-ALL,CHECK-F128  %}
; RUN: %if x86-registered-target     %{ llc < %s -mtriple=x86_64-unknown-linux-musl    | FileCheck %s --check-prefixes=CHECK-ALL,CHECK-USELD %}
;
; FIXME(#144006): Windows-MSVC should also be run but has a ldexp selection
; failure.
; %if x86-registered-target     %{ llc < %s -mtriple=x86_64-pc-windows-msvc       -verify-machineinstrs | FileCheck %s --check-prefixes=CHECK-ALL,CHECK-F128  %}

define fp128 @test_acos(fp128 %a) {
; CHECK-ALL-LABEL:  test_acos:
; CHECK-F128:       acosf128
; CHECK-USELD:      acosl
; CHECK-S390X:      acosl
start:
  %0 = tail call fp128 @llvm.acos.f128(fp128 %a)
  ret fp128 %0
}

define fp128 @test_asin(fp128 %a) {
; CHECK-ALL-LABEL:  test_asin:
; CHECK-F128:       asinf128
; CHECK-USELD:      asinl
; CHECK-S390X:      asinl
start:
  %0 = tail call fp128 @llvm.asin.f128(fp128 %a)
  ret fp128 %0
}

define fp128 @test_atan(fp128 %a) {
; CHECK-ALL-LABEL:      test_atan:
; CHECK-F128:       atanf128
; CHECK-USELD:      atanl
; CHECK-S390X:      atanl
start:
  %0 = tail call fp128 @llvm.atan.f128(fp128 %a)
  ret fp128 %0
}

define fp128 @test_ceil(fp128 %a) {
; CHECK-ALL-LABEL:      test_ceil:
; CHECK-F128:       ceilf128
; CHECK-USELD:      ceill
; CHECK-S390X:      ceill
start:
  %0 = tail call fp128 @llvm.ceil.f128(fp128 %a)
  ret fp128 %0
}

define fp128 @test_copysign(fp128 %a, fp128 %b) {
; copysign should always get lowered to assembly.
; CHECK-ALL-LABEL:      test_copysign:
; CHECK-ALL-NOT:        copysignf128
; CHECK-ALL-NOT:        copysignl
start:
  %0 = tail call fp128 @llvm.copysign.f128(fp128 %a, fp128 %b)
  ret fp128 %0
}

define fp128 @test_cos(fp128 %a) {
; CHECK-ALL-LABEL:  test_cos:
; CHECK-F128:       cosf128
; CHECK-USELD:      cosl
; CHECK-S390X:      cosl
start:
  %0 = tail call fp128 @llvm.cos.f128(fp128 %a)
  ret fp128 %0
}

define fp128 @test_exp10(fp128 %a) {
; CHECK-ALL-LABEL:  test_exp10:
; CHECK-F128:       exp10f128
; CHECK-USELD:      exp10l
; CHECK-S390X:      exp10l
start:
  %0 = tail call fp128 @llvm.exp10.f128(fp128 %a)
  ret fp128 %0
}

define fp128 @test_exp2(fp128 %a) {
; CHECK-ALL-LABEL:  test_exp2:
; CHECK-F128:       exp2f128
; CHECK-USELD:      exp2l
; CHECK-S390X:      exp2l
start:
  %0 = tail call fp128 @llvm.exp2.f128(fp128 %a)
  ret fp128 %0
}


define fp128 @test_exp(fp128 %a) {
; CHECK-ALL-LABEL:  test_exp:
; CHECK-F128:       expf128
; CHECK-USELD:      expl
; CHECK-S390X:      expl
start:
  %0 = tail call fp128 @llvm.exp.f128(fp128 %a)
  ret fp128 %0
}

define fp128 @test_fabs(fp128 %a) {
; fabs should always get lowered to assembly.
; CHECK-ALL-LABEL:  test_fabs:
; CHECK-ALL-NOT:    fabsf128
; CHECK-ALL-NOT:    fabsl
start:
  %0 = tail call fp128 @llvm.fabs.f128(fp128 %a)
  ret fp128 %0
}

define fp128 @test_floor(fp128 %a) {
; CHECK-ALL-LABEL:  test_floor:
; CHECK-F128:       floorf128
; CHECK-USELD:      floorl
; CHECK-S390X:      floorl
start:
  %0 = tail call fp128 @llvm.floor.f128(fp128 %a)
  ret fp128 %0
}

define fp128 @test_fma(fp128 %a, fp128 %b, fp128 %c) {
; CHECK-ALL-LABEL:  test_fma:
; CHECK-F128:       fmaf128
; CHECK-USELD:      fmal
; CHECK-S390X:      fmal
start:
  %0 = tail call fp128 @llvm.fma.f128(fp128 %a, fp128 %b, fp128 %c)
  ret fp128 %0
}

define { fp128, i32 } @test_frexp(fp128 %a) {
; CHECK-ALL-LABEL:  test_frexp:
; CHECK-F128:       frexpf128
; CHECK-USELD:      frexpl
; CHECK-S390X:      frexpl
start:
  %0 = tail call { fp128, i32 } @llvm.frexp.f128(fp128 %a)
  ret { fp128, i32 } %0
}

define fp128 @test_ldexp(fp128 %a, i32 %b) {
; CHECK-ALL-LABEL:  test_ldexp:
; CHECK-F128:       ldexpf128
; CHECK-USELD:      ldexpl
; CHECK-S390X:      ldexpl
start:
  %0 = tail call fp128 @llvm.ldexp.f128(fp128 %a, i32 %b)
  ret fp128 %0
}

define i64 @test_llrint(fp128 %a) {
; CHECK-ALL-LABEL:  test_llrint:
; CHECK-F128:       llrintf128
; CHECK-USELD:      llrintl
; CHECK-S390X:      llrintl
start:
  %0 = tail call i64 @llvm.llrint.f128(fp128 %a)
  ret i64 %0
}

define i64 @test_llround(fp128 %a) {
; CHECK-ALL-LABEL:  test_llround:
; CHECK-F128:       llroundf128
; CHECK-USELD:      llroundl
; CHECK-S390X:      llroundl
start:
  %0 = tail call i64 @llvm.llround.i64.f128(fp128 %a)
  ret i64 %0
}

define fp128 @test_log10(fp128 %a) {
; CHECK-ALL-LABEL:  test_log10:
; CHECK-F128:       log10f128
; CHECK-USELD:      log10l
; CHECK-S390X:      log10l
start:
  %0 = tail call fp128 @llvm.log10.f128(fp128 %a)
  ret fp128 %0
}

define fp128 @test_log2(fp128 %a) {
; CHECK-ALL-LABEL:  test_log2:
; CHECK-F128:       log2f128
; CHECK-USELD:      log2l
; CHECK-S390X:      log2l
start:
  %0 = tail call fp128 @llvm.log2.f128(fp128 %a)
  ret fp128 %0
}

define fp128 @test_log(fp128 %a) {
; CHECK-ALL-LABEL:  test_log:
; CHECK-F128:       logf128
; CHECK-USELD:      logl
; CHECK-S390X:      logl
start:
  %0 = tail call fp128 @llvm.log.f128(fp128 %a)
  ret fp128 %0
}

define i64 @test_lrint(fp128 %a) {
; CHECK-ALL-LABEL:  test_lrint:
; CHECK-F128:       lrintf128
; CHECK-USELD:      lrintl
; CHECK-S390X:      lrintl
start:
  %0 = tail call i64 @llvm.lrint.f128(fp128 %a)
  ret i64 %0
}

define i64 @test_lround(fp128 %a) {
; CHECK-ALL-LABEL:  test_lround:
; CHECK-F128:       lroundf128
; CHECK-USELD:      lroundl
; CHECK-S390X:      lroundl
start:
  %0 = tail call i64 @llvm.lround.i64.f128(fp128 %a)
  ret i64 %0
}

define fp128 @test_nearbyint(fp128 %a) {
; CHECK-ALL-LABEL:  test_nearbyint:
; CHECK-F128:       nearbyintf128
; CHECK-USELD:      nearbyintl
; CHECK-S390X:      nearbyintl
start:
  %0 = tail call fp128 @llvm.nearbyint.f128(fp128 %a)
  ret fp128 %0
}

define fp128 @test_pow(fp128 %a, fp128 %b) {
; CHECK-ALL-LABEL:  test_pow:
; CHECK-F128:       powf128
; CHECK-USELD:      powl
; CHECK-S390X:      powl
start:
  %0 = tail call fp128 @llvm.pow.f128(fp128 %a, fp128 %b)
  ret fp128 %0
}

define fp128 @test_rint(fp128 %a) {
; CHECK-ALL-LABEL:  test_rint:
; CHECK-F128:       rintf128
; CHECK-USELD:      rintl
; CHECK-S390X:      fixbr {{%.*}}, 0, {{%.*}}
start:
  %0 = tail call fp128 @llvm.rint.f128(fp128 %a)
  ret fp128 %0
}

define fp128 @test_roundeven(fp128 %a) {
; CHECK-ALL-LABEL:  test_roundeven:
; CHECK-F128:       roundevenf128
; CHECK-USELD:      roundevenl
; CHECK-S390X:      roundevenl
start:
  %0 = tail call fp128 @llvm.roundeven.f128(fp128 %a)
  ret fp128 %0
}

define fp128 @test_round(fp128 %a) {
; CHECK-ALL-LABEL:  test_round:
; CHECK-F128:       roundf128
; CHECK-USELD:      roundl
; CHECK-S390X:      roundl
start:
  %0 = tail call fp128 @llvm.round.f128(fp128 %a)
  ret fp128 %0
}

define fp128 @test_sin(fp128 %a) {
; CHECK-ALL-LABEL:  test_sin:
; CHECK-F128:       sinf128
; CHECK-USELD:      sinl
; CHECK-S390X:      sinl
start:
  %0 = tail call fp128 @llvm.sin.f128(fp128 %a)
  ret fp128 %0
}

define fp128 @test_sqrt(fp128 %a) {
; CHECK-ALL-LABEL:  test_sqrt:
; CHECK-F128:       sqrtf128
; CHECK-USELD:      sqrtl
; CHECK-S390X:      sqxbr {{%.*}}, {{%.*}}
start:
  %0 = tail call fp128 @llvm.sqrt.f128(fp128 %a)
  ret fp128 %0
}

define fp128 @test_tan(fp128 %a) {
; CHECK-ALL-LABEL:  test_tan:
; CHECK-F128:       tanf128
; CHECK-USELD:      tanl
; CHECK-S390X:      tanl
start:
  %0 = tail call fp128 @llvm.tan.f128(fp128 %a)
  ret fp128 %0
}

define fp128 @test_trunc(fp128 %a) {
; CHECK-ALL-LABEL:  test_trunc:
; CHECK-F128:       truncf128
; CHECK-USELD:      truncl
; CHECK-S390X:      truncl
start:
  %0 = tail call fp128 @llvm.trunc.f128(fp128 %a)
  ret fp128 %0
}
