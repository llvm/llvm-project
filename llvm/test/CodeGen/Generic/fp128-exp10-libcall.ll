; RUN: %if aarch64-registered-target %{ llc < %s -mtriple=aarch64-unknown-linux-gnu    | FileCheck %s --check-prefixes=CHECK-ALL,CHECK-USELD %}
; RUN: %if aarch64-registered-target %{ llc < %s -mtriple=aarch64-unknown-linux-musl   | FileCheck %s --check-prefixes=CHECK-ALL,CHECK-USELD %}
; RUN: %if aarch64-registered-target %{ llc < %s -mtriple=aarch64-unknown-none         | FileCheck %s --check-prefixes=CHECK-ALL,CHECK-USELD %}
; RUN: %if aarch64-registered-target %{ not llc -mtriple=arm64-apple-macosx -filetype=null %s 2>&1 | FileCheck --check-prefix=ERR %s %}
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
; RUN %if x86-registered-target     %{ llc < %s -mtriple=x86_64-pc-windows-msvc        | FileCheck %s --check-prefixes=CHECK-ALL,CHECK-F128  %}

; ERR: error: no libcall available for fexp10
define fp128 @test_exp10(fp128 %a) {
; CHECK-ALL-LABEL:  test_exp10:
; CHECK-F128:       exp10f128
; CHECK-USELD:      exp10l
; CHECK-S390X:      exp10l
start:
  %0 = tail call fp128 @llvm.exp10.f128(fp128 %a)
  ret fp128 %0
}

