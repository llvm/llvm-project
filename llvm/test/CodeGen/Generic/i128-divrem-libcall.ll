; 64-bit targets: fused __divmodti4 / __udivmodti4
; RUN: %if x86-registered-target			   %{ llc < %s -mtriple=x86_64-linux-gnu              | FileCheck %s --check-prefixes=CHECK,FUSED %}
; RUN: %if x86-registered-target         %{ llc < %s -mtriple=x86_64-pc-windows-msvc        | FileCheck %s --check-prefixes=CHECK,FUSED %}
; RUN: %if aarch64-registered-target     %{ llc < %s -mtriple=aarch64-linux-gnu             | FileCheck %s --check-prefixes=CHECK,FUSED %}
; RUN: %if riscv-registered-target       %{ llc < %s -mtriple=riscv64-linux-gnu             | FileCheck %s --check-prefixes=CHECK,FUSED %}
; RUN: %if riscv-registered-target       %{ llc < %s -mtriple=riscv64-linux-gnu -mattr=+m   | FileCheck %s --check-prefixes=CHECK,FUSED %}
; RUN: %if powerpc-registered-target     %{ llc < %s -mtriple=powerpc64-linux-gnu           | FileCheck %s --check-prefixes=CHECK,FUSED %}
; RUN: %if sparc-registered-target       %{ llc < %s -mtriple=sparcv9-linux-gnu             | FileCheck %s --check-prefixes=CHECK,FUSED %}
; RUN: %if loongarch-registered-target   %{ llc < %s -mtriple=loongarch64-linux-gnu         | FileCheck %s --check-prefixes=CHECK,FUSED %}
; RUN: %if webassembly-registered-target %{ llc < %s -mtriple=wasm32-unknown-unknown        | FileCheck %s --check-prefixes=CHECK,FUSED %}
; RUN: %if webassembly-registered-target %{ llc < %s -mtriple=wasm64-unknown-unknown        | FileCheck %s --check-prefixes=CHECK,FUSED %}

; 32-bit / ILP32 targets: no fused libcall 
; RUN: %if x86-registered-target      %{ llc < %s -mtriple=i386-linux-gnu                | FileCheck %s --check-prefixes=CHECK,SPLIT %}
; RUN: %if x86-registered-target      %{ llc < %s -mtriple=i686-linux-gnu                | FileCheck %s --check-prefixes=CHECK,SPLIT %}
; RUN: %if arm-registered-target      %{ llc < %s -mtriple=armv6-linux-gnueabihf         | FileCheck %s --check-prefixes=CHECK,SPLIT %}
; RUN: %if arm-registered-target      %{ llc < %s -mtriple=armv7-linux-gnueabi           | FileCheck %s --check-prefixes=CHECK,SPLIT %}
; RUN: %if aarch64-registered-target  %{ llc < %s -mtriple=aarch64_32-apple-watchos      | FileCheck %s --check-prefixes=CHECK,SPLIT %}
; RUN: %if riscv-registered-target    %{ llc < %s -mtriple=riscv32-linux-gnu             | FileCheck %s --check-prefixes=CHECK,SPLIT %}
; RUN: %if riscv-registered-target    %{ llc < %s -mtriple=riscv32-linux-gnu -mattr=+m   | FileCheck %s --check-prefixes=CHECK,SPLIT %}
; RUN: %if arm-registered-target      %{ llc < %s -mtriple=armv7-none-eabi               | FileCheck %s --check-prefixes=CHECK,SPLIT %}

; Verify that sdiv+srem / udiv+urem on i128 fuse into a single __divmodti4 /
; __udivmodti4 call on targets where the libcall is available (64-bit targets
; and wasm), and do not on targets where it is not (32-bit / ILP32).
;
; The lowering varies by target:
;   64-bit targets and wasm: fused __divmodti4 / __udivmodti4
;   32-bit targets that lack the fused call may lower to:
;     - separate __divti3 + __modti3 / __udivti3 + __umodti3 calls, or
;     - fully inline expansion (e.g. i686)

define void @sdivrem_i128(ptr %q_out, ptr %r_out, i128 %n, i128 %d) {
; CHECK-LABEL: sdivrem_i128:
; FUSED:           __divmodti4
; SPLIT-NOT:       __divmodti4
  %q = sdiv i128 %n, %d
  %r = srem i128 %n, %d
  store i128 %q, ptr %q_out
  store i128 %r, ptr %r_out
  ret void
}

define void @udivrem_i128(ptr %q_out, ptr %r_out, i128 %n, i128 %d) {
; CHECK-LABEL: udivrem_i128:
; FUSED:           __udivmodti4
; SPLIT-NOT:       __udivmodti4
  %q = udiv i128 %n, %d
  %r = urem i128 %n, %d
  store i128 %q, ptr %q_out
  store i128 %r, ptr %r_out
  ret void
}
