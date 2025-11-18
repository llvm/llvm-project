; Same as `bfloat.ll`, but for `fneg`, `fabs`, `copysign` and `fma`.
; Can be merged back into `bfloat.ll` once they have the same platform coverage.
; Once all targets are fixed, the `CHECK-*` prefixes should all be merged into a single `CHECK` prefix and the `BAD-*` prefixes should be removed.

; RUN: %if aarch64-registered-target     %{ llc %s -o - -mtriple=aarch64-apple-darwin            | FileCheck %s --check-prefixes=ALL,CHECK-COPYSIGN,CHECK-FMA %}
; RUN: %if aarch64-registered-target     %{ llc %s -o - -mtriple=aarch64-pc-windows-msvc         | FileCheck %s --check-prefixes=ALL,CHECK-COPYSIGN,CHECK-FMA %}
; RUN: %if aarch64-registered-target     %{ llc %s -o - -mtriple=aarch64-unknown-linux-gnu       | FileCheck %s --check-prefixes=ALL,CHECK-COPYSIGN,CHECK-FMA %}
; RUN: %if aarch64-registered-target     %{ llc %s -o - -mtriple=arm64ec-pc-windows-msvc         | FileCheck %s --check-prefixes=ALL,CHECK-COPYSIGN,CHECK-FMA %}
; RUN: %if amdgpu-registered-target      %{ llc %s -o - -mtriple=amdgcn-amd-amdhsa               | FileCheck %s --check-prefixes=ALL,CHECK-COPYSIGN,CHECK-FMA %}
; RUN: %if arc-registered-target         %{ llc %s -o - -mtriple=arc-elf                         | FileCheck %s --check-prefixes=ALL,CHECK-COPYSIGN,BAD-FMA %}
; RUN: %if arm-registered-target         %{ llc %s -o - -mtriple=arm-unknown-linux-gnueabi       | FileCheck %s --check-prefixes=ALL,CHECK-COPYSIGN,BAD-FMA %}
; RUN: %if arm-registered-target         %{ llc %s -o - -mtriple=thumbv7em-none-eabi             | FileCheck %s --check-prefixes=ALL,CHECK-COPYSIGN,BAD-FMA %}
; RUN: %if avr-registered-target         %{ llc %s -o - -mtriple=avr-none                        | FileCheck %s --check-prefixes=ALL,CHECK-COPYSIGN,BAD-FMA %}
; FIXME: BPF has a compiler error
; RUN: %if csky-registered-target        %{ llc %s -o - -mtriple=csky-unknown-linux-gnuabiv2     | FileCheck %s --check-prefixes=ALL,CHECK-COPYSIGN,BAD-FMA %}
; FIXME: hard float csky crashes
; FIXME: directx has a compiler error
; FIXME: hexagon crashes
; RUN: %if lanai-registered-target       %{ llc %s -o - -mtriple=lanai-unknown-unknown           | FileCheck %s --check-prefixes=ALL,CHECK-COPYSIGN,BAD-FMA %}
; RUN: %if loongarch-registered-target   %{ llc %s -o - -mtriple=loongarch32-unknown-linux-gnu   | FileCheck %s --check-prefixes=ALL,CHECK-COPYSIGN,BAD-FMA %}
; RUN: %if loongarch-registered-target   %{ llc %s -o - -mtriple=loongarch64-unknown-linux-gnu   | FileCheck %s --check-prefixes=ALL,CHECK-COPYSIGN,BAD-FMA %}
; RUN: %if loongarch-registered-target   %{ llc %s -o - -mtriple=loongarch64-unknown-linux-gnu -mattr=+f | FileCheck %s --check-prefixes=ALL,CHECK-COPYSIGN,BAD-FMA %}
; RUN: %if m68k-registered-target        %{ llc %s -o - -mtriple=m68k-unknown-linux-gnu          | FileCheck %s --check-prefixes=ALL,CHECK-COPYSIGN,BAD-FMA %}
; FIXME: mips crashes
; RUN: %if msp430-registered-target      %{ llc %s -o - -mtriple=msp430-none-elf                 | FileCheck %s --check-prefixes=ALL,CHECK-COPYSIGN,BAD-FMA %}
; RUN: %if nvptx-registered-target       %{ llc %s -o - -mtriple=nvptx64-nvidia-cuda             | FileCheck %s --check-prefixes=NOCRASH %}
; FIXME: powerpc crashes
; RUN: %if riscv-registered-target       %{ llc %s -o - -mtriple=riscv32-unknown-linux-gnu       | FileCheck %s --check-prefixes=ALL,CHECK-COPYSIGN,BAD-FMA %}
; RUN: %if riscv-registered-target       %{ llc %s -o - -mtriple=riscv64-unknown-linux-gnu       | FileCheck %s --check-prefixes=ALL,CHECK-COPYSIGN,BAD-FMA %}
; FIXME: sparc crashes
; FIXME: spirv crashes
; FIXME: s390x crashes
; FIXME: ve crashes
; FIXME: wasm crashes
; RUN: %if x86-registered-target         %{ llc %s -o - -mtriple=i686-unknown-linux-gnu          | FileCheck %s --check-prefixes=ALL,CHECK-COPYSIGN,BAD-FMA %}
; RUN: %if x86-registered-target         %{ llc %s -o - -mtriple=x86_64-pc-windows-msvc          | FileCheck %s --check-prefixes=ALL,CHECK-COPYSIGN,BAD-FMA %}
; RUN: %if x86-registered-target         %{ llc %s -o - -mtriple=x86_64-unknown-linux-gnu        | FileCheck %s --check-prefixes=ALL,CHECK-COPYSIGN,BAD-FMA %}
; RUN: %if xcore-registered-target       %{ llc %s -o - -mtriple=xcore-unknown-unknown           | FileCheck %s --check-prefixes=ALL,CHECK-COPYSIGN,BAD-FMA %}
; RUN: %if xtensa-registered-target      %{ llc %s -o - -mtriple=xtensa-none-elf                 | FileCheck %s --check-prefixes=ALL,BAD-COPYSIGN,CHECK-FMA %}

; Note that arm64ec labels are quoted, hence the `{{"?}}:`.

; Codegen tests don't work the same for graphics targets. Add a dummy directive
; for filecheck, just make sure we don't crash.
; NOCRASH: {{.*}}

; fneg, fabs and copysign all need to not quieten signalling NaNs, so should not call any conversion functions which do.
; These tests won't catch cases where the everything is done using native instructions instead of builtins.

define void @test_fneg(ptr %p1, ptr %p2) #0 {
; ALL-LABEL: test_fneg{{"?}}:
; ALL-NEG-NOT: __extend
; ALL-NEG-NOT: __trunc
; ALL-NEG-NOT: __gnu
; ALL-NEG-NOT: __aeabi
  %v = load bfloat, ptr %p1
  %res = fneg bfloat %v
  store bfloat %res, ptr %p2
  ret void
}

define void @test_fabs(ptr %p1, ptr %p2) {
; ALL-LABEL: test_fabs{{"?}}:
; ALL-ABS-NOT: __extend
; ALL-ABS-NOT: __trunc
; ALL-ABS-NOT: __gnu
; ALL-ABS-NOT: __aeabi
  %a = load bfloat, ptr %p1
  %r = call bfloat @llvm.fabs.f16(bfloat %a)
  store bfloat %r, ptr %p2
  ret void
}

define void @test_copysign(ptr %p1, ptr %p2, ptr %p3) {
; ALL-LABEL: test_copysign{{"?}}:
; CHECK-COPYSIGN-NOT: __extend
; CHECK-COPYSIGN-NOT: __trunc
; CHECK-COPYSIGN-NOT: __gnu
; CHECK-COPYSIGN-NOT: __aeabi
; BAD-COPYSIGN: __truncsfbf2
  %a = load bfloat, ptr %p1
  %b = load bfloat, ptr %p2
  %r = call bfloat @llvm.copysign.f16(bfloat %a, bfloat %b)
  store bfloat %r, ptr %p3
  ret void
}

; There is no floating-point type LLVM supports that is large enough to promote bfloat FMA to
; without causing double rounding issues. This checks for libcalls to f32/f64 fma and truncating
; f32/f64 to bf16. See https://github.com/llvm/llvm-project/issues/131531

define void @test_fma(ptr %p1, ptr %p2, ptr %p3, ptr %p4) {
; ALL-LABEL: test_fma{{"?}}:
; CHECK-FMA-NOT: {{\bfmaf?\b}}
; CHECK-FMA-NOT: __truncsfbf2
; CHECK-FMA-NOT: __truncdfbf2
; BAD-FMA: {{__truncsfbf2|\bfmaf?\b}}
  %a = load bfloat, ptr %p1
  %b = load bfloat, ptr %p2
  %c = load bfloat, ptr %p3
  %r = call bfloat @llvm.fma.f16(bfloat %a, bfloat %b, bfloat %c)
  store bfloat %r, ptr %p4
  ret void
}
