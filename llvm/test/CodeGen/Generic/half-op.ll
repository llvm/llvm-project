; Same as `half.ll`, but for `fneg`, `fabs`, `copysign` and `fma`.
; Can be merged back into `half.ll` once BPF doesn't have a compiler error.

; RUN: %if aarch64-registered-target     %{ llc %s -o - -mtriple=aarch64-apple-darwin            | FileCheck %s --check-prefixes=ALL %}
; RUN: %if aarch64-registered-target     %{ llc %s -o - -mtriple=aarch64-pc-windows-msvc         | FileCheck %s --check-prefixes=ALL %}
; RUN: %if aarch64-registered-target     %{ llc %s -o - -mtriple=aarch64-unknown-linux-gnu       | FileCheck %s --check-prefixes=ALL %}
; RUN: %if aarch64-registered-target     %{ llc %s -o - -mtriple=arm64ec-pc-windows-msvc         | FileCheck %s --check-prefixes=ALL %}
; RUN: %if amdgpu-registered-target      %{ llc %s -o - -mtriple=amdgcn-amd-amdhsa               | FileCheck %s --check-prefixes=ALL %}
; RUN: %if arc-registered-target         %{ llc %s -o - -mtriple=arc-elf                         | FileCheck %s --check-prefixes=ALL %}
; RUN: %if arm-registered-target         %{ llc %s -o - -mtriple=arm-unknown-linux-gnueabi       | FileCheck %s --check-prefixes=ALL %}
; RUN: %if arm-registered-target         %{ llc %s -o - -mtriple=thumbv7em-none-eabi             | FileCheck %s --check-prefixes=ALL %}
; RUN: %if avr-registered-target         %{ llc %s -o - -mtriple=avr-none                        | FileCheck %s --check-prefixes=ALL %}
; FIXME: BPF has a compiler error
; RUN: %if csky-registered-target        %{ llc %s -o - -mtriple=csky-unknown-linux-gnuabiv2     | FileCheck %s --check-prefixes=ALL %}
; RUN: %if csky-registered-target        %{ llc %s -o - -mtriple=csky-unknown-linux-gnuabiv2 -mcpu=ck860fv -mattr=+hard-float | FileCheck %s --check-prefixes=ALL %}
; FIXME: directx has a compiler error
; RUN: %if hexagon-registered-target     %{ llc %s -o - -mtriple=hexagon-unknown-linux-musl      | FileCheck %s --check-prefixes=ALL %}
; RUN: %if lanai-registered-target       %{ llc %s -o - -mtriple=lanai-unknown-unknown           | FileCheck %s --check-prefixes=ALL %}
; RUN: %if loongarch-registered-target   %{ llc %s -o - -mtriple=loongarch32-unknown-linux-gnu   | FileCheck %s --check-prefixes=ALL %}
; RUN: %if loongarch-registered-target   %{ llc %s -o - -mtriple=loongarch64-unknown-linux-gnu   | FileCheck %s --check-prefixes=ALL %}
; RUN: %if loongarch-registered-target   %{ llc %s -o - -mtriple=loongarch64-unknown-linux-gnu -mattr=+f | FileCheck %s --check-prefixes=ALL %}
; RUN: %if m68k-registered-target        %{ llc %s -o - -mtriple=m68k-unknown-linux-gnu          | FileCheck %s --check-prefixes=ALL %}
; RUN: %if mips-registered-target        %{ llc %s -o - -mtriple=mips-unknown-linux-gnu          | FileCheck %s --check-prefixes=ALL %}
; RUN: %if mips-registered-target        %{ llc %s -o - -mtriple=mips64-unknown-linux-gnuabi64   | FileCheck %s --check-prefixes=ALL %}
; RUN: %if mips-registered-target        %{ llc %s -o - -mtriple=mips64el-unknown-linux-gnuabi64 | FileCheck %s --check-prefixes=ALL %}
; RUN: %if mips-registered-target        %{ llc %s -o - -mtriple=mipsel-unknown-linux-gnu        | FileCheck %s --check-prefixes=ALL %}
; RUN: %if msp430-registered-target      %{ llc %s -o - -mtriple=msp430-none-elf                 | FileCheck %s --check-prefixes=ALL %}
; RUN: %if nvptx-registered-target       %{ llc %s -o - -mtriple=nvptx64-nvidia-cuda             | FileCheck %s --check-prefixes=NOCRASH %}
; RUN: %if powerpc-registered-target     %{ llc %s -o - -mtriple=powerpc-unknown-linux-gnu       | FileCheck %s --check-prefixes=ALL %}
; RUN: %if powerpc-registered-target     %{ llc %s -o - -mtriple=powerpc64-unknown-linux-gnu     | FileCheck %s --check-prefixes=ALL %}
; RUN: %if powerpc-registered-target     %{ llc %s -o - -mtriple=powerpc64le-unknown-linux-gnu   | FileCheck %s --check-prefixes=ALL %}
; RUN: %if riscv-registered-target       %{ llc %s -o - -mtriple=riscv32-unknown-linux-gnu       | FileCheck %s --check-prefixes=ALL %}
; RUN: %if riscv-registered-target       %{ llc %s -o - -mtriple=riscv64-unknown-linux-gnu       | FileCheck %s --check-prefixes=ALL %}
; RUN: %if sparc-registered-target       %{ llc %s -o - -mtriple=sparc-unknown-linux-gnu         | FileCheck %s --check-prefixes=ALL %}
; RUN: %if sparc-registered-target       %{ llc %s -o - -mtriple=sparc64-unknown-linux-gnu       | FileCheck %s --check-prefixes=ALL %}
; RUN: %if spirv-registered-target       %{ llc %s -o - -mtriple=spirv-unknown-unknown           | FileCheck %s --check-prefixes=NOCRASH %}
; RUN: %if systemz-registered-target     %{ llc %s -o - -mtriple=s390x-unknown-linux-gnu         | FileCheck %s --check-prefixes=ALL %}
; RUN: %if ve-registered-target          %{ llc %s -o - -mtriple=ve-unknown-unknown              | FileCheck %s --check-prefixes=ALL %}
; RUN: %if webassembly-registered-target %{ llc %s -o - -mtriple=wasm32-unknown-unknown          | FileCheck %s --check-prefixes=ALL %}
; RUN: %if x86-registered-target         %{ llc %s -o - -mtriple=i686-unknown-linux-gnu          | FileCheck %s --check-prefixes=ALL %}
; RUN: %if x86-registered-target         %{ llc %s -o - -mtriple=x86_64-pc-windows-msvc          | FileCheck %s --check-prefixes=ALL %}
; RUN: %if x86-registered-target         %{ llc %s -o - -mtriple=x86_64-unknown-linux-gnu        | FileCheck %s --check-prefixes=ALL %}
; RUN: %if xcore-registered-target       %{ llc %s -o - -mtriple=xcore-unknown-unknown           | FileCheck %s --check-prefixes=ALL %}
; RUN: %if xtensa-registered-target      %{ llc %s -o - -mtriple=xtensa-none-elf                 | FileCheck %s --check-prefixes=NOCRASH %}

; Note that arm64ec labels are quoted, hence the `{{"?}}:`.

; Codegen tests don't work the same for graphics targets. Add a dummy directive
; for filecheck, just make sure we don't crash. Xtensa should pass but the
; symbol name list is emitted before the label, making things difficult to
; split up correctly.
; NOCRASH: {{.*}}

; fneg, fabs and copysign all need to not quieten signalling NaNs, so should not call any conversion functions which do.
; These tests won't catch cases where the everything is done using native instructions instead of builtins.
; See https://github.com/llvm/llvm-project/issues/104915

define void @test_fneg(ptr %p1, ptr %p2) #0 {
; ALL-LABEL: test_fneg{{"?}}:
; ALL-NOT: __extend
; ALL-NOT: __trunc
; ALL-NOT: __gnu
; ALL-NOT: __aeabi
  %v = load half, ptr %p1
  %res = fneg half %v
  store half %res, ptr %p2
  ret void
}

define void @test_fabs(ptr %p1, ptr %p2) {
; ALL-LABEL: test_fabs{{"?}}:
; ALL-NOT: __extend
; ALL-NOT: __trunc
; ALL-NOT: __gnu
; ALL-NOT: __aeabi
  %a = load half, ptr %p1
  %r = call half @llvm.fabs.f16(half %a)
  store half %r, ptr %p2
  ret void
}

define void @test_copysign(ptr %p1, ptr %p2, ptr %p3) {
; ALL-LABEL: test_copysign{{"?}}:
; ALL-NOT: __extend
; ALL-NOT: __trunc
; ALL-NOT: __gnu
; ALL-NOT: __aeabi
  %a = load half, ptr %p1
  %b = load half, ptr %p2
  %r = call half @llvm.copysign.f16(half %a, half %b)
  store half %r, ptr %p3
  ret void
}

; If promoting, fma must promote at least to f64 to avoid double rounding issues.
; This checks for calls to f32 fmaf and truncating f32 to f16.
; See https://github.com/llvm/llvm-project/issues/98389

define void @test_fma(ptr %p1, ptr %p2, ptr %p3, ptr %p4) {
; ALL-LABEL: test_fma{{"?}}:
; Allow fmaf16
; ALL-NOT: fmaf{{\b}}
; ALL-NOT: __truncsfhf2
; ALL-NOT: __gnu_f2h_ieee
; ALL-NOT: __aeabi_f2h
  %a = load half, ptr %p1
  %b = load half, ptr %p2
  %c = load half, ptr %p3
  %r = call half @llvm.fma.f16(half %a, half %b, half %c)
  store half %r, ptr %p4
  ret void
}
