; Simple cross-platform smoke checks for basic f16 operations.
;
; There shouldn't be any architectures that crash when trying to use `half`;
; check that here. Additionally do a small handful of smoke tests that work
; well cross-platform.

; RUN: %if aarch64-registered-target     %{ llc %s -o - -mtriple=aarch64-apple-darwin            | FileCheck %s --check-prefixes=ALL,CHECK %}
; RUN: %if aarch64-registered-target     %{ llc %s -o - -mtriple=aarch64-pc-windows-msvc         | FileCheck %s --check-prefixes=ALL,CHECK %}
; RUN: %if aarch64-registered-target     %{ llc %s -o - -mtriple=aarch64-unknown-linux-gnu       | FileCheck %s --check-prefixes=ALL,CHECK %}
; FIXME(#94434) unsupported on arm64ec
; RUN: %if aarch64-registered-target     %{ ! llc %s -o - -mtriple=arm64ec-pc-windows-msvc -filetype=null %}
; RUN: %if amdgpu-registered-target      %{ llc %s -o - -mtriple=amdgcn-amd-amdhsa               | FileCheck %s --check-prefixes=ALL,CHECK %}
; RUN: %if arc-registered-target         %{ llc %s -o - -mtriple=arc-elf                         | FileCheck %s --check-prefixes=ALL,CHECK %}
; RUN: %if arm-registered-target         %{ llc %s -o - -mtriple=arm-unknown-linux-gnueabi       | FileCheck %s --check-prefixes=ALL,CHECK %}
; RUN: %if arm-registered-target         %{ llc %s -o - -mtriple=thumbv7em-none-eabi             | FileCheck %s --check-prefixes=ALL,CHECK %}
; RUN: %if avr-registered-target         %{ llc %s -o - -mtriple=avr-none                        | FileCheck %s --check-prefixes=ALL,CHECK %}
; RUN: %if bpf-registered-target         %{ llc %s -o - -mtriple=bpfel                           | FileCheck %s --check-prefixes=ALL,CHECK %}
; RUN: %if csky-registered-target        %{ llc %s -o - -mtriple=csky-unknown-linux-gnuabiv2     | FileCheck %s --check-prefixes=ALL,CHECK %}
; RUN: %if csky-registered-target        %{ llc %s -o - -mtriple=csky-unknown-linux-gnuabiv2 -mcpu=ck860fv -mattr=+hard-float | FileCheck %s --check-prefixes=ALL,BAD %}
; RUN: %if directx-registered-target     %{ llc %s -o - -mtriple=dxil-pc-shadermodel6.3-library  | FileCheck %s --check-prefixes=NOCRASH %}
; RUN: %if hexagon-registered-target     %{ llc %s -o - -mtriple=hexagon-unknown-linux-musl      | FileCheck %s --check-prefixes=ALL,CHECK %}
; RUN: %if lanai-registered-target       %{ llc %s -o - -mtriple=lanai-unknown-unknown           | FileCheck %s --check-prefixes=ALL,CHECK %}
; RUN: %if loongarch-registered-target   %{ llc %s -o - -mtriple=loongarch32-unknown-linux-gnu   | FileCheck %s --check-prefixes=ALL,CHECK %}
; RUN: %if loongarch-registered-target   %{ llc %s -o - -mtriple=loongarch64-unknown-linux-gnu   | FileCheck %s --check-prefixes=ALL,CHECK %}
; RUN: %if loongarch-registered-target   %{ llc %s -o - -mtriple=loongarch64-unknown-linux-gnu -mattr=+f | FileCheck %s --check-prefixes=ALL,CHECK %}
; RUN: %if m68k-registered-target        %{ llc %s -o - -mtriple=m68k-unknown-linux-gnu          | FileCheck %s --check-prefixes=ALL,CHECK %}
; RUN: %if mips-registered-target        %{ llc %s -o - -mtriple=mips-unknown-linux-gnu          | FileCheck %s --check-prefixes=ALL,CHECK %}
; RUN: %if mips-registered-target        %{ llc %s -o - -mtriple=mips64-unknown-linux-gnuabi64   | FileCheck %s --check-prefixes=ALL,CHECK %}
; RUN: %if mips-registered-target        %{ llc %s -o - -mtriple=mips64el-unknown-linux-gnuabi64 | FileCheck %s --check-prefixes=ALL,CHECK %}
; RUN: %if mips-registered-target        %{ llc %s -o - -mtriple=mipsel-unknown-linux-gnu        | FileCheck %s --check-prefixes=ALL,CHECK %}
; RUN: %if msp430-registered-target      %{ llc %s -o - -mtriple=msp430-none-elf                 | FileCheck %s --check-prefixes=ALL,CHECK %}
; RUN: %if nvptx-registered-target       %{ llc %s -o - -mtriple=nvptx64-nvidia-cuda             | FileCheck %s --check-prefixes=NOCRASH   %}
; RUN: %if powerpc-registered-target     %{ llc %s -o - -mtriple=powerpc-unknown-linux-gnu       | FileCheck %s --check-prefixes=ALL,BAD   %}
; RUN: %if powerpc-registered-target     %{ llc %s -o - -mtriple=powerpc64-unknown-linux-gnu     | FileCheck %s --check-prefixes=ALL,BAD   %}
; RUN: %if powerpc-registered-target     %{ llc %s -o - -mtriple=powerpc64le-unknown-linux-gnu   | FileCheck %s --check-prefixes=ALL,BAD   %}
; RUN: %if riscv-registered-target       %{ llc %s -o - -mtriple=riscv32-unknown-linux-gnu       | FileCheck %s --check-prefixes=ALL,CHECK %}
; RUN: %if riscv-registered-target       %{ llc %s -o - -mtriple=riscv64-unknown-linux-gnu       | FileCheck %s --check-prefixes=ALL,CHECK %}
; RUN: %if sparc-registered-target       %{ llc %s -o - -mtriple=sparc-unknown-linux-gnu         | FileCheck %s --check-prefixes=ALL,BAD   %}
; RUN: %if sparc-registered-target       %{ llc %s -o - -mtriple=sparc64-unknown-linux-gnu       | FileCheck %s --check-prefixes=ALL,BAD   %}
; RUN: %if spirv-registered-target       %{ llc %s -o - -mtriple=spirv-unknown-unknown           | FileCheck %s --check-prefixes=NOCRASH   %}
; RUN: %if systemz-registered-target     %{ llc %s -o - -mtriple=s390x-unknown-linux-gnu         | FileCheck %s --check-prefixes=ALL,CHECK %}
; RUN: %if ve-registered-target          %{ llc %s -o - -mtriple=ve-unknown-unknown              | FileCheck %s --check-prefixes=ALL,BAD   %}
; RUN: %if webassembly-registered-target %{ llc %s -o - -mtriple=wasm32-unknown-unknown          | FileCheck %s --check-prefixes=ALL,BAD   %}
; RUN: %if x86-registered-target         %{ llc %s -o - -mtriple=i686-unknown-linux-gnu          | FileCheck %s --check-prefixes=ALL,CHECK %}
; RUN: %if x86-registered-target         %{ llc %s -o - -mtriple=x86_64-pc-windows-msvc          | FileCheck %s --check-prefixes=ALL,CHECK %}
; RUN: %if x86-registered-target         %{ llc %s -o - -mtriple=x86_64-unknown-linux-gnu        | FileCheck %s --check-prefixes=ALL,CHECK %}
; RUN: %if xcore-registered-target       %{ llc %s -o - -mtriple=xcore-unknown-unknown           | FileCheck %s --check-prefixes=ALL,CHECK %}
; RUN: %if xtensa-registered-target      %{ llc %s -o - -mtriple=xtensa-none-elf                 | FileCheck %s --check-prefixes=ALL,CHECK %}

; Codegen tests don't work the same for graphics targets. Add a dummy directive
; for filecheck, just make sure we don't crash.
; NOCRASH: {{.*}}

; All backends need to be able to bitcast without converting to another format,
; so we assert against __extendhfsf2, __truncsfhf2, __gnu_{h2f,f2h}_ieee. This
; doesn't catch issues on platforms with hardware f32<->f16, but those tend to
; work better anyway.
; Regression test for https://github.com/llvm/llvm-project/issues/97981.

define half @from_bits(i16 %bits) nounwind {
; ALL-LABEL: from_bits:
; CHECK-NOT: __extend
; CHECK-NOT: __trunc
; CHECK-NOT: __gnu
; BAD:       __extendhfsf2
    %f = bitcast i16 %bits to half
    ret half %f
}

define i16 @to_bits(half %f) nounwind {
; ALL-LABEL: to_bits:
; CHECK-NOT: __extend
; CHECK-NOT: __trunc
; CHECK-NOT: __gnu
; BAD:       __truncsfhf2
    %bits = bitcast half %f to i16
    ret i16 %bits
}

; Some platforms have had problems freezing. Regression test for
; https://github.com/llvm/llvm-project/issues/117337 and similar issues.

define half @check_freeze(half %f) nounwind {
; ALL-LABEL: check_freeze:
  %t0 = freeze half %f
  ret half %t0
}
