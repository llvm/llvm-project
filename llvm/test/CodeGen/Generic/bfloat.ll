; Simple cross-platform smoke checks for basic bf16 operations.
;
; There shouldn't be any architectures that crash when trying to use `bfloat`;
; check that here. Additionally do a small handful of smoke tests that work
; well cross-platform.

; RUN: %if aarch64-registered-target     %{ llc %s -o - -mtriple=aarch64-apple-darwin            | FileCheck %s --check-prefixes=ALL,CHECK %}
; RUN: %if aarch64-registered-target     %{ llc %s -o - -mtriple=aarch64-pc-windows-msvc         | FileCheck %s --check-prefixes=ALL,CHECK %}
; RUN: %if aarch64-registered-target     %{ llc %s -o - -mtriple=aarch64-unknown-linux-gnu       | FileCheck %s --check-prefixes=ALL,CHECK %}
; FIXME: arm64ec crashes when passing/returning bfloat
; RUN: %if amdgpu-registered-target      %{ llc %s -o - -mtriple=amdgcn-amd-amdhsa               | FileCheck %s --check-prefixes=ALL,CHECK %}
; RUN: %if arc-registered-target         %{ llc %s -o - -mtriple=arc-elf                         | FileCheck %s --check-prefixes=ALL,CHECK %}
; RUN: %if arm-registered-target         %{ llc %s -o - -mtriple=arm-unknown-linux-gnueabi       | FileCheck %s --check-prefixes=ALL,CHECK %}
; RUN: %if arm-registered-target         %{ llc %s -o - -mtriple=thumbv7em-none-eabi             | FileCheck %s --check-prefixes=ALL,CHECK %}
; RUN: %if avr-registered-target         %{ llc %s -o - -mtriple=avr-none                        | FileCheck %s --check-prefixes=ALL,CHECK %}
; RUN: %if bpf-registered-target         %{ llc %s -o - -mtriple=bpfel                           | FileCheck %s --check-prefixes=ALL,CHECK %}
; RUN: %if csky-registered-target        %{ llc %s -o - -mtriple=csky-unknown-linux-gnuabiv2     | FileCheck %s --check-prefixes=ALL,CHECK %}
; FIXME: hard float csky crashes
; RUN: %if directx-registered-target     %{ llc %s -o - -mtriple=dxil-pc-shadermodel6.3-library  | FileCheck %s --check-prefixes=NOCRASH %}
; FIXME: hexagon crashes
; RUN: %if lanai-registered-target       %{ llc %s -o - -mtriple=lanai-unknown-unknown           | FileCheck %s --check-prefixes=ALL,CHECK %}
; RUN: %if loongarch-registered-target   %{ llc %s -o - -mtriple=loongarch32-unknown-linux-gnu   | FileCheck %s --check-prefixes=ALL,CHECK %}
; RUN: %if loongarch-registered-target   %{ llc %s -o - -mtriple=loongarch64-unknown-linux-gnu   | FileCheck %s --check-prefixes=ALL,CHECK %}
; RUN: %if loongarch-registered-target   %{ llc %s -o - -mtriple=loongarch64-unknown-linux-gnu -mattr=+f | FileCheck %s --check-prefixes=ALL,CHECK %}
; RUN: %if m68k-registered-target        %{ llc %s -o - -mtriple=m68k-unknown-linux-gnu          | FileCheck %s --check-prefixes=ALL,CHECK %}
; FIXME: mips crashes
; RUN: %if msp430-registered-target      %{ llc %s -o - -mtriple=msp430-none-elf                 | FileCheck %s --check-prefixes=ALL,CHECK %}
; RUN: %if nvptx-registered-target       %{ llc %s -o - -mtriple=nvptx64-nvidia-cuda             | FileCheck %s --check-prefixes=NOCRASH   %}
; FIXME: powerpc crashes
; RUN: %if riscv-registered-target       %{ llc %s -o - -mtriple=riscv32-unknown-linux-gnu       | FileCheck %s --check-prefixes=ALL,CHECK %}
; RUN: %if riscv-registered-target       %{ llc %s -o - -mtriple=riscv64-unknown-linux-gnu       | FileCheck %s --check-prefixes=ALL,CHECK %}
; FIXME: sparc crashes
; FIXME: spirv crashes
; FIXME: s390x crashes
; FIXME: ve crashes
; FIXME: wasm crashes
; RUN: %if x86-registered-target         %{ llc %s -o - -mtriple=i686-unknown-linux-gnu          | FileCheck %s --check-prefixes=ALL,BAD %}
; RUN: %if x86-registered-target         %{ llc %s -o - -mtriple=x86_64-pc-windows-msvc          | FileCheck %s --check-prefixes=ALL,CHECK %}
; RUN: %if x86-registered-target         %{ llc %s -o - -mtriple=x86_64-unknown-linux-gnu        | FileCheck %s --check-prefixes=ALL,CHECK %}
; RUN: %if xcore-registered-target       %{ llc %s -o - -mtriple=xcore-unknown-unknown           | FileCheck %s --check-prefixes=ALL,CHECK %}
; RUN: %if xtensa-registered-target      %{ llc %s -o - -mtriple=xtensa-none-elf                 | FileCheck %s --check-prefixes=ALL,CHECK %}

; Note that arm64ec labels are quoted, hence the `{{"?}}:`.

; Codegen tests don't work the same for graphics targets. Add a dummy directive
; for filecheck, just make sure we don't crash.
; NOCRASH: {{.*}}

; All backends need to be able to bitcast without converting to another format,
; so we assert against libcalls (specifically __truncsfbf2). This won't catch hardware conversions.

define bfloat @from_bits(i16 %bits) nounwind {
; ALL-LABEL: from_bits{{"?}}:
; ALL-NOT: __extend
; ALL-NOT: __trunc
; ALL-NOT: __gnu
    %f = bitcast i16 %bits to bfloat
    ret bfloat %f
}

define i16 @to_bits(bfloat %f) nounwind {
; ALL-LABEL: to_bits{{"?}}:
; CHECK-NOT: __extend
; CHECK-NOT: __trunc
; CHECK-NOT: __gnu
; BAD:       __truncsfbf2
    %bits = bitcast bfloat %f to i16
    ret i16 %bits
}

define bfloat @check_freeze(bfloat %f) nounwind {
; ALL-LABEL: check_freeze{{"?}}:
  %t0 = freeze bfloat %f
  ret bfloat %t0
}
