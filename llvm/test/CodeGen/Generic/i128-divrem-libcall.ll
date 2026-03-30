; 64-bit targets: fused __divmodti4 / __udivmodti4
; RUN: %if x86-registered-target         %{ llc < %s -mtriple=x86_64-linux-gnu              | FileCheck %s --check-prefixes=CHECK,FUSED,SYSV-X64 %}
; RUN: %if x86-registered-target         %{ llc < %s -mtriple=x86_64-linux-gnux32           | FileCheck %s --check-prefixes=CHECK,FUSED %}
; RUN: %if x86-registered-target         %{ llc < %s -mtriple=x86_64-pc-windows-msvc        | FileCheck %s --check-prefixes=CHECK,WIN64 %}
; RUN: %if x86-registered-target         %{ llc < %s -mtriple=x86_64-w64-mingw32            | FileCheck %s --check-prefixes=CHECK,FUSED %}
; RUN: %if aarch64-registered-target     %{ llc < %s -mtriple=aarch64-linux-gnu             | FileCheck %s --check-prefixes=CHECK,FUSED,SYSV-A64 %}
; RUN: %if riscv-registered-target       %{ llc < %s -mtriple=riscv64-linux-gnu             | FileCheck %s --check-prefixes=CHECK,FUSED %}
; RUN: %if riscv-registered-target       %{ llc < %s -mtriple=riscv64-linux-gnu -mattr=+m   | FileCheck %s --check-prefixes=CHECK,FUSED %}
; RUN: %if powerpc-registered-target     %{ llc < %s -mtriple=powerpc64-linux-gnu           | FileCheck %s --check-prefixes=CHECK,FUSED %}
; RUN: %if sparc-registered-target       %{ llc < %s -mtriple=sparcv9-linux-gnu             | FileCheck %s --check-prefixes=CHECK,FUSED %}
; RUN: %if loongarch-registered-target   %{ llc < %s -mtriple=loongarch64-linux-gnu         | FileCheck %s --check-prefixes=CHECK,FUSED %}
; RUN: %if webassembly-registered-target %{ llc < %s -mtriple=wasm32-unknown-unknown        | FileCheck %s --check-prefixes=CHECK,FUSED %}
; RUN: %if webassembly-registered-target %{ llc < %s -mtriple=wasm64-unknown-unknown        | FileCheck %s --check-prefixes=CHECK,FUSED %}

; 32-bit x86 Linux/MinGW: fused __divmodti4 / __udivmodti4 via sret calling convention
; (i128 returned via hidden pointer as first arg; all args on stack)
; RUN: %if x86-registered-target      %{ llc < %s -mtriple=i386-linux-gnu                | FileCheck %s --check-prefixes=CHECK,FUSED,SYSV-X86 %}
; RUN: %if x86-registered-target      %{ llc < %s -mtriple=i686-linux-gnu                | FileCheck %s --check-prefixes=CHECK,FUSED,SYSV-X86 %}
; RUN: %if riscv-registered-target    %{ llc < %s -mtriple=riscv32-linux-gnu             | FileCheck %s --check-prefixes=CHECK,INLINE %}
; RUN: %if riscv-registered-target    %{ llc < %s -mtriple=riscv32-linux-gnu -mattr=+m   | FileCheck %s --check-prefixes=CHECK,INLINE %}
; RUN: %if arm-registered-target      %{ llc < %s -mtriple=armv6-linux-gnueabihf         | FileCheck %s --check-prefixes=CHECK,INLINE %}
; RUN: %if arm-registered-target      %{ llc < %s -mtriple=armv7-linux-gnueabi           | FileCheck %s --check-prefixes=CHECK,INLINE %}
; RUN: %if arm-registered-target      %{ llc < %s -mtriple=armv7-none-eabi               | FileCheck %s --check-prefixes=CHECK,INLINE %}

; Win32: fused ___divmodti4 (extra underscore from Windows cdecl decoration, same sret ABI)
; RUN: %if x86-registered-target      %{ llc < %s -mtriple=i686-pc-windows-msvc          | FileCheck %s --check-prefixes=CHECK,WIN32 %}

; ILP32 targets that fall back to separate __divti3 + __modti3 calls
; RUN: %if aarch64-registered-target  %{ llc < %s -mtriple=aarch64_32-apple-watchos      | FileCheck %s --check-prefixes=CHECK,DIVMOD %}

; 64-bit Mac OS: fused ___divmodti4 (extra underscore, same ABI as Linux AArch64)
; RUN: %if x86-registered-target         %{ llc < %s -mtriple=x86_64-apple-macosx           | FileCheck %s --check-prefixes=CHECK,FUSED-DARWIN %}
; RUN: %if aarch64-registered-target     %{ llc < %s -mtriple=arm64-apple-macosx            | FileCheck %s --check-prefixes=CHECK,DARWIN-A64 %}

; Verify that sdiv+srem / udiv+urem on i128 fuse into a single __divmodti4 /
; __udivmodti4 call on targets where the libcall is available, and do not on
; targets where it is not (bare-metal 32-bit without a runtime library).
;
; Detailed ABI checks for the calling conventions:
;   WIN64     (x86_64 Windows): all args spilled to stack and passed as pointers
;             in %rcx/%rdx/%r8, quotient returned in %xmm0.
;   DARWIN-A64 (AArch64 macOS): identical to SYSV-A64 but symbol has an extra
;             leading underscore (___divmodti4).
;   SYSV-X64  (x86_64 Linux/BSD): i128 args in register pairs, rem pointer via
;             %rsp in %r8, quotient returned in %rax:%rdx.
;   SYSV-A64  (AArch64 Linux): i128 args in x0:x1/x2:x3, rem pointer via sp in
;             x4, quotient returned in x0:x1.
;   SYSV-X86  (x86 Linux): i128 returned via sret hidden pointer at (%esp);
;             all args on stack; symbol is __divmodti4.
;   Win32 (i686-windows-msvc): same sret stack convention as SYSV-X86;
;             symbol has an extra leading underscore (___divmodti4).
;   32-bit targets that lack the fused call may lower to:
;     - separate __divti3 + __modti3 / __udivti3 + __umodti3 calls, or
;     - fully inline expansion (e.g. bare metal)

define void @sdivrem_i128(ptr %q_out, ptr %r_out, i128 %n, i128 %d) {
; CHECK-LABEL: sdivrem_i128:
; SYSV-X64:        movq    %rsp, %r8
; SYSV-A64:        mov     x4, sp
; SYSV-X86:        movl    %{{.*}}, (%esp)
; WIN32:           movl    %{{.*}}, (%esp)
; FUSED:           __divmodti4
; FUSED-DARWIN:    ___divmodti4
; SYSV-X64:        movq    (%rsp),
; SYSV-X64:        movq    %rax,
; SYSV-X64:        movq    %rdx,
; SYSV-A64:        ldp     {{.*}}, [sp]
; SYSV-A64:        stp     x0, x1,
; DARWIN-A64:      mov     x4, sp
; DARWIN-A64:      bl      ___divmodti4
; DARWIN-A64:      ldp     {{.*}}, [sp]
; DARWIN-A64:      stp     x0, x1,
; WIN64:           leaq    {{[0-9]+}}(%rsp), %rcx
; WIN64:           leaq    {{[0-9]+}}(%rsp), %rdx
; WIN64:           leaq    {{[0-9]+}}(%rsp), %r8
; WIN64:           callq   __divmodti4
; WIN64:           movaps  {{[0-9]+}}(%rsp), %xmm1
; WIN64:           movaps  %xmm0,
; WIN32:           calll   ___divmodti4
; SYSV-X86:        movl    {{[0-9]+}}(%esp),
; WIN32:           movl    {{[0-9]+}}(%esp),
; DIVMOD:          __divti3
; DIVMOD:          __modti3
; INLINE-NOT:      __divmodti4
; INLINE-NOT:      __divti3
; INLINE-NOT:      __modti3
  %q = sdiv i128 %n, %d
  %r = srem i128 %n, %d
  store i128 %q, ptr %q_out
  store i128 %r, ptr %r_out
  ret void
}

define void @udivrem_i128(ptr %q_out, ptr %r_out, i128 %n, i128 %d) {
; CHECK-LABEL: udivrem_i128:
; SYSV-X64:        movq    %rsp, %r8
; SYSV-A64:        mov     x4, sp
; SYSV-X86:        movl    %{{.*}}, (%esp)
; WIN32:           movl    %{{.*}}, (%esp)
; FUSED:           __udivmodti4
; FUSED-DARWIN:    ___udivmodti4
; SYSV-X64:        movq    (%rsp),
; SYSV-X64:        movq    %rax,
; SYSV-X64:        movq    %rdx,
; SYSV-A64:        ldp     {{.*}}, [sp]
; SYSV-A64:        stp     x0, x1,
; DARWIN-A64:      mov     x4, sp
; DARWIN-A64:      bl      ___udivmodti4
; DARWIN-A64:      ldp     {{.*}}, [sp]
; DARWIN-A64:      stp     x0, x1,
; WIN64:           leaq    {{[0-9]+}}(%rsp), %rcx
; WIN64:           leaq    {{[0-9]+}}(%rsp), %rdx
; WIN64:           leaq    {{[0-9]+}}(%rsp), %r8
; WIN64:           callq   __udivmodti4
; WIN64:           movaps  {{[0-9]+}}(%rsp), %xmm1
; WIN64:           movaps  %xmm0,
; WIN32:           calll   ___udivmodti4
; SYSV-X86:        movl    {{[0-9]+}}(%esp),
; WIN32:           movl    {{[0-9]+}}(%esp),
; DIVMOD:          __udivti3
; DIVMOD:          __umodti3
; INLINE-NOT:      __udivmodti4
; INLINE-NOT:      __udivti3
; INLINE-NOT:      __umodti3
  %q = udiv i128 %n, %d
  %r = urem i128 %n, %d
  store i128 %q, ptr %q_out
  store i128 %r, ptr %r_out
  ret void
}
