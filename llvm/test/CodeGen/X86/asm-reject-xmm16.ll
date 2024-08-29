; RUN: not llc -o /dev/null %s -mtriple=x86_64-unknown-unknown 2>&1 | FileCheck %s -check-prefix X64
; RUN: not llc -o /dev/null %s -mtriple=i386-unknown-unknown -mattr=avx512vl 2>&1 | FileCheck %s -check-prefix X86

; X64: error: couldn't allocate output register for constraint '{xmm16}': register is unavailable without AVX512F
; X86: error: couldn't allocate output register for constraint '{xmm16}': register is unavailable in 32-bit mode
define i64 @blup() {
  %v = tail call i64 asm "", "={xmm16},0"(i64 0)
  ret i64 %v
}
