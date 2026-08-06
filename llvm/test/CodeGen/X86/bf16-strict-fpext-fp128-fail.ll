; RUN: not --crash llc -o /dev/null %s -mtriple=x86_64-unknown-unknown -mattr=avx512bf16,avx512vl 2>&1 | FileCheck %s

; FIXME: this should not crash -- see bfloat-strict-fpext.ll for the cases that
; already work.

; CHECK: LLVM ERROR: unsupported library call operation
define fp128 @strict_fpext_bf16_to_fp128(bfloat %a) nounwind strictfp {
  %r = call fp128 @llvm.experimental.constrained.fpext.f128.bf16(bfloat %a, metadata !"fpexcept.strict")
  ret fp128 %r
}

declare fp128 @llvm.experimental.constrained.fpext.f128.bf16(bfloat, metadata)
