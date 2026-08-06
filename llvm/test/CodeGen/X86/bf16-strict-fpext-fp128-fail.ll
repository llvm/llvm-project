; RUN: llc -o /dev/null %s -mtriple=x86_64-unknown-unknown -mattr=avx512bf16,avx512vl 2>&1 | FileCheck %s
XFAIL: *

; CHECK-NOT: {{Unable to legalize as libcall|unsupported library call operation}}
define fp128 @strict_fpext_bf16_to_fp128(bfloat %a) nounwind strictfp {
  %r = call fp128 @llvm.experimental.constrained.fpext.f128.bf16(bfloat %a, metadata !"fpexcept.strict")
  ret fp128 %r
}

declare fp128 @llvm.experimental.constrained.fpext.f128.bf16(bfloat, metadata)
