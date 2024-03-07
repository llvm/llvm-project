// clang-format off
// RUN: %libomptarget-compileoptxx-generic -fopenmp-target-jit \
// RUN:     -DTGT1_DIRECTIVE="target"                          \
// RUN:     -DLOOP_DIRECTIVE="for"
// RUN: env LIBOMPTARGET_JIT_PRE_OPT_IR_MODULE=%t.pre.ll     \
// RUN:     LIBOMPTARGET_JIT_SKIP_OPT=true                   \
// RUN:     %libomptarget-run-generic
// RUN: %fcheck-plain-generic --input-file %t.pre.ll %S/empty_kernel.inc --check-prefix=FIRST
// RUN: %libomptarget-compileoptxx-generic -fopenmp-target-jit \
// RUN:     -DTGT1_DIRECTIVE="target"                          \
// RUN:     -DTGT2_DIRECTIVE="target"                          \
// RUN:     -DLOOP_DIRECTIVE="for"
// RUN: env LIBOMPTARGET_JIT_PRE_OPT_IR_MODULE=%t.pre.ll     \
// RUN:     LIBOMPTARGET_JIT_SKIP_OPT=true                   \
// RUN:     %libomptarget-run-generic
// RUN: %fcheck-plain-generic --input-file %t.pre.ll %S/empty_kernel.inc --check-prefixes=FIRST,SECOND
//
// RUN: %libomptarget-compileoptxx-generic -fopenmp-target-jit \
// RUN:     -DTGT1_DIRECTIVE="target"                          \
// RUN:     -DLOOP_DIRECTIVE="parallel for"
// RUN: env LIBOMPTARGET_JIT_PRE_OPT_IR_MODULE=%t.pre.ll     \
// RUN:     LIBOMPTARGET_JIT_SKIP_OPT=true                   \
// RUN:     %libomptarget-run-generic
// TODO:
// RUN: not %fcheck-plain-generic --input-file %t.pre.ll %S/empty_kernel.inc --check-prefix=FIRST
// RUN: %libomptarget-compileoptxx-generic -fopenmp-target-jit \
// RUN:     -DTGT1_DIRECTIVE="target"                          \
// RUN:     -DTGT2_DIRECTIVE="target teams"                    \
// RUN:     -DLOOP_DIRECTIVE="parallel for"
// RUN: env LIBOMPTARGET_JIT_PRE_OPT_IR_MODULE=%t.pre.ll     \
// RUN:     LIBOMPTARGET_JIT_SKIP_OPT=true                   \
// RUN:     %libomptarget-run-generic
// TODO:
// RUN: not %fcheck-plain-generic --input-file %t.pre.ll %S/empty_kernel.inc --check-prefixes=FIRST,SECOND
//
// RUN: %libomptarget-compileoptxx-generic -fopenmp-target-jit \
// RUN:     -DTGT1_DIRECTIVE="target teams"                    \
// RUN:     -DLOOP_DIRECTIVE="distribute"
// RUN: env LIBOMPTARGET_JIT_PRE_OPT_IR_MODULE=%t.pre.ll     \
// RUN:     LIBOMPTARGET_JIT_SKIP_OPT=true                   \
// RUN:     %libomptarget-run-generic
// TODO:
// RUN: not %fcheck-plain-generic --input-file %t.pre.ll %S/empty_kernel.inc --check-prefix=FIRST
// RUN: %libomptarget-compileoptxx-generic -fopenmp-target-jit \
// RUN:     -DTGT1_DIRECTIVE="target teams"                    \
// RUN:     -DTGT2_DIRECTIVE="target teams"                    \
// RUN:     -DLOOP_DIRECTIVE="distribute"
// RUN: env LIBOMPTARGET_JIT_PRE_OPT_IR_MODULE=%t.pre.ll     \
// RUN:     LIBOMPTARGET_JIT_SKIP_OPT=true                   \
// RUN:     %libomptarget-run-generic
// TODO:
// RUN: not %fcheck-plain-generic --input-file %t.pre.ll %S/empty_kernel.inc --check-prefixes=FIRST,SECOND
//
// RUN: %libomptarget-compileoptxx-generic -fopenmp-target-jit \
// RUN:     -DTGT1_DIRECTIVE="target teams"                    \
// RUN:     -DLOOP_DIRECTIVE="distribute parallel for"
// RUN: env LIBOMPTARGET_JIT_PRE_OPT_IR_MODULE=%t.pre.ll     \
// RUN:     LIBOMPTARGET_JIT_SKIP_OPT=true                   \
// RUN:     %libomptarget-run-generic
// TODO:
// RUN: not %fcheck-plain-generic --input-file %t.pre.ll %S/empty_kernel.inc --check-prefix=FIRST
// RUN: %libomptarget-compileoptxx-generic -fopenmp-target-jit \
// RUN:     -DTGT1_DIRECTIVE="target teams"                    \
// RUN:     -DTGT2_DIRECTIVE="target teams"                    \
// RUN:     -DLOOP_DIRECTIVE="distribute parallel for"
// RUN: env LIBOMPTARGET_JIT_PRE_OPT_IR_MODULE=%t.pre.ll     \
// RUN:     LIBOMPTARGET_JIT_SKIP_OPT=true                   \
// RUN:     %libomptarget-run-generic
// TODO:
// RUN: not %fcheck-plain-generic --input-file %t.pre.ll %S/empty_kernel.inc --check-prefixes=FIRST,SECOND
//
// RUN: %libomptarget-compileoptxx-generic -fopenmp-target-jit \
// RUN:     -DTGT1_DIRECTIVE="target teams"                    \
// RUN:     -DLOOP_DIRECTIVE="distribute parallel for simd"
// RUN: env LIBOMPTARGET_JIT_PRE_OPT_IR_MODULE=%t.pre.ll     \
// RUN:     LIBOMPTARGET_JIT_SKIP_OPT=true                   \
// RUN:     %libomptarget-run-generic
// TODO:
// RUN: not %fcheck-plain-generic --input-file %t.pre.ll %S/empty_kernel.inc --check-prefix=FIRST
// RUN: %libomptarget-compileoptxx-generic -fopenmp-target-jit \
// RUN:     -DTGT1_DIRECTIVE="target teams"                    \
// RUN:     -DTGT2_DIRECTIVE="target teams"                    \
// RUN:     -DLOOP_DIRECTIVE="distribute parallel for simd"
// RUN: env LIBOMPTARGET_JIT_PRE_OPT_IR_MODULE=%t.pre.ll     \
// RUN:     LIBOMPTARGET_JIT_SKIP_OPT=true                   \
// RUN:     %libomptarget-run-generic
// TODO:
// RUN: not %fcheck-plain-generic --input-file %t.pre.ll %S/empty_kernel.inc --check-prefixes=FIRST,SECOND
// clang-format on

// UNSUPPORTED: aarch64-unknown-linux-gnu
// UNSUPPORTED: aarch64-unknown-linux-gnu-LTO
// UNSUPPORTED: x86_64-pc-linux-gnu
// UNSUPPORTED: x86_64-pc-linux-gnu-LTO

#include "empty_kernel.inc"
