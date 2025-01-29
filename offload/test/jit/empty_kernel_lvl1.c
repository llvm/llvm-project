// clang-format off
// RUN: %libomptarget-compileopt-generic -fopenmp-target-jit \
// RUN:     -DTGT1_DIRECTIVE="target"
// RUN: env LIBOMPTARGET_JIT_PRE_OPT_IR_MODULE=%t.pre.ll     \
// RUN:     LIBOMPTARGET_JIT_SKIP_OPT=true                   \
// RUN:     %libomptarget-run-generic
// RUN: %fcheck-plain-generic --input-file %t.pre.ll %S/empty_kernel.inc --check-prefix=FIRST
// RUN: %libomptarget-compileopt-generic -fopenmp-target-jit \
// RUN:     -DTGT1_DIRECTIVE="target"                        \
// RUN:     -DTGT2_DIRECTIVE="target"
// RUN: env LIBOMPTARGET_JIT_PRE_OPT_IR_MODULE=%t.pre.ll     \
// RUN:     LIBOMPTARGET_JIT_SKIP_OPT=true                   \
// RUN:     %libomptarget-run-generic
// RUN: %fcheck-plain-generic --input-file %t.pre.ll %S/empty_kernel.inc --check-prefixes=FIRST
//
// RUN: %libomptarget-compileopt-generic -fopenmp-target-jit \
// RUN:     -DTGT1_DIRECTIVE="target teams"
// RUN: env LIBOMPTARGET_JIT_PRE_OPT_IR_MODULE=%t.pre.ll     \
// RUN:     LIBOMPTARGET_JIT_SKIP_OPT=true                   \
// RUN:     %libomptarget-run-generic
// RUN: %fcheck-plain-generic --input-file %t.pre.ll %S/empty_kernel.inc --check-prefix=FIRST
// RUN: %libomptarget-compileopt-generic -fopenmp-target-jit \
// RUN:     -DTGT1_DIRECTIVE="target teams"                  \
// RUN:     -DTGT2_DIRECTIVE="target teams"
// RUN: env LIBOMPTARGET_JIT_PRE_OPT_IR_MODULE=%t.pre.ll     \
// RUN:     LIBOMPTARGET_JIT_SKIP_OPT=true                   \
// RUN:     %libomptarget-run-generic
// RUN: %fcheck-plain-generic --input-file %t.pre.ll %S/empty_kernel.inc --check-prefixes=FIRST,SECOND
// clang-format on

// REQUIRES: gpu

#include "empty_kernel.inc"
