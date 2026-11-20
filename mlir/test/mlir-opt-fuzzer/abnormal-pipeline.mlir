// RUN: not mlir-opt-fuzzer %s -ignore_remaining_args=1 2>&1 | FileCheck %s --check-prefix=PIPELINE-MISSING
// RUN: not mlir-opt-fuzzer %s -ignore_remaining_args=1 --pipeline=nope 2>&1 | FileCheck %s --check-prefix=PIPELINE-UNKNOWN

// PIPELINE-MISSING: no --pipeline given
// PIPELINE-UNKNOWN: unknown pipeline slug 'nope'
