// RUN: not mlir-opt %s -eliminate-vector-masks="vscale-min=2" 2>&1 | FileCheck %s --check-prefixes=CHECK,HALF
// RUN: not mlir-opt %s -eliminate-vector-masks="vscale-min=0 vscale-max=4" 2>&1 | FileCheck %s --check-prefixes=CHECK,ZERO
// RUN: not mlir-opt %s -eliminate-vector-masks="vscale-min=8 vscale-max=4" 2>&1 | FileCheck %s --check-prefixes=CHECK,INVERTED

/// A half-specified or inverted vscale range is rejected rather than silently
/// ignored, which would leave scalable masks unproven for no visible reason.
/// The error is reported once, not once per function.

// HALF:       error: invalid vscale range 'vscale-min=2 vscale-max=0':
// ZERO:       error: invalid vscale range 'vscale-min=0 vscale-max=4':
// INVERTED:   error: invalid vscale range 'vscale-min=8 vscale-max=4':
// CHECK-SAME: expected both to be 0 (unknown), or both non-zero with 'vscale-min' <= 'vscale-max'
// CHECK-NOT:  error:
func.func @vscale_range(%n: index) -> vector<[4]xi1> {
  %mask = vector.create_mask %n : vector<[4]xi1>
  return %mask : vector<[4]xi1>
}

func.func @other() {
  return
}
