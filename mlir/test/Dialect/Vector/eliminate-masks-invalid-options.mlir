// RUN: not mlir-opt %s -eliminate-vector-masks="vscale-min=2" 2>&1 | FileCheck %s
// RUN: not mlir-opt %s -eliminate-vector-masks="vscale-min=8 vscale-max=4" 2>&1 | FileCheck %s

/// A half-specified or inverted vscale range is rejected rather than silently
/// ignored, which would leave scalable masks unproven for no visible reason.

// CHECK: error: expected 'vscale-min' and 'vscale-max' to both be set, with 'vscale-min' <= 'vscale-max'
func.func @vscale_range(%n: index) -> vector<[4]xi1> {
  %mask = vector.create_mask %n : vector<[4]xi1>
  return %mask : vector<[4]xi1>
}
