// RUN: mlir-opt %s -test-lower-to-llvm  | \
// RUN: mlir-runner -e entry -entry-point-result=void  \
// RUN:   -shared-libs=%mlir_c_runner_utils | \
// RUN: FileCheck %s

func.func @maximumf_finite() {
  // The neutral (-inf) must order below every active lane. The masked-off
  // lane is the largest.
  // max(<-inf>, -5, -7, -5) = -5
  %mask = arith.constant dense<[true, true, true, false]> : vector<4xi1>
  %v = arith.constant dense<[-5.0, -7.0, -5.0, -1.0]> : vector<4xf32>
  %0 = vector.mask %mask {
    vector.reduction <maximumf>, %v : vector<4xf32> into f32
  } : vector<4xi1> -> f32
  vector.print str "maximumf_finite\n"
  vector.print %0 : f32
  return
}
// CHECK-LABEL: maximumf_finite
// CHECK-NEXT: -5

func.func @minimumf_finite() {
  // min(<+inf>, 5, 7, 5) = 5
  %mask = arith.constant dense<[true, true, true, false]> : vector<4xi1>
  %v = arith.constant dense<[5.0, 7.0, 5.0, 1.0]> : vector<4xf32>
  %0 = vector.mask %mask {
    vector.reduction <minimumf>, %v : vector<4xf32> into f32
  } : vector<4xi1> -> f32
  vector.print str "minimumf_finite\n"
  vector.print %0 : f32
  return
}
// CHECK-LABEL: minimumf_finite
// CHECK-NEXT: 5

func.func @maximumf_no_active_lane() {
  // With no active lane the neutral wins.
  // max(<-inf>) = -inf
  %mask = arith.constant dense<false> : vector<4xi1>
  %v = arith.constant dense<[-5.0, -7.0, -5.0, -1.0]> : vector<4xf32>
  %0 = vector.mask %mask {
    vector.reduction <maximumf>, %v : vector<4xf32> into f32
  } : vector<4xi1> -> f32
  vector.print str "maximumf_no_active_lane\n"
  vector.print %0 : f32
  return
}
// CHECK-LABEL: maximumf_no_active_lane
// CHECK-NEXT: -inf

func.func @minimumf_no_active_lane() {
  // min(<+inf>) = +inf
  %mask = arith.constant dense<false> : vector<4xi1>
  %v = arith.constant dense<[5.0, 7.0, 5.0, 1.0]> : vector<4xf32>
  %0 = vector.mask %mask {
    vector.reduction <minimumf>, %v : vector<4xf32> into f32
  } : vector<4xi1> -> f32
  vector.print str "minimumf_no_active_lane\n"
  vector.print %0 : f32
  return
}
// CHECK-LABEL: minimumf_no_active_lane
// CHECK-NEXT: inf

func.func @maximumf_ninf_no_active_lane() {
  // Under `ninf` the neutral is -FLT_MAX instead of -inf.
  // max(<-FLT_MAX>) = -FLT_MAX
  %mask = arith.constant dense<false> : vector<4xi1>
  %v = arith.constant dense<[-5.0, -7.0, -5.0, -1.0]> : vector<4xf32>
  %0 = vector.mask %mask {
    vector.reduction <maximumf>, %v fastmath<ninf> : vector<4xf32> into f32
  } : vector<4xi1> -> f32
  vector.print str "maximumf_ninf_no_active_lane\n"
  vector.print %0 : f32
  return
}
// CHECK-LABEL: maximumf_ninf_no_active_lane
// CHECK-NEXT: -3.40282e+38

func.func @minimumf_ninf_no_active_lane() {
  // min(<+FLT_MAX>) = +FLT_MAX
  %mask = arith.constant dense<false> : vector<4xi1>
  %v = arith.constant dense<[5.0, 7.0, 5.0, 1.0]> : vector<4xf32>
  %0 = vector.mask %mask {
    vector.reduction <minimumf>, %v fastmath<ninf> : vector<4xf32> into f32
  } : vector<4xi1> -> f32
  vector.print str "minimumf_ninf_no_active_lane\n"
  vector.print %0 : f32
  return
}
// CHECK-LABEL: minimumf_ninf_no_active_lane
// CHECK-NEXT: 3.40282e+38

func.func @maximumf_acc() {
  // An accumulator takes the place of the neutral.
  // max(<-3>, -5, -7, -5) = -3
  %mask = arith.constant dense<[true, true, true, false]> : vector<4xi1>
  %v = arith.constant dense<[-5.0, -7.0, -5.0, -1.0]> : vector<4xf32>
  %acc = arith.constant -3.0 : f32
  %0 = vector.mask %mask {
    vector.reduction <maximumf>, %v, %acc : vector<4xf32> into f32
  } : vector<4xi1> -> f32
  vector.print str "maximumf_acc\n"
  vector.print %0 : f32
  return
}
// CHECK-LABEL: maximumf_acc
// CHECK-NEXT: -3

func.func @entry() {
  call @maximumf_finite() : () -> ()
  call @minimumf_finite() : () -> ()
  call @maximumf_no_active_lane() : () -> ()
  call @minimumf_no_active_lane() : () -> ()
  call @maximumf_ninf_no_active_lane() : () -> ()
  call @minimumf_ninf_no_active_lane() : () -> ()
  call @maximumf_acc() : () -> ()
  return
}
