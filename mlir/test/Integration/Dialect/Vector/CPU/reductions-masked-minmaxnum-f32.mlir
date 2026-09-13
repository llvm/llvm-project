// RUN: mlir-opt %s -test-lower-to-llvm  | \
// RUN: mlir-runner -e entry -entry-point-result=void  \
// RUN:   -shared-libs=%mlir_c_runner_utils | \
// RUN: FileCheck %s

func.func @maximumnumf_finite() {
  // The neutral (qNaN) is ignored by maximumnum. The masked-off lane is the
  // largest.
  // maximumnum(<qNaN>, -5, -7, -5) = -5
  %mask = arith.constant dense<[true, true, true, false]> : vector<4xi1>
  %v = arith.constant dense<[-5.0, -7.0, -5.0, -1.0]> : vector<4xf32>
  %0 = vector.mask %mask {
    vector.reduction <maximumnumf>, %v : vector<4xf32> into f32
  } : vector<4xi1> -> f32
  vector.print str "maximumnumf_finite\n"
  vector.print %0 : f32
  return
}
// CHECK-LABEL: maximumnumf_finite
// CHECK-NEXT: -5

func.func @minimumnumf_finite() {
  // minimumnum(<qNaN>, 5, 7, 5) = 5
  %mask = arith.constant dense<[true, true, true, false]> : vector<4xi1>
  %v = arith.constant dense<[5.0, 7.0, 5.0, 1.0]> : vector<4xf32>
  %0 = vector.mask %mask {
    vector.reduction <minimumnumf>, %v : vector<4xf32> into f32
  } : vector<4xi1> -> f32
  vector.print str "minimumnumf_finite\n"
  vector.print %0 : f32
  return
}
// CHECK-LABEL: minimumnumf_finite
// CHECK-NEXT: 5

func.func @maximumnumf_nan_active_lane() {
  // A NaN in an active lane is ignored, unlike maximumf.
  // maximumnum(<qNaN>, qNaN, -7, -5) = -5
  %mask = arith.constant dense<[true, true, true, false]> : vector<4xi1>
  %v = arith.constant dense<[0x7FC00000, -7.0, -5.0, -1.0]> : vector<4xf32>
  %0 = vector.mask %mask {
    vector.reduction <maximumnumf>, %v : vector<4xf32> into f32
  } : vector<4xi1> -> f32
  vector.print str "maximumnumf_nan_active_lane\n"
  vector.print %0 : f32
  return
}
// CHECK-LABEL: maximumnumf_nan_active_lane
// CHECK-NEXT: -5

func.func @maximumnumf_all_active_lanes_nan() {
  // The result is NaN only when every active lane is NaN.
  // maximumnum(<qNaN>, qNaN, qNaN) = qNaN
  %mask = arith.constant dense<[true, true, false, false]> : vector<4xi1>
  %v = arith.constant dense<[0x7FC00000, 0x7FC00000, -5.0, -1.0]> : vector<4xf32>
  %0 = vector.mask %mask {
    vector.reduction <maximumnumf>, %v : vector<4xf32> into f32
  } : vector<4xi1> -> f32
  vector.print str "maximumnumf_all_active_lanes_nan\n"
  vector.print %0 : f32
  return
}
// CHECK-LABEL: maximumnumf_all_active_lanes_nan
// CHECK-NEXT: nan

func.func @maximumnumf_no_active_lane() {
  // With no active lane the neutral wins.
  // maximumnum(<qNaN>) = qNaN
  %mask = arith.constant dense<false> : vector<4xi1>
  %v = arith.constant dense<[-5.0, -7.0, -5.0, -1.0]> : vector<4xf32>
  %0 = vector.mask %mask {
    vector.reduction <maximumnumf>, %v : vector<4xf32> into f32
  } : vector<4xi1> -> f32
  vector.print str "maximumnumf_no_active_lane\n"
  vector.print %0 : f32
  return
}
// CHECK-LABEL: maximumnumf_no_active_lane
// CHECK-NEXT: nan

func.func @minimumnumf_no_active_lane() {
  // minimumnum(<qNaN>) = qNaN
  %mask = arith.constant dense<false> : vector<4xi1>
  %v = arith.constant dense<[5.0, 7.0, 5.0, 1.0]> : vector<4xf32>
  %0 = vector.mask %mask {
    vector.reduction <minimumnumf>, %v : vector<4xf32> into f32
  } : vector<4xi1> -> f32
  vector.print str "minimumnumf_no_active_lane\n"
  vector.print %0 : f32
  return
}
// CHECK-LABEL: minimumnumf_no_active_lane
// CHECK-NEXT: nan

func.func @maximumnumf_nnan_no_active_lane() {
  // Under `nnan` the neutral is -inf instead of qNaN.
  // maximumnum(<-inf>) = -inf
  %mask = arith.constant dense<false> : vector<4xi1>
  %v = arith.constant dense<[-5.0, -7.0, -5.0, -1.0]> : vector<4xf32>
  %0 = vector.mask %mask {
    vector.reduction <maximumnumf>, %v fastmath<nnan> : vector<4xf32> into f32
  } : vector<4xi1> -> f32
  vector.print str "maximumnumf_nnan_no_active_lane\n"
  vector.print %0 : f32
  return
}
// CHECK-LABEL: maximumnumf_nnan_no_active_lane
// CHECK-NEXT: -inf

func.func @minimumnumf_nnan_no_active_lane() {
  // minimumnum(<+inf>) = +inf
  %mask = arith.constant dense<false> : vector<4xi1>
  %v = arith.constant dense<[5.0, 7.0, 5.0, 1.0]> : vector<4xf32>
  %0 = vector.mask %mask {
    vector.reduction <minimumnumf>, %v fastmath<nnan> : vector<4xf32> into f32
  } : vector<4xi1> -> f32
  vector.print str "minimumnumf_nnan_no_active_lane\n"
  vector.print %0 : f32
  return
}
// CHECK-LABEL: minimumnumf_nnan_no_active_lane
// CHECK-NEXT: inf

func.func @maximumnumf_nnan_ninf_no_active_lane() {
  // Under `nnan, ninf` the neutral is -FLT_MAX.
  // maximumnum(<-FLT_MAX>) = -FLT_MAX
  %mask = arith.constant dense<false> : vector<4xi1>
  %v = arith.constant dense<[-5.0, -7.0, -5.0, -1.0]> : vector<4xf32>
  %0 = vector.mask %mask {
    vector.reduction <maximumnumf>, %v fastmath<nnan,ninf> : vector<4xf32> into f32
  } : vector<4xi1> -> f32
  vector.print str "maximumnumf_nnan_ninf_no_active_lane\n"
  vector.print %0 : f32
  return
}
// CHECK-LABEL: maximumnumf_nnan_ninf_no_active_lane
// CHECK-NEXT: -3.40282e+38

func.func @minimumnumf_nnan_ninf_no_active_lane() {
  // minimumnum(<+FLT_MAX>) = +FLT_MAX
  %mask = arith.constant dense<false> : vector<4xi1>
  %v = arith.constant dense<[5.0, 7.0, 5.0, 1.0]> : vector<4xf32>
  %0 = vector.mask %mask {
    vector.reduction <minimumnumf>, %v fastmath<nnan,ninf> : vector<4xf32> into f32
  } : vector<4xi1> -> f32
  vector.print str "minimumnumf_nnan_ninf_no_active_lane\n"
  vector.print %0 : f32
  return
}
// CHECK-LABEL: minimumnumf_nnan_ninf_no_active_lane
// CHECK-NEXT: 3.40282e+38

func.func @maximumnumf_acc() {
  // An accumulator takes the place of the neutral.
  // maximumnum(<-3>, -5, -7, -5) = -3
  %mask = arith.constant dense<[true, true, true, false]> : vector<4xi1>
  %v = arith.constant dense<[-5.0, -7.0, -5.0, -1.0]> : vector<4xf32>
  %acc = arith.constant -3.0 : f32
  %0 = vector.mask %mask {
    vector.reduction <maximumnumf>, %v, %acc : vector<4xf32> into f32
  } : vector<4xi1> -> f32
  vector.print str "maximumnumf_acc\n"
  vector.print %0 : f32
  return
}
// CHECK-LABEL: maximumnumf_acc
// CHECK-NEXT: -3

func.func @entry() {
  call @maximumnumf_finite() : () -> ()
  call @minimumnumf_finite() : () -> ()
  call @maximumnumf_nan_active_lane() : () -> ()
  call @maximumnumf_all_active_lanes_nan() : () -> ()
  call @maximumnumf_no_active_lane() : () -> ()
  call @minimumnumf_no_active_lane() : () -> ()
  call @maximumnumf_nnan_no_active_lane() : () -> ()
  call @minimumnumf_nnan_no_active_lane() : () -> ()
  call @maximumnumf_nnan_ninf_no_active_lane() : () -> ()
  call @minimumnumf_nnan_ninf_no_active_lane() : () -> ()
  call @maximumnumf_acc() : () -> ()
  return
}
