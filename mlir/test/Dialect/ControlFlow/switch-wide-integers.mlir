// RUN: mlir-opt %s | mlir-opt | FileCheck %s
// RUN: mlir-opt %s --mlir-print-op-generic | FileCheck %s --check-prefix=GENERIC
// RUN: mlir-opt %s | mlir-opt --mlir-print-op-generic | FileCheck %s --check-prefix=GENERIC
// RUN: mlir-opt %s | mlir-opt --canonicalize | FileCheck %s --check-prefix=FOLD

// Start with generic switches so the custom printer must preserve both values.
// After reparsing, canonicalization must select the first and second cases,
// respectively, rather than the default or the same case twice.
// CHECK-LABEL: func.func @switch_i128_large_cases
// CHECK:       18446744073709551616:
// CHECK-NEXT:  18446744073709551617:
// CHECK:       18446744073709551616:
// CHECK-NEXT:  18446744073709551617:
// GENERIC-LABEL: sym_name = "switch_i128_large_cases"
// GENERIC: case_values = dense<[18446744073709551616, 18446744073709551617]> : vector<2xi128>
// GENERIC: case_values = dense<[18446744073709551616, 18446744073709551617]> : vector<2xi128>
// FOLD-LABEL: func.func @switch_i128_large_cases
// FOLD-DAG:   %[[ONE:.*]] = arith.constant 1 : i32
// FOLD-DAG:   %[[TWO:.*]] = arith.constant 2 : i32
// FOLD-NEXT:  return %[[ONE]], %[[TWO]] : i32, i32
// FOLD-NEXT:  }
func.func @switch_i128_large_cases() -> (i32, i32) {
  %first = arith.constant 18446744073709551616 : i128
  %second = arith.constant 18446744073709551617 : i128
  %zero = arith.constant 0 : i32
  %one = arith.constant 1 : i32
  %two = arith.constant 2 : i32
  "cf.switch"(%first, %zero, %one, %two)[^bb1, ^bb1, ^bb1] <{case_operand_segments = array<i32: 1, 1>, case_values = dense<[18446744073709551616, 18446744073709551617]> : vector<2xi128>, operandSegmentSizes = array<i32: 1, 1, 2>}> : (i128, i32, i32, i32) -> ()
^bb1(%first_result: i32):
  "cf.switch"(%second, %zero, %one, %two)[^bb2, ^bb2, ^bb2] <{case_operand_segments = array<i32: 1, 1>, case_values = dense<[18446744073709551616, 18446744073709551617]> : vector<2xi128>, operandSegmentSizes = array<i32: 1, 1, 2>}> : (i128, i32, i32, i32) -> ()
^bb2(%second_result: i32):
  return %first_result, %second_result : i32, i32
}

// Negative literals must be sign-extended to the declared width, including
// literals that already require more than 64 bits to parse.
// CHECK-LABEL: func.func @switch_i128_negative
// CHECK:       -1:
// CHECK-NEXT:  -9223372036854775808:
// CHECK-NEXT:  -9223372036854775809:
// CHECK-NEXT:  -18446744073709551617:
// CHECK-NEXT:  -170141183460469231731687303715884105728:
// CHECK-NEXT:  170141183460469231731687303715884105727:
// GENERIC-LABEL: sym_name = "switch_i128_negative"
// GENERIC: case_values = dense<[-1, -9223372036854775808, -9223372036854775809, -18446744073709551617, -170141183460469231731687303715884105728, 170141183460469231731687303715884105727]> : vector<6xi128>
func.func @switch_i128_negative(%flag: i128) {
  cf.switch %flag : i128, [
    default: ^bb1,
    -1: ^bb1,
    -9223372036854775808: ^bb1,
    -9223372036854775809: ^bb1,
    -18446744073709551617: ^bb1,
    -170141183460469231731687303715884105728: ^bb1,
    170141183460469231731687303715884105727: ^bb1
  ]
^bb1:
  return
}

// CHECK-LABEL: func.func @switch_i1
// CHECK:       default:
// CHECK-NEXT:  0:
// CHECK-NEXT:  -1:
// GENERIC-LABEL: sym_name = "switch_i1"
// GENERIC: case_values = dense<[false, true]> : vector<2xi1>
func.func @switch_i1(%flag: i1) {
  cf.switch %flag : i1, [
    default: ^bb1,
    0: ^bb1,
    1: ^bb1
  ]
^bb1:
  return
}

// Preserve ordinary small values and truncation to the declared width.
// CHECK-LABEL: func.func @switch_i8
// CHECK:       42:
// CHECK-NEXT:  -42:
// CHECK-NEXT:  -128:
// CHECK-NEXT:  -1:
// CHECK-NEXT:  0:
// CHECK-NEXT:  127:
// GENERIC-LABEL: sym_name = "switch_i8"
// GENERIC: case_values = dense<[42, -42, -128, -1, 0, 127]> : vector<6xi8>
func.func @switch_i8(%flag: i8) {
  cf.switch %flag : i8, [
    default: ^bb1,
    42: ^bb1,
    -42: ^bb1,
    128: ^bb1,
    255: ^bb1,
    256: ^bb1,
    -129: ^bb1
  ]
^bb1:
  return
}

// CHECK-LABEL: func.func @switch_i64_boundaries
// CHECK:       9223372036854775807:
// CHECK-NEXT:  -9223372036854775808:
// CHECK-NEXT:  -1:
// GENERIC-LABEL: sym_name = "switch_i64_boundaries"
// GENERIC: case_values = dense<[9223372036854775807, -9223372036854775808, -1]> : vector<3xi64>
func.func @switch_i64_boundaries(%flag: i64) {
  cf.switch %flag : i64, [
    default: ^bb1,
    9223372036854775807: ^bb1,
    -9223372036854775808: ^bb1,
    18446744073709551615: ^bb1
  ]
^bb1:
  return
}

// The same positive literal must remain positive when the declared width has
// room for its sign bit, and become negative when it sets that sign bit.
// CHECK-LABEL: func.func @switch_i65_boundaries
// CHECK:       18446744073709551615:
// CHECK-NEXT:  -18446744073709551616:
// CHECK-NEXT:  -1:
// GENERIC-LABEL: sym_name = "switch_i65_boundaries"
// GENERIC: case_values = dense<[18446744073709551615, -18446744073709551616, -1]> : vector<3xi65>
func.func @switch_i65_boundaries(%flag: i65) {
  cf.switch %flag : i65, [
    default: ^bb1,
    18446744073709551615: ^bb1,
    18446744073709551616: ^bb1,
    -1: ^bb1
  ]
^bb1:
  return
}
