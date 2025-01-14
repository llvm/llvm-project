// RUN: mlir-opt %s -test-lower-to-llvm  | \
// RUN: mlir-cpu-runner -e entry -entry-point-result=void  \
// RUN:   -shared-libs=%mlir_c_runner_utils | \
// RUN: FileCheck %s

//
// Test various signless, signed, unsigned integer types.
//
func.func @entry() {
  %0 = arith.constant dense<[true, false, -1, 0, 1]> : vector<5xi1>
  vector.print %0 : vector<5xi1>
  // CHECK: ( 1, 0, 1, 0, 1 )

  %1 = arith.constant dense<[true, false, -1, 0]> : vector<4xi1>
  %cast_1 = vector.bitcast %1 : vector<4xi1> to vector<4xsi1>
  vector.print %cast_1 : vector<4xsi1>
  // CHECK: ( 1, 0, 1, 0 )

  %2 = arith.constant dense<[true, false, 0, 1]> : vector<4xi1>
  %cast_2 = vector.bitcast %2 : vector<4xi1> to vector<4xui1>
  vector.print %cast_2 : vector<4xui1>
  // CHECK: ( 1, 0, 0, 1 )

  %3 = arith.constant dense<[-128, -127, -1, 0, 1, 127, 128, 254, 255]> : vector<9xi8>
  vector.print %3 : vector<9xi8>
  // CHECK: ( -128, -127, -1, 0, 1, 127, -128, -2, -1 )

  %4 = arith.constant dense<[-128, -127, -1, 0, 1, 127]> : vector<6xi8>
  %cast_4 = vector.bitcast %4 : vector<6xi8> to vector<6xsi8>
  vector.print %cast_4 : vector<6xsi8>
  // CHECK: ( -128, -127, -1, 0, 1, 127 )

  %5 = arith.constant dense<[0, 1, 127, 128, 254, 255]> : vector<6xi8>
  %cast_5 = vector.bitcast %5 : vector<6xi8> to vector<6xui8>
  vector.print %cast_5 : vector<6xui8>
  // CHECK: ( 0, 1, 127, 128, 254, 255 )

  %6 = arith.constant dense<[-32768, -32767, -1, 0, 1, 32767, 32768, 65534, 65535]> : vector<9xi16>
  vector.print %6 : vector<9xi16>
  // CHECK: ( -32768, -32767, -1, 0, 1, 32767, -32768, -2, -1 )

  %7 = arith.constant dense<[-32768, -32767, -1, 0, 1, 32767]> : vector<6xi16>
  %cast_7 = vector.bitcast %7 : vector<6xi16> to vector<6xsi16>
  vector.print %cast_7 : vector<6xsi16>
  // CHECK: ( -32768, -32767, -1, 0, 1, 32767 )

  %8 = arith.constant dense<[0, 1, 32767, 32768, 65534, 65535]> : vector<6xi16>
  %cast_8 = vector.bitcast %8 : vector<6xi16> to vector<6xui16>
  vector.print %cast_8 : vector<6xui16>
  // CHECK: ( 0, 1, 32767, 32768, 65534, 65535 )

  %9 = arith.constant dense<[-2147483648, -2147483647, -1, 0, 1,
                           2147483647, 2147483648, 4294967294, 4294967295]> : vector<9xi32>
  vector.print %9 : vector<9xi32>
  // CHECK: ( -2147483648, -2147483647, -1, 0, 1, 2147483647, -2147483648, -2, -1 )

  %10 = arith.constant dense<[-2147483648, -2147483647, -1, 0, 1, 2147483647]> : vector<6xi32>
  %cast_10 = vector.bitcast %10 : vector<6xi32> to vector<6xsi32>
  vector.print %cast_10 : vector<6xsi32>
  // CHECK: ( -2147483648, -2147483647, -1, 0, 1, 2147483647 )

  %11 = arith.constant dense<[0, 1, 2147483647, 2147483648, 4294967294, 4294967295]> : vector<6xi32>
  %cast_11 = vector.bitcast %11 : vector<6xi32> to vector<6xui32>
  vector.print %cast_11 : vector<6xui32>
  // CHECK: ( 0, 1, 2147483647, 2147483648, 4294967294, 4294967295 )

  %12 = arith.constant dense<[-9223372036854775808, -9223372036854775807, -1, 0, 1,
                            9223372036854775807, 9223372036854775808,
                            18446744073709551614, 18446744073709551615]> : vector<9xi64>
  vector.print %12 : vector<9xi64>
  // CHECK: ( -9223372036854775808, -9223372036854775807, -1, 0, 1, 9223372036854775807, -9223372036854775808, -2, -1 )

  %13 = arith.constant dense<[-9223372036854775808, -9223372036854775807, -1, 0, 1,
                            9223372036854775807]> : vector<6xi64>
  %cast_13 = vector.bitcast %13 : vector<6xi64> to vector<6xsi64>
  vector.print %cast_13 : vector<6xsi64>
  // CHECK: ( -9223372036854775808, -9223372036854775807, -1, 0, 1, 9223372036854775807 )

  %14 = arith.constant dense<[0, 1, 9223372036854775807, 9223372036854775808,
                           18446744073709551614, 18446744073709551615]> : vector<6xi64>
  %cast_14 = vector.bitcast %14 : vector<6xi64> to vector<6xui64>
  vector.print %cast_14 : vector<6xui64>
  // CHECK: ( 0, 1, 9223372036854775807, 9223372036854775808, 18446744073709551614, 18446744073709551615 )

  return
}
