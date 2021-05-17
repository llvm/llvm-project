// RUN: mlir-opt %s -convert-scf-to-std -convert-vector-to-llvm -convert-std-to-llvm | \
// RUN: mlir-cpu-runner -e entry -entry-point-result=void  \
// RUN:   -shared-libs=%mlir_integration_test_dir/libmlir_c_runner_utils%shlibext | \
// RUN: FileCheck %s

//
// Test various signless, signed, unsigned integer types.
//
func @entry() {
  %0 = std.constant dense<[true, false, -1, 0, 1]> : vector<5xi1>
  vector.print %0 : vector<5xi1>
  // CHECK: ( 1, 0, 1, 0, 1 )

  %1 = std.constant dense<[true, false, -1, 0]> : vector<4xsi1>
  vector.print %1 : vector<4xsi1>
  // CHECK: ( 1, 0, 1, 0 )

  %2 = std.constant dense<[true, false, 0, 1]> : vector<4xui1>
  vector.print %2 : vector<4xui1>
  // CHECK: ( 1, 0, 0, 1 )

  %3 = std.constant dense<[-128, -127, -1, 0, 1, 127, 128, 254, 255]> : vector<9xi8>
  vector.print %3 : vector<9xi8>
  // CHECK: ( -128, -127, -1, 0, 1, 127, -128, -2, -1 )

  %4 = std.constant dense<[-128, -127, -1, 0, 1, 127]> : vector<6xsi8>
  vector.print %4 : vector<6xsi8>
  // CHECK: ( -128, -127, -1, 0, 1, 127 )

  %5 = std.constant dense<[0, 1, 127, 128, 254, 255]> : vector<6xui8>
  vector.print %5 : vector<6xui8>
  // CHECK: ( 0, 1, 127, 128, 254, 255 )

  %6 = std.constant dense<[-32768, -32767, -1, 0, 1, 32767, 32768, 65534, 65535]> : vector<9xi16>
  vector.print %6 : vector<9xi16>
  // CHECK: ( -32768, -32767, -1, 0, 1, 32767, -32768, -2, -1 )

  %7 = std.constant dense<[-32768, -32767, -1, 0, 1, 32767]> : vector<6xsi16>
  vector.print %7 : vector<6xsi16>
  // CHECK: ( -32768, -32767, -1, 0, 1, 32767 )

  %8 = std.constant dense<[0, 1, 32767, 32768, 65534, 65535]> : vector<6xui16>
  vector.print %8 : vector<6xui16>
  // CHECK: ( 0, 1, 32767, 32768, 65534, 65535 )

  %9 = std.constant dense<[-2147483648, -2147483647, -1, 0, 1,
                            2147483647, 2147483648, 4294967294, 4294967295]> : vector<9xi32>
  vector.print %9 : vector<9xi32>
  // CHECK: ( -2147483648, -2147483647, -1, 0, 1, 2147483647, -2147483648, -2, -1 )

  %10 = std.constant dense<[-2147483648, -2147483647, -1, 0, 1, 2147483647]> : vector<6xsi32>
  vector.print %10 : vector<6xsi32>
  // CHECK: ( -2147483648, -2147483647, -1, 0, 1, 2147483647 )

  %11 = std.constant dense<[0, 1, 2147483647, 2147483648, 4294967294, 4294967295]> : vector<6xui32>
  vector.print %11 : vector<6xui32>
  // CHECK: ( 0, 1, 2147483647, 2147483648, 4294967294, 4294967295 )

  %12 = std.constant dense<[-9223372036854775808, -9223372036854775807, -1, 0, 1,
                             9223372036854775807, 9223372036854775808,
                             18446744073709551614, 18446744073709551615]> : vector<9xi64>
  vector.print %12 : vector<9xi64>
  // CHECK: ( -9223372036854775808, -9223372036854775807, -1, 0, 1, 9223372036854775807, -9223372036854775808, -2, -1 )

  %13 = std.constant dense<[-9223372036854775808, -9223372036854775807, -1, 0, 1,
                             9223372036854775807]> : vector<6xsi64>
  vector.print %13 : vector<6xsi64>
  // CHECK: ( -9223372036854775808, -9223372036854775807, -1, 0, 1, 9223372036854775807 )

  %14 = std.constant dense<[0, 1, 9223372036854775807, 9223372036854775808,
                            18446744073709551614, 18446744073709551615]> : vector<6xui64>
  vector.print %14 : vector<6xui64>
  // CHECK: ( 0, 1, 9223372036854775807, 9223372036854775808, 18446744073709551614, 18446744073709551615 )

  return
}
