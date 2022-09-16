// Check that the wide integer multiplication emulation produces the same result as wide
// multiplication. Emulate i16 ops with i8 ops.

// RUN: mlir-opt %s --convert-scf-to-cf --convert-cf-to-llvm --convert-vector-to-llvm \
// RUN:             --convert-func-to-llvm --convert-arith-to-llvm | \
// RUN:   mlir-cpu-runner -e entry -entry-point-result=void \
// RUN:                   --shared-libs=%mlir_lib_dir/libmlir_c_runner_utils%shlibext | \
// RUN:   FileCheck %s --match-full-lines --check-prefix=WIDE

// RUN: mlir-opt %s --arith-emulate-wide-int="widest-int-supported=8" \
// RUN:             --convert-scf-to-cf --convert-cf-to-llvm --convert-vector-to-llvm \
// RUN:             --convert-func-to-llvm --convert-arith-to-llvm | \
// RUN:   mlir-cpu-runner -e entry -entry-point-result=void \
// RUN:                   --shared-libs=%mlir_lib_dir/libmlir_c_runner_utils%shlibext | \
// RUN:   FileCheck %s --match-full-lines --check-prefix=EMULATED

func.func @check_muli(%lhs : i16, %rhs : i16) -> () {
  %res = arith.muli %lhs, %rhs : i16
  vector.print %res : i16
  return
}

func.func @entry() {
  %cst0 = arith.constant 0 : i16
  %cst1 = arith.constant 1 : i16
  %cst_1 = arith.constant -1 : i16
  %cst_3 = arith.constant -3 : i16

  %cst13 = arith.constant 13 : i16
  %cst37 = arith.constant 37 : i16
  %cst42 = arith.constant 42 : i16

  %cst256 = arith.constant 256 : i16
  %cst_i16_max = arith.constant 32767 : i16
  %cst_i16_min = arith.constant -32768 : i16

  // WIDE: 0
  // EMULATED: ( 0, 0 )
  func.call @check_muli(%cst0, %cst0) : (i16, i16) -> ()
  // WIDE-NEXT: 0
  // EMULATED-NEXT: ( 0, 0 )
  func.call @check_muli(%cst0, %cst1) : (i16, i16) -> ()
  // WIDE-NEXT: 1
  // EMULATED-NEXT: ( 1, 0 )
  func.call @check_muli(%cst1, %cst1) : (i16, i16) -> ()
  // WIDE-NEXT: -1
  // EMULATED-NEXT: ( -1, -1 )
  func.call @check_muli(%cst1, %cst_1) : (i16, i16) -> ()
  // WIDE-NEXT: 1
  // EMULATED-NEXT: ( 1, 0 )
  func.call @check_muli(%cst_1, %cst_1) : (i16, i16) -> ()
  // WIDE-NEXT: -3
  // EMULATED-NEXT: ( -3, -1 )
  func.call @check_muli(%cst1, %cst_3) : (i16, i16) -> ()

  // WIDE-NEXT: 169
  // EMULATED-NEXT: ( -87, 0 )
  func.call @check_muli(%cst13, %cst13) : (i16, i16) -> ()
  // WIDE-NEXT: 481
  // EMULATED-NEXT: ( -31, 1 )
  func.call @check_muli(%cst13, %cst37) : (i16, i16) -> ()
  // WIDE-NEXT: 1554
  // EMULATED-NEXT: ( 18, 6 )
  func.call @check_muli(%cst37, %cst42) : (i16, i16) -> ()

  // WIDE-NEXT: -256
  // EMULATED-NEXT: ( 0, -1 )
  func.call @check_muli(%cst_1, %cst256) : (i16, i16) -> ()
  // WIDE-NEXT: 3328
  // EMULATED-NEXT: ( 0, 13 )
  func.call @check_muli(%cst256, %cst13) : (i16, i16) -> ()
  // WIDE-NEXT: 9472
  // EMULATED-NEXT: ( 0, 37 )
  func.call @check_muli(%cst256, %cst37) : (i16, i16) -> ()
  // WIDE-NEXT: -768
  // EMULATED-NEXT: ( 0, -3 )
  func.call @check_muli(%cst256, %cst_3) : (i16, i16) -> ()

  // WIDE-NEXT: 32755
  // EMULATED-NEXT: ( -13, 127 )
  func.call @check_muli(%cst13, %cst_i16_max) : (i16, i16) -> ()
  // WIDE-NEXT: -32768
  // EMULATED-NEXT: ( 0, -128 )
  func.call @check_muli(%cst_i16_min, %cst37) : (i16, i16) -> ()

  // WIDE-NEXT: 1
  // EMULATED-NEXT: ( 1, 0 )
  func.call @check_muli(%cst_i16_max, %cst_i16_max) : (i16, i16) -> ()
  // WIDE-NEXT: -32768
  // EMULATED-NEXT: ( 0, -128 )
  func.call @check_muli(%cst_i16_min, %cst13) : (i16, i16) -> ()
  // WIDE-NEXT: 0
  // EMULATED-NEXT: ( 0, 0 )
  func.call @check_muli(%cst_i16_min, %cst_i16_min) : (i16, i16) -> ()

  return
}
