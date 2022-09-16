// Check that the wide integer constant emulation produces the same result as wide
// constants and that printing works. Emulate i16 ops with i8 ops.

// RUN: mlir-opt %s --arith-emulate-wide-int="widest-int-supported=8" \
// RUN:             --convert-scf-to-cf --convert-cf-to-llvm --convert-vector-to-llvm \
// RUN:             --convert-func-to-llvm --convert-arith-to-llvm | \
// RUN:   mlir-cpu-runner -e entry -entry-point-result=void \
// RUN:                   --shared-libs=%mlir_lib_dir/libmlir_c_runner_utils%shlibext | \
// RUN:   FileCheck %s --match-full-lines --check-prefix=EMULATED

func.func @entry() {
  %cst0 = arith.constant 0 : i16
  %cst1 = arith.constant 1 : i16
  %cst_1 = arith.constant -1 : i16
  %cst_3 = arith.constant -3 : i16

  %cst13 = arith.constant 13 : i16
  %cst256 = arith.constant 256 : i16

  %cst_i16_max = arith.constant 32767 : i16
  %cst_i16_min = arith.constant -32768 : i16

  // EMULATED: ( 0, 0 )
  vector.print %cst0 : i16
  // EMULATED-NEXT: ( 1, 0 )
  vector.print %cst1 : i16

  // EMULATED-NEXT: ( -1, -1 )
  vector.print %cst_1 : i16
  // EMULATED-NEXT: ( -3, -1 )
  vector.print %cst_3 : i16

  // EMULATED-NEXT: ( 13, 0 )
  vector.print %cst13 : i16
  // EMULATED-NEXT: ( 0, 1 )
  vector.print %cst256 : i16

  // EMULATED-NEXT: ( -1, 127 )
  vector.print %cst_i16_max : i16
  // EMULATED-NEXT: ( 0, -128 )
  vector.print %cst_i16_min : i16

  return
}
