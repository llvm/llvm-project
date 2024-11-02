// Check that the wide integer `arith.cmpi` emulation produces the same result as wide
// `arith.cmpi`. Emulate i16 ops with i8 ops.
// Ops in functions prefixed with `emulate` will be emulated using i8 types.

// RUN: mlir-opt %s --convert-scf-to-cf --convert-cf-to-llvm --convert-vector-to-llvm \
// RUN:             --convert-func-to-llvm --convert-arith-to-llvm | \
// RUN:   mlir-cpu-runner -e entry -entry-point-result=void \
// RUN:                   --shared-libs=%mlir_lib_dir/libmlir_c_runner_utils%shlibext | \
// RUN:   FileCheck %s --match-full-lines

// RUN: mlir-opt %s --test-arith-emulate-wide-int="widest-int-supported=8" \
// RUN:             --convert-scf-to-cf --convert-cf-to-llvm --convert-vector-to-llvm \
// RUN:             --convert-func-to-llvm --convert-arith-to-llvm | \
// RUN:   mlir-cpu-runner -e entry -entry-point-result=void \
// RUN:                   --shared-libs=%mlir_lib_dir/libmlir_c_runner_utils%shlibext | \
// RUN:   FileCheck %s --match-full-lines

func.func @emulate_cmpi_eq(%lhs : i16, %rhs : i16) -> (i1) {
  %res = arith.cmpi eq, %lhs, %rhs : i16
  return %res : i1
}

func.func @check_cmpi_eq(%lhs : i16, %rhs : i16) -> () {
  %res = func.call @emulate_cmpi_eq(%lhs, %rhs) : (i16, i16) -> (i1)
  vector.print %res : i1
  return
}

func.func @emulate_cmpi_ne(%lhs : i16, %rhs : i16) -> (i1) {
  %res = arith.cmpi ne, %lhs, %rhs : i16
  return %res : i1
}

func.func @check_cmpi_ne(%lhs : i16, %rhs : i16) -> () {
  %res = func.call @emulate_cmpi_ne(%lhs, %rhs) : (i16, i16) -> (i1)
  vector.print %res : i1
  return
}

func.func @emulate_cmpi_sge(%lhs : i16, %rhs : i16) -> (i1) {
  %res = arith.cmpi sge, %lhs, %rhs : i16
  return %res : i1
}

func.func @check_cmpi_sge(%lhs : i16, %rhs : i16) -> () {
  %res = func.call @emulate_cmpi_sge(%lhs, %rhs) : (i16, i16) -> (i1)
  vector.print %res : i1
  return
}

func.func @emulate_cmpi_sgt(%lhs : i16, %rhs : i16) -> (i1) {
  %res = arith.cmpi sgt, %lhs, %rhs : i16
  return %res : i1
}

func.func @check_cmpi_sgt(%lhs : i16, %rhs : i16) -> () {
  %res = func.call @emulate_cmpi_sgt(%lhs, %rhs) : (i16, i16) -> (i1)
  vector.print %res : i1
  return
}

func.func @emulate_cmpi_sle(%lhs : i16, %rhs : i16) -> (i1) {
  %res = arith.cmpi sle, %lhs, %rhs : i16
  return %res : i1
}

func.func @check_cmpi_sle(%lhs : i16, %rhs : i16) -> () {
  %res = func.call @emulate_cmpi_sle(%lhs, %rhs) : (i16, i16) -> (i1)
  vector.print %res : i1
  return
}

func.func @emulate_cmpi_slt(%lhs : i16, %rhs : i16) -> (i1) {
  %res = arith.cmpi slt, %lhs, %rhs : i16
  return %res : i1
}

func.func @check_cmpi_slt(%lhs : i16, %rhs : i16) -> () {
  %res = func.call @emulate_cmpi_slt(%lhs, %rhs) : (i16, i16) -> (i1)
  vector.print %res : i1
  return
}

func.func @emulate_cmpi_uge(%lhs : i16, %rhs : i16) -> (i1) {
  %res = arith.cmpi uge, %lhs, %rhs : i16
  return %res : i1
}

func.func @check_cmpi_uge(%lhs : i16, %rhs : i16) -> () {
  %res = func.call @emulate_cmpi_uge(%lhs, %rhs) : (i16, i16) -> (i1)
  vector.print %res : i1
  return
}

func.func @emulate_cmpi_ugt(%lhs : i16, %rhs : i16) -> (i1) {
  %res = arith.cmpi ugt, %lhs, %rhs : i16
  return %res : i1
}

func.func @check_cmpi_ugt(%lhs : i16, %rhs : i16) -> () {
  %res = func.call @emulate_cmpi_ugt(%lhs, %rhs) : (i16, i16) -> (i1)
  vector.print %res : i1
  return
}

func.func @emulate_cmpi_ule(%lhs : i16, %rhs : i16) -> (i1) {
  %res = arith.cmpi ule, %lhs, %rhs : i16
  return %res : i1
}

func.func @check_cmpi_ule(%lhs : i16, %rhs : i16) -> () {
  %res = func.call @emulate_cmpi_ule(%lhs, %rhs) : (i16, i16) -> (i1)
  vector.print %res : i1
  return
}

func.func @emulate_cmpi_ult(%lhs : i16, %rhs : i16) -> (i1) {
  %res = arith.cmpi ult, %lhs, %rhs : i16
  return %res : i1
}

func.func @check_cmpi_ult(%lhs : i16, %rhs : i16) -> () {
  %res = func.call @emulate_cmpi_ult(%lhs, %rhs) : (i16, i16) -> (i1)
  vector.print %res : i1
  return
}

func.func @entry() {
  %cst0 = arith.constant 0 : i16
  %cst1 = arith.constant 1 : i16
  %cst7 = arith.constant 7 : i16
  %cst_n1 = arith.constant -1 : i16
  %cst1337 = arith.constant 1337 : i16
  %cst4096 = arith.constant 4096 : i16
  %cst_i16_min = arith.constant -32768 : i16

  // CHECK:      1
  // CHECK-NEXT: 0
  // CHECK-NEXT: 0
  // CHECK-NEXT: 1
  // CHECK-NEXT: 0
  // CHECK-NEXT: 0
  // CHECK-NEXT: 1
  // CHECK-NEXT: 1
  // CHECK-NEXT: 0
  func.call @check_cmpi_eq(%cst0, %cst0) : (i16, i16) -> ()
  func.call @check_cmpi_eq(%cst0, %cst1) : (i16, i16) -> ()
  func.call @check_cmpi_eq(%cst1, %cst0) : (i16, i16) -> ()
  func.call @check_cmpi_eq(%cst1, %cst1) : (i16, i16) -> ()
  func.call @check_cmpi_eq(%cst_n1, %cst1) : (i16, i16) -> ()
  func.call @check_cmpi_eq(%cst_n1, %cst1337) : (i16, i16) -> ()
  func.call @check_cmpi_eq(%cst1337, %cst1337) : (i16, i16) -> ()
  func.call @check_cmpi_eq(%cst4096, %cst4096) : (i16, i16) -> ()
  func.call @check_cmpi_eq(%cst1337, %cst_i16_min) : (i16, i16) -> ()

  // CHECK-NEXT: 0
  // CHECK-NEXT: 1
  // CHECK-NEXT: 1
  // CHECK-NEXT: 0
  // CHECK-NEXT: 1
  // CHECK-NEXT: 1
  // CHECK-NEXT: 0
  // CHECK-NEXT: 0
  // CHECK-NEXT: 1
  func.call @check_cmpi_ne(%cst0, %cst0) : (i16, i16) -> ()
  func.call @check_cmpi_ne(%cst0, %cst1) : (i16, i16) -> ()
  func.call @check_cmpi_ne(%cst1, %cst0) : (i16, i16) -> ()
  func.call @check_cmpi_ne(%cst1, %cst1) : (i16, i16) -> ()
  func.call @check_cmpi_ne(%cst_n1, %cst1) : (i16, i16) -> ()
  func.call @check_cmpi_ne(%cst_n1, %cst1337) : (i16, i16) -> ()
  func.call @check_cmpi_ne(%cst1337, %cst1337) : (i16, i16) -> ()
  func.call @check_cmpi_ne(%cst4096, %cst4096) : (i16, i16) -> ()
  func.call @check_cmpi_ne(%cst1337, %cst_i16_min) : (i16, i16) -> ()

  // CHECK-NEXT: 1
  // CHECK-NEXT: 0
  // CHECK-NEXT: 1
  // CHECK-NEXT: 1
  // CHECK-NEXT: 0
  // CHECK-NEXT: 1
  // CHECK-NEXT: 0
  // CHECK-NEXT: 1
  // CHECK-NEXT: 1
  // CHECK-NEXT: 1
  func.call @check_cmpi_sge(%cst0, %cst0) : (i16, i16) -> ()
  func.call @check_cmpi_sge(%cst0, %cst1) : (i16, i16) -> ()
  func.call @check_cmpi_sge(%cst1, %cst0) : (i16, i16) -> ()
  func.call @check_cmpi_sge(%cst1, %cst1) : (i16, i16) -> ()
  func.call @check_cmpi_sge(%cst_n1, %cst1) : (i16, i16) -> ()
  func.call @check_cmpi_sge(%cst1, %cst_n1) : (i16, i16) -> ()
  func.call @check_cmpi_sge(%cst_n1, %cst1337) : (i16, i16) -> ()
  func.call @check_cmpi_sge(%cst1337, %cst1337) : (i16, i16) -> ()
  func.call @check_cmpi_sge(%cst4096, %cst4096) : (i16, i16) -> ()
  func.call @check_cmpi_sge(%cst1337, %cst_i16_min) : (i16, i16) -> ()

  // CHECK-NEXT: 0
  // CHECK-NEXT: 0
  // CHECK-NEXT: 1
  // CHECK-NEXT: 0
  // CHECK-NEXT: 0
  // CHECK-NEXT: 1
  // CHECK-NEXT: 0
  // CHECK-NEXT: 0
  // CHECK-NEXT: 0
  // CHECK-NEXT: 1
  func.call @check_cmpi_sgt(%cst0, %cst0) : (i16, i16) -> ()
  func.call @check_cmpi_sgt(%cst0, %cst1) : (i16, i16) -> ()
  func.call @check_cmpi_sgt(%cst1, %cst0) : (i16, i16) -> ()
  func.call @check_cmpi_sgt(%cst1, %cst1) : (i16, i16) -> ()
  func.call @check_cmpi_sgt(%cst_n1, %cst1) : (i16, i16) -> ()
  func.call @check_cmpi_sgt(%cst1, %cst_n1) : (i16, i16) -> ()
  func.call @check_cmpi_sgt(%cst_n1, %cst1337) : (i16, i16) -> ()
  func.call @check_cmpi_sgt(%cst1337, %cst1337) : (i16, i16) -> ()
  func.call @check_cmpi_sgt(%cst4096, %cst4096) : (i16, i16) -> ()
  func.call @check_cmpi_sgt(%cst1337, %cst_i16_min) : (i16, i16) -> ()

  // CHECK-NEXT: 1
  // CHECK-NEXT: 1
  // CHECK-NEXT: 0
  // CHECK-NEXT: 1
  // CHECK-NEXT: 1
  // CHECK-NEXT: 0
  // CHECK-NEXT: 1
  // CHECK-NEXT: 1
  // CHECK-NEXT: 1
  // CHECK-NEXT: 0
  func.call @check_cmpi_sle(%cst0, %cst0) : (i16, i16) -> ()
  func.call @check_cmpi_sle(%cst0, %cst1) : (i16, i16) -> ()
  func.call @check_cmpi_sle(%cst1, %cst0) : (i16, i16) -> ()
  func.call @check_cmpi_sle(%cst1, %cst1) : (i16, i16) -> ()
  func.call @check_cmpi_sle(%cst_n1, %cst1) : (i16, i16) -> ()
  func.call @check_cmpi_sle(%cst1, %cst_n1) : (i16, i16) -> ()
  func.call @check_cmpi_sle(%cst_n1, %cst1337) : (i16, i16) -> ()
  func.call @check_cmpi_sle(%cst1337, %cst1337) : (i16, i16) -> ()
  func.call @check_cmpi_sle(%cst4096, %cst4096) : (i16, i16) -> ()
  func.call @check_cmpi_sle(%cst1337, %cst_i16_min) : (i16, i16) -> ()

  // CHECK-NEXT: 0
  // CHECK-NEXT: 1
  // CHECK-NEXT: 0
  // CHECK-NEXT: 0
  // CHECK-NEXT: 1
  // CHECK-NEXT: 0
  // CHECK-NEXT: 1
  // CHECK-NEXT: 0
  // CHECK-NEXT: 0
  // CHECK-NEXT: 0
  func.call @check_cmpi_slt(%cst0, %cst0) : (i16, i16) -> ()
  func.call @check_cmpi_slt(%cst0, %cst1) : (i16, i16) -> ()
  func.call @check_cmpi_slt(%cst1, %cst0) : (i16, i16) -> ()
  func.call @check_cmpi_slt(%cst1, %cst1) : (i16, i16) -> ()
  func.call @check_cmpi_slt(%cst_n1, %cst1) : (i16, i16) -> ()
  func.call @check_cmpi_slt(%cst1, %cst_n1) : (i16, i16) -> ()
  func.call @check_cmpi_slt(%cst_n1, %cst1337) : (i16, i16) -> ()
  func.call @check_cmpi_slt(%cst1337, %cst1337) : (i16, i16) -> ()
  func.call @check_cmpi_slt(%cst4096, %cst4096) : (i16, i16) -> ()
  func.call @check_cmpi_slt(%cst1337, %cst_i16_min) : (i16, i16) -> ()

  // CHECK-NEXT: 1
  // CHECK-NEXT: 0
  // CHECK-NEXT: 1
  // CHECK-NEXT: 1
  // CHECK-NEXT: 1
  // CHECK-NEXT: 0
  // CHECK-NEXT: 1
  // CHECK-NEXT: 1
  // CHECK-NEXT: 1
  // CHECK-NEXT: 0
  func.call @check_cmpi_uge(%cst0, %cst0) : (i16, i16) -> ()
  func.call @check_cmpi_uge(%cst0, %cst1) : (i16, i16) -> ()
  func.call @check_cmpi_uge(%cst1, %cst0) : (i16, i16) -> ()
  func.call @check_cmpi_uge(%cst1, %cst1) : (i16, i16) -> ()
  func.call @check_cmpi_uge(%cst_n1, %cst1) : (i16, i16) -> ()
  func.call @check_cmpi_uge(%cst1, %cst_n1) : (i16, i16) -> ()
  func.call @check_cmpi_uge(%cst_n1, %cst1337) : (i16, i16) -> ()
  func.call @check_cmpi_uge(%cst1337, %cst1337) : (i16, i16) -> ()
  func.call @check_cmpi_uge(%cst4096, %cst4096) : (i16, i16) -> ()
  func.call @check_cmpi_uge(%cst1337, %cst_i16_min) : (i16, i16) -> ()

  // CHECK-NEXT: 0
  // CHECK-NEXT: 0
  // CHECK-NEXT: 1
  // CHECK-NEXT: 0
  // CHECK-NEXT: 1
  // CHECK-NEXT: 0
  // CHECK-NEXT: 1
  // CHECK-NEXT: 0
  // CHECK-NEXT: 0
  // CHECK-NEXT: 0
  func.call @check_cmpi_ugt(%cst0, %cst0) : (i16, i16) -> ()
  func.call @check_cmpi_ugt(%cst0, %cst1) : (i16, i16) -> ()
  func.call @check_cmpi_ugt(%cst1, %cst0) : (i16, i16) -> ()
  func.call @check_cmpi_ugt(%cst1, %cst1) : (i16, i16) -> ()
  func.call @check_cmpi_ugt(%cst_n1, %cst1) : (i16, i16) -> ()
  func.call @check_cmpi_ugt(%cst1, %cst_n1) : (i16, i16) -> ()
  func.call @check_cmpi_ugt(%cst_n1, %cst1337) : (i16, i16) -> ()
  func.call @check_cmpi_ugt(%cst1337, %cst1337) : (i16, i16) -> ()
  func.call @check_cmpi_ugt(%cst4096, %cst4096) : (i16, i16) -> ()
  func.call @check_cmpi_ugt(%cst1337, %cst_i16_min) : (i16, i16) -> ()

  // CHECK-NEXT: 1
  // CHECK-NEXT: 1
  // CHECK-NEXT: 0
  // CHECK-NEXT: 1
  // CHECK-NEXT: 0
  // CHECK-NEXT: 1
  // CHECK-NEXT: 0
  // CHECK-NEXT: 1
  // CHECK-NEXT: 1
  // CHECK-NEXT: 1
  func.call @check_cmpi_ule(%cst0, %cst0) : (i16, i16) -> ()
  func.call @check_cmpi_ule(%cst0, %cst1) : (i16, i16) -> ()
  func.call @check_cmpi_ule(%cst1, %cst0) : (i16, i16) -> ()
  func.call @check_cmpi_ule(%cst1, %cst1) : (i16, i16) -> ()
  func.call @check_cmpi_ule(%cst_n1, %cst1) : (i16, i16) -> ()
  func.call @check_cmpi_ule(%cst1, %cst_n1) : (i16, i16) -> ()
  func.call @check_cmpi_ule(%cst_n1, %cst1337) : (i16, i16) -> ()
  func.call @check_cmpi_ule(%cst1337, %cst1337) : (i16, i16) -> ()
  func.call @check_cmpi_ule(%cst4096, %cst4096) : (i16, i16) -> ()
  func.call @check_cmpi_ule(%cst1337, %cst_i16_min) : (i16, i16) -> ()

  // CHECK-NEXT: 0
  // CHECK-NEXT: 1
  // CHECK-NEXT: 0
  // CHECK-NEXT: 0
  // CHECK-NEXT: 0
  // CHECK-NEXT: 1
  // CHECK-NEXT: 0
  // CHECK-NEXT: 0
  // CHECK-NEXT: 0
  // CHECK-NEXT: 1
  func.call @check_cmpi_ult(%cst0, %cst0) : (i16, i16) -> ()
  func.call @check_cmpi_ult(%cst0, %cst1) : (i16, i16) -> ()
  func.call @check_cmpi_ult(%cst1, %cst0) : (i16, i16) -> ()
  func.call @check_cmpi_ult(%cst1, %cst1) : (i16, i16) -> ()
  func.call @check_cmpi_ult(%cst_n1, %cst1) : (i16, i16) -> ()
  func.call @check_cmpi_ult(%cst1, %cst_n1) : (i16, i16) -> ()
  func.call @check_cmpi_ult(%cst_n1, %cst1337) : (i16, i16) -> ()
  func.call @check_cmpi_ult(%cst1337, %cst1337) : (i16, i16) -> ()
  func.call @check_cmpi_ult(%cst4096, %cst4096) : (i16, i16) -> ()
  func.call @check_cmpi_ult(%cst1337, %cst_i16_min) : (i16, i16) -> ()

  return
}
