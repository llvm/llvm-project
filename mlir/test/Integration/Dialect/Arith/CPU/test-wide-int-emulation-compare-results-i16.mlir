// Check that the wide integer emulation produces the same result as wide
// calculations. Emulate i16 ops with i8 ops.

// RUN: mlir-opt %s --test-arith-emulate-wide-int="widest-int-supported=8" \
// RUN:             --convert-scf-to-cf --convert-cf-to-llvm --convert-vector-to-llvm \
// RUN:             --convert-func-to-llvm --convert-arith-to-llvm | \
// RUN:   mlir-cpu-runner -e entry -entry-point-result=void \
// RUN:      --shared-libs="%mlir_lib_dir/libmlir_c_runner_utils%shlibext,%mlir_lib_dir/libmlir_runner_utils%shlibext" | \
// RUN:   FileCheck %s

// CHECK-NOT: Mismatch

//===----------------------------------------------------------------------===//
// Common Utility Functions
//===----------------------------------------------------------------------===//

llvm.mlir.global internal constant @str_mismatch("Mismatch\0A")
func.func private @printCString(!llvm.ptr<i8>) -> ()
// Prints 'Mismatch' to stdout.
func.func @printMismatch() -> () {
  %0 = llvm.mlir.addressof @str_mismatch : !llvm.ptr<array<9 x i8>>
  %1 = llvm.mlir.constant(0 : index) : i64
  %2 = llvm.getelementptr %0[%1, %1]
    : (!llvm.ptr<array<9 x i8>>, i64, i64) -> !llvm.ptr<i8>
  func.call @printCString(%2) : (!llvm.ptr<i8>) -> ()
  return
}

// Prints both binary op operands and the first result. If the second result
// does not match, prints the second result and a 'Mismatch' message.
func.func @check_results(%lhs : i16, %rhs : i16, %res0 : i16, %res1 : i16) -> () {
  %vec_zero = arith.constant dense<0> : vector<2xi16>
  %ins0 = vector.insert %lhs, %vec_zero[0] : i16 into vector<2xi16>
  %operands = vector.insert %rhs, %ins0[1] : i16 into vector<2xi16>
  vector.print %operands : vector<2xi16>
  vector.print %res0 : i16
  %mismatch = arith.cmpi ne, %res0, %res1 : i16
  scf.if %mismatch -> () {
    vector.print %res1 : i16
    func.call @printMismatch() : () -> ()
  }
  return
}

func.func @xorshift(%i : i16) -> (i16) {
  %cst8 = arith.constant 8 : i16
  %shifted = arith.shrui %i, %cst8 : i16
  %res = arith.xori %i, %shifted : i16
  return %res : i16
}

// Returns a hash of the input number. This is used we want to sample a bunch
// of i16 inputs with close to uniform distribution but without fixed offsets
// between each sample.
func.func @xhash(%i : i16) -> (i16) {
  %pattern = arith.constant 21845 : i16 // Alternating ones and zeros.
  %prime = arith.constant 25867 : i16   // Large i16 prime.
  %xi = func.call @xorshift(%i) : (i16) -> (i16)
  %inner = arith.muli %xi, %pattern : i16
  %xinner = func.call @xorshift(%inner) : (i16) -> (i16)
  %res = arith.muli %xinner, %prime : i16
  return %res : i16
}

//===----------------------------------------------------------------------===//
// Test arith.addi
//===----------------------------------------------------------------------===//

// Ops in this function will be emulated using i8 ops.
func.func @emulate_addi(%lhs : i16, %rhs : i16) -> (i16) {
  %res = arith.addi %lhs, %rhs : i16
  return %res : i16
}

// Performs both wide and emulated `arith.muli`, and checks that the results
// match.
func.func @check_addi(%lhs : i16, %rhs : i16) -> () {
  %wide = arith.addi %lhs, %rhs : i16
  %emulated = func.call @emulate_addi(%lhs, %rhs) : (i16, i16) -> (i16)
  func.call @check_results(%lhs, %rhs, %wide, %emulated) : (i16, i16, i16, i16) -> ()
  return
}

// Checks that `arith.addi` is emulated properly by sampling the input space.
// In total, this test function checks 500 * 500 = 250k input pairs.
func.func @test_addi() -> () {
  %idx0 = arith.constant 0 : index
  %idx1 = arith.constant 1 : index
  %idx500 = arith.constant 500 : index

  %cst0 = arith.constant 0 : i16
  %cst1 = arith.constant 1 : i16

  scf.for %lhs_idx = %idx0 to %idx500 step %idx1 iter_args(%lhs = %cst0) -> (i16) {
    %arg_lhs = func.call @xhash(%lhs) : (i16) -> (i16)

    scf.for %rhs_idx = %idx0 to %idx500 step %idx1 iter_args(%rhs = %cst0) -> (i16) {
        %arg_rhs = func.call @xhash(%rhs) : (i16) -> (i16)
        func.call @check_addi(%arg_lhs, %arg_rhs) : (i16, i16) -> ()

        %rhs_next = arith.addi %rhs, %cst1 : i16
        scf.yield %rhs_next : i16
    }

    %lhs_next = arith.addi %lhs, %cst1 : i16
    scf.yield %lhs_next : i16
  }

  return
}

//===----------------------------------------------------------------------===//
// Test arith.muli
//===----------------------------------------------------------------------===//

// Ops in this function will be emulated using i8 ops.
func.func @emulate_muli(%lhs : i16, %rhs : i16) -> (i16) {
  %res = arith.muli %lhs, %rhs : i16
  return %res : i16
}

// Performs both wide and emulated `arith.muli`, and checks that the results
// match.
func.func @check_muli(%lhs : i16, %rhs : i16) -> () {
  %wide = arith.muli %lhs, %rhs : i16
  %emulated = func.call @emulate_muli(%lhs, %rhs) : (i16, i16) -> (i16)
  func.call @check_results(%lhs, %rhs, %wide, %emulated) : (i16, i16, i16, i16) -> ()
  return
}

// Checks that `arith.muli` is emulated properly by sampling the input space.
// In total, this test function checks 500 * 500 = 250k input pairs.
func.func @test_muli() -> () {
  %idx0 = arith.constant 0 : index
  %idx1 = arith.constant 1 : index
  %idx500 = arith.constant 500 : index

  %cst0 = arith.constant 0 : i16
  %cst1 = arith.constant 1 : i16

  scf.for %lhs_idx = %idx0 to %idx500 step %idx1 iter_args(%lhs = %cst0) -> (i16) {
    %arg_lhs = func.call @xhash(%lhs) : (i16) -> (i16)

    scf.for %rhs_idx = %idx0 to %idx500 step %idx1 iter_args(%rhs = %cst0) -> (i16) {
        %arg_rhs = func.call @xhash(%rhs) : (i16) -> (i16)
        func.call @check_muli(%arg_lhs, %arg_rhs) : (i16, i16) -> ()

        %rhs_next = arith.addi %rhs, %cst1 : i16
        scf.yield %rhs_next : i16
    }

    %lhs_next = arith.addi %lhs, %cst1 : i16
    scf.yield %lhs_next : i16
  }

  return
}

//===----------------------------------------------------------------------===//
// Test arith.shli
//===----------------------------------------------------------------------===//

// Ops in this function will be emulated using i8 ops.
func.func @emulate_shli(%lhs : i16, %rhs : i16) -> (i16) {
  %res = arith.shli %lhs, %rhs : i16
  return %res : i16
}

// Performs both wide and emulated `arith.shli`, and checks that the results
// match.
func.func @check_shli(%lhs : i16, %rhs : i16) -> () {
  %wide = arith.shli %lhs, %rhs : i16
  %emulated = func.call @emulate_shli(%lhs, %rhs) : (i16, i16) -> (i16)
  func.call @check_results(%lhs, %rhs, %wide, %emulated) : (i16, i16, i16, i16) -> ()
  return
}

// Checks that `arith.shli` is emulated properly by sampling the input space.
// Checks all valid shift amounts for i16: 0 to 15.
// In total, this test function checks 100 * 16 = 1.6k input pairs.
func.func @test_shli() -> () {
  %idx0 = arith.constant 0 : index
  %idx1 = arith.constant 1 : index
  %idx16 = arith.constant 16 : index
  %idx100 = arith.constant 100 : index

  %cst0 = arith.constant 0 : i16
  %cst1 = arith.constant 1 : i16

  scf.for %lhs_idx = %idx0 to %idx100 step %idx1 iter_args(%lhs = %cst0) -> (i16) {
    %arg_lhs = func.call @xhash(%lhs) : (i16) -> (i16)

    scf.for %rhs_idx = %idx0 to %idx16 step %idx1 iter_args(%rhs = %cst0) -> (i16) {
        func.call @check_shli(%arg_lhs, %rhs) : (i16, i16) -> ()
        %rhs_next = arith.addi %rhs, %cst1 : i16
        scf.yield %rhs_next : i16
    }

    %lhs_next = arith.addi %lhs, %cst1 : i16
    scf.yield %lhs_next : i16
  }

  return
}

//===----------------------------------------------------------------------===//
// Test arith.shrui
//===----------------------------------------------------------------------===//

// Ops in this function will be emulated using i8 ops.
func.func @emulate_shrui(%lhs : i16, %rhs : i16) -> (i16) {
  %res = arith.shrui %lhs, %rhs : i16
  return %res : i16
}

// Performs both wide and emulated `arith.shrui`, and checks that the results
// match.
func.func @check_shrui(%lhs : i16, %rhs : i16) -> () {
  %wide = arith.shrui %lhs, %rhs : i16
  %emulated = func.call @emulate_shrui(%lhs, %rhs) : (i16, i16) -> (i16)
  func.call @check_results(%lhs, %rhs, %wide, %emulated) : (i16, i16, i16, i16) -> ()
  return
}

// Checks that `arith.shrui` is emulated properly by sampling the input space.
// Checks all valid shift amounts for i16: 0 to 15.
// In total, this test function checks 100 * 16 = 1.6k input pairs.
func.func @test_shrui() -> () {
  %idx0 = arith.constant 0 : index
  %idx1 = arith.constant 1 : index
  %idx16 = arith.constant 16 : index
  %idx100 = arith.constant 100 : index

  %cst0 = arith.constant 0 : i16
  %cst1 = arith.constant 1 : i16

  scf.for %lhs_idx = %idx0 to %idx100 step %idx1 iter_args(%lhs = %cst0) -> (i16) {
    %arg_lhs = func.call @xhash(%lhs) : (i16) -> (i16)

    scf.for %rhs_idx = %idx0 to %idx16 step %idx1 iter_args(%rhs = %cst0) -> (i16) {
        func.call @check_shrui(%arg_lhs, %rhs) : (i16, i16) -> ()
        %rhs_next = arith.addi %rhs, %cst1 : i16
        scf.yield %rhs_next : i16
    }

    %lhs_next = arith.addi %lhs, %cst1 : i16
    scf.yield %lhs_next : i16
  }

  return
}

//===----------------------------------------------------------------------===//
// Entry Point
//===----------------------------------------------------------------------===//

func.func @entry() {
  func.call @test_addi() : () -> ()
  func.call @test_muli() : () -> ()
  func.call @test_shli() : () -> ()
  func.call @test_shrui() : () -> ()
  return
}
