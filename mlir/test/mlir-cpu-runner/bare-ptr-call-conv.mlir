// RUN: mlir-opt %s -pass-pipeline="builtin.module(func.func(convert-scf-to-cf,convert-arith-to-llvm),finalize-memref-to-llvm,convert-func-to-llvm{use-bare-ptr-memref-call-conv=1},reconcile-unrealized-casts)" | mlir-cpu-runner -shared-libs=%mlir_c_runner_utils -entry-point-result=void | FileCheck %s

// Verify bare pointer memref calling convention. `simple_add1_add2_test`
// gets two 2xf32 memrefs, adds 1.0f to the first one and 2.0f to the second
// one. 'main' calls 'simple_add1_add2_test' with {1, 1} and {2, 2} so {2, 2}
// and {4, 4} are the expected outputs.

func.func @simple_add1_add2_test(%arg0: memref<2xf32>, %arg1: memref<2xf32>) {
  %c2 = arith.constant 2 : index
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %cst = arith.constant 1.000000e+00 : f32
  %cst_0 = arith.constant 2.000000e+00 : f32
  scf.for %arg2 = %c0 to %c2 step %c1 {
    %0 = memref.load %arg0[%arg2] : memref<2xf32>
    %1 = arith.addf %0, %cst : f32
    memref.store %1, %arg0[%arg2] : memref<2xf32>
    // CHECK: 2, 2

    %2 = memref.load %arg1[%arg2] : memref<2xf32>
    %3 = arith.addf %1, %cst_0 : f32
    memref.store %3, %arg1[%arg2] : memref<2xf32>
    // CHECK-NEXT: 4, 4
  }
  return
}

// External declarations.
llvm.func @malloc(i64) -> !llvm.ptr<i8>
llvm.func @free(!llvm.ptr<i8>)
func.func private @printF32(%arg0: f32)
func.func private @printComma()
func.func private @printNewline()

func.func @main()
{
  %c2 = arith.constant 2 : index
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %cst = arith.constant 1.000000e+00 : f32
  %cst_0 = arith.constant 2.000000e+00 : f32
  %a = memref.alloc() : memref<2xf32>
  %b = memref.alloc() : memref<2xf32>
  scf.for %i = %c0 to %c2 step %c1 {
    memref.store %cst, %a[%i] : memref<2xf32>
    memref.store %cst, %b[%i] : memref<2xf32>
  }

  call @simple_add1_add2_test(%a, %b) : (memref<2xf32>, memref<2xf32>) -> ()

  %l0 = memref.load %a[%c0] : memref<2xf32>
  call @printF32(%l0) : (f32) -> ()
  call @printComma() : () -> ()
  %l1 = memref.load %a[%c1] : memref<2xf32>
  call @printF32(%l1) : (f32) -> ()
  call @printNewline() : () -> ()

  %l2 = memref.load %b[%c0] : memref<2xf32>
  call @printF32(%l2) : (f32) -> ()
  call @printComma() : () -> ()
  %l3 = memref.load %b[%c1] : memref<2xf32>
  call @printF32(%l3) : (f32) -> ()
  call @printNewline() : () -> ()

  memref.dealloc %a : memref<2xf32>
  memref.dealloc %b : memref<2xf32>
  return
}
