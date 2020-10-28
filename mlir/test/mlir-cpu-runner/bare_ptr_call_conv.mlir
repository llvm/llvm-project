// RUN: mlir-opt %s -convert-scf-to-std -convert-std-to-llvm='use-bare-ptr-memref-call-conv=1' | mlir-cpu-runner -shared-libs=%linalg_test_lib_dir/libmlir_c_runner_utils%shlibext -entry-point-result=void | FileCheck %s

// Verify bare pointer memref calling convention. `simple_add1_add2_test`
// gets two 2xf32 memrefs, adds 1.0f to the first one and 2.0f to the second
// one. 'main' calls 'simple_add1_add2_test' with {1, 1} and {2, 2} so {2, 2}
// and {4, 4} are the expected outputs.

func @simple_add1_add2_test(%arg0: memref<2xf32>, %arg1: memref<2xf32>) {
  %c2 = constant 2 : index
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %cst = constant 1.000000e+00 : f32
  %cst_0 = constant 2.000000e+00 : f32
  scf.for %arg2 = %c0 to %c2 step %c1 {
    %0 = load %arg0[%arg2] : memref<2xf32>
    %1 = addf %0, %cst : f32
    store %1, %arg0[%arg2] : memref<2xf32>
    // CHECK: 2, 2

    %2 = load %arg1[%arg2] : memref<2xf32>
    %3 = addf %1, %cst_0 : f32
    store %3, %arg1[%arg2] : memref<2xf32>
    // CHECK-NEXT: 4, 4
  }
  return
}

// External declarations.
llvm.func @malloc(!llvm.i64) -> !llvm.ptr<i8>
llvm.func @free(!llvm.ptr<i8>)
func @print_f32(%arg0: f32)
func @print_comma()
func @print_newline()

// TODO: 'main' function currently has to be provided in LLVM dialect since
// 'call' op is not yet supported by the bare pointer calling convention. The
// LLVM dialect version was generated using the following loop/std dialect
// version and minor changes around the 'simple_add1_add2_test' call.

//func @main()
//{
//  %c2 = constant 2 : index
//  %c0 = constant 0 : index
//  %c1 = constant 1 : index
//  %cst = constant 1.000000e+00 : f32
//  %cst_0 = constant 2.000000e+00 : f32
//  %a = alloc() : memref<2xf32>
//  %b = alloc() : memref<2xf32>
//  scf.for %i = %c0 to %c2 step %c1 {
//    store %cst, %a[%i] : memref<2xf32>
//    store %cst, %b[%i] : memref<2xf32>
//  }
//
//  call @simple_add1_add2_test(%a, %b) : (memref<2xf32>, memref<2xf32>) -> ()
//
//  %l0 = load %a[%c0] : memref<2xf32>
//  call @print_f32(%l0) : (f32) -> ()
//  call @print_comma() : () -> ()
//  %l1 = load %a[%c1] : memref<2xf32>
//  call @print_f32(%l1) : (f32) -> ()
//  call @print_newline() : () -> ()
//
//  %l2 = load %b[%c0] : memref<2xf32>
//  call @print_f32(%l2) : (f32) -> ()
//  call @print_comma() : () -> ()
//  %l3 = load %b[%c1] : memref<2xf32>
//  call @print_f32(%l3) : (f32) -> ()
//  call @print_newline() : () -> ()
//
//  dealloc %a : memref<2xf32>
//  dealloc %b : memref<2xf32>
//  return
//}

llvm.func @main() {
  %0 = llvm.mlir.constant(2 : index) : !llvm.i64
  %1 = llvm.mlir.constant(0 : index) : !llvm.i64
  %2 = llvm.mlir.constant(1 : index) : !llvm.i64
  %3 = llvm.mlir.constant(1.000000e+00 : f32) : !llvm.float
  %4 = llvm.mlir.constant(2.000000e+00 : f32) : !llvm.float
  %5 = llvm.mlir.constant(2 : index) : !llvm.i64
  %6 = llvm.mlir.null : !llvm.ptr<float>
  %7 = llvm.mlir.constant(1 : index) : !llvm.i64
  %8 = llvm.getelementptr %6[%7] : (!llvm.ptr<float>, !llvm.i64) -> !llvm.ptr<float>
  %9 = llvm.ptrtoint %8 : !llvm.ptr<float> to !llvm.i64
  %10 = llvm.mul %5, %9 : !llvm.i64
  %11 = llvm.call @malloc(%10) : (!llvm.i64) -> !llvm.ptr<i8>
  %12 = llvm.bitcast %11 : !llvm.ptr<i8> to !llvm.ptr<float>
  %13 = llvm.mlir.undef : !llvm.struct<(ptr<float>, ptr<float>, i64, array<1 x i64>, array<1 x i64>)>
  %14 = llvm.insertvalue %12, %13[0] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<1 x i64>, array<1 x i64>)>
  %15 = llvm.insertvalue %12, %14[1] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<1 x i64>, array<1 x i64>)>
  %16 = llvm.mlir.constant(0 : index) : !llvm.i64
  %17 = llvm.insertvalue %16, %15[2] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<1 x i64>, array<1 x i64>)>
  %18 = llvm.mlir.constant(1 : index) : !llvm.i64
  %19 = llvm.insertvalue %5, %17[3, 0] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<1 x i64>, array<1 x i64>)>
  %20 = llvm.insertvalue %18, %19[4, 0] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<1 x i64>, array<1 x i64>)>
  %21 = llvm.mlir.constant(2 : index) : !llvm.i64
  %22 = llvm.mlir.null : !llvm.ptr<float>
  %23 = llvm.mlir.constant(1 : index) : !llvm.i64
  %24 = llvm.getelementptr %22[%23] : (!llvm.ptr<float>, !llvm.i64) -> !llvm.ptr<float>
  %25 = llvm.ptrtoint %24 : !llvm.ptr<float> to !llvm.i64
  %26 = llvm.mul %21, %25 : !llvm.i64
  %27 = llvm.call @malloc(%26) : (!llvm.i64) -> !llvm.ptr<i8>
  %28 = llvm.bitcast %27 : !llvm.ptr<i8> to !llvm.ptr<float>
  %29 = llvm.mlir.undef : !llvm.struct<(ptr<float>, ptr<float>, i64, array<1 x i64>, array<1 x i64>)>
  %30 = llvm.insertvalue %28, %29[0] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<1 x i64>, array<1 x i64>)>
  %31 = llvm.insertvalue %28, %30[1] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<1 x i64>, array<1 x i64>)>
  %32 = llvm.mlir.constant(0 : index) : !llvm.i64
  %33 = llvm.insertvalue %32, %31[2] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<1 x i64>, array<1 x i64>)>
  %34 = llvm.mlir.constant(1 : index) : !llvm.i64
  %35 = llvm.insertvalue %21, %33[3, 0] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<1 x i64>, array<1 x i64>)>
  %36 = llvm.insertvalue %34, %35[4, 0] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<1 x i64>, array<1 x i64>)>
  llvm.br ^bb1(%1 : !llvm.i64)
^bb1(%37: !llvm.i64):	// 2 preds: ^bb0, ^bb2
  %38 = llvm.icmp "slt" %37, %0 : !llvm.i64
  llvm.cond_br %38, ^bb2, ^bb3
^bb2:	// pred: ^bb1
  %39 = llvm.extractvalue %20[1] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<1 x i64>, array<1 x i64>)>
  %40 = llvm.mlir.constant(0 : index) : !llvm.i64
  %41 = llvm.mlir.constant(1 : index) : !llvm.i64
  %42 = llvm.mul %37, %41 : !llvm.i64
  %43 = llvm.add %40, %42 : !llvm.i64
  %44 = llvm.getelementptr %39[%43] : (!llvm.ptr<float>, !llvm.i64) -> !llvm.ptr<float>
  llvm.store %3, %44 : !llvm.ptr<float>
  %45 = llvm.extractvalue %36[1] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<1 x i64>, array<1 x i64>)>
  %46 = llvm.mlir.constant(0 : index) : !llvm.i64
  %47 = llvm.mlir.constant(1 : index) : !llvm.i64
  %48 = llvm.mul %37, %47 : !llvm.i64
  %49 = llvm.add %46, %48 : !llvm.i64
  %50 = llvm.getelementptr %45[%49] : (!llvm.ptr<float>, !llvm.i64) -> !llvm.ptr<float>
  llvm.store %3, %50 : !llvm.ptr<float>
  %51 = llvm.add %37, %2 : !llvm.i64
  llvm.br ^bb1(%51 : !llvm.i64)
^bb3:	// pred: ^bb1
  %52 = llvm.mlir.constant(1 : index) : !llvm.i64
  %53 = llvm.mlir.constant(1 : index) : !llvm.i64
  %54 = llvm.extractvalue %20[1] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<1 x i64>, array<1 x i64>)>
  %55 = llvm.extractvalue %36[1] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<1 x i64>, array<1 x i64>)>
  llvm.call @simple_add1_add2_test(%54, %55) : (!llvm.ptr<float>, !llvm.ptr<float>) -> ()
  %56 = llvm.extractvalue %20[1] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<1 x i64>, array<1 x i64>)>
  %57 = llvm.mlir.constant(0 : index) : !llvm.i64
  %58 = llvm.mlir.constant(1 : index) : !llvm.i64
  %59 = llvm.mul %1, %58 : !llvm.i64
  %60 = llvm.add %57, %59 : !llvm.i64
  %61 = llvm.getelementptr %56[%60] : (!llvm.ptr<float>, !llvm.i64) -> !llvm.ptr<float>
  %62 = llvm.load %61 : !llvm.ptr<float>
  llvm.call @print_f32(%62) : (!llvm.float) -> ()
  llvm.call @print_comma() : () -> ()
  %63 = llvm.extractvalue %20[1] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<1 x i64>, array<1 x i64>)>
  %64 = llvm.mlir.constant(0 : index) : !llvm.i64
  %65 = llvm.mlir.constant(1 : index) : !llvm.i64
  %66 = llvm.mul %2, %65 : !llvm.i64
  %67 = llvm.add %64, %66 : !llvm.i64
  %68 = llvm.getelementptr %63[%67] : (!llvm.ptr<float>, !llvm.i64) -> !llvm.ptr<float>
  %69 = llvm.load %68 : !llvm.ptr<float>
  llvm.call @print_f32(%69) : (!llvm.float) -> ()
  llvm.call @print_newline() : () -> ()
  %70 = llvm.extractvalue %36[1] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<1 x i64>, array<1 x i64>)>
  %71 = llvm.mlir.constant(0 : index) : !llvm.i64
  %72 = llvm.mlir.constant(1 : index) : !llvm.i64
  %73 = llvm.mul %1, %72 : !llvm.i64
  %74 = llvm.add %71, %73 : !llvm.i64
  %75 = llvm.getelementptr %70[%74] : (!llvm.ptr<float>, !llvm.i64) -> !llvm.ptr<float>
  %76 = llvm.load %75 : !llvm.ptr<float>
  llvm.call @print_f32(%76) : (!llvm.float) -> ()
  llvm.call @print_comma() : () -> ()
  %77 = llvm.extractvalue %36[1] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<1 x i64>, array<1 x i64>)>
  %78 = llvm.mlir.constant(0 : index) : !llvm.i64
  %79 = llvm.mlir.constant(1 : index) : !llvm.i64
  %80 = llvm.mul %2, %79 : !llvm.i64
  %81 = llvm.add %78, %80 : !llvm.i64
  %82 = llvm.getelementptr %77[%81] : (!llvm.ptr<float>, !llvm.i64) -> !llvm.ptr<float>
  %83 = llvm.load %82 : !llvm.ptr<float>
  llvm.call @print_f32(%83) : (!llvm.float) -> ()
  llvm.call @print_newline() : () -> ()
  %84 = llvm.extractvalue %20[0] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<1 x i64>, array<1 x i64>)>
  %85 = llvm.bitcast %84 : !llvm.ptr<float> to !llvm.ptr<i8>
  llvm.call @free(%85) : (!llvm.ptr<i8>) -> ()
  %86 = llvm.extractvalue %36[0] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<1 x i64>, array<1 x i64>)>
  %87 = llvm.bitcast %86 : !llvm.ptr<float> to !llvm.ptr<i8>
  llvm.call @free(%87) : (!llvm.ptr<i8>) -> ()
  llvm.return
}
