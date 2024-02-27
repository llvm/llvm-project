// RUN: mlir-translate -mlir-to-llvmir %s -debug-only=llvm-dialect-to-llvm-ir 2>&1 | FileCheck %s

// CHECK: Convert initializer for dup_const
// CHECK: 6 new constants hit
// CHECK: 3 dangling constants erased
// CHECK: Convert initializer for unique_const
// CHECK: 6 new constants hit
// CHECK: 5 dangling constants erased


// CHECK:@dup_const = global { [2 x double], [2 x double], [2 x double] } { [2 x double] [double 3.612250e-02, double 5.119230e-02], [2 x double] [double 3.612250e-02, double 5.119230e-02], [2 x double] [double 3.612250e-02, double 5.119230e-02] }

llvm.mlir.global @dup_const() : !llvm.struct<(array<2 x f64>, array<2 x f64>, array<2 x f64>)> {
    %c0 = llvm.mlir.constant(3.612250e-02 : f64) : f64
    %c1 = llvm.mlir.constant(5.119230e-02 : f64) : f64

    %empty0 = llvm.mlir.undef : !llvm.array<2 x f64>
    %a00 = llvm.insertvalue %c0, %empty0[0] : !llvm.array<2 x f64>

    %empty1 = llvm.mlir.undef : !llvm.array<2 x f64>
    %a10 = llvm.insertvalue %c0, %empty1[0] : !llvm.array<2 x f64>

    %empty2 = llvm.mlir.undef : !llvm.array<2 x f64>
    %a20 = llvm.insertvalue %c0, %empty2[0] : !llvm.array<2 x f64>

// NOTE: a00, a10, a20 are all same ConstantAggregate which not used at this point.
//       should not delete it before all of the uses of the ConstantAggregate finished.

    %a01 = llvm.insertvalue %c1, %a00[1] : !llvm.array<2 x f64>
    %a11 = llvm.insertvalue %c1, %a10[1] : !llvm.array<2 x f64>
    %a21 = llvm.insertvalue %c1, %a20[1] : !llvm.array<2 x f64>
    %empty_r = llvm.mlir.undef : !llvm.struct<(array<2 x f64>, array<2 x f64>, array<2 x f64>)>
    %r0 = llvm.insertvalue %a01, %empty_r[0] : !llvm.struct<(array<2 x f64>, array<2 x f64>, array<2 x f64>)>
    %r1 = llvm.insertvalue %a11, %r0[1] : !llvm.struct<(array<2 x f64>, array<2 x f64>, array<2 x f64>)>
    %r2 = llvm.insertvalue %a21, %r1[2] : !llvm.struct<(array<2 x f64>, array<2 x f64>, array<2 x f64>)>

    llvm.return %r2 : !llvm.struct<(array<2 x f64>, array<2 x f64>, array<2 x f64>)>
  }

// CHECK:@unique_const = global { [2 x double], [2 x double], [2 x double] } { [2 x double] [double 3.612250e-02, double 5.119230e-02], [2 x double] [double 3.312250e-02, double 5.219230e-02], [2 x double] [double 3.412250e-02, double 5.419230e-02] }

llvm.mlir.global @unique_const() : !llvm.struct<(array<2 x f64>, array<2 x f64>, array<2 x f64>)> {
    %c0 = llvm.mlir.constant(3.612250e-02 : f64) : f64
    %c1 = llvm.mlir.constant(5.119230e-02 : f64) : f64

    %c2 = llvm.mlir.constant(3.312250e-02 : f64) : f64
    %c3 = llvm.mlir.constant(5.219230e-02 : f64) : f64

    %c4 = llvm.mlir.constant(3.412250e-02 : f64) : f64
    %c5 = llvm.mlir.constant(5.419230e-02 : f64) : f64

    %2 = llvm.mlir.undef : !llvm.struct<(array<2 x f64>, array<2 x f64>, array<2 x f64>)>

    %3 = llvm.mlir.undef : !llvm.array<2 x f64>

    %4 = llvm.insertvalue %c0, %3[0] : !llvm.array<2 x f64>
    %5 = llvm.insertvalue %c1, %4[1] : !llvm.array<2 x f64>

    %6 = llvm.insertvalue %5, %2[0] : !llvm.struct<(array<2 x f64>, array<2 x f64>, array<2 x f64>)>

    %7 = llvm.insertvalue %c2, %3[0] : !llvm.array<2 x f64>
    %8 = llvm.insertvalue %c3, %7[1] : !llvm.array<2 x f64>

    %9 = llvm.insertvalue %8, %6[1] : !llvm.struct<(array<2 x f64>, array<2 x f64>, array<2 x f64>)>

    %10 = llvm.insertvalue %c4, %3[0] : !llvm.array<2 x f64>
    %11 = llvm.insertvalue %c5, %10[1] : !llvm.array<2 x f64>

    %12 = llvm.insertvalue %11, %9[2] : !llvm.struct<(array<2 x f64>, array<2 x f64>, array<2 x f64>)>

    llvm.return %12 : !llvm.struct<(array<2 x f64>, array<2 x f64>, array<2 x f64>)>
}
