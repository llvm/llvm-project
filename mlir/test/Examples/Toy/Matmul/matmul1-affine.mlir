module {
  func.func @main() {
    %cst = arith.constant 0.000000e+00 : f64
    %cst_0 = arith.constant 6.000000e+00 : f64
    %cst_1 = arith.constant 5.000000e+00 : f64
    %cst_2 = arith.constant 4.000000e+00 : f64
    %cst_3 = arith.constant 3.000000e+00 : f64
    %cst_4 = arith.constant 2.000000e+00 : f64
    %cst_5 = arith.constant 1.000000e+00 : f64
    %alloc = memref.alloc() : memref<3x2xf64>
    %alloc_6 = memref.alloc() : memref<2x3xf64>
    affine.store %cst_5, %alloc_6[0, 0] : memref<2x3xf64>
    affine.store %cst_4, %alloc_6[0, 1] : memref<2x3xf64>
    affine.store %cst_3, %alloc_6[0, 2] : memref<2x3xf64>
    affine.store %cst_2, %alloc_6[1, 0] : memref<2x3xf64>
    affine.store %cst_1, %alloc_6[1, 1] : memref<2x3xf64>
    affine.store %cst_0, %alloc_6[1, 2] : memref<2x3xf64>
    affine.store %cst_5, %alloc[0, 0] : memref<3x2xf64>
    affine.store %cst_4, %alloc[0, 1] : memref<3x2xf64>
    affine.store %cst_3, %alloc[1, 0] : memref<3x2xf64>
    affine.store %cst_2, %alloc[1, 1] : memref<3x2xf64>
    affine.store %cst_1, %alloc[2, 0] : memref<3x2xf64>
    affine.store %cst_0, %alloc[2, 1] : memref<3x2xf64>
    %alloc_7 = memref.alloc() : memref<2x2xf64>
    %0 = affine.load %alloc_6[0, 0] : memref<2x3xf64>
    %1 = affine.load %alloc[0, 0] : memref<3x2xf64>
    %2 = arith.mulf %0, %1 : f64
    %3 = arith.addf %2, %cst : f64
    %4 = affine.load %alloc_6[0, 1] : memref<2x3xf64>
    %5 = affine.load %alloc[1, 0] : memref<3x2xf64>
    %6 = arith.mulf %4, %5 : f64
    %7 = arith.addf %3, %6 : f64
    %8 = affine.load %alloc_6[0, 2] : memref<2x3xf64>
    %9 = affine.load %alloc[2, 0] : memref<3x2xf64>
    %10 = arith.mulf %8, %9 : f64
    %11 = arith.addf %7, %10 : f64
    affine.store %11, %alloc_7[0, 0] : memref<2x2xf64>
    %12 = affine.load %alloc_6[0, 0] : memref<2x3xf64>
    %13 = affine.load %alloc[0, 1] : memref<3x2xf64>
    %14 = arith.mulf %12, %13 : f64
    %15 = arith.addf %14, %cst : f64
    %16 = affine.load %alloc_6[0, 1] : memref<2x3xf64>
    %17 = affine.load %alloc[1, 1] : memref<3x2xf64>
    %18 = arith.mulf %16, %17 : f64
    %19 = arith.addf %15, %18 : f64
    %20 = affine.load %alloc_6[0, 2] : memref<2x3xf64>
    %21 = affine.load %alloc[2, 1] : memref<3x2xf64>
    %22 = arith.mulf %20, %21 : f64
    %23 = arith.addf %19, %22 : f64
    affine.store %23, %alloc_7[0, 1] : memref<2x2xf64>
    %24 = affine.load %alloc_6[1, 0] : memref<2x3xf64>
    %25 = affine.load %alloc[0, 0] : memref<3x2xf64>
    %26 = arith.mulf %24, %25 : f64
    %27 = arith.addf %26, %cst : f64
    %28 = affine.load %alloc_6[1, 1] : memref<2x3xf64>
    %29 = affine.load %alloc[1, 0] : memref<3x2xf64>
    %30 = arith.mulf %28, %29 : f64
    %31 = arith.addf %27, %30 : f64
    %32 = affine.load %alloc_6[1, 2] : memref<2x3xf64>
    %33 = affine.load %alloc[2, 0] : memref<3x2xf64>
    %34 = arith.mulf %32, %33 : f64
    %35 = arith.addf %31, %34 : f64
    affine.store %35, %alloc_7[1, 0] : memref<2x2xf64>
    %36 = affine.load %alloc_6[1, 0] : memref<2x3xf64>
    %37 = affine.load %alloc[0, 1] : memref<3x2xf64>
    %38 = arith.mulf %36, %37 : f64
    %39 = arith.addf %38, %cst : f64
    %40 = affine.load %alloc_6[1, 1] : memref<2x3xf64>
    %41 = affine.load %alloc[1, 1] : memref<3x2xf64>
    %42 = arith.mulf %40, %41 : f64
    %43 = arith.addf %39, %42 : f64
    %44 = affine.load %alloc_6[1, 2] : memref<2x3xf64>
    %45 = affine.load %alloc[2, 1] : memref<3x2xf64>
    %46 = arith.mulf %44, %45 : f64
    %47 = arith.addf %43, %46 : f64
    affine.store %47, %alloc_7[1, 1] : memref<2x2xf64>
    toy.print %alloc_7 : memref<2x2xf64>
    memref.dealloc %alloc_6 : memref<2x3xf64>
    memref.dealloc %alloc : memref<3x2xf64>
    return
  }
}
