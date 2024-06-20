module {
  func.func @main() {
    %cst = arith.constant 0.000000e+00 : f64
    %cst_0 = arith.constant 6.000000e+00 : f64
    %cst_1 = arith.constant 5.000000e+00 : f64
    %cst_2 = arith.constant 4.000000e+00 : f64
    %cst_3 = arith.constant 3.000000e+00 : f64
    %cst_4 = arith.constant 2.000000e+00 : f64
    %cst_5 = arith.constant 1.000000e+00 : f64
    %alloc = memref.alloc() : memref<1x6xf64>
    %alloc_6 = memref.alloc() : memref<6x1xf64>
    affine.store %cst_5, %alloc_6[0, 0] : memref<6x1xf64>
    affine.store %cst_4, %alloc_6[1, 0] : memref<6x1xf64>
    affine.store %cst_3, %alloc_6[2, 0] : memref<6x1xf64>
    affine.store %cst_2, %alloc_6[3, 0] : memref<6x1xf64>
    affine.store %cst_1, %alloc_6[4, 0] : memref<6x1xf64>
    affine.store %cst_0, %alloc_6[5, 0] : memref<6x1xf64>
    affine.store %cst_5, %alloc[0, 0] : memref<1x6xf64>
    affine.store %cst_4, %alloc[0, 1] : memref<1x6xf64>
    affine.store %cst_3, %alloc[0, 2] : memref<1x6xf64>
    affine.store %cst_2, %alloc[0, 3] : memref<1x6xf64>
    affine.store %cst_1, %alloc[0, 4] : memref<1x6xf64>
    affine.store %cst_0, %alloc[0, 5] : memref<1x6xf64>
    %alloc_7 = memref.alloc() : memref<1x1xf64>
    %0 = affine.load %alloc[0, 0] : memref<1x6xf64>
    %1 = affine.load %alloc_6[0, 0] : memref<6x1xf64>
    %2 = arith.mulf %0, %1 : f64
    %3 = arith.addf %2, %cst : f64
    %4 = affine.load %alloc[0, 1] : memref<1x6xf64>
    %5 = affine.load %alloc_6[1, 0] : memref<6x1xf64>
    %6 = arith.mulf %4, %5 : f64
    %7 = arith.addf %3, %6 : f64
    %8 = affine.load %alloc[0, 2] : memref<1x6xf64>
    %9 = affine.load %alloc_6[2, 0] : memref<6x1xf64>
    %10 = arith.mulf %8, %9 : f64
    %11 = arith.addf %7, %10 : f64
    %12 = affine.load %alloc[0, 3] : memref<1x6xf64>
    %13 = affine.load %alloc_6[3, 0] : memref<6x1xf64>
    %14 = arith.mulf %12, %13 : f64
    %15 = arith.addf %11, %14 : f64
    %16 = affine.load %alloc[0, 4] : memref<1x6xf64>
    %17 = affine.load %alloc_6[4, 0] : memref<6x1xf64>
    %18 = arith.mulf %16, %17 : f64
    %19 = arith.addf %15, %18 : f64
    %20 = affine.load %alloc[0, 5] : memref<1x6xf64>
    %21 = affine.load %alloc_6[5, 0] : memref<6x1xf64>
    %22 = arith.mulf %20, %21 : f64
    %23 = arith.addf %19, %22 : f64
    affine.store %23, %alloc_7[0, 0] : memref<1x1xf64>
    toy.print %alloc_7 : memref<1x1xf64>
    memref.dealloc %alloc_6 : memref<6x1xf64>
    memref.dealloc %alloc : memref<1x6xf64>
    return
  }
}
