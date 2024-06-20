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
    %alloc_7 = memref.alloc() : memref<6x6xf64>
    %0 = affine.load %alloc_6[0, 0] : memref<6x1xf64>
    %1 = affine.load %alloc[0, 0] : memref<1x6xf64>
    %2 = arith.mulf %0, %1 : f64
    %3 = arith.addf %2, %cst : f64
    affine.store %3, %alloc_7[0, 0] : memref<6x6xf64>
    %4 = affine.load %alloc_6[0, 0] : memref<6x1xf64>
    %5 = affine.load %alloc[0, 1] : memref<1x6xf64>
    %6 = arith.mulf %4, %5 : f64
    %7 = arith.addf %6, %cst : f64
    affine.store %7, %alloc_7[0, 1] : memref<6x6xf64>
    %8 = affine.load %alloc_6[0, 0] : memref<6x1xf64>
    %9 = affine.load %alloc[0, 2] : memref<1x6xf64>
    %10 = arith.mulf %8, %9 : f64
    %11 = arith.addf %10, %cst : f64
    affine.store %11, %alloc_7[0, 2] : memref<6x6xf64>
    %12 = affine.load %alloc_6[0, 0] : memref<6x1xf64>
    %13 = affine.load %alloc[0, 3] : memref<1x6xf64>
    %14 = arith.mulf %12, %13 : f64
    %15 = arith.addf %14, %cst : f64
    affine.store %15, %alloc_7[0, 3] : memref<6x6xf64>
    %16 = affine.load %alloc_6[0, 0] : memref<6x1xf64>
    %17 = affine.load %alloc[0, 4] : memref<1x6xf64>
    %18 = arith.mulf %16, %17 : f64
    %19 = arith.addf %18, %cst : f64
    affine.store %19, %alloc_7[0, 4] : memref<6x6xf64>
    %20 = affine.load %alloc_6[0, 0] : memref<6x1xf64>
    %21 = affine.load %alloc[0, 5] : memref<1x6xf64>
    %22 = arith.mulf %20, %21 : f64
    %23 = arith.addf %22, %cst : f64
    affine.store %23, %alloc_7[0, 5] : memref<6x6xf64>
    %24 = affine.load %alloc_6[1, 0] : memref<6x1xf64>
    %25 = affine.load %alloc[0, 0] : memref<1x6xf64>
    %26 = arith.mulf %24, %25 : f64
    %27 = arith.addf %26, %cst : f64
    affine.store %27, %alloc_7[1, 0] : memref<6x6xf64>
    %28 = affine.load %alloc_6[1, 0] : memref<6x1xf64>
    %29 = affine.load %alloc[0, 1] : memref<1x6xf64>
    %30 = arith.mulf %28, %29 : f64
    %31 = arith.addf %30, %cst : f64
    affine.store %31, %alloc_7[1, 1] : memref<6x6xf64>
    %32 = affine.load %alloc_6[1, 0] : memref<6x1xf64>
    %33 = affine.load %alloc[0, 2] : memref<1x6xf64>
    %34 = arith.mulf %32, %33 : f64
    %35 = arith.addf %34, %cst : f64
    affine.store %35, %alloc_7[1, 2] : memref<6x6xf64>
    %36 = affine.load %alloc_6[1, 0] : memref<6x1xf64>
    %37 = affine.load %alloc[0, 3] : memref<1x6xf64>
    %38 = arith.mulf %36, %37 : f64
    %39 = arith.addf %38, %cst : f64
    affine.store %39, %alloc_7[1, 3] : memref<6x6xf64>
    %40 = affine.load %alloc_6[1, 0] : memref<6x1xf64>
    %41 = affine.load %alloc[0, 4] : memref<1x6xf64>
    %42 = arith.mulf %40, %41 : f64
    %43 = arith.addf %42, %cst : f64
    affine.store %43, %alloc_7[1, 4] : memref<6x6xf64>
    %44 = affine.load %alloc_6[1, 0] : memref<6x1xf64>
    %45 = affine.load %alloc[0, 5] : memref<1x6xf64>
    %46 = arith.mulf %44, %45 : f64
    %47 = arith.addf %46, %cst : f64
    affine.store %47, %alloc_7[1, 5] : memref<6x6xf64>
    %48 = affine.load %alloc_6[2, 0] : memref<6x1xf64>
    %49 = affine.load %alloc[0, 0] : memref<1x6xf64>
    %50 = arith.mulf %48, %49 : f64
    %51 = arith.addf %50, %cst : f64
    affine.store %51, %alloc_7[2, 0] : memref<6x6xf64>
    %52 = affine.load %alloc_6[2, 0] : memref<6x1xf64>
    %53 = affine.load %alloc[0, 1] : memref<1x6xf64>
    %54 = arith.mulf %52, %53 : f64
    %55 = arith.addf %54, %cst : f64
    affine.store %55, %alloc_7[2, 1] : memref<6x6xf64>
    %56 = affine.load %alloc_6[2, 0] : memref<6x1xf64>
    %57 = affine.load %alloc[0, 2] : memref<1x6xf64>
    %58 = arith.mulf %56, %57 : f64
    %59 = arith.addf %58, %cst : f64
    affine.store %59, %alloc_7[2, 2] : memref<6x6xf64>
    %60 = affine.load %alloc_6[2, 0] : memref<6x1xf64>
    %61 = affine.load %alloc[0, 3] : memref<1x6xf64>
    %62 = arith.mulf %60, %61 : f64
    %63 = arith.addf %62, %cst : f64
    affine.store %63, %alloc_7[2, 3] : memref<6x6xf64>
    %64 = affine.load %alloc_6[2, 0] : memref<6x1xf64>
    %65 = affine.load %alloc[0, 4] : memref<1x6xf64>
    %66 = arith.mulf %64, %65 : f64
    %67 = arith.addf %66, %cst : f64
    affine.store %67, %alloc_7[2, 4] : memref<6x6xf64>
    %68 = affine.load %alloc_6[2, 0] : memref<6x1xf64>
    %69 = affine.load %alloc[0, 5] : memref<1x6xf64>
    %70 = arith.mulf %68, %69 : f64
    %71 = arith.addf %70, %cst : f64
    affine.store %71, %alloc_7[2, 5] : memref<6x6xf64>
    %72 = affine.load %alloc_6[3, 0] : memref<6x1xf64>
    %73 = affine.load %alloc[0, 0] : memref<1x6xf64>
    %74 = arith.mulf %72, %73 : f64
    %75 = arith.addf %74, %cst : f64
    affine.store %75, %alloc_7[3, 0] : memref<6x6xf64>
    %76 = affine.load %alloc_6[3, 0] : memref<6x1xf64>
    %77 = affine.load %alloc[0, 1] : memref<1x6xf64>
    %78 = arith.mulf %76, %77 : f64
    %79 = arith.addf %78, %cst : f64
    affine.store %79, %alloc_7[3, 1] : memref<6x6xf64>
    %80 = affine.load %alloc_6[3, 0] : memref<6x1xf64>
    %81 = affine.load %alloc[0, 2] : memref<1x6xf64>
    %82 = arith.mulf %80, %81 : f64
    %83 = arith.addf %82, %cst : f64
    affine.store %83, %alloc_7[3, 2] : memref<6x6xf64>
    %84 = affine.load %alloc_6[3, 0] : memref<6x1xf64>
    %85 = affine.load %alloc[0, 3] : memref<1x6xf64>
    %86 = arith.mulf %84, %85 : f64
    %87 = arith.addf %86, %cst : f64
    affine.store %87, %alloc_7[3, 3] : memref<6x6xf64>
    %88 = affine.load %alloc_6[3, 0] : memref<6x1xf64>
    %89 = affine.load %alloc[0, 4] : memref<1x6xf64>
    %90 = arith.mulf %88, %89 : f64
    %91 = arith.addf %90, %cst : f64
    affine.store %91, %alloc_7[3, 4] : memref<6x6xf64>
    %92 = affine.load %alloc_6[3, 0] : memref<6x1xf64>
    %93 = affine.load %alloc[0, 5] : memref<1x6xf64>
    %94 = arith.mulf %92, %93 : f64
    %95 = arith.addf %94, %cst : f64
    affine.store %95, %alloc_7[3, 5] : memref<6x6xf64>
    %96 = affine.load %alloc_6[4, 0] : memref<6x1xf64>
    %97 = affine.load %alloc[0, 0] : memref<1x6xf64>
    %98 = arith.mulf %96, %97 : f64
    %99 = arith.addf %98, %cst : f64
    affine.store %99, %alloc_7[4, 0] : memref<6x6xf64>
    %100 = affine.load %alloc_6[4, 0] : memref<6x1xf64>
    %101 = affine.load %alloc[0, 1] : memref<1x6xf64>
    %102 = arith.mulf %100, %101 : f64
    %103 = arith.addf %102, %cst : f64
    affine.store %103, %alloc_7[4, 1] : memref<6x6xf64>
    %104 = affine.load %alloc_6[4, 0] : memref<6x1xf64>
    %105 = affine.load %alloc[0, 2] : memref<1x6xf64>
    %106 = arith.mulf %104, %105 : f64
    %107 = arith.addf %106, %cst : f64
    affine.store %107, %alloc_7[4, 2] : memref<6x6xf64>
    %108 = affine.load %alloc_6[4, 0] : memref<6x1xf64>
    %109 = affine.load %alloc[0, 3] : memref<1x6xf64>
    %110 = arith.mulf %108, %109 : f64
    %111 = arith.addf %110, %cst : f64
    affine.store %111, %alloc_7[4, 3] : memref<6x6xf64>
    %112 = affine.load %alloc_6[4, 0] : memref<6x1xf64>
    %113 = affine.load %alloc[0, 4] : memref<1x6xf64>
    %114 = arith.mulf %112, %113 : f64
    %115 = arith.addf %114, %cst : f64
    affine.store %115, %alloc_7[4, 4] : memref<6x6xf64>
    %116 = affine.load %alloc_6[4, 0] : memref<6x1xf64>
    %117 = affine.load %alloc[0, 5] : memref<1x6xf64>
    %118 = arith.mulf %116, %117 : f64
    %119 = arith.addf %118, %cst : f64
    affine.store %119, %alloc_7[4, 5] : memref<6x6xf64>
    %120 = affine.load %alloc_6[5, 0] : memref<6x1xf64>
    %121 = affine.load %alloc[0, 0] : memref<1x6xf64>
    %122 = arith.mulf %120, %121 : f64
    %123 = arith.addf %122, %cst : f64
    affine.store %123, %alloc_7[5, 0] : memref<6x6xf64>
    %124 = affine.load %alloc_6[5, 0] : memref<6x1xf64>
    %125 = affine.load %alloc[0, 1] : memref<1x6xf64>
    %126 = arith.mulf %124, %125 : f64
    %127 = arith.addf %126, %cst : f64
    affine.store %127, %alloc_7[5, 1] : memref<6x6xf64>
    %128 = affine.load %alloc_6[5, 0] : memref<6x1xf64>
    %129 = affine.load %alloc[0, 2] : memref<1x6xf64>
    %130 = arith.mulf %128, %129 : f64
    %131 = arith.addf %130, %cst : f64
    affine.store %131, %alloc_7[5, 2] : memref<6x6xf64>
    %132 = affine.load %alloc_6[5, 0] : memref<6x1xf64>
    %133 = affine.load %alloc[0, 3] : memref<1x6xf64>
    %134 = arith.mulf %132, %133 : f64
    %135 = arith.addf %134, %cst : f64
    affine.store %135, %alloc_7[5, 3] : memref<6x6xf64>
    %136 = affine.load %alloc_6[5, 0] : memref<6x1xf64>
    %137 = affine.load %alloc[0, 4] : memref<1x6xf64>
    %138 = arith.mulf %136, %137 : f64
    %139 = arith.addf %138, %cst : f64
    affine.store %139, %alloc_7[5, 4] : memref<6x6xf64>
    %140 = affine.load %alloc_6[5, 0] : memref<6x1xf64>
    %141 = affine.load %alloc[0, 5] : memref<1x6xf64>
    %142 = arith.mulf %140, %141 : f64
    %143 = arith.addf %142, %cst : f64
    affine.store %143, %alloc_7[5, 5] : memref<6x6xf64>
    toy.print %alloc_7 : memref<6x6xf64>
    memref.dealloc %alloc_6 : memref<6x1xf64>
    memref.dealloc %alloc : memref<1x6xf64>
    return
  }
}
