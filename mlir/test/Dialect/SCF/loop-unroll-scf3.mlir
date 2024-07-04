module {
  func.func @main() -> f32 {
    %sum = arith.constant 0.0 : f32
    %val = arith.constant 2.0 : f32
    %N = arith.constant 4 : index
    %num = arith.constant 10 : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index

    %result = scf.for %i = %c0 to %N step %c1 iter_args(%iter_sum = %sum) -> (f32) {
      %new_sum = arith.addf %iter_sum, %val : f32
      %result2 = scf.for %j = %c0 to %num step %c1 iter_args(%iter_sum2 = %new_sum) -> (f32) {
        %new_sum2 = arith.addf %iter_sum2, %val : f32
        scf.yield %new_sum2 : f32
      }
      %new_sum3 = arith.addf %result2, %val : f32
      scf.yield %new_sum : f32
    }
    return %result : f32
  }
}

//CHECK-LABEL: func.func @main() -> f32 {
//CHECK-NEXT: %cst = arith.constant 0.000000e+00 : f32
//CHECK-NEXT: %cst_0 = arith.constant 2.000000e+00 : f32
//CHECK-NEXT: %c4 = arith.constant 4 : index
//CHECK-NEXT: %c10 = arith.constant 10 : index
//CHECK-NEXT: %c0 = arith.constant 0 : index
//CHECK-NEXT: %c1 = arith.constant 1 : index
//CHECK-NEXT: %0 = scf.for %arg0 = %c0 to %c4 step %c1 iter_args(%arg1 = %cst) -> (f32) {
//CHECK-NEXT:   %1 = arith.addf %arg1, %cst_0 : f32
//CHECK-NEXT:   %c8 = arith.constant 8 : index
//CHECK-NEXT:   %c4_1 = arith.constant 4 : index
//CHECK-NEXT:   %2 = scf.for %arg2 = %c0 to %c10 step %c4_1 iter_args(%arg3 = %1) -> (f32) {
//CHECK-NEXT:     %5 = arith.addf %arg3, %cst_0 : f32
//CHECK-NEXT:     %6 = arith.addf %5, %cst_0 : f32
//CHECK-NEXT:     %7 = arith.addf %6, %cst_0 : f32
//CHECK-NEXT:     %8 = arith.addf %7, %cst_0 : f32
//CHECK-NEXT:     scf.yield %8 : f32
//CHECK-NEXT:   }
//CHECK-NEXT:   %3 = scf.for %arg2 = %c8 to %c10 step %c1 iter_args(%arg3 = %2) -> (f32) {
//CHECK-NEXT:     %5 = arith.addf %arg3, %cst_0 : f32
//CHECK-NEXT:     scf.yield %5 : f32
//CHECK-NEXT:   }
//CHECK-NEXT:   %4 = arith.addf %3, %cst_0 : f32
//CHECK-NEXT:   scf.yield %1 : f32
//CHECK-NEXT: }
//CHECK-NEXT: return %0 : f32
//CHECK-NEXT: }
