// RUN: mlir-translate -mlir-to-llvmir %s | FileCheck %s

// CHECK-LABEL: define i64 @tcgen05_mma_smem_desc_test(i32 %0, i32 %1, i32 %2, i8 %3, i1 %4, i8 %5) {
llvm.func @tcgen05_mma_smem_desc_test(%startAddr: i32, %leadingDimOffset: i32, %strideDimOffset: i32,
                                      %baseOffset: i8, %leadingDimMode: i1, %swizzleMode: i8) -> i64 {
  // CHECK-NEXT: %7 = and i32 %0, 16383
  // CHECK-NEXT: %8 = zext i32 %7 to i64
  // CHECK-NEXT: %9 = shl i64 %8, 0
  // CHECK-NEXT: %10 = or i64 0, %9
  // CHECK-NEXT: %11 = and i32 %1, 16383
  // CHECK-NEXT: %12 = zext i32 %11 to i64
  // CHECK-NEXT: %13 = shl i64 %12, 16
  // CHECK-NEXT: %14 = or i64 %10, %13
  // CHECK-NEXT: %15 = and i32 %2, 16383
  // CHECK-NEXT: %16 = zext i32 %15 to i64
  // CHECK-NEXT: %17 = shl i64 %16, 32
  // CHECK-NEXT: %18 = or i64 %14, %17
  // CHECK-NEXT: %19 = or i64 %18, 70368744177664
  // CHECK-NEXT: %20 = zext i8 %3 to i32
  // CHECK-NEXT: %21 = and i32 %20, 7
  // CHECK-NEXT: %22 = zext i32 %21 to i64
  // CHECK-NEXT: %23 = shl i64 %22, 49
  // CHECK-NEXT: %24 = or i64 %19, %23
  // CHECK-NEXT: %25 = zext i1 %4 to i32
  // CHECK-NEXT: %26 = and i32 %25, 1
  // CHECK-NEXT: %27 = zext i32 %26 to i64
  // CHECK-NEXT: %28 = shl i64 %27, 52
  // CHECK-NEXT: %29 = or i64 %24, %28
  // CHECK-NEXT: %30 = zext i8 %5 to i32
  // CHECK-NEXT: %31 = and i32 %30, 7
  // CHECK-NEXT: %32 = zext i32 %31 to i64
  // CHECK-NEXT: %33 = shl i64 %32, 61
  // CHECK-NEXT: %34 = or i64 %29, %33
  // CHECK-NEXT: ret i64 %34
  // CHECK-NEXT: }
  %desc = nvvm.tcgen05.mma_smem_desc (%startAddr, %leadingDimOffset, %strideDimOffset, %baseOffset, %leadingDimMode, %swizzleMode) : (i32, i32, i32, i8, i1, i8) -> i64
  llvm.return %desc : i64
}
