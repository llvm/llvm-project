// RUN: mlir-opt -convert-openacc-to-llvm='use-opaque-pointers=1' -split-input-file %s | FileCheck %s

func.func @testparallelop(%a: memref<10xf32>, %b: memref<10xf32>) -> () {
  acc.parallel copy(%b : memref<10xf32>) copyout(%a : memref<10xf32>) {
  }
  return
}

// CHECK: acc.parallel copy(%{{.*}}: !llvm.struct<"openacc_data", (struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, ptr, i64)>) copyout(%{{.*}}: !llvm.struct<"openacc_data.1", (struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, ptr, i64)>)

// -----

func.func @testparallelop(%a: !llvm.ptr, %b: memref<10xf32>, %c: !llvm.ptr) -> () {
  acc.parallel copyin(%b : memref<10xf32>) deviceptr(%c: !llvm.ptr) attach(%a : !llvm.ptr) {
  }
  return
}

// CHECK: acc.parallel attach(%{{.*}}: !llvm.ptr) copyin(%{{.*}}: !llvm.struct<"openacc_data", (struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, ptr, i64)>) deviceptr(%{{.*}} : !llvm.ptr) 

// -----

func.func @testparallelop(%a: memref<10xf32>, %b: memref<10xf32>) -> () {
  %ifCond = arith.constant true
  acc.parallel if(%ifCond) copyin_readonly(%b : memref<10xf32>) copyout_zero(%a : memref<10xf32>) {
  }
  return
}

// CHECK: acc.parallel copyin_readonly(%{{.*}}: !llvm.struct<"openacc_data", (struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, ptr, i64)>) copyout_zero(%{{.*}}: !llvm.struct<"openacc_data.1", (struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, ptr, i64)>) if(%{{.*}})

// -----

func.func @testparallelop(%a: !llvm.ptr, %b: memref<10xf32>, %c: !llvm.ptr) -> () {
  acc.parallel create(%b : memref<10xf32>) create_zero(%c: !llvm.ptr) no_create(%a : !llvm.ptr) {
  }
  return
}

// CHECK: acc.parallel create(%{{.*}}: !llvm.struct<"openacc_data", (struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, ptr, i64)>) create_zero(%{{.*}}: !llvm.ptr) no_create(%{{.*}}: !llvm.ptr)

// -----

func.func @testparallelop(%a: memref<10xf32>, %b: memref<10xf32>) -> () {
  acc.parallel present(%a, %b : memref<10xf32>, memref<10xf32>) {
  }
  return
}

// CHECK: acc.parallel present(%{{.*}}, %{{.*}} : !llvm.struct<"openacc_data", (struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, ptr, i64)>, !llvm.struct<"openacc_data.1", (struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, ptr, i64)>)

// -----

func.func @testparallelop(%i: i64, %a: memref<10xf32>, %b: memref<10xf32>) -> () {
  acc.parallel num_gangs(%i: i64) present(%a, %b : memref<10xf32>, memref<10xf32>) {
    %0 = arith.constant 0 : i32
    acc.yield
  } attributes {async}
  return
}

// CHECK: acc.parallel num_gangs(%{{.*}}: i64) present(%{{.*}}, %{{.*}} : !llvm.struct<"openacc_data", (struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, ptr, i64)>, !llvm.struct<"openacc_data.1", (struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, ptr, i64)>)
// CHECK:   %c0_i32 = arith.constant 0 : i32
// CHECK:   acc.yield
// CHECK: } attributes {async}
