// RUN: mlir-opt -convert-openacc-to-llvm -split-input-file %s | FileCheck %s

func @testenterdataop(%a: memref<10xf32>, %b: memref<10xf32>) -> () {
  acc.enter_data copyin(%b : memref<10xf32>) create(%a : memref<10xf32>)
  return
}

// CHECK: acc.enter_data copyin(%{{.*}} : !llvm.struct<"openacc_data", (struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>, ptr<f32>, i64)>) create(%{{.*}} : !llvm.struct<"openacc_data.1", (struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>, ptr<f32>, i64)>)

// -----

func @testenterdataop(%a: !llvm.ptr<f32>, %b: memref<10xf32>) -> () {
  acc.enter_data copyin(%b : memref<10xf32>) create(%a : !llvm.ptr<f32>)
  return
}

// CHECK: acc.enter_data copyin(%{{.*}} : !llvm.struct<"openacc_data", (struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>, ptr<f32>, i64)>) create(%{{.*}} : !llvm.ptr<f32>)

// -----

func @testenterdataop(%a: memref<10xi64>, %b: memref<10xf32>) -> () {
  acc.enter_data copyin(%b : memref<10xf32>) create_zero(%a : memref<10xi64>) attributes {async}
  return
}

// CHECK: acc.enter_data copyin(%{{.*}} : !llvm.struct<"openacc_data", (struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>, ptr<f32>, i64)>) create_zero(%{{.*}} : !llvm.struct<"openacc_data.1", (struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>, ptr<i64>, i64)>) attributes {async}

// -----

func @testenterdataop(%a: memref<10xf32>, %b: memref<10xf32>) -> () {
  %ifCond = constant true
  acc.enter_data if(%ifCond) copyin(%b : memref<10xf32>) create(%a : memref<10xf32>)
  return
}

// CHECK: acc.enter_data if(%{{.*}}) copyin(%{{.*}} : !llvm.struct<"openacc_data", (struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>, ptr<f32>, i64)>) create(%{{.*}} : !llvm.struct<"openacc_data.1", (struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>, ptr<f32>, i64)>)

// -----

func @testexitdataop(%a: memref<10xf32>, %b: memref<10xf32>) -> () {
  acc.exit_data copyout(%b : memref<10xf32>) delete(%a : memref<10xf32>)
  return
}

// CHECK: acc.exit_data copyout(%{{.*}} : !llvm.struct<"openacc_data", (struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>, ptr<f32>, i64)>) delete(%{{.*}} : !llvm.struct<"openacc_data.1", (struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>, ptr<f32>, i64)>)

// -----

func @testexitdataop(%a: !llvm.ptr<f32>, %b: memref<10xf32>) -> () {
  acc.exit_data copyout(%b : memref<10xf32>) delete(%a : !llvm.ptr<f32>)
  return
}

// CHECK: acc.exit_data copyout(%{{.*}} : !llvm.struct<"openacc_data", (struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>, ptr<f32>, i64)>) delete(%{{.*}} : !llvm.ptr<f32>)

// -----

func @testexitdataop(%a: memref<10xi64>, %b: memref<10xf32>) -> () {
  acc.exit_data copyout(%b : memref<10xf32>) delete(%a : memref<10xi64>) attributes {async}
  return
}

// CHECK: acc.exit_data copyout(%{{.*}} : !llvm.struct<"openacc_data", (struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>, ptr<f32>, i64)>) delete(%{{.*}} : !llvm.struct<"openacc_data.1", (struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>, ptr<i64>, i64)>) attributes {async}

// -----

func @testexitdataop(%a: memref<10xf32>, %b: memref<10xf32>) -> () {
  %ifCond = constant true
  acc.exit_data if(%ifCond) copyout(%b : memref<10xf32>) delete(%a : memref<10xf32>)
  return
}

// CHECK: acc.exit_data if(%{{.*}}) copyout(%{{.*}} : !llvm.struct<"openacc_data", (struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>, ptr<f32>, i64)>) delete(%{{.*}} : !llvm.struct<"openacc_data.1", (struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>, ptr<f32>, i64)>)

// -----

func @testupdateop(%a: memref<10xf32>, %b: memref<10xf32>) -> () {
  acc.update host(%b : memref<10xf32>) device(%a : memref<10xf32>)
  return
}

// CHECK: acc.update host(%{{.*}} : !llvm.struct<"openacc_data", (struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>, ptr<f32>, i64)>) device(%{{.*}} : !llvm.struct<"openacc_data.1", (struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>, ptr<f32>, i64)>)

// -----

func @testupdateop(%a: !llvm.ptr<f32>, %b: memref<10xf32>) -> () {
  acc.update host(%b : memref<10xf32>) device(%a : !llvm.ptr<f32>)
  return
}

// CHECK: acc.update host(%{{.*}} : !llvm.struct<"openacc_data", (struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>, ptr<f32>, i64)>) device(%{{.*}} : !llvm.ptr<f32>)

// -----

func @testupdateop(%a: memref<10xi64>, %b: memref<10xf32>) -> () {
  acc.update host(%b : memref<10xf32>) device(%a : memref<10xi64>) attributes {async}
  return
}

// CHECK: acc.update host(%{{.*}} : !llvm.struct<"openacc_data", (struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>, ptr<f32>, i64)>) device(%{{.*}} : !llvm.struct<"openacc_data.1", (struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>, ptr<i64>, i64)>) attributes {async}

// -----

func @testupdateop(%a: memref<10xf32>, %b: memref<10xf32>) -> () {
  %ifCond = constant true
  acc.update if(%ifCond) host(%b : memref<10xf32>) device(%a : memref<10xf32>)
  return
}

// CHECK: acc.update if(%{{.*}}) host(%{{.*}} : !llvm.struct<"openacc_data", (struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>, ptr<f32>, i64)>) device(%{{.*}} : !llvm.struct<"openacc_data.1", (struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>, ptr<f32>, i64)>)
