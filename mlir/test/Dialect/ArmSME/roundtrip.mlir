// RUN: mlir-opt -split-input-file -verify-diagnostics %s | mlir-opt | FileCheck %s

func.func @arm_sme_cast_tile_to_vector_i8(%tile_id : i8) -> vector<[16]x[16]xi8> {
  // CHECK: arm_sme.cast_tile_to_vector {{.*}} : i8 to vector<[16]x[16]xi8>
  %0 = arm_sme.cast_tile_to_vector %tile_id : i8 to vector<[16]x[16]xi8>
  return %0 : vector<[16]x[16]xi8>
}

// -----

func.func @arm_sme_cast_tile_to_vector_i16(%tile_id : i16) -> vector<[8]x[8]xi16> {
  // CHECK: arm_sme.cast_tile_to_vector {{.*}} : i16 to vector<[8]x[8]xi16>
  %0 = arm_sme.cast_tile_to_vector %tile_id : i16 to vector<[8]x[8]xi16>
  return %0 : vector<[8]x[8]xi16>
}

// -----

func.func @arm_sme_cast_tile_to_vector_i32(%tile_id : i32) -> vector<[4]x[4]xi32> {
  // CHECK: arm_sme.cast_tile_to_vector {{.*}} : i32 to vector<[4]x[4]xi32>
  %0 = arm_sme.cast_tile_to_vector %tile_id : i32 to vector<[4]x[4]xi32>
  return %0 : vector<[4]x[4]xi32>
}

// -----

func.func @arm_sme_cast_tile_to_vector_i64(%tile_id : i64) -> vector<[2]x[2]xi64> {
  // CHECK: arm_sme.cast_tile_to_vector {{.*}} : i64 to vector<[2]x[2]xi64>
  %0 = arm_sme.cast_tile_to_vector %tile_id : i64 to vector<[2]x[2]xi64>
  return %0 : vector<[2]x[2]xi64>
}

// -----

func.func @arm_sme_cast_tile_to_vector_i128(%tile_id : i128) -> vector<[1]x[1]xi128> {
  // CHECK: arm_sme.cast_tile_to_vector {{.*}} : i128 to vector<[1]x[1]xi128>
  %0 = arm_sme.cast_tile_to_vector %tile_id : i128 to vector<[1]x[1]xi128>
  return %0 : vector<[1]x[1]xi128>
}

// -----

func.func @arm_sme_cast_tile_to_vector_f16(%tile_id : i16) -> vector<[8]x[8]xf16> {
  // CHECK: arm_sme.cast_tile_to_vector {{.*}} : i16 to vector<[8]x[8]xf16>
  %0 = arm_sme.cast_tile_to_vector %tile_id : i16 to vector<[8]x[8]xf16>
  return %0 : vector<[8]x[8]xf16>
}

// -----

func.func @arm_sme_cast_tile_to_vector_bf16(%tile_id : i16) -> vector<[8]x[8]xbf16> {
  // CHECK: arm_sme.cast_tile_to_vector {{.*}} : i16 to vector<[8]x[8]xbf16>
  %0 = arm_sme.cast_tile_to_vector %tile_id : i16 to vector<[8]x[8]xbf16>
  return %0 : vector<[8]x[8]xbf16>
}

// -----

func.func @arm_sme_cast_tile_to_vector_f32(%tile_id : i32) -> vector<[4]x[4]xf32> {
  // CHECK: arm_sme.cast_tile_to_vector {{.*}} : i32 to vector<[4]x[4]xf32>
  %0 = arm_sme.cast_tile_to_vector %tile_id : i32 to vector<[4]x[4]xf32>
  return %0 : vector<[4]x[4]xf32>
}

// -----

func.func @arm_sme_cast_tile_to_vector_f64(%tile_id : i64) -> vector<[2]x[2]xf64> {
  // CHECK: arm_sme.cast_tile_to_vector {{.*}} : i64 to vector<[2]x[2]xf64>
  %0 = arm_sme.cast_tile_to_vector %tile_id : i64 to vector<[2]x[2]xf64>
  return %0 : vector<[2]x[2]xf64>
}

// -----

func.func @arm_sme_cast_vector_to_tile_i8(%vector : vector<[16]x[16]xi8>) -> i8 {
  // CHECK: arm_sme.cast_vector_to_tile {{.*}} : vector<[16]x[16]xi8> to i8
  %0 = arm_sme.cast_vector_to_tile %vector : vector<[16]x[16]xi8> to i8
  return %0 : i8
}

// -----

func.func @arm_sme_cast_vector_to_tile_i16(%vector : vector<[8]x[8]xi16>) -> i16 {
  // CHECK: arm_sme.cast_vector_to_tile {{.*}} : vector<[8]x[8]xi16> to i16
  %0 = arm_sme.cast_vector_to_tile %vector : vector<[8]x[8]xi16> to i16
  return %0 : i16
}

// -----

func.func @arm_sme_cast_vector_to_tile_i32(%vector : vector<[4]x[4]xi32>) -> i32 {
  // CHECK: arm_sme.cast_vector_to_tile {{.*}} : vector<[4]x[4]xi32> to i32
  %0 = arm_sme.cast_vector_to_tile %vector : vector<[4]x[4]xi32> to i32
  return %0 : i32
}

// -----

func.func @arm_sme_cast_vector_to_tile_i64(%vector : vector<[2]x[2]xi64>) -> i64 {
  // CHECK: arm_sme.cast_vector_to_tile {{.*}} : vector<[2]x[2]xi64> to i64
  %0 = arm_sme.cast_vector_to_tile %vector : vector<[2]x[2]xi64> to i64
  return %0 : i64
}

// -----

func.func @arm_sme_cast_vector_to_tile_i128(%vector : vector<[1]x[1]xi128>) -> i128 {
  // CHECK: arm_sme.cast_vector_to_tile {{.*}} : vector<[1]x[1]xi128> to i128
  %0 = arm_sme.cast_vector_to_tile %vector : vector<[1]x[1]xi128> to i128
  return %0 : i128
}

// -----

func.func @arm_sme_cast_vector_to_tile_f16(%vector : vector<[8]x[8]xf16>) -> i16 {
  // CHECK: arm_sme.cast_vector_to_tile {{.*}} : vector<[8]x[8]xf16> to i16
  %0 = arm_sme.cast_vector_to_tile %vector : vector<[8]x[8]xf16> to i16
  return %0 : i16
}

// -----

func.func @arm_sme_cast_vector_to_tile_bf16(%vector : vector<[8]x[8]xbf16>) -> i16 {
  // CHECK: arm_sme.cast_vector_to_tile {{.*}} : vector<[8]x[8]xbf16> to i16
  %0 = arm_sme.cast_vector_to_tile %vector : vector<[8]x[8]xbf16> to i16
  return %0 : i16
}

// -----

func.func @arm_sme_cast_vector_to_tile_f32(%vector : vector<[4]x[4]xf32>) -> i32 {
  // CHECK: arm_sme.cast_vector_to_tile {{.*}} : vector<[4]x[4]xf32> to i32
  %0 = arm_sme.cast_vector_to_tile %vector : vector<[4]x[4]xf32> to i32
  return %0 : i32
}

// -----

func.func @arm_sme_cast_vector_to_tile_f64(%vector : vector<[2]x[2]xf64>) -> i64 {
  // CHECK: arm_sme.cast_vector_to_tile {{.*}} : vector<[2]x[2]xf64> to i64
  %0 = arm_sme.cast_vector_to_tile %vector : vector<[2]x[2]xf64> to i64
  return %0 : i64
}

// -----

func.func @arm_sme_get_tile_id_i8() -> i8 {
  // CHECK: arm_sme.get_tile_id : i8
  %0 = arm_sme.get_tile_id : i8
  return %0 : i8
}

// -----

func.func @arm_sme_get_tile_id_i16() -> i16 {
  // CHECK: arm_sme.get_tile_id : i16
  %0 = arm_sme.get_tile_id : i16
  return %0 : i16
}

// -----

func.func @arm_sme_get_tile_id_i32() -> i32 {
  // CHECK: arm_sme.get_tile_id : i32
  %0 = arm_sme.get_tile_id : i32
  return %0 : i32
}

// -----

func.func @arm_sme_get_tile_id_i64() -> i64 {
  // CHECK: arm_sme.get_tile_id : i64
  %0 = arm_sme.get_tile_id : i64
  return %0 : i64
}

// -----

func.func @arm_sme_get_tile_id_i128() -> i128 {
  // CHECK: arm_sme.get_tile_id : i128
  %0 = arm_sme.get_tile_id : i128
  return %0 : i128
}

// -----

func.func @arm_sme_zero() -> () {
  // CHECK: arm_sme.zero : vector<[16]x[16]xi8>
  %0 = arm_sme.zero : vector<[16]x[16]xi8>
  return
}

// -----

func.func @arm_sme_tile_load_i8(%src : memref<?x?xi8>) -> () {
  // CHECK: arm_sme.tile_load {{.*}} : memref<?x?xi8>, vector<[16]x[16]xi8>
  %c0 = arith.constant 0 : index
  %tile = arm_sme.tile_load %src[%c0, %c0] : memref<?x?xi8>, vector<[16]x[16]xi8>
  return
}

// -----

func.func @arm_sme_tile_load_i16(%src : memref<?x?xi16>) -> () {
  // CHECK: arm_sme.tile_load {{.*}} : memref<?x?xi16>, vector<[8]x[8]xi16>
  %c0 = arith.constant 0 : index
  %tile = arm_sme.tile_load %src[%c0, %c0] : memref<?x?xi16>, vector<[8]x[8]xi16>
  return
}

// -----

func.func @arm_sme_tile_load_i32(%src : memref<?x?xi32>) -> () {
  // CHECK: arm_sme.tile_load {{.*}} : memref<?x?xi32>, vector<[4]x[4]xi32>
  %c0 = arith.constant 0 : index
  %tile = arm_sme.tile_load %src[%c0, %c0] : memref<?x?xi32>, vector<[4]x[4]xi32>
  return
}

// -----

func.func @arm_sme_tile_load_i64(%src : memref<?x?xi64>) -> () {
  // CHECK: arm_sme.tile_load {{.*}} : memref<?x?xi64>, vector<[2]x[2]xi64>
  %c0 = arith.constant 0 : index
  %tile = arm_sme.tile_load %src[%c0, %c0] : memref<?x?xi64>, vector<[2]x[2]xi64>
  return
}

// -----

func.func @arm_sme_tile_load_i128(%src : memref<?x?xi128>) -> () {
  // CHECK: arm_sme.tile_load {{.*}} : memref<?x?xi128>, vector<[1]x[1]xi128>
  %c0 = arith.constant 0 : index
  %tile = arm_sme.tile_load %src[%c0, %c0] : memref<?x?xi128>, vector<[1]x[1]xi128>
  return
}

// -----

func.func @arm_sme_tile_load_f16(%src : memref<?x?xf16>) -> () {
  // CHECK: arm_sme.tile_load {{.*}} : memref<?x?xf16>, vector<[8]x[8]xf16>
  %c0 = arith.constant 0 : index
  %tile = arm_sme.tile_load %src[%c0, %c0] : memref<?x?xf16>, vector<[8]x[8]xf16>
  return
}

// -----

func.func @arm_sme_tile_load_bf16(%src : memref<?x?xbf16>) -> () {
  // CHECK: arm_sme.tile_load {{.*}} : memref<?x?xbf16>, vector<[8]x[8]xbf16>
  %c0 = arith.constant 0 : index
  %tile = arm_sme.tile_load %src[%c0, %c0] : memref<?x?xbf16>, vector<[8]x[8]xbf16>
  return
}

// -----

func.func @arm_sme_tile_load_f32(%src : memref<?x?xf32>) -> () {
  // CHECK: arm_sme.tile_load {{.*}} : memref<?x?xf32>, vector<[4]x[4]xf32>
  %c0 = arith.constant 0 : index
  %tile = arm_sme.tile_load %src[%c0, %c0] : memref<?x?xf32>, vector<[4]x[4]xf32>
  return
}

// -----

func.func @arm_sme_tile_load_f64(%src : memref<?x?xf64>) -> () {
  // CHECK: arm_sme.tile_load {{.*}} : memref<?x?xf64>, vector<[2]x[2]xf64>
  %c0 = arith.constant 0 : index
  %tile = arm_sme.tile_load %src[%c0, %c0] : memref<?x?xf64>, vector<[2]x[2]xf64>
  return
}

// -----

func.func @arm_sme_store_tile(%tile : vector<[16]x[16]xi8>, %dest : memref<?x?xi8>) -> () {
  // CHECK: arm_sme.tile_store {{.*}} : memref<?x?xi8>, vector<[16]x[16]xi8>
  %c0 = arith.constant 0 : index
  arm_sme.tile_store %tile, %dest[%c0, %c0] : memref<?x?xi8>, vector<[16]x[16]xi8>
  return
}
