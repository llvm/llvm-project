// RUN: mlir-opt -split-input-file -verify-diagnostics %s | mlir-opt | FileCheck %s

//===----------------------------------------------------------------------===//
// arm_sme.get_tile
//===----------------------------------------------------------------------===//


func.func @arm_sme_get_tile_i8() {
  // CHECK: arm_sme.get_tile : vector<[16]x[16]xi8>
  %0 = arm_sme.get_tile : vector<[16]x[16]xi8>
  return
}

// -----

func.func @arm_sme_get_tile_i16() {
  // CHECK: arm_sme.get_tile : vector<[8]x[8]xi16>
  %0 = arm_sme.get_tile : vector<[8]x[8]xi16>
  return
}

// -----

func.func @arm_sme_get_tile_i32() {
  // CHECK: arm_sme.get_tile : vector<[4]x[4]xi32>
  %0 = arm_sme.get_tile : vector<[4]x[4]xi32>
  return
}

// -----

func.func @arm_sme_get_tile_i64() {
  // CHECK: arm_sme.get_tile : vector<[2]x[2]xi64>
  %0 = arm_sme.get_tile : vector<[2]x[2]xi64>
  return
}

// -----

func.func @arm_sme_get_tile_i128() {
  // CHECK: arm_sme.get_tile : vector<[1]x[1]xi128>
  %0 = arm_sme.get_tile : vector<[1]x[1]xi128>
  return
}

// -----

func.func @arm_sme_get_tile_f16() {
  // CHECK: arm_sme.get_tile : vector<[8]x[8]xf16>
  %0 = arm_sme.get_tile : vector<[8]x[8]xf16>
  return
}

// -----

func.func @arm_sme_get_tile_bf16() {
  // CHECK: arm_sme.get_tile : vector<[8]x[8]xbf16>
  %0 = arm_sme.get_tile : vector<[8]x[8]xbf16>
  return
}

// -----

func.func @arm_sme_get_tile_f32() {
  // CHECK: arm_sme.get_tile : vector<[4]x[4]xf32>
  %0 = arm_sme.get_tile : vector<[4]x[4]xf32>
  return
}

// -----

func.func @arm_sme_get_tile_f64() {
  // CHECK: arm_sme.get_tile : vector<[2]x[2]xf64>
  %0 = arm_sme.get_tile : vector<[2]x[2]xf64>
  return
}

//===----------------------------------------------------------------------===//
// arm_sme.zero
//===----------------------------------------------------------------------===//

// -----

func.func @arm_sme_zero_i8() {
  // CHECK: arm_sme.zero : vector<[16]x[16]xi8>
  %0 = arm_sme.zero : vector<[16]x[16]xi8>
  return
}

// -----

func.func @arm_sme_zero_i16() {
  // CHECK: arm_sme.zero : vector<[8]x[8]xi16>
  %0 = arm_sme.zero : vector<[8]x[8]xi16>
  return
}

// -----

func.func @arm_sme_zero_i32() {
  // CHECK: arm_sme.zero : vector<[4]x[4]xi32>
  %0 = arm_sme.zero : vector<[4]x[4]xi32>
  return
}

// -----

func.func @arm_sme_zero_i64() {
  // CHECK: arm_sme.zero : vector<[2]x[2]xi64>
  %0 = arm_sme.zero : vector<[2]x[2]xi64>
  return
}

// -----

func.func @arm_sme_zero_i128() {
  // CHECK: arm_sme.zero : vector<[1]x[1]xi128>
  %0 = arm_sme.zero : vector<[1]x[1]xi128>
  return
}

// -----

func.func @arm_sme_zero_f16() {
  // CHECK: arm_sme.zero : vector<[8]x[8]xf16>
  %0 = arm_sme.zero : vector<[8]x[8]xf16>
  return
}

// -----

func.func @arm_sme_zero_bf16() {
  // CHECK: arm_sme.zero : vector<[8]x[8]xbf16>
  %0 = arm_sme.zero : vector<[8]x[8]xbf16>
  return
}

// -----

func.func @arm_sme_zero_f32() {
  // CHECK: arm_sme.zero : vector<[4]x[4]xf32>
  %0 = arm_sme.zero : vector<[4]x[4]xf32>
  return
}

// -----

func.func @arm_sme_zero_f64() {
  // CHECK: arm_sme.zero : vector<[2]x[2]xf64>
  %0 = arm_sme.zero : vector<[2]x[2]xf64>
  return
}

//===----------------------------------------------------------------------===//
// arm_sme.tile_load
//===----------------------------------------------------------------------===//

// -----

func.func @arm_sme_tile_load_hor_i8(%src : memref<?x?xi8>) {
  // CHECK: arm_sme.tile_load %{{.*}}[{{.*}}] : memref<?x?xi8>, vector<[16]x[16]xi8>
  %c0 = arith.constant 0 : index
  %tile = arm_sme.tile_load %src[%c0, %c0] : memref<?x?xi8>, vector<[16]x[16]xi8>
  return
}

// -----

func.func @arm_sme_tile_load_hor_i16(%src : memref<?x?xi16>) {
  // CHECK: arm_sme.tile_load %{{.*}}[{{.*}}] : memref<?x?xi16>, vector<[8]x[8]xi16>
  %c0 = arith.constant 0 : index
  %tile = arm_sme.tile_load %src[%c0, %c0] : memref<?x?xi16>, vector<[8]x[8]xi16>
  return
}

// -----

func.func @arm_sme_tile_load_hor_i32(%src : memref<?x?xi32>) {
  // CHECK: arm_sme.tile_load %{{.*}}[{{.*}}] : memref<?x?xi32>, vector<[4]x[4]xi32>
  %c0 = arith.constant 0 : index
  %tile = arm_sme.tile_load %src[%c0, %c0] : memref<?x?xi32>, vector<[4]x[4]xi32>
  return
}

// -----

func.func @arm_sme_tile_load_hor_i64(%src : memref<?x?xi64>) {
  // CHECK: arm_sme.tile_load %{{.*}}[{{.*}}] : memref<?x?xi64>, vector<[2]x[2]xi64>
  %c0 = arith.constant 0 : index
  %tile = arm_sme.tile_load %src[%c0, %c0] : memref<?x?xi64>, vector<[2]x[2]xi64>
  return
}

// -----

func.func @arm_sme_tile_load_hor_i128(%src : memref<?x?xi128>) {
  // CHECK: arm_sme.tile_load %{{.*}}[{{.*}}] : memref<?x?xi128>, vector<[1]x[1]xi128>
  %c0 = arith.constant 0 : index
  %tile = arm_sme.tile_load %src[%c0, %c0] : memref<?x?xi128>, vector<[1]x[1]xi128>
  return
}

// -----

func.func @arm_sme_tile_load_hor_f16(%src : memref<?x?xf16>) {
  // CHECK: arm_sme.tile_load %{{.*}}[{{.*}}] : memref<?x?xf16>, vector<[8]x[8]xf16>
  %c0 = arith.constant 0 : index
  %tile = arm_sme.tile_load %src[%c0, %c0] : memref<?x?xf16>, vector<[8]x[8]xf16>
  return
}

// -----

func.func @arm_sme_tile_load_hor_bf16(%src : memref<?x?xbf16>) {
  // CHECK: arm_sme.tile_load %{{.*}}[{{.*}}] : memref<?x?xbf16>, vector<[8]x[8]xbf16>
  %c0 = arith.constant 0 : index
  %tile = arm_sme.tile_load %src[%c0, %c0] : memref<?x?xbf16>, vector<[8]x[8]xbf16>
  return
}

// -----

func.func @arm_sme_tile_load_hor_f32(%src : memref<?x?xf32>) {
  // CHECK: arm_sme.tile_load %{{.*}}[{{.*}}] : memref<?x?xf32>, vector<[4]x[4]xf32>
  %c0 = arith.constant 0 : index
  %tile = arm_sme.tile_load %src[%c0, %c0] : memref<?x?xf32>, vector<[4]x[4]xf32>
  return
}

// -----

func.func @arm_sme_tile_load_hor_f64(%src : memref<?x?xf64>) {
  // CHECK: arm_sme.tile_load %{{.*}}[{{.*}}] : memref<?x?xf64>, vector<[2]x[2]xf64>
  %c0 = arith.constant 0 : index
  %tile = arm_sme.tile_load %src[%c0, %c0] : memref<?x?xf64>, vector<[2]x[2]xf64>
  return
}

// -----

func.func @arm_sme_tile_load_ver_i8(%src : memref<?x?xi8>) {
  // CHECK: arm_sme.tile_load {{.*}} layout<vertical> : memref<?x?xi8>, vector<[16]x[16]xi8>
  %c0 = arith.constant 0 : index
  %tile = arm_sme.tile_load %src[%c0, %c0] layout<vertical> : memref<?x?xi8>, vector<[16]x[16]xi8>
  return
}

// -----

func.func @arm_sme_tile_load_ver_i16(%src : memref<?x?xi16>) {
  // CHECK: arm_sme.tile_load {{.*}} layout<vertical> : memref<?x?xi16>, vector<[8]x[8]xi16>
  %c0 = arith.constant 0 : index
  %tile = arm_sme.tile_load %src[%c0, %c0] layout<vertical> : memref<?x?xi16>, vector<[8]x[8]xi16>
  return
}

// -----

func.func @arm_sme_tile_load_ver_i32(%src : memref<?x?xi32>) {
  // CHECK: arm_sme.tile_load {{.*}} layout<vertical> : memref<?x?xi32>, vector<[4]x[4]xi32>
  %c0 = arith.constant 0 : index
  %tile = arm_sme.tile_load %src[%c0, %c0] layout<vertical> : memref<?x?xi32>, vector<[4]x[4]xi32>
  return
}

// -----

func.func @arm_sme_tile_load_ver_i64(%src : memref<?x?xi64>) {
  // CHECK: arm_sme.tile_load {{.*}} layout<vertical> : memref<?x?xi64>, vector<[2]x[2]xi64>
  %c0 = arith.constant 0 : index
  %tile = arm_sme.tile_load %src[%c0, %c0] layout<vertical> : memref<?x?xi64>, vector<[2]x[2]xi64>
  return
}

// -----

func.func @arm_sme_tile_load_ver_i128(%src : memref<?x?xi128>) {
  // CHECK: arm_sme.tile_load {{.*}} layout<vertical> : memref<?x?xi128>, vector<[1]x[1]xi128>
  %c0 = arith.constant 0 : index
  %tile = arm_sme.tile_load %src[%c0, %c0] layout<vertical> : memref<?x?xi128>, vector<[1]x[1]xi128>
  return
}

// -----

func.func @arm_sme_tile_load_ver_f16(%src : memref<?x?xf16>) {
  // CHECK: arm_sme.tile_load {{.*}} layout<vertical> : memref<?x?xf16>, vector<[8]x[8]xf16>
  %c0 = arith.constant 0 : index
  %tile = arm_sme.tile_load %src[%c0, %c0] layout<vertical> : memref<?x?xf16>, vector<[8]x[8]xf16>
  return
}

// -----

func.func @arm_sme_tile_load_ver_bf16(%src : memref<?x?xbf16>) {
  // CHECK: arm_sme.tile_load {{.*}} layout<vertical> : memref<?x?xbf16>, vector<[8]x[8]xbf16>
  %c0 = arith.constant 0 : index
  %tile = arm_sme.tile_load %src[%c0, %c0] layout<vertical> : memref<?x?xbf16>, vector<[8]x[8]xbf16>
  return
}

// -----

func.func @arm_sme_tile_load_ver_f32(%src : memref<?x?xf32>) {
  // CHECK: arm_sme.tile_load {{.*}} layout<vertical> : memref<?x?xf32>, vector<[4]x[4]xf32>
  %c0 = arith.constant 0 : index
  %tile = arm_sme.tile_load %src[%c0, %c0] layout<vertical> : memref<?x?xf32>, vector<[4]x[4]xf32>
  return
}

// -----

func.func @arm_sme_tile_load_ver_f64(%src : memref<?x?xf64>) {
  // CHECK: arm_sme.tile_load {{.*}} layout<vertical> : memref<?x?xf64>, vector<[2]x[2]xf64>
  %c0 = arith.constant 0 : index
  %tile = arm_sme.tile_load %src[%c0, %c0] layout<vertical> : memref<?x?xf64>, vector<[2]x[2]xf64>
  return
}

// -----

/// Padding and mask are optional
func.func @arm_sme_tile_load_hor_pad_f64(%src : memref<?x?xf64>, %pad : f64, %mask : vector<[2]x[2]xi1>) {
  // CHECK: arm_sme.tile_load %{{.*}}[{{.*}}], {{.*}}, {{.*}} : memref<?x?xf64>, vector<[2]x[2]xf64>
  %c0 = arith.constant 0 : index
  %tile = arm_sme.tile_load %src[%c0, %c0], %pad, %mask : memref<?x?xf64>, vector<[2]x[2]xf64>
  return
}

// -----

/// Layout is optional and horizontal is the default, verify it's still parsed.
func.func @arm_sme_tile_load_explicit_hor(%src : memref<?x?xi8>) {
  // CHECK: arm_sme.tile_load %{{.*}}[{{.*}}] : memref<?x?xi8>, vector<[16]x[16]xi8>
  %c0 = arith.constant 0 : index
  %tile = arm_sme.tile_load %src[%c0, %c0] layout<horizontal> : memref<?x?xi8>, vector<[16]x[16]xi8>
  return
}

//===----------------------------------------------------------------------===//
// arm_sme.tile_store
//===----------------------------------------------------------------------===//

// -----

func.func @arm_sme_tile_store_hor_i8(%tile : vector<[16]x[16]xi8>, %dest : memref<?x?xi8>) {
  // CHECK: arm_sme.tile_store %{{.*}}[{{.*}}] : memref<?x?xi8>, vector<[16]x[16]xi8>
  %c0 = arith.constant 0 : index
  arm_sme.tile_store %tile, %dest[%c0, %c0] : memref<?x?xi8>, vector<[16]x[16]xi8>
  return
}

// -----

func.func @arm_sme_tile_store_hor_i16(%tile : vector<[8]x[8]xi16>, %dest : memref<?x?xi16>) {
  // CHECK: arm_sme.tile_store %{{.*}}[{{.*}}] : memref<?x?xi16>, vector<[8]x[8]xi16>
  %c0 = arith.constant 0 : index
  arm_sme.tile_store %tile, %dest[%c0, %c0] : memref<?x?xi16>, vector<[8]x[8]xi16>
  return
}

// -----

func.func @arm_sme_tile_store_hor_i32(%tile : vector<[4]x[4]xi32>, %dest : memref<?x?xi32>) {
  // CHECK: arm_sme.tile_store %{{.*}}[{{.*}}] : memref<?x?xi32>, vector<[4]x[4]xi32>
  %c0 = arith.constant 0 : index
  arm_sme.tile_store %tile, %dest[%c0, %c0] : memref<?x?xi32>, vector<[4]x[4]xi32>
  return
}

// -----

func.func @arm_sme_tile_store_hor_i64(%tile : vector<[2]x[2]xi64>, %dest : memref<?x?xi64>) {
  // CHECK: arm_sme.tile_store %{{.*}}[{{.*}}] : memref<?x?xi64>, vector<[2]x[2]xi64>
  %c0 = arith.constant 0 : index
  arm_sme.tile_store %tile, %dest[%c0, %c0] : memref<?x?xi64>, vector<[2]x[2]xi64>
  return
}

// -----

func.func @arm_sme_tile_store_hor_i128(%tile : vector<[1]x[1]xi128>, %dest : memref<?x?xi128>) {
  // CHECK: arm_sme.tile_store %{{.*}}[{{.*}}] : memref<?x?xi128>, vector<[1]x[1]xi128>
  %c0 = arith.constant 0 : index
  arm_sme.tile_store %tile, %dest[%c0, %c0] : memref<?x?xi128>, vector<[1]x[1]xi128>
  return
}

// -----

func.func @arm_sme_tile_store_hor_f16(%tile : vector<[8]x[8]xf16>, %dest : memref<?x?xf16>) {
  // CHECK: arm_sme.tile_store %{{.*}}[{{.*}}] : memref<?x?xf16>, vector<[8]x[8]xf16>
  %c0 = arith.constant 0 : index
  arm_sme.tile_store %tile, %dest[%c0, %c0] : memref<?x?xf16>, vector<[8]x[8]xf16>
  return
}

// -----

func.func @arm_sme_tile_store_hor_bf16(%tile : vector<[8]x[8]xbf16>, %dest : memref<?x?xbf16>) {
  // CHECK: arm_sme.tile_store %{{.*}}[{{.*}}] : memref<?x?xbf16>, vector<[8]x[8]xbf16>
  %c0 = arith.constant 0 : index
  arm_sme.tile_store %tile, %dest[%c0, %c0] : memref<?x?xbf16>, vector<[8]x[8]xbf16>
  return
}

// -----

func.func @arm_sme_tile_store_hor_f32(%tile : vector<[4]x[4]xf32>, %dest : memref<?x?xf32>) {
  // CHECK: arm_sme.tile_store %{{.*}}[{{.*}}] : memref<?x?xf32>, vector<[4]x[4]xf32>
  %c0 = arith.constant 0 : index
  arm_sme.tile_store %tile, %dest[%c0, %c0] : memref<?x?xf32>, vector<[4]x[4]xf32>
  return
}

// -----

func.func @arm_sme_tile_store_hor_f64(%tile : vector<[2]x[2]xf64>, %dest : memref<?x?xf64>) {
  // CHECK: arm_sme.tile_store %{{.*}}[{{.*}}] : memref<?x?xf64>, vector<[2]x[2]xf64>
  %c0 = arith.constant 0 : index
  arm_sme.tile_store %tile, %dest[%c0, %c0] : memref<?x?xf64>, vector<[2]x[2]xf64>
  return
}

// -----

func.func @arm_sme_tile_store_ver_i8(%tile : vector<[16]x[16]xi8>, %dest : memref<?x?xi8>) {
  // CHECK: arm_sme.tile_store {{.*}} layout<vertical> : memref<?x?xi8>, vector<[16]x[16]xi8>
  %c0 = arith.constant 0 : index
  arm_sme.tile_store %tile, %dest[%c0, %c0] layout<vertical> : memref<?x?xi8>, vector<[16]x[16]xi8>
  return
}

// -----

func.func @arm_sme_tile_store_ver_i16(%tile : vector<[8]x[8]xi16>, %dest : memref<?x?xi16>) {
  // CHECK: arm_sme.tile_store {{.*}} layout<vertical> : memref<?x?xi16>, vector<[8]x[8]xi16>
  %c0 = arith.constant 0 : index
  arm_sme.tile_store %tile, %dest[%c0, %c0] layout<vertical> : memref<?x?xi16>, vector<[8]x[8]xi16>
  return
}

// -----

func.func @arm_sme_tile_store_ver_i32(%tile : vector<[4]x[4]xi32>, %dest : memref<?x?xi32>) {
  // CHECK: arm_sme.tile_store {{.*}} layout<vertical> : memref<?x?xi32>, vector<[4]x[4]xi32>
  %c0 = arith.constant 0 : index
  arm_sme.tile_store %tile, %dest[%c0, %c0] layout<vertical> : memref<?x?xi32>, vector<[4]x[4]xi32>
  return
}

// -----

func.func @arm_sme_tile_store_ver_i64(%tile : vector<[2]x[2]xi64>, %dest : memref<?x?xi64>) {
  // CHECK: arm_sme.tile_store {{.*}} layout<vertical> : memref<?x?xi64>, vector<[2]x[2]xi64>
  %c0 = arith.constant 0 : index
  arm_sme.tile_store %tile, %dest[%c0, %c0] layout<vertical> : memref<?x?xi64>, vector<[2]x[2]xi64>
  return
}

// -----

func.func @arm_sme_tile_store_ver_i128(%tile : vector<[1]x[1]xi128>, %dest : memref<?x?xi128>) {
  // CHECK: arm_sme.tile_store {{.*}} layout<vertical> : memref<?x?xi128>, vector<[1]x[1]xi128>
  %c0 = arith.constant 0 : index
  arm_sme.tile_store %tile, %dest[%c0, %c0] layout<vertical> : memref<?x?xi128>, vector<[1]x[1]xi128>
  return
}

// -----

func.func @arm_sme_tile_store_ver_f16(%tile : vector<[8]x[8]xf16>, %dest : memref<?x?xf16>) {
  // CHECK: arm_sme.tile_store {{.*}} layout<vertical> : memref<?x?xf16>, vector<[8]x[8]xf16>
  %c0 = arith.constant 0 : index
  arm_sme.tile_store %tile, %dest[%c0, %c0] layout<vertical> : memref<?x?xf16>, vector<[8]x[8]xf16>
  return
}

// -----

func.func @arm_sme_tile_store_ver_bf16(%tile : vector<[8]x[8]xbf16>, %dest : memref<?x?xbf16>) {
  // CHECK: arm_sme.tile_store {{.*}} layout<vertical> : memref<?x?xbf16>, vector<[8]x[8]xbf16>
  %c0 = arith.constant 0 : index
  arm_sme.tile_store %tile, %dest[%c0, %c0] layout<vertical> : memref<?x?xbf16>, vector<[8]x[8]xbf16>
  return
}

// -----

func.func @arm_sme_tile_store_ver_f32(%tile : vector<[4]x[4]xf32>, %dest : memref<?x?xf32>) {
  // CHECK: arm_sme.tile_store {{.*}} layout<vertical> : memref<?x?xf32>, vector<[4]x[4]xf32>
  %c0 = arith.constant 0 : index
  arm_sme.tile_store %tile, %dest[%c0, %c0] layout<vertical> : memref<?x?xf32>, vector<[4]x[4]xf32>
  return
}

// -----

func.func @arm_sme_tile_store_ver_f64(%tile : vector<[2]x[2]xf64>, %dest : memref<?x?xf64>) {
  // CHECK: arm_sme.tile_store {{.*}} layout<vertical> : memref<?x?xf64>, vector<[2]x[2]xf64>
  %c0 = arith.constant 0 : index
  arm_sme.tile_store %tile, %dest[%c0, %c0] layout<vertical> : memref<?x?xf64>, vector<[2]x[2]xf64>
  return
}

// -----

func.func @arm_sme_tile_store_with_mask_ver_f32(%tile : vector<[4]x[4]xf32>, %dest : memref<?x?xf32>, %mask : vector<[4]x[4]xi1>) {
  // CHECK: arm_sme.tile_store {{.*}} layout<vertical> : memref<?x?xf32>, vector<[4]x[4]xf32>
  %c0 = arith.constant 0 : index
  arm_sme.tile_store %tile, %dest[%c0, %c0], %mask layout<vertical> : memref<?x?xf32>, vector<[4]x[4]xf32>
  return
}

// -----

/// Layout is optional and horizontal is the default, verify it's still parsed.
func.func @arm_sme_tile_store_ver_i8(%tile : vector<[16]x[16]xi8>, %dest : memref<?x?xi8>) {
  // CHECK: arm_sme.tile_store %{{.*}}[{{.*}}] : memref<?x?xi8>, vector<[16]x[16]xi8>
  %c0 = arith.constant 0 : index
  arm_sme.tile_store %tile, %dest[%c0, %c0] layout<horizontal> : memref<?x?xi8>, vector<[16]x[16]xi8>
  return
}

//===----------------------------------------------------------------------===//
// arm_sme.load_tile_slice
//===----------------------------------------------------------------------===//

// -----

func.func @arm_sme_load_tile_slice_hor_i8(%src : memref<?x?xi8>, %mask : vector<[16]xi1>, %tile : vector<[16]x[16]xi8>, %tile_slice_index : index) {
  // CHECK: arm_sme.load_tile_slice %{{.*}}[{{.*}}], %{{.*}}, %{{.*}} : memref<?x?xi8>, vector<[16]xi1>, vector<[16]x[16]xi8>
  %c0 = arith.constant 0 : index
  %tile_update = arm_sme.load_tile_slice %src[%c0], %mask, %tile, %tile_slice_index : memref<?x?xi8>, vector<[16]xi1>, vector<[16]x[16]xi8>
  return
}

// -----

func.func @arm_sme_load_tile_slice_hor_i16(%src : memref<?x?xi16>, %mask : vector<[8]xi1>, %tile : vector<[8]x[8]xi16>, %tile_slice_index : index) {
  // CHECK: arm_sme.load_tile_slice %{{.*}}[{{.*}}], %{{.*}}, %{{.*}} : memref<?x?xi16>, vector<[8]xi1>, vector<[8]x[8]xi16>
  %c0 = arith.constant 0 : index
  %tile_update = arm_sme.load_tile_slice %src[%c0], %mask, %tile, %tile_slice_index : memref<?x?xi16>, vector<[8]xi1>, vector<[8]x[8]xi16>
  return
}

// -----

func.func @arm_sme_load_tile_slice_hor_i32(%src : memref<?x?xi32>, %mask : vector<[4]xi1>, %tile : vector<[4]x[4]xi32>, %tile_slice_index : index) {
  // CHECK: arm_sme.load_tile_slice %{{.*}}[{{.*}}], %{{.*}}, %{{.*}} : memref<?x?xi32>, vector<[4]xi1>, vector<[4]x[4]xi32>
  %c0 = arith.constant 0 : index
  %tile_update = arm_sme.load_tile_slice %src[%c0], %mask, %tile, %tile_slice_index : memref<?x?xi32>, vector<[4]xi1>, vector<[4]x[4]xi32>
  return
}

// -----

func.func @arm_sme_load_tile_slice_hor_i64(%src : memref<?x?xi64>, %mask : vector<[2]xi1>, %tile : vector<[2]x[2]xi64>, %tile_slice_index : index) {
  // CHECK: arm_sme.load_tile_slice %{{.*}}[{{.*}}], %{{.*}}, %{{.*}} : memref<?x?xi64>, vector<[2]xi1>, vector<[2]x[2]xi64>
  %c0 = arith.constant 0 : index
  %tile_update = arm_sme.load_tile_slice %src[%c0], %mask, %tile, %tile_slice_index : memref<?x?xi64>, vector<[2]xi1>, vector<[2]x[2]xi64>
  return
}

// -----

func.func @arm_sme_load_tile_slice_hor_i128(%src : memref<?x?xi128>, %mask : vector<[1]xi1>, %tile : vector<[1]x[1]xi128>, %tile_slice_index : index) {
  // CHECK: arm_sme.load_tile_slice %{{.*}}[{{.*}}], %{{.*}}, %{{.*}} : memref<?x?xi128>, vector<[1]xi1>, vector<[1]x[1]xi128>
  %c0 = arith.constant 0 : index
  %tile_update = arm_sme.load_tile_slice %src[%c0], %mask, %tile, %tile_slice_index : memref<?x?xi128>, vector<[1]xi1>, vector<[1]x[1]xi128>
  return
}

// -----

func.func @arm_sme_load_tile_slice_hor_f16(%src : memref<?x?xf16>, %mask : vector<[8]xi1>, %tile : vector<[8]x[8]xf16>, %tile_slice_index : index) {
  // CHECK: arm_sme.load_tile_slice %{{.*}}[{{.*}}], %{{.*}}, %{{.*}} : memref<?x?xf16>, vector<[8]xi1>, vector<[8]x[8]xf16>
  %c0 = arith.constant 0 : index
  %tile_update = arm_sme.load_tile_slice %src[%c0], %mask, %tile, %tile_slice_index : memref<?x?xf16>, vector<[8]xi1>, vector<[8]x[8]xf16>
  return
}

// -----

func.func @arm_sme_load_tile_slice_hor_bf16(%src : memref<?x?xbf16>, %mask : vector<[8]xi1>, %tile : vector<[8]x[8]xbf16>, %tile_slice_index : index) {
  // CHECK: arm_sme.load_tile_slice %{{.*}}[{{.*}}], %{{.*}}, %{{.*}} : memref<?x?xbf16>, vector<[8]xi1>, vector<[8]x[8]xbf16>
  %c0 = arith.constant 0 : index
  %tile_update = arm_sme.load_tile_slice %src[%c0], %mask, %tile, %tile_slice_index : memref<?x?xbf16>, vector<[8]xi1>, vector<[8]x[8]xbf16>
  return
}

// -----

func.func @arm_sme_load_tile_slice_hor_f32(%src : memref<?x?xf32>, %mask : vector<[4]xi1>, %tile : vector<[4]x[4]xf32>, %tile_slice_index : index) {
  // CHECK: arm_sme.load_tile_slice %{{.*}}[{{.*}}], %{{.*}}, %{{.*}} : memref<?x?xf32>, vector<[4]xi1>, vector<[4]x[4]xf32>
  %c0 = arith.constant 0 : index
  %tile_update = arm_sme.load_tile_slice %src[%c0], %mask, %tile, %tile_slice_index : memref<?x?xf32>, vector<[4]xi1>, vector<[4]x[4]xf32>
  return
}

// -----

func.func @arm_sme_load_tile_slice_hor_f64(%src : memref<?x?xf64>, %mask : vector<[2]xi1>, %tile : vector<[2]x[2]xf64>, %tile_slice_index : index) {
  // CHECK: arm_sme.load_tile_slice %{{.*}}[{{.*}}], %{{.*}}, %{{.*}} : memref<?x?xf64>, vector<[2]xi1>, vector<[2]x[2]xf64>
  %c0 = arith.constant 0 : index
  %tile_update = arm_sme.load_tile_slice %src[%c0], %mask, %tile, %tile_slice_index : memref<?x?xf64>, vector<[2]xi1>, vector<[2]x[2]xf64>
  return
}

// -----

func.func @arm_sme_load_tile_slice_ver_i8(%src : memref<?x?xi8>, %mask : vector<[16]xi1>, %tile : vector<[16]x[16]xi8>, %tile_slice_index : index) {
  // CHECK: arm_sme.load_tile_slice {{.*}} layout<vertical> : memref<?x?xi8>, vector<[16]xi1>, vector<[16]x[16]xi8>
  %c0 = arith.constant 0 : index
  %tile_update = arm_sme.load_tile_slice %src[%c0], %mask, %tile, %tile_slice_index layout<vertical> : memref<?x?xi8>, vector<[16]xi1>, vector<[16]x[16]xi8>
  return
}

// -----

func.func @arm_sme_load_tile_slice_ver_i16(%src : memref<?x?xi16>, %mask : vector<[8]xi1>, %tile : vector<[8]x[8]xi16>, %tile_slice_index : index) {
  // CHECK: arm_sme.load_tile_slice {{.*}} layout<vertical> : memref<?x?xi16>, vector<[8]xi1>, vector<[8]x[8]xi16>
  %c0 = arith.constant 0 : index
  %tile_update = arm_sme.load_tile_slice %src[%c0], %mask, %tile, %tile_slice_index layout<vertical> : memref<?x?xi16>, vector<[8]xi1>, vector<[8]x[8]xi16>
  return
}

// -----

func.func @arm_sme_load_tile_slice_ver_i32(%src : memref<?x?xi32>, %mask : vector<[4]xi1>, %tile : vector<[4]x[4]xi32>, %tile_slice_index : index) {
  // CHECK: arm_sme.load_tile_slice {{.*}} layout<vertical> : memref<?x?xi32>, vector<[4]xi1>, vector<[4]x[4]xi32>
  %c0 = arith.constant 0 : index
  %tile_update = arm_sme.load_tile_slice %src[%c0], %mask, %tile, %tile_slice_index layout<vertical> : memref<?x?xi32>, vector<[4]xi1>, vector<[4]x[4]xi32>
  return
}

// -----

func.func @arm_sme_load_tile_slice_ver_i64(%src : memref<?x?xi64>, %mask : vector<[2]xi1>, %tile : vector<[2]x[2]xi64>, %tile_slice_index : index) {
  // CHECK: arm_sme.load_tile_slice {{.*}} layout<vertical> : memref<?x?xi64>, vector<[2]xi1>, vector<[2]x[2]xi64>
  %c0 = arith.constant 0 : index
  %tile_update = arm_sme.load_tile_slice %src[%c0], %mask, %tile, %tile_slice_index layout<vertical> : memref<?x?xi64>, vector<[2]xi1>, vector<[2]x[2]xi64>
  return
}

// -----

func.func @arm_sme_load_tile_slice_ver_i128(%src : memref<?x?xi128>, %mask : vector<[1]xi1>, %tile : vector<[1]x[1]xi128>, %tile_slice_index : index) {
  // CHECK: arm_sme.load_tile_slice {{.*}} layout<vertical> : memref<?x?xi128>, vector<[1]xi1>, vector<[1]x[1]xi128>
  %c0 = arith.constant 0 : index
  %tile_update = arm_sme.load_tile_slice %src[%c0], %mask, %tile, %tile_slice_index layout<vertical> : memref<?x?xi128>, vector<[1]xi1>, vector<[1]x[1]xi128>
  return
}

// -----

func.func @arm_sme_load_tile_slice_ver_f16(%src : memref<?x?xf16>, %mask : vector<[8]xi1>, %tile : vector<[8]x[8]xf16>, %tile_slice_index : index) {
  // CHECK: arm_sme.load_tile_slice {{.*}} layout<vertical> : memref<?x?xf16>, vector<[8]xi1>, vector<[8]x[8]xf16>
  %c0 = arith.constant 0 : index
  %tile_update = arm_sme.load_tile_slice %src[%c0], %mask, %tile, %tile_slice_index layout<vertical> : memref<?x?xf16>, vector<[8]xi1>, vector<[8]x[8]xf16>
  return
}

// -----

func.func @arm_sme_load_tile_slice_ver_bf16(%src : memref<?x?xbf16>, %mask : vector<[8]xi1>, %tile : vector<[8]x[8]xbf16>, %tile_slice_index : index) {
  // CHECK: arm_sme.load_tile_slice {{.*}} layout<vertical> : memref<?x?xbf16>, vector<[8]xi1>, vector<[8]x[8]xbf16>
  %c0 = arith.constant 0 : index
  %tile_update = arm_sme.load_tile_slice %src[%c0], %mask, %tile, %tile_slice_index layout<vertical> : memref<?x?xbf16>, vector<[8]xi1>, vector<[8]x[8]xbf16>
  return
}

// -----

func.func @arm_sme_load_tile_slice_ver_f32(%src : memref<?x?xf32>, %mask : vector<[4]xi1>, %tile : vector<[4]x[4]xf32>, %tile_slice_index : index) {
  // CHECK: arm_sme.load_tile_slice {{.*}} layout<vertical> : memref<?x?xf32>, vector<[4]xi1>, vector<[4]x[4]xf32>
  %c0 = arith.constant 0 : index
  %tile_update = arm_sme.load_tile_slice %src[%c0], %mask, %tile, %tile_slice_index layout<vertical> : memref<?x?xf32>, vector<[4]xi1>, vector<[4]x[4]xf32>
  return
}

// -----

func.func @arm_sme_load_tile_slice_ver_f64(%src : memref<?x?xf64>, %mask : vector<[2]xi1>, %tile : vector<[2]x[2]xf64>, %tile_slice_index : index) {
  // CHECK: arm_sme.load_tile_slice {{.*}} layout<vertical> : memref<?x?xf64>, vector<[2]xi1>, vector<[2]x[2]xf64>
  %c0 = arith.constant 0 : index
  %tile_update = arm_sme.load_tile_slice %src[%c0], %mask, %tile, %tile_slice_index layout<vertical> : memref<?x?xf64>, vector<[2]xi1>, vector<[2]x[2]xf64>
  return
}

// -----

/// Layout is optional and horizontal is the default, verify it's still parsed.
func.func @arm_sme_load_tile_slice_hor_i8(%src : memref<?x?xi8>, %mask : vector<[16]xi1>, %tile : vector<[16]x[16]xi8>, %tile_slice_index : index) {
  // CHECK: arm_sme.load_tile_slice %{{.*}}[{{.*}}], %{{.*}}, %{{.*}} : memref<?x?xi8>, vector<[16]xi1>, vector<[16]x[16]xi8>
  %c0 = arith.constant 0 : index
  %tile_update = arm_sme.load_tile_slice %src[%c0], %mask, %tile, %tile_slice_index layout<horizontal> : memref<?x?xi8>, vector<[16]xi1>, vector<[16]x[16]xi8>
  return
}

//===----------------------------------------------------------------------===//
// arm_sme.store_tile_slice
//===----------------------------------------------------------------------===//

// -----

func.func @arm_sme_store_tile_slice_hor_i8(%tile : vector<[16]x[16]xi8>, %tile_slice_index : index, %mask : vector<[16]xi1>, %dest : memref<?x?xi8>) -> () {
  // CHECK: arm_sme.store_tile_slice {{.*}}, {{.*}}, %{{.*}}[{{.*}}] : memref<?x?xi8>, vector<[16]xi1>, vector<[16]x[16]xi8>
  %c0 = arith.constant 0 : index
  arm_sme.store_tile_slice %tile, %tile_slice_index, %mask, %dest[%c0] : memref<?x?xi8>, vector<[16]xi1>, vector<[16]x[16]xi8>
  return
}

// -----

func.func @arm_sme_store_tile_slice_hor_i16(%tile : vector<[8]x[8]xi16>, %tile_slice_index : index, %mask : vector<[8]xi1>, %dest : memref<?x?xi16>) -> () {
  // CHECK: arm_sme.store_tile_slice {{.*}}, {{.*}}, %{{.*}}[{{.*}}] : memref<?x?xi16>, vector<[8]xi1>, vector<[8]x[8]xi16>
  %c0 = arith.constant 0 : index
  arm_sme.store_tile_slice %tile, %tile_slice_index, %mask, %dest[%c0] : memref<?x?xi16>, vector<[8]xi1>, vector<[8]x[8]xi16>
  return
}

// -----

func.func @arm_sme_store_tile_slice_hor_i32(%tile : vector<[4]x[4]xi32>, %tile_slice_index : index, %mask : vector<[4]xi1>, %dest : memref<?x?xi32>) -> () {
  // CHECK: arm_sme.store_tile_slice {{.*}}, {{.*}}, %{{.*}}[{{.*}}] : memref<?x?xi32>, vector<[4]xi1>, vector<[4]x[4]xi32>
  %c0 = arith.constant 0 : index
  arm_sme.store_tile_slice %tile, %tile_slice_index, %mask, %dest[%c0] : memref<?x?xi32>, vector<[4]xi1>, vector<[4]x[4]xi32>
  return
}

// -----

func.func @arm_sme_store_tile_slice_hor_i64(%tile : vector<[2]x[2]xi64>, %tile_slice_index : index, %mask : vector<[2]xi1>, %dest : memref<?x?xi64>) -> () {
  // CHECK: arm_sme.store_tile_slice {{.*}}, {{.*}}, %{{.*}}[{{.*}}] : memref<?x?xi64>, vector<[2]xi1>, vector<[2]x[2]xi64>
  %c0 = arith.constant 0 : index
  arm_sme.store_tile_slice %tile, %tile_slice_index, %mask, %dest[%c0] : memref<?x?xi64>, vector<[2]xi1>, vector<[2]x[2]xi64>
  return
}

// -----

func.func @arm_sme_store_tile_slice_hor_i128(%tile : vector<[1]x[1]xi128>, %tile_slice_index : index, %mask : vector<[1]xi1>, %dest : memref<?x?xi128>) -> () {
  // CHECK: arm_sme.store_tile_slice {{.*}}, {{.*}}, %{{.*}}[{{.*}}] : memref<?x?xi128>, vector<[1]xi1>, vector<[1]x[1]xi128>
  %c0 = arith.constant 0 : index
  arm_sme.store_tile_slice %tile, %tile_slice_index, %mask, %dest[%c0] : memref<?x?xi128>, vector<[1]xi1>, vector<[1]x[1]xi128>
  return
}

// -----

func.func @arm_sme_store_tile_slice_hor_f16(%tile : vector<[8]x[8]xf16>, %tile_slice_index : index, %mask : vector<[8]xi1>, %dest : memref<?x?xf16>) -> () {
  // CHECK: arm_sme.store_tile_slice {{.*}}, {{.*}}, %{{.*}}[{{.*}}] : memref<?x?xf16>, vector<[8]xi1>, vector<[8]x[8]xf16>
  %c0 = arith.constant 0 : index
  arm_sme.store_tile_slice %tile, %tile_slice_index, %mask, %dest[%c0] : memref<?x?xf16>, vector<[8]xi1>, vector<[8]x[8]xf16>
  return
}

// -----

func.func @arm_sme_store_tile_slice_hor_bf16(%tile : vector<[8]x[8]xbf16>, %tile_slice_index : index, %mask : vector<[8]xi1>, %dest : memref<?x?xbf16>) -> () {
  // CHECK: arm_sme.store_tile_slice {{.*}}, {{.*}}, %{{.*}}[{{.*}}] : memref<?x?xbf16>, vector<[8]xi1>, vector<[8]x[8]xbf16>
  %c0 = arith.constant 0 : index
  arm_sme.store_tile_slice %tile, %tile_slice_index, %mask, %dest[%c0] : memref<?x?xbf16>, vector<[8]xi1>, vector<[8]x[8]xbf16>
  return
}

// -----

func.func @arm_sme_store_tile_slice_hor_f32(%tile : vector<[4]x[4]xf32>, %tile_slice_index : index, %mask : vector<[4]xi1>, %dest : memref<?x?xf32>) -> () {
  // CHECK: arm_sme.store_tile_slice {{.*}}, {{.*}}, %{{.*}}[{{.*}}] : memref<?x?xf32>, vector<[4]xi1>, vector<[4]x[4]xf32>
  %c0 = arith.constant 0 : index
  arm_sme.store_tile_slice %tile, %tile_slice_index, %mask, %dest[%c0] : memref<?x?xf32>, vector<[4]xi1>, vector<[4]x[4]xf32>
  return
}

// -----

func.func @arm_sme_store_tile_slice_hor_f64(%tile : vector<[2]x[2]xf64>, %tile_slice_index : index, %mask : vector<[2]xi1>, %dest : memref<?x?xf64>) -> () {
  // CHECK: arm_sme.store_tile_slice {{.*}}, {{.*}}, %{{.*}}[{{.*}}] : memref<?x?xf64>, vector<[2]xi1>, vector<[2]x[2]xf64>
  %c0 = arith.constant 0 : index
  arm_sme.store_tile_slice %tile, %tile_slice_index, %mask, %dest[%c0] : memref<?x?xf64>, vector<[2]xi1>, vector<[2]x[2]xf64>
  return
}

// -----

func.func @arm_sme_store_tile_slice_ver_i8(%tile : vector<[16]x[16]xi8>, %tile_slice_index : index, %mask : vector<[16]xi1>, %dest : memref<?x?xi8>) -> () {
  // CHECK: arm_sme.store_tile_slice {{.*}} layout<vertical> : memref<?x?xi8>, vector<[16]xi1>, vector<[16]x[16]xi8>
  %c0 = arith.constant 0 : index
  arm_sme.store_tile_slice %tile, %tile_slice_index, %mask, %dest[%c0] layout<vertical> : memref<?x?xi8>, vector<[16]xi1>, vector<[16]x[16]xi8>
  return
}

// -----

func.func @arm_sme_store_tile_slice_ver_i16(%tile : vector<[8]x[8]xi16>, %tile_slice_index : index, %mask : vector<[8]xi1>, %dest : memref<?x?xi16>) -> () {
  // CHECK: arm_sme.store_tile_slice {{.*}} layout<vertical> : memref<?x?xi16>, vector<[8]xi1>, vector<[8]x[8]xi16>
  %c0 = arith.constant 0 : index
  arm_sme.store_tile_slice %tile, %tile_slice_index, %mask, %dest[%c0] layout<vertical> : memref<?x?xi16>, vector<[8]xi1>, vector<[8]x[8]xi16>
  return
}

// -----

func.func @arm_sme_store_tile_slice_ver_i32(%tile : vector<[4]x[4]xi32>, %tile_slice_index : index, %mask : vector<[4]xi1>, %dest : memref<?x?xi32>) -> () {
  // CHECK: arm_sme.store_tile_slice {{.*}} layout<vertical> : memref<?x?xi32>, vector<[4]xi1>, vector<[4]x[4]xi32>
  %c0 = arith.constant 0 : index
  arm_sme.store_tile_slice %tile, %tile_slice_index, %mask, %dest[%c0] layout<vertical> : memref<?x?xi32>, vector<[4]xi1>, vector<[4]x[4]xi32>
  return
}

// -----

func.func @arm_sme_store_tile_slice_ver_i64(%tile : vector<[2]x[2]xi64>, %tile_slice_index : index, %mask : vector<[2]xi1>, %dest : memref<?x?xi64>) -> () {
  // CHECK: arm_sme.store_tile_slice {{.*}} layout<vertical> : memref<?x?xi64>, vector<[2]xi1>, vector<[2]x[2]xi64>
  %c0 = arith.constant 0 : index
  arm_sme.store_tile_slice %tile, %tile_slice_index, %mask, %dest[%c0] layout<vertical> : memref<?x?xi64>, vector<[2]xi1>, vector<[2]x[2]xi64>
  return
}

// -----

func.func @arm_sme_store_tile_slice_ver_i128(%tile : vector<[1]x[1]xi128>, %tile_slice_index : index, %mask : vector<[1]xi1>, %dest : memref<?x?xi128>) -> () {
  // CHECK: arm_sme.store_tile_slice {{.*}} layout<vertical> : memref<?x?xi128>, vector<[1]xi1>, vector<[1]x[1]xi128>
  %c0 = arith.constant 0 : index
  arm_sme.store_tile_slice %tile, %tile_slice_index, %mask, %dest[%c0] layout<vertical> : memref<?x?xi128>, vector<[1]xi1>, vector<[1]x[1]xi128>
  return
}

// -----

func.func @arm_sme_store_tile_slice_ver_f16(%tile : vector<[8]x[8]xf16>, %tile_slice_index : index, %mask : vector<[8]xi1>, %dest : memref<?x?xf16>) -> () {
  // CHECK: arm_sme.store_tile_slice {{.*}} layout<vertical> : memref<?x?xf16>, vector<[8]xi1>, vector<[8]x[8]xf16>
  %c0 = arith.constant 0 : index
  arm_sme.store_tile_slice %tile, %tile_slice_index, %mask, %dest[%c0] layout<vertical> : memref<?x?xf16>, vector<[8]xi1>, vector<[8]x[8]xf16>
  return
}

// -----

func.func @arm_sme_store_tile_slice_ver_bf16(%tile : vector<[8]x[8]xbf16>, %tile_slice_index : index, %mask : vector<[8]xi1>, %dest : memref<?x?xbf16>) -> () {
  // CHECK: arm_sme.store_tile_slice {{.*}} layout<vertical> : memref<?x?xbf16>, vector<[8]xi1>, vector<[8]x[8]xbf16>
  %c0 = arith.constant 0 : index
  arm_sme.store_tile_slice %tile, %tile_slice_index, %mask, %dest[%c0] layout<vertical> : memref<?x?xbf16>, vector<[8]xi1>, vector<[8]x[8]xbf16>
  return
}

// -----

func.func @arm_sme_store_tile_slice_ver_f32(%tile : vector<[4]x[4]xf32>, %tile_slice_index : index, %mask : vector<[4]xi1>, %dest : memref<?x?xf32>) -> () {
  // CHECK: arm_sme.store_tile_slice {{.*}} layout<vertical> : memref<?x?xf32>, vector<[4]xi1>, vector<[4]x[4]xf32>
  %c0 = arith.constant 0 : index
  arm_sme.store_tile_slice %tile, %tile_slice_index, %mask, %dest[%c0] layout<vertical> : memref<?x?xf32>, vector<[4]xi1>, vector<[4]x[4]xf32>
  return
}

// -----

func.func @arm_sme_store_tile_slice_ver_f64(%tile : vector<[2]x[2]xf64>, %tile_slice_index : index, %mask : vector<[2]xi1>, %dest : memref<?x?xf64>) -> () {
  // CHECK: arm_sme.store_tile_slice {{.*}} layout<vertical> : memref<?x?xf64>, vector<[2]xi1>, vector<[2]x[2]xf64>
  %c0 = arith.constant 0 : index
  arm_sme.store_tile_slice %tile, %tile_slice_index, %mask, %dest[%c0] layout<vertical> : memref<?x?xf64>, vector<[2]xi1>, vector<[2]x[2]xf64>
  return
}

// -----

/// Layout is optional and horizontal is the default, verify it's still parsed.
func.func @arm_sme_store_tile_slice_hor_i8(%tile : vector<[16]x[16]xi8>, %tile_slice_index : index, %mask : vector<[16]xi1>, %dest : memref<?x?xi8>) -> () {
  // CHECK: arm_sme.store_tile_slice {{.*}}, {{.*}}, %{{.*}}[{{.*}}] : memref<?x?xi8>, vector<[16]xi1>, vector<[16]x[16]xi8>
  %c0 = arith.constant 0 : index
  arm_sme.store_tile_slice %tile, %tile_slice_index, %mask, %dest[%c0] layout<horizontal> : memref<?x?xi8>, vector<[16]xi1>, vector<[16]x[16]xi8>
  return
}

//===----------------------------------------------------------------------===//
// arm_sme.move_vector_to_tile_slice
//===----------------------------------------------------------------------===//

// -----

func.func @arm_sme_move_vector_to_tile_slice_i8(%vector : vector<[16]xi8>, %tile : vector<[16]x[16]xi8>, %tile_slice_index : index) -> () {
  // CHECK: arm_sme.move_vector_to_tile_slice {{.*}} : vector<[16]xi8> into vector<[16]x[16]xi8>
  %c0 = arith.constant 0 : index
  arm_sme.move_vector_to_tile_slice %vector, %tile, %tile_slice_index : vector<[16]xi8> into vector<[16]x[16]xi8>
  return
}

// -----

func.func @arm_sme_move_vector_to_tile_slice_i16(%vector : vector<[8]xi16>, %tile : vector<[8]x[8]xi16>, %tile_slice_index : index) -> () {
  // CHECK: arm_sme.move_vector_to_tile_slice {{.*}} : vector<[8]xi16> into vector<[8]x[8]xi16>
  %c0 = arith.constant 0 : index
  arm_sme.move_vector_to_tile_slice %vector, %tile, %tile_slice_index : vector<[8]xi16> into vector<[8]x[8]xi16>
  return
}

// -----

func.func @arm_sme_move_vector_to_tile_slice_i32(%vector : vector<[4]xi32>, %tile : vector<[4]x[4]xi32>, %tile_slice_index : index) -> () {
  // CHECK: arm_sme.move_vector_to_tile_slice {{.*}} : vector<[4]xi32> into vector<[4]x[4]xi32>
  %c0 = arith.constant 0 : index
  arm_sme.move_vector_to_tile_slice %vector, %tile, %tile_slice_index : vector<[4]xi32> into vector<[4]x[4]xi32>
  return
}

// -----

func.func @arm_sme_move_vector_to_tile_slice_i64(%vector : vector<[2]xi64>, %tile : vector<[2]x[2]xi64>, %tile_slice_index : index) -> () {
  // CHECK: arm_sme.move_vector_to_tile_slice {{.*}} : vector<[2]xi64> into vector<[2]x[2]xi64>
  %c0 = arith.constant 0 : index
  arm_sme.move_vector_to_tile_slice %vector, %tile, %tile_slice_index : vector<[2]xi64> into vector<[2]x[2]xi64>
  return
}

// -----

func.func @arm_sme_move_vector_to_tile_slice_i128(%vector : vector<[1]xi128>, %tile : vector<[1]x[1]xi128>, %tile_slice_index : index) -> () {
  // CHECK: arm_sme.move_vector_to_tile_slice {{.*}} : vector<[1]xi128> into vector<[1]x[1]xi128>
  %c0 = arith.constant 0 : index
  arm_sme.move_vector_to_tile_slice %vector, %tile, %tile_slice_index : vector<[1]xi128> into vector<[1]x[1]xi128>
  return
}

// -----

func.func @arm_sme_move_vector_to_tile_slice_f16(%vector : vector<[8]xf16>, %tile : vector<[8]x[8]xf16>, %tile_slice_index : index) -> () {
  // CHECK: arm_sme.move_vector_to_tile_slice {{.*}} : vector<[8]xf16> into vector<[8]x[8]xf16>
  %c0 = arith.constant 0 : index
  arm_sme.move_vector_to_tile_slice %vector, %tile, %tile_slice_index : vector<[8]xf16> into vector<[8]x[8]xf16>
  return
}

// -----

func.func @arm_sme_move_vector_to_tile_slice_bf16(%vector : vector<[8]xbf16>, %tile : vector<[8]x[8]xbf16>, %tile_slice_index : index) -> () {
  // CHECK: arm_sme.move_vector_to_tile_slice {{.*}} : vector<[8]xbf16> into vector<[8]x[8]xbf16>
  %c0 = arith.constant 0 : index
  arm_sme.move_vector_to_tile_slice %vector, %tile, %tile_slice_index : vector<[8]xbf16> into vector<[8]x[8]xbf16>
  return
}

// -----

func.func @arm_sme_move_vector_to_tile_slice_f32(%vector : vector<[4]xf32>, %tile : vector<[4]x[4]xf32>, %tile_slice_index : index) -> () {
  // CHECK: arm_sme.move_vector_to_tile_slice {{.*}} : vector<[4]xf32> into vector<[4]x[4]xf32>
  %c0 = arith.constant 0 : index
  arm_sme.move_vector_to_tile_slice %vector, %tile, %tile_slice_index : vector<[4]xf32> into vector<[4]x[4]xf32>
  return
}

// -----

func.func @arm_sme_move_vector_to_tile_slice_f64(%vector : vector<[2]xf64>, %tile : vector<[2]x[2]xf64>, %tile_slice_index : index) -> () {
  // CHECK: arm_sme.move_vector_to_tile_slice {{.*}} : vector<[2]xf64> into vector<[2]x[2]xf64>
  %c0 = arith.constant 0 : index
  arm_sme.move_vector_to_tile_slice %vector, %tile, %tile_slice_index : vector<[2]xf64> into vector<[2]x[2]xf64>
  return
}

// -----

func.func @arm_sme_move_vector_to_tile_slice_ver_i8(%vector : vector<[16]xi8>, %tile : vector<[16]x[16]xi8>, %tile_slice_index : index) -> () {
  // CHECK: arm_sme.move_vector_to_tile_slice {{.*}} layout<vertical> : vector<[16]xi8> into vector<[16]x[16]xi8>
  %c0 = arith.constant 0 : index
  arm_sme.move_vector_to_tile_slice %vector, %tile, %tile_slice_index layout<vertical> : vector<[16]xi8> into vector<[16]x[16]xi8>
  return
}

//===----------------------------------------------------------------------===//
// arm_sme.move_tile_slice_to_vector
//===----------------------------------------------------------------------===//

// -----

func.func @arm_sme_move_tile_slice_to_vector_i8(%tile : vector<[16]x[16]xi8>, %tile_slice_index : index) -> vector<[16]xi8> {
  // CHECK: arm_sme.move_tile_slice_to_vector {{.*}} : vector<[16]xi8> from vector<[16]x[16]xi8>
  %slice = arm_sme.move_tile_slice_to_vector %tile[%tile_slice_index] : vector<[16]xi8> from vector<[16]x[16]xi8>
  return %slice : vector<[16]xi8>
}

// -----

func.func @arm_sme_move_tile_slice_to_vector_i16(%tile : vector<[8]x[8]xi16>, %tile_slice_index : index) -> vector<[8]xi16> {
  // CHECK: arm_sme.move_tile_slice_to_vector {{.*}} : vector<[8]xi16> from vector<[8]x[8]xi16>
  %slice = arm_sme.move_tile_slice_to_vector %tile[%tile_slice_index] : vector<[8]xi16> from vector<[8]x[8]xi16>
  return %slice : vector<[8]xi16>
}

// -----

func.func @arm_sme_move_tile_slice_to_vector_i32(%tile : vector<[4]x[4]xi32>, %tile_slice_index : index) -> vector<[4]xi32> {
  // CHECK: arm_sme.move_tile_slice_to_vector {{.*}} : vector<[4]xi32> from vector<[4]x[4]xi32>
  %slice = arm_sme.move_tile_slice_to_vector %tile[%tile_slice_index] : vector<[4]xi32> from vector<[4]x[4]xi32>
  return %slice : vector<[4]xi32>
}

// -----

func.func @arm_sme_move_tile_slice_to_vector_i64(%tile : vector<[2]x[2]xi64>, %tile_slice_index : index) -> vector<[2]xi64> {
  // CHECK: arm_sme.move_tile_slice_to_vector {{.*}} : vector<[2]xi64> from vector<[2]x[2]xi64>
  %slice = arm_sme.move_tile_slice_to_vector %tile[%tile_slice_index] : vector<[2]xi64> from vector<[2]x[2]xi64>
  return %slice : vector<[2]xi64>
}

// -----

func.func @arm_sme_move_tile_slice_to_vector_i128(%tile : vector<[1]x[1]xi128>, %tile_slice_index : index) -> vector<[1]xi128> {
  // CHECK: arm_sme.move_tile_slice_to_vector {{.*}} : vector<[1]xi128> from vector<[1]x[1]xi128>
  %slice = arm_sme.move_tile_slice_to_vector %tile[%tile_slice_index] : vector<[1]xi128> from vector<[1]x[1]xi128>
  return %slice : vector<[1]xi128>
}

// -----

func.func @arm_sme_move_tile_slice_to_vector_f16(%tile : vector<[8]x[8]xf16>, %tile_slice_index : index) -> vector<[8]xf16> {
  // CHECK: arm_sme.move_tile_slice_to_vector {{.*}} : vector<[8]xf16> from vector<[8]x[8]xf16>
  %slice = arm_sme.move_tile_slice_to_vector %tile[%tile_slice_index] : vector<[8]xf16> from vector<[8]x[8]xf16>
  return %slice : vector<[8]xf16>
}

// -----

func.func @arm_sme_move_tile_slice_to_vector_bf16(%tile : vector<[8]x[8]xbf16>, %tile_slice_index : index) -> vector<[8]xbf16> {
  // CHECK: arm_sme.move_tile_slice_to_vector {{.*}} : vector<[8]xbf16> from vector<[8]x[8]xbf16>
  %slice = arm_sme.move_tile_slice_to_vector %tile[%tile_slice_index] : vector<[8]xbf16> from vector<[8]x[8]xbf16>
  return %slice : vector<[8]xbf16>
}

// -----

func.func @arm_sme_move_tile_slice_to_vector_f32(%tile : vector<[4]x[4]xf32>, %tile_slice_index : index) -> vector<[4]xf32> {
  // CHECK: arm_sme.move_tile_slice_to_vector {{.*}} : vector<[4]xf32> from vector<[4]x[4]xf32>
  %slice = arm_sme.move_tile_slice_to_vector %tile[%tile_slice_index] : vector<[4]xf32> from vector<[4]x[4]xf32>
  return %slice : vector<[4]xf32>
}

// -----

func.func @arm_sme_move_tile_slice_to_vector_f64(%tile : vector<[2]x[2]xf64>, %tile_slice_index : index) -> vector<[2]xf64> {
  // CHECK: arm_sme.move_tile_slice_to_vector {{.*}} : vector<[2]xf64> from vector<[2]x[2]xf64>
  %slice = arm_sme.move_tile_slice_to_vector %tile[%tile_slice_index] : vector<[2]xf64> from vector<[2]x[2]xf64>
  return %slice : vector<[2]xf64>
}

// -----

func.func @arm_sme_move_tile_slice_to_vector_ver_f64(%tile : vector<[2]x[2]xf64>, %tile_slice_index : index) -> vector<[2]xf64> {
  // CHECK: arm_sme.move_tile_slice_to_vector {{.*}} layout<vertical> : vector<[2]xf64> from vector<[2]x[2]xf64>
  %slice = arm_sme.move_tile_slice_to_vector %tile[%tile_slice_index] layout<vertical> : vector<[2]xf64> from vector<[2]x[2]xf64>
  return %slice : vector<[2]xf64>
}

//===----------------------------------------------------------------------===//
// arm_sme.outerproduct
//===----------------------------------------------------------------------===//

// -----

func.func @arm_sme_outerproduct(%vecA: vector<[8]xi16>, %vecB: vector<[8]xi16>) -> vector<[8]x[8]xi16> {
  // CHECK: arm_sme.outerproduct {{.*}}, {{.*}} : vector<[8]xi16>, vector<[8]xi16>
  %result = arm_sme.outerproduct %vecA, %vecB : vector<[8]xi16>, vector<[8]xi16>
  return %result : vector<[8]x[8]xi16>
}

// -----

func.func @arm_sme_outerproduct_with_masking(%vecA: vector<[4]xf32>, %vecB: vector<[4]xf32>, %maskA: vector<[4]xi1>, %maskB: vector<[4]xi1>) -> vector<[4]x[4]xf32> {
  // CHECK: arm_sme.outerproduct {{.*}}, {{.*}} masks({{.*}}, {{.*}}) : vector<[4]xf32>, vector<[4]xf32>
  %result = arm_sme.outerproduct %vecA, %vecB masks(%maskA, %maskB) : vector<[4]xf32>, vector<[4]xf32>
  return %result : vector<[4]x[4]xf32>
}

// -----

func.func @arm_sme_outerproduct_with_acc(%vecA: vector<[2]xi64>, %vecB: vector<[2]xi64>, %acc: vector<[2]x[2]xi64>) -> vector<[2]x[2]xi64> {
  // CHECK: arm_sme.outerproduct {{.*}}, {{.*}} acc({{.*}}) : vector<[2]xi64>, vector<[2]xi64>
  %result = arm_sme.outerproduct %vecA, %vecB acc(%acc) : vector<[2]xi64>, vector<[2]xi64>
  return %result : vector<[2]x[2]xi64>
}

// -----

func.func @arm_sme_outerproduct_with_kind(%vecA: vector<[2]xf64>, %vecB: vector<[2]xf64>) -> vector<[2]x[2]xf64>  {
  // CHECK: arm_sme.outerproduct {{.*}}, {{.*}} kind<sub> : vector<[2]xf64>, vector<[2]xf64>
  %result = arm_sme.outerproduct %vecA, %vecB kind<sub> : vector<[2]xf64>, vector<[2]xf64>
  return %result : vector<[2]x[2]xf64>
}

// -----

func.func @arm_sme_outerproduct_with_everything(%vecA: vector<[16]xi8>, %vecB: vector<[16]xi8>, %acc: vector<[16]x[16]xi8>, %maskA: vector<[16]xi1>, %maskB: vector<[16]xi1>) -> vector<[16]x[16]xi8> {
  // CHECK: arm_sme.outerproduct {{.*}}, {{.*}} kind<sub> acc({{.*}}) masks({{.*}}, {{.*}}) : vector<[16]xi8>, vector<[16]xi8>
  %result = arm_sme.outerproduct %vecA, %vecB kind<sub> acc(%acc) masks(%maskA, %maskB) : vector<[16]xi8>, vector<[16]xi8>
  return %result : vector<[16]x[16]xi8>
}

//===----------------------------------------------------------------------===//
// arm_sme.streaming_vl
//===----------------------------------------------------------------------===//

// -----

func.func @arm_sme_streaming_vl_bytes() -> index {
  // CHECK: arm_sme.streaming_vl <byte>
  %svl_b = arm_sme.streaming_vl <byte>
  return %svl_b : index
}

// -----

func.func @arm_sme_streaming_vl_half_words() -> index {
  // CHECK: arm_sme.streaming_vl <half>
  %svl_h = arm_sme.streaming_vl <half>
  return %svl_h : index
}

// -----

func.func @arm_sme_streaming_vl_words() -> index {
  // CHECK: arm_sme.streaming_vl <word>
  %svl_w = arm_sme.streaming_vl <word>
  return %svl_w : index
}

// -----

func.func @arm_sme_streaming_vl_double_words() -> index {
  // CHECK: arm_sme.streaming_vl <double>
  %svl_d = arm_sme.streaming_vl <double>
  return %svl_d : index
}

//===----------------------------------------------------------------------===//
// arm_sme.fmopa_2way
//===----------------------------------------------------------------------===//

// -----

func.func @arm_sme_fmopa_2way_f16f16_to_f32(%vecA: vector<[8]xf16>, %vecB: vector<[8]xf16>) -> vector<[4]x[4]xf32> {
  // CHECK: arm_sme.fmopa_2way {{.*}}, {{.*}} : vector<[8]xf16>, vector<[8]xf16> into vector<[4]x[4]xf32>
  %result = arm_sme.fmopa_2way %vecA, %vecB : vector<[8]xf16>, vector<[8]xf16> into vector<[4]x[4]xf32>
  return %result : vector<[4]x[4]xf32>
}

// -----

func.func @arm_sme_fmopa_2way_bf16bf16_to_f32(%vecA: vector<[8]xbf16>, %vecB: vector<[8]xbf16>) -> vector<[4]x[4]xf32> {
  // CHECK: arm_sme.fmopa_2way {{.*}}, {{.*}} : vector<[8]xbf16>, vector<[8]xbf16> into vector<[4]x[4]xf32>
  %result = arm_sme.fmopa_2way %vecA, %vecB : vector<[8]xbf16>, vector<[8]xbf16> into vector<[4]x[4]xf32>
  return %result : vector<[4]x[4]xf32>
}

// -----

func.func @arm_sme_fmopa_2way_with_masking(%vecA: vector<[8]xf16>, %vecB: vector<[8]xf16>, %maskA: vector<[8]xi1>, %maskB: vector<[8]xi1>) -> vector<[4]x[4]xf32> {
  // CHECK: arm_sme.fmopa_2way {{.*}}, {{.*}} masks({{.*}}, {{.*}}) : vector<[8]xf16>, vector<[8]xf16> into vector<[4]x[4]xf32>
  %result = arm_sme.fmopa_2way %vecA, %vecB masks(%maskA, %maskB) : vector<[8]xf16>, vector<[8]xf16> into vector<[4]x[4]xf32>
  return %result : vector<[4]x[4]xf32>
}

// -----

func.func @arm_sme_fmopa_2way_with_acc(%vecA: vector<[8]xf16>, %vecB: vector<[8]xf16>, %acc : vector<[4]x[4]xf32>) -> vector<[4]x[4]xf32> {
  // CHECK: arm_sme.fmopa_2way {{.*}}, {{.*}} acc({{.*}}) : vector<[8]xf16>, vector<[8]xf16> into vector<[4]x[4]xf32>
  %result = arm_sme.fmopa_2way %vecA, %vecB acc(%acc) : vector<[8]xf16>, vector<[8]xf16> into vector<[4]x[4]xf32>
  return %result : vector<[4]x[4]xf32>
}

// -----

func.func @arm_sme_fmopa_2way_with_everything(%vecA: vector<[8]xf16>, %vecB: vector<[8]xf16>, %acc : vector<[4]x[4]xf32>, %maskA: vector<[8]xi1>, %maskB: vector<[8]xi1>) -> vector<[4]x[4]xf32> {
  // CHECK: arm_sme.fmopa_2way {{.*}}, {{.*}} acc({{.*}}) masks({{.*}}, {{.*}}) : vector<[8]xf16>, vector<[8]xf16> into vector<[4]x[4]xf32>
  %result = arm_sme.fmopa_2way %vecA, %vecB acc(%acc) masks(%maskA, %maskB) : vector<[8]xf16>, vector<[8]xf16> into vector<[4]x[4]xf32>
  return %result : vector<[4]x[4]xf32>
}

//===----------------------------------------------------------------------===//
// arm_sme.fmops_2way
//===----------------------------------------------------------------------===//

// -----

func.func @arm_sme_fmops_2way_f16f16_to_f32(%vecA: vector<[8]xf16>, %vecB: vector<[8]xf16>) -> vector<[4]x[4]xf32> {
  // CHECK: arm_sme.fmops_2way {{.*}}, {{.*}} : vector<[8]xf16>, vector<[8]xf16> into vector<[4]x[4]xf32>
  %result = arm_sme.fmops_2way %vecA, %vecB : vector<[8]xf16>, vector<[8]xf16> into vector<[4]x[4]xf32>
  return %result : vector<[4]x[4]xf32>
}

// -----

func.func @arm_sme_fmops_2way_bf16bf16_to_f32(%vecA: vector<[8]xbf16>, %vecB: vector<[8]xbf16>) -> vector<[4]x[4]xf32> {
  // CHECK: arm_sme.fmops_2way {{.*}}, {{.*}} : vector<[8]xbf16>, vector<[8]xbf16> into vector<[4]x[4]xf32>
  %result = arm_sme.fmops_2way %vecA, %vecB : vector<[8]xbf16>, vector<[8]xbf16> into vector<[4]x[4]xf32>
  return %result : vector<[4]x[4]xf32>
}

//===----------------------------------------------------------------------===//
// arm_sme.smopa_2way
//===----------------------------------------------------------------------===//

// -----

func.func @arm_sme_smopa_2way_i16i16_to_i32(%vecA: vector<[8]xi16>, %vecB: vector<[8]xi16>) -> vector<[4]x[4]xi32> {
  // CHECK: arm_sme.smopa_2way {{.*}}, {{.*}} : vector<[8]xi16>, vector<[8]xi16> into vector<[4]x[4]xi32>
  %result = arm_sme.smopa_2way %vecA, %vecB : vector<[8]xi16>, vector<[8]xi16> into vector<[4]x[4]xi32>
  return %result : vector<[4]x[4]xi32>
}

//===----------------------------------------------------------------------===//
// arm_sme.smops_2way
//===----------------------------------------------------------------------===//

// -----

func.func @arm_sme_smops_2way_i16i16_to_i32(%vecA: vector<[8]xi16>, %vecB: vector<[8]xi16>) -> vector<[4]x[4]xi32> {
  // CHECK: arm_sme.smops_2way {{.*}}, {{.*}} : vector<[8]xi16>, vector<[8]xi16> into vector<[4]x[4]xi32>
  %result = arm_sme.smops_2way %vecA, %vecB : vector<[8]xi16>, vector<[8]xi16> into vector<[4]x[4]xi32>
  return %result : vector<[4]x[4]xi32>
}

//===----------------------------------------------------------------------===//
// arm_sme.umopa_2way
//===----------------------------------------------------------------------===//

// -----

func.func @arm_sme_umopa_2way_i16i16_to_i32(%vecA: vector<[8]xi16>, %vecB: vector<[8]xi16>) -> vector<[4]x[4]xi32> {
  // CHECK: arm_sme.umopa_2way {{.*}}, {{.*}} : vector<[8]xi16>, vector<[8]xi16> into vector<[4]x[4]xi32>
  %result = arm_sme.umopa_2way %vecA, %vecB : vector<[8]xi16>, vector<[8]xi16> into vector<[4]x[4]xi32>
  return %result : vector<[4]x[4]xi32>
}

//===----------------------------------------------------------------------===//
// arm_sme.umops_2way
//===----------------------------------------------------------------------===//

// -----

func.func @arm_sme_umops_2way_i16i16_to_i32(%vecA: vector<[8]xi16>, %vecB: vector<[8]xi16>) -> vector<[4]x[4]xi32> {
  // CHECK: arm_sme.umops_2way {{.*}}, {{.*}} : vector<[8]xi16>, vector<[8]xi16> into vector<[4]x[4]xi32>
  %result = arm_sme.umops_2way %vecA, %vecB : vector<[8]xi16>, vector<[8]xi16> into vector<[4]x[4]xi32>
  return %result : vector<[4]x[4]xi32>
}
