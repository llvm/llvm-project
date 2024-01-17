// RUN: mlir-opt %s | FileCheck %s
// Verify the printed output can be parsed.
// RUN: mlir-opt %s | mlir-opt | FileCheck %s
// Verify the generic form can be parsed.
// RUN: mlir-opt -mlir-print-op-generic %s | mlir-opt | FileCheck %s

#sg_map_fp16 = #xegpu.sg_map<wi_layout = [2, 8], wi_data = [1, 2]>

func.func @test_create_nd_tdesc_0(%src: memref<24x32xf16>) {
  %c0 = arith.constant 2 : index
  %c1 = arith.constant 4 : index

  // CHECK:  xegpu.create_nd_tdesc
  // CHECK-SAME: memref<24x32xf16> -> !xegpu.tensor_desc<8x16xf16, #xegpu.sg_map<wi_layout = [2, 8], wi_data = [1, 2]>>
  %1 = xegpu.create_nd_tdesc %src[%c0, %c1]
      : memref<24x32xf16> -> !xegpu.tensor_desc<8x16xf16, #sg_map_fp16>

  // CHECK: xegpu.create_nd_tdesc
  // CHECK-SAME: memref<24x32xf16> -> !xegpu.tensor_desc<8x16xf16, #xegpu.sg_map<wi_layout = [2, 8], wi_data = [1, 2]>>
  %2 = xegpu.create_nd_tdesc %src[2, 4]
      : memref<24x32xf16> -> !xegpu.tensor_desc<8x16xf16, #sg_map_fp16>

  return
}

// CHECK-LABEL: func @test_create_nd_tdesc_1({{.*}}) {
func.func @test_create_nd_tdesc_1(%src: memref<24x32xf16>, %x : index, %y : index) {
  // CHECK: xegpu.create_nd_tdesc
  // CHECK-SAME: memref<24x32xf16> -> !xegpu.tensor_desc<8x16xf16, #xegpu.sg_map<wi_layout = [2, 8], wi_data = [1, 2]>>
  %1 = xegpu.create_nd_tdesc %src[%x, %y]
      : memref<24x32xf16> -> !xegpu.tensor_desc<8x16xf16, #sg_map_fp16>
  return
}

// CHECK-LABEL: func @test_create_nd_tdesc_2({{.*}}) {
func.func @test_create_nd_tdesc_2(%src: ui64, %w : index, %h : index, %x : index, %y : index) {
  %c1 = arith.constant 1 : index
  // CHECK: xegpu.create_nd_tdesc
  // CHECK-SAME: ui64 -> !xegpu.tensor_desc<8x16xf16, #xegpu.sg_map<wi_layout = [2, 8], wi_data = [1, 2]>>
  %1 = xegpu.create_nd_tdesc %src[%x, %y], [%h, %w], [%w, %c1]  : ui64 -> !xegpu.tensor_desc<8x16xf16, #sg_map_fp16>
  return
}

// CHECK-LABEL: func @test_create_nd_tdesc_3({{.*}}) {
func.func @test_create_nd_tdesc_3(%src: memref<?x?xf16>, %w : index, %h : index, %x : index, %y : index) {
  %c1 = arith.constant 1 : index
  // CHECK: xegpu.create_nd_tdesc
  // CHECK-SAME: memref<?x?xf16> -> !xegpu.tensor_desc<8x16xf16, #xegpu.sg_map<wi_layout = [2, 8], wi_data = [1, 2]>>
  %1 = xegpu.create_nd_tdesc %src[%x, %y], [%h, %w], [%w, %c1]  : memref<?x?xf16> -> !xegpu.tensor_desc<8x16xf16, #sg_map_fp16>
  return
}


// CHECK-LABEL: func @test_create_nd_tdesc_4({{.*}}) {
func.func @test_create_nd_tdesc_4(%src: memref<?x?xf16>, %w : index, %h : index, %x : index, %y : index) {
  %c1 = arith.constant 1 : index
  // CHECK: xegpu.create_nd_tdesc
  // CHECK-SAME: memref<?x?xf16> -> !xegpu.tensor_desc<8x16xf16, #xegpu.sg_map<wi_layout = [2, 8], wi_data = [1, 2]>>
  %1 = xegpu.create_nd_tdesc %src[%x, %y], [%h, %w], [%w, %c1]
          : memref<?x?xf16> -> !xegpu.tensor_desc<8x16xf16, #sg_map_fp16>
  return
}

// CHECK-LABEL: func @test_create_nd_tdesc_5({{.*}}) {
func.func @test_create_nd_tdesc_5(%src: memref<?x?xf16>, %w : index, %h : index, %x : index, %y : index) {
  %c1 = arith.constant 1 : index
  // CHECK: xegpu.create_nd_tdesc
  // CHECK-SAME: memref<?x?xf16> -> !xegpu.tensor_desc<8x16xf16, #xegpu.tdesc_attr<memory_scope = slm, map = <wi_layout = [2, 8], wi_data = [1, 2]>>>
  %1 = xegpu.create_nd_tdesc %src[%x, %y], [%h, %w], [%w, %c1]
                                  : memref<?x?xf16> -> !xegpu.tensor_desc<8x16xf16, #xegpu.tdesc_attr<memory_scope = slm, map = #sg_map_fp16>>
  return
}

// CHECK-LABEL: func @test_create_nd_tdesc_6({{.*}}) {
func.func @test_create_nd_tdesc_6(%src: memref<?x?xf16>, %w : index, %h : index, %x : index, %y : index) {
  %c1 = arith.constant 1 : index
  // CHECK: xegpu.create_nd_tdesc
  // CHECK-SAME: memref<?x?xf16> -> !xegpu.tensor_desc<8x16xf16, #xegpu.tdesc_attr<memory_scope = slm, map = <wi_layout = [2, 8], wi_data = [1, 2]>>>
  %1 = xegpu.create_nd_tdesc %src[%x, %y], [%h, %w], [%w, %c1]
                            : memref<?x?xf16> -> !xegpu.tensor_desc<8x16xf16, #xegpu.tdesc_attr<memory_scope = slm, map = #sg_map_fp16>>
  return
}

// CHECK-LABEL: func @test_create_nd_tdesc_7({{.*}}) {
func.func @test_create_nd_tdesc_7(%src: memref<1024xf16>, %offset : index) {
  // CHECK: xegpu.create_nd_tdesc
  // CHECK-SAME: memref<1024xf16> -> !xegpu.tensor_desc<16xf16, #xegpu.sg_map<wi_layout = [2, 8], wi_data = [1, 2]>>
  %1 = xegpu.create_nd_tdesc %src[%offset] : memref<1024xf16> -> !xegpu.tensor_desc<16xf16, #sg_map_fp16>
  return
}


// CHECK-LABEL: func @test_create_nd_tdesc_8({{.*}}) {
func.func @test_create_nd_tdesc_8(%src: memref<?x?xf16>, %w : index, %h : index, %x : index) {
  %c1 = arith.constant 1 : index
  // CHECK: xegpu.create_nd_tdesc
  // CHECK-SAME: memref<?x?xf16> -> !xegpu.tensor_desc<8x16xf16, #xegpu.tdesc_attr<memory_scope = slm, map = <wi_layout = [2, 8], wi_data = [1, 2]>>>
  %1 = xegpu.create_nd_tdesc %src[8, %x], [%h, %w], [%w, %c1]
                                    : memref<?x?xf16> -> !xegpu.tensor_desc<8x16xf16, #xegpu.tdesc_attr<memory_scope = slm, map = #sg_map_fp16>>
  return
}

// CHECK-LABEL: func @test_create_nd_tdesc_9({{.*}}) {
func.func @test_create_nd_tdesc_9(%src: memref<?x?xf16>, %w : index, %h : index, %x : index) {
  %c1 = arith.constant 1 : index
  // CHECK: xegpu.create_nd_tdesc
  // CHECK-SAME: memref<?x?xf16> -> !xegpu.tensor_desc<64x128xf16, #xegpu.tdesc_attr<memory_scope = slm, map = <wi_layout = [2, 8], wi_data = [1, 2]>>>
  %1 = xegpu.create_nd_tdesc %src[8, %x], [%h, %w], [%w, %c1] : memref<?x?xf16>
            -> !xegpu.tensor_desc<64x128xf16, #xegpu.tdesc_attr<memory_scope = slm, map = #sg_map_fp16>>
  return
}
