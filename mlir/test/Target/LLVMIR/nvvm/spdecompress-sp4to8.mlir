// RUN: mlir-translate -mlir-to-llvmir %s | FileCheck %s

// rpf=1: metadata=1, compressed_data=1, data=2
llvm.func @spdecompress_sp4to8_elmsz_b8_idxsz_b4_rpf_1(%metadata: i32, %compdata: i32) -> vector<2xi32> {
  // CHECK-LABEL: define <2 x i32> @spdecompress_sp4to8_elmsz_b8_idxsz_b4_rpf_1(i32 %0, i32 %1) {
  // CHECK-NEXT:    %[[CDATA:[0-9]+]] = bitcast i32 %1 to <4 x i8>
  // CHECK-NEXT:    %[[RES:[0-9]+]] = call <8 x i8> @llvm.nvvm.spdecompress.v8i8.i32.v4i8(i32 %0, <4 x i8> %[[CDATA]], /* idx_size= */ i32 4, /* num_tgt= */ i32 8)
  // CHECK-NEXT:    %[[DATA:[0-9]+]] = bitcast <8 x i8> %[[RES]] to <2 x i32>
  // CHECK-NEXT:    ret <2 x i32> %[[DATA]]
  // CHECK-NEXT:  }
  %res = nvvm.spdecompress %metadata, %compdata
      comp_factor = sp4to8 index_size = b4 elem_size = b8 rep_factor = x1
      : i32, i32 -> vector<2xi32>
  llvm.return %res : vector<2xi32>
}

// rpf=2: metadata=1, compressed_data=2, data=4
llvm.func @spdecompress_sp4to8_elmsz_b8_idxsz_b4_rpf_2(%metadata: i32, %compdata: vector<2xi32>) -> vector<4xi32> {
  // CHECK-LABEL: define <4 x i32> @spdecompress_sp4to8_elmsz_b8_idxsz_b4_rpf_2(i32 %0, <2 x i32> %1) {
  // CHECK-NEXT:    %[[CDATA:[0-9]+]] = bitcast <2 x i32> %1 to <8 x i8>
  // CHECK-NEXT:    %[[RES:[0-9]+]] = call <16 x i8> @llvm.nvvm.spdecompress.v16i8.i32.v8i8(i32 %0, <8 x i8> %[[CDATA]], /* idx_size= */ i32 4, /* num_tgt= */ i32 8)
  // CHECK-NEXT:    %[[DATA:[0-9]+]] = bitcast <16 x i8> %[[RES]] to <4 x i32>
  // CHECK-NEXT:    ret <4 x i32> %[[DATA]]
  // CHECK-NEXT:  }
  %res = nvvm.spdecompress %metadata, %compdata
      comp_factor = sp4to8 index_size = b4 elem_size = b8 rep_factor = x2
      : i32, vector<2xi32> -> vector<4xi32>
  llvm.return %res : vector<4xi32>
}

// rpf=4: metadata=2, compressed_data=4, data=8
llvm.func @spdecompress_sp4to8_elmsz_b8_idxsz_b4_rpf_4(%metadata: vector<2xi32>, %compdata: vector<4xi32>) -> vector<8xi32> {
  // CHECK-LABEL: define <8 x i32> @spdecompress_sp4to8_elmsz_b8_idxsz_b4_rpf_4(<2 x i32> %0, <4 x i32> %1) {
  // CHECK-NEXT:    %[[CDATA:[0-9]+]] = bitcast <4 x i32> %1 to <16 x i8>
  // CHECK-NEXT:    %[[RES:[0-9]+]] = call <32 x i8> @llvm.nvvm.spdecompress.v32i8.v2i32.v16i8(<2 x i32> %0, <16 x i8> %[[CDATA]], /* idx_size= */ i32 4, /* num_tgt= */ i32 8)
  // CHECK-NEXT:    %[[DATA:[0-9]+]] = bitcast <32 x i8> %[[RES]] to <8 x i32>
  // CHECK-NEXT:    ret <8 x i32> %[[DATA]]
  // CHECK-NEXT:  }
  %res = nvvm.spdecompress %metadata, %compdata
      comp_factor = sp4to8 index_size = b4 elem_size = b8 rep_factor = x4
      : vector<2xi32>, vector<4xi32> -> vector<8xi32>
  llvm.return %res : vector<8xi32>
}

// rpf=8: metadata=4, compressed_data=8, data=16
llvm.func @spdecompress_sp4to8_elmsz_b8_idxsz_b4_rpf_8(%metadata: vector<4xi32>, %compdata: vector<8xi32>) -> vector<16xi32> {
  // CHECK-LABEL: define <16 x i32> @spdecompress_sp4to8_elmsz_b8_idxsz_b4_rpf_8(<4 x i32> %0, <8 x i32> %1) {
  // CHECK-NEXT:    %[[CDATA:[0-9]+]] = bitcast <8 x i32> %1 to <32 x i8>
  // CHECK-NEXT:    %[[RES:[0-9]+]] = call <64 x i8> @llvm.nvvm.spdecompress.v64i8.v4i32.v32i8(<4 x i32> %0, <32 x i8> %[[CDATA]], /* idx_size= */ i32 4, /* num_tgt= */ i32 8)
  // CHECK-NEXT:    %[[DATA:[0-9]+]] = bitcast <64 x i8> %[[RES]] to <16 x i32>
  // CHECK-NEXT:    ret <16 x i32> %[[DATA]]
  // CHECK-NEXT:  }
  %res = nvvm.spdecompress %metadata, %compdata
      comp_factor = sp4to8 index_size = b4 elem_size = b8 rep_factor = x8
      : vector<4xi32>, vector<8xi32> -> vector<16xi32>
  llvm.return %res : vector<16xi32>
}

// rpf=16: metadata=8, compressed_data=16, data=32
llvm.func @spdecompress_sp4to8_elmsz_b8_idxsz_b4_rpf_16(%metadata: vector<8xi32>, %compdata: vector<16xi32>) -> vector<32xi32> {
  // CHECK-LABEL: define <32 x i32> @spdecompress_sp4to8_elmsz_b8_idxsz_b4_rpf_16(<8 x i32> %0, <16 x i32> %1) {
  // CHECK-NEXT:    %[[CDATA:[0-9]+]] = bitcast <16 x i32> %1 to <64 x i8>
  // CHECK-NEXT:    %[[RES:[0-9]+]] = call <128 x i8> @llvm.nvvm.spdecompress.v128i8.v8i32.v64i8(<8 x i32> %0, <64 x i8> %[[CDATA]], /* idx_size= */ i32 4, /* num_tgt= */ i32 8)
  // CHECK-NEXT:    %[[DATA:[0-9]+]] = bitcast <128 x i8> %[[RES]] to <32 x i32>
  // CHECK-NEXT:    ret <32 x i32> %[[DATA]]
  // CHECK-NEXT:  }
  %res = nvvm.spdecompress %metadata, %compdata
      comp_factor = sp4to8 index_size = b4 elem_size = b8 rep_factor = x16
      : vector<8xi32>, vector<16xi32> -> vector<32xi32>
  llvm.return %res : vector<32xi32>
}

// rpf=32: metadata=16, compressed_data=32, data=64
llvm.func @spdecompress_sp4to8_elmsz_b8_idxsz_b4_rpf_32(%metadata: vector<16xi32>, %compdata: vector<32xi32>) -> vector<64xi32> {
  // CHECK-LABEL: define <64 x i32> @spdecompress_sp4to8_elmsz_b8_idxsz_b4_rpf_32(<16 x i32> %0, <32 x i32> %1) {
  // CHECK-NEXT:    %[[CDATA:[0-9]+]] = bitcast <32 x i32> %1 to <128 x i8>
  // CHECK-NEXT:    %[[RES:[0-9]+]] = call <256 x i8> @llvm.nvvm.spdecompress.v256i8.v16i32.v128i8(<16 x i32> %0, <128 x i8> %[[CDATA]], /* idx_size= */ i32 4, /* num_tgt= */ i32 8)
  // CHECK-NEXT:    %[[DATA:[0-9]+]] = bitcast <256 x i8> %[[RES]] to <64 x i32>
  // CHECK-NEXT:    ret <64 x i32> %[[DATA]]
  // CHECK-NEXT:  }
  %res = nvvm.spdecompress %metadata, %compdata
      comp_factor = sp4to8 index_size = b4 elem_size = b8 rep_factor = x32
      : vector<16xi32>, vector<32xi32> -> vector<64xi32>
  llvm.return %res : vector<64xi32>
}

// rpf=64: metadata=32, compressed_data=64, data=128
llvm.func @spdecompress_sp4to8_elmsz_b8_idxsz_b4_rpf_64(%metadata: vector<32xi32>, %compdata: vector<64xi32>) -> vector<128xi32> {
  // CHECK-LABEL: define <128 x i32> @spdecompress_sp4to8_elmsz_b8_idxsz_b4_rpf_64(<32 x i32> %0, <64 x i32> %1) {
  // CHECK-NEXT:    %[[CDATA:[0-9]+]] = bitcast <64 x i32> %1 to <256 x i8>
  // CHECK-NEXT:    %[[RES:[0-9]+]] = call <512 x i8> @llvm.nvvm.spdecompress.v512i8.v32i32.v256i8(<32 x i32> %0, <256 x i8> %[[CDATA]], /* idx_size= */ i32 4, /* num_tgt= */ i32 8)
  // CHECK-NEXT:    %[[DATA:[0-9]+]] = bitcast <512 x i8> %[[RES]] to <128 x i32>
  // CHECK-NEXT:    ret <128 x i32> %[[DATA]]
  // CHECK-NEXT:  }
  %res = nvvm.spdecompress %metadata, %compdata
      comp_factor = sp4to8 index_size = b4 elem_size = b8 rep_factor = x64
      : vector<32xi32>, vector<64xi32> -> vector<128xi32>
  llvm.return %res : vector<128xi32>
}
