// RUN: mlir-translate -mlir-to-llvmir %s | FileCheck %s

// rpf=1: metadata=1 register(s)
llvm.func @spdecompress_sp2to16_elmsz_b8_idxsz_b4_rpf_1(%metadata: i32, %compdata: vector<2xi8>) -> vector<16xi8> {
  // CHECK-LABEL: define <16 x i8> @spdecompress_sp2to16_elmsz_b8_idxsz_b4_rpf_1(i32 %0, <2 x i8> %1) {
  // CHECK-NEXT:    %[[RES:[0-9]+]] = call <16 x i8> @llvm.nvvm.spdecompress.v16i8.i32.v2i8(i32 %0, <2 x i8> %1, /* idx_size= */ i32 4, /* num_tgt= */ i32 16)
  // CHECK-NEXT:    ret <16 x i8> %[[RES]]
  // CHECK-NEXT:  }
  %res = nvvm.sparse.decompress %metadata, %compdata idx_size = 4 num_tgt = 16
      : i32, vector<2xi8> -> vector<16xi8>
  llvm.return %res : vector<16xi8>
}

// rpf=2: metadata=1 register(s)
llvm.func @spdecompress_sp2to16_elmsz_b8_idxsz_b4_rpf_2(%metadata: i32, %compdata: vector<4xi8>) -> vector<32xi8> {
  // CHECK-LABEL: define <32 x i8> @spdecompress_sp2to16_elmsz_b8_idxsz_b4_rpf_2(i32 %0, <4 x i8> %1) {
  // CHECK-NEXT:    %[[RES:[0-9]+]] = call <32 x i8> @llvm.nvvm.spdecompress.v32i8.i32.v4i8(i32 %0, <4 x i8> %1, /* idx_size= */ i32 4, /* num_tgt= */ i32 16)
  // CHECK-NEXT:    ret <32 x i8> %[[RES]]
  // CHECK-NEXT:  }
  %res = nvvm.sparse.decompress %metadata, %compdata idx_size = 4 num_tgt = 16
      : i32, vector<4xi8> -> vector<32xi8>
  llvm.return %res : vector<32xi8>
}

// rpf=4: metadata=1 register(s)
llvm.func @spdecompress_sp2to16_elmsz_b8_idxsz_b4_rpf_4(%metadata: i32, %compdata: vector<8xi8>) -> vector<64xi8> {
  // CHECK-LABEL: define <64 x i8> @spdecompress_sp2to16_elmsz_b8_idxsz_b4_rpf_4(i32 %0, <8 x i8> %1) {
  // CHECK-NEXT:    %[[RES:[0-9]+]] = call <64 x i8> @llvm.nvvm.spdecompress.v64i8.i32.v8i8(i32 %0, <8 x i8> %1, /* idx_size= */ i32 4, /* num_tgt= */ i32 16)
  // CHECK-NEXT:    ret <64 x i8> %[[RES]]
  // CHECK-NEXT:  }
  %res = nvvm.sparse.decompress %metadata, %compdata idx_size = 4 num_tgt = 16
      : i32, vector<8xi8> -> vector<64xi8>
  llvm.return %res : vector<64xi8>
}

// rpf=8: metadata=2 register(s)
llvm.func @spdecompress_sp2to16_elmsz_b8_idxsz_b4_rpf_8(%metadata: vector<2xi32>, %compdata: vector<16xi8>) -> vector<128xi8> {
  // CHECK-LABEL: define <128 x i8> @spdecompress_sp2to16_elmsz_b8_idxsz_b4_rpf_8(<2 x i32> %0, <16 x i8> %1) {
  // CHECK-NEXT:    %[[RES:[0-9]+]] = call <128 x i8> @llvm.nvvm.spdecompress.v128i8.v2i32.v16i8(<2 x i32> %0, <16 x i8> %1, /* idx_size= */ i32 4, /* num_tgt= */ i32 16)
  // CHECK-NEXT:    ret <128 x i8> %[[RES]]
  // CHECK-NEXT:  }
  %res = nvvm.sparse.decompress %metadata, %compdata idx_size = 4 num_tgt = 16
      : vector<2xi32>, vector<16xi8> -> vector<128xi8>
  llvm.return %res : vector<128xi8>
}

// rpf=16: metadata=4 register(s)
llvm.func @spdecompress_sp2to16_elmsz_b8_idxsz_b4_rpf_16(%metadata: vector<4xi32>, %compdata: vector<32xi8>) -> vector<256xi8> {
  // CHECK-LABEL: define <256 x i8> @spdecompress_sp2to16_elmsz_b8_idxsz_b4_rpf_16(<4 x i32> %0, <32 x i8> %1) {
  // CHECK-NEXT:    %[[RES:[0-9]+]] = call <256 x i8> @llvm.nvvm.spdecompress.v256i8.v4i32.v32i8(<4 x i32> %0, <32 x i8> %1, /* idx_size= */ i32 4, /* num_tgt= */ i32 16)
  // CHECK-NEXT:    ret <256 x i8> %[[RES]]
  // CHECK-NEXT:  }
  %res = nvvm.sparse.decompress %metadata, %compdata idx_size = 4 num_tgt = 16
      : vector<4xi32>, vector<32xi8> -> vector<256xi8>
  llvm.return %res : vector<256xi8>
}

// rpf=32: metadata=8 register(s)
llvm.func @spdecompress_sp2to16_elmsz_b8_idxsz_b4_rpf_32(%metadata: vector<8xi32>, %compdata: vector<64xi8>) -> vector<512xi8> {
  // CHECK-LABEL: define <512 x i8> @spdecompress_sp2to16_elmsz_b8_idxsz_b4_rpf_32(<8 x i32> %0, <64 x i8> %1) {
  // CHECK-NEXT:    %[[RES:[0-9]+]] = call <512 x i8> @llvm.nvvm.spdecompress.v512i8.v8i32.v64i8(<8 x i32> %0, <64 x i8> %1, /* idx_size= */ i32 4, /* num_tgt= */ i32 16)
  // CHECK-NEXT:    ret <512 x i8> %[[RES]]
  // CHECK-NEXT:  }
  %res = nvvm.sparse.decompress %metadata, %compdata idx_size = 4 num_tgt = 16
      : vector<8xi32>, vector<64xi8> -> vector<512xi8>
  llvm.return %res : vector<512xi8>
}

// rpf=1: metadata=1 register(s)
llvm.func @spdecompress_sp2to16_elmsz_b16_idxsz_b4_rpf_1(%metadata: i32, %compdata: vector<2xi16>) -> vector<16xi16> {
  // CHECK-LABEL: define <16 x i16> @spdecompress_sp2to16_elmsz_b16_idxsz_b4_rpf_1(i32 %0, <2 x i16> %1) {
  // CHECK-NEXT:    %[[RES:[0-9]+]] = call <16 x i16> @llvm.nvvm.spdecompress.v16i16.i32.v2i16(i32 %0, <2 x i16> %1, /* idx_size= */ i32 4, /* num_tgt= */ i32 16)
  // CHECK-NEXT:    ret <16 x i16> %[[RES]]
  // CHECK-NEXT:  }
  %res = nvvm.sparse.decompress %metadata, %compdata idx_size = 4 num_tgt = 16
      : i32, vector<2xi16> -> vector<16xi16>
  llvm.return %res : vector<16xi16>
}

// rpf=2: metadata=1 register(s)
llvm.func @spdecompress_sp2to16_elmsz_b16_idxsz_b4_rpf_2(%metadata: i32, %compdata: vector<4xi16>) -> vector<32xi16> {
  // CHECK-LABEL: define <32 x i16> @spdecompress_sp2to16_elmsz_b16_idxsz_b4_rpf_2(i32 %0, <4 x i16> %1) {
  // CHECK-NEXT:    %[[RES:[0-9]+]] = call <32 x i16> @llvm.nvvm.spdecompress.v32i16.i32.v4i16(i32 %0, <4 x i16> %1, /* idx_size= */ i32 4, /* num_tgt= */ i32 16)
  // CHECK-NEXT:    ret <32 x i16> %[[RES]]
  // CHECK-NEXT:  }
  %res = nvvm.sparse.decompress %metadata, %compdata idx_size = 4 num_tgt = 16
      : i32, vector<4xi16> -> vector<32xi16>
  llvm.return %res : vector<32xi16>
}

// rpf=4: metadata=1 register(s)
llvm.func @spdecompress_sp2to16_elmsz_b16_idxsz_b4_rpf_4(%metadata: i32, %compdata: vector<8xi16>) -> vector<64xi16> {
  // CHECK-LABEL: define <64 x i16> @spdecompress_sp2to16_elmsz_b16_idxsz_b4_rpf_4(i32 %0, <8 x i16> %1) {
  // CHECK-NEXT:    %[[RES:[0-9]+]] = call <64 x i16> @llvm.nvvm.spdecompress.v64i16.i32.v8i16(i32 %0, <8 x i16> %1, /* idx_size= */ i32 4, /* num_tgt= */ i32 16)
  // CHECK-NEXT:    ret <64 x i16> %[[RES]]
  // CHECK-NEXT:  }
  %res = nvvm.sparse.decompress %metadata, %compdata idx_size = 4 num_tgt = 16
      : i32, vector<8xi16> -> vector<64xi16>
  llvm.return %res : vector<64xi16>
}

// rpf=8: metadata=2 register(s)
llvm.func @spdecompress_sp2to16_elmsz_b16_idxsz_b4_rpf_8(%metadata: vector<2xi32>, %compdata: vector<16xi16>) -> vector<128xi16> {
  // CHECK-LABEL: define <128 x i16> @spdecompress_sp2to16_elmsz_b16_idxsz_b4_rpf_8(<2 x i32> %0, <16 x i16> %1) {
  // CHECK-NEXT:    %[[RES:[0-9]+]] = call <128 x i16> @llvm.nvvm.spdecompress.v128i16.v2i32.v16i16(<2 x i32> %0, <16 x i16> %1, /* idx_size= */ i32 4, /* num_tgt= */ i32 16)
  // CHECK-NEXT:    ret <128 x i16> %[[RES]]
  // CHECK-NEXT:  }
  %res = nvvm.sparse.decompress %metadata, %compdata idx_size = 4 num_tgt = 16
      : vector<2xi32>, vector<16xi16> -> vector<128xi16>
  llvm.return %res : vector<128xi16>
}

// rpf=16: metadata=4 register(s)
llvm.func @spdecompress_sp2to16_elmsz_b16_idxsz_b4_rpf_16(%metadata: vector<4xi32>, %compdata: vector<32xi16>) -> vector<256xi16> {
  // CHECK-LABEL: define <256 x i16> @spdecompress_sp2to16_elmsz_b16_idxsz_b4_rpf_16(<4 x i32> %0, <32 x i16> %1) {
  // CHECK-NEXT:    %[[RES:[0-9]+]] = call <256 x i16> @llvm.nvvm.spdecompress.v256i16.v4i32.v32i16(<4 x i32> %0, <32 x i16> %1, /* idx_size= */ i32 4, /* num_tgt= */ i32 16)
  // CHECK-NEXT:    ret <256 x i16> %[[RES]]
  // CHECK-NEXT:  }
  %res = nvvm.sparse.decompress %metadata, %compdata idx_size = 4 num_tgt = 16
      : vector<4xi32>, vector<32xi16> -> vector<256xi16>
  llvm.return %res : vector<256xi16>
}
