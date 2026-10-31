// RUN: mlir-translate -mlir-to-llvmir %s | FileCheck %s

// rpf=1: metadata=1 register(s)
llvm.func @spcompress_elmsz_b8_idxsz_b2_rpf_1(%spdesc: i32, %data: vector<8xi8>) -> i32 {
  // CHECK-LABEL: define i32 @spcompress_elmsz_b8_idxsz_b2_rpf_1(i32 %0, <8 x i8> %1) {
  // CHECK-NEXT:    %[[RES:[0-9]+]] = call { i32, <4 x i8> } @llvm.nvvm.spcompress.i32.v4i8.v8i8(<8 x i8> %1, i32 %0, /* idx_size= */ i32 2, /* num_tgt= */ i32 4)
  // CHECK-NEXT:    %[[MDATA:[0-9]+]] = extractvalue { i32, <4 x i8> } %[[RES]], 0
  // CHECK-NEXT:    %{{[0-9]+}} = extractvalue { i32, <4 x i8> } %[[RES]], 1
  // CHECK-NEXT:    ret i32 %[[MDATA]]
  // CHECK-NEXT:  }
  %metadata, %compressed_data = nvvm.sparse.compress %spdesc, %data idx_size = 2
      : i32, vector<8xi8> -> i32, vector<4xi8>
  llvm.return %metadata : i32
}

// rpf=2: metadata=1 register(s)
llvm.func @spcompress_elmsz_b8_idxsz_b2_rpf_2(%spdesc: i32, %data: vector<16xi8>) -> i32 {
  // CHECK-LABEL: define i32 @spcompress_elmsz_b8_idxsz_b2_rpf_2(i32 %0, <16 x i8> %1) {
  // CHECK-NEXT:    %[[RES:[0-9]+]] = call { i32, <8 x i8> } @llvm.nvvm.spcompress.i32.v8i8.v16i8(<16 x i8> %1, i32 %0, /* idx_size= */ i32 2, /* num_tgt= */ i32 4)
  // CHECK-NEXT:    %[[MDATA:[0-9]+]] = extractvalue { i32, <8 x i8> } %[[RES]], 0
  // CHECK-NEXT:    %{{[0-9]+}} = extractvalue { i32, <8 x i8> } %[[RES]], 1
  // CHECK-NEXT:    ret i32 %[[MDATA]]
  // CHECK-NEXT:  }
  %metadata, %compressed_data = nvvm.sparse.compress %spdesc, %data idx_size = 2
      : i32, vector<16xi8> -> i32, vector<8xi8>
  llvm.return %metadata : i32
}

// rpf=4: metadata=1 register(s)
llvm.func @spcompress_elmsz_b8_idxsz_b2_rpf_4(%spdesc: i32, %data: vector<32xi8>) -> i32 {
  // CHECK-LABEL: define i32 @spcompress_elmsz_b8_idxsz_b2_rpf_4(i32 %0, <32 x i8> %1) {
  // CHECK-NEXT:    %[[RES:[0-9]+]] = call { i32, <16 x i8> } @llvm.nvvm.spcompress.i32.v16i8.v32i8(<32 x i8> %1, i32 %0, /* idx_size= */ i32 2, /* num_tgt= */ i32 4)
  // CHECK-NEXT:    %[[MDATA:[0-9]+]] = extractvalue { i32, <16 x i8> } %[[RES]], 0
  // CHECK-NEXT:    %{{[0-9]+}} = extractvalue { i32, <16 x i8> } %[[RES]], 1
  // CHECK-NEXT:    ret i32 %[[MDATA]]
  // CHECK-NEXT:  }
  %metadata, %compressed_data = nvvm.sparse.compress %spdesc, %data idx_size = 2
      : i32, vector<32xi8> -> i32, vector<16xi8>
  llvm.return %metadata : i32
}

// rpf=8: metadata=2 register(s)
llvm.func @spcompress_elmsz_b8_idxsz_b2_rpf_8(%spdesc: i32, %data: vector<64xi8>) -> vector<2xi32> {
  // CHECK-LABEL: define <2 x i32> @spcompress_elmsz_b8_idxsz_b2_rpf_8(i32 %0, <64 x i8> %1) {
  // CHECK-NEXT:    %[[RES:[0-9]+]] = call { <2 x i32>, <32 x i8> } @llvm.nvvm.spcompress.v2i32.v32i8.v64i8(<64 x i8> %1, i32 %0, /* idx_size= */ i32 2, /* num_tgt= */ i32 4)
  // CHECK-NEXT:    %[[MDATA:[0-9]+]] = extractvalue { <2 x i32>, <32 x i8> } %[[RES]], 0
  // CHECK-NEXT:    %{{[0-9]+}} = extractvalue { <2 x i32>, <32 x i8> } %[[RES]], 1
  // CHECK-NEXT:    ret <2 x i32> %[[MDATA]]
  // CHECK-NEXT:  }
  %metadata, %compressed_data = nvvm.sparse.compress %spdesc, %data idx_size = 2
      : i32, vector<64xi8> -> vector<2xi32>, vector<32xi8>
  llvm.return %metadata : vector<2xi32>
}

// rpf=16: metadata=4 register(s)
llvm.func @spcompress_elmsz_b8_idxsz_b2_rpf_16(%spdesc: i32, %data: vector<128xi8>) -> vector<4xi32> {
  // CHECK-LABEL: define <4 x i32> @spcompress_elmsz_b8_idxsz_b2_rpf_16(i32 %0, <128 x i8> %1) {
  // CHECK-NEXT:    %[[RES:[0-9]+]] = call { <4 x i32>, <64 x i8> } @llvm.nvvm.spcompress.v4i32.v64i8.v128i8(<128 x i8> %1, i32 %0, /* idx_size= */ i32 2, /* num_tgt= */ i32 4)
  // CHECK-NEXT:    %[[MDATA:[0-9]+]] = extractvalue { <4 x i32>, <64 x i8> } %[[RES]], 0
  // CHECK-NEXT:    %{{[0-9]+}} = extractvalue { <4 x i32>, <64 x i8> } %[[RES]], 1
  // CHECK-NEXT:    ret <4 x i32> %[[MDATA]]
  // CHECK-NEXT:  }
  %metadata, %compressed_data = nvvm.sparse.compress %spdesc, %data idx_size = 2
      : i32, vector<128xi8> -> vector<4xi32>, vector<64xi8>
  llvm.return %metadata : vector<4xi32>
}

// rpf=32: metadata=8 register(s)
llvm.func @spcompress_elmsz_b8_idxsz_b2_rpf_32(%spdesc: i32, %data: vector<256xi8>) -> vector<8xi32> {
  // CHECK-LABEL: define <8 x i32> @spcompress_elmsz_b8_idxsz_b2_rpf_32(i32 %0, <256 x i8> %1) {
  // CHECK-NEXT:    %[[RES:[0-9]+]] = call { <8 x i32>, <128 x i8> } @llvm.nvvm.spcompress.v8i32.v128i8.v256i8(<256 x i8> %1, i32 %0, /* idx_size= */ i32 2, /* num_tgt= */ i32 4)
  // CHECK-NEXT:    %[[MDATA:[0-9]+]] = extractvalue { <8 x i32>, <128 x i8> } %[[RES]], 0
  // CHECK-NEXT:    %{{[0-9]+}} = extractvalue { <8 x i32>, <128 x i8> } %[[RES]], 1
  // CHECK-NEXT:    ret <8 x i32> %[[MDATA]]
  // CHECK-NEXT:  }
  %metadata, %compressed_data = nvvm.sparse.compress %spdesc, %data idx_size = 2
      : i32, vector<256xi8> -> vector<8xi32>, vector<128xi8>
  llvm.return %metadata : vector<8xi32>
}

// rpf=64: metadata=16 register(s)
llvm.func @spcompress_elmsz_b8_idxsz_b2_rpf_64(%spdesc: i32, %data: vector<512xi8>) -> vector<16xi32> {
  // CHECK-LABEL: define <16 x i32> @spcompress_elmsz_b8_idxsz_b2_rpf_64(i32 %0, <512 x i8> %1) {
  // CHECK-NEXT:    %[[RES:[0-9]+]] = call { <16 x i32>, <256 x i8> } @llvm.nvvm.spcompress.v16i32.v256i8.v512i8(<512 x i8> %1, i32 %0, /* idx_size= */ i32 2, /* num_tgt= */ i32 4)
  // CHECK-NEXT:    %[[MDATA:[0-9]+]] = extractvalue { <16 x i32>, <256 x i8> } %[[RES]], 0
  // CHECK-NEXT:    %{{[0-9]+}} = extractvalue { <16 x i32>, <256 x i8> } %[[RES]], 1
  // CHECK-NEXT:    ret <16 x i32> %[[MDATA]]
  // CHECK-NEXT:  }
  %metadata, %compressed_data = nvvm.sparse.compress %spdesc, %data idx_size = 2
      : i32, vector<512xi8> -> vector<16xi32>, vector<256xi8>
  llvm.return %metadata : vector<16xi32>
}

// rpf=1: metadata=1 register(s)
llvm.func @spcompress_elmsz_b8_idxsz_b4_rpf_1(%spdesc: i32, %data: vector<8xi8>) -> i32 {
  // CHECK-LABEL: define i32 @spcompress_elmsz_b8_idxsz_b4_rpf_1(i32 %0, <8 x i8> %1) {
  // CHECK-NEXT:    %[[RES:[0-9]+]] = call { i32, <4 x i8> } @llvm.nvvm.spcompress.i32.v4i8.v8i8(<8 x i8> %1, i32 %0, /* idx_size= */ i32 4, /* num_tgt= */ i32 4)
  // CHECK-NEXT:    %[[MDATA:[0-9]+]] = extractvalue { i32, <4 x i8> } %[[RES]], 0
  // CHECK-NEXT:    %{{[0-9]+}} = extractvalue { i32, <4 x i8> } %[[RES]], 1
  // CHECK-NEXT:    ret i32 %[[MDATA]]
  // CHECK-NEXT:  }
  %metadata, %compressed_data = nvvm.sparse.compress %spdesc, %data idx_size = 4
      : i32, vector<8xi8> -> i32, vector<4xi8>
  llvm.return %metadata : i32
}

// rpf=2: metadata=1 register(s)
llvm.func @spcompress_elmsz_b8_idxsz_b4_rpf_2(%spdesc: i32, %data: vector<16xi8>) -> i32 {
  // CHECK-LABEL: define i32 @spcompress_elmsz_b8_idxsz_b4_rpf_2(i32 %0, <16 x i8> %1) {
  // CHECK-NEXT:    %[[RES:[0-9]+]] = call { i32, <8 x i8> } @llvm.nvvm.spcompress.i32.v8i8.v16i8(<16 x i8> %1, i32 %0, /* idx_size= */ i32 4, /* num_tgt= */ i32 4)
  // CHECK-NEXT:    %[[MDATA:[0-9]+]] = extractvalue { i32, <8 x i8> } %[[RES]], 0
  // CHECK-NEXT:    %{{[0-9]+}} = extractvalue { i32, <8 x i8> } %[[RES]], 1
  // CHECK-NEXT:    ret i32 %[[MDATA]]
  // CHECK-NEXT:  }
  %metadata, %compressed_data = nvvm.sparse.compress %spdesc, %data idx_size = 4
      : i32, vector<16xi8> -> i32, vector<8xi8>
  llvm.return %metadata : i32
}

// rpf=4: metadata=2 register(s)
llvm.func @spcompress_elmsz_b8_idxsz_b4_rpf_4(%spdesc: i32, %data: vector<32xi8>) -> vector<2xi32> {
  // CHECK-LABEL: define <2 x i32> @spcompress_elmsz_b8_idxsz_b4_rpf_4(i32 %0, <32 x i8> %1) {
  // CHECK-NEXT:    %[[RES:[0-9]+]] = call { <2 x i32>, <16 x i8> } @llvm.nvvm.spcompress.v2i32.v16i8.v32i8(<32 x i8> %1, i32 %0, /* idx_size= */ i32 4, /* num_tgt= */ i32 4)
  // CHECK-NEXT:    %[[MDATA:[0-9]+]] = extractvalue { <2 x i32>, <16 x i8> } %[[RES]], 0
  // CHECK-NEXT:    %{{[0-9]+}} = extractvalue { <2 x i32>, <16 x i8> } %[[RES]], 1
  // CHECK-NEXT:    ret <2 x i32> %[[MDATA]]
  // CHECK-NEXT:  }
  %metadata, %compressed_data = nvvm.sparse.compress %spdesc, %data idx_size = 4
      : i32, vector<32xi8> -> vector<2xi32>, vector<16xi8>
  llvm.return %metadata : vector<2xi32>
}

// rpf=8: metadata=4 register(s)
llvm.func @spcompress_elmsz_b8_idxsz_b4_rpf_8(%spdesc: i32, %data: vector<64xi8>) -> vector<4xi32> {
  // CHECK-LABEL: define <4 x i32> @spcompress_elmsz_b8_idxsz_b4_rpf_8(i32 %0, <64 x i8> %1) {
  // CHECK-NEXT:    %[[RES:[0-9]+]] = call { <4 x i32>, <32 x i8> } @llvm.nvvm.spcompress.v4i32.v32i8.v64i8(<64 x i8> %1, i32 %0, /* idx_size= */ i32 4, /* num_tgt= */ i32 4)
  // CHECK-NEXT:    %[[MDATA:[0-9]+]] = extractvalue { <4 x i32>, <32 x i8> } %[[RES]], 0
  // CHECK-NEXT:    %{{[0-9]+}} = extractvalue { <4 x i32>, <32 x i8> } %[[RES]], 1
  // CHECK-NEXT:    ret <4 x i32> %[[MDATA]]
  // CHECK-NEXT:  }
  %metadata, %compressed_data = nvvm.sparse.compress %spdesc, %data idx_size = 4
      : i32, vector<64xi8> -> vector<4xi32>, vector<32xi8>
  llvm.return %metadata : vector<4xi32>
}

// rpf=16: metadata=8 register(s)
llvm.func @spcompress_elmsz_b8_idxsz_b4_rpf_16(%spdesc: i32, %data: vector<128xi8>) -> vector<8xi32> {
  // CHECK-LABEL: define <8 x i32> @spcompress_elmsz_b8_idxsz_b4_rpf_16(i32 %0, <128 x i8> %1) {
  // CHECK-NEXT:    %[[RES:[0-9]+]] = call { <8 x i32>, <64 x i8> } @llvm.nvvm.spcompress.v8i32.v64i8.v128i8(<128 x i8> %1, i32 %0, /* idx_size= */ i32 4, /* num_tgt= */ i32 4)
  // CHECK-NEXT:    %[[MDATA:[0-9]+]] = extractvalue { <8 x i32>, <64 x i8> } %[[RES]], 0
  // CHECK-NEXT:    %{{[0-9]+}} = extractvalue { <8 x i32>, <64 x i8> } %[[RES]], 1
  // CHECK-NEXT:    ret <8 x i32> %[[MDATA]]
  // CHECK-NEXT:  }
  %metadata, %compressed_data = nvvm.sparse.compress %spdesc, %data idx_size = 4
      : i32, vector<128xi8> -> vector<8xi32>, vector<64xi8>
  llvm.return %metadata : vector<8xi32>
}

// rpf=32: metadata=16 register(s)
llvm.func @spcompress_elmsz_b8_idxsz_b4_rpf_32(%spdesc: i32, %data: vector<256xi8>) -> vector<16xi32> {
  // CHECK-LABEL: define <16 x i32> @spcompress_elmsz_b8_idxsz_b4_rpf_32(i32 %0, <256 x i8> %1) {
  // CHECK-NEXT:    %[[RES:[0-9]+]] = call { <16 x i32>, <128 x i8> } @llvm.nvvm.spcompress.v16i32.v128i8.v256i8(<256 x i8> %1, i32 %0, /* idx_size= */ i32 4, /* num_tgt= */ i32 4)
  // CHECK-NEXT:    %[[MDATA:[0-9]+]] = extractvalue { <16 x i32>, <128 x i8> } %[[RES]], 0
  // CHECK-NEXT:    %{{[0-9]+}} = extractvalue { <16 x i32>, <128 x i8> } %[[RES]], 1
  // CHECK-NEXT:    ret <16 x i32> %[[MDATA]]
  // CHECK-NEXT:  }
  %metadata, %compressed_data = nvvm.sparse.compress %spdesc, %data idx_size = 4
      : i32, vector<256xi8> -> vector<16xi32>, vector<128xi8>
  llvm.return %metadata : vector<16xi32>
}

// rpf=64: metadata=32 register(s)
llvm.func @spcompress_elmsz_b8_idxsz_b4_rpf_64(%spdesc: i32, %data: vector<512xi8>) -> vector<32xi32> {
  // CHECK-LABEL: define <32 x i32> @spcompress_elmsz_b8_idxsz_b4_rpf_64(i32 %0, <512 x i8> %1) {
  // CHECK-NEXT:    %[[RES:[0-9]+]] = call { <32 x i32>, <256 x i8> } @llvm.nvvm.spcompress.v32i32.v256i8.v512i8(<512 x i8> %1, i32 %0, /* idx_size= */ i32 4, /* num_tgt= */ i32 4)
  // CHECK-NEXT:    %[[MDATA:[0-9]+]] = extractvalue { <32 x i32>, <256 x i8> } %[[RES]], 0
  // CHECK-NEXT:    %{{[0-9]+}} = extractvalue { <32 x i32>, <256 x i8> } %[[RES]], 1
  // CHECK-NEXT:    ret <32 x i32> %[[MDATA]]
  // CHECK-NEXT:  }
  %metadata, %compressed_data = nvvm.sparse.compress %spdesc, %data idx_size = 4
      : i32, vector<512xi8> -> vector<32xi32>, vector<256xi8>
  llvm.return %metadata : vector<32xi32>
}

// rpf=1: metadata=1 register(s)
llvm.func @spcompress_elmsz_b16_idxsz_b2_rpf_1(%spdesc: i32, %data: vector<4xi16>) -> i32 {
  // CHECK-LABEL: define i32 @spcompress_elmsz_b16_idxsz_b2_rpf_1(i32 %0, <4 x i16> %1) {
  // CHECK-NEXT:    %[[RES:[0-9]+]] = call { i32, <2 x i16> } @llvm.nvvm.spcompress.i32.v2i16.v4i16(<4 x i16> %1, i32 %0, /* idx_size= */ i32 2, /* num_tgt= */ i32 4)
  // CHECK-NEXT:    %[[MDATA:[0-9]+]] = extractvalue { i32, <2 x i16> } %[[RES]], 0
  // CHECK-NEXT:    %{{[0-9]+}} = extractvalue { i32, <2 x i16> } %[[RES]], 1
  // CHECK-NEXT:    ret i32 %[[MDATA]]
  // CHECK-NEXT:  }
  %metadata, %compressed_data = nvvm.sparse.compress %spdesc, %data idx_size = 2
      : i32, vector<4xi16> -> i32, vector<2xi16>
  llvm.return %metadata : i32
}

// rpf=2: metadata=1 register(s)
llvm.func @spcompress_elmsz_b16_idxsz_b2_rpf_2(%spdesc: i32, %data: vector<8xi16>) -> i32 {
  // CHECK-LABEL: define i32 @spcompress_elmsz_b16_idxsz_b2_rpf_2(i32 %0, <8 x i16> %1) {
  // CHECK-NEXT:    %[[RES:[0-9]+]] = call { i32, <4 x i16> } @llvm.nvvm.spcompress.i32.v4i16.v8i16(<8 x i16> %1, i32 %0, /* idx_size= */ i32 2, /* num_tgt= */ i32 4)
  // CHECK-NEXT:    %[[MDATA:[0-9]+]] = extractvalue { i32, <4 x i16> } %[[RES]], 0
  // CHECK-NEXT:    %{{[0-9]+}} = extractvalue { i32, <4 x i16> } %[[RES]], 1
  // CHECK-NEXT:    ret i32 %[[MDATA]]
  // CHECK-NEXT:  }
  %metadata, %compressed_data = nvvm.sparse.compress %spdesc, %data idx_size = 2
      : i32, vector<8xi16> -> i32, vector<4xi16>
  llvm.return %metadata : i32
}

// rpf=4: metadata=1 register(s)
llvm.func @spcompress_elmsz_b16_idxsz_b2_rpf_4(%spdesc: i32, %data: vector<16xi16>) -> i32 {
  // CHECK-LABEL: define i32 @spcompress_elmsz_b16_idxsz_b2_rpf_4(i32 %0, <16 x i16> %1) {
  // CHECK-NEXT:    %[[RES:[0-9]+]] = call { i32, <8 x i16> } @llvm.nvvm.spcompress.i32.v8i16.v16i16(<16 x i16> %1, i32 %0, /* idx_size= */ i32 2, /* num_tgt= */ i32 4)
  // CHECK-NEXT:    %[[MDATA:[0-9]+]] = extractvalue { i32, <8 x i16> } %[[RES]], 0
  // CHECK-NEXT:    %{{[0-9]+}} = extractvalue { i32, <8 x i16> } %[[RES]], 1
  // CHECK-NEXT:    ret i32 %[[MDATA]]
  // CHECK-NEXT:  }
  %metadata, %compressed_data = nvvm.sparse.compress %spdesc, %data idx_size = 2
      : i32, vector<16xi16> -> i32, vector<8xi16>
  llvm.return %metadata : i32
}

// rpf=8: metadata=1 register(s)
llvm.func @spcompress_elmsz_b16_idxsz_b2_rpf_8(%spdesc: i32, %data: vector<32xi16>) -> i32 {
  // CHECK-LABEL: define i32 @spcompress_elmsz_b16_idxsz_b2_rpf_8(i32 %0, <32 x i16> %1) {
  // CHECK-NEXT:    %[[RES:[0-9]+]] = call { i32, <16 x i16> } @llvm.nvvm.spcompress.i32.v16i16.v32i16(<32 x i16> %1, i32 %0, /* idx_size= */ i32 2, /* num_tgt= */ i32 4)
  // CHECK-NEXT:    %[[MDATA:[0-9]+]] = extractvalue { i32, <16 x i16> } %[[RES]], 0
  // CHECK-NEXT:    %{{[0-9]+}} = extractvalue { i32, <16 x i16> } %[[RES]], 1
  // CHECK-NEXT:    ret i32 %[[MDATA]]
  // CHECK-NEXT:  }
  %metadata, %compressed_data = nvvm.sparse.compress %spdesc, %data idx_size = 2
      : i32, vector<32xi16> -> i32, vector<16xi16>
  llvm.return %metadata : i32
}

// rpf=16: metadata=2 register(s)
llvm.func @spcompress_elmsz_b16_idxsz_b2_rpf_16(%spdesc: i32, %data: vector<64xi16>) -> vector<2xi32> {
  // CHECK-LABEL: define <2 x i32> @spcompress_elmsz_b16_idxsz_b2_rpf_16(i32 %0, <64 x i16> %1) {
  // CHECK-NEXT:    %[[RES:[0-9]+]] = call { <2 x i32>, <32 x i16> } @llvm.nvvm.spcompress.v2i32.v32i16.v64i16(<64 x i16> %1, i32 %0, /* idx_size= */ i32 2, /* num_tgt= */ i32 4)
  // CHECK-NEXT:    %[[MDATA:[0-9]+]] = extractvalue { <2 x i32>, <32 x i16> } %[[RES]], 0
  // CHECK-NEXT:    %{{[0-9]+}} = extractvalue { <2 x i32>, <32 x i16> } %[[RES]], 1
  // CHECK-NEXT:    ret <2 x i32> %[[MDATA]]
  // CHECK-NEXT:  }
  %metadata, %compressed_data = nvvm.sparse.compress %spdesc, %data idx_size = 2
      : i32, vector<64xi16> -> vector<2xi32>, vector<32xi16>
  llvm.return %metadata : vector<2xi32>
}

// rpf=32: metadata=4 register(s)
llvm.func @spcompress_elmsz_b16_idxsz_b2_rpf_32(%spdesc: i32, %data: vector<128xi16>) -> vector<4xi32> {
  // CHECK-LABEL: define <4 x i32> @spcompress_elmsz_b16_idxsz_b2_rpf_32(i32 %0, <128 x i16> %1) {
  // CHECK-NEXT:    %[[RES:[0-9]+]] = call { <4 x i32>, <64 x i16> } @llvm.nvvm.spcompress.v4i32.v64i16.v128i16(<128 x i16> %1, i32 %0, /* idx_size= */ i32 2, /* num_tgt= */ i32 4)
  // CHECK-NEXT:    %[[MDATA:[0-9]+]] = extractvalue { <4 x i32>, <64 x i16> } %[[RES]], 0
  // CHECK-NEXT:    %{{[0-9]+}} = extractvalue { <4 x i32>, <64 x i16> } %[[RES]], 1
  // CHECK-NEXT:    ret <4 x i32> %[[MDATA]]
  // CHECK-NEXT:  }
  %metadata, %compressed_data = nvvm.sparse.compress %spdesc, %data idx_size = 2
      : i32, vector<128xi16> -> vector<4xi32>, vector<64xi16>
  llvm.return %metadata : vector<4xi32>
}

// rpf=64: metadata=8 register(s)
llvm.func @spcompress_elmsz_b16_idxsz_b2_rpf_64(%spdesc: i32, %data: vector<256xi16>) -> vector<8xi32> {
  // CHECK-LABEL: define <8 x i32> @spcompress_elmsz_b16_idxsz_b2_rpf_64(i32 %0, <256 x i16> %1) {
  // CHECK-NEXT:    %[[RES:[0-9]+]] = call { <8 x i32>, <128 x i16> } @llvm.nvvm.spcompress.v8i32.v128i16.v256i16(<256 x i16> %1, i32 %0, /* idx_size= */ i32 2, /* num_tgt= */ i32 4)
  // CHECK-NEXT:    %[[MDATA:[0-9]+]] = extractvalue { <8 x i32>, <128 x i16> } %[[RES]], 0
  // CHECK-NEXT:    %{{[0-9]+}} = extractvalue { <8 x i32>, <128 x i16> } %[[RES]], 1
  // CHECK-NEXT:    ret <8 x i32> %[[MDATA]]
  // CHECK-NEXT:  }
  %metadata, %compressed_data = nvvm.sparse.compress %spdesc, %data idx_size = 2
      : i32, vector<256xi16> -> vector<8xi32>, vector<128xi16>
  llvm.return %metadata : vector<8xi32>
}

// rpf=1: metadata=1 register(s)
llvm.func @spcompress_elmsz_b16_idxsz_b4_rpf_1(%spdesc: i32, %data: vector<4xi16>) -> i32 {
  // CHECK-LABEL: define i32 @spcompress_elmsz_b16_idxsz_b4_rpf_1(i32 %0, <4 x i16> %1) {
  // CHECK-NEXT:    %[[RES:[0-9]+]] = call { i32, <2 x i16> } @llvm.nvvm.spcompress.i32.v2i16.v4i16(<4 x i16> %1, i32 %0, /* idx_size= */ i32 4, /* num_tgt= */ i32 4)
  // CHECK-NEXT:    %[[MDATA:[0-9]+]] = extractvalue { i32, <2 x i16> } %[[RES]], 0
  // CHECK-NEXT:    %{{[0-9]+}} = extractvalue { i32, <2 x i16> } %[[RES]], 1
  // CHECK-NEXT:    ret i32 %[[MDATA]]
  // CHECK-NEXT:  }
  %metadata, %compressed_data = nvvm.sparse.compress %spdesc, %data idx_size = 4
      : i32, vector<4xi16> -> i32, vector<2xi16>
  llvm.return %metadata : i32
}

// rpf=2: metadata=1 register(s)
llvm.func @spcompress_elmsz_b16_idxsz_b4_rpf_2(%spdesc: i32, %data: vector<8xi16>) -> i32 {
  // CHECK-LABEL: define i32 @spcompress_elmsz_b16_idxsz_b4_rpf_2(i32 %0, <8 x i16> %1) {
  // CHECK-NEXT:    %[[RES:[0-9]+]] = call { i32, <4 x i16> } @llvm.nvvm.spcompress.i32.v4i16.v8i16(<8 x i16> %1, i32 %0, /* idx_size= */ i32 4, /* num_tgt= */ i32 4)
  // CHECK-NEXT:    %[[MDATA:[0-9]+]] = extractvalue { i32, <4 x i16> } %[[RES]], 0
  // CHECK-NEXT:    %{{[0-9]+}} = extractvalue { i32, <4 x i16> } %[[RES]], 1
  // CHECK-NEXT:    ret i32 %[[MDATA]]
  // CHECK-NEXT:  }
  %metadata, %compressed_data = nvvm.sparse.compress %spdesc, %data idx_size = 4
      : i32, vector<8xi16> -> i32, vector<4xi16>
  llvm.return %metadata : i32
}

// rpf=4: metadata=1 register(s)
llvm.func @spcompress_elmsz_b16_idxsz_b4_rpf_4(%spdesc: i32, %data: vector<16xi16>) -> i32 {
  // CHECK-LABEL: define i32 @spcompress_elmsz_b16_idxsz_b4_rpf_4(i32 %0, <16 x i16> %1) {
  // CHECK-NEXT:    %[[RES:[0-9]+]] = call { i32, <8 x i16> } @llvm.nvvm.spcompress.i32.v8i16.v16i16(<16 x i16> %1, i32 %0, /* idx_size= */ i32 4, /* num_tgt= */ i32 4)
  // CHECK-NEXT:    %[[MDATA:[0-9]+]] = extractvalue { i32, <8 x i16> } %[[RES]], 0
  // CHECK-NEXT:    %{{[0-9]+}} = extractvalue { i32, <8 x i16> } %[[RES]], 1
  // CHECK-NEXT:    ret i32 %[[MDATA]]
  // CHECK-NEXT:  }
  %metadata, %compressed_data = nvvm.sparse.compress %spdesc, %data idx_size = 4
      : i32, vector<16xi16> -> i32, vector<8xi16>
  llvm.return %metadata : i32
}

// rpf=8: metadata=2 register(s)
llvm.func @spcompress_elmsz_b16_idxsz_b4_rpf_8(%spdesc: i32, %data: vector<32xi16>) -> vector<2xi32> {
  // CHECK-LABEL: define <2 x i32> @spcompress_elmsz_b16_idxsz_b4_rpf_8(i32 %0, <32 x i16> %1) {
  // CHECK-NEXT:    %[[RES:[0-9]+]] = call { <2 x i32>, <16 x i16> } @llvm.nvvm.spcompress.v2i32.v16i16.v32i16(<32 x i16> %1, i32 %0, /* idx_size= */ i32 4, /* num_tgt= */ i32 4)
  // CHECK-NEXT:    %[[MDATA:[0-9]+]] = extractvalue { <2 x i32>, <16 x i16> } %[[RES]], 0
  // CHECK-NEXT:    %{{[0-9]+}} = extractvalue { <2 x i32>, <16 x i16> } %[[RES]], 1
  // CHECK-NEXT:    ret <2 x i32> %[[MDATA]]
  // CHECK-NEXT:  }
  %metadata, %compressed_data = nvvm.sparse.compress %spdesc, %data idx_size = 4
      : i32, vector<32xi16> -> vector<2xi32>, vector<16xi16>
  llvm.return %metadata : vector<2xi32>
}

// rpf=16: metadata=4 register(s)
llvm.func @spcompress_elmsz_b16_idxsz_b4_rpf_16(%spdesc: i32, %data: vector<64xi16>) -> vector<4xi32> {
  // CHECK-LABEL: define <4 x i32> @spcompress_elmsz_b16_idxsz_b4_rpf_16(i32 %0, <64 x i16> %1) {
  // CHECK-NEXT:    %[[RES:[0-9]+]] = call { <4 x i32>, <32 x i16> } @llvm.nvvm.spcompress.v4i32.v32i16.v64i16(<64 x i16> %1, i32 %0, /* idx_size= */ i32 4, /* num_tgt= */ i32 4)
  // CHECK-NEXT:    %[[MDATA:[0-9]+]] = extractvalue { <4 x i32>, <32 x i16> } %[[RES]], 0
  // CHECK-NEXT:    %{{[0-9]+}} = extractvalue { <4 x i32>, <32 x i16> } %[[RES]], 1
  // CHECK-NEXT:    ret <4 x i32> %[[MDATA]]
  // CHECK-NEXT:  }
  %metadata, %compressed_data = nvvm.sparse.compress %spdesc, %data idx_size = 4
      : i32, vector<64xi16> -> vector<4xi32>, vector<32xi16>
  llvm.return %metadata : vector<4xi32>
}

// rpf=32: metadata=8 register(s)
llvm.func @spcompress_elmsz_b16_idxsz_b4_rpf_32(%spdesc: i32, %data: vector<128xi16>) -> vector<8xi32> {
  // CHECK-LABEL: define <8 x i32> @spcompress_elmsz_b16_idxsz_b4_rpf_32(i32 %0, <128 x i16> %1) {
  // CHECK-NEXT:    %[[RES:[0-9]+]] = call { <8 x i32>, <64 x i16> } @llvm.nvvm.spcompress.v8i32.v64i16.v128i16(<128 x i16> %1, i32 %0, /* idx_size= */ i32 4, /* num_tgt= */ i32 4)
  // CHECK-NEXT:    %[[MDATA:[0-9]+]] = extractvalue { <8 x i32>, <64 x i16> } %[[RES]], 0
  // CHECK-NEXT:    %{{[0-9]+}} = extractvalue { <8 x i32>, <64 x i16> } %[[RES]], 1
  // CHECK-NEXT:    ret <8 x i32> %[[MDATA]]
  // CHECK-NEXT:  }
  %metadata, %compressed_data = nvvm.sparse.compress %spdesc, %data idx_size = 4
      : i32, vector<128xi16> -> vector<8xi32>, vector<64xi16>
  llvm.return %metadata : vector<8xi32>
}

// rpf=64: metadata=16 register(s)
llvm.func @spcompress_elmsz_b16_idxsz_b4_rpf_64(%spdesc: i32, %data: vector<256xi16>) -> vector<16xi32> {
  // CHECK-LABEL: define <16 x i32> @spcompress_elmsz_b16_idxsz_b4_rpf_64(i32 %0, <256 x i16> %1) {
  // CHECK-NEXT:    %[[RES:[0-9]+]] = call { <16 x i32>, <128 x i16> } @llvm.nvvm.spcompress.v16i32.v128i16.v256i16(<256 x i16> %1, i32 %0, /* idx_size= */ i32 4, /* num_tgt= */ i32 4)
  // CHECK-NEXT:    %[[MDATA:[0-9]+]] = extractvalue { <16 x i32>, <128 x i16> } %[[RES]], 0
  // CHECK-NEXT:    %{{[0-9]+}} = extractvalue { <16 x i32>, <128 x i16> } %[[RES]], 1
  // CHECK-NEXT:    ret <16 x i32> %[[MDATA]]
  // CHECK-NEXT:  }
  %metadata, %compressed_data = nvvm.sparse.compress %spdesc, %data idx_size = 4
      : i32, vector<256xi16> -> vector<16xi32>, vector<128xi16>
  llvm.return %metadata : vector<16xi32>
}

// rpf=1: metadata=1 register(s), returned as vector<1xi32>
llvm.func @spcompress_elmsz_b8_idxsz_b2_rpf_1_vec1(%spdesc: i32, %data: vector<8xi8>) -> vector<1xi32> {
  // CHECK-LABEL: define <1 x i32> @spcompress_elmsz_b8_idxsz_b2_rpf_1_vec1(i32 %0, <8 x i8> %1) {
  // CHECK-NEXT:    %[[RES:[0-9]+]] = call { i32, <4 x i8> } @llvm.nvvm.spcompress.i32.v4i8.v8i8(<8 x i8> %1, i32 %0, /* idx_size= */ i32 2, /* num_tgt= */ i32 4)
  // CHECK-NEXT:    %[[MDATA_I32:[0-9]+]] = extractvalue { i32, <4 x i8> } %[[RES]], 0
  // CHECK-NEXT:    %[[MDATA:[0-9]+]] = bitcast i32 %[[MDATA_I32]] to <1 x i32>
  // CHECK-NEXT:    %{{[0-9]+}} = extractvalue { i32, <4 x i8> } %[[RES]], 1
  // CHECK-NEXT:    ret <1 x i32> %[[MDATA]]
  // CHECK-NEXT:  }
  %metadata, %compressed_data = nvvm.sparse.compress %spdesc, %data idx_size = 2
      : i32, vector<8xi8> -> vector<1xi32>, vector<4xi8>
  llvm.return %metadata : vector<1xi32>
}
