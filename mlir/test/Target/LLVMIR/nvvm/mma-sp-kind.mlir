// RUN: mlir-translate --mlir-to-llvmir %s | FileCheck %s

// CHECK-LABEL: @nvvm_mma_sp_kind_m16n8k64_e4m3_f16
llvm.func @nvvm_mma_sp_kind_m16n8k64_e4m3_f16(
    %a0: i32, %a1: i32, %a2: i32, %a3: i32,
    %b0: i32, %b1: i32, %b2: i32, %b3: i32,
    %c0: vector<2xf16>, %c1: vector<2xf16>,
    %meta: i32) -> !llvm.struct<(vector<2xf16>, vector<2xf16>)> {
  %sel = llvm.mlir.constant(0 : i32) : i32
  // CHECK: call { <2 x half>, <2 x half> } @llvm.nvvm.mma.sp.ordered.metadata.m16n8k64.row.col.kind.f8f6f4.f16.e4m3.e4m3.f16(i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, <2 x half> {{%[0-9]+}}, <2 x half> {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 0)
  %res = nvvm.mma.sp.sync A[%a0, %a1, %a2, %a3] B[%b0, %b1, %b2, %b3] C[%c0, %c1]
      sparseMetadata[%meta] selector[%sel]
      {kind = #nvvm.mma_kind<f8f6f4>,
       multiplicandAPtxType = #nvvm.mma_type<e4m3>,
       multiplicandBPtxType = #nvvm.mma_type<e4m3>,
       orderedMetadata,
       shape = #nvvm.shape<m = 16, n = 8, k = 64>}
      : (i32, i32, vector<2xf16>) -> !llvm.struct<(vector<2xf16>, vector<2xf16>)>
  llvm.return %res : !llvm.struct<(vector<2xf16>, vector<2xf16>)>
}

// CHECK-LABEL: @nvvm_mma_sp_kind_m16n8k64_e4m3_f32
llvm.func @nvvm_mma_sp_kind_m16n8k64_e4m3_f32(
    %a0: i32, %a1: i32, %a2: i32, %a3: i32,
    %b0: i32, %b1: i32, %b2: i32, %b3: i32,
    %c0: f32, %c1: f32, %c2: f32, %c3: f32,
    %meta: i32) -> !llvm.struct<(f32, f32, f32, f32)> {
  %sel = llvm.mlir.constant(0 : i32) : i32
  // CHECK: call { float, float, float, float } @llvm.nvvm.mma.sp.ordered.metadata.m16n8k64.row.col.f32.e4m3.e4m3.f32(i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, float {{%[0-9]+}}, float {{%[0-9]+}}, float {{%[0-9]+}}, float {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 0)
  %res = nvvm.mma.sp.sync A[%a0, %a1, %a2, %a3] B[%b0, %b1, %b2, %b3] C[%c0, %c1, %c2, %c3]
      sparseMetadata[%meta] selector[%sel]
      {kind = #nvvm.mma_kind<f8f6f4>,
       multiplicandAPtxType = #nvvm.mma_type<e4m3>,
       multiplicandBPtxType = #nvvm.mma_type<e4m3>,
       orderedMetadata,
       shape = #nvvm.shape<m = 16, n = 8, k = 64>}
      : (i32, i32, f32) -> !llvm.struct<(f32, f32, f32, f32)>
  llvm.return %res : !llvm.struct<(f32, f32, f32, f32)>
}

// CHECK-LABEL: @nvvm_mma_sp_kind_m16n8k64_e5m2_f16
llvm.func @nvvm_mma_sp_kind_m16n8k64_e5m2_f16(
    %a0: i32, %a1: i32, %a2: i32, %a3: i32,
    %b0: i32, %b1: i32, %b2: i32, %b3: i32,
    %c0: vector<2xf16>, %c1: vector<2xf16>,
    %meta: i32) -> !llvm.struct<(vector<2xf16>, vector<2xf16>)> {
  %sel = llvm.mlir.constant(0 : i32) : i32
  // CHECK: call { <2 x half>, <2 x half> } @llvm.nvvm.mma.sp.ordered.metadata.m16n8k64.row.col.kind.f8f6f4.f16.e5m2.e5m2.f16(i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, <2 x half> {{%[0-9]+}}, <2 x half> {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 0)
  %res = nvvm.mma.sp.sync A[%a0, %a1, %a2, %a3] B[%b0, %b1, %b2, %b3] C[%c0, %c1]
      sparseMetadata[%meta] selector[%sel]
      {kind = #nvvm.mma_kind<f8f6f4>,
       multiplicandAPtxType = #nvvm.mma_type<e5m2>,
       multiplicandBPtxType = #nvvm.mma_type<e5m2>,
       orderedMetadata,
       shape = #nvvm.shape<m = 16, n = 8, k = 64>}
      : (i32, i32, vector<2xf16>) -> !llvm.struct<(vector<2xf16>, vector<2xf16>)>
  llvm.return %res : !llvm.struct<(vector<2xf16>, vector<2xf16>)>
}

// CHECK-LABEL: @nvvm_mma_sp_kind_m16n8k64_e5m2_f32
llvm.func @nvvm_mma_sp_kind_m16n8k64_e5m2_f32(
    %a0: i32, %a1: i32, %a2: i32, %a3: i32,
    %b0: i32, %b1: i32, %b2: i32, %b3: i32,
    %c0: f32, %c1: f32, %c2: f32, %c3: f32,
    %meta: i32) -> !llvm.struct<(f32, f32, f32, f32)> {
  %sel = llvm.mlir.constant(0 : i32) : i32
  // CHECK: call { float, float, float, float } @llvm.nvvm.mma.sp.ordered.metadata.m16n8k64.row.col.f32.e5m2.e5m2.f32(i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, float {{%[0-9]+}}, float {{%[0-9]+}}, float {{%[0-9]+}}, float {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 0)
  %res = nvvm.mma.sp.sync A[%a0, %a1, %a2, %a3] B[%b0, %b1, %b2, %b3] C[%c0, %c1, %c2, %c3]
      sparseMetadata[%meta] selector[%sel]
      {kind = #nvvm.mma_kind<f8f6f4>,
       multiplicandAPtxType = #nvvm.mma_type<e5m2>,
       multiplicandBPtxType = #nvvm.mma_type<e5m2>,
       orderedMetadata,
       shape = #nvvm.shape<m = 16, n = 8, k = 64>}
      : (i32, i32, f32) -> !llvm.struct<(f32, f32, f32, f32)>
  llvm.return %res : !llvm.struct<(f32, f32, f32, f32)>
}

// CHECK-LABEL: @nvvm_mma_sp_kind_m16n8k64_e3m2_f16
llvm.func @nvvm_mma_sp_kind_m16n8k64_e3m2_f16(
    %a0: i32, %a1: i32, %a2: i32, %a3: i32,
    %b0: i32, %b1: i32, %b2: i32, %b3: i32,
    %c0: vector<2xf16>, %c1: vector<2xf16>,
    %meta: i32) -> !llvm.struct<(vector<2xf16>, vector<2xf16>)> {
  %sel = llvm.mlir.constant(0 : i32) : i32
  // CHECK: call { <2 x half>, <2 x half> } @llvm.nvvm.mma.sp.ordered.metadata.m16n8k64.row.col.kind.f8f6f4.f16.e3m2.e3m2.f16(i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, <2 x half> {{%[0-9]+}}, <2 x half> {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 0)
  %res = nvvm.mma.sp.sync A[%a0, %a1, %a2, %a3] B[%b0, %b1, %b2, %b3] C[%c0, %c1]
      sparseMetadata[%meta] selector[%sel]
      {kind = #nvvm.mma_kind<f8f6f4>,
       multiplicandAPtxType = #nvvm.mma_type<e3m2>,
       multiplicandBPtxType = #nvvm.mma_type<e3m2>,
       orderedMetadata,
       shape = #nvvm.shape<m = 16, n = 8, k = 64>}
      : (i32, i32, vector<2xf16>) -> !llvm.struct<(vector<2xf16>, vector<2xf16>)>
  llvm.return %res : !llvm.struct<(vector<2xf16>, vector<2xf16>)>
}

// CHECK-LABEL: @nvvm_mma_sp_kind_m16n8k64_e3m2_f32
llvm.func @nvvm_mma_sp_kind_m16n8k64_e3m2_f32(
    %a0: i32, %a1: i32, %a2: i32, %a3: i32,
    %b0: i32, %b1: i32, %b2: i32, %b3: i32,
    %c0: f32, %c1: f32, %c2: f32, %c3: f32,
    %meta: i32) -> !llvm.struct<(f32, f32, f32, f32)> {
  %sel = llvm.mlir.constant(0 : i32) : i32
  // CHECK: call { float, float, float, float } @llvm.nvvm.mma.sp.ordered.metadata.m16n8k64.row.col.kind.f8f6f4.f32.e3m2.e3m2.f32(i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, float {{%[0-9]+}}, float {{%[0-9]+}}, float {{%[0-9]+}}, float {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 0)
  %res = nvvm.mma.sp.sync A[%a0, %a1, %a2, %a3] B[%b0, %b1, %b2, %b3] C[%c0, %c1, %c2, %c3]
      sparseMetadata[%meta] selector[%sel]
      {kind = #nvvm.mma_kind<f8f6f4>,
       multiplicandAPtxType = #nvvm.mma_type<e3m2>,
       multiplicandBPtxType = #nvvm.mma_type<e3m2>,
       orderedMetadata,
       shape = #nvvm.shape<m = 16, n = 8, k = 64>}
      : (i32, i32, f32) -> !llvm.struct<(f32, f32, f32, f32)>
  llvm.return %res : !llvm.struct<(f32, f32, f32, f32)>
}

// CHECK-LABEL: @nvvm_mma_sp_kind_m16n8k64_e2m3_f16
llvm.func @nvvm_mma_sp_kind_m16n8k64_e2m3_f16(
    %a0: i32, %a1: i32, %a2: i32, %a3: i32,
    %b0: i32, %b1: i32, %b2: i32, %b3: i32,
    %c0: vector<2xf16>, %c1: vector<2xf16>,
    %meta: i32) -> !llvm.struct<(vector<2xf16>, vector<2xf16>)> {
  %sel = llvm.mlir.constant(0 : i32) : i32
  // CHECK: call { <2 x half>, <2 x half> } @llvm.nvvm.mma.sp.ordered.metadata.m16n8k64.row.col.kind.f8f6f4.f16.e2m3.e2m3.f16(i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, <2 x half> {{%[0-9]+}}, <2 x half> {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 0)
  %res = nvvm.mma.sp.sync A[%a0, %a1, %a2, %a3] B[%b0, %b1, %b2, %b3] C[%c0, %c1]
      sparseMetadata[%meta] selector[%sel]
      {kind = #nvvm.mma_kind<f8f6f4>,
       multiplicandAPtxType = #nvvm.mma_type<e2m3>,
       multiplicandBPtxType = #nvvm.mma_type<e2m3>,
       orderedMetadata,
       shape = #nvvm.shape<m = 16, n = 8, k = 64>}
      : (i32, i32, vector<2xf16>) -> !llvm.struct<(vector<2xf16>, vector<2xf16>)>
  llvm.return %res : !llvm.struct<(vector<2xf16>, vector<2xf16>)>
}

// CHECK-LABEL: @nvvm_mma_sp_kind_m16n8k64_e2m3_f32
llvm.func @nvvm_mma_sp_kind_m16n8k64_e2m3_f32(
    %a0: i32, %a1: i32, %a2: i32, %a3: i32,
    %b0: i32, %b1: i32, %b2: i32, %b3: i32,
    %c0: f32, %c1: f32, %c2: f32, %c3: f32,
    %meta: i32) -> !llvm.struct<(f32, f32, f32, f32)> {
  %sel = llvm.mlir.constant(0 : i32) : i32
  // CHECK: call { float, float, float, float } @llvm.nvvm.mma.sp.ordered.metadata.m16n8k64.row.col.kind.f8f6f4.f32.e2m3.e2m3.f32(i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, float {{%[0-9]+}}, float {{%[0-9]+}}, float {{%[0-9]+}}, float {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 0)
  %res = nvvm.mma.sp.sync A[%a0, %a1, %a2, %a3] B[%b0, %b1, %b2, %b3] C[%c0, %c1, %c2, %c3]
      sparseMetadata[%meta] selector[%sel]
      {kind = #nvvm.mma_kind<f8f6f4>,
       multiplicandAPtxType = #nvvm.mma_type<e2m3>,
       multiplicandBPtxType = #nvvm.mma_type<e2m3>,
       orderedMetadata,
       shape = #nvvm.shape<m = 16, n = 8, k = 64>}
      : (i32, i32, f32) -> !llvm.struct<(f32, f32, f32, f32)>
  llvm.return %res : !llvm.struct<(f32, f32, f32, f32)>
}

// CHECK-LABEL: @nvvm_mma_sp_kind_m16n8k64_e2m1_f16
llvm.func @nvvm_mma_sp_kind_m16n8k64_e2m1_f16(
    %a0: i32, %a1: i32, %a2: i32, %a3: i32,
    %b0: i32, %b1: i32, %b2: i32, %b3: i32,
    %c0: vector<2xf16>, %c1: vector<2xf16>,
    %meta: i32) -> !llvm.struct<(vector<2xf16>, vector<2xf16>)> {
  %sel = llvm.mlir.constant(0 : i32) : i32
  // CHECK: call { <2 x half>, <2 x half> } @llvm.nvvm.mma.sp.ordered.metadata.m16n8k64.row.col.kind.f8f6f4.f16.e2m1.e2m1.f16(i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, <2 x half> {{%[0-9]+}}, <2 x half> {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 0)
  %res = nvvm.mma.sp.sync A[%a0, %a1, %a2, %a3] B[%b0, %b1, %b2, %b3] C[%c0, %c1]
      sparseMetadata[%meta] selector[%sel]
      {kind = #nvvm.mma_kind<f8f6f4>,
       multiplicandAPtxType = #nvvm.mma_type<e2m1>,
       multiplicandBPtxType = #nvvm.mma_type<e2m1>,
       orderedMetadata,
       shape = #nvvm.shape<m = 16, n = 8, k = 64>}
      : (i32, i32, vector<2xf16>) -> !llvm.struct<(vector<2xf16>, vector<2xf16>)>
  llvm.return %res : !llvm.struct<(vector<2xf16>, vector<2xf16>)>
}

// CHECK-LABEL: @nvvm_mma_sp_kind_m16n8k64_e2m1_f32
llvm.func @nvvm_mma_sp_kind_m16n8k64_e2m1_f32(
    %a0: i32, %a1: i32, %a2: i32, %a3: i32,
    %b0: i32, %b1: i32, %b2: i32, %b3: i32,
    %c0: f32, %c1: f32, %c2: f32, %c3: f32,
    %meta: i32) -> !llvm.struct<(f32, f32, f32, f32)> {
  %sel = llvm.mlir.constant(0 : i32) : i32
  // CHECK: call { float, float, float, float } @llvm.nvvm.mma.sp.ordered.metadata.m16n8k64.row.col.kind.f8f6f4.f32.e2m1.e2m1.f32(i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, float {{%[0-9]+}}, float {{%[0-9]+}}, float {{%[0-9]+}}, float {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 0)
  %res = nvvm.mma.sp.sync A[%a0, %a1, %a2, %a3] B[%b0, %b1, %b2, %b3] C[%c0, %c1, %c2, %c3]
      sparseMetadata[%meta] selector[%sel]
      {kind = #nvvm.mma_kind<f8f6f4>,
       multiplicandAPtxType = #nvvm.mma_type<e2m1>,
       multiplicandBPtxType = #nvvm.mma_type<e2m1>,
       orderedMetadata,
       shape = #nvvm.shape<m = 16, n = 8, k = 64>}
      : (i32, i32, f32) -> !llvm.struct<(f32, f32, f32, f32)>
  llvm.return %res : !llvm.struct<(f32, f32, f32, f32)>
}
