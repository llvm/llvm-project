// RUN: mlir-translate --mlir-to-llvmir %s | FileCheck %s

// CHECK-LABEL: @nvvm_mxf8f6f4_sp_blockscale_mma_e2m1_e2m1
llvm.func @nvvm_mxf8f6f4_sp_blockscale_mma_e2m1_e2m1(
    %a0: i32, %a1: i32, %a2: i32, %a3: i32,
    %b0: i32, %b1: i32, %b2: i32, %b3: i32,
    %c0: f32, %c1: f32, %c2: f32, %c3: f32,
    %meta: i32,
    %scaleA0: i32, %scaleA1: i16, %scaleA2: i16,
    %scaleB0: i32, %scaleB1: i16, %scaleB2: i16) -> !llvm.struct<(f32, f32, f32, f32)> {
  %sel = llvm.mlir.constant(0 : i32) : i32
  // CHECK: call { float, float, float, float } @llvm.nvvm.mma.sp.ordered.metadata.block.scale.m16n8k64.row.col.mxf8f6f4.scale.1x.f32.e2m1.e2m1.f32.ue8m0(i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, float {{%[0-9]+}}, float {{%[0-9]+}}, float {{%[0-9]+}}, float {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 0, i32 {{%[0-9]+}}, i16 {{%[0-9]+}}, i16 {{%[0-9]+}}, i32 {{%[0-9]+}}, i16 {{%[0-9]+}}, i16 {{%[0-9]+}})
  %res = nvvm.mma.sp.block_scale
      A[%a0, %a1, %a2, %a3]
      B[%b0, %b1, %b2, %b3]
      C[%c0, %c1, %c2, %c3]
      sparseMetadata[%meta]
      selector[%sel]
      scaleA[%scaleA0, %scaleA1, %scaleA2]
      scaleB[%scaleB0, %scaleB1, %scaleB2]
      {blockScaleFormat = #nvvm.block_scale_format<ue8m0>,
       kind = #nvvm.block_scale_kind<mxf8f6f4>,
       multiplicandAPtxType = #nvvm.mma_type<e2m1>,
       multiplicandBPtxType = #nvvm.mma_type<e2m1>,
       orderedMetadata,
       scaleVecSize = #nvvm.scale_vec_size<x1>,
       shape = #nvvm.shape<m = 16, n = 8, k = 64>}
      : (i32, i32, f32) -> !llvm.struct<(f32, f32, f32, f32)>
  llvm.return %res : !llvm.struct<(f32, f32, f32, f32)>
}

// CHECK-LABEL: @nvvm_mxf8f6f4_sp_blockscale_mma_e2m1_e2m3
llvm.func @nvvm_mxf8f6f4_sp_blockscale_mma_e2m1_e2m3(
    %a0: i32, %a1: i32, %a2: i32, %a3: i32,
    %b0: i32, %b1: i32, %b2: i32, %b3: i32,
    %c0: f32, %c1: f32, %c2: f32, %c3: f32,
    %meta: i32,
    %scaleA0: i32, %scaleA1: i16, %scaleA2: i16,
    %scaleB0: i32, %scaleB1: i16, %scaleB2: i16) -> !llvm.struct<(f32, f32, f32, f32)> {
  %sel = llvm.mlir.constant(0 : i32) : i32
  // CHECK: call { float, float, float, float } @llvm.nvvm.mma.sp.ordered.metadata.block.scale.m16n8k64.row.col.mxf8f6f4.scale.1x.f32.e2m1.e2m3.f32.ue8m0(i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, float {{%[0-9]+}}, float {{%[0-9]+}}, float {{%[0-9]+}}, float {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 0, i32 {{%[0-9]+}}, i16 {{%[0-9]+}}, i16 {{%[0-9]+}}, i32 {{%[0-9]+}}, i16 {{%[0-9]+}}, i16 {{%[0-9]+}})
  %res = nvvm.mma.sp.block_scale
      A[%a0, %a1, %a2, %a3]
      B[%b0, %b1, %b2, %b3]
      C[%c0, %c1, %c2, %c3]
      sparseMetadata[%meta]
      selector[%sel]
      scaleA[%scaleA0, %scaleA1, %scaleA2]
      scaleB[%scaleB0, %scaleB1, %scaleB2]
      {blockScaleFormat = #nvvm.block_scale_format<ue8m0>,
       kind = #nvvm.block_scale_kind<mxf8f6f4>,
       multiplicandAPtxType = #nvvm.mma_type<e2m1>,
       multiplicandBPtxType = #nvvm.mma_type<e2m3>,
       orderedMetadata,
       scaleVecSize = #nvvm.scale_vec_size<x1>,
       shape = #nvvm.shape<m = 16, n = 8, k = 64>}
      : (i32, i32, f32) -> !llvm.struct<(f32, f32, f32, f32)>
  llvm.return %res : !llvm.struct<(f32, f32, f32, f32)>
}

// CHECK-LABEL: @nvvm_mxf8f6f4_sp_blockscale_mma_e2m1_e3m2
llvm.func @nvvm_mxf8f6f4_sp_blockscale_mma_e2m1_e3m2(
    %a0: i32, %a1: i32, %a2: i32, %a3: i32,
    %b0: i32, %b1: i32, %b2: i32, %b3: i32,
    %c0: f32, %c1: f32, %c2: f32, %c3: f32,
    %meta: i32,
    %scaleA0: i32, %scaleA1: i16, %scaleA2: i16,
    %scaleB0: i32, %scaleB1: i16, %scaleB2: i16) -> !llvm.struct<(f32, f32, f32, f32)> {
  %sel = llvm.mlir.constant(0 : i32) : i32
  // CHECK: call { float, float, float, float } @llvm.nvvm.mma.sp.ordered.metadata.block.scale.m16n8k64.row.col.mxf8f6f4.scale.1x.f32.e2m1.e3m2.f32.ue8m0(i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, float {{%[0-9]+}}, float {{%[0-9]+}}, float {{%[0-9]+}}, float {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 0, i32 {{%[0-9]+}}, i16 {{%[0-9]+}}, i16 {{%[0-9]+}}, i32 {{%[0-9]+}}, i16 {{%[0-9]+}}, i16 {{%[0-9]+}})
  %res = nvvm.mma.sp.block_scale
      A[%a0, %a1, %a2, %a3]
      B[%b0, %b1, %b2, %b3]
      C[%c0, %c1, %c2, %c3]
      sparseMetadata[%meta]
      selector[%sel]
      scaleA[%scaleA0, %scaleA1, %scaleA2]
      scaleB[%scaleB0, %scaleB1, %scaleB2]
      {blockScaleFormat = #nvvm.block_scale_format<ue8m0>,
       kind = #nvvm.block_scale_kind<mxf8f6f4>,
       multiplicandAPtxType = #nvvm.mma_type<e2m1>,
       multiplicandBPtxType = #nvvm.mma_type<e3m2>,
       orderedMetadata,
       scaleVecSize = #nvvm.scale_vec_size<x1>,
       shape = #nvvm.shape<m = 16, n = 8, k = 64>}
      : (i32, i32, f32) -> !llvm.struct<(f32, f32, f32, f32)>
  llvm.return %res : !llvm.struct<(f32, f32, f32, f32)>
}

// CHECK-LABEL: @nvvm_mxf8f6f4_sp_blockscale_mma_e2m1_e4m3
llvm.func @nvvm_mxf8f6f4_sp_blockscale_mma_e2m1_e4m3(
    %a0: i32, %a1: i32, %a2: i32, %a3: i32,
    %b0: i32, %b1: i32, %b2: i32, %b3: i32,
    %c0: f32, %c1: f32, %c2: f32, %c3: f32,
    %meta: i32,
    %scaleA0: i32, %scaleA1: i16, %scaleA2: i16,
    %scaleB0: i32, %scaleB1: i16, %scaleB2: i16) -> !llvm.struct<(f32, f32, f32, f32)> {
  %sel = llvm.mlir.constant(0 : i32) : i32
  // CHECK: call { float, float, float, float } @llvm.nvvm.mma.sp.ordered.metadata.block.scale.m16n8k64.row.col.mxf8f6f4.scale.1x.f32.e2m1.e4m3.f32.ue8m0(i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, float {{%[0-9]+}}, float {{%[0-9]+}}, float {{%[0-9]+}}, float {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 0, i32 {{%[0-9]+}}, i16 {{%[0-9]+}}, i16 {{%[0-9]+}}, i32 {{%[0-9]+}}, i16 {{%[0-9]+}}, i16 {{%[0-9]+}})
  %res = nvvm.mma.sp.block_scale
      A[%a0, %a1, %a2, %a3]
      B[%b0, %b1, %b2, %b3]
      C[%c0, %c1, %c2, %c3]
      sparseMetadata[%meta]
      selector[%sel]
      scaleA[%scaleA0, %scaleA1, %scaleA2]
      scaleB[%scaleB0, %scaleB1, %scaleB2]
      {blockScaleFormat = #nvvm.block_scale_format<ue8m0>,
       kind = #nvvm.block_scale_kind<mxf8f6f4>,
       multiplicandAPtxType = #nvvm.mma_type<e2m1>,
       multiplicandBPtxType = #nvvm.mma_type<e4m3>,
       orderedMetadata,
       scaleVecSize = #nvvm.scale_vec_size<x1>,
       shape = #nvvm.shape<m = 16, n = 8, k = 64>}
      : (i32, i32, f32) -> !llvm.struct<(f32, f32, f32, f32)>
  llvm.return %res : !llvm.struct<(f32, f32, f32, f32)>
}

// CHECK-LABEL: @nvvm_mxf8f6f4_sp_blockscale_mma_e2m1_e5m2
llvm.func @nvvm_mxf8f6f4_sp_blockscale_mma_e2m1_e5m2(
    %a0: i32, %a1: i32, %a2: i32, %a3: i32,
    %b0: i32, %b1: i32, %b2: i32, %b3: i32,
    %c0: f32, %c1: f32, %c2: f32, %c3: f32,
    %meta: i32,
    %scaleA0: i32, %scaleA1: i16, %scaleA2: i16,
    %scaleB0: i32, %scaleB1: i16, %scaleB2: i16) -> !llvm.struct<(f32, f32, f32, f32)> {
  %sel = llvm.mlir.constant(0 : i32) : i32
  // CHECK: call { float, float, float, float } @llvm.nvvm.mma.sp.ordered.metadata.block.scale.m16n8k64.row.col.mxf8f6f4.scale.1x.f32.e2m1.e5m2.f32.ue8m0(i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, float {{%[0-9]+}}, float {{%[0-9]+}}, float {{%[0-9]+}}, float {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 0, i32 {{%[0-9]+}}, i16 {{%[0-9]+}}, i16 {{%[0-9]+}}, i32 {{%[0-9]+}}, i16 {{%[0-9]+}}, i16 {{%[0-9]+}})
  %res = nvvm.mma.sp.block_scale
      A[%a0, %a1, %a2, %a3]
      B[%b0, %b1, %b2, %b3]
      C[%c0, %c1, %c2, %c3]
      sparseMetadata[%meta]
      selector[%sel]
      scaleA[%scaleA0, %scaleA1, %scaleA2]
      scaleB[%scaleB0, %scaleB1, %scaleB2]
      {blockScaleFormat = #nvvm.block_scale_format<ue8m0>,
       kind = #nvvm.block_scale_kind<mxf8f6f4>,
       multiplicandAPtxType = #nvvm.mma_type<e2m1>,
       multiplicandBPtxType = #nvvm.mma_type<e5m2>,
       orderedMetadata,
       scaleVecSize = #nvvm.scale_vec_size<x1>,
       shape = #nvvm.shape<m = 16, n = 8, k = 64>}
      : (i32, i32, f32) -> !llvm.struct<(f32, f32, f32, f32)>
  llvm.return %res : !llvm.struct<(f32, f32, f32, f32)>
}

// CHECK-LABEL: @nvvm_mxf8f6f4_sp_blockscale_mma_e2m3_e2m1
llvm.func @nvvm_mxf8f6f4_sp_blockscale_mma_e2m3_e2m1(
    %a0: i32, %a1: i32, %a2: i32, %a3: i32,
    %b0: i32, %b1: i32, %b2: i32, %b3: i32,
    %c0: f32, %c1: f32, %c2: f32, %c3: f32,
    %meta: i32,
    %scaleA0: i32, %scaleA1: i16, %scaleA2: i16,
    %scaleB0: i32, %scaleB1: i16, %scaleB2: i16) -> !llvm.struct<(f32, f32, f32, f32)> {
  %sel = llvm.mlir.constant(0 : i32) : i32
  // CHECK: call { float, float, float, float } @llvm.nvvm.mma.sp.ordered.metadata.block.scale.m16n8k64.row.col.mxf8f6f4.scale.1x.f32.e2m3.e2m1.f32.ue8m0(i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, float {{%[0-9]+}}, float {{%[0-9]+}}, float {{%[0-9]+}}, float {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 0, i32 {{%[0-9]+}}, i16 {{%[0-9]+}}, i16 {{%[0-9]+}}, i32 {{%[0-9]+}}, i16 {{%[0-9]+}}, i16 {{%[0-9]+}})
  %res = nvvm.mma.sp.block_scale
      A[%a0, %a1, %a2, %a3]
      B[%b0, %b1, %b2, %b3]
      C[%c0, %c1, %c2, %c3]
      sparseMetadata[%meta]
      selector[%sel]
      scaleA[%scaleA0, %scaleA1, %scaleA2]
      scaleB[%scaleB0, %scaleB1, %scaleB2]
      {blockScaleFormat = #nvvm.block_scale_format<ue8m0>,
       kind = #nvvm.block_scale_kind<mxf8f6f4>,
       multiplicandAPtxType = #nvvm.mma_type<e2m3>,
       multiplicandBPtxType = #nvvm.mma_type<e2m1>,
       orderedMetadata,
       scaleVecSize = #nvvm.scale_vec_size<x1>,
       shape = #nvvm.shape<m = 16, n = 8, k = 64>}
      : (i32, i32, f32) -> !llvm.struct<(f32, f32, f32, f32)>
  llvm.return %res : !llvm.struct<(f32, f32, f32, f32)>
}

// CHECK-LABEL: @nvvm_mxf8f6f4_sp_blockscale_mma_e2m3_e2m3
llvm.func @nvvm_mxf8f6f4_sp_blockscale_mma_e2m3_e2m3(
    %a0: i32, %a1: i32, %a2: i32, %a3: i32,
    %b0: i32, %b1: i32, %b2: i32, %b3: i32,
    %c0: f32, %c1: f32, %c2: f32, %c3: f32,
    %meta: i32,
    %scaleA0: i32, %scaleA1: i16, %scaleA2: i16,
    %scaleB0: i32, %scaleB1: i16, %scaleB2: i16) -> !llvm.struct<(f32, f32, f32, f32)> {
  %sel = llvm.mlir.constant(0 : i32) : i32
  // CHECK: call { float, float, float, float } @llvm.nvvm.mma.sp.ordered.metadata.block.scale.m16n8k64.row.col.mxf8f6f4.scale.1x.f32.e2m3.e2m3.f32.ue8m0(i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, float {{%[0-9]+}}, float {{%[0-9]+}}, float {{%[0-9]+}}, float {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 0, i32 {{%[0-9]+}}, i16 {{%[0-9]+}}, i16 {{%[0-9]+}}, i32 {{%[0-9]+}}, i16 {{%[0-9]+}}, i16 {{%[0-9]+}})
  %res = nvvm.mma.sp.block_scale
      A[%a0, %a1, %a2, %a3]
      B[%b0, %b1, %b2, %b3]
      C[%c0, %c1, %c2, %c3]
      sparseMetadata[%meta]
      selector[%sel]
      scaleA[%scaleA0, %scaleA1, %scaleA2]
      scaleB[%scaleB0, %scaleB1, %scaleB2]
      {blockScaleFormat = #nvvm.block_scale_format<ue8m0>,
       kind = #nvvm.block_scale_kind<mxf8f6f4>,
       multiplicandAPtxType = #nvvm.mma_type<e2m3>,
       multiplicandBPtxType = #nvvm.mma_type<e2m3>,
       orderedMetadata,
       scaleVecSize = #nvvm.scale_vec_size<x1>,
       shape = #nvvm.shape<m = 16, n = 8, k = 64>}
      : (i32, i32, f32) -> !llvm.struct<(f32, f32, f32, f32)>
  llvm.return %res : !llvm.struct<(f32, f32, f32, f32)>
}

// CHECK-LABEL: @nvvm_mxf8f6f4_sp_blockscale_mma_e2m3_e3m2
llvm.func @nvvm_mxf8f6f4_sp_blockscale_mma_e2m3_e3m2(
    %a0: i32, %a1: i32, %a2: i32, %a3: i32,
    %b0: i32, %b1: i32, %b2: i32, %b3: i32,
    %c0: f32, %c1: f32, %c2: f32, %c3: f32,
    %meta: i32,
    %scaleA0: i32, %scaleA1: i16, %scaleA2: i16,
    %scaleB0: i32, %scaleB1: i16, %scaleB2: i16) -> !llvm.struct<(f32, f32, f32, f32)> {
  %sel = llvm.mlir.constant(0 : i32) : i32
  // CHECK: call { float, float, float, float } @llvm.nvvm.mma.sp.ordered.metadata.block.scale.m16n8k64.row.col.mxf8f6f4.scale.1x.f32.e2m3.e3m2.f32.ue8m0(i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, float {{%[0-9]+}}, float {{%[0-9]+}}, float {{%[0-9]+}}, float {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 0, i32 {{%[0-9]+}}, i16 {{%[0-9]+}}, i16 {{%[0-9]+}}, i32 {{%[0-9]+}}, i16 {{%[0-9]+}}, i16 {{%[0-9]+}})
  %res = nvvm.mma.sp.block_scale
      A[%a0, %a1, %a2, %a3]
      B[%b0, %b1, %b2, %b3]
      C[%c0, %c1, %c2, %c3]
      sparseMetadata[%meta]
      selector[%sel]
      scaleA[%scaleA0, %scaleA1, %scaleA2]
      scaleB[%scaleB0, %scaleB1, %scaleB2]
      {blockScaleFormat = #nvvm.block_scale_format<ue8m0>,
       kind = #nvvm.block_scale_kind<mxf8f6f4>,
       multiplicandAPtxType = #nvvm.mma_type<e2m3>,
       multiplicandBPtxType = #nvvm.mma_type<e3m2>,
       orderedMetadata,
       scaleVecSize = #nvvm.scale_vec_size<x1>,
       shape = #nvvm.shape<m = 16, n = 8, k = 64>}
      : (i32, i32, f32) -> !llvm.struct<(f32, f32, f32, f32)>
  llvm.return %res : !llvm.struct<(f32, f32, f32, f32)>
}

// CHECK-LABEL: @nvvm_mxf8f6f4_sp_blockscale_mma_e2m3_e4m3
llvm.func @nvvm_mxf8f6f4_sp_blockscale_mma_e2m3_e4m3(
    %a0: i32, %a1: i32, %a2: i32, %a3: i32,
    %b0: i32, %b1: i32, %b2: i32, %b3: i32,
    %c0: f32, %c1: f32, %c2: f32, %c3: f32,
    %meta: i32,
    %scaleA0: i32, %scaleA1: i16, %scaleA2: i16,
    %scaleB0: i32, %scaleB1: i16, %scaleB2: i16) -> !llvm.struct<(f32, f32, f32, f32)> {
  %sel = llvm.mlir.constant(0 : i32) : i32
  // CHECK: call { float, float, float, float } @llvm.nvvm.mma.sp.ordered.metadata.block.scale.m16n8k64.row.col.mxf8f6f4.scale.1x.f32.e2m3.e4m3.f32.ue8m0(i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, float {{%[0-9]+}}, float {{%[0-9]+}}, float {{%[0-9]+}}, float {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 0, i32 {{%[0-9]+}}, i16 {{%[0-9]+}}, i16 {{%[0-9]+}}, i32 {{%[0-9]+}}, i16 {{%[0-9]+}}, i16 {{%[0-9]+}})
  %res = nvvm.mma.sp.block_scale
      A[%a0, %a1, %a2, %a3]
      B[%b0, %b1, %b2, %b3]
      C[%c0, %c1, %c2, %c3]
      sparseMetadata[%meta]
      selector[%sel]
      scaleA[%scaleA0, %scaleA1, %scaleA2]
      scaleB[%scaleB0, %scaleB1, %scaleB2]
      {blockScaleFormat = #nvvm.block_scale_format<ue8m0>,
       kind = #nvvm.block_scale_kind<mxf8f6f4>,
       multiplicandAPtxType = #nvvm.mma_type<e2m3>,
       multiplicandBPtxType = #nvvm.mma_type<e4m3>,
       orderedMetadata,
       scaleVecSize = #nvvm.scale_vec_size<x1>,
       shape = #nvvm.shape<m = 16, n = 8, k = 64>}
      : (i32, i32, f32) -> !llvm.struct<(f32, f32, f32, f32)>
  llvm.return %res : !llvm.struct<(f32, f32, f32, f32)>
}

// CHECK-LABEL: @nvvm_mxf8f6f4_sp_blockscale_mma_e2m3_e5m2
llvm.func @nvvm_mxf8f6f4_sp_blockscale_mma_e2m3_e5m2(
    %a0: i32, %a1: i32, %a2: i32, %a3: i32,
    %b0: i32, %b1: i32, %b2: i32, %b3: i32,
    %c0: f32, %c1: f32, %c2: f32, %c3: f32,
    %meta: i32,
    %scaleA0: i32, %scaleA1: i16, %scaleA2: i16,
    %scaleB0: i32, %scaleB1: i16, %scaleB2: i16) -> !llvm.struct<(f32, f32, f32, f32)> {
  %sel = llvm.mlir.constant(0 : i32) : i32
  // CHECK: call { float, float, float, float } @llvm.nvvm.mma.sp.ordered.metadata.block.scale.m16n8k64.row.col.mxf8f6f4.scale.1x.f32.e2m3.e5m2.f32.ue8m0(i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, float {{%[0-9]+}}, float {{%[0-9]+}}, float {{%[0-9]+}}, float {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 0, i32 {{%[0-9]+}}, i16 {{%[0-9]+}}, i16 {{%[0-9]+}}, i32 {{%[0-9]+}}, i16 {{%[0-9]+}}, i16 {{%[0-9]+}})
  %res = nvvm.mma.sp.block_scale
      A[%a0, %a1, %a2, %a3]
      B[%b0, %b1, %b2, %b3]
      C[%c0, %c1, %c2, %c3]
      sparseMetadata[%meta]
      selector[%sel]
      scaleA[%scaleA0, %scaleA1, %scaleA2]
      scaleB[%scaleB0, %scaleB1, %scaleB2]
      {blockScaleFormat = #nvvm.block_scale_format<ue8m0>,
       kind = #nvvm.block_scale_kind<mxf8f6f4>,
       multiplicandAPtxType = #nvvm.mma_type<e2m3>,
       multiplicandBPtxType = #nvvm.mma_type<e5m2>,
       orderedMetadata,
       scaleVecSize = #nvvm.scale_vec_size<x1>,
       shape = #nvvm.shape<m = 16, n = 8, k = 64>}
      : (i32, i32, f32) -> !llvm.struct<(f32, f32, f32, f32)>
  llvm.return %res : !llvm.struct<(f32, f32, f32, f32)>
}

// CHECK-LABEL: @nvvm_mxf8f6f4_sp_blockscale_mma_e3m2_e2m1
llvm.func @nvvm_mxf8f6f4_sp_blockscale_mma_e3m2_e2m1(
    %a0: i32, %a1: i32, %a2: i32, %a3: i32,
    %b0: i32, %b1: i32, %b2: i32, %b3: i32,
    %c0: f32, %c1: f32, %c2: f32, %c3: f32,
    %meta: i32,
    %scaleA0: i32, %scaleA1: i16, %scaleA2: i16,
    %scaleB0: i32, %scaleB1: i16, %scaleB2: i16) -> !llvm.struct<(f32, f32, f32, f32)> {
  %sel = llvm.mlir.constant(0 : i32) : i32
  // CHECK: call { float, float, float, float } @llvm.nvvm.mma.sp.ordered.metadata.block.scale.m16n8k64.row.col.mxf8f6f4.scale.1x.f32.e3m2.e2m1.f32.ue8m0(i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, float {{%[0-9]+}}, float {{%[0-9]+}}, float {{%[0-9]+}}, float {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 0, i32 {{%[0-9]+}}, i16 {{%[0-9]+}}, i16 {{%[0-9]+}}, i32 {{%[0-9]+}}, i16 {{%[0-9]+}}, i16 {{%[0-9]+}})
  %res = nvvm.mma.sp.block_scale
      A[%a0, %a1, %a2, %a3]
      B[%b0, %b1, %b2, %b3]
      C[%c0, %c1, %c2, %c3]
      sparseMetadata[%meta]
      selector[%sel]
      scaleA[%scaleA0, %scaleA1, %scaleA2]
      scaleB[%scaleB0, %scaleB1, %scaleB2]
      {blockScaleFormat = #nvvm.block_scale_format<ue8m0>,
       kind = #nvvm.block_scale_kind<mxf8f6f4>,
       multiplicandAPtxType = #nvvm.mma_type<e3m2>,
       multiplicandBPtxType = #nvvm.mma_type<e2m1>,
       orderedMetadata,
       scaleVecSize = #nvvm.scale_vec_size<x1>,
       shape = #nvvm.shape<m = 16, n = 8, k = 64>}
      : (i32, i32, f32) -> !llvm.struct<(f32, f32, f32, f32)>
  llvm.return %res : !llvm.struct<(f32, f32, f32, f32)>
}

// CHECK-LABEL: @nvvm_mxf8f6f4_sp_blockscale_mma_e3m2_e2m3
llvm.func @nvvm_mxf8f6f4_sp_blockscale_mma_e3m2_e2m3(
    %a0: i32, %a1: i32, %a2: i32, %a3: i32,
    %b0: i32, %b1: i32, %b2: i32, %b3: i32,
    %c0: f32, %c1: f32, %c2: f32, %c3: f32,
    %meta: i32,
    %scaleA0: i32, %scaleA1: i16, %scaleA2: i16,
    %scaleB0: i32, %scaleB1: i16, %scaleB2: i16) -> !llvm.struct<(f32, f32, f32, f32)> {
  %sel = llvm.mlir.constant(0 : i32) : i32
  // CHECK: call { float, float, float, float } @llvm.nvvm.mma.sp.ordered.metadata.block.scale.m16n8k64.row.col.mxf8f6f4.scale.1x.f32.e3m2.e2m3.f32.ue8m0(i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, float {{%[0-9]+}}, float {{%[0-9]+}}, float {{%[0-9]+}}, float {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 0, i32 {{%[0-9]+}}, i16 {{%[0-9]+}}, i16 {{%[0-9]+}}, i32 {{%[0-9]+}}, i16 {{%[0-9]+}}, i16 {{%[0-9]+}})
  %res = nvvm.mma.sp.block_scale
      A[%a0, %a1, %a2, %a3]
      B[%b0, %b1, %b2, %b3]
      C[%c0, %c1, %c2, %c3]
      sparseMetadata[%meta]
      selector[%sel]
      scaleA[%scaleA0, %scaleA1, %scaleA2]
      scaleB[%scaleB0, %scaleB1, %scaleB2]
      {blockScaleFormat = #nvvm.block_scale_format<ue8m0>,
       kind = #nvvm.block_scale_kind<mxf8f6f4>,
       multiplicandAPtxType = #nvvm.mma_type<e3m2>,
       multiplicandBPtxType = #nvvm.mma_type<e2m3>,
       orderedMetadata,
       scaleVecSize = #nvvm.scale_vec_size<x1>,
       shape = #nvvm.shape<m = 16, n = 8, k = 64>}
      : (i32, i32, f32) -> !llvm.struct<(f32, f32, f32, f32)>
  llvm.return %res : !llvm.struct<(f32, f32, f32, f32)>
}

// CHECK-LABEL: @nvvm_mxf8f6f4_sp_blockscale_mma_e3m2_e3m2
llvm.func @nvvm_mxf8f6f4_sp_blockscale_mma_e3m2_e3m2(
    %a0: i32, %a1: i32, %a2: i32, %a3: i32,
    %b0: i32, %b1: i32, %b2: i32, %b3: i32,
    %c0: f32, %c1: f32, %c2: f32, %c3: f32,
    %meta: i32,
    %scaleA0: i32, %scaleA1: i16, %scaleA2: i16,
    %scaleB0: i32, %scaleB1: i16, %scaleB2: i16) -> !llvm.struct<(f32, f32, f32, f32)> {
  %sel = llvm.mlir.constant(0 : i32) : i32
  // CHECK: call { float, float, float, float } @llvm.nvvm.mma.sp.ordered.metadata.block.scale.m16n8k64.row.col.mxf8f6f4.scale.1x.f32.e3m2.e3m2.f32.ue8m0(i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, float {{%[0-9]+}}, float {{%[0-9]+}}, float {{%[0-9]+}}, float {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 0, i32 {{%[0-9]+}}, i16 {{%[0-9]+}}, i16 {{%[0-9]+}}, i32 {{%[0-9]+}}, i16 {{%[0-9]+}}, i16 {{%[0-9]+}})
  %res = nvvm.mma.sp.block_scale
      A[%a0, %a1, %a2, %a3]
      B[%b0, %b1, %b2, %b3]
      C[%c0, %c1, %c2, %c3]
      sparseMetadata[%meta]
      selector[%sel]
      scaleA[%scaleA0, %scaleA1, %scaleA2]
      scaleB[%scaleB0, %scaleB1, %scaleB2]
      {blockScaleFormat = #nvvm.block_scale_format<ue8m0>,
       kind = #nvvm.block_scale_kind<mxf8f6f4>,
       multiplicandAPtxType = #nvvm.mma_type<e3m2>,
       multiplicandBPtxType = #nvvm.mma_type<e3m2>,
       orderedMetadata,
       scaleVecSize = #nvvm.scale_vec_size<x1>,
       shape = #nvvm.shape<m = 16, n = 8, k = 64>}
      : (i32, i32, f32) -> !llvm.struct<(f32, f32, f32, f32)>
  llvm.return %res : !llvm.struct<(f32, f32, f32, f32)>
}

// CHECK-LABEL: @nvvm_mxf8f6f4_sp_blockscale_mma_e3m2_e4m3
llvm.func @nvvm_mxf8f6f4_sp_blockscale_mma_e3m2_e4m3(
    %a0: i32, %a1: i32, %a2: i32, %a3: i32,
    %b0: i32, %b1: i32, %b2: i32, %b3: i32,
    %c0: f32, %c1: f32, %c2: f32, %c3: f32,
    %meta: i32,
    %scaleA0: i32, %scaleA1: i16, %scaleA2: i16,
    %scaleB0: i32, %scaleB1: i16, %scaleB2: i16) -> !llvm.struct<(f32, f32, f32, f32)> {
  %sel = llvm.mlir.constant(0 : i32) : i32
  // CHECK: call { float, float, float, float } @llvm.nvvm.mma.sp.ordered.metadata.block.scale.m16n8k64.row.col.mxf8f6f4.scale.1x.f32.e3m2.e4m3.f32.ue8m0(i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, float {{%[0-9]+}}, float {{%[0-9]+}}, float {{%[0-9]+}}, float {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 0, i32 {{%[0-9]+}}, i16 {{%[0-9]+}}, i16 {{%[0-9]+}}, i32 {{%[0-9]+}}, i16 {{%[0-9]+}}, i16 {{%[0-9]+}})
  %res = nvvm.mma.sp.block_scale
      A[%a0, %a1, %a2, %a3]
      B[%b0, %b1, %b2, %b3]
      C[%c0, %c1, %c2, %c3]
      sparseMetadata[%meta]
      selector[%sel]
      scaleA[%scaleA0, %scaleA1, %scaleA2]
      scaleB[%scaleB0, %scaleB1, %scaleB2]
      {blockScaleFormat = #nvvm.block_scale_format<ue8m0>,
       kind = #nvvm.block_scale_kind<mxf8f6f4>,
       multiplicandAPtxType = #nvvm.mma_type<e3m2>,
       multiplicandBPtxType = #nvvm.mma_type<e4m3>,
       orderedMetadata,
       scaleVecSize = #nvvm.scale_vec_size<x1>,
       shape = #nvvm.shape<m = 16, n = 8, k = 64>}
      : (i32, i32, f32) -> !llvm.struct<(f32, f32, f32, f32)>
  llvm.return %res : !llvm.struct<(f32, f32, f32, f32)>
}

// CHECK-LABEL: @nvvm_mxf8f6f4_sp_blockscale_mma_e3m2_e5m2
llvm.func @nvvm_mxf8f6f4_sp_blockscale_mma_e3m2_e5m2(
    %a0: i32, %a1: i32, %a2: i32, %a3: i32,
    %b0: i32, %b1: i32, %b2: i32, %b3: i32,
    %c0: f32, %c1: f32, %c2: f32, %c3: f32,
    %meta: i32,
    %scaleA0: i32, %scaleA1: i16, %scaleA2: i16,
    %scaleB0: i32, %scaleB1: i16, %scaleB2: i16) -> !llvm.struct<(f32, f32, f32, f32)> {
  %sel = llvm.mlir.constant(0 : i32) : i32
  // CHECK: call { float, float, float, float } @llvm.nvvm.mma.sp.ordered.metadata.block.scale.m16n8k64.row.col.mxf8f6f4.scale.1x.f32.e3m2.e5m2.f32.ue8m0(i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, float {{%[0-9]+}}, float {{%[0-9]+}}, float {{%[0-9]+}}, float {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 0, i32 {{%[0-9]+}}, i16 {{%[0-9]+}}, i16 {{%[0-9]+}}, i32 {{%[0-9]+}}, i16 {{%[0-9]+}}, i16 {{%[0-9]+}})
  %res = nvvm.mma.sp.block_scale
      A[%a0, %a1, %a2, %a3]
      B[%b0, %b1, %b2, %b3]
      C[%c0, %c1, %c2, %c3]
      sparseMetadata[%meta]
      selector[%sel]
      scaleA[%scaleA0, %scaleA1, %scaleA2]
      scaleB[%scaleB0, %scaleB1, %scaleB2]
      {blockScaleFormat = #nvvm.block_scale_format<ue8m0>,
       kind = #nvvm.block_scale_kind<mxf8f6f4>,
       multiplicandAPtxType = #nvvm.mma_type<e3m2>,
       multiplicandBPtxType = #nvvm.mma_type<e5m2>,
       orderedMetadata,
       scaleVecSize = #nvvm.scale_vec_size<x1>,
       shape = #nvvm.shape<m = 16, n = 8, k = 64>}
      : (i32, i32, f32) -> !llvm.struct<(f32, f32, f32, f32)>
  llvm.return %res : !llvm.struct<(f32, f32, f32, f32)>
}

// CHECK-LABEL: @nvvm_mxf8f6f4_sp_blockscale_mma_e4m3_e2m1
llvm.func @nvvm_mxf8f6f4_sp_blockscale_mma_e4m3_e2m1(
    %a0: i32, %a1: i32, %a2: i32, %a3: i32,
    %b0: i32, %b1: i32, %b2: i32, %b3: i32,
    %c0: f32, %c1: f32, %c2: f32, %c3: f32,
    %meta: i32,
    %scaleA0: i32, %scaleA1: i16, %scaleA2: i16,
    %scaleB0: i32, %scaleB1: i16, %scaleB2: i16) -> !llvm.struct<(f32, f32, f32, f32)> {
  %sel = llvm.mlir.constant(0 : i32) : i32
  // CHECK: call { float, float, float, float } @llvm.nvvm.mma.sp.ordered.metadata.block.scale.m16n8k64.row.col.mxf8f6f4.scale.1x.f32.e4m3.e2m1.f32.ue8m0(i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, float {{%[0-9]+}}, float {{%[0-9]+}}, float {{%[0-9]+}}, float {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 0, i32 {{%[0-9]+}}, i16 {{%[0-9]+}}, i16 {{%[0-9]+}}, i32 {{%[0-9]+}}, i16 {{%[0-9]+}}, i16 {{%[0-9]+}})
  %res = nvvm.mma.sp.block_scale
      A[%a0, %a1, %a2, %a3]
      B[%b0, %b1, %b2, %b3]
      C[%c0, %c1, %c2, %c3]
      sparseMetadata[%meta]
      selector[%sel]
      scaleA[%scaleA0, %scaleA1, %scaleA2]
      scaleB[%scaleB0, %scaleB1, %scaleB2]
      {blockScaleFormat = #nvvm.block_scale_format<ue8m0>,
       kind = #nvvm.block_scale_kind<mxf8f6f4>,
       multiplicandAPtxType = #nvvm.mma_type<e4m3>,
       multiplicandBPtxType = #nvvm.mma_type<e2m1>,
       orderedMetadata,
       scaleVecSize = #nvvm.scale_vec_size<x1>,
       shape = #nvvm.shape<m = 16, n = 8, k = 64>}
      : (i32, i32, f32) -> !llvm.struct<(f32, f32, f32, f32)>
  llvm.return %res : !llvm.struct<(f32, f32, f32, f32)>
}

// CHECK-LABEL: @nvvm_mxf8f6f4_sp_blockscale_mma_e4m3_e2m3
llvm.func @nvvm_mxf8f6f4_sp_blockscale_mma_e4m3_e2m3(
    %a0: i32, %a1: i32, %a2: i32, %a3: i32,
    %b0: i32, %b1: i32, %b2: i32, %b3: i32,
    %c0: f32, %c1: f32, %c2: f32, %c3: f32,
    %meta: i32,
    %scaleA0: i32, %scaleA1: i16, %scaleA2: i16,
    %scaleB0: i32, %scaleB1: i16, %scaleB2: i16) -> !llvm.struct<(f32, f32, f32, f32)> {
  %sel = llvm.mlir.constant(0 : i32) : i32
  // CHECK: call { float, float, float, float } @llvm.nvvm.mma.sp.ordered.metadata.block.scale.m16n8k64.row.col.mxf8f6f4.scale.1x.f32.e4m3.e2m3.f32.ue8m0(i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, float {{%[0-9]+}}, float {{%[0-9]+}}, float {{%[0-9]+}}, float {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 0, i32 {{%[0-9]+}}, i16 {{%[0-9]+}}, i16 {{%[0-9]+}}, i32 {{%[0-9]+}}, i16 {{%[0-9]+}}, i16 {{%[0-9]+}})
  %res = nvvm.mma.sp.block_scale
      A[%a0, %a1, %a2, %a3]
      B[%b0, %b1, %b2, %b3]
      C[%c0, %c1, %c2, %c3]
      sparseMetadata[%meta]
      selector[%sel]
      scaleA[%scaleA0, %scaleA1, %scaleA2]
      scaleB[%scaleB0, %scaleB1, %scaleB2]
      {blockScaleFormat = #nvvm.block_scale_format<ue8m0>,
       kind = #nvvm.block_scale_kind<mxf8f6f4>,
       multiplicandAPtxType = #nvvm.mma_type<e4m3>,
       multiplicandBPtxType = #nvvm.mma_type<e2m3>,
       orderedMetadata,
       scaleVecSize = #nvvm.scale_vec_size<x1>,
       shape = #nvvm.shape<m = 16, n = 8, k = 64>}
      : (i32, i32, f32) -> !llvm.struct<(f32, f32, f32, f32)>
  llvm.return %res : !llvm.struct<(f32, f32, f32, f32)>
}

// CHECK-LABEL: @nvvm_mxf8f6f4_sp_blockscale_mma_e4m3_e3m2
llvm.func @nvvm_mxf8f6f4_sp_blockscale_mma_e4m3_e3m2(
    %a0: i32, %a1: i32, %a2: i32, %a3: i32,
    %b0: i32, %b1: i32, %b2: i32, %b3: i32,
    %c0: f32, %c1: f32, %c2: f32, %c3: f32,
    %meta: i32,
    %scaleA0: i32, %scaleA1: i16, %scaleA2: i16,
    %scaleB0: i32, %scaleB1: i16, %scaleB2: i16) -> !llvm.struct<(f32, f32, f32, f32)> {
  %sel = llvm.mlir.constant(0 : i32) : i32
  // CHECK: call { float, float, float, float } @llvm.nvvm.mma.sp.ordered.metadata.block.scale.m16n8k64.row.col.mxf8f6f4.scale.1x.f32.e4m3.e3m2.f32.ue8m0(i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, float {{%[0-9]+}}, float {{%[0-9]+}}, float {{%[0-9]+}}, float {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 0, i32 {{%[0-9]+}}, i16 {{%[0-9]+}}, i16 {{%[0-9]+}}, i32 {{%[0-9]+}}, i16 {{%[0-9]+}}, i16 {{%[0-9]+}})
  %res = nvvm.mma.sp.block_scale
      A[%a0, %a1, %a2, %a3]
      B[%b0, %b1, %b2, %b3]
      C[%c0, %c1, %c2, %c3]
      sparseMetadata[%meta]
      selector[%sel]
      scaleA[%scaleA0, %scaleA1, %scaleA2]
      scaleB[%scaleB0, %scaleB1, %scaleB2]
      {blockScaleFormat = #nvvm.block_scale_format<ue8m0>,
       kind = #nvvm.block_scale_kind<mxf8f6f4>,
       multiplicandAPtxType = #nvvm.mma_type<e4m3>,
       multiplicandBPtxType = #nvvm.mma_type<e3m2>,
       orderedMetadata,
       scaleVecSize = #nvvm.scale_vec_size<x1>,
       shape = #nvvm.shape<m = 16, n = 8, k = 64>}
      : (i32, i32, f32) -> !llvm.struct<(f32, f32, f32, f32)>
  llvm.return %res : !llvm.struct<(f32, f32, f32, f32)>
}

// CHECK-LABEL: @nvvm_mxf8f6f4_sp_blockscale_mma_e4m3_e4m3
llvm.func @nvvm_mxf8f6f4_sp_blockscale_mma_e4m3_e4m3(
    %a0: i32, %a1: i32, %a2: i32, %a3: i32,
    %b0: i32, %b1: i32, %b2: i32, %b3: i32,
    %c0: f32, %c1: f32, %c2: f32, %c3: f32,
    %meta: i32,
    %scaleA0: i32, %scaleA1: i16, %scaleA2: i16,
    %scaleB0: i32, %scaleB1: i16, %scaleB2: i16) -> !llvm.struct<(f32, f32, f32, f32)> {
  %sel = llvm.mlir.constant(0 : i32) : i32
  // CHECK: call { float, float, float, float } @llvm.nvvm.mma.sp.ordered.metadata.block.scale.m16n8k64.row.col.mxf8f6f4.scale.1x.f32.e4m3.e4m3.f32.ue8m0(i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, float {{%[0-9]+}}, float {{%[0-9]+}}, float {{%[0-9]+}}, float {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 0, i32 {{%[0-9]+}}, i16 {{%[0-9]+}}, i16 {{%[0-9]+}}, i32 {{%[0-9]+}}, i16 {{%[0-9]+}}, i16 {{%[0-9]+}})
  %res = nvvm.mma.sp.block_scale
      A[%a0, %a1, %a2, %a3]
      B[%b0, %b1, %b2, %b3]
      C[%c0, %c1, %c2, %c3]
      sparseMetadata[%meta]
      selector[%sel]
      scaleA[%scaleA0, %scaleA1, %scaleA2]
      scaleB[%scaleB0, %scaleB1, %scaleB2]
      {blockScaleFormat = #nvvm.block_scale_format<ue8m0>,
       kind = #nvvm.block_scale_kind<mxf8f6f4>,
       multiplicandAPtxType = #nvvm.mma_type<e4m3>,
       multiplicandBPtxType = #nvvm.mma_type<e4m3>,
       orderedMetadata,
       scaleVecSize = #nvvm.scale_vec_size<x1>,
       shape = #nvvm.shape<m = 16, n = 8, k = 64>}
      : (i32, i32, f32) -> !llvm.struct<(f32, f32, f32, f32)>
  llvm.return %res : !llvm.struct<(f32, f32, f32, f32)>
}

// CHECK-LABEL: @nvvm_mxf8f6f4_sp_blockscale_mma_e4m3_e5m2
llvm.func @nvvm_mxf8f6f4_sp_blockscale_mma_e4m3_e5m2(
    %a0: i32, %a1: i32, %a2: i32, %a3: i32,
    %b0: i32, %b1: i32, %b2: i32, %b3: i32,
    %c0: f32, %c1: f32, %c2: f32, %c3: f32,
    %meta: i32,
    %scaleA0: i32, %scaleA1: i16, %scaleA2: i16,
    %scaleB0: i32, %scaleB1: i16, %scaleB2: i16) -> !llvm.struct<(f32, f32, f32, f32)> {
  %sel = llvm.mlir.constant(0 : i32) : i32
  // CHECK: call { float, float, float, float } @llvm.nvvm.mma.sp.ordered.metadata.block.scale.m16n8k64.row.col.mxf8f6f4.scale.1x.f32.e4m3.e5m2.f32.ue8m0(i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, float {{%[0-9]+}}, float {{%[0-9]+}}, float {{%[0-9]+}}, float {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 0, i32 {{%[0-9]+}}, i16 {{%[0-9]+}}, i16 {{%[0-9]+}}, i32 {{%[0-9]+}}, i16 {{%[0-9]+}}, i16 {{%[0-9]+}})
  %res = nvvm.mma.sp.block_scale
      A[%a0, %a1, %a2, %a3]
      B[%b0, %b1, %b2, %b3]
      C[%c0, %c1, %c2, %c3]
      sparseMetadata[%meta]
      selector[%sel]
      scaleA[%scaleA0, %scaleA1, %scaleA2]
      scaleB[%scaleB0, %scaleB1, %scaleB2]
      {blockScaleFormat = #nvvm.block_scale_format<ue8m0>,
       kind = #nvvm.block_scale_kind<mxf8f6f4>,
       multiplicandAPtxType = #nvvm.mma_type<e4m3>,
       multiplicandBPtxType = #nvvm.mma_type<e5m2>,
       orderedMetadata,
       scaleVecSize = #nvvm.scale_vec_size<x1>,
       shape = #nvvm.shape<m = 16, n = 8, k = 64>}
      : (i32, i32, f32) -> !llvm.struct<(f32, f32, f32, f32)>
  llvm.return %res : !llvm.struct<(f32, f32, f32, f32)>
}

// CHECK-LABEL: @nvvm_mxf8f6f4_sp_blockscale_mma_e5m2_e2m1
llvm.func @nvvm_mxf8f6f4_sp_blockscale_mma_e5m2_e2m1(
    %a0: i32, %a1: i32, %a2: i32, %a3: i32,
    %b0: i32, %b1: i32, %b2: i32, %b3: i32,
    %c0: f32, %c1: f32, %c2: f32, %c3: f32,
    %meta: i32,
    %scaleA0: i32, %scaleA1: i16, %scaleA2: i16,
    %scaleB0: i32, %scaleB1: i16, %scaleB2: i16) -> !llvm.struct<(f32, f32, f32, f32)> {
  %sel = llvm.mlir.constant(0 : i32) : i32
  // CHECK: call { float, float, float, float } @llvm.nvvm.mma.sp.ordered.metadata.block.scale.m16n8k64.row.col.mxf8f6f4.scale.1x.f32.e5m2.e2m1.f32.ue8m0(i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, float {{%[0-9]+}}, float {{%[0-9]+}}, float {{%[0-9]+}}, float {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 0, i32 {{%[0-9]+}}, i16 {{%[0-9]+}}, i16 {{%[0-9]+}}, i32 {{%[0-9]+}}, i16 {{%[0-9]+}}, i16 {{%[0-9]+}})
  %res = nvvm.mma.sp.block_scale
      A[%a0, %a1, %a2, %a3]
      B[%b0, %b1, %b2, %b3]
      C[%c0, %c1, %c2, %c3]
      sparseMetadata[%meta]
      selector[%sel]
      scaleA[%scaleA0, %scaleA1, %scaleA2]
      scaleB[%scaleB0, %scaleB1, %scaleB2]
      {blockScaleFormat = #nvvm.block_scale_format<ue8m0>,
       kind = #nvvm.block_scale_kind<mxf8f6f4>,
       multiplicandAPtxType = #nvvm.mma_type<e5m2>,
       multiplicandBPtxType = #nvvm.mma_type<e2m1>,
       orderedMetadata,
       scaleVecSize = #nvvm.scale_vec_size<x1>,
       shape = #nvvm.shape<m = 16, n = 8, k = 64>}
      : (i32, i32, f32) -> !llvm.struct<(f32, f32, f32, f32)>
  llvm.return %res : !llvm.struct<(f32, f32, f32, f32)>
}

// CHECK-LABEL: @nvvm_mxf8f6f4_sp_blockscale_mma_e5m2_e2m3
llvm.func @nvvm_mxf8f6f4_sp_blockscale_mma_e5m2_e2m3(
    %a0: i32, %a1: i32, %a2: i32, %a3: i32,
    %b0: i32, %b1: i32, %b2: i32, %b3: i32,
    %c0: f32, %c1: f32, %c2: f32, %c3: f32,
    %meta: i32,
    %scaleA0: i32, %scaleA1: i16, %scaleA2: i16,
    %scaleB0: i32, %scaleB1: i16, %scaleB2: i16) -> !llvm.struct<(f32, f32, f32, f32)> {
  %sel = llvm.mlir.constant(0 : i32) : i32
  // CHECK: call { float, float, float, float } @llvm.nvvm.mma.sp.ordered.metadata.block.scale.m16n8k64.row.col.mxf8f6f4.scale.1x.f32.e5m2.e2m3.f32.ue8m0(i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, float {{%[0-9]+}}, float {{%[0-9]+}}, float {{%[0-9]+}}, float {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 0, i32 {{%[0-9]+}}, i16 {{%[0-9]+}}, i16 {{%[0-9]+}}, i32 {{%[0-9]+}}, i16 {{%[0-9]+}}, i16 {{%[0-9]+}})
  %res = nvvm.mma.sp.block_scale
      A[%a0, %a1, %a2, %a3]
      B[%b0, %b1, %b2, %b3]
      C[%c0, %c1, %c2, %c3]
      sparseMetadata[%meta]
      selector[%sel]
      scaleA[%scaleA0, %scaleA1, %scaleA2]
      scaleB[%scaleB0, %scaleB1, %scaleB2]
      {blockScaleFormat = #nvvm.block_scale_format<ue8m0>,
       kind = #nvvm.block_scale_kind<mxf8f6f4>,
       multiplicandAPtxType = #nvvm.mma_type<e5m2>,
       multiplicandBPtxType = #nvvm.mma_type<e2m3>,
       orderedMetadata,
       scaleVecSize = #nvvm.scale_vec_size<x1>,
       shape = #nvvm.shape<m = 16, n = 8, k = 64>}
      : (i32, i32, f32) -> !llvm.struct<(f32, f32, f32, f32)>
  llvm.return %res : !llvm.struct<(f32, f32, f32, f32)>
}

// CHECK-LABEL: @nvvm_mxf8f6f4_sp_blockscale_mma_e5m2_e3m2
llvm.func @nvvm_mxf8f6f4_sp_blockscale_mma_e5m2_e3m2(
    %a0: i32, %a1: i32, %a2: i32, %a3: i32,
    %b0: i32, %b1: i32, %b2: i32, %b3: i32,
    %c0: f32, %c1: f32, %c2: f32, %c3: f32,
    %meta: i32,
    %scaleA0: i32, %scaleA1: i16, %scaleA2: i16,
    %scaleB0: i32, %scaleB1: i16, %scaleB2: i16) -> !llvm.struct<(f32, f32, f32, f32)> {
  %sel = llvm.mlir.constant(0 : i32) : i32
  // CHECK: call { float, float, float, float } @llvm.nvvm.mma.sp.ordered.metadata.block.scale.m16n8k64.row.col.mxf8f6f4.scale.1x.f32.e5m2.e3m2.f32.ue8m0(i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, float {{%[0-9]+}}, float {{%[0-9]+}}, float {{%[0-9]+}}, float {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 0, i32 {{%[0-9]+}}, i16 {{%[0-9]+}}, i16 {{%[0-9]+}}, i32 {{%[0-9]+}}, i16 {{%[0-9]+}}, i16 {{%[0-9]+}})
  %res = nvvm.mma.sp.block_scale
      A[%a0, %a1, %a2, %a3]
      B[%b0, %b1, %b2, %b3]
      C[%c0, %c1, %c2, %c3]
      sparseMetadata[%meta]
      selector[%sel]
      scaleA[%scaleA0, %scaleA1, %scaleA2]
      scaleB[%scaleB0, %scaleB1, %scaleB2]
      {blockScaleFormat = #nvvm.block_scale_format<ue8m0>,
       kind = #nvvm.block_scale_kind<mxf8f6f4>,
       multiplicandAPtxType = #nvvm.mma_type<e5m2>,
       multiplicandBPtxType = #nvvm.mma_type<e3m2>,
       orderedMetadata,
       scaleVecSize = #nvvm.scale_vec_size<x1>,
       shape = #nvvm.shape<m = 16, n = 8, k = 64>}
      : (i32, i32, f32) -> !llvm.struct<(f32, f32, f32, f32)>
  llvm.return %res : !llvm.struct<(f32, f32, f32, f32)>
}

// CHECK-LABEL: @nvvm_mxf8f6f4_sp_blockscale_mma_e5m2_e4m3
llvm.func @nvvm_mxf8f6f4_sp_blockscale_mma_e5m2_e4m3(
    %a0: i32, %a1: i32, %a2: i32, %a3: i32,
    %b0: i32, %b1: i32, %b2: i32, %b3: i32,
    %c0: f32, %c1: f32, %c2: f32, %c3: f32,
    %meta: i32,
    %scaleA0: i32, %scaleA1: i16, %scaleA2: i16,
    %scaleB0: i32, %scaleB1: i16, %scaleB2: i16) -> !llvm.struct<(f32, f32, f32, f32)> {
  %sel = llvm.mlir.constant(0 : i32) : i32
  // CHECK: call { float, float, float, float } @llvm.nvvm.mma.sp.ordered.metadata.block.scale.m16n8k64.row.col.mxf8f6f4.scale.1x.f32.e5m2.e4m3.f32.ue8m0(i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, float {{%[0-9]+}}, float {{%[0-9]+}}, float {{%[0-9]+}}, float {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 0, i32 {{%[0-9]+}}, i16 {{%[0-9]+}}, i16 {{%[0-9]+}}, i32 {{%[0-9]+}}, i16 {{%[0-9]+}}, i16 {{%[0-9]+}})
  %res = nvvm.mma.sp.block_scale
      A[%a0, %a1, %a2, %a3]
      B[%b0, %b1, %b2, %b3]
      C[%c0, %c1, %c2, %c3]
      sparseMetadata[%meta]
      selector[%sel]
      scaleA[%scaleA0, %scaleA1, %scaleA2]
      scaleB[%scaleB0, %scaleB1, %scaleB2]
      {blockScaleFormat = #nvvm.block_scale_format<ue8m0>,
       kind = #nvvm.block_scale_kind<mxf8f6f4>,
       multiplicandAPtxType = #nvvm.mma_type<e5m2>,
       multiplicandBPtxType = #nvvm.mma_type<e4m3>,
       orderedMetadata,
       scaleVecSize = #nvvm.scale_vec_size<x1>,
       shape = #nvvm.shape<m = 16, n = 8, k = 64>}
      : (i32, i32, f32) -> !llvm.struct<(f32, f32, f32, f32)>
  llvm.return %res : !llvm.struct<(f32, f32, f32, f32)>
}

// CHECK-LABEL: @nvvm_mxf8f6f4_sp_blockscale_mma_e5m2_e5m2
llvm.func @nvvm_mxf8f6f4_sp_blockscale_mma_e5m2_e5m2(
    %a0: i32, %a1: i32, %a2: i32, %a3: i32,
    %b0: i32, %b1: i32, %b2: i32, %b3: i32,
    %c0: f32, %c1: f32, %c2: f32, %c3: f32,
    %meta: i32,
    %scaleA0: i32, %scaleA1: i16, %scaleA2: i16,
    %scaleB0: i32, %scaleB1: i16, %scaleB2: i16) -> !llvm.struct<(f32, f32, f32, f32)> {
  %sel = llvm.mlir.constant(0 : i32) : i32
  // CHECK: call { float, float, float, float } @llvm.nvvm.mma.sp.ordered.metadata.block.scale.m16n8k64.row.col.mxf8f6f4.scale.1x.f32.e5m2.e5m2.f32.ue8m0(i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, float {{%[0-9]+}}, float {{%[0-9]+}}, float {{%[0-9]+}}, float {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 0, i32 {{%[0-9]+}}, i16 {{%[0-9]+}}, i16 {{%[0-9]+}}, i32 {{%[0-9]+}}, i16 {{%[0-9]+}}, i16 {{%[0-9]+}})
  %res = nvvm.mma.sp.block_scale
      A[%a0, %a1, %a2, %a3]
      B[%b0, %b1, %b2, %b3]
      C[%c0, %c1, %c2, %c3]
      sparseMetadata[%meta]
      selector[%sel]
      scaleA[%scaleA0, %scaleA1, %scaleA2]
      scaleB[%scaleB0, %scaleB1, %scaleB2]
      {blockScaleFormat = #nvvm.block_scale_format<ue8m0>,
       kind = #nvvm.block_scale_kind<mxf8f6f4>,
       multiplicandAPtxType = #nvvm.mma_type<e5m2>,
       multiplicandBPtxType = #nvvm.mma_type<e5m2>,
       orderedMetadata,
       scaleVecSize = #nvvm.scale_vec_size<x1>,
       shape = #nvvm.shape<m = 16, n = 8, k = 64>}
      : (i32, i32, f32) -> !llvm.struct<(f32, f32, f32, f32)>
  llvm.return %res : !llvm.struct<(f32, f32, f32, f32)>
}

// CHECK-LABEL: @nvvm_mxf4_sp_blockscale_mma
llvm.func @nvvm_mxf4_sp_blockscale_mma(
    %a0: i32, %a1: i32, %a2: i32, %a3: i32,
    %b0: i32, %b1: i32, %b2: i32, %b3: i32,
    %c0: f32, %c1: f32, %c2: f32, %c3: f32,
    %meta: i32,
    %scaleA0: i32, %scaleA1: i16, %scaleA2: i16,
    %scaleB0: i32, %scaleB1: i16, %scaleB2: i16) -> !llvm.struct<(f32, f32, f32, f32)> {
  %sel = llvm.mlir.constant(0 : i32) : i32
  // CHECK: call { float, float, float, float } @llvm.nvvm.mma.sp.ordered.metadata.block.scale.m16n8k128.row.col.mxf4.scale.2x.f32.e2m1.e2m1.f32.ue8m0(i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, float {{%[0-9]+}}, float {{%[0-9]+}}, float {{%[0-9]+}}, float {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 0, i32 {{%[0-9]+}}, i16 {{%[0-9]+}}, i16 {{%[0-9]+}}, i32 {{%[0-9]+}}, i16 {{%[0-9]+}}, i16 {{%[0-9]+}})
  %res = nvvm.mma.sp.block_scale
      A[%a0, %a1, %a2, %a3]
      B[%b0, %b1, %b2, %b3]
      C[%c0, %c1, %c2, %c3]
      sparseMetadata[%meta]
      selector[%sel]
      scaleA[%scaleA0, %scaleA1, %scaleA2]
      scaleB[%scaleB0, %scaleB1, %scaleB2]
      {blockScaleFormat = #nvvm.block_scale_format<ue8m0>,
       kind = #nvvm.block_scale_kind<mxf4>,
       multiplicandAPtxType = #nvvm.mma_type<e2m1>,
       multiplicandBPtxType = #nvvm.mma_type<e2m1>,
       orderedMetadata,
       scaleVecSize = #nvvm.scale_vec_size<x2>,
       shape = #nvvm.shape<m = 16, n = 8, k = 128>}
      : (i32, i32, f32) -> !llvm.struct<(f32, f32, f32, f32)>
  llvm.return %res : !llvm.struct<(f32, f32, f32, f32)>
}

// CHECK-LABEL: @nvvm_mxf4nvf4_sp_blockscale_mma_ue8m0
llvm.func @nvvm_mxf4nvf4_sp_blockscale_mma_ue8m0(
    %a0: i32, %a1: i32, %a2: i32, %a3: i32,
    %b0: i32, %b1: i32, %b2: i32, %b3: i32,
    %c0: f32, %c1: f32, %c2: f32, %c3: f32,
    %meta: i32,
    %scaleA0: i32, %scaleA1: i16, %scaleA2: i16,
    %scaleB0: i32, %scaleB1: i16, %scaleB2: i16) -> !llvm.struct<(f32, f32, f32, f32)> {
  %sel = llvm.mlir.constant(0 : i32) : i32
  // CHECK: call { float, float, float, float } @llvm.nvvm.mma.sp.ordered.metadata.block.scale.m16n8k128.row.col.mxf4nvf4.scale.2x.f32.e2m1.e2m1.f32.ue8m0(i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, float {{%[0-9]+}}, float {{%[0-9]+}}, float {{%[0-9]+}}, float {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 0, i32 {{%[0-9]+}}, i16 {{%[0-9]+}}, i16 {{%[0-9]+}}, i32 {{%[0-9]+}}, i16 {{%[0-9]+}}, i16 {{%[0-9]+}})
  %res = nvvm.mma.sp.block_scale
      A[%a0, %a1, %a2, %a3]
      B[%b0, %b1, %b2, %b3]
      C[%c0, %c1, %c2, %c3]
      sparseMetadata[%meta]
      selector[%sel]
      scaleA[%scaleA0, %scaleA1, %scaleA2]
      scaleB[%scaleB0, %scaleB1, %scaleB2]
      {blockScaleFormat = #nvvm.block_scale_format<ue8m0>,
       kind = #nvvm.block_scale_kind<mxf4nvf4>,
       multiplicandAPtxType = #nvvm.mma_type<e2m1>,
       multiplicandBPtxType = #nvvm.mma_type<e2m1>,
       orderedMetadata,
       scaleVecSize = #nvvm.scale_vec_size<x2>,
       shape = #nvvm.shape<m = 16, n = 8, k = 128>}
      : (i32, i32, f32) -> !llvm.struct<(f32, f32, f32, f32)>
  llvm.return %res : !llvm.struct<(f32, f32, f32, f32)>
}

// CHECK-LABEL: @nvvm_mxf4nvf4_sp_blockscale_mma_ue4m3
llvm.func @nvvm_mxf4nvf4_sp_blockscale_mma_ue4m3(
    %a0: i32, %a1: i32, %a2: i32, %a3: i32,
    %b0: i32, %b1: i32, %b2: i32, %b3: i32,
    %c0: f32, %c1: f32, %c2: f32, %c3: f32,
    %meta: i32,
    %scaleA0: i32, %scaleA1: i16, %scaleA2: i16,
    %scaleB0: i32, %scaleB1: i16, %scaleB2: i16) -> !llvm.struct<(f32, f32, f32, f32)> {
  %sel = llvm.mlir.constant(0 : i32) : i32
  // CHECK: call { float, float, float, float } @llvm.nvvm.mma.sp.ordered.metadata.block.scale.m16n8k128.row.col.mxf4nvf4.scale.4x.f32.e2m1.e2m1.f32.ue4m3(i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, float {{%[0-9]+}}, float {{%[0-9]+}}, float {{%[0-9]+}}, float {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 0, i32 {{%[0-9]+}}, i16 {{%[0-9]+}}, i16 {{%[0-9]+}}, i32 {{%[0-9]+}}, i16 {{%[0-9]+}}, i16 {{%[0-9]+}})
  %res = nvvm.mma.sp.block_scale
      A[%a0, %a1, %a2, %a3]
      B[%b0, %b1, %b2, %b3]
      C[%c0, %c1, %c2, %c3]
      sparseMetadata[%meta]
      selector[%sel]
      scaleA[%scaleA0, %scaleA1, %scaleA2]
      scaleB[%scaleB0, %scaleB1, %scaleB2]
      {blockScaleFormat = #nvvm.block_scale_format<ue4m3>,
       kind = #nvvm.block_scale_kind<mxf4nvf4>,
       multiplicandAPtxType = #nvvm.mma_type<e2m1>,
       multiplicandBPtxType = #nvvm.mma_type<e2m1>,
       orderedMetadata,
       scaleVecSize = #nvvm.scale_vec_size<x4>,
       shape = #nvvm.shape<m = 16, n = 8, k = 128>}
      : (i32, i32, f32) -> !llvm.struct<(f32, f32, f32, f32)>
  llvm.return %res : !llvm.struct<(f32, f32, f32, f32)>
}

// CHECK-LABEL: @nvvm_mxf4nvf4_sp_blockscale_mma_ue8m0_x4
llvm.func @nvvm_mxf4nvf4_sp_blockscale_mma_ue8m0_x4(
    %a0: i32, %a1: i32, %a2: i32, %a3: i32,
    %b0: i32, %b1: i32, %b2: i32, %b3: i32,
    %c0: f32, %c1: f32, %c2: f32, %c3: f32,
    %meta: i32,
    %scaleA0: i32, %scaleA1: i16, %scaleA2: i16,
    %scaleB0: i32, %scaleB1: i16, %scaleB2: i16) -> !llvm.struct<(f32, f32, f32, f32)> {
  %sel = llvm.mlir.constant(0 : i32) : i32
  // CHECK: call { float, float, float, float } @llvm.nvvm.mma.sp.ordered.metadata.block.scale.m16n8k128.row.col.mxf4nvf4.scale.4x.f32.e2m1.e2m1.f32.ue8m0(i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, float {{%[0-9]+}}, float {{%[0-9]+}}, float {{%[0-9]+}}, float {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 0, i32 {{%[0-9]+}}, i16 {{%[0-9]+}}, i16 {{%[0-9]+}}, i32 {{%[0-9]+}}, i16 {{%[0-9]+}}, i16 {{%[0-9]+}})
  %res = nvvm.mma.sp.block_scale
      A[%a0, %a1, %a2, %a3]
      B[%b0, %b1, %b2, %b3]
      C[%c0, %c1, %c2, %c3]
      sparseMetadata[%meta]
      selector[%sel]
      scaleA[%scaleA0, %scaleA1, %scaleA2]
      scaleB[%scaleB0, %scaleB1, %scaleB2]
      {blockScaleFormat = #nvvm.block_scale_format<ue8m0>,
       kind = #nvvm.block_scale_kind<mxf4nvf4>,
       multiplicandAPtxType = #nvvm.mma_type<e2m1>,
       multiplicandBPtxType = #nvvm.mma_type<e2m1>,
       orderedMetadata,
       scaleVecSize = #nvvm.scale_vec_size<x4>,
       shape = #nvvm.shape<m = 16, n = 8, k = 128>}
      : (i32, i32, f32) -> !llvm.struct<(f32, f32, f32, f32)>
  llvm.return %res : !llvm.struct<(f32, f32, f32, f32)>
}
