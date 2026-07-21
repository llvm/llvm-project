// RUN: mlir-translate --mlir-to-llvmir %s | FileCheck %s

// CHECK-LABEL: @nvvm_mma_sp_ordered_m16n8k16_f16_f16
llvm.func @nvvm_mma_sp_ordered_m16n8k16_f16_f16(
    %a0: vector<2xf16>, %a1: vector<2xf16>,
    %b0: vector<2xf16>, %b1: vector<2xf16>,
    %c0: vector<2xf16>, %c1: vector<2xf16>,
    %meta: i32) -> !llvm.struct<(vector<2xf16>, vector<2xf16>)> {
  %sel = llvm.mlir.constant(0 : i32) : i32
  // CHECK: call { <2 x half>, <2 x half> } @llvm.nvvm.mma.sp.ordered.metadata.m16n8k16.row.col.f16.f16(<2 x half> {{%[0-9]+}}, <2 x half> {{%[0-9]+}}, <2 x half> {{%[0-9]+}}, <2 x half> {{%[0-9]+}}, <2 x half> {{%[0-9]+}}, <2 x half> {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 0)
  %res = nvvm.mma.sp.sync A[%a0, %a1] B[%b0, %b1] C[%c0, %c1]
      sparseMetadata[%meta] selector[%sel]
      {orderedMetadata,
       shape = #nvvm.shape<m = 16, n = 8, k = 16>}
      : (vector<2xf16>, vector<2xf16>, vector<2xf16>) -> !llvm.struct<(vector<2xf16>, vector<2xf16>)>
  llvm.return %res : !llvm.struct<(vector<2xf16>, vector<2xf16>)>
}

// CHECK-LABEL: @nvvm_mma_sp_ordered_m16n8k16_f16_f32
llvm.func @nvvm_mma_sp_ordered_m16n8k16_f16_f32(
    %a0: vector<2xf16>, %a1: vector<2xf16>,
    %b0: vector<2xf16>, %b1: vector<2xf16>,
    %c0: f32, %c1: f32, %c2: f32, %c3: f32,
    %meta: i32) -> !llvm.struct<(f32, f32, f32, f32)> {
  %sel = llvm.mlir.constant(0 : i32) : i32
  // CHECK: call { float, float, float, float } @llvm.nvvm.mma.sp.ordered.metadata.m16n8k16.row.col.f32.f32(<2 x half> {{%[0-9]+}}, <2 x half> {{%[0-9]+}}, <2 x half> {{%[0-9]+}}, <2 x half> {{%[0-9]+}}, float {{%[0-9]+}}, float {{%[0-9]+}}, float {{%[0-9]+}}, float {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 0)
  %res = nvvm.mma.sp.sync A[%a0, %a1] B[%b0, %b1] C[%c0, %c1, %c2, %c3]
      sparseMetadata[%meta] selector[%sel]
      {orderedMetadata,
       shape = #nvvm.shape<m = 16, n = 8, k = 16>}
      : (vector<2xf16>, vector<2xf16>, f32) -> !llvm.struct<(f32, f32, f32, f32)>
  llvm.return %res : !llvm.struct<(f32, f32, f32, f32)>
}

// CHECK-LABEL: @nvvm_mma_sp_ordered_m16n8k32_f16_f16
llvm.func @nvvm_mma_sp_ordered_m16n8k32_f16_f16(
    %a0: vector<2xf16>, %a1: vector<2xf16>, %a2: vector<2xf16>, %a3: vector<2xf16>,
    %b0: vector<2xf16>, %b1: vector<2xf16>, %b2: vector<2xf16>, %b3: vector<2xf16>,
    %c0: vector<2xf16>, %c1: vector<2xf16>,
    %meta: i32) -> !llvm.struct<(vector<2xf16>, vector<2xf16>)> {
  %sel = llvm.mlir.constant(0 : i32) : i32
  // CHECK: call { <2 x half>, <2 x half> } @llvm.nvvm.mma.sp.ordered.metadata.m16n8k32.row.col.f16.f16(<2 x half> {{%[0-9]+}}, <2 x half> {{%[0-9]+}}, <2 x half> {{%[0-9]+}}, <2 x half> {{%[0-9]+}}, <2 x half> {{%[0-9]+}}, <2 x half> {{%[0-9]+}}, <2 x half> {{%[0-9]+}}, <2 x half> {{%[0-9]+}}, <2 x half> {{%[0-9]+}}, <2 x half> {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 0)
  %res = nvvm.mma.sp.sync A[%a0, %a1, %a2, %a3] B[%b0, %b1, %b2, %b3] C[%c0, %c1]
      sparseMetadata[%meta] selector[%sel]
      {orderedMetadata,
       shape = #nvvm.shape<m = 16, n = 8, k = 32>}
      : (vector<2xf16>, vector<2xf16>, vector<2xf16>) -> !llvm.struct<(vector<2xf16>, vector<2xf16>)>
  llvm.return %res : !llvm.struct<(vector<2xf16>, vector<2xf16>)>
}

// CHECK-LABEL: @nvvm_mma_sp_ordered_m16n8k32_f16_f32
llvm.func @nvvm_mma_sp_ordered_m16n8k32_f16_f32(
    %a0: vector<2xf16>, %a1: vector<2xf16>, %a2: vector<2xf16>, %a3: vector<2xf16>,
    %b0: vector<2xf16>, %b1: vector<2xf16>, %b2: vector<2xf16>, %b3: vector<2xf16>,
    %c0: f32, %c1: f32, %c2: f32, %c3: f32,
    %meta: i32) -> !llvm.struct<(f32, f32, f32, f32)> {
  %sel = llvm.mlir.constant(0 : i32) : i32
  // CHECK: call { float, float, float, float } @llvm.nvvm.mma.sp.ordered.metadata.m16n8k32.row.col.f32.f32(<2 x half> {{%[0-9]+}}, <2 x half> {{%[0-9]+}}, <2 x half> {{%[0-9]+}}, <2 x half> {{%[0-9]+}}, <2 x half> {{%[0-9]+}}, <2 x half> {{%[0-9]+}}, <2 x half> {{%[0-9]+}}, <2 x half> {{%[0-9]+}}, float {{%[0-9]+}}, float {{%[0-9]+}}, float {{%[0-9]+}}, float {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 0)
  %res = nvvm.mma.sp.sync A[%a0, %a1, %a2, %a3] B[%b0, %b1, %b2, %b3] C[%c0, %c1, %c2, %c3]
      sparseMetadata[%meta] selector[%sel]
      {orderedMetadata,
       shape = #nvvm.shape<m = 16, n = 8, k = 32>}
      : (vector<2xf16>, vector<2xf16>, f32) -> !llvm.struct<(f32, f32, f32, f32)>
  llvm.return %res : !llvm.struct<(f32, f32, f32, f32)>
}

// CHECK-LABEL: @nvvm_mma_sp_ordered_m16n8k16_bf16_f32
llvm.func @nvvm_mma_sp_ordered_m16n8k16_bf16_f32(
    %a0: i32, %a1: i32,
    %b0: i32, %b1: i32,
    %c0: f32, %c1: f32, %c2: f32, %c3: f32,
    %meta: i32) -> !llvm.struct<(f32, f32, f32, f32)> {
  %sel = llvm.mlir.constant(0 : i32) : i32
  // CHECK: call { float, float, float, float } @llvm.nvvm.mma.sp.ordered.metadata.m16n8k16.row.col.bf16(i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, float {{%[0-9]+}}, float {{%[0-9]+}}, float {{%[0-9]+}}, float {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 0)
  %res = nvvm.mma.sp.sync A[%a0, %a1] B[%b0, %b1] C[%c0, %c1, %c2, %c3]
      sparseMetadata[%meta] selector[%sel]
      {multiplicandAPtxType = #nvvm.mma_type<bf16>,
       multiplicandBPtxType = #nvvm.mma_type<bf16>,
       orderedMetadata,
       shape = #nvvm.shape<m = 16, n = 8, k = 16>}
      : (i32, i32, f32) -> !llvm.struct<(f32, f32, f32, f32)>
  llvm.return %res : !llvm.struct<(f32, f32, f32, f32)>
}

// CHECK-LABEL: @nvvm_mma_sp_ordered_m16n8k32_bf16_f32
llvm.func @nvvm_mma_sp_ordered_m16n8k32_bf16_f32(
    %a0: i32, %a1: i32, %a2: i32, %a3: i32,
    %b0: i32, %b1: i32, %b2: i32, %b3: i32,
    %c0: f32, %c1: f32, %c2: f32, %c3: f32,
    %meta: i32) -> !llvm.struct<(f32, f32, f32, f32)> {
  %sel = llvm.mlir.constant(0 : i32) : i32
  // CHECK: call { float, float, float, float } @llvm.nvvm.mma.sp.ordered.metadata.m16n8k32.row.col.bf16(i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, float {{%[0-9]+}}, float {{%[0-9]+}}, float {{%[0-9]+}}, float {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 0)
  %res = nvvm.mma.sp.sync A[%a0, %a1, %a2, %a3] B[%b0, %b1, %b2, %b3] C[%c0, %c1, %c2, %c3]
      sparseMetadata[%meta] selector[%sel]
      {multiplicandAPtxType = #nvvm.mma_type<bf16>,
       multiplicandBPtxType = #nvvm.mma_type<bf16>,
       orderedMetadata,
       shape = #nvvm.shape<m = 16, n = 8, k = 32>}
      : (i32, i32, f32) -> !llvm.struct<(f32, f32, f32, f32)>
  llvm.return %res : !llvm.struct<(f32, f32, f32, f32)>
}

// CHECK-LABEL: @nvvm_mma_sp_ordered_m16n8k8_tf32_f32
llvm.func @nvvm_mma_sp_ordered_m16n8k8_tf32_f32(
    %a0: i32, %a1: i32,
    %b0: i32, %b1: i32,
    %c0: f32, %c1: f32, %c2: f32, %c3: f32,
    %meta: i32) -> !llvm.struct<(f32, f32, f32, f32)> {
  %sel = llvm.mlir.constant(0 : i32) : i32
  // CHECK: call { float, float, float, float } @llvm.nvvm.mma.sp.ordered.metadata.m16n8k8.row.col.tf32(i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, float {{%[0-9]+}}, float {{%[0-9]+}}, float {{%[0-9]+}}, float {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 0)
  %res = nvvm.mma.sp.sync A[%a0, %a1] B[%b0, %b1] C[%c0, %c1, %c2, %c3]
      sparseMetadata[%meta] selector[%sel]
      {multiplicandAPtxType = #nvvm.mma_type<tf32>,
       multiplicandBPtxType = #nvvm.mma_type<tf32>,
       orderedMetadata,
       shape = #nvvm.shape<m = 16, n = 8, k = 8>}
      : (i32, i32, f32) -> !llvm.struct<(f32, f32, f32, f32)>
  llvm.return %res : !llvm.struct<(f32, f32, f32, f32)>
}

// CHECK-LABEL: @nvvm_mma_sp_ordered_m16n8k16_tf32_f32
llvm.func @nvvm_mma_sp_ordered_m16n8k16_tf32_f32(
    %a0: i32, %a1: i32, %a2: i32, %a3: i32,
    %b0: i32, %b1: i32, %b2: i32, %b3: i32,
    %c0: f32, %c1: f32, %c2: f32, %c3: f32,
    %meta: i32) -> !llvm.struct<(f32, f32, f32, f32)> {
  %sel = llvm.mlir.constant(0 : i32) : i32
  // CHECK: call { float, float, float, float } @llvm.nvvm.mma.sp.ordered.metadata.m16n8k16.row.col.tf32(i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, float {{%[0-9]+}}, float {{%[0-9]+}}, float {{%[0-9]+}}, float {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 0)
  %res = nvvm.mma.sp.sync A[%a0, %a1, %a2, %a3] B[%b0, %b1, %b2, %b3] C[%c0, %c1, %c2, %c3]
      sparseMetadata[%meta] selector[%sel]
      {multiplicandAPtxType = #nvvm.mma_type<tf32>,
       multiplicandBPtxType = #nvvm.mma_type<tf32>,
       orderedMetadata,
       shape = #nvvm.shape<m = 16, n = 8, k = 16>}
      : (i32, i32, f32) -> !llvm.struct<(f32, f32, f32, f32)>
  llvm.return %res : !llvm.struct<(f32, f32, f32, f32)>
}

// CHECK-LABEL: @nvvm_mma_sp_ordered_m16n8k32_s8_s32
llvm.func @nvvm_mma_sp_ordered_m16n8k32_s8_s32(
    %a0: i32, %a1: i32,
    %b0: i32, %b1: i32,
    %c0: i32, %c1: i32, %c2: i32, %c3: i32,
    %meta: i32) -> !llvm.struct<(i32, i32, i32, i32)> {
  %sel = llvm.mlir.constant(0 : i32) : i32
  // CHECK: call { i32, i32, i32, i32 } @llvm.nvvm.mma.sp.ordered.metadata.m16n8k32.row.col.s8(i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 0)
  %res = nvvm.mma.sp.sync A[%a0, %a1] B[%b0, %b1] C[%c0, %c1, %c2, %c3]
      sparseMetadata[%meta] selector[%sel]
      {intOverflowBehavior = #nvvm.mma_int_overflow<wrapped>,
       multiplicandAPtxType = #nvvm.mma_type<s8>,
       multiplicandBPtxType = #nvvm.mma_type<s8>,
       orderedMetadata,
       shape = #nvvm.shape<m = 16, n = 8, k = 32>}
      : (i32, i32, i32) -> !llvm.struct<(i32, i32, i32, i32)>
  llvm.return %res : !llvm.struct<(i32, i32, i32, i32)>
}

// CHECK-LABEL: @nvvm_mma_sp_ordered_m16n8k32_s8_s32_satfinite
llvm.func @nvvm_mma_sp_ordered_m16n8k32_s8_s32_satfinite(
    %a0: i32, %a1: i32,
    %b0: i32, %b1: i32,
    %c0: i32, %c1: i32, %c2: i32, %c3: i32,
    %meta: i32) -> !llvm.struct<(i32, i32, i32, i32)> {
  %sel = llvm.mlir.constant(0 : i32) : i32
  // CHECK: call { i32, i32, i32, i32 } @llvm.nvvm.mma.sp.ordered.metadata.m16n8k32.row.col.satfinite.s8(i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 0)
  %res = nvvm.mma.sp.sync A[%a0, %a1] B[%b0, %b1] C[%c0, %c1, %c2, %c3]
      sparseMetadata[%meta] selector[%sel]
      {intOverflowBehavior = #nvvm.mma_int_overflow<satfinite>,
       multiplicandAPtxType = #nvvm.mma_type<s8>,
       multiplicandBPtxType = #nvvm.mma_type<s8>,
       orderedMetadata,
       shape = #nvvm.shape<m = 16, n = 8, k = 32>}
      : (i32, i32, i32) -> !llvm.struct<(i32, i32, i32, i32)>
  llvm.return %res : !llvm.struct<(i32, i32, i32, i32)>
}

// CHECK-LABEL: @nvvm_mma_sp_ordered_m16n8k64_s8_s32
llvm.func @nvvm_mma_sp_ordered_m16n8k64_s8_s32(
    %a0: i32, %a1: i32, %a2: i32, %a3: i32,
    %b0: i32, %b1: i32, %b2: i32, %b3: i32,
    %c0: i32, %c1: i32, %c2: i32, %c3: i32,
    %meta: i32) -> !llvm.struct<(i32, i32, i32, i32)> {
  %sel = llvm.mlir.constant(0 : i32) : i32
  // CHECK: call { i32, i32, i32, i32 } @llvm.nvvm.mma.sp.ordered.metadata.m16n8k64.row.col.s8(i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 0)
  %res = nvvm.mma.sp.sync A[%a0, %a1, %a2, %a3] B[%b0, %b1, %b2, %b3] C[%c0, %c1, %c2, %c3]
      sparseMetadata[%meta] selector[%sel]
      {intOverflowBehavior = #nvvm.mma_int_overflow<wrapped>,
       multiplicandAPtxType = #nvvm.mma_type<s8>,
       multiplicandBPtxType = #nvvm.mma_type<s8>,
       orderedMetadata,
       shape = #nvvm.shape<m = 16, n = 8, k = 64>}
      : (i32, i32, i32) -> !llvm.struct<(i32, i32, i32, i32)>
  llvm.return %res : !llvm.struct<(i32, i32, i32, i32)>
}

// CHECK-LABEL: @nvvm_mma_sp_ordered_m16n8k32_u8_s32
llvm.func @nvvm_mma_sp_ordered_m16n8k32_u8_s32(
    %a0: i32, %a1: i32,
    %b0: i32, %b1: i32,
    %c0: i32, %c1: i32, %c2: i32, %c3: i32,
    %meta: i32) -> !llvm.struct<(i32, i32, i32, i32)> {
  %sel = llvm.mlir.constant(0 : i32) : i32
  // CHECK: call { i32, i32, i32, i32 } @llvm.nvvm.mma.sp.ordered.metadata.m16n8k32.row.col.u8(i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 0)
  %res = nvvm.mma.sp.sync A[%a0, %a1] B[%b0, %b1] C[%c0, %c1, %c2, %c3]
      sparseMetadata[%meta] selector[%sel]
      {intOverflowBehavior = #nvvm.mma_int_overflow<wrapped>,
       multiplicandAPtxType = #nvvm.mma_type<u8>,
       multiplicandBPtxType = #nvvm.mma_type<u8>,
       orderedMetadata,
       shape = #nvvm.shape<m = 16, n = 8, k = 32>}
      : (i32, i32, i32) -> !llvm.struct<(i32, i32, i32, i32)>
  llvm.return %res : !llvm.struct<(i32, i32, i32, i32)>
}

// CHECK-LABEL: @nvvm_mma_sp_ordered_m16n8k64_u8_s32
llvm.func @nvvm_mma_sp_ordered_m16n8k64_u8_s32(
    %a0: i32, %a1: i32, %a2: i32, %a3: i32,
    %b0: i32, %b1: i32, %b2: i32, %b3: i32,
    %c0: i32, %c1: i32, %c2: i32, %c3: i32,
    %meta: i32) -> !llvm.struct<(i32, i32, i32, i32)> {
  %sel = llvm.mlir.constant(0 : i32) : i32
  // CHECK: call { i32, i32, i32, i32 } @llvm.nvvm.mma.sp.ordered.metadata.m16n8k64.row.col.u8(i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 0)
  %res = nvvm.mma.sp.sync A[%a0, %a1, %a2, %a3] B[%b0, %b1, %b2, %b3] C[%c0, %c1, %c2, %c3]
      sparseMetadata[%meta] selector[%sel]
      {intOverflowBehavior = #nvvm.mma_int_overflow<wrapped>,
       multiplicandAPtxType = #nvvm.mma_type<u8>,
       multiplicandBPtxType = #nvvm.mma_type<u8>,
       orderedMetadata,
       shape = #nvvm.shape<m = 16, n = 8, k = 64>}
      : (i32, i32, i32) -> !llvm.struct<(i32, i32, i32, i32)>
  llvm.return %res : !llvm.struct<(i32, i32, i32, i32)>
}

// CHECK-LABEL: @nvvm_mma_sp_ordered_m16n8k64_s4_s32
llvm.func @nvvm_mma_sp_ordered_m16n8k64_s4_s32(
    %a0: i32, %a1: i32,
    %b0: i32, %b1: i32,
    %c0: i32, %c1: i32, %c2: i32, %c3: i32,
    %meta: i32) -> !llvm.struct<(i32, i32, i32, i32)> {
  %sel = llvm.mlir.constant(0 : i32) : i32
  // CHECK: call { i32, i32, i32, i32 } @llvm.nvvm.mma.sp.ordered.metadata.m16n8k64.row.col.s4(i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 0)
  %res = nvvm.mma.sp.sync A[%a0, %a1] B[%b0, %b1] C[%c0, %c1, %c2, %c3]
      sparseMetadata[%meta] selector[%sel]
      {intOverflowBehavior = #nvvm.mma_int_overflow<wrapped>,
       multiplicandAPtxType = #nvvm.mma_type<s4>,
       multiplicandBPtxType = #nvvm.mma_type<s4>,
       orderedMetadata,
       shape = #nvvm.shape<m = 16, n = 8, k = 64>}
      : (i32, i32, i32) -> !llvm.struct<(i32, i32, i32, i32)>
  llvm.return %res : !llvm.struct<(i32, i32, i32, i32)>
}

// CHECK-LABEL: @nvvm_mma_sp_ordered_m16n8k128_s4_s32
llvm.func @nvvm_mma_sp_ordered_m16n8k128_s4_s32(
    %a0: i32, %a1: i32, %a2: i32, %a3: i32,
    %b0: i32, %b1: i32, %b2: i32, %b3: i32,
    %c0: i32, %c1: i32, %c2: i32, %c3: i32,
    %meta: i32) -> !llvm.struct<(i32, i32, i32, i32)> {
  %sel = llvm.mlir.constant(0 : i32) : i32
  // CHECK: call { i32, i32, i32, i32 } @llvm.nvvm.mma.sp.ordered.metadata.m16n8k128.row.col.s4(i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 0)
  %res = nvvm.mma.sp.sync A[%a0, %a1, %a2, %a3] B[%b0, %b1, %b2, %b3] C[%c0, %c1, %c2, %c3]
      sparseMetadata[%meta] selector[%sel]
      {intOverflowBehavior = #nvvm.mma_int_overflow<wrapped>,
       multiplicandAPtxType = #nvvm.mma_type<s4>,
       multiplicandBPtxType = #nvvm.mma_type<s4>,
       orderedMetadata,
       shape = #nvvm.shape<m = 16, n = 8, k = 128>}
      : (i32, i32, i32) -> !llvm.struct<(i32, i32, i32, i32)>
  llvm.return %res : !llvm.struct<(i32, i32, i32, i32)>
}

// CHECK-LABEL: @nvvm_mma_sp_ordered_m16n8k64_u4_s32
llvm.func @nvvm_mma_sp_ordered_m16n8k64_u4_s32(
    %a0: i32, %a1: i32,
    %b0: i32, %b1: i32,
    %c0: i32, %c1: i32, %c2: i32, %c3: i32,
    %meta: i32) -> !llvm.struct<(i32, i32, i32, i32)> {
  %sel = llvm.mlir.constant(0 : i32) : i32
  // CHECK: call { i32, i32, i32, i32 } @llvm.nvvm.mma.sp.ordered.metadata.m16n8k64.row.col.u4(i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 0)
  %res = nvvm.mma.sp.sync A[%a0, %a1] B[%b0, %b1] C[%c0, %c1, %c2, %c3]
      sparseMetadata[%meta] selector[%sel]
      {intOverflowBehavior = #nvvm.mma_int_overflow<wrapped>,
       multiplicandAPtxType = #nvvm.mma_type<u4>,
       multiplicandBPtxType = #nvvm.mma_type<u4>,
       orderedMetadata,
       shape = #nvvm.shape<m = 16, n = 8, k = 64>}
      : (i32, i32, i32) -> !llvm.struct<(i32, i32, i32, i32)>
  llvm.return %res : !llvm.struct<(i32, i32, i32, i32)>
}

// CHECK-LABEL: @nvvm_mma_sp_ordered_m16n8k128_u4_s32
llvm.func @nvvm_mma_sp_ordered_m16n8k128_u4_s32(
    %a0: i32, %a1: i32, %a2: i32, %a3: i32,
    %b0: i32, %b1: i32, %b2: i32, %b3: i32,
    %c0: i32, %c1: i32, %c2: i32, %c3: i32,
    %meta: i32) -> !llvm.struct<(i32, i32, i32, i32)> {
  %sel = llvm.mlir.constant(0 : i32) : i32
  // CHECK: call { i32, i32, i32, i32 } @llvm.nvvm.mma.sp.ordered.metadata.m16n8k128.row.col.u4(i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 0)
  %res = nvvm.mma.sp.sync A[%a0, %a1, %a2, %a3] B[%b0, %b1, %b2, %b3] C[%c0, %c1, %c2, %c3]
      sparseMetadata[%meta] selector[%sel]
      {intOverflowBehavior = #nvvm.mma_int_overflow<wrapped>,
       multiplicandAPtxType = #nvvm.mma_type<u4>,
       multiplicandBPtxType = #nvvm.mma_type<u4>,
       orderedMetadata,
       shape = #nvvm.shape<m = 16, n = 8, k = 128>}
      : (i32, i32, i32) -> !llvm.struct<(i32, i32, i32, i32)>
  llvm.return %res : !llvm.struct<(i32, i32, i32, i32)>
}

// CHECK-LABEL: @nvvm_mma_sp_ordered_m16n8k64_e4m3_f16
llvm.func @nvvm_mma_sp_ordered_m16n8k64_e4m3_f16(
    %a0: i32, %a1: i32, %a2: i32, %a3: i32,
    %b0: i32, %b1: i32, %b2: i32, %b3: i32,
    %c0: vector<2xf16>, %c1: vector<2xf16>,
    %meta: i32) -> !llvm.struct<(vector<2xf16>, vector<2xf16>)> {
  %sel = llvm.mlir.constant(0 : i32) : i32
  // CHECK: call { <2 x half>, <2 x half> } @llvm.nvvm.mma.sp.ordered.metadata.m16n8k64.row.col.kind.f8f6f4.f16.e4m3.e4m3.f16(i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, <2 x half> {{%[0-9]+}}, <2 x half> {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 0)
  %res = nvvm.mma.sp.sync A[%a0, %a1, %a2, %a3] B[%b0, %b1, %b2, %b3] C[%c0, %c1]
      sparseMetadata[%meta] selector[%sel]
      {multiplicandAPtxType = #nvvm.mma_type<e4m3>,
       multiplicandBPtxType = #nvvm.mma_type<e4m3>,
       orderedMetadata,
       shape = #nvvm.shape<m = 16, n = 8, k = 64>}
      : (i32, i32, vector<2xf16>) -> !llvm.struct<(vector<2xf16>, vector<2xf16>)>
  llvm.return %res : !llvm.struct<(vector<2xf16>, vector<2xf16>)>
}

// CHECK-LABEL: @nvvm_mma_sp_ordered_m16n8k64_e4m3_f32
llvm.func @nvvm_mma_sp_ordered_m16n8k64_e4m3_f32(
    %a0: i32, %a1: i32, %a2: i32, %a3: i32,
    %b0: i32, %b1: i32, %b2: i32, %b3: i32,
    %c0: f32, %c1: f32, %c2: f32, %c3: f32,
    %meta: i32) -> !llvm.struct<(f32, f32, f32, f32)> {
  %sel = llvm.mlir.constant(0 : i32) : i32
  // CHECK: call { float, float, float, float } @llvm.nvvm.mma.sp.ordered.metadata.m16n8k64.row.col.f32.e4m3.e4m3.f32(i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, float {{%[0-9]+}}, float {{%[0-9]+}}, float {{%[0-9]+}}, float {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 0)
  %res = nvvm.mma.sp.sync A[%a0, %a1, %a2, %a3] B[%b0, %b1, %b2, %b3] C[%c0, %c1, %c2, %c3]
      sparseMetadata[%meta] selector[%sel]
      {multiplicandAPtxType = #nvvm.mma_type<e4m3>,
       multiplicandBPtxType = #nvvm.mma_type<e4m3>,
       orderedMetadata,
       shape = #nvvm.shape<m = 16, n = 8, k = 64>}
      : (i32, i32, f32) -> !llvm.struct<(f32, f32, f32, f32)>
  llvm.return %res : !llvm.struct<(f32, f32, f32, f32)>
}

// CHECK-LABEL: @nvvm_mma_sp_ordered_m16n8k64_e5m2_f16
llvm.func @nvvm_mma_sp_ordered_m16n8k64_e5m2_f16(
    %a0: i32, %a1: i32, %a2: i32, %a3: i32,
    %b0: i32, %b1: i32, %b2: i32, %b3: i32,
    %c0: vector<2xf16>, %c1: vector<2xf16>,
    %meta: i32) -> !llvm.struct<(vector<2xf16>, vector<2xf16>)> {
  %sel = llvm.mlir.constant(0 : i32) : i32
  // CHECK: call { <2 x half>, <2 x half> } @llvm.nvvm.mma.sp.ordered.metadata.m16n8k64.row.col.kind.f8f6f4.f16.e5m2.e5m2.f16(i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, <2 x half> {{%[0-9]+}}, <2 x half> {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 0)
  %res = nvvm.mma.sp.sync A[%a0, %a1, %a2, %a3] B[%b0, %b1, %b2, %b3] C[%c0, %c1]
      sparseMetadata[%meta] selector[%sel]
      {multiplicandAPtxType = #nvvm.mma_type<e5m2>,
       multiplicandBPtxType = #nvvm.mma_type<e5m2>,
       orderedMetadata,
       shape = #nvvm.shape<m = 16, n = 8, k = 64>}
      : (i32, i32, vector<2xf16>) -> !llvm.struct<(vector<2xf16>, vector<2xf16>)>
  llvm.return %res : !llvm.struct<(vector<2xf16>, vector<2xf16>)>
}

// CHECK-LABEL: @nvvm_mma_sp_ordered_m16n8k64_e5m2_f32
llvm.func @nvvm_mma_sp_ordered_m16n8k64_e5m2_f32(
    %a0: i32, %a1: i32, %a2: i32, %a3: i32,
    %b0: i32, %b1: i32, %b2: i32, %b3: i32,
    %c0: f32, %c1: f32, %c2: f32, %c3: f32,
    %meta: i32) -> !llvm.struct<(f32, f32, f32, f32)> {
  %sel = llvm.mlir.constant(0 : i32) : i32
  // CHECK: call { float, float, float, float } @llvm.nvvm.mma.sp.ordered.metadata.m16n8k64.row.col.f32.e5m2.e5m2.f32(i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 {{%[0-9]+}}, float {{%[0-9]+}}, float {{%[0-9]+}}, float {{%[0-9]+}}, float {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 0)
  %res = nvvm.mma.sp.sync A[%a0, %a1, %a2, %a3] B[%b0, %b1, %b2, %b3] C[%c0, %c1, %c2, %c3]
      sparseMetadata[%meta] selector[%sel]
      {multiplicandAPtxType = #nvvm.mma_type<e5m2>,
       multiplicandBPtxType = #nvvm.mma_type<e5m2>,
       orderedMetadata,
       shape = #nvvm.shape<m = 16, n = 8, k = 64>}
      : (i32, i32, f32) -> !llvm.struct<(f32, f32, f32, f32)>
  llvm.return %res : !llvm.struct<(f32, f32, f32, f32)>
}
