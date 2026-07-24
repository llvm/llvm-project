// RUN: mlir-opt --xegpu-wg-to-sg-distribute -split-input-file %s | FileCheck %s

// The SLM privatization pre-phase of WG-to-SG distribution demotes a shared
// local memory (space 3) buffer to subgroup-private memory (space 4) when every
// matrix access to it uses the same sg_layout, sg_data, data shape and offsets,
// and the workgroup tile is evenly distributed to the subgroups (no broadcast).
// In that case each subgroup reads/writes an identical, non-overlapping region,
// so the buffer and mem_desc are shrunk to the per-subgroup size and the matrix
// ops are re-indexed with local, subgroup-id-free offsets.

gpu.module @test {
  // A buffer whose load_matrix and store_matrix share the same sg_layout,
  // sg_data and offsets is private to each subgroup: it is moved to memory space
  // 4 and shrunk from 64x128 to the per-subgroup 32x32 tile.
  // CHECK-LABEL: gpu.func @privatize_same_offset_layout
  // CHECK: %[[ALLOCA:.*]] = memref.alloca() : memref<4096xi8, 4>
  // CHECK: %[[MD:.*]] = xegpu.create_mem_desc %[[ALLOCA]] : memref<4096xi8, 4> -> !xegpu.mem_desc<32x32xf32>
  // CHECK: xegpu.load_matrix %[[MD]][0, 0] : !xegpu.mem_desc<32x32xf32> -> vector<32x32xf32>
  // CHECK: xegpu.store_matrix %{{.*}}, %[[MD]][0, 0] : vector<32x32xf32>, !xegpu.mem_desc<32x32xf32>
  gpu.func @privatize_same_offset_layout() {
    %cst = arith.constant dense<1.0> : vector<64x128xf32>
    %a = memref.alloca() : memref<32768xi8, 3>
    %md = xegpu.create_mem_desc %a : memref<32768xi8, 3> -> !xegpu.mem_desc<64x128xf32>
    %ld = xegpu.load_matrix %md[0, 0] <{layout = #xegpu.layout<sg_layout = [2, 4], sg_data = [32, 32]>}>
      : !xegpu.mem_desc<64x128xf32> -> vector<64x128xf32>
    xegpu.store_matrix %cst, %md[0, 0] <{layout = #xegpu.layout<sg_layout = [2, 4], sg_data = [32, 32]>}>
      : vector<64x128xf32>, !xegpu.mem_desc<64x128xf32>
    gpu.return
  }
}

// -----

gpu.module @test {
  // When a dimension spans multiple distribution rounds (128 / sg_layout 8 = 16
  // = 2 rounds of sg_data 8), the shrunk buffer holds both rounds and each round
  // is indexed with a local offset (0 and 8).
  // CHECK-LABEL: gpu.func @privatize_multi_round
  // CHECK: %[[ALLOCA:.*]] = memref.alloca() : memref<64xi8, 4>
  // CHECK: %[[MD:.*]] = xegpu.create_mem_desc %[[ALLOCA]] : memref<64xi8, 4> -> !xegpu.mem_desc<16xf32>
  // CHECK: xegpu.store_matrix %{{.*}}, %[[MD]][0] : vector<8xf32>, !xegpu.mem_desc<16xf32>
  // CHECK: xegpu.store_matrix %{{.*}}, %[[MD]][8] : vector<8xf32>, !xegpu.mem_desc<16xf32>
  // CHECK: xegpu.load_matrix %[[MD]][0] : !xegpu.mem_desc<16xf32> -> vector<8xf32>
  // CHECK: xegpu.load_matrix %[[MD]][8] : !xegpu.mem_desc<16xf32> -> vector<8xf32>
  gpu.func @privatize_multi_round() {
    %cst = arith.constant dense<1.0> : vector<128xf32>
    %a = memref.alloca() : memref<512xi8, 3>
    %md = xegpu.create_mem_desc %a : memref<512xi8, 3> -> !xegpu.mem_desc<128xf32>
    xegpu.store_matrix %cst, %md[0] <{layout = #xegpu.layout<sg_layout = [8], sg_data = [8]>}>
      : vector<128xf32>, !xegpu.mem_desc<128xf32>
    %l = xegpu.load_matrix %md[0] <{layout = #xegpu.layout<sg_layout = [8], sg_data = [8]>}>
      : !xegpu.mem_desc<128xf32> -> vector<128xf32>
    gpu.return
  }
}

// -----

gpu.module @test {
  // Accesses use different offsets, so the region seen by each subgroup is not
  // identical: the buffer stays in shared local memory (space 3), full size.
  // CHECK-LABEL: gpu.func @no_privatize_diff_offsets
  // CHECK: %[[ALLOCA:.*]] = memref.alloca() : memref<65536xi8, 3>
  // CHECK: xegpu.create_mem_desc %[[ALLOCA]] : memref<65536xi8, 3> -> !xegpu.mem_desc<128x128xf32>
  gpu.func @no_privatize_diff_offsets() {
    %cst = arith.constant dense<1.0> : vector<64x128xf32>
    %a = memref.alloca() : memref<65536xi8, 3>
    %md = xegpu.create_mem_desc %a : memref<65536xi8, 3> -> !xegpu.mem_desc<128x128xf32>
    %ld = xegpu.load_matrix %md[0, 0] <{layout = #xegpu.layout<sg_layout = [2, 4], sg_data = [32, 32]>}>
      : !xegpu.mem_desc<128x128xf32> -> vector<64x128xf32>
    xegpu.store_matrix %cst, %md[32, 0] <{layout = #xegpu.layout<sg_layout = [2, 4], sg_data = [32, 32]>}>
      : vector<64x128xf32>, !xegpu.mem_desc<128x128xf32>
    gpu.return
  }
}

// -----

gpu.module @test {
  // Accesses use different sg_layouts, so subgroups partition the region
  // differently: the buffer stays in shared local memory (space 3).
  // CHECK-LABEL: gpu.func @no_privatize_diff_layout
  // CHECK: %[[ALLOCA:.*]] = memref.alloca() : memref<32768xi8, 3>
  // CHECK: xegpu.create_mem_desc %[[ALLOCA]] : memref<32768xi8, 3> -> !xegpu.mem_desc<64x128xf32>
  gpu.func @no_privatize_diff_layout() {
    %cst = arith.constant dense<1.0> : vector<64x128xf32>
    %a = memref.alloca() : memref<32768xi8, 3>
    %md = xegpu.create_mem_desc %a : memref<32768xi8, 3> -> !xegpu.mem_desc<64x128xf32>
    %ld = xegpu.load_matrix %md[0, 0] <{layout = #xegpu.layout<sg_layout = [2, 4], sg_data = [32, 32]>}>
      : !xegpu.mem_desc<64x128xf32> -> vector<64x128xf32>
    xegpu.store_matrix %cst, %md[0, 0] <{layout = #xegpu.layout<sg_layout = [4, 2], sg_data = [16, 64]>}>
      : vector<64x128xf32>, !xegpu.mem_desc<64x128xf32>
    gpu.return
  }
}

// -----

gpu.module @test {
  // Accesses use different data (vector) shapes, so the per-subgroup regions
  // differ: the buffer stays in shared local memory (space 3).
  // CHECK-LABEL: gpu.func @no_privatize_diff_data_shape
  // CHECK: %[[ALLOCA:.*]] = memref.alloca() : memref<32768xi8, 3>
  // CHECK: xegpu.create_mem_desc %[[ALLOCA]] : memref<32768xi8, 3> -> !xegpu.mem_desc<64x128xf32>
  gpu.func @no_privatize_diff_data_shape() {
    %c0 = arith.constant dense<1.0> : vector<64x128xf32>
    %c1 = arith.constant dense<1.0> : vector<32x128xf32>
    %a = memref.alloca() : memref<32768xi8, 3>
    %md = xegpu.create_mem_desc %a : memref<32768xi8, 3> -> !xegpu.mem_desc<64x128xf32>
    xegpu.store_matrix %c0, %md[0, 0] <{layout = #xegpu.layout<sg_layout = [2, 4], sg_data = [32, 32]>}>
      : vector<64x128xf32>, !xegpu.mem_desc<64x128xf32>
    %ld = xegpu.load_matrix %md[0, 0] <{layout = #xegpu.layout<sg_layout = [2, 4], sg_data = [16, 32]>}>
      : !xegpu.mem_desc<64x128xf32> -> vector<32x128xf32>
    gpu.return
  }
}

// -----

gpu.module @test {
  // The workgroup tile wraps around (sg_layout 4 * sg_data 8 = 32 > tile 8), so
  // the single tile is broadcast to all four subgroups and their regions
  // overlap: the buffer is not private and stays in shared local memory.
  // CHECK-LABEL: gpu.func @no_privatize_broadcast
  // CHECK: %[[ALLOCA:.*]] = memref.alloca() : memref<32xi8, 3>
  // CHECK: xegpu.create_mem_desc %[[ALLOCA]] : memref<32xi8, 3> -> !xegpu.mem_desc<8xf32>
  gpu.func @no_privatize_broadcast() {
    %cst = arith.constant dense<1.0> : vector<8xf32>
    %a = memref.alloca() : memref<32xi8, 3>
    %md = xegpu.create_mem_desc %a : memref<32xi8, 3> -> !xegpu.mem_desc<8xf32>
    xegpu.store_matrix %cst, %md[0] <{layout = #xegpu.layout<sg_layout = [4], sg_data = [8]>}>
      : vector<8xf32>, !xegpu.mem_desc<8xf32>
    %l = xegpu.load_matrix %md[0] <{layout = #xegpu.layout<sg_layout = [4], sg_data = [8]>}>
      : !xegpu.mem_desc<8xf32> -> vector<8xf32>
    gpu.return
  }
}

// -----

gpu.module @test {
  // The buffer escapes the function (passed to a call), so it may be observed
  // by other subgroups and must not be privatized: it stays in space 3.
  func.func private @use(memref<32768xi8, 3>)
  // CHECK-LABEL: gpu.func @no_privatize_escaping_buffer
  // CHECK: %[[ALLOCA:.*]] = memref.alloca() : memref<32768xi8, 3>
  // CHECK: xegpu.create_mem_desc %[[ALLOCA]] : memref<32768xi8, 3> -> !xegpu.mem_desc<64x128xf32>
  gpu.func @no_privatize_escaping_buffer() {
    %cst = arith.constant dense<1.0> : vector<64x128xf32>
    %a = memref.alloca() : memref<32768xi8, 3>
    %md = xegpu.create_mem_desc %a : memref<32768xi8, 3> -> !xegpu.mem_desc<64x128xf32>
    xegpu.store_matrix %cst, %md[0, 0] <{layout = #xegpu.layout<sg_layout = [2, 4], sg_data = [32, 32]>}>
      : vector<64x128xf32>, !xegpu.mem_desc<64x128xf32>
    func.call @use(%a) : (memref<32768xi8, 3>) -> ()
    gpu.return
  }
}

// -----

gpu.module @test {
  // End-to-end softmax-like kernel (after vector-to-xegpu, rewritten to use
  // load_matrix/store_matrix). Both SLM scratch buffers (running max and running
  // sum) are read and written by every subgroup with the same sg_layout, sg_data
  // and offset, so both are demoted to subgroup-private memory and shrunk from
  // 64 elements to the per-subgroup 8 elements.
  // CHECK-LABEL: gpu.func @payload_kernel
  // CHECK-DAG: %[[MAX:.*]] = memref.alloca() : memref<32xi8, 4>
  // CHECK-DAG: %[[MD_MAX:.*]] = xegpu.create_mem_desc %[[MAX]] : memref<32xi8, 4> -> !xegpu.mem_desc<8xf32>
  // CHECK-DAG: %[[SUM:.*]] = memref.alloca() : memref<32xi8, 4>
  // CHECK-DAG: %[[MD_SUM:.*]] = xegpu.create_mem_desc %[[SUM]] : memref<32xi8, 4> -> !xegpu.mem_desc<8xf32>
  // CHECK: xegpu.store_matrix %{{.*}}, %[[MD_MAX]][0] : vector<8xf32>, !xegpu.mem_desc<8xf32>
  // CHECK: xegpu.store_matrix %{{.*}}, %[[MD_SUM]][0] : vector<8xf32>, !xegpu.mem_desc<8xf32>
  // CHECK: scf.for
  // CHECK: xegpu.load_matrix %[[MD_MAX]][0] : !xegpu.mem_desc<8xf32> -> vector<8xf32>
  // CHECK: xegpu.store_matrix %{{.*}}, %[[MD_MAX]][0] : vector<8xf32>, !xegpu.mem_desc<8xf32>
  // CHECK: xegpu.load_matrix %[[MD_SUM]][0] : !xegpu.mem_desc<8xf32> -> vector<8xf32>
  // CHECK: xegpu.store_matrix %{{.*}}, %[[MD_SUM]][0] : vector<8xf32>, !xegpu.mem_desc<8xf32>
  gpu.func @payload_kernel() kernel {
    %c16 = arith.constant 16 : index
    %c512 = arith.constant 512 : index
    %cst = arith.constant dense<0.000000e+00> : vector<64xf32>
    %c0 = arith.constant 0 : index
    %cst_0 = arith.constant dense<0xFFC00000> : vector<64xf32>
    %alloca = memref.alloca() : memref<256xi8, 3>
    %alloca_1 = memref.alloca() : memref<256xi8, 3>
    %md_max = xegpu.create_mem_desc %alloca_1 : memref<256xi8, 3> -> !xegpu.mem_desc<64xf32>
    %md_sum = xegpu.create_mem_desc %alloca : memref<256xi8, 3> -> !xegpu.mem_desc<64xf32>
    xegpu.store_matrix %cst_0, %md_max[0] <{layout = #xegpu.layout<sg_layout = [8], sg_data = [8]>}>
      : vector<64xf32>, !xegpu.mem_desc<64xf32>
    xegpu.store_matrix %cst, %md_sum[0] <{layout = #xegpu.layout<sg_layout = [8], sg_data = [8]>}>
      : vector<64xf32>, !xegpu.mem_desc<64xf32>
    scf.for %arg2 = %c0 to %c512 step %c16 {
      %4 = xegpu.load_matrix %md_max[0] <{layout = #xegpu.layout<sg_layout = [8], sg_data = [8]>}>
        : !xegpu.mem_desc<64xf32> -> vector<64xf32>
      %5 = arith.subf %cst, %4 : vector<64xf32>
      xegpu.store_matrix %5, %md_max[0] <{layout = #xegpu.layout<sg_layout = [8], sg_data = [8]>}>
        : vector<64xf32>, !xegpu.mem_desc<64xf32>
      %11 = xegpu.load_matrix %md_sum[0] <{layout = #xegpu.layout<sg_layout = [8], sg_data = [8]>}>
        : !xegpu.mem_desc<64xf32> -> vector<64xf32>
      %12 = arith.addf %11, %5 : vector<64xf32>
      xegpu.store_matrix %12, %md_sum[0] <{layout = #xegpu.layout<sg_layout = [8], sg_data = [8]>}>
        : vector<64xf32>, !xegpu.mem_desc<64xf32>
    }
    gpu.return
  }
}
