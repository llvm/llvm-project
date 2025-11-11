// RUN: mlir-opt %s -split-input-file -verify-diagnostics | FileCheck %s

// CHECK-LABEL: @nvvm_special_regs
func.func @nvvm_special_regs() -> i32 {
  // CHECK: nvvm.read.ptx.sreg.tid.x : i32
  %0 = nvvm.read.ptx.sreg.tid.x : i32
  // CHECK: nvvm.read.ptx.sreg.tid.y : i32
  %1 = nvvm.read.ptx.sreg.tid.y : i32
  // CHECK: nvvm.read.ptx.sreg.tid.z : i32
  %2 = nvvm.read.ptx.sreg.tid.z : i32
  // CHECK: nvvm.read.ptx.sreg.ntid.x : i32
  %3 = nvvm.read.ptx.sreg.ntid.x : i32
  // CHECK: nvvm.read.ptx.sreg.ntid.y : i32
  %4 = nvvm.read.ptx.sreg.ntid.y : i32
  // CHECK: nvvm.read.ptx.sreg.ntid.z : i32
  %5 = nvvm.read.ptx.sreg.ntid.z : i32
  // CHECK: nvvm.read.ptx.sreg.ctaid.x : i32
  %6 = nvvm.read.ptx.sreg.ctaid.x : i32
  // CHECK: nvvm.read.ptx.sreg.ctaid.y : i32
  %7 = nvvm.read.ptx.sreg.ctaid.y : i32
  // CHECK: nvvm.read.ptx.sreg.ctaid.z : i32
  %8 = nvvm.read.ptx.sreg.ctaid.z : i32
  // CHECK: nvvm.read.ptx.sreg.nctaid.x : i32
  %9 = nvvm.read.ptx.sreg.nctaid.x : i32
  // CHECK: nvvm.read.ptx.sreg.nctaid.y : i32
  %10 = nvvm.read.ptx.sreg.nctaid.y : i32
  // CHECK: nvvm.read.ptx.sreg.nctaid.z : i32
  %11 = nvvm.read.ptx.sreg.nctaid.z : i32
  llvm.return %0 : i32
}

// CHECK-LABEL: @nvvm_rcp
func.func @nvvm_rcp(%arg0: f32) -> f32 {
  // CHECK: nvvm.rcp.approx.ftz.f %arg0 : f32
  %0 = nvvm.rcp.approx.ftz.f %arg0 : f32
  llvm.return %0 : f32
}

// CHECK-LABEL: @llvm_nvvm_barrier0
func.func @llvm_nvvm_barrier0() {
  // CHECK: nvvm.barrier0
  nvvm.barrier0
  llvm.return
}

// CHECK-LABEL: @llvm_nvvm_barrier
// CHECK-SAME: (%[[barId:.*]]: i32, %[[numberOfThreads:.*]]: i32)
llvm.func @llvm_nvvm_barrier(%barId : i32, %numberOfThreads : i32) {
  // CHECK: nvvm.barrier 
  nvvm.barrier 
  // CHECK: nvvm.barrier id = %[[barId]]
  nvvm.barrier id = %barId
  // CHECK: nvvm.barrier id = %[[barId]] number_of_threads = %[[numberOfThreads]]
  nvvm.barrier id = %barId number_of_threads = %numberOfThreads
  llvm.return
}

// CHECK-LABEL: @llvm_nvvm_barrier_arrive
// CHECK-SAME: (%[[barId:.*]]: i32, %[[numberOfThreads:.*]]: i32)
llvm.func @llvm_nvvm_barrier_arrive(%barId : i32, %numberOfThreads : i32) {
  // CHECK: nvvm.barrier.arrive number_of_threads = %[[numberOfThreads]]
  nvvm.barrier.arrive number_of_threads = %numberOfThreads
  // CHECK: nvvm.barrier.arrive id = %[[barId]] number_of_threads = %[[numberOfThreads]]
  nvvm.barrier.arrive id = %barId number_of_threads = %numberOfThreads
  llvm.return
}

// CHECK-LABEL: @llvm_nvvm_cluster_arrive
func.func @llvm_nvvm_cluster_arrive() {
  // CHECK: nvvm.cluster.arrive
  nvvm.cluster.arrive
  // CHECK: nvvm.cluster.arrive {aligned}
  nvvm.cluster.arrive {aligned}
  llvm.return
}

// CHECK-LABEL: @llvm_nvvm_cluster_arrive_relaxed
func.func @llvm_nvvm_cluster_arrive_relaxed() {
  // CHECK: nvvm.cluster.arrive.relaxed
  nvvm.cluster.arrive.relaxed
  // CHECK: nvvm.cluster.arrive.relaxed {aligned}
  nvvm.cluster.arrive.relaxed {aligned}
  llvm.return
}

// CHECK-LABEL: @llvm_nvvm_cluster_wait
func.func @llvm_nvvm_cluster_wait() {
  // CHECK: nvvm.cluster.wait
  nvvm.cluster.wait
  // CHECK: nvvm.cluster.wait {aligned}
  nvvm.cluster.wait {aligned}
  llvm.return
}

// CHECK-LABEL: @llvm_nvvm_fence_sc_cluster
func.func @llvm_nvvm_fence_sc_cluster() {
  // CHECK: nvvm.fence.sc.cluster
  nvvm.fence.sc.cluster
  llvm.return
}

// CHECK-LABEL: @nvvm_shfl
func.func @nvvm_shfl(
    %arg0 : i32, %arg1 : i32, %arg2 : i32,
    %arg3 : i32, %arg4 : f32) -> i32 {
  // CHECK: nvvm.shfl.sync bfly %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : i32 -> i32
  %0 = nvvm.shfl.sync bfly %arg0, %arg3, %arg1, %arg2 : i32 -> i32
  // CHECK: nvvm.shfl.sync bfly %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : f32 -> f32
  %1 = nvvm.shfl.sync bfly %arg0, %arg4, %arg1, %arg2 : f32 -> f32
  // CHECK: nvvm.shfl.sync up %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : f32 -> f32
  %2 = nvvm.shfl.sync up %arg0, %arg4, %arg1, %arg2 : f32 -> f32
  // CHECK: nvvm.shfl.sync down %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : f32 -> f32
  %3 = nvvm.shfl.sync down %arg0, %arg4, %arg1, %arg2 : f32 -> f32
  // CHECK: nvvm.shfl.sync idx %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : f32 -> f32
  %4 = nvvm.shfl.sync idx %arg0, %arg4, %arg1, %arg2 : f32 -> f32
  llvm.return %0 : i32
}

// CHECK-LABEL: @nvvm_shfl_pred
func.func @nvvm_shfl_pred(
    %arg0 : i32, %arg1 : i32, %arg2 : i32,
    %arg3 : i32, %arg4 : f32) -> !llvm.struct<(i32, i1)> {
  // CHECK: nvvm.shfl.sync bfly %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} {return_value_and_is_valid} : i32 -> !llvm.struct<(i32, i1)>
  %0 = nvvm.shfl.sync bfly %arg0, %arg3, %arg1, %arg2 {return_value_and_is_valid} : i32 -> !llvm.struct<(i32, i1)>
  // CHECK: nvvm.shfl.sync bfly %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} {return_value_and_is_valid} : f32 -> !llvm.struct<(f32, i1)>
  %1 = nvvm.shfl.sync bfly %arg0, %arg4, %arg1, %arg2 {return_value_and_is_valid} : f32 -> !llvm.struct<(f32, i1)>
  llvm.return %0 : !llvm.struct<(i32, i1)>
}

// CHECK-LABEL: @nvvm_vote(
func.func @nvvm_vote(%arg0 : i32, %arg1 : i1) -> i32 {
  // CHECK: nvvm.vote.sync ballot %{{.*}}, %{{.*}} -> i32
  %0 = nvvm.vote.sync ballot %arg0, %arg1 -> i32
  // CHECK: nvvm.vote.sync all %{{.*}}, %{{.*}} -> i1
  %1 = nvvm.vote.sync all %arg0, %arg1 -> i1
  // CHECK: nvvm.vote.sync any %{{.*}}, %{{.*}} -> i1
  %2 = nvvm.vote.sync any %arg0, %arg1 -> i1
  // CHECK: nvvm.vote.sync uni %{{.*}}, %{{.*}} -> i1
  %3 = nvvm.vote.sync uni %arg0, %arg1 -> i1
  llvm.return %0 : i32
}

// CHECK-LABEL: @llvm_nvvm_bar_warp_sync
func.func @llvm_nvvm_bar_warp_sync(%mask : i32) {
  // CHECK: nvvm.bar.warp.sync %{{.*}}
  nvvm.bar.warp.sync %mask : i32
  llvm.return
}

// CHECK-LABEL: @nvvm_mma_m8n8k4_row_col_f32_f32
func.func @nvvm_mma_m8n8k4_row_col_f32_f32(%a0 : vector<2xf16>, %a1 : vector<2xf16>,
               %b0 : vector<2xf16>, %b1 : vector<2xf16>,
               %c0 : f32, %c1 : f32, %c2 : f32, %c3 : f32, %c4 : f32, %c5 : f32, %c6 : f32, %c7 : f32) {
  // CHECK: nvvm.mma.sync
  %0 = nvvm.mma.sync A[%a0, %a1] B[%b0, %b1] C[%c0, %c1, %c2, %c3, %c4, %c5, %c6, %c7]
    {layoutA = #nvvm.mma_layout<row>, layoutB = #nvvm.mma_layout<col>,
     shape = #nvvm.shape<m = 8, n = 8, k = 4>} : (vector<2xf16>, vector<2xf16>, f32) -> !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f32)>
  llvm.return %0 : !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f32)>
}

// CHECK-LABEL: @nvvm_mma_m8n8k4_f16_f16
func.func @nvvm_mma_m8n8k4_f16_f16(%a0 : vector<2xf16>, %a1 : vector<2xf16>,
                              %b0 : vector<2xf16>, %b1 : vector<2xf16>,
                              %c0 : vector<2xf16>, %c1 : vector<2xf16>, %c2 : vector<2xf16>, %c3 : vector<2xf16>) {
  // CHECK: nvvm.mma.sync A[{{.*}}] B[{{.*}}] C[{{.*}}]
  %0 = nvvm.mma.sync A[%a0, %a1] B[%b0, %b1] C[%c0, %c1, %c2, %c3]
    {layoutA = #nvvm.mma_layout<row>, layoutB = #nvvm.mma_layout<col>,
     shape = #nvvm.shape<m = 8, n = 8, k = 4>} : (vector<2xf16>,vector<2xf16>,vector<2xf16>) -> !llvm.struct<(vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>)>
  llvm.return %0 : !llvm.struct<(vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>)>
}

// CHECK-LABEL: @nvvm_mma_m16n8k8_bf16_bf16
func.func @nvvm_mma_m16n8k8_bf16_bf16(%a0 : i32, %a1 : i32, %b0 : i32,
                              %c0 : f32, %c1 : f32, %c2 : f32, %c3 : f32) {
  // CHECK: nvvm.mma.sync A[{{.*}}] B[{{.*}}] C[{{.*}}] {layoutA = #nvvm.mma_layout<row>, layoutB = #nvvm.mma_layout<col>, multiplicandAPtxType = #nvvm.mma_type<bf16>, multiplicandBPtxType = #nvvm.mma_type<bf16>, shape = #nvvm.shape<m = 16, n = 8, k = 8>} : (i32, i32, f32) -> !llvm.struct<(f32, f32, f32, f32)>
  %0 = nvvm.mma.sync A[%a0, %a1] B[%b0] C[%c0, %c1, %c2, %c3]
    {layoutA = #nvvm.mma_layout<row>, layoutB = #nvvm.mma_layout<col>,
     multiplicandAPtxType = #nvvm.mma_type<bf16>, multiplicandBPtxType = #nvvm.mma_type<bf16>,
     shape = #nvvm.shape<m = 16, n = 8, k = 8>} : (i32, i32, f32) -> !llvm.struct<(f32, f32, f32, f32)>
  llvm.return %0 : !llvm.struct<(f32, f32, f32, f32)>
}

// CHECK-LABEL: @nvvm_mma_m16n8k16_bf16_bf16
func.func @nvvm_mma_m16n8k16_bf16_bf16(%a0 : i32, %a1 : i32, %a2 : i32, %a3 : i32,
                              %b0 : i32, %b1 : i32,
                              %c0 : f32, %c1 : f32, %c2 : f32, %c3 : f32) {
  // CHECK: nvvm.mma.sync A[{{.*}}] B[{{.*}}] C[{{.*}}] {layoutA = #nvvm.mma_layout<row>, layoutB = #nvvm.mma_layout<col>, multiplicandAPtxType = #nvvm.mma_type<bf16>, multiplicandBPtxType = #nvvm.mma_type<bf16>, shape = #nvvm.shape<m = 16, n = 8, k = 16>} : (i32, i32, f32) -> !llvm.struct<(f32, f32, f32, f32)>
  %0 = nvvm.mma.sync A[%a0, %a1, %a2, %a3] B[%b0, %b1] C[%c0, %c1, %c2, %c3]
    {layoutA = #nvvm.mma_layout<row>, layoutB = #nvvm.mma_layout<col>,
     multiplicandAPtxType = #nvvm.mma_type<bf16>, multiplicandBPtxType = #nvvm.mma_type<bf16>,
     shape = #nvvm.shape<m = 16, n = 8, k = 16>} : (i32, i32, f32) -> !llvm.struct<(f32, f32, f32, f32)>
  llvm.return %0 : !llvm.struct<(f32, f32, f32, f32)>
}

// CHECK-LABEL: @nvvm_mma_m8n8k16_s8_s8
func.func @nvvm_mma_m8n8k16_s8_s8(%a0 : i32, %b0 : i32,
                             %c0 : i32, %c1 : i32) {
  // CHECK: nvvm.mma.sync A[{{.*}}] B[{{.*}}] C[{{.*}}, {{.*}}] {intOverflowBehavior = #nvvm.mma_int_overflow<wrapped>, layoutA = #nvvm.mma_layout<row>, layoutB = #nvvm.mma_layout<col>, multiplicandAPtxType = #nvvm.mma_type<s8>, multiplicandBPtxType = #nvvm.mma_type<s8>, shape = #nvvm.shape<m = 8, n = 8, k = 16>} : (i32, i32, i32) -> !llvm.struct<(i32, i32)>
  %0 = nvvm.mma.sync A[%a0] B[%b0] C[%c0, %c1]
    {layoutA = #nvvm.mma_layout<row>, layoutB = #nvvm.mma_layout<col>,
     multiplicandAPtxType = #nvvm.mma_type<s8>, multiplicandBPtxType = #nvvm.mma_type<s8>,
     intOverflowBehavior=#nvvm.mma_int_overflow<wrapped>,
     shape = #nvvm.shape<m = 8, n = 8, k = 16>} : (i32, i32, i32) -> !llvm.struct<(i32, i32)>
  llvm.return %0 : !llvm.struct<(i32, i32)>
}

// CHECK-LABEL: @nvvm_mma_m16n8k8_f16_f16
func.func @nvvm_mma_m16n8k8_f16_f16(%a0 : vector<2xf16>, %a1 : vector<2xf16>,
                               %b0 : vector<2xf16>,
                               %c0 : vector<2xf16>, %c1 : vector<2xf16>) {
  // CHECK: nvvm.mma.sync A[%{{.*}}, %{{.*}}] B[%{{.*}}] C[%{{.*}}, %{{.*}}] {{{.*}}} : (vector<2xf16>, vector<2xf16>, vector<2xf16>) -> !llvm.struct<(vector<2xf16>, vector<2xf16>)>
  %0 = nvvm.mma.sync A[%a0, %a1] B[%b0] C[%c0, %c1]
    {layoutA = #nvvm.mma_layout<row>, layoutB = #nvvm.mma_layout<col>,
     shape = #nvvm.shape<m = 16, n = 8, k = 8>} : (vector<2xf16>, vector<2xf16>, vector<2xf16>) -> !llvm.struct<(vector<2xf16>, vector<2xf16>)>
  llvm.return %0 : !llvm.struct<(vector<2xf16>, vector<2xf16>)>
}

// CHECK-LABEL: @nvvm_mma_m16n8k16_f16_f16
func.func @nvvm_mma_m16n8k16_f16_f16(%a0 : vector<2xf16>, %a1 : vector<2xf16>,
                                %a2 : vector<2xf16>, %a3 : vector<2xf16>,
                                %b0 : vector<2xf16>, %b1 : vector<2xf16>,
                                %c0 : vector<2xf16>, %c1 : vector<2xf16>) {
  // CHECK: nvvm.mma.sync A[{{.*}}, {{.*}}, {{.*}}, {{.*}}] B[{{.*}}, {{.*}}] C[{{.*}}, {{.*}}] {layoutA = #nvvm.mma_layout<row>, layoutB = #nvvm.mma_layout<col>, shape = #nvvm.shape<m = 16, n = 8, k = 16>} : (vector<2xf16>, vector<2xf16>, vector<2xf16>) -> !llvm.struct<(vector<2xf16>, vector<2xf16>)>
  %0 = nvvm.mma.sync A[%a0, %a1, %a2, %a3] B[%b0, %b1] C[%c0, %c1]
    {layoutA = #nvvm.mma_layout<row>, layoutB = #nvvm.mma_layout<col>,
     shape = #nvvm.shape<m = 16, n = 8, k = 16>} : (vector<2xf16>, vector<2xf16>, vector<2xf16>) -> !llvm.struct<(vector<2xf16>, vector<2xf16>)>
  llvm.return %0 : !llvm.struct<(vector<2xf16>, vector<2xf16>)>
}

// CHECK-LABEL: @nvvm_mma_m16n8k16_f32_f32
func.func @nvvm_mma_m16n8k16_f32_f32(%a0 : vector<2xf16>, %a1 : vector<2xf16>,
                                %a2 : vector<2xf16>, %a3 : vector<2xf16>,
                                %b0 : vector<2xf16>, %b1 : vector<2xf16>,
                                %c0 : f32, %c1 : f32, %c2 : f32, %c3 : f32) {
  // CHECK: nvvm.mma.sync A[{{.*}}, {{.*}}, {{.*}}, {{.*}}] B[{{.*}}, {{.*}}] C[{{.*}}, {{.*}}, {{.*}}, {{.*}}] {layoutA = #nvvm.mma_layout<row>, layoutB = #nvvm.mma_layout<col>, shape = #nvvm.shape<m = 16, n = 8, k = 16>} : (vector<2xf16>, vector<2xf16>, f32) -> !llvm.struct<(f32, f32, f32, f32)>
  %0 = nvvm.mma.sync A[%a0, %a1, %a2, %a3] B[%b0, %b1] C[%c0, %c1, %c2, %c3]
    {layoutA = #nvvm.mma_layout<row>, layoutB = #nvvm.mma_layout<col>,
     shape = #nvvm.shape<m = 16, n = 8, k = 16>} : (vector<2xf16>, vector<2xf16>, f32) -> !llvm.struct<(f32, f32, f32, f32)>
  llvm.return %0 : !llvm.struct<(f32, f32, f32, f32)>
}

// CHECK-LABEL: @nvvm_mma_m16n8k4_tf32_f32
func.func @nvvm_mma_m16n8k4_tf32_f32(%a0 : i32, %a1 : i32,
                                     %b0 : i32,
                                     %c0 : f32, %c1 : f32, %c2 : f32, %c3 : f32) {
  // CHECK: nvvm.mma.sync A[{{.*}}, {{.*}}] B[{{.*}}] C[{{.*}}, {{.*}}, {{.*}}, {{.*}}] {layoutA = #nvvm.mma_layout<row>, layoutB = #nvvm.mma_layout<col>, multiplicandAPtxType = #nvvm.mma_type<tf32>, multiplicandBPtxType = #nvvm.mma_type<tf32>, shape = #nvvm.shape<m = 16, n = 8, k = 4>} : (i32, i32, f32) -> !llvm.struct<(f32, f32, f32, f32)>
  %0 = nvvm.mma.sync A[%a0, %a1] B[%b0] C[%c0, %c1, %c2, %c3]
    {layoutA = #nvvm.mma_layout<row>, layoutB = #nvvm.mma_layout<col>,
     multiplicandAPtxType = #nvvm.mma_type<tf32>, multiplicandBPtxType = #nvvm.mma_type<tf32>,
     shape = #nvvm.shape<m = 16, n = 8, k = 4>} : (i32, i32, f32) -> !llvm.struct<(f32, f32, f32, f32)>
  llvm.return %0 : !llvm.struct<(f32, f32, f32, f32)>
}

// CHECK-LABEL: @nvvm_mma_m16n8k16_s8_s8
func.func @nvvm_mma_m16n8k16_s8_s8(%a0 : i32, %a1 : i32, %b0 : i32,
                              %c0 : i32, %c1 : i32, %c2 : i32, %c3 : i32) {
  // CHECK: nvvm.mma.sync A[{{.*}}, {{.*}}] B[{{.*}}] C[{{.*}}, {{.*}}, {{.*}}, {{.*}}] {intOverflowBehavior = #nvvm.mma_int_overflow<wrapped>, layoutA = #nvvm.mma_layout<row>, layoutB = #nvvm.mma_layout<col>, multiplicandAPtxType = #nvvm.mma_type<s8>, multiplicandBPtxType = #nvvm.mma_type<s8>, shape = #nvvm.shape<m = 16, n = 8, k = 16>} : (i32, i32, i32) -> !llvm.struct<(i32, i32, i32, i32)>
  %0 = nvvm.mma.sync A[%a0, %a1] B[%b0] C[%c0, %c1, %c2, %c3]
    {layoutA = #nvvm.mma_layout<row>, layoutB = #nvvm.mma_layout<col>,
     multiplicandAPtxType = #nvvm.mma_type<s8>, multiplicandBPtxType = #nvvm.mma_type<s8>,
     intOverflowBehavior=#nvvm.mma_int_overflow<wrapped>,
     shape = #nvvm.shape<m = 16, n = 8, k = 16>} : (i32, i32, i32) -> !llvm.struct<(i32,i32,i32,i32)>
  llvm.return %0 : !llvm.struct<(i32,i32,i32,i32)>
}

// CHECK-LABEL: @nvvm_mma_m16n8k16_s8_u8
func.func @nvvm_mma_m16n8k16_s8_u8(%a0 : i32, %a1 : i32,
                                %b0 : i32,
                                %c0 : i32, %c1 : i32, %c2 : i32, %c3 : i32) {
  // CHECK: nvvm.mma.sync A[{{.*}}, {{.*}}] B[{{.*}}] C[{{.*}}, {{.*}}, {{.*}}, {{.*}}] {intOverflowBehavior = #nvvm.mma_int_overflow<satfinite>, layoutA = #nvvm.mma_layout<row>, layoutB = #nvvm.mma_layout<col>, multiplicandAPtxType = #nvvm.mma_type<s8>, multiplicandBPtxType = #nvvm.mma_type<u8>, shape = #nvvm.shape<m = 16, n = 8, k = 16>} : (i32, i32, i32) -> !llvm.struct<(i32, i32, i32, i32)>
  %0 = nvvm.mma.sync A[%a0, %a1] B[%b0] C[%c0, %c1, %c2, %c3]
    {layoutA = #nvvm.mma_layout<row>, layoutB = #nvvm.mma_layout<col>,
     multiplicandAPtxType = #nvvm.mma_type<s8>, multiplicandBPtxType = #nvvm.mma_type<u8>,
     intOverflowBehavior=#nvvm.mma_int_overflow<satfinite>,
     shape = #nvvm.shape<m = 16, n = 8, k = 16>} : (i32, i32, i32) -> !llvm.struct<(i32,i32,i32,i32)>
  llvm.return %0 : !llvm.struct<(i32,i32,i32,i32)>
}

// CHECK-LABEL: @nvvm_mma_m16n8k256_b1_b1
func.func @nvvm_mma_m16n8k256_b1_b1(%a0 : i32, %a1 : i32, %a2 : i32, %a3 : i32,
                               %b0 : i32, %b1 : i32,
                               %c0 : i32, %c1 : i32, %c2 : i32, %c3 : i32) {
  // CHECK: nvvm.mma.sync A[{{.*}}, {{.*}}, {{.*}}, {{.*}}] B[{{.*}}] C[{{.*}}, {{.*}}, {{.*}}, {{.*}}] {b1Op = #nvvm.mma_b1op<xor_popc>, layoutA = #nvvm.mma_layout<row>, layoutB = #nvvm.mma_layout<col>, multiplicandAPtxType = #nvvm.mma_type<b1>, multiplicandBPtxType = #nvvm.mma_type<b1>, shape = #nvvm.shape<m = 16, n = 8, k = 256>} : (i32, i32, i32) -> !llvm.struct<(i32, i32, i32, i32)>
  %0 = nvvm.mma.sync A[%a0, %a1, %a2, %a3] B[%b0, %b1] C[%c0, %c1, %c2, %c3]
    {layoutA = #nvvm.mma_layout<row>, layoutB = #nvvm.mma_layout<col>,
     multiplicandAPtxType = #nvvm.mma_type<b1>, multiplicandBPtxType = #nvvm.mma_type<b1>,
     b1Op = #nvvm.mma_b1op<xor_popc>, shape = #nvvm.shape<m = 16, n = 8, k = 256>} : (i32, i32, i32) -> !llvm.struct<(i32,i32,i32,i32)>
  llvm.return %0 : !llvm.struct<(i32,i32,i32,i32)>
}

// CHECK-LABEL: @nvvm_mma_m16n8k128_b1_b1
func.func @nvvm_mma_m16n8k128_b1_b1(%a0 : i32, %a1 : i32,
                               %b0 : i32,
                               %c0 : i32, %c1 : i32, %c2 : i32, %c3 : i32) {
  // CHECK: nvvm.mma.sync A[{{.*}}, {{.*}}] B[{{.*}}] C[{{.*}}, {{.*}}, {{.*}}, {{.*}}] {b1Op = #nvvm.mma_b1op<xor_popc>, layoutA = #nvvm.mma_layout<row>, layoutB = #nvvm.mma_layout<col>, multiplicandAPtxType = #nvvm.mma_type<b1>, multiplicandBPtxType = #nvvm.mma_type<b1>, shape = #nvvm.shape<m = 16, n = 8, k = 128>} : (i32, i32, i32) -> !llvm.struct<(i32, i32, i32, i32)>
  %0 = nvvm.mma.sync A[%a0, %a1] B[%b0] C[%c0, %c1, %c2, %c3]
    {layoutA = #nvvm.mma_layout<row>, layoutB = #nvvm.mma_layout<col>,
     multiplicandAPtxType = #nvvm.mma_type<b1>, multiplicandBPtxType = #nvvm.mma_type<b1>,
     b1Op = #nvvm.mma_b1op<xor_popc>,
     shape = #nvvm.shape<m = 16, n = 8, k = 128>} : (i32, i32, i32) -> !llvm.struct<(i32,i32,i32,i32)>
  llvm.return %0 : !llvm.struct<(i32,i32,i32,i32)>
}

// CHECK-LABEL: @nvvm_mma_m8n8k128_b1_b1
func.func @nvvm_mma_m8n8k128_b1_b1(%a0 : i32,
                              %b0 : i32,
                              %c0 : i32, %c1 : i32) {
  // CHECK: nvvm.mma.sync A[{{.*}}] B[{{.*}}] C[{{.*}}, {{.*}}] {b1Op = #nvvm.mma_b1op<xor_popc>, layoutA = #nvvm.mma_layout<row>, layoutB = #nvvm.mma_layout<col>, multiplicandAPtxType = #nvvm.mma_type<b1>, multiplicandBPtxType = #nvvm.mma_type<b1>, shape = #nvvm.shape<m = 8, n = 8, k = 128>} : (i32, i32, i32) -> !llvm.struct<(i32, i32)>
  %0 = nvvm.mma.sync A[%a0] B[%b0] C[%c0, %c1]
    {layoutA = #nvvm.mma_layout<row>, layoutB = #nvvm.mma_layout<col>,
     multiplicandAPtxType = #nvvm.mma_type<b1>, multiplicandBPtxType = #nvvm.mma_type<b1>,
     b1Op = #nvvm.mma_b1op<xor_popc>, shape = #nvvm.shape<m = 8, n = 8, k = 128>} : (i32, i32, i32) -> !llvm.struct<(i32,i32)>
  llvm.return %0 : !llvm.struct<(i32,i32)>
}

// CHECK-LABEL: @nvvm_mma_m16n8k32_s4_s4
func.func @nvvm_mma_m16n8k32_s4_s4(%a0 : i32, %a1 : i32,
                               %b0 : i32,
                               %c0 : i32, %c1 : i32, %c2 : i32, %c3 : i32) {
  // CHECK: nvvm.mma.sync A[{{.*}}, {{.*}}] B[{{.*}}] C[{{.*}}, {{.*}}, {{.*}}, {{.*}}] {intOverflowBehavior = #nvvm.mma_int_overflow<wrapped>, layoutA = #nvvm.mma_layout<row>, layoutB = #nvvm.mma_layout<col>, multiplicandAPtxType = #nvvm.mma_type<s4>, multiplicandBPtxType = #nvvm.mma_type<s4>, shape = #nvvm.shape<m = 16, n = 8, k = 32>} : (i32, i32, i32) -> !llvm.struct<(i32, i32, i32, i32)>
  %0 = nvvm.mma.sync A[%a0, %a1] B[%b0] C[%c0, %c1, %c2, %c3]
    {layoutA = #nvvm.mma_layout<row>, layoutB = #nvvm.mma_layout<col>,
     multiplicandAPtxType = #nvvm.mma_type<s4>, multiplicandBPtxType = #nvvm.mma_type<s4>,
     intOverflowBehavior=#nvvm.mma_int_overflow<wrapped>,
     shape = #nvvm.shape<m = 16, n = 8, k = 32>} : (i32, i32, i32) -> !llvm.struct<(i32,i32,i32,i32)>
  llvm.return %0 : !llvm.struct<(i32,i32,i32,i32)>
}

// CHECK-LABEL: @nvvm_wmma_load_tf32
func.func @nvvm_wmma_load_tf32(%arg0: !llvm.ptr, %arg1 : i32) -> !llvm.struct<(i32, i32, i32, i32)> {
  // CHECK: nvvm.wmma.load {{.*}} {eltype = #nvvm.mma_type<tf32>, frag = #nvvm.mma_frag<a>, k = 8 : i32, layout = #nvvm.mma_layout<row>, m = 16 : i32, n = 16 : i32}
  %0 = nvvm.wmma.load %arg0, %arg1
    {eltype = #nvvm.mma_type<tf32>, frag = #nvvm.mma_frag<a>, k = 8 : i32, layout = #nvvm.mma_layout<row>, m = 16 : i32, n = 16 : i32}
    : (!llvm.ptr) -> !llvm.struct<(i32, i32, i32, i32)>
  llvm.return %0 : !llvm.struct<(i32, i32, i32, i32)>
}

// CHECK-LABEL: @nvvm_wmma_mma
func.func @nvvm_wmma_mma(%0 : i32, %1 : i32, %2 : i32, %3 : i32, %4 : i32, %5 : i32,
                    %6 : i32, %7 : i32, %8 : f32, %9 : f32, %10 : f32,
                    %11 : f32, %12 : f32, %13 : f32, %14 : f32, %15 : f32)
                   -> !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f32)> {
  // CHECK: nvvm.wmma.mma {{.*}} {eltypeA = #nvvm.mma_type<tf32>, eltypeB = #nvvm.mma_type<f32>, k = 8 : i32, layoutA = #nvvm.mma_layout<row>, layoutB = #nvvm.mma_layout<row>, m = 16 : i32, n = 16 : i32}
  %r = nvvm.wmma.mma %0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15
    {eltypeA = #nvvm.mma_type<tf32>, eltypeB = #nvvm.mma_type<f32>, k = 8 : i32, layoutA = #nvvm.mma_layout<row>, layoutB = #nvvm.mma_layout<row>, m = 16 : i32, n = 16 : i32}
    : (i32, i32, i32, i32, i32, i32, i32, i32, f32, f32, f32, f32, f32, f32, f32, f32)
    -> !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f32)>
  llvm.return %r : !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f32)>
}

// CHECK-LABEL: @cp_async
llvm.func @cp_async(%arg0: !llvm.ptr<3>, %arg1: !llvm.ptr<1>) {
// CHECK:  nvvm.cp.async.shared.global %{{.*}}, %{{.*}}, 16, cache = ca
  nvvm.cp.async.shared.global %arg0, %arg1, 16, cache = ca : !llvm.ptr<3>, !llvm.ptr<1>
// CHECK:  nvvm.cp.async.shared.global %{{.*}}, %{{.*}}, 16, cache = cg
  nvvm.cp.async.shared.global %arg0, %arg1, 16, cache = cg : !llvm.ptr<3>, !llvm.ptr<1>
// CHECK: nvvm.cp.async.commit.group
  nvvm.cp.async.commit.group
// CHECK: nvvm.cp.async.wait.group 0
  nvvm.cp.async.wait.group 0
  llvm.return
}

// CHECK-LABEL: llvm.func @redux_sync
llvm.func @redux_sync(%value : i32, %offset : i32) -> i32 {
  // CHECK: nvvm.redux.sync  add %{{.*}}
  %r1 = nvvm.redux.sync add %value, %offset : i32 -> i32
  // CHECK: nvvm.redux.sync  max %{{.*}}
  %r2 = nvvm.redux.sync max %value, %offset : i32 -> i32
  // CHECK: nvvm.redux.sync  min %{{.*}}
  %r3 = nvvm.redux.sync min %value, %offset : i32 -> i32
  // CHECK: nvvm.redux.sync  umax %{{.*}}
  %r5 = nvvm.redux.sync umax %value, %offset : i32 -> i32
  // CHECK: nvvm.redux.sync  umin %{{.*}}
  %r6 = nvvm.redux.sync umin %value, %offset : i32 -> i32
  // CHECK: nvvm.redux.sync  and %{{.*}}
  %r7 = nvvm.redux.sync and %value, %offset : i32 -> i32
  // CHECK: nvvm.redux.sync  or %{{.*}}
  %r8 = nvvm.redux.sync or %value, %offset : i32 -> i32
  // CHECK: nvvm.redux.sync  xor %{{.*}}
  %r9 = nvvm.redux.sync xor %value, %offset : i32 -> i32
  llvm.return %r1 : i32
}

llvm.func @redux_sync_f32(%value: f32, %offset: i32) -> f32 {
  // CHECK: nvvm.redux.sync fmin %{{.*}}
  %r1 = nvvm.redux.sync fmin %value, %offset: f32 -> f32
  // CHECK: nvvm.redux.sync fmin %{{.*}}
  %r2 = nvvm.redux.sync fmin %value, %offset {abs = true}: f32 -> f32
  // CHECK: nvvm.redux.sync fmin %{{.*}}
  %r3 = nvvm.redux.sync fmin %value, %offset {NaN = true}: f32 -> f32
  // CHECK: nvvm.redux.sync fmin %{{.*}}
  %r4 = nvvm.redux.sync fmin %value, %offset {abs = true, NaN = true}: f32 -> f32
  // CHECK: nvvm.redux.sync fmax %{{.*}}
  %r5 = nvvm.redux.sync fmax %value, %offset: f32 -> f32
  // CHECK: nvvm.redux.sync fmax %{{.*}}
  %r6 = nvvm.redux.sync fmax %value, %offset {abs = true}: f32 -> f32
  // CHECK: nvvm.redux.sync fmax %{{.*}}
  %r7 = nvvm.redux.sync fmax %value, %offset {NaN = true}: f32 -> f32
  // CHECK: nvvm.redux.sync fmax %{{.*}}
  %r8 = nvvm.redux.sync fmax %value, %offset {abs = true, NaN = true}: f32 -> f32
  llvm.return %r1 : f32
}

// -----

// expected-error@below {{attribute attached to unexpected op}}
func.func private @expected_llvm_func() attributes { nvvm.kernel }

// -----

llvm.func private @mbarrier_init_generic(%barrier: !llvm.ptr) {
  %count = nvvm.read.ptx.sreg.ntid.x : i32
  // CHECK:   nvvm.mbarrier.init %{{.*}}, %{{.*}} : !llvm.ptr, i32
  nvvm.mbarrier.init %barrier, %count : !llvm.ptr, i32
  llvm.return
}


llvm.func private @mbarrier_init_shared(%barrier: !llvm.ptr<3>) {
  %count = nvvm.read.ptx.sreg.ntid.x : i32
  // CHECK:   nvvm.mbarrier.init.shared %{{.*}}, %{{.*}} : !llvm.ptr<3>, i32
  nvvm.mbarrier.init.shared %barrier, %count : !llvm.ptr<3>, i32
  llvm.return
}


llvm.func private @mbarrier_inval_generic(%barrier: !llvm.ptr) {
  // CHECK:   nvvm.mbarrier.inval %{{.*}} : !llvm.ptr
  nvvm.mbarrier.inval %barrier : !llvm.ptr
  llvm.return
}


llvm.func private @mbarrier_inval_shared(%barrier: !llvm.ptr<3>) {
  // CHECK:   nvvm.mbarrier.inval.shared %{{.*}} : !llvm.ptr<3>
  nvvm.mbarrier.inval.shared %barrier : !llvm.ptr<3>
  llvm.return
}

llvm.func private @mbarrier_arrive(%barrier: !llvm.ptr) {
  // CHECK:   nvvm.mbarrier.arrive %{{.*}} : !llvm.ptr
  %0 = nvvm.mbarrier.arrive %barrier : !llvm.ptr  -> i64
  llvm.return
}

llvm.func private @mbarrier_arrive_shared(%barrier: !llvm.ptr<3>) {
  // CHECK:   nvvm.mbarrier.arrive.shared %{{.*}} : !llvm.ptr<3>
  %0 = nvvm.mbarrier.arrive.shared %barrier : !llvm.ptr<3> -> i64
  llvm.return
}

llvm.func private @mbarrier_arrive_nocomplete(%barrier: !llvm.ptr) {
  %count = nvvm.read.ptx.sreg.ntid.x : i32
  // CHECK:   nvvm.mbarrier.arrive.nocomplete %{{.*}} : !llvm.ptr
  %0 = nvvm.mbarrier.arrive.nocomplete %barrier, %count : !llvm.ptr, i32 -> i64
  llvm.return
}

llvm.func private @mbarrier_arrive_nocomplete_shared(%barrier: !llvm.ptr<3>) {
  %count = nvvm.read.ptx.sreg.ntid.x : i32
  // CHECK:   nvvm.mbarrier.arrive.nocomplete.shared %{{.*}} : !llvm.ptr<3>
  %0 = nvvm.mbarrier.arrive.nocomplete.shared %barrier, %count : !llvm.ptr<3>, i32  -> i64
  llvm.return
}

llvm.func private @mbarrier_test_wait(%barrier: !llvm.ptr, %token : i64) -> i1 {  
  // CHECK:   nvvm.mbarrier.test.wait %{{.*}}
  %isComplete = nvvm.mbarrier.test.wait %barrier, %token : !llvm.ptr, i64 -> i1
  llvm.return %isComplete : i1
}

llvm.func private @mbarrier_test_wait_shared(%barrier: !llvm.ptr<3>, %token : i64) {
  %count = nvvm.read.ptx.sreg.ntid.x : i32
  // CHECK:   nvvm.mbarrier.test.wait.shared %{{.*}}
  %isComplete = nvvm.mbarrier.test.wait.shared %barrier, %token : !llvm.ptr<3>, i64 -> i1
  llvm.return
}

// CHECK-LABEL: @wgmma_fence_aligned
func.func @wgmma_fence_aligned() {
  // CHECK: nvvm.wgmma.fence.aligned
  nvvm.wgmma.fence.aligned
  return
}

// CHECK-LABEL: @wgmma_commit_group_sync_aligned
func.func @wgmma_commit_group_sync_aligned() {
  // CHECK: nvvm.wgmma.commit.group.sync.aligned
  nvvm.wgmma.commit.group.sync.aligned
  return
}


// CHECK-LABEL: @wgmma_wait_group_sync_aligned
func.func @wgmma_wait_group_sync_aligned() {
  // CHECK: nvvm.wgmma.wait.group.sync.aligned
  nvvm.wgmma.wait.group.sync.aligned 0
  return
}

func.func @griddepcontrol_wait() {
  // CHECK: nvvm.griddepcontrol wait
  nvvm.griddepcontrol wait
  return
}

func.func @griddepcontrol_launch_dependents()
{
  // CHECK: nvvm.griddepcontrol launch_dependents
  nvvm.griddepcontrol launch_dependents
  return
}

// CHECK-LABEL: @mapa
func.func @mapa(%a: !llvm.ptr, %a_shared: !llvm.ptr<3>, %b : i32) {
  // CHECK:   nvvm.mapa %{{.*}}
  %0 = nvvm.mapa %a, %b: !llvm.ptr -> !llvm.ptr
  // CHECK:   nvvm.mapa %{{.*}}
  %1 = nvvm.mapa %a_shared, %b: !llvm.ptr<3> -> !llvm.ptr<7>
  return
}

// CHECK-LABEL: @match_sync
func.func @match_sync(%val32: i32, %val64: i64, %thread_mask: i32) {
  // CHECK: nvvm.match.sync any %{{.*}}, %{{.*}} : i32 -> i32
  %0 = nvvm.match.sync any %thread_mask, %val32 : i32 -> i32
  // CHECK: nvvm.match.sync all %{{.*}}, %{{.*}} : i32 -> !llvm.struct<(i32, i1)>
  %1 = nvvm.match.sync all %thread_mask, %val32 : i32 -> !llvm.struct<(i32, i1)>
  // CHECK: nvvm.match.sync any %{{.*}}, %{{.*}} : i64 -> i32
  %2 = nvvm.match.sync any %thread_mask, %val64 : i64 -> i32
  // CHECK: nvvm.match.sync all %{{.*}}, %{{.*}} : i64 -> !llvm.struct<(i32, i1)>
  %3 = nvvm.match.sync all %thread_mask, %val64 : i64 -> !llvm.struct<(i32, i1)>
  return 
}

// CHECK-LABEL: @st_bulk
func.func @st_bulk(%addr_gen: !llvm.ptr, %addr_shared: !llvm.ptr<3>, %size: i64) {
  // CHECK:   nvvm.st.bulk %{{.*}}, size = %{{.*}} : !llvm.ptr
  nvvm.st.bulk %addr_gen, size = %size, init = 0 : !llvm.ptr
  // CHECK:   nvvm.st.bulk %{{.*}}, size = %{{.*}} : !llvm.ptr<3>
  nvvm.st.bulk %addr_shared, size = %size, init = 0 : !llvm.ptr<3>
  return
}

// CHECK-LABEL: @dot_accumulate_4way
func.func @dot_accumulate_4way(%a_vec: vector<4xi8>, %b_vec: vector<4xi8>, %c: i32) {
  // CHECK:   nvvm.dot.accumulate.4way %{{.*}}, %{{.*}}, %{{.*}} : vector<4xi8>, vector<4xi8>
  %1 = nvvm.dot.accumulate.4way %a_vec <unsigned>, %b_vec <unsigned>, %c: vector<4xi8>, vector<4xi8>
  // CHECK:   nvvm.dot.accumulate.4way %{{.*}}, %{{.*}}, %{{.*}} : vector<4xi8>, vector<4xi8>
  %3 = nvvm.dot.accumulate.4way %a_vec <signed>, %b_vec <signed>, %c: vector<4xi8>, vector<4xi8>
  return
}

// CHECK-LABEL: @dot_accumulate_2way
func.func @dot_accumulate_2way(%a_vec: vector<2xi16>, %b_vec: vector<4xi8>, %c: i32) {
  // CHECK:   nvvm.dot.accumulate.2way %{{.*}}, %{{.*}}, %{{.*}} {b_hi = false} : vector<2xi16>, vector<4xi8>
  %1 = nvvm.dot.accumulate.2way %a_vec <unsigned>, %b_vec <unsigned>, %c {b_hi = false}: vector<2xi16>, vector<4xi8>
  // CHECK:   nvvm.dot.accumulate.2way %{{.*}}, %{{.*}}, %{{.*}} {b_hi = true} : vector<2xi16>, vector<4xi8>
  %3 = nvvm.dot.accumulate.2way %a_vec <signed>, %b_vec <signed>, %c {b_hi = true}: vector<2xi16>, vector<4xi8>
  return
}

// CHECK-LABEL: @prefetch
func.func @prefetch(%gen_ptr: !llvm.ptr, %local_ptr: !llvm.ptr<5>, %global_ptr: !llvm.ptr<1>, %const_ptr: !llvm.ptr<4>) {
  // CHECK:   nvvm.prefetch level = L1, %{{.*}}
  nvvm.prefetch level = L1, %gen_ptr : !llvm.ptr<0>
  // CHECK:   nvvm.prefetch level = L1, %{{.*}}
  nvvm.prefetch level = L1, %local_ptr : !llvm.ptr<5>
  // CHECK:   nvvm.prefetch level = L1, %{{.*}}
  nvvm.prefetch level = L1, %global_ptr : !llvm.ptr<1>
  // CHECK:   nvvm.prefetch level = L2, %{{.*}}
  nvvm.prefetch level = L2, %gen_ptr : !llvm.ptr<0>
  // CHECK:   nvvm.prefetch level = L2, %{{.*}}
  nvvm.prefetch level = L2, %local_ptr : !llvm.ptr<5>
  // CHECK:   nvvm.prefetch level = L2, %{{.*}}
  nvvm.prefetch level = L2, %global_ptr : !llvm.ptr<1>
  // CHECK:   nvvm.prefetch level = L2, evict_priority = evict_last, %{{.*}}
  nvvm.prefetch level = L2, evict_priority = evict_last, %global_ptr :
  !llvm.ptr<1>
  // CHECK:   nvvm.prefetch level = L2, evict_priority = evict_normal, %{{.*}}
  nvvm.prefetch level = L2, evict_priority = evict_normal, %global_ptr : !llvm.ptr<1>
  // CHECK:   nvvm.prefetch level = L1 uniform, %{{.*}}
  nvvm.prefetch level = L1 uniform, %gen_ptr : !llvm.ptr
  // CHECK:   nvvm.prefetch tensormap, %{{.*}}
  nvvm.prefetch tensormap, %gen_ptr : !llvm.ptr
  // CHECK:   nvvm.prefetch tensormap, %{{.*}}
  nvvm.prefetch tensormap, %const_ptr : !llvm.ptr<4>
  // CHECK:   nvvm.prefetch tensormap in_param_space, %{{.*}}
  nvvm.prefetch tensormap in_param_space, %gen_ptr : !llvm.ptr
  return
}

// CHECK-LABEL: @prefetch_tensormap
func.func @prefetch_tensormap(%gen_ptr: !llvm.ptr, %const_ptr: !llvm.ptr<4>) {
  return
}

// CHECK-LABEL: @nvvm_address_space
func.func private @nvvm_address_space(
    !ptr.ptr<#nvvm.memory_space<global>>,
    !ptr.ptr<#nvvm.memory_space<shared>>,
    !ptr.ptr<#nvvm.memory_space<constant>>,
    !ptr.ptr<#nvvm.memory_space<local>>,
    !ptr.ptr<#nvvm.memory_space<tensor>>,
    !ptr.ptr<#nvvm.memory_space<shared_cluster>>
  ) -> !ptr.ptr<#nvvm.memory_space<generic>>

// -----

// Just check these don't emit errors.
gpu.module @module_1 [#nvvm.target<chip = "sm_90", features = "+ptx70", link = ["my_device_lib.bc"], flags = {fast, ftz}>] {
}

gpu.module @module_2 [#nvvm.target<chip = "sm_90">, #nvvm.target<chip = "sm_80">, #nvvm.target<chip = "sm_70">] {
}

// CHECK-LABEL: nvvm.grid_constant
llvm.func @kernel_func(%arg0: !llvm.ptr {llvm.byval = i32, nvvm.grid_constant}) attributes {nvvm.kernel} {
  llvm.return
}

// -----

// expected-error @below {{'"nvvm.grid_constant"' attribute must be present only on kernel arguments}}
llvm.func @kernel_func(%arg0: !llvm.ptr {llvm.byval = i32, nvvm.grid_constant}) {
  llvm.return
}

// -----

// expected-error @below {{'"nvvm.grid_constant"' attribute requires the argument to also have attribute 'llvm.byval'}}
llvm.func @kernel_func(%arg0: !llvm.ptr {nvvm.grid_constant}) attributes {nvvm.kernel} {
  llvm.return
}

// -----

// expected-error @below {{'"nvvm.grid_constant"' must be a unit attribute}}
llvm.func @kernel_func(%arg0: !llvm.ptr {llvm.byval = i32, nvvm.grid_constant = true}) attributes {nvvm.kernel} {
  llvm.return
}
