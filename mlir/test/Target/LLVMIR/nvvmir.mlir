// RUN: mlir-translate -mlir-to-llvmir %s  -split-input-file --verify-diagnostics | FileCheck %s

// CHECK-LABEL: @nvvm_special_regs
llvm.func @nvvm_special_regs() -> i32 {
  // CHECK: %1 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %1 = nvvm.read.ptx.sreg.tid.x : i32
  // CHECK: call i32 @llvm.nvvm.read.ptx.sreg.tid.y()
  %2 = nvvm.read.ptx.sreg.tid.y : i32
  // CHECK: call i32 @llvm.nvvm.read.ptx.sreg.tid.z()
  %3 = nvvm.read.ptx.sreg.tid.z : i32
  // CHECK: call i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
  %4 = nvvm.read.ptx.sreg.ntid.x : i32
  // CHECK: call i32 @llvm.nvvm.read.ptx.sreg.ntid.y()
  %5 = nvvm.read.ptx.sreg.ntid.y : i32
  // CHECK: call i32 @llvm.nvvm.read.ptx.sreg.ntid.z()
  %6 = nvvm.read.ptx.sreg.ntid.z : i32
  // CHECK: call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
  %7 = nvvm.read.ptx.sreg.ctaid.x : i32
  // CHECK: call i32 @llvm.nvvm.read.ptx.sreg.ctaid.y()
  %8 = nvvm.read.ptx.sreg.ctaid.y : i32
  // CHECK: call i32 @llvm.nvvm.read.ptx.sreg.ctaid.z()
  %9 = nvvm.read.ptx.sreg.ctaid.z : i32
  // CHECK: call i32 @llvm.nvvm.read.ptx.sreg.nctaid.x()
  %10 = nvvm.read.ptx.sreg.nctaid.x : i32
  // CHECK: call i32 @llvm.nvvm.read.ptx.sreg.nctaid.y()
  %11 = nvvm.read.ptx.sreg.nctaid.y : i32
  // CHECK: call i32 @llvm.nvvm.read.ptx.sreg.nctaid.z()
  %12 = nvvm.read.ptx.sreg.nctaid.z : i32
  // CHECK: call i32 @llvm.nvvm.read.ptx.sreg.warpsize()
  %13 = nvvm.read.ptx.sreg.warpsize : i32
  // CHECK: call i32 @llvm.nvvm.read.ptx.sreg.laneid()
  %14 = nvvm.read.ptx.sreg.laneid : i32
  // CHECK: call i32 @llvm.nvvm.read.ptx.sreg.clusterid.x
  %15 = nvvm.read.ptx.sreg.clusterid.x : i32
  // CHECK: call i32 @llvm.nvvm.read.ptx.sreg.clusterid.y
  %16 = nvvm.read.ptx.sreg.clusterid.y : i32
  // CHECK: call i32 @llvm.nvvm.read.ptx.sreg.clusterid.z
  %17 = nvvm.read.ptx.sreg.clusterid.z : i32
  // CHECK: call i32 @llvm.nvvm.read.ptx.sreg.nclusterid.x
  %18 = nvvm.read.ptx.sreg.nclusterid.x : i32
  // CHECK: call i32 @llvm.nvvm.read.ptx.sreg.nclusterid.y
  %19 = nvvm.read.ptx.sreg.nclusterid.y : i32
  // CHECK: call i32 @llvm.nvvm.read.ptx.sreg.nclusterid.z
  %20 = nvvm.read.ptx.sreg.nclusterid.z : i32
  // CHECK: call i32 @llvm.nvvm.read.ptx.sreg.cluster.ctaid
  %21 = nvvm.read.ptx.sreg.cluster.ctaid.x : i32
  // CHECK: call i32 @llvm.nvvm.read.ptx.sreg.cluster.ctaid
  %22 = nvvm.read.ptx.sreg.cluster.ctaid.y : i32
  // CHECK: call i32 @llvm.nvvm.read.ptx.sreg.cluster.ctaid
  %23 = nvvm.read.ptx.sreg.cluster.ctaid.z : i32
  // CHECK: call i32 @llvm.nvvm.read.ptx.sreg.cluster.nctaid
  %24 = nvvm.read.ptx.sreg.cluster.nctaid.x : i32
  // CHECK: call i32 @llvm.nvvm.read.ptx.sreg.cluster.nctaid
  %25 = nvvm.read.ptx.sreg.cluster.nctaid.y : i32
  // CHECK: call i32 @llvm.nvvm.read.ptx.sreg.cluster.nctaid
  %26 = nvvm.read.ptx.sreg.cluster.nctaid.z : i32
  // CHECK: call i32 @llvm.nvvm.read.ptx.sreg.cluster.ctarank
  %27 = nvvm.read.ptx.sreg.cluster.ctarank : i32
  // CHECK: call i32 @llvm.nvvm.read.ptx.sreg.cluster.nctarank
  %28 = nvvm.read.ptx.sreg.cluster.nctarank : i32
  // CHECK: call i32 @llvm.nvvm.read.ptx.sreg.clock
  %29 = nvvm.read.ptx.sreg.clock : i32
  // CHECK: call i64 @llvm.nvvm.read.ptx.sreg.clock64
  %30 = nvvm.read.ptx.sreg.clock64 : i64
  // CHECK: call i64 @llvm.nvvm.read.ptx.sreg.globaltimer
  %31 = nvvm.read.ptx.sreg.globaltimer : i64
  // CHECK: %32 = call range(i32 0, 64) i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %32 = nvvm.read.ptx.sreg.tid.x range <i32, 0, 64> : i32
  // CHECK: call i32 @llvm.nvvm.read.ptx.sreg.warpid
  %33 = nvvm.read.ptx.sreg.warpid : i32
  // CHECK: call i32 @llvm.nvvm.read.ptx.sreg.nwarpid
  %34 = nvvm.read.ptx.sreg.nwarpid : i32
  // CHECK: call i32 @llvm.nvvm.read.ptx.sreg.smid
  %35 = nvvm.read.ptx.sreg.smid : i32
  // CHECK: call i32 @llvm.nvvm.read.ptx.sreg.nsmid
  %36 = nvvm.read.ptx.sreg.nsmid : i32
  // CHECK: call i32 @llvm.nvvm.read.ptx.sreg.gridid
  %37 = nvvm.read.ptx.sreg.gridid : i32
  //CHECK: call i32 @llvm.nvvm.read.ptx.sreg.envreg0
  %38 = nvvm.read.ptx.sreg.envreg0 : i32
  //CHECK: call i32 @llvm.nvvm.read.ptx.sreg.envreg1
  %39 = nvvm.read.ptx.sreg.envreg1 : i32
  //CHECK: call i32 @llvm.nvvm.read.ptx.sreg.envreg2
  %40 = nvvm.read.ptx.sreg.envreg2 : i32
  //CHECK: call i32 @llvm.nvvm.read.ptx.sreg.envreg3
  %41 = nvvm.read.ptx.sreg.envreg3 : i32
  //CHECK: call i32 @llvm.nvvm.read.ptx.sreg.envreg4
  %42 = nvvm.read.ptx.sreg.envreg4 : i32
  //CHECK: call i32 @llvm.nvvm.read.ptx.sreg.envreg5
  %43 = nvvm.read.ptx.sreg.envreg5 : i32
  //CHECK: call i32 @llvm.nvvm.read.ptx.sreg.envreg6
  %44 = nvvm.read.ptx.sreg.envreg6 : i32
  //CHECK: call i32 @llvm.nvvm.read.ptx.sreg.envreg7
  %45 = nvvm.read.ptx.sreg.envreg7 : i32
  //CHECK: call i32 @llvm.nvvm.read.ptx.sreg.envreg8
  %46 = nvvm.read.ptx.sreg.envreg8 : i32
  //CHECK: call i32 @llvm.nvvm.read.ptx.sreg.envreg9
  %47 = nvvm.read.ptx.sreg.envreg9 : i32
  //CHECK: call i32 @llvm.nvvm.read.ptx.sreg.envreg10
  %48 = nvvm.read.ptx.sreg.envreg10 : i32
  //CHECK: call i32 @llvm.nvvm.read.ptx.sreg.envreg11
  %49 = nvvm.read.ptx.sreg.envreg11 : i32
  //CHECK: call i32 @llvm.nvvm.read.ptx.sreg.envreg12
  %50 = nvvm.read.ptx.sreg.envreg12 : i32
  //CHECK: call i32 @llvm.nvvm.read.ptx.sreg.envreg13
  %51 = nvvm.read.ptx.sreg.envreg13 : i32
  //CHECK: call i32 @llvm.nvvm.read.ptx.sreg.envreg14
  %52 = nvvm.read.ptx.sreg.envreg14 : i32
  //CHECK: call i32 @llvm.nvvm.read.ptx.sreg.envreg15
  %53 = nvvm.read.ptx.sreg.envreg15 : i32
  //CHECK: call i32 @llvm.nvvm.read.ptx.sreg.envreg16
  %54 = nvvm.read.ptx.sreg.envreg16 : i32
  //CHECK: call i32 @llvm.nvvm.read.ptx.sreg.envreg17
  %55 = nvvm.read.ptx.sreg.envreg17 : i32
  //CHECK: call i32 @llvm.nvvm.read.ptx.sreg.envreg18
  %56 = nvvm.read.ptx.sreg.envreg18 : i32
  //CHECK: call i32 @llvm.nvvm.read.ptx.sreg.envreg19
  %57 = nvvm.read.ptx.sreg.envreg19 : i32
  //CHECK: call i32 @llvm.nvvm.read.ptx.sreg.envreg20
  %58 = nvvm.read.ptx.sreg.envreg20 : i32
  //CHECK: call i32 @llvm.nvvm.read.ptx.sreg.envreg21
  %59 = nvvm.read.ptx.sreg.envreg21 : i32
  //CHECK: call i32 @llvm.nvvm.read.ptx.sreg.envreg22
  %60 = nvvm.read.ptx.sreg.envreg22 : i32
  //CHECK: call i32 @llvm.nvvm.read.ptx.sreg.envreg23
  %61 = nvvm.read.ptx.sreg.envreg23 : i32
  //CHECK: call i32 @llvm.nvvm.read.ptx.sreg.envreg24
  %62 = nvvm.read.ptx.sreg.envreg24 : i32
  //CHECK: call i32 @llvm.nvvm.read.ptx.sreg.envreg25
  %63 = nvvm.read.ptx.sreg.envreg25 : i32
  //CHECK: call i32 @llvm.nvvm.read.ptx.sreg.envreg26
  %64 = nvvm.read.ptx.sreg.envreg26 : i32
  //CHECK: call i32 @llvm.nvvm.read.ptx.sreg.envreg27
  %65 = nvvm.read.ptx.sreg.envreg27 : i32
  //CHECK: call i32 @llvm.nvvm.read.ptx.sreg.envreg28
  %66 = nvvm.read.ptx.sreg.envreg28 : i32
  //CHECK: call i32 @llvm.nvvm.read.ptx.sreg.envreg29
  %67 = nvvm.read.ptx.sreg.envreg29 : i32
  //CHECK: call i32 @llvm.nvvm.read.ptx.sreg.envreg30
  %68 = nvvm.read.ptx.sreg.envreg30 : i32
  //CHECK: call i32 @llvm.nvvm.read.ptx.sreg.envreg31
  %69 = nvvm.read.ptx.sreg.envreg31 : i32
  //CHECK: call i32 @llvm.nvvm.read.ptx.sreg.lanemask.eq
  %70 = nvvm.read.ptx.sreg.lanemask.eq : i32
  //CHECK: call i32 @llvm.nvvm.read.ptx.sreg.lanemask.le
  %71 = nvvm.read.ptx.sreg.lanemask.le : i32
  //CHECK: call i32 @llvm.nvvm.read.ptx.sreg.lanemask.lt
  %72 = nvvm.read.ptx.sreg.lanemask.lt : i32
  //CHECK: call i32 @llvm.nvvm.read.ptx.sreg.lanemask.ge
  %73 = nvvm.read.ptx.sreg.lanemask.ge : i32
  //CHECK: call i32 @llvm.nvvm.read.ptx.sreg.lanemask.gt
  %74 = nvvm.read.ptx.sreg.lanemask.gt : i32
  llvm.return %1 : i32
}

// CHECK-LABEL: @nvvm_rcp
llvm.func @nvvm_rcp(%0: f32) -> f32 {
  // CHECK: call float @llvm.nvvm.rcp.approx.ftz.f
  %1 = nvvm.rcp.approx.ftz.f %0 : f32
  llvm.return %1 : f32
}

// CHECK-LABEL: @llvm_nvvm_barrier0
llvm.func @llvm_nvvm_barrier0() {
  // CHECK: call void @llvm.nvvm.barrier0()
  nvvm.barrier0
  llvm.return
}

// CHECK-LABEL: @llvm_nvvm_barrier(
// CHECK-SAME: i32 %[[barId:.*]], i32 %[[numThreads:.*]])
llvm.func @llvm_nvvm_barrier(%barID : i32, %numberOfThreads : i32) {
  // CHECK: call void @llvm.nvvm.barrier0()
  nvvm.barrier 
  // CHECK: call void @llvm.nvvm.barrier.n(i32 %[[barId]])
  nvvm.barrier id = %barID
  // CHECK: call void @llvm.nvvm.barrier(i32 %[[barId]], i32 %[[numThreads]])
  nvvm.barrier id = %barID number_of_threads = %numberOfThreads
  llvm.return
}

// CHECK-LABEL: @llvm_nvvm_cluster_arrive
llvm.func @llvm_nvvm_cluster_arrive() {
  // CHECK: call void @llvm.nvvm.barrier.cluster.arrive()
  nvvm.cluster.arrive
  // CHECK: call void @llvm.nvvm.barrier.cluster.arrive.aligned()
  nvvm.cluster.arrive {aligned}
  llvm.return
}

// CHECK-LABEL: @llvm_nvvm_cluster_arrive_relaxed
llvm.func @llvm_nvvm_cluster_arrive_relaxed() {
  // CHECK: call void @llvm.nvvm.barrier.cluster.arrive.relaxed()
  nvvm.cluster.arrive.relaxed
  // CHECK: call void @llvm.nvvm.barrier.cluster.arrive.relaxed.aligned()
  nvvm.cluster.arrive.relaxed {aligned}
  llvm.return
}

// CHECK-LABEL: @llvm_nvvm_cluster_wait
llvm.func @llvm_nvvm_cluster_wait() {
  // CHECK: call void @llvm.nvvm.barrier.cluster.wait()
  nvvm.cluster.wait
  // CHECK: call void @llvm.nvvm.barrier.cluster.wait.aligned()
  nvvm.cluster.wait {aligned}
  llvm.return
}

// CHECK-LABEL: @nvvm_shfl
llvm.func @nvvm_shfl(
    %0 : i32, %1 : i32, %2 : i32,
    %3 : i32, %4 : f32) -> i32 {
  // CHECK: call i32 @llvm.nvvm.shfl.sync.bfly.i32(i32 %{{.*}}, i32 %{{.*}}, i32 %{{.*}}, i32 %{{.*}})
  %6 = nvvm.shfl.sync bfly %0, %3, %1, %2 : i32 -> i32
  // CHECK: call float @llvm.nvvm.shfl.sync.bfly.f32(i32 %{{.*}}, float %{{.*}}, i32 %{{.*}}, i32 %{{.*}})
  %7 = nvvm.shfl.sync bfly %0, %4, %1, %2 : f32 -> f32
  // CHECK: call i32 @llvm.nvvm.shfl.sync.up.i32(i32 %{{.*}}, i32 %{{.*}}, i32 %{{.*}}, i32 %{{.*}})
  %8 = nvvm.shfl.sync up %0, %3, %1, %2 : i32 -> i32
  // CHECK: call float @llvm.nvvm.shfl.sync.up.f32(i32 %{{.*}}, float %{{.*}}, i32 %{{.*}}, i32 %{{.*}})
  %9 = nvvm.shfl.sync up %0, %4, %1, %2 : f32 -> f32
  // CHECK: call i32 @llvm.nvvm.shfl.sync.down.i32(i32 %{{.*}}, i32 %{{.*}}, i32 %{{.*}}, i32 %{{.*}})
  %10 = nvvm.shfl.sync down %0, %3, %1, %2 : i32 -> i32
  // CHECK: call float @llvm.nvvm.shfl.sync.down.f32(i32 %{{.*}}, float %{{.*}}, i32 %{{.*}}, i32 %{{.*}})
  %11 = nvvm.shfl.sync down %0, %4, %1, %2 : f32 -> f32
  // CHECK: call i32 @llvm.nvvm.shfl.sync.idx.i32(i32 %{{.*}}, i32 %{{.*}}, i32 %{{.*}}, i32 %{{.*}})
  %12 = nvvm.shfl.sync idx %0, %3, %1, %2 : i32 -> i32
  // CHECK: call float @llvm.nvvm.shfl.sync.idx.f32(i32 %{{.*}}, float %{{.*}}, i32 %{{.*}}, i32 %{{.*}})
  %13 = nvvm.shfl.sync idx %0, %4, %1, %2 : f32 -> f32
  llvm.return %6 : i32
}

// CHECK-LABEL: @nvvm_shfl_pred
llvm.func @nvvm_shfl_pred(
    %0 : i32, %1 : i32, %2 : i32,
    %3 : i32, %4 : f32) -> !llvm.struct<(i32, i1)> {
  // CHECK: call { i32, i1 } @llvm.nvvm.shfl.sync.bfly.i32p(i32 %{{.*}}, i32 %{{.*}}, i32 %{{.*}}, i32 %{{.*}})
  %6 = nvvm.shfl.sync bfly %0, %3, %1, %2 {return_value_and_is_valid} : i32 -> !llvm.struct<(i32, i1)>
  // CHECK: call { float, i1 } @llvm.nvvm.shfl.sync.bfly.f32p(i32 %{{.*}}, float %{{.*}}, i32 %{{.*}}, i32 %{{.*}})
  %7 = nvvm.shfl.sync bfly %0, %4, %1, %2 {return_value_and_is_valid} : f32 -> !llvm.struct<(f32, i1)>
  // CHECK: call { i32, i1 } @llvm.nvvm.shfl.sync.up.i32p(i32 %{{.*}}, i32 %{{.*}}, i32 %{{.*}}, i32 %{{.*}})
  %8 = nvvm.shfl.sync up %0, %3, %1, %2 {return_value_and_is_valid} : i32 -> !llvm.struct<(i32, i1)>
  // CHECK: call { float, i1 } @llvm.nvvm.shfl.sync.up.f32p(i32 %{{.*}}, float %{{.*}}, i32 %{{.*}}, i32 %{{.*}})
  %9 = nvvm.shfl.sync up %0, %4, %1, %2 {return_value_and_is_valid} : f32 -> !llvm.struct<(f32, i1)>
  // CHECK: call { i32, i1 } @llvm.nvvm.shfl.sync.down.i32p(i32 %{{.*}}, i32 %{{.*}}, i32 %{{.*}}, i32 %{{.*}})
  %10 = nvvm.shfl.sync down %0, %3, %1, %2 {return_value_and_is_valid} : i32 -> !llvm.struct<(i32, i1)>
  // CHECK: call { float, i1 } @llvm.nvvm.shfl.sync.down.f32p(i32 %{{.*}}, float %{{.*}}, i32 %{{.*}}, i32 %{{.*}})
  %11 = nvvm.shfl.sync down %0, %4, %1, %2 {return_value_and_is_valid} : f32 -> !llvm.struct<(f32, i1)>
  // CHECK: call { i32, i1 } @llvm.nvvm.shfl.sync.idx.i32p(i32 %{{.*}}, i32 %{{.*}}, i32 %{{.*}}, i32 %{{.*}})
  %12 = nvvm.shfl.sync idx %0, %3, %1, %2 {return_value_and_is_valid} : i32 -> !llvm.struct<(i32, i1)>
  // CHECK: call { float, i1 } @llvm.nvvm.shfl.sync.idx.f32p(i32 %{{.*}}, float %{{.*}}, i32 %{{.*}}, i32 %{{.*}})
  %13 = nvvm.shfl.sync idx %0, %4, %1, %2 {return_value_and_is_valid} : f32 -> !llvm.struct<(f32, i1)>
  llvm.return %6 : !llvm.struct<(i32, i1)>
}

// CHECK-LABEL: @nvvm_vote
llvm.func @nvvm_vote(%0 : i32, %1 : i1) -> i32 {
  // CHECK: call i32 @llvm.nvvm.vote.ballot.sync(i32 %{{.*}}, i1 %{{.*}})
  %3 = nvvm.vote.ballot.sync %0, %1 : i32
  llvm.return %3 : i32
}

// CHECK-LABEL: @nvvm_elect_sync
llvm.func @nvvm_elect_sync() -> i1 {
  // CHECK: %[[RES:.*]] = call { i32, i1 } @llvm.nvvm.elect.sync(i32 -1)
  // CHECK-NEXT: %[[PRED:.*]] = extractvalue { i32, i1 } %[[RES]], 1
  // CHECK-NEXT: ret i1 %[[PRED]]
  %0 = nvvm.elect.sync -> i1
  llvm.return %0 : i1
}

// CHECK-LABEL: @nvvm_mma_mn8n8k4_row_col_f32_f32
llvm.func @nvvm_mma_mn8n8k4_row_col_f32_f32(%a0 : vector<2xf16>, %a1 : vector<2xf16>,
                    %b0 : vector<2xf16>, %b1 : vector<2xf16>,
                    %c0 : f32, %c1 : f32, %c2 : f32, %c3 : f32,
                    %c4 : f32, %c5 : f32, %c6 : f32, %c7 : f32) -> !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f32)> {
  // CHECK: call { float, float, float, float, float, float, float, float } @llvm.nvvm.mma.m8n8k4.row.col.f32.f32
  %0 = nvvm.mma.sync A[%a0, %a1] B[%b0, %b1] C[%c0, %c1, %c2, %c3, %c4, %c5, %c6, %c7]
  {layoutA = #nvvm.mma_layout<row>, layoutB = #nvvm.mma_layout<col>, shape = #nvvm.shape<m = 8, n = 8, k = 4>} : (vector<2xf16>, vector<2xf16>, f32) -> !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f32)>
  llvm.return %0 : !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f32)>
}

// CHECK-LABEL: @nvvm_mma_m16n8k16_f16_f16
llvm.func @nvvm_mma_m16n8k16_f16_f16(%a0 : vector<2xf16>, %a1 : vector<2xf16>,
                                %a2 : vector<2xf16>, %a3 : vector<2xf16>,
                                %b0 : vector<2xf16>, %b1 : vector<2xf16>,
                                %c0 : vector<2xf16>, %c1 : vector<2xf16>) -> !llvm.struct<(vector<2xf16>, vector<2xf16>)> {
  // CHECK: call { <2 x half>, <2 x half> } @llvm.nvvm.mma.m16n8k16.row.col.f16.f16
  %0 = nvvm.mma.sync A[ %a0, %a1, %a2, %a3 ] B[ %b0, %b1 ] C[ %c0, %c1 ]
    {layoutA = #nvvm.mma_layout<row>, layoutB = #nvvm.mma_layout<col>, shape = #nvvm.shape<m = 16, n = 8, k = 16>}
     : (vector<2xf16>, vector<2xf16>, vector<2xf16>) -> !llvm.struct<(vector<2xf16>, vector<2xf16>)>
  llvm.return %0 : !llvm.struct<(vector<2xf16>, vector<2xf16>)>
}

// f32 return type, f16 accumulate type
// CHECK-LABEL: @nvvm_mma_m16n8k16_f32_f16
llvm.func @nvvm_mma_m16n8k16_f32_f16(%a0 : vector<2xf16>, %a1 : vector<2xf16>,
                                %a2 : vector<2xf16>, %a3 : vector<2xf16>,
                                %b0 : vector<2xf16>, %b1 : vector<2xf16>,
                                %c0 : vector<2xf16>, %c1 : vector<2xf16>) -> !llvm.struct<(f32, f32, f32, f32)> {
  // CHECK: call { float, float, float, float } @llvm.nvvm.mma.m16n8k16.row.col.f32.f16
  %0 = nvvm.mma.sync A[%a0, %a1, %a2, %a3] B[%b0, %b1] C[%c0, %c1]
    {layoutA = #nvvm.mma_layout<row>, layoutB = #nvvm.mma_layout<col>,
     shape = #nvvm.shape<m = 16, n = 8, k = 16>} : (vector<2xf16>, vector<2xf16>, vector<2xf16>) -> !llvm.struct<(f32, f32, f32, f32)>
  llvm.return %0 : !llvm.struct<(f32, f32, f32, f32)>
}

// f16 return type, f32 accumulate type
// CHECK-LABEL: @nvvm_mma_m16n8k16_f16_f32
llvm.func @nvvm_mma_m16n8k16_f16_f32(%a0 : vector<2xf16>, %a1 : vector<2xf16>,
                                %a2 : vector<2xf16>, %a3 : vector<2xf16>,
                                %b0 : vector<2xf16>, %b1 : vector<2xf16>,
                                %c0 : f32, %c1 : f32, %c2 : f32, %c3 : f32) -> !llvm.struct<(vector<2xf16>, vector<2xf16>)> {
  // CHECK: call { <2 x half>, <2 x half> } @llvm.nvvm.mma.m16n8k16.row.col.f16.f32
  %0 = nvvm.mma.sync A[%a0, %a1, %a2, %a3] B[%b0, %b1] C[%c0, %c1, %c2, %c3]
    {layoutA = #nvvm.mma_layout<row>, layoutB = #nvvm.mma_layout<col>,
     shape = #nvvm.shape<m = 16, n = 8, k = 16>} : (vector<2xf16>, vector<2xf16>, f32) -> !llvm.struct<(vector<2xf16>, vector<2xf16>)>
  llvm.return %0 : !llvm.struct<(vector<2xf16>, vector<2xf16>)>
}

// f32 return type, f32 accumulate type
// CHECK-LABEL: @nvvm_mma_m16n8k16_f32_f32
llvm.func @nvvm_mma_m16n8k16_f32_f32(%a0 : vector<2xf16>, %a1 : vector<2xf16>,
                                %a2 : vector<2xf16>, %a3 : vector<2xf16>,
                                %b0 : vector<2xf16>, %b1 : vector<2xf16>,
                                %c0 : f32, %c1 : f32, %c2 : f32, %c3 : f32) -> !llvm.struct<(f32, f32, f32, f32)> {
  // CHECK: call { float, float, float, float } @llvm.nvvm.mma.m16n8k16.row.col.f32.f32
  %0 = nvvm.mma.sync A[%a0, %a1, %a2, %a3] B[%b0, %b1] C[%c0, %c1, %c2, %c3]
    {layoutA = #nvvm.mma_layout<row>, layoutB = #nvvm.mma_layout<col>,
     shape = #nvvm.shape<m = 16, n = 8, k = 16>} : (vector<2xf16>, vector<2xf16>, f32) -> !llvm.struct<(f32, f32, f32, f32)>
  llvm.return %0 : !llvm.struct<(f32, f32, f32, f32)>
}

// CHECK-LABEL: @nvvm_mma_m16n8k16_s8_s8
llvm.func @nvvm_mma_m16n8k16_s8_s8(%a0 : i32, %a1 : i32,
                                %b0 : i32,
                                %c0 : i32, %c1 : i32, %c2 : i32, %c3 : i32) -> !llvm.struct<(i32, i32, i32, i32)> {
  // CHECK: call { i32, i32, i32, i32 } @llvm.nvvm.mma.m16n8k16.row.col.s8
  %0 = nvvm.mma.sync A[%a0, %a1] B[%b0] C[%c0, %c1, %c2, %c3]
    {layoutA = #nvvm.mma_layout<row>, layoutB = #nvvm.mma_layout<col>,
     multiplicandAPtxType = #nvvm.mma_type<s8>, multiplicandBPtxType = #nvvm.mma_type<s8>,
     intOverflowBehavior=#nvvm.mma_int_overflow<wrapped>,
     shape = #nvvm.shape<m = 16, n = 8, k = 16>} : (i32, i32, i32) -> !llvm.struct<(i32,i32,i32,i32)>
  llvm.return %0 : !llvm.struct<(i32,i32,i32,i32)>
}

// CHECK-LABEL: @nvvm_mma_m16n8k16_s8_u8
llvm.func @nvvm_mma_m16n8k16_s8_u8(%a0 : i32, %a1 : i32,
                                %b0 : i32,
                                %c0 : i32, %c1 : i32, %c2 : i32, %c3 : i32) -> !llvm.struct<(i32, i32, i32, i32)> {
  // CHECK: call { i32, i32, i32, i32 } @llvm.nvvm.mma.m16n8k16.row.col.satfinite.s8.u8
  %0 = nvvm.mma.sync A[%a0, %a1] B[%b0] C[%c0, %c1, %c2, %c3]
    {layoutA = #nvvm.mma_layout<row>, layoutB = #nvvm.mma_layout<col>,
     multiplicandAPtxType = #nvvm.mma_type<s8>, multiplicandBPtxType = #nvvm.mma_type<u8>,
     intOverflowBehavior=#nvvm.mma_int_overflow<satfinite>,
     shape = #nvvm.shape<m = 16, n = 8, k = 16>} : (i32, i32, i32) -> !llvm.struct<(i32,i32,i32,i32)>
  llvm.return %0 : !llvm.struct<(i32,i32,i32,i32)>
}

// CHECK-LABEL: @nvvm_mma_m16n8k128_b1_b1
llvm.func @nvvm_mma_m16n8k128_b1_b1(%a0 : i32, %a1 : i32,
                                    %b0 : i32,
                                    %c0 : i32, %c1 : i32, %c2 : i32, %c3 : i32) -> !llvm.struct<(i32,i32,i32,i32)> {
  // CHECK: call { i32, i32, i32, i32 } @llvm.nvvm.mma.xor.popc.m16n8k128.row.col.b1
  %0 = nvvm.mma.sync A[%a0, %a1] B[%b0] C[%c0, %c1, %c2, %c3]
    {layoutA = #nvvm.mma_layout<row>, layoutB = #nvvm.mma_layout<col>,
     multiplicandAPtxType = #nvvm.mma_type<b1>, multiplicandBPtxType = #nvvm.mma_type<b1>,
     b1Op = #nvvm.mma_b1op<xor_popc>, shape = #nvvm.shape<m = 16, n = 8, k = 128>} : (i32, i32, i32) -> !llvm.struct<(i32,i32,i32,i32)>
  llvm.return %0 : !llvm.struct<(i32,i32,i32,i32)>
}

// CHECK-LABEL: @nvvm_mma_m16n8k32_s4_s4
llvm.func @nvvm_mma_m16n8k32_s4_s4(%a0 : i32, %a1 : i32,
                               %b0 : i32,
                               %c0 : i32, %c1 : i32, %c2 : i32, %c3 : i32) -> !llvm.struct<(i32,i32,i32,i32)> {
  // CHECK: call { i32, i32, i32, i32 } @llvm.nvvm.mma.m16n8k32.row.col.satfinite.s4
  %0 = nvvm.mma.sync A[%a0, %a1] B[%b0] C[%c0, %c1, %c2, %c3]
    {layoutA = #nvvm.mma_layout<row>, layoutB = #nvvm.mma_layout<col>,
     multiplicandAPtxType = #nvvm.mma_type<s4>, multiplicandBPtxType = #nvvm.mma_type<s4>,
     intOverflowBehavior=#nvvm.mma_int_overflow<satfinite>,
     shape = #nvvm.shape<m = 16, n = 8, k = 32>} : (i32, i32, i32) -> !llvm.struct<(i32,i32,i32,i32)>
  llvm.return %0 : !llvm.struct<(i32,i32,i32,i32)>
}

// CHECK-LABEL: @nvvm_mma_m8n8k4_f64_f64
llvm.func @nvvm_mma_m8n8k4_f64_f64(%a0 : f64,
                                   %b0 : f64,
                                   %c0 : f64, %c1 : f64) -> !llvm.struct<(f64, f64)> {
  // CHECK: call { double, double } @llvm.nvvm.mma.m8n8k4.row.col.f64
  %0 = nvvm.mma.sync A[%a0] B[%b0] C[%c0, %c1]
    {layoutA = #nvvm.mma_layout<row>, layoutB = #nvvm.mma_layout<col>,
     shape = #nvvm.shape<m = 8, n = 8, k = 4>} : (f64, f64, f64) -> !llvm.struct<(f64, f64)>
  llvm.return %0 : !llvm.struct<(f64, f64)>
}

// CHECK-LABEL: @nvvm_mma_m16n8k4_tf32_f32
llvm.func @nvvm_mma_m16n8k4_tf32_f32(%a0 : i32, %a1 : i32,
                                     %b0 : i32,
                                     %c0 : f32, %c1 : f32, %c2 : f32, %c3 : f32) -> !llvm.struct<(f32, f32, f32, f32)> {
  // CHECK: call { float, float, float, float } @llvm.nvvm.mma.m16n8k4.row.col.tf32
  %0 = nvvm.mma.sync A[%a0, %a1] B[%b0] C[%c0, %c1, %c2, %c3]
    {layoutA = #nvvm.mma_layout<row>, layoutB = #nvvm.mma_layout<col>,
     multiplicandAPtxType = #nvvm.mma_type<tf32>, multiplicandBPtxType = #nvvm.mma_type<tf32>,
     shape = #nvvm.shape<m = 16, n = 8, k = 4>} : (i32, i32, f32) -> !llvm.struct<(f32, f32, f32, f32)>
  llvm.return %0 : !llvm.struct<(f32, f32, f32, f32)>
}

// The test below checks the correct mapping of the nvvm.wmma.*.load.* op to the correct intrinsic
// in the LLVM NVPTX backend.
// CHECK-LABEL: @gpu_wmma_load_op
llvm.func @gpu_wmma_load_op(%arg0: !llvm.ptr<3>, %arg1: i32) {
  // CHECK: call { <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half> } @llvm.nvvm.wmma.m16n16k16.load.a.row.stride.f16.p3(ptr addrspace(3) %{{.*}}, i32 %{{.*}})
  %0 = nvvm.wmma.load %arg0, %arg1
    {eltype = #nvvm.mma_type<f16>, frag = #nvvm.mma_frag<a>, k = 16 : i32, layout = #nvvm.mma_layout<row>, m = 16 : i32, n = 16 : i32}
    : (!llvm.ptr<3>) -> !llvm.struct<(vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>)>

  llvm.return
}

// The test below checks the correct mapping of the nvvm.wmma.*.store.* op to the correct intrinsic
// in the LLVM NVPTX backend.
// CHECK-LABEL: @gpu_wmma_store_op
llvm.func @gpu_wmma_store_op(%arg0: !llvm.ptr<3>, %arg1: i32,
                            %arg2: vector<2 x f16>, %arg3: vector<2 x f16>,
                            %arg4: vector<2 xf16>, %arg5: vector<2 x f16>) {
  // CHECK: call void @llvm.nvvm.wmma.m16n16k16.store.d.row.stride.f16.p3(ptr addrspace(3) %{{.*}}, <2 x half> {{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, i32 %{{.*}})
  nvvm.wmma.store %arg0, %arg1, %arg2, %arg3, %arg4, %arg5
    {eltype = #nvvm.mma_type<f16>, k = 16 : i32, layout = #nvvm.mma_layout<row>, m = 16 : i32, n = 16 : i32}
    : !llvm.ptr<3>, vector<2 x f16>, vector<2 x f16>, vector<2 x f16>, vector<2 x f16>
  llvm.return
}

// The test below checks the correct mapping of the nvvm.wmma.*.mma.* op to the correct intrinsic
// in the LLVM NVPTX backend.
// CHECK-LABEL: @gpu_wmma_mma_op
llvm.func @gpu_wmma_mma_op(%arg0: vector<2 x f16>, %arg1: vector<2 x f16>,
                        %arg2: vector<2 x f16>, %arg3: vector<2 x f16>,
                        %arg4: vector<2 x f16>, %arg5: vector<2 x f16>,
                        %arg6: vector<2 x f16>, %arg7: vector<2 x f16>,
                        %arg8: vector<2 x f16>, %arg9: vector<2 x f16>,
                        %arg10: vector<2 x f16>, %arg11: vector<2 x f16>,
                        %arg12: vector<2 x f16>, %arg13: vector<2 x f16>,
                        %arg14: vector<2 x f16>, %arg15: vector<2 x f16>,
                        %arg16: vector<2 x f16>, %arg17: vector<2 x f16>,
                        %arg18: vector<2 x f16>, %arg19: vector<2 x f16>) {
  // CHECK: call { <2 x half>, <2 x half>, <2 x half>, <2 x half> } @llvm.nvvm.wmma.m16n16k16.mma.row.row.f16.f16(<2 x half> {{.*}}, <2 x half> {{.*}}, <2 x half> {{.*}}, <2 x half> {{.*}}, <2 x half> {{.*}}, <2 x half> {{.*}}, <2 x half> {{.*}}, <2 x half> {{.*}}, <2 x half> {{.*}}, <2 x half> {{.*}}, <2 x half> {{.*}}, <2 x half> {{.*}}, <2 x half> {{.*}}, <2 x half> {{.*}}, <2 x half> {{.*}}, <2 x half> {{.*}}, <2 x half> {{.*}}, <2 x half> {{.*}}, <2 x half> {{.*}}, <2 x half> {{.*}})
  %0 = nvvm.wmma.mma %arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15, %arg16, %arg17, %arg18, %arg19
    {eltypeA = #nvvm.mma_type<f16>, eltypeB = #nvvm.mma_type<f16>, k = 16 : i32, layoutA = #nvvm.mma_layout<row>, layoutB = #nvvm.mma_layout<row>, m = 16 : i32, n = 16 : i32}
    : (vector<2 x f16>, vector<2 x f16>, vector<2 x f16>, vector<2 x f16>, vector<2 x f16>,
       vector<2 x f16>, vector<2 x f16>, vector<2 x f16>, vector<2 x f16>, vector<2 x f16>,
       vector<2 x f16>, vector<2 x f16>, vector<2 x f16>, vector<2 x f16>, vector<2 x f16>,
       vector<2 x f16>, vector<2 x f16>, vector<2 x f16>, vector<2 x f16>, vector<2 x f16>)
      -> !llvm.struct<(vector<2 x f16>, vector<2 x f16>, vector<2 x f16>, vector<2 x f16>)>
  llvm.return
}

// CHECK-LABEL: @nvvm_wmma_load_tf32
llvm.func @nvvm_wmma_load_tf32(%arg0: !llvm.ptr, %arg1 : i32) {
  // CHECK: call { i32, i32, i32, i32 } @llvm.nvvm.wmma.m16n16k8.load.a.row.stride.tf32.p0(ptr %{{.*}}, i32 %{{.*}})
  %0 = nvvm.wmma.load %arg0, %arg1
    {eltype = #nvvm.mma_type<tf32>, frag = #nvvm.mma_frag<a>, k = 8 : i32, layout = #nvvm.mma_layout<row>, m = 16 : i32, n = 16 : i32}
    : (!llvm.ptr) -> !llvm.struct<(i32, i32, i32, i32)>
  llvm.return
}

// CHECK-LABEL: @nvvm_wmma_mma
llvm.func @nvvm_wmma_mma(%0 : i32, %1 : i32, %2 : i32, %3 : i32, %4 : i32, %5 : i32,
                    %6 : i32, %7 : i32, %8 : f32, %9 : f32, %10 : f32,
                    %11 : f32, %12 : f32, %13 : f32, %14 : f32, %15 : f32) {
  // CHECK: { float, float, float, float, float, float, float, float } @llvm.nvvm.wmma.m16n16k8.mma.row.row.tf32(i32 %{{.*}}, i32 %{{.*}}, i32 %{{.*}}, i32 %{{.*}}, i32 %{{.*}}, i32 %{{.*}}, i32 %{{.*}}, i32 %{{.*}}, float %{{.*}}, float %{{.*}}, float %{{.*}}, float %{{.*}}, float %{{.*}}, float %{{.*}}, float %{{.*}}, float %{{.*}})
  %r = nvvm.wmma.mma %0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15
    {eltypeA = #nvvm.mma_type<tf32>, eltypeB = #nvvm.mma_type<f32>, k = 8 : i32, layoutA = #nvvm.mma_layout<row>, layoutB = #nvvm.mma_layout<row>, m = 16 : i32, n = 16 : i32}
    : (i32, i32, i32, i32, i32, i32, i32, i32, f32, f32, f32, f32, f32, f32, f32, f32)
    -> !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f32)>
  llvm.return
}

// CHECK-LABEL: @cp_async
llvm.func @cp_async(%arg0: !llvm.ptr<3>, %arg1: !llvm.ptr<1>) {
// CHECK: call void @llvm.nvvm.cp.async.ca.shared.global.4(ptr addrspace(3) %{{.*}}, ptr addrspace(1) %{{.*}})
  nvvm.cp.async.shared.global %arg0, %arg1, 4, cache =  ca : !llvm.ptr<3>, !llvm.ptr<1>
// CHECK: call void @llvm.nvvm.cp.async.ca.shared.global.8(ptr addrspace(3) %{{.*}}, ptr addrspace(1) %{{.*}})
  nvvm.cp.async.shared.global %arg0, %arg1, 8, cache =  ca : !llvm.ptr<3>, !llvm.ptr<1>
// CHECK: call void @llvm.nvvm.cp.async.ca.shared.global.16(ptr addrspace(3) %{{.*}}, ptr addrspace(1) %{{.*}})
  nvvm.cp.async.shared.global %arg0, %arg1, 16, cache =  ca : !llvm.ptr<3>, !llvm.ptr<1>
// CHECK: call void @llvm.nvvm.cp.async.cg.shared.global.16(ptr addrspace(3) %{{.*}}, ptr addrspace(1) %{{.*}})
  nvvm.cp.async.shared.global %arg0, %arg1, 16, cache =  cg : !llvm.ptr<3>, !llvm.ptr<1>
// CHECK: call void @llvm.nvvm.cp.async.commit.group()
  nvvm.cp.async.commit.group
// CHECK: call void @llvm.nvvm.cp.async.wait.group(i32 0)
  nvvm.cp.async.wait.group 0
  llvm.return
}

// CHECK-LABEL: @cp_async_mbarrier_arrive
llvm.func @cp_async_mbarrier_arrive(%bar_shared: !llvm.ptr<3>, %bar_gen: !llvm.ptr) {
  // CHECK: call void @llvm.nvvm.cp.async.mbarrier.arrive(ptr %{{.*}})
  nvvm.cp.async.mbarrier.arrive %bar_gen : !llvm.ptr
  // CHECK: call void @llvm.nvvm.cp.async.mbarrier.arrive.noinc(ptr %{{.*}})
  nvvm.cp.async.mbarrier.arrive %bar_gen {noinc = true} : !llvm.ptr
  // CHECK: call void @llvm.nvvm.cp.async.mbarrier.arrive.shared(ptr addrspace(3) %{{.*}})
  nvvm.cp.async.mbarrier.arrive.shared %bar_shared : !llvm.ptr<3>
  // CHECK: call void @llvm.nvvm.cp.async.mbarrier.arrive.noinc.shared(ptr addrspace(3) %{{.*}})
  nvvm.cp.async.mbarrier.arrive.shared %bar_shared {noinc = true} : !llvm.ptr<3>
  llvm.return
}

// CHECK-LABEL: @llvm_nvvm_setmaxregister
llvm.func @llvm_nvvm_setmaxregister() {
  // CHECK: call void @llvm.nvvm.setmaxnreg.inc.sync.aligned.u32(i32 256)
  nvvm.setmaxregister increase 256
  // CHECK: call void @llvm.nvvm.setmaxnreg.dec.sync.aligned.u32(i32 24)
  nvvm.setmaxregister decrease 24
  llvm.return
}

// CHECK-LABEL: @llvm_nvvm_cp_async_bulk_commit_group
llvm.func @llvm_nvvm_cp_async_bulk_commit_group() {
  // CHECK: call void @llvm.nvvm.cp.async.bulk.commit.group()
  nvvm.cp.async.bulk.commit.group
  llvm.return
}

// CHECK-LABEL: @llvm_nvvm_cp_async_bulk_wait_group
llvm.func @llvm_nvvm_cp_async_bulk_wait_group() {
  // CHECK: call void @llvm.nvvm.cp.async.bulk.wait.group(i32 0)
  nvvm.cp.async.bulk.wait_group 0
  // CHECK: call void @llvm.nvvm.cp.async.bulk.wait.group(i32 3)
  nvvm.cp.async.bulk.wait_group 3
  // CHECK: call void @llvm.nvvm.cp.async.bulk.wait.group.read(i32 0)
  nvvm.cp.async.bulk.wait_group 0 {read}
  // CHECK: call void @llvm.nvvm.cp.async.bulk.wait.group.read(i32 3)
  nvvm.cp.async.bulk.wait_group 3 {read}
  llvm.return
}

// CHECK-LABEL: @ld_matrix
llvm.func @ld_matrix(%arg0: !llvm.ptr<3>) {
  // CHECK: call i32 @llvm.nvvm.ldmatrix.sync.aligned.m8n8.x1.b16.p3(ptr addrspace(3) %{{.*}})
  %l1 = nvvm.ldmatrix %arg0 {num = 1 : i32, layout = #nvvm.mma_layout<row>} : (!llvm.ptr<3>) -> i32
  // CHECK: call { i32, i32 } @llvm.nvvm.ldmatrix.sync.aligned.m8n8.x2.b16.p3(ptr addrspace(3) %{{.*}})
  %l2 = nvvm.ldmatrix %arg0 {num = 2 : i32, layout = #nvvm.mma_layout<row>} : (!llvm.ptr<3>) -> !llvm.struct<(i32, i32)>
  // CHECK: call { i32, i32, i32, i32 } @llvm.nvvm.ldmatrix.sync.aligned.m8n8.x4.b16.p3(ptr addrspace(3) %{{.*}})
  %l4 = nvvm.ldmatrix %arg0 {num = 4 : i32, layout = #nvvm.mma_layout<row>} : (!llvm.ptr<3>) -> !llvm.struct<(i32, i32, i32, i32)>
   // CHECK: call i32 @llvm.nvvm.ldmatrix.sync.aligned.m8n8.x1.trans.b16.p3(ptr addrspace(3) %{{.*}})
  %l1t = nvvm.ldmatrix %arg0 {num = 1 : i32, layout = #nvvm.mma_layout<col>} : (!llvm.ptr<3>) -> i32
  // CHECK: call { i32, i32 } @llvm.nvvm.ldmatrix.sync.aligned.m8n8.x2.trans.b16.p3(ptr addrspace(3) %{{.*}})
  %l2t = nvvm.ldmatrix %arg0 {num = 2 : i32, layout = #nvvm.mma_layout<col>} : (!llvm.ptr<3>) -> !llvm.struct<(i32, i32)>
  // CHECK: call { i32, i32, i32, i32 } @llvm.nvvm.ldmatrix.sync.aligned.m8n8.x4.trans.b16.p3(ptr addrspace(3) %{{.*}})
  %l4t = nvvm.ldmatrix %arg0 {num = 4 : i32, layout = #nvvm.mma_layout<col>} : (!llvm.ptr<3>) -> !llvm.struct<(i32, i32, i32, i32)>
  llvm.return
}

// This function has the "kernel" attribute attached and should appear in the
// NVVM annotations after conversion.
llvm.func @kernel_func() attributes {nvvm.kernel} {
  llvm.return
}

// CHECK: ptx_kernel void @kernel_func

// -----

llvm.func @kernel_func() attributes {nvvm.kernel, nvvm.maxntid = array<i32: 1, 23, 32>} {
  llvm.return
}

// CHECK: define ptx_kernel void @kernel_func
// CHECK:     !nvvm.annotations =
// CHECK:     {ptr @kernel_func, !"maxntidx", i32 1}
// CHECK:     {ptr @kernel_func, !"maxntidy", i32 23}
// CHECK:     {ptr @kernel_func, !"maxntidz", i32 32}
// -----

llvm.func @kernel_func() attributes {nvvm.kernel, nvvm.reqntid = array<i32: 1, 23, 32>} {
  llvm.return
}

// CHECK: define ptx_kernel void @kernel_func
// CHECK:     !nvvm.annotations =
// CHECK:     {ptr @kernel_func, !"reqntidx", i32 1}
// CHECK:     {ptr @kernel_func, !"reqntidy", i32 23}
// CHECK:     {ptr @kernel_func, !"reqntidz", i32 32}
// -----

llvm.func @kernel_func() attributes {nvvm.kernel, nvvm.cluster_dim = array<i32: 3, 5, 7>} {
  llvm.return
}

// CHECK: define ptx_kernel void @kernel_func
// CHECK:     !nvvm.annotations =
// CHECK:     {ptr @kernel_func, !"cluster_dim_x", i32 3}
// CHECK:     {ptr @kernel_func, !"cluster_dim_y", i32 5}
// CHECK:     {ptr @kernel_func, !"cluster_dim_z", i32 7}
// -----

llvm.func @kernel_func() attributes {nvvm.kernel, nvvm.cluster_max_blocks = 8} {
  llvm.return
}

// CHECK: define ptx_kernel void @kernel_func
// CHECK:     !nvvm.annotations =
// CHECK:     {ptr @kernel_func, !"cluster_max_blocks", i32 8}
// -----

llvm.func @kernel_func() attributes {nvvm.kernel, nvvm.minctasm = 16} {
  llvm.return
}

// CHECK: define ptx_kernel void @kernel_func
// CHECK:     !nvvm.annotations =
// CHECK:     {ptr @kernel_func, !"minctasm", i32 16}
// -----

llvm.func @kernel_func() attributes {nvvm.kernel, nvvm.maxnreg = 16} {
  llvm.return
}

// CHECK: define ptx_kernel void @kernel_func
// CHECK:     !nvvm.annotations =
// CHECK:     {ptr @kernel_func, !"maxnreg", i32 16}
// -----

llvm.func @kernel_func() attributes {nvvm.kernel, nvvm.maxntid = array<i32: 1, 23, 32>,
                                     nvvm.minctasm = 16, nvvm.maxnreg = 32} {
  llvm.return
}

// CHECK: define ptx_kernel void @kernel_func
// CHECK:     !nvvm.annotations =
// CHECK:     {ptr @kernel_func, !"maxnreg", i32 32}
// CHECK:     {ptr @kernel_func, !"maxntidx", i32 1}
// CHECK:     {ptr @kernel_func, !"maxntidy", i32 23}
// CHECK:     {ptr @kernel_func, !"maxntidz", i32 32}
// CHECK:     {ptr @kernel_func, !"minctasm", i32 16}

// -----
// CHECK: define ptx_kernel void @kernel_func
// CHECK: !nvvm.annotations =
// CHECK: !1 = !{ptr @kernel_func, !"grid_constant", !2}
// CHECK: !2 = !{i32 1}
llvm.func @kernel_func(%arg0: !llvm.ptr {llvm.byval = i32, nvvm.grid_constant}) attributes {nvvm.kernel} {
  llvm.return
}

// -----
// CHECK: define ptx_kernel void @kernel_func
// CHECK: !nvvm.annotations =
// CHECK: !1 = !{ptr @kernel_func, !"grid_constant", !2}
// CHECK: !2 = !{i32 1, i32 3}
llvm.func @kernel_func(%arg0: !llvm.ptr {llvm.byval = i32, nvvm.grid_constant}, %arg1: f32, %arg2: !llvm.ptr {llvm.byval = f32, nvvm.grid_constant}) attributes {nvvm.kernel} {
  llvm.return
}


// -----
// CHECK-LABEL: @nvvm_fence_proxy_tensormap_generic_release
llvm.func @nvvm_fence_proxy_tensormap_generic_release() {
  %c128 = llvm.mlir.constant(128) : i32
  // CHECK: call void @llvm.nvvm.fence.proxy.tensormap_generic.release.cta()
  nvvm.fence.proxy.release #nvvm.mem_scope<cta>

  // CHECK: call void @llvm.nvvm.fence.proxy.tensormap_generic.release.cluster()
  nvvm.fence.proxy.release #nvvm.mem_scope<cluster>

  // CHECK: call void @llvm.nvvm.fence.proxy.tensormap_generic.release.gpu()
  nvvm.fence.proxy.release #nvvm.mem_scope<gpu>

  // CHECK: call void @llvm.nvvm.fence.proxy.tensormap_generic.release.sys()
  nvvm.fence.proxy.release #nvvm.mem_scope<sys>
  llvm.return
}

// -----
// CHECK-LABEL: @nvvm_fence_proxy_tensormap_generic_acquire
llvm.func @nvvm_fence_proxy_tensormap_generic_acquire(%addr : !llvm.ptr) {
  %c128 = llvm.mlir.constant(128) : i32
  // CHECK: call void @llvm.nvvm.fence.proxy.tensormap_generic.acquire.cta(ptr {{%[0-9]+}}, i32 128)
  nvvm.fence.proxy.acquire #nvvm.mem_scope<cta> %addr, %c128

  // CHECK: call void @llvm.nvvm.fence.proxy.tensormap_generic.acquire.cluster(ptr {{%[0-9]+}}, i32 128)
  nvvm.fence.proxy.acquire #nvvm.mem_scope<cluster> %addr, %c128

  // CHECK: call void @llvm.nvvm.fence.proxy.tensormap_generic.acquire.gpu(ptr {{%[0-9]+}}, i32 128)
  nvvm.fence.proxy.acquire #nvvm.mem_scope<gpu> %addr, %c128

  // CHECK: call void @llvm.nvvm.fence.proxy.tensormap_generic.acquire.sys(ptr {{%[0-9]+}}, i32 128)
  nvvm.fence.proxy.acquire #nvvm.mem_scope<sys> %addr, %c128
  llvm.return
}
// -----

// CHECK-LABEL: @nvvm_exit
llvm.func @nvvm_exit() {
  // CHECK: call void @llvm.nvvm.exit()
  nvvm.exit
  llvm.return
}



// -----
// CHECK-LABEL: @nvvm_breakpoint
llvm.func @nvvm_breakpoint() {
  // CHECK: call void @llvm.debugtrap()
  nvvm.breakpoint
  llvm.return
}

// -----
// CHECK-LABEL: @nvvm_wgmma_fence_aligned
llvm.func @nvvm_wgmma_fence_aligned() {
  // CHECK: call void @llvm.nvvm.wgmma.fence.sync.aligned()
  nvvm.wgmma.fence.aligned
  llvm.return
}

// -----
// CHECK-LABEL: @nvvm_wgmma_commit_group_aligned
llvm.func @nvvm_wgmma_commit_group_aligned() {
  // CHECK: call void @llvm.nvvm.wgmma.commit_group.sync.aligned()
  nvvm.wgmma.commit.group.sync.aligned
  llvm.return
}

// -----
// CHECK-LABEL: @nvvm_wgmma_wait_group_aligned
llvm.func @nvvm_wgmma_wait_group_aligned() {
  // CHECK: call void @llvm.nvvm.wgmma.wait_group.sync.aligned(i64 0)
  nvvm.wgmma.wait.group.sync.aligned 0
  // CHECK: call void @llvm.nvvm.wgmma.wait_group.sync.aligned(i64 20)
  nvvm.wgmma.wait.group.sync.aligned 20
  llvm.return
}
