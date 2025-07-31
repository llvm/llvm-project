// RUN: mlir-translate -mlir-to-llvmir %s | FileCheck %s

// CHECK: atomicrmw add ptr %loadgep_, i32 1 monotonic, align 4, !amdgpu.ignore.denormal.mode !{{.*}}, !amdgpu.no.fine.grained.memory !{{.*}}, !amdgpu.no.remote.memory !{{.*}}

module attributes {dlti.dl_spec = #dlti.dl_spec<!llvm.ptr = dense<64> : vector<4xi64>, !llvm.ptr<1> = dense<64> : vector<4xi64>, !llvm.ptr<2> = dense<32> : vector<4xi64>, !llvm.ptr<3> = dense<32> : vector<4xi64>, !llvm.ptr<4> = dense<64> : vector<4xi64>, !llvm.ptr<5> = dense<32> : vector<4xi64>, !llvm.ptr<6> = dense<32> : vector<4xi64>, !llvm.ptr<7> = dense<[160, 256, 256, 32]> : vector<4xi64>, !llvm.ptr<8> = dense<[128, 128, 128, 48]> : vector<4xi64>, !llvm.ptr<9> = dense<[192, 256, 256, 32]> : vector<4xi64>, i64 = dense<64> : vector<2xi64>, i1 = dense<8> : vector<2xi64>, i8 = dense<8> : vector<2xi64>, i16 = dense<16> : vector<2xi64>, i32 = dense<32> : vector<2xi64>, f16 = dense<16> : vector<2xi64>, f64 = dense<64> : vector<2xi64>, f128 = dense<128> : vector<2xi64>, "dlti.endianness" = "little", "dlti.legal_int_widths" = array<i32: 32, 64>, "dlti.stack_alignment" = 32 : i64, "dlti.alloca_memory_space" = 5 : ui64, "dlti.global_memory_space" = 1 : ui64>, fir.atomic_ignore_denormal_mode, fir.defaultkind = "a1c4d8i4l4r4", fir.kindmap = "", fir.target_cpu = "generic-hsa", llvm.data_layout = "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128:128:48-p9:192:256:256:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8:9", llvm.target_triple = "amdgcn-amd-amdhsa", omp.flags = #omp.flags<openmp_device_version = 31>, omp.is_gpu = true, omp.is_target_device = true, omp.requires = #omp<clause_requires none>, omp.target_triples = [], omp.version = #omp.version<version = 31>} {
  llvm.func @_QQmain() attributes {fir.bindc_name = "TEST", omp.declare_target = #omp.declaretarget<device_type = (host), capture_clause = (to)>, target_cpu = "generic-hsa"} {
    %0 = llvm.mlir.constant(1 : i64) : i64
    %1 = llvm.alloca %0 x i32 {bindc_name = "threads"} : (i64) -> !llvm.ptr<5>
    %2 = llvm.addrspacecast %1 : !llvm.ptr<5> to !llvm.ptr
    %3 = llvm.mlir.constant(1 : i64) : i64
    %4 = llvm.alloca %3 x i32 {bindc_name = "a"} : (i64) -> !llvm.ptr<5>
    %5 = llvm.addrspacecast %4 : !llvm.ptr<5> to !llvm.ptr
    %6 = llvm.mlir.constant(0 : i32) : i32
    %7 = llvm.mlir.constant(128 : i32) : i32
    %8 = llvm.mlir.constant(1 : i64) : i64
    %9 = llvm.mlir.constant(1 : i64) : i64
    llvm.store %7, %2 : i32, !llvm.ptr
    llvm.store %6, %5 : i32, !llvm.ptr
    %10 = omp.map.info var_ptr(%2 : !llvm.ptr, i32) map_clauses(implicit, exit_release_or_enter_alloc) capture(ByCopy) -> !llvm.ptr {name = "threads"}
    %11 = omp.map.info var_ptr(%5 : !llvm.ptr, i32) map_clauses(implicit, exit_release_or_enter_alloc) capture(ByCopy) -> !llvm.ptr {name = "a"}
    omp.target map_entries(%10 -> %arg0, %11 -> %arg1 : !llvm.ptr, !llvm.ptr) {
      %12 = llvm.mlir.constant(1 : i32) : i32
      %13 = llvm.load %arg0 : !llvm.ptr -> i32
      omp.parallel num_threads(%13 : i32) {
        omp.atomic.update %arg1 : !llvm.ptr {
        ^bb0(%arg2: i32):
          %14 = llvm.add %arg2, %12 : i32
          omp.yield(%14 : i32)
        } {atomic_control = #omp.atomic_control<ignore_denormal_mode = true>}
        omp.terminator
      }
      omp.terminator
    }
    llvm.return
  }
}
