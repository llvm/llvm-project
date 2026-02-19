
// RUN: mlir-translate -mlir-to-llvmir %s | FileCheck %s

// Tests that we correctly lower the different variations of reference pointer
// and attach semantics.

module attributes {omp.is_gpu = false, omp.is_target_device = false, omp.requires = #omp<clause_requires none>, omp.target_triples = ["amdgcn-amd-amdhsa"], omp.version = #omp.version<version = 61>} {
  llvm.func @attach_always_(%arg0: !llvm.ptr, %arg1: !llvm.ptr) {
    %map1 = omp.map.info var_ptr(%arg0 : !llvm.ptr, !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8)>) map_clauses(tofrom) capture(ByRef) var_ptr_ptr(%arg1 : !llvm.ptr, i32) -> !llvm.ptr {name = ""}
    %map2 = omp.map.info var_ptr(%arg0 : !llvm.ptr, !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8)>) map_clauses(always, to) capture(ByRef) members(%map1 : [0] : !llvm.ptr) -> !llvm.ptr {name = "x"}
    %map3 = omp.map.info var_ptr(%arg0 : !llvm.ptr, !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8)>) map_clauses(always, attach, ref_ptr_ptee) capture(ByRef) var_ptr_ptr(%arg1 : !llvm.ptr, i32) -> !llvm.ptr {name = "x"}
    omp.target map_entries(%map2 -> %arg2, %map3 -> %arg3, %map1 -> %arg4 : !llvm.ptr, !llvm.ptr, !llvm.ptr) {
      omp.terminator
    }
    llvm.return
  }

  llvm.func @attach_never_(%arg0: !llvm.ptr, %arg1: !llvm.ptr) {
    %map1 = omp.map.info var_ptr(%arg0 : !llvm.ptr, !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8)>) map_clauses(tofrom) capture(ByRef) var_ptr_ptr(%arg1 : !llvm.ptr, i32) -> !llvm.ptr {name = ""}
    %map2 = omp.map.info var_ptr(%arg0 : !llvm.ptr, !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8)>) map_clauses(always, to) capture(ByRef) members(%map1 : [0] : !llvm.ptr) -> !llvm.ptr {name = "x"}
    omp.target map_entries(%map2 -> %arg2, %map1 -> %arg3 : !llvm.ptr, !llvm.ptr) {
      omp.terminator
    }
    llvm.return
  }

  llvm.func @attach_auto_(%arg0: !llvm.ptr, %arg1: !llvm.ptr) {
    %map1 = omp.map.info var_ptr(%arg0 : !llvm.ptr, !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8)>) map_clauses(tofrom) capture(ByRef) var_ptr_ptr(%arg1 : !llvm.ptr, i32) -> !llvm.ptr {name = ""}
    %map2 = omp.map.info var_ptr(%arg0 : !llvm.ptr, !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8)>) map_clauses(always, to) capture(ByRef) members(%map1 : [0] : !llvm.ptr) -> !llvm.ptr {name = "x"}
    %map3 = omp.map.info var_ptr(%arg0 : !llvm.ptr, !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8)>) map_clauses(attach, ref_ptr_ptee) capture(ByRef) var_ptr_ptr(%arg1 : !llvm.ptr, i32) -> !llvm.ptr {name = "x"}
    omp.target map_entries(%map2 -> %arg2, %map3 -> %arg3, %map1 -> %arg4 : !llvm.ptr, !llvm.ptr, !llvm.ptr) {
      omp.terminator
    }
    llvm.return
  }

  llvm.func @ref_ptr_ptee_(%arg0: !llvm.ptr, %arg1: !llvm.ptr) {
    %map1 = omp.map.info var_ptr(%arg0 : !llvm.ptr, !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8)>) map_clauses(to, ref_ptr_ptee) capture(ByRef) var_ptr_ptr(%arg1 : !llvm.ptr, i32) -> !llvm.ptr {name = ""}
    %map2 = omp.map.info var_ptr(%arg0 : !llvm.ptr, !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8)>) map_clauses(to, ref_ptr_ptee) capture(ByRef) members(%map1 : [0] : !llvm.ptr) -> !llvm.ptr {name = "x"}
    %map3 = omp.map.info var_ptr(%arg0 : !llvm.ptr, !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8)>) map_clauses(attach, ref_ptr_ptee) capture(ByRef) var_ptr_ptr(%arg1 : !llvm.ptr, i32) -> !llvm.ptr {name = "x"}
    omp.target map_entries(%map2 -> %arg2, %map3 -> %arg3, %map1 -> %arg4 : !llvm.ptr, !llvm.ptr, !llvm.ptr) {
      omp.terminator
    }
    llvm.return
  }

  llvm.func @ref_ptr_(%arg0: !llvm.ptr, %arg1: !llvm.ptr) {
    %map1 = omp.map.info var_ptr(%arg0 : !llvm.ptr, !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8)>) map_clauses(to, ref_ptr) capture(ByRef) -> !llvm.ptr {name = ""}
    %map2 = omp.map.info var_ptr(%arg0 : !llvm.ptr, !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8)>) map_clauses(attach, ref_ptr) capture(ByRef) var_ptr_ptr(%arg1 : !llvm.ptr, i32) -> !llvm.ptr {name = "x"}
    omp.target map_entries(%map2 -> %arg2, %map1 -> %arg3 : !llvm.ptr, !llvm.ptr) {
      omp.terminator
    }
    llvm.return
  }

  llvm.func @ref_ptee_(%arg0: !llvm.ptr, %arg1: !llvm.ptr) {
    %map1 = omp.map.info var_ptr(%arg0 : !llvm.ptr, !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8)>) map_clauses(to, ref_ptee) capture(ByRef) var_ptr_ptr(%arg1 : !llvm.ptr, i32) -> !llvm.ptr {name = ""}
    %map2 = omp.map.info var_ptr(%arg0 : !llvm.ptr, !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8)>) map_clauses(attach, ref_ptee) capture(ByRef) var_ptr_ptr(%arg1 : !llvm.ptr, i32) -> !llvm.ptr {name = "x"}
    omp.target map_entries(%map2 -> %arg2, %map1 -> %arg3 : !llvm.ptr, !llvm.ptr) {
      omp.terminator
    }
    llvm.return
  }

  llvm.func @ref_ptr_ptee_attach_never_(%arg0: !llvm.ptr, %arg1: !llvm.ptr) {
    %map1 = omp.map.info var_ptr(%arg0 : !llvm.ptr, !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8)>) map_clauses(to, ref_ptr_ptee) capture(ByRef) var_ptr_ptr(%arg1 : !llvm.ptr, i32) -> !llvm.ptr {name = ""}
    %map2 = omp.map.info var_ptr(%arg0 : !llvm.ptr, !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8)>) map_clauses(to, ref_ptr_ptee) capture(ByRef) members(%map1 : [0] : !llvm.ptr) -> !llvm.ptr {name = "x"}
    omp.target map_entries(%map2 -> %arg2, %map1 -> %arg3 : !llvm.ptr, !llvm.ptr) {
      omp.terminator
    }
    llvm.return
  }
}

// CHECK: @.offload_maptypes{{.*}} = private unnamed_addr constant [4 x i64] [i64 32, i64 281474976710661, i64 3, i64 16388]
// CHECK: @.offload_maptypes{{.*}} = private unnamed_addr constant [3 x i64] [i64 32, i64 281474976710661, i64 3]
// CHECK: @.offload_maptypes{{.*}} = private unnamed_addr constant [4 x i64] [i64 32, i64 281474976710661, i64 3, i64 16384]
// CHECK: @.offload_maptypes{{.*}} = private unnamed_addr constant [4 x i64] [i64 32, i64 281474976710657, i64 1, i64 16384]
// CHECK: @.offload_sizes{{.*}} = private unnamed_addr constant [2 x i64] [i64 0, i64 24]
// CHECK: @.offload_maptypes{{.*}} = private unnamed_addr constant [2 x i64] [i64 16384, i64 33]
// CHECK: @.offload_maptypes{{.*}} = private unnamed_addr constant [2 x i64] [i64 16384, i64 33]
// CHECK: @.offload_maptypes{{.*}} = private unnamed_addr constant [3 x i64] [i64 32, i64 281474976710657, i64 1]

// CHECK: define void @attach_always_(ptr %[[ARG0:.*]], ptr %[[ARG1:.*]])
// CHECK:  %[[VAL_0:.*]] = load ptr, ptr %[[ARG1]], align 8
// CHECK:  %[[VAL_1:.*]] = load ptr, ptr %[[ARG1]], align 8
// CHECK:  %[[VAL_2:.*]] = getelementptr { ptr, i64, i32, i8, i8, i8, i8 }, ptr %[[ARG0]], i32 1
// CHECK:  %[[VAL_3:.*]] = ptrtoint ptr %[[VAL_2]] to i64
// CHECK:  %[[VAL_4:.*]] = ptrtoint ptr %[[ARG0]] to i64
// CHECK:  %[[VAL_5:.*]] = sub i64 %[[VAL_3]], %[[VAL_4]]
// CHECK:  %[[VAL_6:.*]] = sdiv exact i64 %[[VAL_5]], ptrtoint (ptr getelementptr (i8, ptr null, i32 1) to i64)
// CHECK:  %[[VAL_1_CMP:.*]] = icmp eq ptr %[[VAL_1]], null
// CHECK:  %[[VAL_1_SEL:.*]] = select i1 %[[VAL_1_CMP]], i64 0, i64 4
// CHECK:  %[[VAL_2_CMP:.*]] = icmp eq ptr %[[VAL_0]], null
// CHECK:  %[[VAL_2_SEL:.*]] = select i1 %[[VAL_2_CMP]], i64 0, i64 24

// CHECK:  %[[BASEPTRS:.*]] = getelementptr inbounds [4 x ptr], ptr %.offload_baseptrs, i32 0, i32 0
// CHECK:  store ptr %[[ARG0]], ptr %[[BASEPTRS]], align 8
// CHECK:  %[[OFFPTRS:.*]] = getelementptr inbounds [4 x ptr], ptr %.offload_ptrs, i32 0, i32 0
// CHECK:  store ptr %[[ARG0]], ptr %[[OFFPTRS]], align 8
// CHECK:  %[[SIZES:.*]] = getelementptr inbounds [4 x i64], ptr %.offload_sizes, i32 0, i32 0
// CHECK:  store i64 %[[VAL_6]], ptr %[[SIZES]], align 8
// CHECK:  %[[BASEPTRS:.*]] = getelementptr inbounds [4 x ptr], ptr %.offload_baseptrs, i32 0, i32 1
// CHECK:  store ptr %[[ARG0]], ptr %[[BASEPTRS]], align 8
// CHECK:  %[[OFFPTRS:.*]] = getelementptr inbounds [4 x ptr], ptr %.offload_ptrs, i32 0, i32 1
// CHECK:  store ptr %[[ARG0]], ptr %[[OFFPTRS]], align 8
// CHECK:  %[[BASEPTRS:.*]] = getelementptr inbounds [4 x ptr], ptr %.offload_baseptrs, i32 0, i32 2
// CHECK:  store ptr %[[ARG0]], ptr %[[BASEPTRS]], align 8
// CHECK:  %[[OFFPTRS:.*]] = getelementptr inbounds [4 x ptr], ptr %.offload_ptrs, i32 0, i32 2
// CHECK:  store ptr %[[VAL_1]], ptr %[[OFFPTRS]], align 8
// CHECK:  %[[SIZES:.*]] = getelementptr inbounds [4 x i64], ptr %.offload_sizes, i32 0, i32 2
// CHECK:  store i64 %[[VAL_1_SEL]], ptr %[[SIZES]], align 8
// CHECK:  %[[BASEPTRS:.*]] = getelementptr inbounds [4 x ptr], ptr %.offload_baseptrs, i32 0, i32 3
// CHECK:  store ptr %[[ARG0]], ptr %[[BASEPTRS]], align 8
// CHECK:  %[[OFFPTRS:.*]] = getelementptr inbounds [4 x ptr], ptr %.offload_ptrs, i32 0, i32 3
// CHECK:  store ptr %[[VAL_0]], ptr %[[OFFPTRS]], align 8
// CHECK:  %[[SIZES:.*]] = getelementptr inbounds [4 x i64], ptr %.offload_sizes, i32 0, i32 3
// CHECK:  store i64 %[[VAL_2_SEL]], ptr %[[SIZES]], align 8



// CHECK: define void @attach_never_(ptr %[[ARG0:.*]], ptr %[[ARG1:.*]])
// CHECK:  %[[VAL_1:.*]] = load ptr, ptr %[[ARG1]], align 8
// CHECK:  %[[VAL_2:.*]] = getelementptr { ptr, i64, i32, i8, i8, i8, i8 }, ptr %[[ARG0]], i32 1
// CHECK:  %[[VAL_3:.*]] = ptrtoint ptr %[[VAL_2]] to i64
// CHECK:  %[[VAL_4:.*]] = ptrtoint ptr %[[ARG0]] to i64
// CHECK:  %[[VAL_5:.*]] = sub i64 %[[VAL_3]], %[[VAL_4]]
// CHECK:  %[[VAL_6:.*]] = sdiv exact i64 %[[VAL_5]], ptrtoint (ptr getelementptr (i8, ptr null, i32 1) to i64)
// CHECK:  %[[VAL_7:.*]] = icmp eq ptr %[[VAL_1]], null
// CHECK:  %[[VAL_8:.*]] = select i1 %[[VAL_7]], i64 0, i64 4
// CHECK:  %[[BASEPTRS:.*]] = getelementptr inbounds [3 x ptr], ptr %.offload_baseptrs, i32 0, i32 0
// CHECK:  store ptr %[[ARG0]], ptr %[[BASEPTRS]], align 8
// CHECK:  %[[OFFPTRS:.*]] = getelementptr inbounds [3 x ptr], ptr %.offload_ptrs, i32 0, i32 0
// CHECK:  store ptr %[[ARG0]], ptr %[[OFFPTRS]], align 8
// CHECK:  %[[SIZES:.*]] = getelementptr inbounds [3 x i64], ptr %.offload_sizes, i32 0, i32 0
// CHECK:  store i64 %[[VAL_6]], ptr %[[SIZES]], align 8
// CHECK:  %[[BASEPTRS:.*]] = getelementptr inbounds [3 x ptr], ptr %.offload_baseptrs, i32 0, i32 1
// CHECK:  store ptr %[[ARG0]], ptr %[[BASEPTRS]], align 8
// CHECK:  %[[OFFPTRS:.*]] = getelementptr inbounds [3 x ptr], ptr %.offload_ptrs, i32 0, i32 1
// CHECK:  store ptr %[[ARG0]], ptr %[[OFFPTRS]], align 8
// CHECK:  %[[BASEPTRS:.*]] = getelementptr inbounds [3 x ptr], ptr %.offload_baseptrs, i32 0, i32 2
// CHECK:  store ptr %[[ARG0]], ptr %[[BASEPTRS]], align 8
// CHECK:  %[[OFFPTRS:.*]] = getelementptr inbounds [3 x ptr], ptr %.offload_ptrs, i32 0, i32 2
// CHECK:  store ptr %[[VAL_1]], ptr %[[OFFPTRS]], align 8
// CHECK:  %[[SIZES:.*]] = getelementptr inbounds [3 x i64], ptr %.offload_sizes, i32 0, i32 2
// CHECK:  store i64 %[[VAL_8]], ptr %[[SIZES]], align 8


// CHECK: define void @attach_auto_(ptr %[[ARG0:.*]], ptr %[[ARG1:.*]])
// CHECK:  %[[VAL_0:.*]] = load ptr, ptr %[[ARG1]], align 8
// CHECK:  %[[VAL_1:.*]] = load ptr, ptr %[[ARG1]], align 8
// CHECK:  %[[VAL_2:.*]] = getelementptr { ptr, i64, i32, i8, i8, i8, i8 }, ptr %[[ARG0]], i32 1
// CHECK:  %[[VAL_3:.*]] = ptrtoint ptr %[[VAL_2]] to i64
// CHECK:  %[[VAL_4:.*]] = ptrtoint ptr %[[ARG0]] to i64
// CHECK:  %[[VAL_5:.*]] = sub i64 %[[VAL_3]], %[[VAL_4]]
// CHECK:  %[[VAL_6:.*]] = sdiv exact i64 %[[VAL_5]], ptrtoint (ptr getelementptr (i8, ptr null, i32 1) to i64)
// CHECK:  %[[VAL_1_CMP:.*]] = icmp eq ptr %[[VAL_1]], null
// CHECK:  %[[VAL_1_SEL:.*]] = select i1 %[[VAL_1_CMP]], i64 0, i64 4
// CHECK:  %[[VAL_2_CMP:.*]] = icmp eq ptr %[[VAL_0]], null
// CHECK:  %[[VAL_2_SEL:.*]] = select i1 %[[VAL_2_CMP]], i64 0, i64 24
// CHECK:  %[[BASEPTRS:.*]] = getelementptr inbounds [4 x ptr], ptr %.offload_baseptrs, i32 0, i32 0
// CHECK:  store ptr %[[ARG0]], ptr %[[BASEPTRS]], align 8
// CHECK:  %[[OFFPTRS:.*]] = getelementptr inbounds [4 x ptr], ptr %.offload_ptrs, i32 0, i32 0
// CHECK:  store ptr %[[ARG0]], ptr %[[OFFPTRS]], align 8
// CHECK:  %[[SIZES:.*]] = getelementptr inbounds [4 x i64], ptr %.offload_sizes, i32 0, i32 0
// CHECK:  store i64 %[[VAL_6]], ptr %[[SIZES]], align 8
// CHECK:  %[[BASEPTRS:.*]] = getelementptr inbounds [4 x ptr], ptr %.offload_baseptrs, i32 0, i32 1
// CHECK:  store ptr %[[ARG0]], ptr %[[BASEPTRS]], align 8
// CHECK:  %[[OFFPTRS:.*]] = getelementptr inbounds [4 x ptr], ptr %.offload_ptrs, i32 0, i32 1
// CHECK:  store ptr %[[ARG0]], ptr %[[OFFPTRS]], align 8
// CHECK:  %[[BASEPTRS:.*]] = getelementptr inbounds [4 x ptr], ptr %.offload_baseptrs, i32 0, i32 2
// CHECK:  store ptr %[[ARG0]], ptr %[[BASEPTRS]], align 8
// CHECK:  %[[OFFPTRS:.*]] = getelementptr inbounds [4 x ptr], ptr %.offload_ptrs, i32 0, i32 2
// CHECK:  store ptr %[[VAL_1]], ptr %[[OFFPTRS]], align 8
// CHECK:  %[[SIZES:.*]] = getelementptr inbounds [4 x i64], ptr %.offload_sizes, i32 0, i32 2
// CHECK:  store i64 %[[VAL_1_SEL]], ptr %[[SIZES]], align 8
// CHECK:  %[[BASEPTRS:.*]] = getelementptr inbounds [4 x ptr], ptr %.offload_baseptrs, i32 0, i32 3
// CHECK:  store ptr %[[ARG0]], ptr %[[BASEPTRS]], align 8
// CHECK:  %[[OFFPTRS:.*]] = getelementptr inbounds [4 x ptr], ptr %.offload_ptrs, i32 0, i32 3
// CHECK:  store ptr %[[VAL_0]], ptr %[[OFFPTRS]], align 8
// CHECK:  %[[SIZES:.*]] = getelementptr inbounds [4 x i64], ptr %.offload_sizes, i32 0, i32 3
// CHECK:  store i64 %[[VAL_2_SEL]], ptr %[[SIZES]], align 8


// CHECK: define void @ref_ptr_ptee_(ptr %[[ARG0:.*]], ptr %[[ARG1:.*]])
// CHECK:  %[[VAL_0:.*]] = load ptr, ptr %[[ARG1]], align 8
// CHECK:  %[[VAL_1:.*]] = load ptr, ptr %[[ARG1]], align 8
// CHECK:  %[[VAL_2:.*]] = getelementptr { ptr, i64, i32, i8, i8, i8, i8 }, ptr %[[ARG0]], i32 1
// CHECK:  %[[VAL_3:.*]] = ptrtoint ptr %[[VAL_2]] to i64
// CHECK:  %[[VAL_4:.*]] = ptrtoint ptr %[[ARG0]] to i64
// CHECK:  %[[VAL_5:.*]] = sub i64 %[[VAL_3]], %[[VAL_4]]
// CHECK:  %[[VAL_6:.*]] = sdiv exact i64 %[[VAL_5]], ptrtoint (ptr getelementptr (i8, ptr null, i32 1) to i64)
// CHECK:  %[[VAL_1_CMP:.*]] = icmp eq ptr %[[VAL_1]], null
// CHECK:  %[[VAL_1_SEL:.*]] = select i1 %[[VAL_1_CMP]], i64 0, i64 4
// CHECK:  %[[VAL_2_CMP:.*]] = icmp eq ptr %[[VAL_0]], null
// CHECK:  %[[VAL_2_SEL:.*]] = select i1 %[[VAL_2_CMP]], i64 0, i64 24
// CHECK:  %[[BASEPTRS:.*]] = getelementptr inbounds [4 x ptr], ptr %.offload_baseptrs, i32 0, i32 0
// CHECK:  store ptr %[[ARG0]], ptr %[[BASEPTRS]], align 8
// CHECK:  %[[OFFPTRS:.*]] = getelementptr inbounds [4 x ptr], ptr %.offload_ptrs, i32 0, i32 0
// CHECK:  store ptr %[[ARG0]], ptr %[[OFFPTRS]], align 8
// CHECK:  %[[SIZES:.*]] = getelementptr inbounds [4 x i64], ptr %.offload_sizes, i32 0, i32 0
// CHECK:  store i64 %[[VAL_6]], ptr %[[SIZES]], align 8
// CHECK:  %[[BASEPTRS:.*]] = getelementptr inbounds [4 x ptr], ptr %.offload_baseptrs, i32 0, i32 1
// CHECK:  store ptr %[[ARG0]], ptr %[[BASEPTRS]], align 8
// CHECK:  %[[OFFPTRS:.*]] = getelementptr inbounds [4 x ptr], ptr %.offload_ptrs, i32 0, i32 1
// CHECK:  store ptr %[[ARG0]], ptr %[[OFFPTRS]], align 8
// CHECK:  %[[BASEPTRS:.*]] = getelementptr inbounds [4 x ptr], ptr %.offload_baseptrs, i32 0, i32 2
// CHECK:  store ptr %[[ARG1]], ptr %[[BASEPTRS]], align 8
// CHECK:  %[[OFFPTRS:.*]] = getelementptr inbounds [4 x ptr], ptr %.offload_ptrs, i32 0, i32 2
// CHECK:  store ptr %[[VAL_1]], ptr %[[OFFPTRS]], align 8
// CHECK:  %[[SIZES:.*]] = getelementptr inbounds [4 x i64], ptr %.offload_sizes, i32 0, i32 2
// CHECK:  store i64 %[[VAL_1_SEL]], ptr %[[SIZES]], align 8
// CHECK:  %[[BASEPTRS:.*]] = getelementptr inbounds [4 x ptr], ptr %.offload_baseptrs, i32 0, i32 3
// CHECK:  store ptr %[[ARG0]], ptr %[[BASEPTRS]], align 8
// CHECK:  %[[OFFPTRS:.*]] = getelementptr inbounds [4 x ptr], ptr %.offload_ptrs, i32 0, i32 3
// CHECK:  store ptr %[[VAL_0]], ptr %[[OFFPTRS]], align 8
// CHECK:  %[[SIZES:.*]] = getelementptr inbounds [4 x i64], ptr %.offload_sizes, i32 0, i32 3
// CHECK:  store i64 %[[VAL_2_SEL]], ptr %[[SIZES]], align 8

// CHECK: define void @ref_ptr_(ptr %[[ARG0:.*]], ptr %[[ARG1:.*]])
// CHECK:  %[[VAL_0:.*]] = load ptr, ptr %[[ARG1]], align 8
// CHECK:  %[[VAL_1:.*]] = icmp eq ptr %[[VAL_0]], null
// CHECK:  %[[VAL_2:.*]] = select i1 %[[VAL_1]], i64 0, i64 24
// CHECK:  %[[BASEPTRS:.*]] = getelementptr inbounds [2 x ptr], ptr %.offload_baseptrs, i32 0, i32 0
// CHECK:  store ptr %[[ARG0]], ptr %[[BASEPTRS]], align 8
// CHECK:  %[[OFFPTRS:.*]] = getelementptr inbounds [2 x ptr], ptr %.offload_ptrs, i32 0, i32 0
// CHECK:  store ptr %[[VAL_0]], ptr %[[OFFPTRS]], align 8
// CHECK:  %[[SIZES:.*]] = getelementptr inbounds [2 x i64], ptr %.offload_sizes, i32 0, i32 0
// CHECK:  store i64 %[[VAL_2]], ptr %[[SIZES]], align 8
// CHECK:  %[[BASEPTRS:.*]] = getelementptr inbounds [2 x ptr], ptr %.offload_baseptrs, i32 0, i32 1
// CHECK:  store ptr %[[ARG0]], ptr %[[BASEPTRS]], align 8
// CHECK:  %[[OFFPTRS:.*]] = getelementptr inbounds [2 x ptr], ptr %.offload_ptrs, i32 0, i32 1
// CHECK:  store ptr %[[ARG0]], ptr %[[OFFPTRS]], align 8

// CHECK: define void @ref_ptee_(ptr %[[ARG0:.*]], ptr %[[ARG1:.*]])
// CHECK: %[[VAL_0:.*]] = load ptr, ptr %[[ARG1]], align 8
// CHECK: %[[VAL_1:.*]] = load ptr, ptr %[[ARG1]], align 8
// CHECK: %[[VAL_2:.*]] = icmp eq ptr %[[VAL_0]], null
// CHECK: %[[VAL_3:.*]] = select i1 %[[VAL_2]], i64 0, i64 24
// CHECK: %[[VAL_4:.*]] = icmp eq ptr %[[VAL_1]], null
// CHECK: %[[VAL_5:.*]] = select i1 %[[VAL_4]], i64 0, i64 4
// CHECK:  %[[BASEPTRS:.*]] = getelementptr inbounds [2 x ptr], ptr %.offload_baseptrs, i32 0, i32 0
// CHECK:  store ptr %[[ARG0]], ptr %[[BASEPTRS]], align 8
// CHECK:  %[[OFFPTRS:.*]] = getelementptr inbounds [2 x ptr], ptr %.offload_ptrs, i32 0, i32 0
// CHECK:  store ptr %[[VAL_0]], ptr %[[OFFPTRS]], align 8
// CHECK:  %[[SIZES:.*]] = getelementptr inbounds [2 x i64], ptr %.offload_sizes, i32 0, i32 0
// CHECK:  store i64 %[[VAL_3]], ptr %[[SIZES]], align 8
// CHECK:  %[[BASEPTRS:.*]] = getelementptr inbounds [2 x ptr], ptr %.offload_baseptrs, i32 0, i32 1
// CHECK:  store ptr %[[ARG1]], ptr %[[BASEPTRS]], align 8
// CHECK:  %[[OFFPTRS:.*]] = getelementptr inbounds [2 x ptr], ptr %.offload_ptrs, i32 0, i32 1
// CHECK:  store ptr %[[VAL_1]], ptr %[[OFFPTRS]], align 8
// CHECK:  %[[SIZES:.*]] = getelementptr inbounds [2 x i64], ptr %.offload_sizes, i32 0, i32 1
// CHECK:  store i64 %[[VAL_5]], ptr %[[SIZES]], align 8

// CHECK: define void @ref_ptr_ptee_attach_never_(ptr %[[ARG0:.*]], ptr %[[ARG1:.*]])
// CHECK:  %[[VAL_1:.*]] = load ptr, ptr %[[ARG1]], align 8
// CHECK:  %[[VAL_2:.*]] = getelementptr { ptr, i64, i32, i8, i8, i8, i8 }, ptr %[[ARG0]], i32 1
// CHECK:  %[[VAL_3:.*]] = ptrtoint ptr %[[VAL_2]] to i64
// CHECK:  %[[VAL_4:.*]] = ptrtoint ptr %[[ARG0]] to i64
// CHECK:  %[[VAL_5:.*]] = sub i64 %[[VAL_3]], %[[VAL_4]]
// CHECK:  %[[VAL_6:.*]] = sdiv exact i64 %[[VAL_5]], ptrtoint (ptr getelementptr (i8, ptr null, i32 1) to i64)
// CHECK:  %[[VAL_7:.*]] = icmp eq ptr %[[VAL_1]], null
// CHECK:  %[[VAL_8:.*]] = select i1 %[[VAL_7]], i64 0, i64 4
// CHECK:  %[[BASEPTRS:.*]] = getelementptr inbounds [3 x ptr], ptr %.offload_baseptrs, i32 0, i32 0
// CHECK:  store ptr %[[ARG0]], ptr %[[BASEPTRS]], align 8
// CHECK:  %[[OFFPTRS:.*]] = getelementptr inbounds [3 x ptr], ptr %.offload_ptrs, i32 0, i32 0
// CHECK:  store ptr %[[ARG0]], ptr %[[OFFPTRS]], align 8
// CHECK:  %[[SIZES:.*]] = getelementptr inbounds [3 x i64], ptr %.offload_sizes, i32 0, i32 0
// CHECK:  store i64 %[[VAL_6]], ptr %[[SIZES]], align 8
// CHECK:  %[[BASEPTRS:.*]] = getelementptr inbounds [3 x ptr], ptr %.offload_baseptrs, i32 0, i32 1
// CHECK:  store ptr %[[ARG0]], ptr %[[BASEPTRS]], align 8
// CHECK:  %[[OFFPTRS:.*]] = getelementptr inbounds [3 x ptr], ptr %.offload_ptrs, i32 0, i32 1
// CHECK:  store ptr %[[ARG0]], ptr %[[OFFPTRS]], align 8
// CHECK:  %[[BASEPTRS:.*]] = getelementptr inbounds [3 x ptr], ptr %.offload_baseptrs, i32 0, i32 2
// CHECK:  store ptr %[[ARG1]], ptr %[[BASEPTRS]], align 8
// CHECK:  %[[OFFPTRS:.*]] = getelementptr inbounds [3 x ptr], ptr %.offload_ptrs, i32 0, i32 2
// CHECK:  store ptr %[[VAL_1]], ptr %[[OFFPTRS]], align 8
// CHECK:  %[[SIZES:.*]] = getelementptr inbounds [3 x i64], ptr %.offload_sizes, i32 0, i32 2
// CHECK:  store i64 %[[VAL_8]], ptr %[[SIZES]], align 8
