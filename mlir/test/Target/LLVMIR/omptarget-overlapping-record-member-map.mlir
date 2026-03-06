// RUN: mlir-translate -mlir-to-llvmir %s | FileCheck %s

module attributes {llvm.target_triple = "x86_64-unknown-linux-gnu", omp.is_gpu = false, omp.is_target_device = false, omp.target_triples = ["amdgcn-amd-amdhsa"]} {
    llvm.func @_QQmain() attributes {fir.bindc_name = "main"} {
        %0 = llvm.mlir.constant(1 : i64) : i64
        %1 = llvm.alloca %0 x !llvm.struct<"_QFTdtype", (f32, i32)> {bindc_name = "dtypev"} : (i64) -> !llvm.ptr
        %2 = llvm.getelementptr %1[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"_QFTdtype", (f32, i32)>
        %3 = omp.map.info var_ptr(%2 : !llvm.ptr, i32) map_clauses(tofrom) capture(ByRef) -> !llvm.ptr {name = "dtypev%value2"}
        %4 = omp.map.info var_ptr(%1 : !llvm.ptr, !llvm.struct<"_QFTdtype", (f32, i32)>) map_clauses(to) capture(ByRef) members(%3 : [1] : !llvm.ptr) -> !llvm.ptr {name = "dtypev"}
        omp.target map_entries(%4 -> %arg0, %3 -> %arg1 : !llvm.ptr, !llvm.ptr) {
          omp.terminator
        }
        llvm.return
    }
}

// CHECK: @.offload_sizes = private unnamed_addr constant [4 x i64] [i64 0, i64 0, i64 0, i64 4]
// CHECK: @.offload_maptypes = private unnamed_addr constant [4 x i64] [i64 32, i64 281474976710657, i64 281474976710657, i64 281474976710659]

// CHECK: %[[ALLOCA:.*]] = alloca %_QFTdtype, i64 1, align 8
// CHECK: %[[ELEMENT_ACC:.*]] = getelementptr %_QFTdtype, ptr %[[ALLOCA]], i32 0, i32 1

// CHECK: %[[SIZE1_CALC_1:.*]] = getelementptr %_QFTdtype, ptr %[[ALLOCA]], i32 1
// CHECK: %[[SIZE1_CALC_2:.*]] = ptrtoaddr ptr %[[SIZE1_CALC_1]] to i64
// CHECK: %[[SIZE1_CALC_3:.*]] = ptrtoaddr ptr %[[ALLOCA]] to i64
// CHECK: %[[SIZE1_CALC_4:.*]] = sub i64 %[[SIZE1_CALC_2]], %[[SIZE1_CALC_3]]

// CHECK:  %[[SIZE2_CALC_1:.*]] = getelementptr %_QFTdtype, ptr %[[ALLOCA]], i32 1
// CHECK:  %[[SIZE2_CALC_2:.*]] = ptrtoaddr ptr %[[ELEMENT_ACC]] to i64
// CHECK:  %[[SIZE2_CALC_3:.*]] = ptrtoaddr ptr %[[ALLOCA]] to i64
// CHECK:  %[[SIZE2_CALC_4:.*]] = sub i64 %[[SIZE2_CALC_2]], %[[SIZE2_CALC_3]]

// CHECK:  %[[SIZE3_CALC_1:.*]] = getelementptr i32, ptr %[[ELEMENT_ACC]], i32 1
// CHECK:  %[[SIZE3_CALC_2:.*]] = ptrtoaddr ptr %[[SIZE2_CALC_1]] to i64
// CHECK:  %[[SIZE3_CALC_3:.*]] = ptrtoaddr ptr %[[SIZE3_CALC_1]] to i64
// CHECK:  %[[SIZE3_CALC_4:.*]] = sub i64 %[[SIZE3_CALC_2]], %[[SIZE3_CALC_3]]

// CHECK: %[[BASEPTR:.*]] = getelementptr inbounds [4 x ptr], ptr %.offload_baseptrs, i32 0, i32 0
// CHECK: store ptr %[[ALLOCA]], ptr %[[BASEPTR]], align 8
// CHECK: %[[PTRS:.*]] = getelementptr inbounds [4 x ptr], ptr %.offload_ptrs, i32 0, i32 0
// CHECK: store ptr %[[ALLOCA]], ptr %[[PTRS]], align 8
// CHECK: %[[SIZES:.*]] = getelementptr inbounds [4 x i64], ptr %.offload_sizes, i32 0, i32 0
// CHECK: store i64 %[[SIZE1_CALC_4]], ptr %[[SIZES]], align 8

// CHECK: %[[BASEPTR:.*]] = getelementptr inbounds [4 x ptr], ptr %.offload_baseptrs, i32 0, i32 1
// CHECK: store ptr %[[ALLOCA]], ptr %[[BASEPTR]], align 8
// CHECK: %[[PTRS:.*]] = getelementptr inbounds [4 x ptr], ptr %.offload_ptrs, i32 0, i32 1
// CHECK: store ptr %[[ALLOCA]], ptr %[[PTRS]], align 8
// CHECK: %[[SIZES:.*]] = getelementptr inbounds [4 x i64], ptr %.offload_sizes, i32 0, i32 1
// CHECK: store i64 %[[SIZE2_CALC_4]], ptr %[[SIZES]], align 8

// CHECK: %[[BASEPTR:.*]] = getelementptr inbounds [4 x ptr], ptr %.offload_baseptrs, i32 0, i32 2
// CHECK: store ptr %[[ALLOCA]], ptr %[[BASEPTR]], align 8
// CHECK: %[[PTRS:.*]] = getelementptr inbounds [4 x ptr], ptr %.offload_ptrs, i32 0, i32 2
// CHECK: store ptr %11, ptr %[[PTRS]], align 8
// CHECK: %[[SIZES:.*]] = getelementptr inbounds [4 x i64], ptr %.offload_sizes, i32 0, i32 2
// CHECK: store i64 %[[SIZE3_CALC_4]], ptr %[[SIZES]], align 8

// CHECK: %[[BASEPTR:.*]] = getelementptr inbounds [4 x ptr], ptr %.offload_baseptrs, i32 0, i32 3
// CHECK: store ptr %[[ALLOCA]], ptr %[[BASEPTR]], align 8
// CHECK: %[[PTRS:.*]] = getelementptr inbounds [4 x ptr], ptr %.offload_ptrs, i32 0, i32 3
// CHECK: store ptr %[[ELEMENT_ACC]], ptr %[[PTRS]], align 8
