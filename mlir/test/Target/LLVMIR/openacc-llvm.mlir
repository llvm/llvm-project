// RUN: mlir-translate -mlir-to-llvmir -split-input-file %s | FileCheck %s

llvm.func @testenterdataop(%arg0: !llvm.ptr, %arg1 : !llvm.ptr) {
  %0 = acc.create varPtr(%arg0 : !llvm.ptr) varType(f32) -> !llvm.ptr
  %1 = acc.copyin varPtr(%arg1 : !llvm.ptr) varType(f32) -> !llvm.ptr
  acc.enter_data dataOperands(%0, %1 : !llvm.ptr, !llvm.ptr)
  llvm.return
}

// CHECK: %struct.ident_t = type { i32, i32, i32, i32, ptr }

// CHECK: @[[LOCSTR:.*]] = private unnamed_addr constant [{{[0-9]*}} x i8] c";{{.*}};testenterdataop;{{[0-9]*}};{{[0-9]*}};;\00", align 1
// CHECK: @[[LOCGLOBAL:.*]] = private unnamed_addr constant %struct.ident_t { i32 0, i32 2, i32 0, i32 {{[0-9]*}}, ptr @[[LOCSTR]] }, align 8
// CHECK: @[[MAPNAME1:.*]] = private unnamed_addr constant [{{[0-9]*}} x i8] c";{{.*}};unknown;{{[0-9]*}};{{[0-9]*}};;\00", align 1
// CHECK: @[[MAPNAME2:.*]] = private unnamed_addr constant [{{[0-9]*}} x i8] c";{{.*}};unknown;{{[0-9]*}};{{[0-9]*}};;\00", align 1
// CHECK: @[[MAPTYPES:.*]] = private unnamed_addr constant [{{[0-9]*}} x i64] [i64 0, i64 1]
// CHECK: @[[MAPNAMES:.*]] = private constant [{{[0-9]*}} x ptr] [ptr @[[MAPNAME1]], ptr @[[MAPNAME2]]]

// CHECK: define void @testenterdataop(ptr %[[PTR0:.*]], ptr %[[PTR1:.*]])
// CHECK: %[[OFFLOAD_BASEPTR:.*]] = alloca [{{[0-9]*}} x ptr]
// CHECK: %[[OFFLOAD_PTRS:.*]] = alloca [{{[0-9]*}} x ptr]
// CHECK: %[[OFFLOAD_SIZES:.*]] = alloca [{{[0-9]*}} x i64]

// CHECK: %[[OFFLOAD_BASEPTR_GEP:.*]] = getelementptr inbounds [2 x ptr], ptr %[[OFFLOAD_BASEPTR]], i32 0, i32 0
// CHECK: store ptr %[[PTR0]], ptr %[[OFFLOAD_BASEPTR_GEP]]
// CHECK: %[[OFFLOAD_PTRS_GEP:.*]] = getelementptr inbounds [2 x ptr], ptr %[[OFFLOAD_PTRS]], i32 0, i32 0
// CHECK: store ptr %[[PTR0]], ptr %[[OFFLOAD_PTRS_GEP]]
// CHECK: %[[OFFLOAD_SIZES_GEP:.*]] = getelementptr inbounds [2 x i64], ptr %[[OFFLOAD_SIZES]], i32 0, i32 0
// CHECK: store i64 ptrtoint (ptr getelementptr (ptr, ptr null, i32 1) to i64), ptr %[[OFFLOAD_SIZES_GEP]]

// CHECK: %[[OFFLOAD_BASEPTR_GEP:.*]] = getelementptr inbounds [2 x ptr], ptr %[[OFFLOAD_BASEPTR]], i32 0, i32 1
// CHECK: store ptr %[[PTR1]], ptr %[[OFFLOAD_BASEPTR_GEP]]
// CHECK: %[[OFFLOAD_PTRS_GEP:.*]] = getelementptr inbounds [2 x ptr], ptr %[[OFFLOAD_PTRS]], i32 0, i32 1
// CHECK: store ptr %[[PTR1]], ptr %[[OFFLOAD_PTRS_GEP]]
// CHECK: %[[OFFLOAD_SIZES_GEP:.*]] = getelementptr inbounds [2 x i64], ptr %[[OFFLOAD_SIZES]], i32 0, i32 1
// CHECK: store i64 ptrtoint (ptr getelementptr (ptr, ptr null, i32 1) to i64), ptr %[[OFFLOAD_SIZES_GEP]]


// CHECK: %[[OFFLOAD_BASEPTR_GEP:.*]] = getelementptr inbounds [2 x ptr], ptr %[[OFFLOAD_BASEPTR]], i32 0, i32 0
// CHECK: %[[OFFLOAD_PTRS_GEP:.*]] = getelementptr inbounds [2 x ptr], ptr %[[OFFLOAD_PTRS]], i32 0, i32 0
// CHECK: %[[OFFLOAD_SIZES_GEP:.*]] = getelementptr inbounds [2 x i64], ptr %[[OFFLOAD_SIZES]], i32 0, i32 0

// CHECK: call void @__tgt_target_data_begin_mapper(ptr @[[LOCGLOBAL]], i64 -1, i32 2, ptr %[[OFFLOAD_BASEPTR_GEP]], ptr %[[OFFLOAD_PTRS_GEP]], ptr %[[OFFLOAD_SIZES_GEP]], ptr @[[MAPTYPES]], ptr @[[MAPNAMES]], ptr null)

// CHECK: declare void @__tgt_target_data_begin_mapper(ptr, i64, i32, ptr, ptr, ptr, ptr, ptr, ptr) #0

// -----


llvm.func @testexitdataop(%arg0: !llvm.ptr, %arg1: !llvm.ptr) {
  %arg0_devptr = acc.getdeviceptr varPtr(%arg0 : !llvm.ptr) varType(f32) -> !llvm.ptr
  %1 = acc.getdeviceptr varPtr(%arg1 : !llvm.ptr) varType(f32) -> !llvm.ptr
  acc.exit_data dataOperands(%arg0_devptr, %1 : !llvm.ptr, !llvm.ptr)
  acc.delete accPtr(%arg0_devptr : !llvm.ptr)
  acc.copyout accPtr(%1 : !llvm.ptr) to varPtr(%arg1 : !llvm.ptr) varType(f32)
  llvm.return
}

// CHECK: %struct.ident_t = type { i32, i32, i32, i32, ptr }

// CHECK: @[[LOCSTR:.*]] = private unnamed_addr constant [{{[0-9]*}} x i8] c";{{.*}};testexitdataop;{{[0-9]*}};{{[0-9]*}};;\00"
// CHECK: @[[LOCGLOBAL:.*]] = private unnamed_addr constant %struct.ident_t { i32 0, i32 2, i32 0, i32 {{[0-9]*}}, ptr @[[LOCSTR]] }
// CHECK: @[[MAPNAME1:.*]] = private unnamed_addr constant [{{[0-9]*}} x i8] c";{{.*}};unknown;{{[0-9]*}};{{[0-9]*}};;\00"
// CHECK: @[[MAPNAME2:.*]] = private unnamed_addr constant [{{[0-9]*}} x i8] c";{{.*}};unknown;{{[0-9]*}};{{[0-9]*}};;\00"
// CHECK: @[[MAPTYPES:.*]] = private unnamed_addr constant [{{[0-9]*}} x i64] [i64 8, i64 2]
// CHECK: @[[MAPNAMES:.*]] = private constant [{{[0-9]*}} x ptr] [ptr @[[MAPNAME1]], ptr @[[MAPNAME2]]]

// CHECK: define void @testexitdataop(ptr %[[PTR0:.*]], ptr %[[PTR1:.*]])
// CHECK: %[[OFFLOAD_BASEPTRS:.*]] = alloca [{{[0-9]*}} x ptr]
// CHECK: %[[OFFLOAD_PTRS:.*]] = alloca [{{[0-9]*}} x ptr]
// CHECK: %[[OFFLOAS_SIZES:.*]] = alloca [{{[0-9]*}} x i64]

// CHECK: %[[OFFLOAD_BASEPTRS_GEP:.*]] = getelementptr inbounds [2 x ptr], ptr %[[OFFLOAD_BASEPTRS]], i32 0, i32 0
// CHECK: store ptr %[[PTR0]], ptr %[[OFFLOAD_BASEPTRS_GEP]]
// CHECK: %[[OFFLOAD_PTRS_GEP:.*]] = getelementptr inbounds [2 x ptr], ptr %[[OFFLOAD_PTRS]], i32 0, i32 0
// CHECK: store ptr %[[PTR0]], ptr %[[OFFLOAD_PTRS_GEP]]
// CHECK: %[[OFFLOAS_SIZES_GEP:.*]] = getelementptr inbounds [2 x i64], ptr %[[OFFLOAS_SIZES]], i32 0, i32 0
// CHECK: store i64 ptrtoint (ptr getelementptr (ptr, ptr null, i32 1) to i64), ptr %[[OFFLOAS_SIZES_GEP]]

// CHECK: %[[OFFLOAD_BASEPTRS_GEP:.*]] = getelementptr inbounds [2 x ptr], ptr %[[OFFLOAD_BASEPTRS]], i32 0, i32 1
// CHECK: store ptr %[[PTR1]], ptr %[[OFFLOAD_BASEPTRS_GEP]]
// CHECK: %[[OFFLOAD_PTRS_GEP:.*]] = getelementptr inbounds [2 x ptr], ptr %[[OFFLOAD_PTRS]], i32 0, i32 1
// CHECK: store ptr %[[PTR1]], ptr %[[OFFLOAD_PTRS_GEP]]
// CHECK: %[[OFFLOAS_SIZES_GEP:.*]] = getelementptr inbounds [2 x i64], ptr %[[OFFLOAS_SIZES]], i32 0, i32 1
// CHECK: store i64 ptrtoint (ptr getelementptr (ptr, ptr null, i32 1) to i64), ptr %[[OFFLOAS_SIZES_GEP]]

// CHECK: %[[OFFLOAD_BASEPTRS_GEP:.*]] = getelementptr inbounds [2 x ptr], ptr %[[OFFLOAD_BASEPTRS]], i32 0, i32 0
// CHECK: %[[OFFLOAD_PTRS_GEP:.*]] = getelementptr inbounds [2 x ptr], ptr %[[OFFLOAD_PTRS]], i32 0, i32 0
// CHECK: %[[OFFLOAS_SIZES_GEP:.*]] = getelementptr inbounds [2 x i64], ptr %[[OFFLOAS_SIZES]], i32 0, i32 0

// CHECK: call void @__tgt_target_data_end_mapper(ptr @[[LOCGLOBAL]], i64 -1, i32 2, ptr %[[OFFLOAD_BASEPTRS_GEP]], ptr %[[OFFLOAD_PTRS_GEP]], ptr %[[OFFLOAS_SIZES_GEP]], ptr @[[MAPTYPES]], ptr @[[MAPNAMES]], ptr null)

// CHECK: declare void @__tgt_target_data_end_mapper(ptr, i64, i32, ptr, ptr, ptr, ptr, ptr, ptr) #0

// -----

llvm.func @testupdateop(%arg1: !llvm.ptr) {
  %0 = acc.update_device varPtr(%arg1 : !llvm.ptr) varType(f32) -> !llvm.ptr
  acc.update dataOperands(%0 : !llvm.ptr)
  llvm.return
}

// CHECK: %struct.ident_t = type { i32, i32, i32, i32, ptr }

// CHECK: [[LOCSTR:@.*]] = private unnamed_addr constant [{{[0-9]*}} x i8] c";{{.*}};testupdateop;{{[0-9]*}};{{[0-9]*}};;\00", align 1
// CHECK: [[LOCGLOBAL:@.*]] = private unnamed_addr constant %struct.ident_t { i32 0, i32 2, i32 0, i32 {{[0-9]*}}, ptr [[LOCSTR]] }, align 8
// CHECK: [[MAPNAME1:@.*]] = private unnamed_addr constant [{{[0-9]*}} x i8] c";{{.*}};unknown;{{[0-9]*}};{{[0-9]*}};;\00", align 1
// CHECK: [[MAPTYPES:@.*]] = private unnamed_addr constant [{{[0-9]*}} x i64] [i64 1]
// CHECK: [[MAPNAMES:@.*]] = private constant [{{[0-9]*}} x ptr] [ptr [[MAPNAME1]]]

// CHECK: define void @testupdateop(ptr %[[SIMPLEPTR:.*]])
// CHECK: %[[OFFLOAD_BASEPTRS:.*]] = alloca [{{[0-9]*}} x ptr]
// CHECK: %[[OFFLOAD_PTRS:.*]] = alloca [{{[0-9]*}} x ptr]
// CHECK: %[[OFFLOAS_SIZES:.*]] = alloca [{{[0-9]*}} x i64]

// CHECK: %[[BASEGEP:.*]] = getelementptr inbounds [1 x ptr], ptr %[[OFFLOAD_BASEPTRS]], i32 0, i32 0
// CHECK: store ptr %[[SIMPLEPTR]], ptr %[[BASEGEP]]
// CHECK: %[[ARGGEP:.*]] = getelementptr inbounds [1 x ptr], ptr %[[OFFLOAD_PTRS]], i32 0, i32 0
// CHECK: store ptr %[[SIMPLEPTR]], ptr %[[ARGGEP]]
// CHECK: %[[SIZEGEP:.*]] = getelementptr inbounds [1 x i64], ptr %[[OFFLOAS_SIZES]], i32 0, i32 0
// CHECK: store i64 ptrtoint (ptr getelementptr (ptr, ptr null, i32 1) to i64), ptr %[[SIZEGEP]]

// CHECK: %[[OFFLOAD_BASEPTRS_GEP:.*]] = getelementptr inbounds [1 x ptr], ptr %[[OFFLOAD_BASEPTRS]], i32 0, i32 0
// CHECK: %[[OFFLOAD_PTRS_GEP:.*]] = getelementptr inbounds [1 x ptr], ptr %[[OFFLOAD_PTRS]], i32 0, i32 0
// CHECK: %[[OFFLOAS_SIZES_GEP:.*]] = getelementptr inbounds [1 x i64], ptr %[[OFFLOAS_SIZES]], i32 0, i32 0

// CHECK: call void @__tgt_target_data_update_mapper(ptr [[LOCGLOBAL]], i64 -1, i32 1, ptr %[[OFFLOAD_BASEPTRS_GEP]], ptr %[[OFFLOAD_PTRS_GEP]], ptr %[[OFFLOAS_SIZES_GEP]], ptr [[MAPTYPES]], ptr [[MAPNAMES]], ptr null)

// CHECK: declare void @__tgt_target_data_update_mapper(ptr, i64, i32, ptr, ptr, ptr, ptr, ptr, ptr) #0

// -----

llvm.func @testdataop(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: !llvm.ptr) {
  
  %0 = acc.copyin varPtr(%arg0 : !llvm.ptr) varType(f32) -> !llvm.ptr
  %1 = acc.create varPtr(%arg1 : !llvm.ptr) varType(f32) -> !llvm.ptr
  acc.data dataOperands(%0, %1 : !llvm.ptr, !llvm.ptr) {
    %9 = llvm.mlir.constant(2 : i32) : i32
    llvm.store %9, %arg2 : i32, !llvm.ptr
    acc.terminator
  }
  acc.copyout accPtr(%0 : !llvm.ptr) to varPtr(%arg0 : !llvm.ptr) varType(f32)
  acc.copyout accPtr(%1 : !llvm.ptr) to varPtr(%arg1 : !llvm.ptr) varType(f32)
  llvm.return
}

// CHECK: %struct.ident_t = type { i32, i32, i32, i32, ptr }
// CHECK: @[[LOCSTR:.*]] = private unnamed_addr constant [{{[0-9]*}} x i8] c";{{.*}};testdataop;{{[0-9]*}};{{[0-9]*}};;\00"
// CHECK: @[[LOCGLOBAL:.*]] = private unnamed_addr constant %struct.ident_t { i32 0, i32 2, i32 0, i32 {{[0-9]*}}, ptr @[[LOCSTR]] }
// CHECK: @[[MAPNAME1:.*]] = private unnamed_addr constant [{{[0-9]*}} x i8] c";{{.*}};unknown;{{[0-9]*}};{{[0-9]*}};;\00"
// CHECK: @[[MAPNAME2:.*]] = private unnamed_addr constant [{{[0-9]*}} x i8] c";{{.*}};unknown;{{[0-9]*}};{{[0-9]*}};;\00"
// CHECK: @[[MAPTYPES:.*]] = private unnamed_addr constant [{{[0-9]*}} x i64] [i64 8193, i64 8192]
// CHECK: @[[MAPNAMES:.*]] = private constant [{{[0-9]*}} x ptr] [ptr @[[MAPNAME1]], ptr @[[MAPNAME2]]]

// CHECK: define void @testdataop(ptr %[[PTR0:.*]], ptr %[[PTR1:.*]], ptr %[[PTR2:.*]])
// CHECK: %[[OFFLOAD_BASEPTRS:.*]] = alloca [{{[0-9]*}} x ptr], align 8
// CHECK: %[[OFFLOAD_PTRS:.*]] = alloca [{{[0-9]*}} x ptr], align 8
// CHECK: %[[OFFLOAS_SIZES:.*]] = alloca [{{[0-9]*}} x i64], align 8

// CHECK: %[[OFFLOAD_BASEPTRS_GEP:.*]] = getelementptr inbounds [2 x ptr], ptr %[[OFFLOAD_BASEPTRS]], i32 0, i32 0
// CHECK: store ptr %[[PTR0]], ptr %[[OFFLOAD_BASEPTRS_GEP]]
// CHECK: %[[OFFLOAD_PTRS_GEP:.*]] = getelementptr inbounds [2 x ptr], ptr %[[OFFLOAD_PTRS]], i32 0, i32 0
// CHECK: store ptr %[[PTR0]], ptr %[[OFFLOAD_PTRS_GEP]]
// CHECK: %[[OFFLOAS_SIZES_GEP:.*]] = getelementptr inbounds [2 x i64], ptr %[[OFFLOAS_SIZES]], i32 0, i32 0
// CHECK: store i64 ptrtoint (ptr getelementptr (ptr, ptr null, i32 1) to i64), ptr %[[OFFLOAS_SIZES_GEP]]

// CHECK: %[[OFFLOAD_BASEPTRS_GEP:.*]] = getelementptr inbounds [2 x ptr], ptr %[[OFFLOAD_BASEPTRS]], i32 0, i32 1
// CHECK: store ptr %[[PTR1]], ptr %[[OFFLOAD_BASEPTRS_GEP]]
// CHECK: %[[OFFLOAD_PTRS_GEP:.*]] = getelementptr inbounds [2 x ptr], ptr %[[OFFLOAD_PTRS]], i32 0, i32 1
// CHECK: store ptr %[[PTR1]], ptr %[[OFFLOAD_PTRS_GEP]]
// CHECK: %[[OFFLOAS_SIZES_GEP:.*]] = getelementptr inbounds [2 x i64], ptr %[[OFFLOAS_SIZES]], i32 0, i32 1
// CHECK: store i64 ptrtoint (ptr getelementptr (ptr, ptr null, i32 1) to i64), ptr %[[OFFLOAS_SIZES_GEP]]

// CHECK: %[[OFFLOAD_BASEPTRS_GEP:.*]] = getelementptr inbounds [2 x ptr], ptr %[[OFFLOAD_BASEPTRS]], i32 0, i32 0
// CHECK: %[[OFFLOAD_PTRS_GEP:.*]] = getelementptr inbounds [2 x ptr], ptr %[[OFFLOAD_PTRS]], i32 0, i32 0
// CHECK: %[[OFFLOAS_SIZES_GEP:.*]] = getelementptr inbounds [2 x i64], ptr %[[OFFLOAS_SIZES]], i32 0, i32 0
// CHECK: call void @__tgt_target_data_begin_mapper(ptr @[[LOCGLOBAL]], i64 -1, i32 2, ptr %[[OFFLOAD_BASEPTRS_GEP]], ptr %[[OFFLOAD_PTRS_GEP]], ptr %[[OFFLOAS_SIZES_GEP]], ptr @[[MAPTYPES]], ptr @[[MAPNAMES]], ptr null)
// CHECK: br label %acc.data

// CHECK:      acc.data:
// CHECK-NEXT:   store i32 2, ptr %{{.*}}
// CHECK-NEXT:   br label %acc.end_data

// CHECK: acc.end_data:
// CHECK:   %[[OFFLOAD_BASEPTRS_GEP:.*]] = getelementptr inbounds [2 x ptr], ptr %[[OFFLOAD_BASEPTRS]], i32 0, i32 0
// CHECK:   %[[OFFLOAD_PTRS_GEP:.*]] = getelementptr inbounds [2 x ptr], ptr %[[OFFLOAD_PTRS]], i32 0, i32 0
// CHECK:   %[[OFFLOAS_SIZES_GEP:.*]] = getelementptr inbounds [2 x i64], ptr %[[OFFLOAS_SIZES]], i32 0, i32 0
// CHECK:   call void @__tgt_target_data_end_mapper(ptr @[[LOCGLOBAL]], i64 -1, i32 2, ptr %[[OFFLOAD_BASEPTRS_GEP]], ptr %[[OFFLOAD_PTRS_GEP]], ptr %[[OFFLOAS_SIZES_GEP]], ptr @[[MAPTYPES]], ptr @[[MAPNAMES]], ptr null)

// CHECK: declare void @__tgt_target_data_begin_mapper(ptr, i64, i32, ptr, ptr, ptr, ptr, ptr, ptr)
// CHECK: declare void @__tgt_target_data_end_mapper(ptr, i64, i32, ptr, ptr, ptr, ptr, ptr, ptr)
