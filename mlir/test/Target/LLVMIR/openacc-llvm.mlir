// RUN: mlir-translate -mlir-to-llvmir -split-input-file %s | FileCheck %s

llvm.func @testenterdataop(%arg0: !llvm.ptr<f32>, %arg1: !llvm.ptr<f32>, %arg2: i64, %arg3: i64, %arg4: i64, %arg5: !llvm.ptr<f32>) {
  %0 = llvm.mlir.undef : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
  %1 = llvm.insertvalue %arg0, %0[0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
  %2 = llvm.insertvalue %arg1, %1[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
  %3 = llvm.insertvalue %arg2, %2[2] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
  %4 = llvm.insertvalue %arg3, %3[3, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
  %5 = llvm.insertvalue %arg4, %4[4, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
  %6 = llvm.mlir.constant(10 : index) : i64
  %7 = llvm.mlir.constant(1 : index) : i64
  %8 = llvm.mlir.null : !llvm.ptr<f32>
  %9 = llvm.getelementptr %8[%6] : (!llvm.ptr<f32>, i64) -> !llvm.ptr<f32>
  %10 = llvm.ptrtoint %9 : !llvm.ptr<f32> to i64
  %11 = llvm.extractvalue %5[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
  %12 = llvm.mlir.undef : !llvm.struct<"openacc_data", (struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>, ptr<f32>, i64)>
  %13 = llvm.insertvalue %5, %12[0] : !llvm.struct<"openacc_data", (struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>, ptr<f32>, i64)>
  %14 = llvm.insertvalue %11, %13[1] : !llvm.struct<"openacc_data", (struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>, ptr<f32>, i64)>
  %15 = llvm.insertvalue %10, %14[2] : !llvm.struct<"openacc_data", (struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>, ptr<f32>, i64)>
  acc.enter_data copyin(%arg5 : !llvm.ptr<f32>) create(%15 : !llvm.struct<"openacc_data", (struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>, ptr<f32>, i64)>)
  llvm.return
}

// CHECK: %struct.ident_t = type { i32, i32, i32, i32, i8* }

// CHECK: [[LOCSTR:@.*]] = private unnamed_addr constant [{{[0-9]*}} x i8] c";{{.*}};testenterdataop;{{[0-9]*}};{{[0-9]*}};;\00", align 1
// CHECK: [[LOCGLOBAL:@.*]] = private unnamed_addr constant %struct.ident_t { i32 0, i32 2, i32 0, i32 0, i8* getelementptr inbounds ([{{[0-9]*}} x i8], [{{[0-9]*}} x i8]* [[LOCSTR]], i32 0, i32 0) }, align 8
// CHECK: [[MAPNAME1:@.*]] = private unnamed_addr constant [{{[0-9]*}} x i8] c";{{.*}};unknown;{{[0-9]*}};{{[0-9]*}};;\00", align 1
// CHECK: [[MAPNAME2:@.*]] = private unnamed_addr constant [{{[0-9]*}} x i8] c";{{.*}};unknown;{{[0-9]*}};{{[0-9]*}};;\00", align 1
// CHECK: [[MAPTYPES:@.*]] = private unnamed_addr constant [{{[0-9]*}} x i64] [i64 0, i64 1]
// CHECK: [[MAPNAMES:@.*]] = private constant [{{[0-9]*}} x i8*] [i8* getelementptr inbounds ([{{[0-9]*}} x i8], [{{[0-9]*}} x i8]* [[MAPNAME1]], i32 0, i32 0), i8* getelementptr inbounds ([{{[0-9]*}} x i8], [{{[0-9]*}} x i8]* [[MAPNAME2]], i32 0, i32 0)]

// CHECK: define void @testenterdataop(float* %{{.*}}, float* %{{.*}}, i64 %{{.*}}, i64 %{{.*}}, i64 %{{.*}}, float* [[SIMPLEPTR:%.*]])
// CHECK: [[ARGBASE_ALLOCA:%.*]] = alloca [{{[0-9]*}} x i8*], align 8
// CHECK: [[ARG_ALLOCA:%.*]] = alloca [{{[0-9]*}} x i8*], align 8
// CHECK: [[SIZE_ALLOCA:%.*]] = alloca [{{[0-9]*}} x i64], align 8

// CHECK: [[ARGBASE:%.*]] = extractvalue %openacc_data %{{.*}}, 0
// CHECK: [[ARG:%.*]] = extractvalue %openacc_data %{{.*}}, 1
// CHECK: [[ARGSIZE:%.*]] = extractvalue %openacc_data %{{.*}}, 2
// CHECK: [[ARGBASEGEP:%.*]] = getelementptr inbounds [2 x i8*], [2 x i8*]* [[ARGBASE_ALLOCA]], i32 0, i32 0
// CHECK: [[ARGBASEGEPCAST:%.*]] = bitcast i8** [[ARGBASEGEP]] to { float*, float*, i64, [1 x i64], [1 x i64] }*
// CHECK: store { float*, float*, i64, [1 x i64], [1 x i64] } [[ARGBASE]], { float*, float*, i64, [1 x i64], [1 x i64] }* [[ARGBASEGEPCAST]], align 8
// CHECK: [[ARGGEP:%.*]] = getelementptr inbounds [2 x i8*], [2 x i8*]* [[ARG_ALLOCA]], i32 0, i32 0
// CHECK: [[ARGGEPCAST:%.*]] = bitcast i8** [[ARGGEP]] to float**
// CHECK: store float* [[ARG]], float** [[ARGGEPCAST]], align 8
// CHECK: [[SIZEGEP:%.*]] = getelementptr inbounds [2 x i64], [2 x i64]* [[SIZE_ALLOCA]], i32 0, i32 0
// CHECK: store i64 [[ARGSIZE]], i64* [[SIZEGEP]], align 4

// CHECK: [[ARGBASEGEP:%.*]] = getelementptr inbounds [2 x i8*], [2 x i8*]* [[ARGBASE_ALLOCA]], i32 0, i32 1
// CHECK: [[ARGBASEGEPCAST:%.*]] = bitcast i8** [[ARGBASEGEP]] to float**
// CHECK: store float* [[SIMPLEPTR]], float** [[ARGBASEGEPCAST]], align 8
// CHECK: [[ARGGEP:%.*]] = getelementptr inbounds [2 x i8*], [2 x i8*]* [[ARG_ALLOCA]], i32 0, i32 1
// CHECK: [[ARGGEPCAST:%.*]] = bitcast i8** [[ARGGEP]] to float**
// CHECK: store float* [[SIMPLEPTR]], float** [[ARGGEPCAST]], align 8
// CHECK: [[SIZEGEP:%.*]] = getelementptr inbounds [2 x i64], [2 x i64]* [[SIZE_ALLOCA]], i32 0, i32 1
// CHECK: store i64 ptrtoint (i1** getelementptr (i1*, i1** null, i32 1) to i64), i64* [[SIZEGEP]], align 4

// CHECK: [[ARGBASE_ALLOCA_GEP:%.*]] = getelementptr inbounds [2 x i8*], [2 x i8*]* [[ARGBASE_ALLOCA]], i32 0, i32 0
// CHECK: [[ARG_ALLOCA_GEP:%.*]] = getelementptr inbounds [2 x i8*], [2 x i8*]* [[ARG_ALLOCA]], i32 0, i32 0
// CHECK: [[SIZE_ALLOCA_GEP:%.*]] = getelementptr inbounds [2 x i64], [2 x i64]* [[SIZE_ALLOCA]], i32 0, i32 0

// CHECK: call void @__tgt_target_data_begin_mapper(%struct.ident_t* [[LOCGLOBAL]], i64 -1, i32 2, i8** [[ARGBASE_ALLOCA_GEP]], i8** [[ARG_ALLOCA_GEP]], i64* [[SIZE_ALLOCA_GEP]], i64* getelementptr inbounds ([{{[0-9]*}} x i64], [{{[0-9]*}} x i64]* [[MAPTYPES]], i32 0, i32 0), i8** getelementptr inbounds ([{{[0-9]*}} x i8*], [{{[0-9]*}} x i8*]* [[MAPNAMES]], i32 0, i32 0), i8** null)

// CHECK: declare void @__tgt_target_data_begin_mapper(%struct.ident_t*, i64, i32, i8**, i8**, i64*, i64*, i8**, i8**) #0

// -----


llvm.func @testexitdataop(%arg0: !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>, %arg1: !llvm.ptr<f32>) {
  %0 = llvm.mlir.constant(10 : index) : i64
  %1 = llvm.mlir.null : !llvm.ptr<f32>
  %2 = llvm.getelementptr %1[%0] : (!llvm.ptr<f32>, i64) -> !llvm.ptr<f32>
  %3 = llvm.ptrtoint %2 : !llvm.ptr<f32> to i64
  %4 = llvm.extractvalue %arg0[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
  %5 = llvm.mlir.undef : !llvm.struct<"openacc_data", (struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>, ptr<f32>, i64)>
  %6 = llvm.insertvalue %arg0, %5[0] : !llvm.struct<"openacc_data", (struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>, ptr<f32>, i64)>
  %7 = llvm.insertvalue %4, %6[1] : !llvm.struct<"openacc_data", (struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>, ptr<f32>, i64)>
  %8 = llvm.insertvalue %3, %7[2] : !llvm.struct<"openacc_data", (struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>, ptr<f32>, i64)>
  acc.exit_data copyout(%arg1 : !llvm.ptr<f32>) delete(%8 : !llvm.struct<"openacc_data", (struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>, ptr<f32>, i64)>)
  llvm.return
}

// CHECK: %struct.ident_t = type { i32, i32, i32, i32, i8* }

// CHECK: [[LOCSTR:@.*]] = private unnamed_addr constant [{{[0-9]*}} x i8] c";{{.*}};testexitdataop;{{[0-9]*}};{{[0-9]*}};;\00", align 1
// CHECK: [[LOCGLOBAL:@.*]] = private unnamed_addr constant %struct.ident_t { i32 0, i32 2, i32 0, i32 0, i8* getelementptr inbounds ([{{[0-9]*}} x i8], [{{[0-9]*}} x i8]* [[LOCSTR]], i32 0, i32 0) }, align 8
// CHECK: [[MAPNAME1:@.*]] = private unnamed_addr constant [{{[0-9]*}} x i8] c";{{.*}};unknown;{{[0-9]*}};{{[0-9]*}};;\00", align 1
// CHECK: [[MAPNAME2:@.*]] = private unnamed_addr constant [{{[0-9]*}} x i8] c";{{.*}};unknown;{{[0-9]*}};{{[0-9]*}};;\00", align 1
// CHECK: [[MAPTYPES:@.*]] = private unnamed_addr constant [{{[0-9]*}} x i64] [i64 8, i64 2]
// CHECK: [[MAPNAMES:@.*]] = private constant [{{[0-9]*}} x i8*] [i8* getelementptr inbounds ([{{[0-9]*}} x i8], [{{[0-9]*}} x i8]* [[MAPNAME1]], i32 0, i32 0), i8* getelementptr inbounds ([{{[0-9]*}} x i8], [{{[0-9]*}} x i8]* [[MAPNAME2]], i32 0, i32 0)]

// CHECK: define void @testexitdataop({ float*, float*, i64, [1 x i64], [1 x i64] } %{{.*}}, float* [[SIMPLEPTR:%.*]])
// CHECK: [[ARGBASE_ALLOCA:%.*]] = alloca [{{[0-9]*}} x i8*], align 8
// CHECK: [[ARG_ALLOCA:%.*]] = alloca [{{[0-9]*}} x i8*], align 8
// CHECK: [[SIZE_ALLOCA:%.*]] = alloca [{{[0-9]*}} x i64], align 8

// CHECK: [[ARGBASE:%.*]] = extractvalue %openacc_data %{{.*}}, 0
// CHECK: [[ARG:%.*]] = extractvalue %openacc_data %{{.*}}, 1
// CHECK: [[ARGSIZE:%.*]] = extractvalue %openacc_data %{{.*}}, 2
// CHECK: [[ARGBASEGEP:%.*]] = getelementptr inbounds [2 x i8*], [2 x i8*]* [[ARGBASE_ALLOCA]], i32 0, i32 0
// CHECK: [[ARGBASEGEPCAST:%.*]] = bitcast i8** [[ARGBASEGEP]] to { float*, float*, i64, [1 x i64], [1 x i64] }*
// CHECK: store { float*, float*, i64, [1 x i64], [1 x i64] } [[ARGBASE]], { float*, float*, i64, [1 x i64], [1 x i64] }* [[ARGBASEGEPCAST]], align 8
// CHECK: [[ARGGEP:%.*]] = getelementptr inbounds [2 x i8*], [2 x i8*]* [[ARG_ALLOCA]], i32 0, i32 0
// CHECK: [[ARGGEPCAST:%.*]] = bitcast i8** [[ARGGEP]] to float**
// CHECK: store float* [[ARG]], float** [[ARGGEPCAST]], align 8
// CHECK: [[SIZEGEP:%.*]] = getelementptr inbounds [2 x i64], [2 x i64]* [[SIZE_ALLOCA]], i32 0, i32 0
// CHECK: store i64 [[ARGSIZE]], i64* [[SIZEGEP]], align 4

// CHECK: [[ARGBASEGEP:%.*]] = getelementptr inbounds [2 x i8*], [2 x i8*]* [[ARGBASE_ALLOCA]], i32 0, i32 1
// CHECK: [[ARGBASEGEPCAST:%.*]] = bitcast i8** [[ARGBASEGEP]] to float**
// CHECK: store float* [[SIMPLEPTR]], float** [[ARGBASEGEPCAST]], align 8
// CHECK: [[ARGGEP:%.*]] = getelementptr inbounds [2 x i8*], [2 x i8*]* [[ARG_ALLOCA]], i32 0, i32 1
// CHECK: [[ARGGEPCAST:%.*]] = bitcast i8** [[ARGGEP]] to float**
// CHECK: store float* [[SIMPLEPTR]], float** [[ARGGEPCAST]], align 8
// CHECK: [[SIZEGEP:%.*]] = getelementptr inbounds [2 x i64], [2 x i64]* [[SIZE_ALLOCA]], i32 0, i32 1
// CHECK: store i64 ptrtoint (i1** getelementptr (i1*, i1** null, i32 1) to i64), i64* [[SIZEGEP]], align 4

// CHECK: [[ARGBASE_ALLOCA_GEP:%.*]] = getelementptr inbounds [2 x i8*], [2 x i8*]* [[ARGBASE_ALLOCA]], i32 0, i32 0
// CHECK: [[ARG_ALLOCA_GEP:%.*]] = getelementptr inbounds [2 x i8*], [2 x i8*]* [[ARG_ALLOCA]], i32 0, i32 0
// CHECK: [[SIZE_ALLOCA_GEP:%.*]] = getelementptr inbounds [2 x i64], [2 x i64]* [[SIZE_ALLOCA]], i32 0, i32 0

// CHECK: call void @__tgt_target_data_end_mapper(%struct.ident_t* [[LOCGLOBAL]], i64 -1, i32 2, i8** [[ARGBASE_ALLOCA_GEP]], i8** [[ARG_ALLOCA_GEP]], i64* [[SIZE_ALLOCA_GEP]], i64* getelementptr inbounds ([{{[0-9]*}} x i64], [{{[0-9]*}} x i64]* [[MAPTYPES]], i32 0, i32 0), i8** getelementptr inbounds ([{{[0-9]*}} x i8*], [{{[0-9]*}} x i8*]* [[MAPNAMES]], i32 0, i32 0), i8** null)

// CHECK: declare void @__tgt_target_data_end_mapper(%struct.ident_t*, i64, i32, i8**, i8**, i64*, i64*, i8**, i8**) #0

// -----

llvm.func @testupdateop(%arg0: !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>, %arg1: !llvm.ptr<f32>) {
  %0 = llvm.mlir.constant(10 : index) : i64
  %1 = llvm.mlir.null : !llvm.ptr<f32>
  %2 = llvm.getelementptr %1[%0] : (!llvm.ptr<f32>, i64) -> !llvm.ptr<f32>
  %3 = llvm.ptrtoint %2 : !llvm.ptr<f32> to i64
  %4 = llvm.extractvalue %arg0[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
  %5 = llvm.mlir.undef : !llvm.struct<"openacc_data", (struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>, ptr<f32>, i64)>
  %6 = llvm.insertvalue %arg0, %5[0] : !llvm.struct<"openacc_data", (struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>, ptr<f32>, i64)>
  %7 = llvm.insertvalue %4, %6[1] : !llvm.struct<"openacc_data", (struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>, ptr<f32>, i64)>
  %8 = llvm.insertvalue %3, %7[2] : !llvm.struct<"openacc_data", (struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>, ptr<f32>, i64)>
  acc.update host(%8 : !llvm.struct<"openacc_data", (struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>, ptr<f32>, i64)>) device(%arg1 : !llvm.ptr<f32>)
  llvm.return
}

// CHECK: %struct.ident_t = type { i32, i32, i32, i32, i8* }

// CHECK: [[LOCSTR:@.*]] = private unnamed_addr constant [{{[0-9]*}} x i8] c";{{.*}};testupdateop;{{[0-9]*}};{{[0-9]*}};;\00", align 1
// CHECK: [[LOCGLOBAL:@.*]] = private unnamed_addr constant %struct.ident_t { i32 0, i32 2, i32 0, i32 0, i8* getelementptr inbounds ([{{[0-9]*}} x i8], [{{[0-9]*}} x i8]* [[LOCSTR]], i32 0, i32 0) }, align 8
// CHECK: [[MAPNAME1:@.*]] = private unnamed_addr constant [{{[0-9]*}} x i8] c";{{.*}};unknown;{{[0-9]*}};{{[0-9]*}};;\00", align 1
// CHECK: [[MAPNAME2:@.*]] = private unnamed_addr constant [{{[0-9]*}} x i8] c";{{.*}};unknown;{{[0-9]*}};{{[0-9]*}};;\00", align 1
// CHECK: [[MAPTYPES:@.*]] = private unnamed_addr constant [{{[0-9]*}} x i64] [i64 2, i64 1]
// CHECK: [[MAPNAMES:@.*]] = private constant [{{[0-9]*}} x i8*] [i8* getelementptr inbounds ([{{[0-9]*}} x i8], [{{[0-9]*}} x i8]* [[MAPNAME1]], i32 0, i32 0), i8* getelementptr inbounds ([{{[0-9]*}} x i8], [{{[0-9]*}} x i8]* [[MAPNAME2]], i32 0, i32 0)]

// CHECK: define void @testupdateop({ float*, float*, i64, [1 x i64], [1 x i64] } %{{.*}}, float* [[SIMPLEPTR:%.*]])
// CHECK: [[ARGBASE_ALLOCA:%.*]] = alloca [{{[0-9]*}} x i8*], align 8
// CHECK: [[ARG_ALLOCA:%.*]] = alloca [{{[0-9]*}} x i8*], align 8
// CHECK: [[SIZE_ALLOCA:%.*]] = alloca [{{[0-9]*}} x i64], align 8

// CHECK: [[ARGBASE:%.*]] = extractvalue %openacc_data %{{.*}}, 0
// CHECK: [[ARG:%.*]] = extractvalue %openacc_data %{{.*}}, 1
// CHECK: [[ARGSIZE:%.*]] = extractvalue %openacc_data %{{.*}}, 2
// CHECK: [[ARGBASEGEP:%.*]] = getelementptr inbounds [2 x i8*], [2 x i8*]* [[ARGBASE_ALLOCA]], i32 0, i32 0
// CHECK: [[ARGBASEGEPCAST:%.*]] = bitcast i8** [[ARGBASEGEP]] to { float*, float*, i64, [1 x i64], [1 x i64] }*
// CHECK: store { float*, float*, i64, [1 x i64], [1 x i64] } [[ARGBASE]], { float*, float*, i64, [1 x i64], [1 x i64] }* [[ARGBASEGEPCAST]], align 8
// CHECK: [[ARGGEP:%.*]] = getelementptr inbounds [2 x i8*], [2 x i8*]* [[ARG_ALLOCA]], i32 0, i32 0
// CHECK: [[ARGGEPCAST:%.*]] = bitcast i8** [[ARGGEP]] to float**
// CHECK: store float* [[ARG]], float** [[ARGGEPCAST]], align 8
// CHECK: [[SIZEGEP:%.*]] = getelementptr inbounds [2 x i64], [2 x i64]* [[SIZE_ALLOCA]], i32 0, i32 0
// CHECK: store i64 [[ARGSIZE]], i64* [[SIZEGEP]], align 4

// CHECK: [[ARGBASEGEP:%.*]] = getelementptr inbounds [2 x i8*], [2 x i8*]* [[ARGBASE_ALLOCA]], i32 0, i32 1
// CHECK: [[ARGBASEGEPCAST:%.*]] = bitcast i8** [[ARGBASEGEP]] to float**
// CHECK: store float* [[SIMPLEPTR]], float** [[ARGBASEGEPCAST]], align 8
// CHECK: [[ARGGEP:%.*]] = getelementptr inbounds [2 x i8*], [2 x i8*]* [[ARG_ALLOCA]], i32 0, i32 1
// CHECK: [[ARGGEPCAST:%.*]] = bitcast i8** [[ARGGEP]] to float**
// CHECK: store float* [[SIMPLEPTR]], float** [[ARGGEPCAST]], align 8
// CHECK: [[SIZEGEP:%.*]] = getelementptr inbounds [2 x i64], [2 x i64]* [[SIZE_ALLOCA]], i32 0, i32 1
// CHECK: store i64 ptrtoint (i1** getelementptr (i1*, i1** null, i32 1) to i64), i64* [[SIZEGEP]], align 4

// CHECK: [[ARGBASE_ALLOCA_GEP:%.*]] = getelementptr inbounds [2 x i8*], [2 x i8*]* [[ARGBASE_ALLOCA]], i32 0, i32 0
// CHECK: [[ARG_ALLOCA_GEP:%.*]] = getelementptr inbounds [2 x i8*], [2 x i8*]* [[ARG_ALLOCA]], i32 0, i32 0
// CHECK: [[SIZE_ALLOCA_GEP:%.*]] = getelementptr inbounds [2 x i64], [2 x i64]* [[SIZE_ALLOCA]], i32 0, i32 0

// CHECK: call void @__tgt_target_data_update_mapper(%struct.ident_t* [[LOCGLOBAL]], i64 -1, i32 2, i8** [[ARGBASE_ALLOCA_GEP]], i8** [[ARG_ALLOCA_GEP]], i64* [[SIZE_ALLOCA_GEP]], i64* getelementptr inbounds ([{{[0-9]*}} x i64], [{{[0-9]*}} x i64]* [[MAPTYPES]], i32 0, i32 0), i8** getelementptr inbounds ([{{[0-9]*}} x i8*], [{{[0-9]*}} x i8*]* [[MAPNAMES]], i32 0, i32 0), i8** null)

// CHECK: declare void @__tgt_target_data_update_mapper(%struct.ident_t*, i64, i32, i8**, i8**, i64*, i64*, i8**, i8**) #0
