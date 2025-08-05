// RUN: mlir-translate -mlir-to-llvmir %s | FileCheck %s

// -----
// CHECK-LABEL: @simd_nontemporal
llvm.func @simd_nontemporal() {
  %0 = llvm.mlir.constant(10 : i64) : i64
  %1 = llvm.mlir.constant(1 : i64) : i64
  %2 = llvm.alloca %1 x i64 : (i64) -> !llvm.ptr
  %3 = llvm.alloca %1 x i64 : (i64) -> !llvm.ptr
  //CHECK: %[[A_ADDR:.*]] = alloca i64, i64 1, align 8
  //CHECK: %[[B_ADDR:.*]] = alloca i64, i64 1, align 8
  //CHECK: %[[B:.*]] = load i64, ptr %[[B_ADDR]], align 4, !nontemporal !1, !llvm.access.group !2
  //CHECK: store i64 %[[B]], ptr %[[A_ADDR]], align 4, !nontemporal !1, !llvm.access.group !2
  omp.simd nontemporal(%2, %3 : !llvm.ptr, !llvm.ptr) {
    omp.loop_nest (%arg0) : i64 = (%1) to (%0) inclusive step (%1) {
      %4 = llvm.load %3 {nontemporal}: !llvm.ptr -> i64
      llvm.store %4, %2 {nontemporal} : i64, !llvm.ptr
      omp.yield
    }
  }
  llvm.return
}

// -----

//CHECK-LABEL:  define void @_QPtest(ptr %0, ptr %1) {
llvm.func @_QPtest(%arg0: !llvm.ptr {fir.bindc_name = "n"}, %arg1: !llvm.ptr {fir.bindc_name = "a"}) {
    %0 = llvm.mlir.constant(1 : i32) : i32
    %1 = llvm.alloca %0 x !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)> {alignment = 8 : i64} : (i32) -> !llvm.ptr
    %2 = llvm.alloca %0 x !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)> {alignment = 8 : i64} : (i32) -> !llvm.ptr
    %3 = llvm.mlir.constant(1 : i64) : i64
    %4 = llvm.alloca %3 x i32 {bindc_name = "i", pinned} : (i64) -> !llvm.ptr
    %6 = llvm.load %arg0 : !llvm.ptr -> i32
    // CHECK:  %[[A_VAL1:.*]] = alloca { ptr, i64, i32, i8, i8, i8, i8, [1 x [3 x i64]] }, align 8
    // CHECK:  %[[A_VAL2:.*]] = alloca { ptr, i64, i32, i8, i8, i8, i8, [1 x [3 x i64]] }, align 8
    omp.simd nontemporal(%arg1 : !llvm.ptr) {
      omp.loop_nest (%arg2) : i32 = (%0) to (%6) inclusive step (%0) {
        llvm.store %arg2, %4 : i32, !llvm.ptr
        // CHECK:  call void @llvm.memcpy.p0.p0.i32(ptr %[[A_VAL2]], ptr %1, i32 48, i1 false)
        %7 = llvm.mlir.constant(48 : i32) : i32
        "llvm.intr.memcpy"(%2, %arg1, %7) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i32) -> ()
        %8 = llvm.load %4 : !llvm.ptr -> i32
        %9 = llvm.sext %8 : i32 to i64
        %10 = llvm.getelementptr %2[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)>
        %11 = llvm.load %10 : !llvm.ptr -> !llvm.ptr
        %12 = llvm.mlir.constant(0 : index) : i64
        %13 = llvm.getelementptr %2[0, 7, %12, 0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)>
        %14 = llvm.load %13 : !llvm.ptr -> i64
        %15 = llvm.getelementptr %2[0, 7, %12, 1] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)>
        %16 = llvm.load %15 : !llvm.ptr -> i64
        %17 = llvm.getelementptr %2[0, 7, %12, 2] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)>
        %18 = llvm.load %17 : !llvm.ptr -> i64
        %19 = llvm.mlir.constant(0 : i64) : i64
        %20 = llvm.sub %9, %14 overflow<nsw> : i64
        %21 = llvm.mul %20, %3 overflow<nsw> : i64
        %22 = llvm.mul %21, %3 overflow<nsw> : i64
        %23 = llvm.add %22,%19 overflow<nsw> : i64
        %24 = llvm.mul %3, %16 overflow<nsw> : i64
        // CHECK:  %[[VAL1:.*]] = getelementptr float, ptr {{.*}}, i64 %{{.*}}
        // CHECK:  %[[LOAD_A:.*]] = load float, ptr %[[VAL1]], align 4, !nontemporal 
        // CHECK:  %[[RES:.*]] = fadd contract float %[[LOAD_A]], 2.000000e+01
        %25 = llvm.getelementptr %11[%23] : (!llvm.ptr, i64) -> !llvm.ptr, f32
        %26 = llvm.load %25 {nontemporal} : !llvm.ptr -> f32
        %27 = llvm.mlir.constant(2.000000e+01 : f32) : f32
        %28 = llvm.fadd %26, %27 {fastmathFlags = #llvm.fastmath<contract>} : f32
        // CHECK:  call void @llvm.memcpy.p0.p0.i32(ptr %[[A_VAL1]], ptr %1, i32 48, i1 false)
        %29 = llvm.mlir.constant(48 : i32) : i32
        "llvm.intr.memcpy"(%1, %arg1, %29) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i32) -> ()
        %30 = llvm.load %4 : !llvm.ptr -> i32
        %31 = llvm.sext %30 : i32 to i64
        %32 = llvm.getelementptr %1[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)>
        %33 = llvm.load %32 : !llvm.ptr -> !llvm.ptr
        %34 = llvm.mlir.constant(0 : index) : i64
        %35 = llvm.getelementptr %1[0, 7, %34, 0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)>
        %36 = llvm.load %35 : !llvm.ptr -> i64
        %37 = llvm.getelementptr %1[0, 7, %34, 1] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)>
        %38 = llvm.load %37 : !llvm.ptr -> i64
        %39 = llvm.getelementptr %1[0, 7, %34, 2] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)>
        %40 = llvm.load %39 : !llvm.ptr -> i64
        %41 = llvm.sub %31, %36 overflow<nsw> : i64
        %42 = llvm.mul %41, %3 overflow<nsw> : i64
        %43 = llvm.mul %42, %3 overflow<nsw> : i64
        %44 = llvm.add %43,%19 overflow<nsw> : i64
        %45 = llvm.mul %3, %38 overflow<nsw> : i64
        // CHECK:  %[[VAL2:.*]] = getelementptr float, ptr %{{.*}}, i64 %{{.*}}
        // CHECK:  store float %[[RES]], ptr %[[VAL2]], align 4, !nontemporal 
        %46 = llvm.getelementptr %33[%44] : (!llvm.ptr, i64) -> !llvm.ptr, f32
        llvm.store %28, %46 {nontemporal} : f32, !llvm.ptr
        omp.yield
      }
    }
    llvm.return
  }

// -----

