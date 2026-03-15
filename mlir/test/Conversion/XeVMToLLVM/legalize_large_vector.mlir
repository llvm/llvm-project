// RUN: mlir-opt --convert-xevm-to-llvm --split-input-file %s | FileCheck %s

module @test_illegal_vector {
  // CHECK-LABEL: llvm.func @test_illegal_vector
  // CHECK: %[[ARG0:.*]]: !llvm.ptr, %[[ARG1:.*]]: !llvm.ptr, %[[ARG2:.*]]: !llvm.ptr, %[[ARG3:.*]]: !llvm.ptr, %[[ARG4:.*]]: !llvm.ptr
  llvm.func @test_illegal_vector(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: !llvm.ptr, %arg3: !llvm.ptr, %arg4: !llvm.ptr) {
    // CHECK: %[[LOAD0:.*]] = llvm.load %[[ARG0]] : !llvm.ptr -> vector<8xi16>
    // CHECK: %[[BITCAST0:.*]] = llvm.bitcast %[[LOAD0]] : vector<8xi16> to vector<8xf16>
    // CHECK: %[[FPEXT0:.*]] = llvm.fpext %[[BITCAST0]] : vector<8xf16> to vector<8xf32>
    // CHECK: %[[GEP0:.*]] = llvm.getelementptr %[[ARG0]][8] : (!llvm.ptr) -> !llvm.ptr, i16
    // CHECK: %[[LOAD1:.*]] = llvm.load %[[GEP0]] : !llvm.ptr -> vector<8xi16>
    // CHECK: %[[BITCAST1:.*]] = llvm.bitcast %[[LOAD1]] : vector<8xi16> to vector<8xf16>
    // CHECK: %[[FPEXT1:.*]] = llvm.fpext %[[BITCAST1]] : vector<8xf16> to vector<8xf32>
    // CHECK: %[[BITCAST2:.*]] = llvm.bitcast %[[FPEXT0]] : vector<8xf32> to vector<8xi32>
    // CHECK: llvm.store %[[BITCAST2]], %[[ARG1]] : vector<8xi32>, !llvm.ptr
    // CHECK: %[[BITCAST3:.*]] = llvm.bitcast %[[FPEXT1]] : vector<8xf32> to vector<8xi32>
    // CHECK: llvm.store %[[BITCAST3]], %[[ARG2]] : vector<8xi32>, !llvm.ptr
    // CHECK: %[[GEP1:.*]] = llvm.getelementptr %[[ARG0]][16] : (!llvm.ptr) -> !llvm.ptr, i16
    // CHECK: %[[LOAD2:.*]] = llvm.load %[[GEP1]] : !llvm.ptr -> vector<8xi16>
    // CHECK: %[[BITCAST4:.*]] = llvm.bitcast %[[LOAD2]] : vector<8xi16> to vector<8xf16>
    // CHECK: %[[FPEXT2:.*]] = llvm.fpext %[[BITCAST4]] : vector<8xf16> to vector<8xf32>
    // CHECK: %[[GEP2:.*]] = llvm.getelementptr %[[ARG0]][24] : (!llvm.ptr) -> !llvm.ptr, i16
    // CHECK: %[[LOAD3:.*]] = llvm.load %[[GEP2]] : !llvm.ptr -> vector<8xi16>
    // CHECK: %[[BITCAST5:.*]] = llvm.bitcast %[[LOAD3]] : vector<8xi16> to vector<8xf16>
    // CHECK: %[[FPEXT3:.*]] = llvm.fpext %[[BITCAST5]] : vector<8xf16> to vector<8xf32>
    // CHECK: %[[BITCAST6:.*]] = llvm.bitcast %[[FPEXT2]] : vector<8xf32> to vector<8xi32>
    // CHECK: llvm.store %[[BITCAST6]], %[[ARG3]] : vector<8xi32>, !llvm.ptr
    // CHECK: %[[BITCAST7:.*]] = llvm.bitcast %[[FPEXT3]] : vector<8xf32> to vector<8xi32>
    // CHECK: llvm.store %[[BITCAST7]], %[[ARG4]] : vector<8xi32>, !llvm.ptr
    // CHECK: llvm.return
      %0 = llvm.load %arg0 : !llvm.ptr -> vector<32xi16>
      %1 = llvm.bitcast %0 : vector<32xi16> to vector<32xf16>
      %2 = llvm.shufflevector %1, %1 [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15] : vector<32xf16>
      %3 = llvm.shufflevector %1, %1 [16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31] : vector<32xf16>
      %4 = llvm.fpext %2 : vector<16xf16> to vector<16xf32>
      %5 = llvm.fpext %3 : vector<16xf16> to vector<16xf32>
      %6 = llvm.shufflevector %4, %4 [0, 1, 2, 3, 4, 5, 6, 7] : vector<16xf32>
      %7 = llvm.shufflevector %4, %4 [8, 9, 10, 11, 12, 13, 14, 15] : vector<16xf32>
      %8 = llvm.bitcast %6 : vector<8xf32> to vector<8xi32>
      llvm.store %8, %arg1 : vector<8xi32>, !llvm.ptr
      %9 = llvm.bitcast %7 : vector<8xf32> to vector<8xi32>
      llvm.store %9, %arg2 : vector<8xi32>, !llvm.ptr
      %10 = llvm.shufflevector %5, %5 [0, 1, 2, 3, 4, 5, 6, 7] : vector<16xf32>
      %11 = llvm.shufflevector %5, %5 [8, 9, 10, 11, 12, 13, 14, 15] : vector<16xf32>
      %12 = llvm.bitcast %10 : vector<8xf32> to vector<8xi32>
      llvm.store %12, %arg3 : vector<8xi32>, !llvm.ptr
      %13 = llvm.bitcast %11 : vector<8xf32> to vector<8xi32>
      llvm.store %13, %arg4 : vector<8xi32>, !llvm.ptr
      llvm.return
  }
}
