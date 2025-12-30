// RUN: mlir-translate -mlir-to-llvmir %s | FileCheck %s

module attributes {omp.is_target_device = false, omp.target_triples = ["amdgcn-amd-amdhsa"]} {
  llvm.func @_QQmain() attributes {fir.bindc_name = "main"} {
    %0 = llvm.mlir.addressof @_QFEi : !llvm.ptr
    %1 = llvm.mlir.addressof @_QFEsp : !llvm.ptr
    %2 = omp.map.info var_ptr(%1 : !llvm.ptr, i32) map_clauses(tofrom) capture(ByRef) -> !llvm.ptr {name = "sp"}
    %3 = omp.map.info var_ptr(%0 : !llvm.ptr, i32) map_clauses(to) capture(ByCopy) -> !llvm.ptr {name = "i"}
    omp.target map_entries(%2 -> %arg0, %3 -> %arg1 : !llvm.ptr, !llvm.ptr) {
      %4 = llvm.load %arg1 : !llvm.ptr -> i32
      llvm.store %4, %arg0 : i32, !llvm.ptr
      omp.terminator
    }
    llvm.return
  }
  llvm.mlir.global internal @_QFEi() {addr_space = 0 : i32} : i32 {
    %0 = llvm.mlir.constant(1 : i32) : i32
    llvm.return %0 : i32
  }
  llvm.mlir.global internal @_QFEsp() {addr_space = 0 : i32} : i32 {
    %0 = llvm.mlir.constant(0 : i32) : i32
    llvm.return %0 : i32
  }
}

// CHECK: define void @_QQmain() {
// CHECK: %[[BYCOPY_ALLOCA:.*]] = alloca ptr, align 8

// CHECK: entry:                                            ; preds = %0
// CHECK: %[[LOAD_VAL:.*]] = load i32, ptr @_QFEi, align 4
// CHECK: store i32 %[[LOAD_VAL]], ptr %[[BYCOPY_ALLOCA]], align 4
// CHECK: %[[BYCOPY_LOAD:.*]] = load ptr, ptr %[[BYCOPY_ALLOCA]], align 8

// CHECK: %[[BASEPTR_BYREF:.*]] = getelementptr inbounds [2 x ptr], ptr %.offload_baseptrs, i32 0, i32 0
// CHECK: store ptr @_QFEsp, ptr %[[BASEPTR_BYREF]], align 8
// CHECK: %[[OFFLOADPTR_BYREF:.*]] = getelementptr inbounds [2 x ptr], ptr %.offload_ptrs, i32 0, i32 0
// CHECK: store ptr @_QFEsp, ptr %[[OFFLOADPTR_BYREF]], align 8

// CHECK: %[[BASEPTR_BYCOPY:.*]] = getelementptr inbounds [2 x ptr], ptr %.offload_baseptrs, i32 0, i32 1
// CHECK: store ptr %[[BYCOPY_LOAD]], ptr %[[BASEPTR_BYCOPY]], align 8
// CHECK: %[[OFFLOADPTR_BYREF:.*]] = getelementptr inbounds [2 x ptr], ptr %.offload_ptrs, i32 0, i32 1
// CHECK: store ptr %[[BYCOPY_LOAD]], ptr %[[OFFLOADPTR_BYREF]], align 8
