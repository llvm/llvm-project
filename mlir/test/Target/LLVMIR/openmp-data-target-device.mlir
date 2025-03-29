// RUN: mlir-translate -mlir-to-llvmir %s | FileCheck %s

// This tests checks that a target op inside a data op
// We are only interested in ensuring that the -mlir-to-llmvir pass doesn't crash.
// CHECK: {{.*}} = add i32 {{.*}}, 1
module attributes { } {
  llvm.mlir.global weak_odr hidden local_unnamed_addr constant @__oclc_ABI_version(400 : i32) {addr_space = 4 : i32} : i32
  llvm.func @_QQmain() attributes {fir.bindc_name = "main", omp.declare_target = #omp.declaretarget<device_type = (host), capture_clause = (to)>} {
    %0 = llvm.mlir.constant(99 : index) : i64
    %1 = llvm.mlir.constant(0 : index) : i64
    %2 = llvm.mlir.constant(1 : index) : i64
    %3 = llvm.mlir.constant(100 : index) : i64
    %4 = llvm.mlir.constant(1 : i64) : i64
    %5 = llvm.alloca %4 x i32 {bindc_name = "array_length"} : (i64) -> !llvm.ptr<5>
    %6 = llvm.addrspacecast %5 : !llvm.ptr<5> to !llvm.ptr
    %7 = llvm.mlir.constant(1 : i64) : i64
    %8 = llvm.alloca %7 x i32 {bindc_name = "index_"} : (i64) -> !llvm.ptr<5>
    %9 = llvm.addrspacecast %8 : !llvm.ptr<5> to !llvm.ptr
    %10 = llvm.mlir.addressof @_QFEint_array : !llvm.ptr
    %11 = omp.map.bounds lower_bound(%1 : i64) upper_bound(%0 : i64) extent(%3 : i64) stride(%2 : i64) start_idx(%2 : i64)
    %12 = omp.map.info var_ptr(%10 : !llvm.ptr, !llvm.array<100 x i32>) map_clauses(from) capture(ByRef) bounds(%11) -> !llvm.ptr {name = "int_array"}
    omp.target_data map_entries(%12 : !llvm.ptr) {
      %13 = omp.map.info var_ptr(%10 : !llvm.ptr, !llvm.array<100 x i32>) map_clauses(from) capture(ByRef) bounds(%11) -> !llvm.ptr {name = "int_array"}
      %14 = omp.map.info var_ptr(%9 : !llvm.ptr, i32) map_clauses(implicit, exit_release_or_enter_alloc) capture(ByCopy) -> !llvm.ptr {name = "index_"}
      omp.target map_entries(%13 -> %arg0, %14 -> %arg1 : !llvm.ptr, !llvm.ptr) {
        %15 = llvm.mlir.constant(100 : i32) : i32
        %16 = llvm.mlir.constant(1 : i32) : i32
        %17 = llvm.mlir.constant(100 : index) : i64
        omp.parallel {
          %18 = llvm.mlir.constant(1 : i64) : i64
          %19 = llvm.alloca %18 x i32 {pinned} : (i64) -> !llvm.ptr<5>
          %20 = llvm.addrspacecast %19 : !llvm.ptr<5> to !llvm.ptr
          omp.wsloop {
            omp.loop_nest (%arg2) : i32 = (%16) to (%15) inclusive step (%16) {
              llvm.store %arg2, %20 : i32, !llvm.ptr
              %21 = llvm.load %20 : !llvm.ptr -> i32
              %22 = llvm.sext %21 : i32 to i64
              %23 = llvm.mlir.constant(1 : i64) : i64
              %24 = llvm.mlir.constant(0 : i64) : i64
              %25 = llvm.sub %22, %23 overflow<nsw>  : i64
              %26 = llvm.mul %25, %23 overflow<nsw>  : i64
              %27 = llvm.mul %26, %23 overflow<nsw>  : i64
              %28 = llvm.add %27, %24 overflow<nsw>  : i64
              %29 = llvm.mul %23, %17 overflow<nsw>  : i64
              %30 = llvm.getelementptr %arg0[%28] : (!llvm.ptr, i64) -> !llvm.ptr, i32
              llvm.store %21, %30 : i32, !llvm.ptr
              omp.yield
            }
          }
          omp.terminator
        }
        omp.terminator
      }
      omp.terminator
    }
    llvm.return
  }
  llvm.mlir.global internal @_QFEint_array() {addr_space = 0 : i32} : !llvm.array<100 x i32> {
    %0 = llvm.mlir.zero : !llvm.array<100 x i32>
    llvm.return %0 : !llvm.array<100 x i32>
  }
}
