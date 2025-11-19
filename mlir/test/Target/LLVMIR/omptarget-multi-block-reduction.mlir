// RUN: mlir-translate -mlir-to-llvmir %s | FileCheck %s

// Verifies that the IR builder can handle reductions with multi-block combiner
// regions on the GPU.

module attributes {dlti.dl_spec = #dlti.dl_spec<"dlti.alloca_memory_space" = 5 : ui64, "dlti.global_memory_space" = 1 : ui64>, llvm.target_triple = "amdgcn-amd-amdhsa", omp.is_gpu = true, omp.is_target_device = true} {
  llvm.func @bar() {}
  llvm.func @baz() {}

  omp.declare_reduction @add_reduction_byref_box_5xf32 : !llvm.ptr alloc {
    %0 = llvm.mlir.constant(1 : i64) : i64
    %1 = llvm.alloca %0 x !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)> : (i64) -> !llvm.ptr<5>
    %2 = llvm.addrspacecast %1 : !llvm.ptr<5> to !llvm.ptr
    omp.yield(%2 : !llvm.ptr)
  } init {
  ^bb0(%arg0: !llvm.ptr, %arg1: !llvm.ptr):
    omp.yield(%arg1 : !llvm.ptr)
  } combiner {
  ^bb0(%arg0: !llvm.ptr, %arg1: !llvm.ptr):
    llvm.call @bar() : () -> ()
    llvm.br ^bb3

  ^bb3:  // pred: ^bb1
    llvm.call @baz() : () -> ()
    omp.yield(%arg0 : !llvm.ptr)
  }
  llvm.func @foo_() {
    %c1 = llvm.mlir.constant(1 : i64) : i64
    %10 = llvm.alloca %c1 x !llvm.array<5 x f32> {bindc_name = "x"} : (i64) -> !llvm.ptr<5>
    %11 = llvm.addrspacecast %10 : !llvm.ptr<5> to !llvm.ptr
    %74 = omp.map.info var_ptr(%11 : !llvm.ptr, !llvm.array<5 x f32>) map_clauses(tofrom) capture(ByRef) -> !llvm.ptr {name = "x"}
    omp.target map_entries(%74 -> %arg0 : !llvm.ptr) {
      %c1_2 = llvm.mlir.constant(1 : i32) : i32
      %c10 = llvm.mlir.constant(10 : i32) : i32
      omp.teams reduction(byref @add_reduction_byref_box_5xf32 %arg0 -> %arg2 : !llvm.ptr) {
        omp.parallel {
          omp.distribute {
            omp.wsloop {
              omp.loop_nest (%arg5) : i32 = (%c1_2) to (%c10) inclusive step (%c1_2) {
                omp.yield
              }
            } {omp.composite}
          } {omp.composite}
          omp.terminator
        } {omp.composite}
        omp.terminator
      }
      omp.terminator
    }
    llvm.return
  }
}

// CHECK:      call void @__kmpc_parallel_51({{.*}}, i32 1, i32 -1, i32 -1,
// CHECK-SAME:   ptr @[[PAR_OUTLINED:.*]], ptr null, ptr %2, i64 1)

// CHECK: define internal void @[[PAR_OUTLINED]]{{.*}} {
// CHECK:   .omp.reduction.then:
// CHECK:     br label %omp.reduction.nonatomic.body

// CHECK:   omp.reduction.nonatomic.body:
// CHECK:     call void @bar()
// CHECK:     br label %[[BODY_2ND_BB:.*]]

// CHECK:   [[BODY_2ND_BB]]:
// CHECK:     call void @baz()
// CHECK:     br label %[[CONT_BB:.*]]

// CHECK:   [[CONT_BB]]:
// CHECK-NEXT: %[[RED_RHS:.*]] = phi ptr [ %final.rhs, %{{.*}} ]
// CHECK-NEXT: store ptr %[[RED_RHS]], ptr %{{.*}}, align 8
// CHECK-NEXT: br label %.omp.reduction.done
// CHECK: }

// CHECK: define internal void @"{{.*}}$reduction$reduction_func"(ptr noundef %0, ptr noundef %1) #0 {
// CHECK:     br label %omp.reduction.nonatomic.body

// CHECK:   [[BODY_2ND_BB:.*]]:
// CHECK:     call void @baz()
// CHECK:     br label %omp.region.cont


// CHECK: omp.reduction.nonatomic.body:
// CHECK:   call void @bar()
// CHECK:     br label %[[BODY_2ND_BB]]

// CHECK: }
