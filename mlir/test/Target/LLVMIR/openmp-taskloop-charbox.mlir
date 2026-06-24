// RUN: mlir-translate --mlir-to-llvmir %s | FileCheck %s

module {
  llvm.func @touch(!llvm.ptr, i64)

  omp.private {type = firstprivate} @box_firstprivate : !llvm.struct<(ptr, i64)> init {
  ^bb0(%arg0: !llvm.struct<(ptr, i64)>, %arg1: !llvm.struct<(ptr, i64)>):
    %0 = llvm.extractvalue %arg0[0] : !llvm.struct<(ptr, i64)>
    %1 = llvm.extractvalue %arg0[1] : !llvm.struct<(ptr, i64)>
    %2 = llvm.mlir.undef : !llvm.struct<(ptr, i64)>
    %3 = llvm.insertvalue %0, %2[0] : !llvm.struct<(ptr, i64)>
    %4 = llvm.insertvalue %1, %3[1] : !llvm.struct<(ptr, i64)>
    omp.yield(%4 : !llvm.struct<(ptr, i64)>)
  } copy {
  ^bb0(%arg0: !llvm.struct<(ptr, i64)>, %arg1: !llvm.struct<(ptr, i64)>):
    %0 = llvm.extractvalue %arg0[0] : !llvm.struct<(ptr, i64)>
    %1 = llvm.extractvalue %arg0[1] : !llvm.struct<(ptr, i64)>
    llvm.call @touch(%0, %1) : (!llvm.ptr, i64) -> ()
    omp.yield(%arg0 : !llvm.struct<(ptr, i64)>)
  } dealloc {
  ^bb0(%arg0: !llvm.struct<(ptr, i64)>):
    %0 = llvm.extractvalue %arg0[0] : !llvm.struct<(ptr, i64)>
    %1 = llvm.extractvalue %arg0[1] : !llvm.struct<(ptr, i64)>
    llvm.call @touch(%0, %1) : (!llvm.ptr, i64) -> ()
    omp.yield
  }

  llvm.func @test(%arg0: !llvm.ptr, %arg1: i64) {
    %0 = llvm.mlir.undef : !llvm.struct<(ptr, i64)>
    %1 = llvm.insertvalue %arg0, %0[0] : !llvm.struct<(ptr, i64)>
    %2 = llvm.insertvalue %arg1, %1[1] : !llvm.struct<(ptr, i64)>
    %c1 = llvm.mlir.constant(1 : i32) : i32
    %c2 = llvm.mlir.constant(2 : i32) : i32
    omp.taskloop.context private(@box_firstprivate %2 -> %arg2 : !llvm.struct<(ptr, i64)>) {
      omp.taskloop.wrapper {
        omp.loop_nest (%arg3) : i32 = (%c1) to (%c2) inclusive step (%c1) {
          %3 = llvm.extractvalue %arg2[0] : !llvm.struct<(ptr, i64)>
          %4 = llvm.extractvalue %arg2[1] : !llvm.struct<(ptr, i64)>
          llvm.call @touch(%3, %4) : (!llvm.ptr, i64) -> ()
          omp.yield
        }
      }
      omp.terminator
    }
    llvm.return
  }
}

// CHECK-LABEL: define void @test(
// CHECK:         call void @__kmpc_taskloop(

// CHECK-LABEL: define internal void @omp_taskloop_dup(
// CHECK:         load { ptr, i64 }, ptr
// CHECK:         call void @touch(
