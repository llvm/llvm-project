// RUN: mlir-translate -mlir-to-llvmir %s | FileCheck %s

// The aim of this test is to verfiy that information of
// alignment of loaded objects is passed to outlined
// functions.

module attributes {llvm.data_layout = "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128-p9:192:256:256:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8:9", llvm.target_triple = "amdgcn-amd-amdhsa", omp.is_gpu = true, omp.is_target_device = true} {
  omp.private {type = private} @_QFEk_private_i32 : i32
  llvm.func @_QQmain()  {
    %0 = llvm.mlir.constant(1 : i32) : i32
    %7 = llvm.alloca %0 x !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)> {alignment = 8 : i64} : (i32) -> !llvm.ptr<5>
    %8 = llvm.addrspacecast %7 : !llvm.ptr<5> to !llvm.ptr
    %12 = llvm.mlir.constant(1 : i64) : i64
    %13 = llvm.alloca %12 x i32 {bindc_name = "k"} : (i64) -> !llvm.ptr<5>
    %14 = llvm.addrspacecast %13 : !llvm.ptr<5> to !llvm.ptr
    %15 = llvm.mlir.constant(1 : i64) : i64
    %16 = llvm.alloca %15 x i32 {bindc_name = "b"} : (i64) -> !llvm.ptr<5>
    %17 = llvm.addrspacecast %16 : !llvm.ptr<5> to !llvm.ptr
    %19 = llvm.mlir.constant(1 : index) : i64
    %20 = llvm.mlir.constant(0 : index) : i64
    %22 = llvm.mlir.addressof @_QFEa : !llvm.ptr
    %25 = llvm.mlir.addressof @_QFECnz : !llvm.ptr
    %60 = llvm.getelementptr %8[0, 7, %20, 0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)>
    %61 = llvm.load %60 : !llvm.ptr -> i64
    %62 = llvm.getelementptr %8[0, 7, %20, 1] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)>
    %63 = llvm.load %62 : !llvm.ptr -> i64
    %64 = llvm.getelementptr %8[0, 7, %20, 2] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)>
    %65 = llvm.load %64 : !llvm.ptr -> i64
    %66 = llvm.sub %63, %19 : i64
    %67 = omp.map.bounds lower_bound(%20 : i64) upper_bound(%66 : i64) extent(%63 : i64) stride(%65 : i64) start_idx(%61 : i64) {stride_in_bytes = true}
    %68 = llvm.getelementptr %22[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)>
    %69 = omp.map.info var_ptr(%22 : !llvm.ptr, i32) map_clauses(tofrom) capture(ByRef) var_ptr_ptr(%68 : !llvm.ptr) bounds(%67) -> !llvm.ptr {name = ""}
    %70 = omp.map.info var_ptr(%22 : !llvm.ptr, !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)>) map_clauses(to) capture(ByRef) members(%69 : [0] : !llvm.ptr) -> !llvm.ptr {name = "a"}
    %71 = omp.map.info var_ptr(%17 : !llvm.ptr, i32) map_clauses(tofrom) capture(ByRef) -> !llvm.ptr {name = "b"}
    %72 = omp.map.info var_ptr(%14 : !llvm.ptr, i32) map_clauses(implicit, exit_release_or_enter_alloc) capture(ByCopy) -> !llvm.ptr {name = "k"}
    %73 = omp.map.info var_ptr(%25 : !llvm.ptr, i32) map_clauses(implicit, exit_release_or_enter_alloc) capture(ByCopy) -> !llvm.ptr {name = "nz"}
    omp.target map_entries(%70 -> %arg0, %71 -> %arg1, %72 -> %arg2, %73 -> %arg3, %69 -> %arg4 : !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr) {
      %106 = llvm.mlir.constant(0 : index) : i64
      %107 = llvm.mlir.constant(13 : i32) : i32
      %108 = llvm.mlir.constant(1000 : i32) : i32
      %109 = llvm.mlir.constant(1 : i32) : i32
      omp.teams {
        omp.parallel private(@_QFEk_private_i32 %arg2 -> %arg5 : !llvm.ptr) {
          %110 = llvm.mlir.constant(1 : i32) : i32
          %111 = llvm.alloca %110 x !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)> {alignment = 8 : i64} : (i32) -> !llvm.ptr<5>
          %112 = llvm.addrspacecast %111 : !llvm.ptr<5> to !llvm.ptr
          omp.distribute {
            omp.wsloop {
              omp.loop_nest (%arg6) : i32 = (%109) to (%108) inclusive step (%109) {
                llvm.store %arg6, %arg5  : i32, !llvm.ptr
                %115 = llvm.mlir.constant(48 : i32) : i32
                "llvm.intr.memcpy"(%112, %arg0, %115) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i32) -> ()
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
  llvm.mlir.global internal @_QFEa() {addr_space = 0 : i32} : !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)> {
    %6 = llvm.mlir.undef : !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)>
    llvm.return %6 : !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)>
  }
  llvm.mlir.global internal constant @_QFECnz() {addr_space = 0 : i32} : i32 {
    %0 = llvm.mlir.constant(1000 : i32) : i32
    llvm.return %0 : i32
  }
}

// CHECK:   call void @__kmpc_distribute_for_static_loop_4u(
// CHECK-SAME:  ptr addrspacecast (ptr addrspace(1) @[[GLOB:[0-9]+]] to ptr),
// CHECK-SAME:  ptr @[[LOOP_BODY_FUNC:.*]], ptr %[[LOOP_BODY_FUNC_ARG:.*]],
// CHEKC-SAME   i32 1000, i32 %1, i32 0, i32 0)


// CHECK:   define internal void @[[LOOP_BODY_FUNC]](i32 %[[CNT:.*]], ptr %[[LOOP_BODY_ARG_PTR:.*]]) #[[ATTRS:[0-9]+]] {
// CHECK:       %[[GEP_PTR_0:.*]] = getelementptr { ptr, ptr, ptr }, ptr %[[LOOP_BODY_ARG_PTR]], i32 0, i32 0
// CHECK:       %[[INT_PTR:.*]] = load ptr, ptr %[[GEP_PTR_0]], align 8, !align ![[ALIGN_INT:[0-9]+]]
// CHECK:       %[[GEP_PTR_1:.*]] = getelementptr { ptr, ptr, ptr }, ptr %[[LOOP_BODY_ARG_PTR]], i32 0, i32 1
// CHECK:       %[[STRUCT_PTR_0:.*]] = load ptr, ptr %[[GEP_PTR_1]], align 8, !align ![[ALIGN_STRUCT:[0-9]+]]
// CHECK:       %[[GEP_PTR_2:.*]] = getelementptr { ptr, ptr, ptr }, ptr %[[LOOP_BODY_ARG_PTR]], i32 0, i32 2
// CHECK:       %[[STRUCT_PTR_1:.*]] = load ptr, ptr %[[GEP_PTR_2]], align 8, !align ![[ALIGN_STRUCT:[0-9]+]]
// CHECK:       store i32 %[[DATA_INT:.*]], ptr %[[INT_PTR]], align 4
// CHECK:       call void @llvm.memcpy.p0.p0.i32(ptr %[[STRUCT_PTR_0]], ptr %[[STRUCT_PTR_1]], i32 48, i1 false)

// CHECK:       ![[ALIGN_STRUCT]] = !{i64 8}
// CHECK:       ![[ALIGN_INT]] = !{i64 4}
