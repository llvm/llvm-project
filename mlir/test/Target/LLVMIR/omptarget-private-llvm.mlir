// RUN: mlir-translate -mlir-to-llvmir %s | FileCheck %s

// Regression tset for calling a function using pointer alloca'ed on
// device for private variable

module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<"dlti.alloca_memory_space", 5 : ui32>>, llvm.data_layout = "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8:9", llvm.target_triple = "amdgcn-amd-amdhsa", omp.is_gpu = true, omp.is_target_device = true} {
  omp.private {type = private} @_QMmodFfailingEi_private_i32 : i32
  llvm.func @_QMotherProutine(%arg0: !llvm.ptr {fir.bindc_name = "i", llvm.nocapture}) attributes {frame_pointer = #llvm.framePointerKind<all>, omp.declare_target = #omp.declaretarget<device_type = (nohost), capture_clause = (to)>, target_cpu = "gfx90a", target_features = #llvm.target_features<["+16-bit-insts", "+atomic-buffer-global-pk-add-f16-insts", "+atomic-fadd-rtn-insts", "+ci-insts", "+dl-insts", "+dot1-insts", "+dot10-insts", "+dot2-insts", "+dot3-insts", "+dot4-insts", "+dot5-insts", "+dot6-insts", "+dot7-insts", "+dpp", "+gfx8-insts", "+gfx9-insts", "+gfx90a-insts", "+gws", "+image-insts", "+mai-insts", "+s-memrealtime", "+s-memtime-inst", "+wavefrontsize64"]>} {
    llvm.return
  }
  llvm.func @_QMmodPfailing(%arg0: !llvm.ptr {fir.bindc_name = "d", llvm.nocapture}) attributes {frame_pointer = #llvm.framePointerKind<all>, omp.declare_target = #omp.declaretarget<device_type = (host), capture_clause = (to)>, target_cpu = "gfx90a", target_features = #llvm.target_features<["+16-bit-insts", "+atomic-buffer-global-pk-add-f16-insts", "+atomic-fadd-rtn-insts", "+ci-insts", "+dl-insts", "+dot1-insts", "+dot10-insts", "+dot2-insts", "+dot3-insts", "+dot4-insts", "+dot5-insts", "+dot6-insts", "+dot7-insts", "+dpp", "+gfx8-insts", "+gfx9-insts", "+gfx90a-insts", "+gws", "+image-insts", "+mai-insts", "+s-memrealtime", "+s-memtime-inst", "+wavefrontsize64"]>} {
    %0 = llvm.mlir.constant(1 : i64) : i64
    %1 = llvm.alloca %0 x i32 {bindc_name = "i"} : (i64) -> !llvm.ptr<5>
    %2 = llvm.addrspacecast %1 : !llvm.ptr<5> to !llvm.ptr
    %3 = llvm.mlir.constant(1 : i64) : i64
    %4 = omp.map.info var_ptr(%2 : !llvm.ptr, i32) map_clauses(implicit, exit_release_or_enter_alloc) capture(ByCopy) -> !llvm.ptr {name = "i"}
    %5 = omp.map.info var_ptr(%arg0 : !llvm.ptr, i32) map_clauses(implicit, exit_release_or_enter_alloc) capture(ByCopy) -> !llvm.ptr {name = "d"}
    omp.target map_entries(%4 -> %arg1, %5 -> %arg2 : !llvm.ptr, !llvm.ptr) {
      %6 = llvm.mlir.constant(1 : i32) : i32
      omp.teams {

// CHECK:    omp.par.entry:
// CHECK:      %[[TID_ADDR_LOCAL:.*]] = alloca i32, align 4, addrspace(5)
// CHECK:      %[[OMP_PRIVATE_ALLOC:omp\.private\.alloc]] = alloca i32, align 4, addrspace(5)
// CHECK-NEXT: %[[CAST:.*]] = addrspacecast ptr addrspace(5) %[[OMP_PRIVATE_ALLOC]] to ptr

        omp.parallel private(@_QMmodFfailingEi_private_i32 %arg1 -> %arg3 : !llvm.ptr) {
          %7 = llvm.load %arg2 : !llvm.ptr -> i32
          omp.distribute {
            omp.wsloop {
              omp.loop_nest (%arg4) : i32 = (%6) to (%7) inclusive step (%6) {
                llvm.store %arg4, %arg3 : i32, !llvm.ptr
                llvm.call @_QMotherProutine(%arg3) {fastmathFlags = #llvm.fastmath<contract>} : (!llvm.ptr) -> ()
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
