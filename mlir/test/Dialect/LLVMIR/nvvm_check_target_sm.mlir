// RUN: mlir-opt %s -split-input-file -verify-diagnostics

gpu.module @tcgen05_alloc_sm90 [#nvvm.target<chip = "sm_90">] {
  func.func @tcgen05_alloc_sm90(%addr: !llvm.ptr, %ncols: i32) {
    // expected-error @below {{'nvvm.tcgen05.alloc' op is not supported on sm_90}}
    nvvm.tcgen05.alloc %addr, %ncols : !llvm.ptr, i32
    return
  }
}

// -----

gpu.module @tcgen05_alloc_sm100 [#nvvm.target<chip = "sm_100">] {
  func.func @tcgen05_alloc_sm100(%addr: !llvm.ptr, %ncols: i32) {
    // expected-error @below {{'nvvm.tcgen05.alloc' op is not supported on sm_100}}
    nvvm.tcgen05.alloc %addr, %ncols : !llvm.ptr, i32
    return
  }
}

// -----

gpu.module @tcgen05_dealloc_sm90a [#nvvm.target<chip = "sm_90a">] {
  func.func @tcgen05_dealloc_sm90a(%taddr: !llvm.ptr<6>, %ncols: i32) {
    // expected-error @below {{'nvvm.tcgen05.dealloc' op is not supported on sm_90a}}
    nvvm.tcgen05.dealloc %taddr, %ncols : !llvm.ptr<6>, i32
    return
  }
}

// -----

gpu.module @tcgen05_relinquish_alloc_permit_sm100 [#nvvm.target<chip = "sm_100">] {
  func.func @tcgen05_relinquish_alloc_permit_sm100() {
    // expected-error @below {{'nvvm.tcgen05.relinquish_alloc_permit' op is not supported on sm_100}}
    nvvm.tcgen05.relinquish_alloc_permit
    return
  }
}

// -----

gpu.module @tcgen05_fence_sm120f [#nvvm.target<chip = "sm_120f">] {
  func.func @tcgen05_fence_sm120f() {
    // expected-error @below {{'nvvm.tcgen05.fence' op is not supported on sm_120f}}
    nvvm.tcgen05.fence #nvvm.tcgen05_fence<before>
    return
  }
}

// -----

gpu.module @tcgen05_wait_sm90 [#nvvm.target<chip = "sm_90">] {
  func.func @tcgen05_wait_sm90() {
    // expected-error @below {{'nvvm.tcgen05.wait' op is not supported on sm_90}}
    nvvm.tcgen05.wait #nvvm.tcgen05_wait<load>
    return
  }
}

// -----

gpu.module @tcgen05_commit_sm100 [#nvvm.target<chip = "sm_100">] {
  func.func @tcgen05_commit_sm100(%barrier: !llvm.ptr) {
    // expected-error @below {{'nvvm.tcgen05.commit' op is not supported on sm_100}}
    nvvm.tcgen05.commit %barrier : !llvm.ptr
    return
  }
}

// -----

gpu.module @tcgen05_cp_sm90a [#nvvm.target<chip = "sm_90a">] {
  func.func @tcgen05_cp_sm90a(%taddr: !llvm.ptr<6>, %sdesc: i64) {
    // expected-error @below {{'nvvm.tcgen05.cp' op is not supported on sm_90a}}
    nvvm.tcgen05.cp %taddr, %sdesc {shape = #nvvm.tcgen05_cp_shape<shape_128x256b>}
    return
  }
}

// -----

gpu.module @tcgen05_ld_sm90 [#nvvm.target<chip = "sm_90">] {
  func.func @tcgen05_ld_sm90(%taddr: !llvm.ptr<6>) {
    // expected-error @below {{'nvvm.tcgen05.ld' op is not supported on sm_90}}
    %0 = nvvm.tcgen05.ld %taddr {shape = #nvvm.tcgen05_ldst_shape<shape_16x64b>} : i32
    return
  }
}

// -----

gpu.module @tcgen05_st_sm120f [#nvvm.target<chip = "sm_120f">] {
  func.func @tcgen05_st_sm120f(%taddr: !llvm.ptr<6>, %val: i32) {
    // expected-error @below {{'nvvm.tcgen05.st' op is not supported on sm_120f}}
    nvvm.tcgen05.st %taddr, %val {shape = #nvvm.tcgen05_ldst_shape<shape_16x64b>} : i32
    return
  }
}

// -----

gpu.module @tcgen05_mma_sm90 [#nvvm.target<chip = "sm_90">] {
  func.func @tcgen05_mma_sm90(%d: !llvm.ptr<6>, %a: i64, %b: i64, %idesc: i32, %eid: i1) {
    // expected-error @below {{'nvvm.tcgen05.mma' op is not supported on sm_90}}
    nvvm.tcgen05.mma %d, %a, %b, %idesc, %eid {kind = #nvvm.tcgen05_mma_kind<f16>, ctaGroup = #nvvm.cta_group<cta_1>} : (!llvm.ptr<6>, i64, i64, i32, i1)
    return
  }
}

// -----

gpu.module @tcgen05_mma_sp_sm100 [#nvvm.target<chip = "sm_100">] {
  func.func @tcgen05_mma_sp_sm100(%d: !llvm.ptr<6>, %a: i64, %b: i64, %idesc: i32, %eid: i1, %sp: !llvm.ptr<6>) {
    // expected-error @below {{'nvvm.tcgen05.mma.sp' op is not supported on sm_100}}
    nvvm.tcgen05.mma.sp %d, %a, %b, %idesc, %eid, %sp {kind = #nvvm.tcgen05_mma_kind<f16>, ctaGroup = #nvvm.cta_group<cta_1>} : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>)
    return
  }
}

// -----

gpu.module @tcgen05_mma_block_scale_sm90a [#nvvm.target<chip = "sm_90a">] {
  func.func @tcgen05_mma_block_scale_sm90a(%d: !llvm.ptr<6>, %a: i64, %b: i64, %idesc: i32, %eid: i1, %sa: !llvm.ptr<6>, %sb: !llvm.ptr<6>) {
    // expected-error @below {{'nvvm.tcgen05.mma.block_scale' op is not supported on sm_90a}}
    nvvm.tcgen05.mma.block_scale %d, %a, %b, %idesc, %eid, %sa, %sb {kind = #nvvm.tcgen05_mma_kind<mxf8f6f4>, ctaGroup = #nvvm.cta_group<cta_1>} : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>, !llvm.ptr<6>)
    return
  }
}

// -----

gpu.module @tcgen05_mma_sp_block_scale_sm90 [#nvvm.target<chip = "sm_90">] {
  func.func @tcgen05_mma_sp_block_scale_sm90(%d: !llvm.ptr<6>, %a: i64, %b: i64, %idesc: i32, %eid: i1, %sp: !llvm.ptr<6>, %sa: !llvm.ptr<6>, %sb: !llvm.ptr<6>) {
    // expected-error @below {{'nvvm.tcgen05.mma.sp.block_scale' op is not supported on sm_90}}
    nvvm.tcgen05.mma.sp.block_scale %d, %a, %b, %idesc, %eid, %sp, %sa, %sb {kind = #nvvm.tcgen05_mma_kind<mxf8f6f4>, ctaGroup = #nvvm.cta_group<cta_1>} : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>, !llvm.ptr<6>, !llvm.ptr<6>)
    return
  }
}

// -----

gpu.module @tcgen05_mma_ws_sm120f [#nvvm.target<chip = "sm_120f">] {
  func.func @tcgen05_mma_ws_sm120f(%d: !llvm.ptr<6>, %a: i64, %b: i64, %idesc: i32, %eid: i1) {
    // expected-error @below {{'nvvm.tcgen05.mma.ws' op is not supported on sm_120f}}
    nvvm.tcgen05.mma.ws %d, %a, %b, %idesc, %eid {kind = #nvvm.tcgen05_mma_kind<f16>} : (!llvm.ptr<6>, i64, i64, i32, i1)
    return
  }
}

// -----

gpu.module @tcgen05_mma_ws_sp_sm90a [#nvvm.target<chip = "sm_90a">] {
  func.func @tcgen05_mma_ws_sp_sm90a(%d: !llvm.ptr<6>, %a: i64, %b: i64, %idesc: i32, %eid: i1, %sp: !llvm.ptr<6>) {
    // expected-error @below {{'nvvm.tcgen05.mma.ws.sp' op is not supported on sm_90a}}
    nvvm.tcgen05.mma.ws.sp %d, %a, %b, %idesc, %eid, %sp {kind = #nvvm.tcgen05_mma_kind<f16>} : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>)
    return
  }
}

// -----

gpu.module @tcgen05_shift_sm90a [#nvvm.target<chip = "sm_90a">] {
  func.func @tcgen05_shift_sm90a(%taddr: !llvm.ptr<6>) {
    // expected-error @below {{'nvvm.tcgen05.shift' op is not supported on sm_90a}}
    nvvm.tcgen05.shift %taddr : !llvm.ptr<6>
    return
  }
}

// -----

gpu.module @tcgen05_shift_sm100f [#nvvm.target<chip = "sm_100f">] {
  func.func @tcgen05_shift_sm100f(%taddr: !llvm.ptr<6>) {
    // expected-error @below {{'nvvm.tcgen05.shift' op is not supported on sm_100f}}
    nvvm.tcgen05.shift %taddr : !llvm.ptr<6>
    return
  }
}

// -----

gpu.module @tcgen05_shift_sm100 [#nvvm.target<chip = "sm_100">] {
  func.func @tcgen05_shift_sm100(%taddr: !llvm.ptr<6>) {
    // expected-error @below {{'nvvm.tcgen05.shift' op is not supported on sm_100}}
    nvvm.tcgen05.shift %taddr : !llvm.ptr<6>
    return
  }
}

// -----

gpu.module @tcgen05_ld_red_sm100a [#nvvm.target<chip = "sm_100a">] {
  func.func @tcgen05_ld_red_sm100a(%addr: !llvm.ptr<6>) {
    // expected-error @below {{'nvvm.tcgen05.ld.red' op is not supported on sm_100a}}
    %data, %rv = nvvm.tcgen05.ld.red min %addr {shape = #nvvm.tcgen05_ldst_shape<shape_32x32b>} : vector<2xi32>, i32
    return
  }
}

// -----

gpu.module @tcgen05_ld_red_sm90a [#nvvm.target<chip = "sm_90a">] {
  func.func @tcgen05_ld_red_sm90a(%addr: !llvm.ptr<6>) {
    // expected-error @below {{'nvvm.tcgen05.ld.red' op is not supported on sm_90a}}
    %data, %rv = nvvm.tcgen05.ld.red min %addr {shape = #nvvm.tcgen05_ldst_shape<shape_32x32b>} : vector<2xi32>, i32
    return
  }
}

// -----

gpu.module @tensormap_replace_sm80 [#nvvm.target<chip = "sm_80">] {
  func.func @tensormap_replace_sm80(%addr: !llvm.ptr<1>, %nv: i64) {
    // expected-error @below {{'nvvm.tensormap.replace' op is not supported on sm_80}}
    nvvm.tensormap.replace field = global_address, new_value = %nv in %addr : !llvm.ptr<1>, i64
    return
  }
}

// -----

gpu.module @tensormap_replace_sm90 [#nvvm.target<chip = "sm_90">] {
  func.func @tensormap_replace_sm90(%addr: !llvm.ptr<1>, %nv: i64) {
    // expected-error @below {{'nvvm.tensormap.replace' op is not supported on sm_90}}
    nvvm.tensormap.replace field = global_address, new_value = %nv in %addr : !llvm.ptr<1>, i64
    return
  }
}
