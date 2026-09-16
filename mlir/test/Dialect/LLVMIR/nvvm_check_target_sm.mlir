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
    nvvm.tcgen05.cp %taddr, %sdesc , shape = shape_128x256b
    return
  }
}

// -----

gpu.module @tcgen05_ld_sm90 [#nvvm.target<chip = "sm_90">] {
  func.func @tcgen05_ld_sm90(%taddr: !llvm.ptr<6>) {
    // expected-error @below {{'nvvm.tcgen05.ld' op is not supported on sm_90}}
    %0 = nvvm.tcgen05.ld %taddr  shape = shape_16x64b : vector<1 x i32>
    return
  }
}

// -----

gpu.module @tcgen05_st_sm120f [#nvvm.target<chip = "sm_120f">] {
  func.func @tcgen05_st_sm120f(%taddr: !llvm.ptr<6>, %val: vector<1 x i32>) {
    // expected-error @below {{'nvvm.tcgen05.st' op is not supported on sm_120f}}
    nvvm.tcgen05.st %taddr, %val  shape = shape_16x64b : vector<1 x i32>
    return
  }
}

// -----

gpu.module @tcgen05_mma_sm90 [#nvvm.target<chip = "sm_90">] {
  func.func @tcgen05_mma_sm90(%d: !llvm.ptr<6>, %a: i64, %b: i64, %idesc: i32, %eid: i1) {
    // expected-error @below {{'nvvm.tcgen05.mma' op is not supported on sm_90}}
    nvvm.tcgen05.mma %d, %a, %b, %idesc, %eid , kind = f16, cta_group = <cta_1> : (!llvm.ptr<6>, i64, i64, i32, i1)
    return
  }
}

// -----

gpu.module @tcgen05_mma_sp_sm100 [#nvvm.target<chip = "sm_100">] {
  func.func @tcgen05_mma_sp_sm100(%d: !llvm.ptr<6>, %a: i64, %b: i64, %idesc: i32, %eid: i1, %sp: !llvm.ptr<6>) {
    // expected-error @below {{'nvvm.tcgen05.mma.sp' op is not supported on sm_100}}
    nvvm.tcgen05.mma.sp %d, %a, %b, %idesc, %eid, %sp , kind = f16, cta_group = <cta_1> : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>)
    return
  }
}

// -----

gpu.module @tcgen05_mma_block_scale_sm90a [#nvvm.target<chip = "sm_90a">] {
  func.func @tcgen05_mma_block_scale_sm90a(%d: !llvm.ptr<6>, %a: i64, %b: i64, %idesc: i32, %eid: i1, %sa: !llvm.ptr<6>, %sb: !llvm.ptr<6>) {
    // expected-error @below {{'nvvm.tcgen05.mma.block_scale' op is not supported on sm_90a}}
    nvvm.tcgen05.mma.block_scale %d, %a, %b, %idesc, %eid, %sa, %sb , kind = mxf8f6f4, cta_group = <cta_1> : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>, !llvm.ptr<6>)
    return
  }
}

// -----

gpu.module @tcgen05_mma_sp_block_scale_sm90 [#nvvm.target<chip = "sm_90">] {
  func.func @tcgen05_mma_sp_block_scale_sm90(%d: !llvm.ptr<6>, %a: i64, %b: i64, %idesc: i32, %eid: i1, %sp: !llvm.ptr<6>, %sa: !llvm.ptr<6>, %sb: !llvm.ptr<6>) {
    // expected-error @below {{'nvvm.tcgen05.mma.sp.block_scale' op is not supported on sm_90}}
    nvvm.tcgen05.mma.sp.block_scale %d, %a, %b, %idesc, %eid, %sp, %sa, %sb , kind = mxf8f6f4, cta_group = <cta_1> : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>, !llvm.ptr<6>, !llvm.ptr<6>)
    return
  }
}

// -----

gpu.module @tcgen05_mma_ws_sm120f [#nvvm.target<chip = "sm_120f">] {
  func.func @tcgen05_mma_ws_sm120f(%d: !llvm.ptr<6>, %a: i64, %b: i64, %idesc: i32, %eid: i1) {
    // expected-error @below {{'nvvm.tcgen05.mma.ws' op is not supported on sm_120f}}
    nvvm.tcgen05.mma.ws %d, %a, %b, %idesc, %eid  kind = f16 : (!llvm.ptr<6>, i64, i64, i32, i1)
    return
  }
}

// -----

gpu.module @tcgen05_mma_ws_sp_sm90a [#nvvm.target<chip = "sm_90a">] {
  func.func @tcgen05_mma_ws_sp_sm90a(%d: !llvm.ptr<6>, %a: i64, %b: i64, %idesc: i32, %eid: i1, %sp: !llvm.ptr<6>) {
    // expected-error @below {{'nvvm.tcgen05.mma.ws.sp' op is not supported on sm_90a}}
    nvvm.tcgen05.mma.ws.sp %d, %a, %b, %idesc, %eid, %sp  kind = f16 : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>)
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
    %data, %rv = nvvm.tcgen05.ld.red min %addr  shape = shape_32x32b : vector<2xi32>, i32
    return
  }
}

// -----

gpu.module @tcgen05_ld_red_sm90a [#nvvm.target<chip = "sm_90a">] {
  func.func @tcgen05_ld_red_sm90a(%addr: !llvm.ptr<6>) {
    // expected-error @below {{'nvvm.tcgen05.ld.red' op is not supported on sm_90a}}
    %data, %rv = nvvm.tcgen05.ld.red min %addr  shape = shape_32x32b : vector<2xi32>, i32
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

// -----

//===----------------------------------------------------------------------===//
// mbarrier
//===----------------------------------------------------------------------===//

// Just check these don't emit errors: every op at exactly its floor.
gpu.module @mbarrier_sm80 [#nvvm.target<chip = "sm_80">] {
  func.func @mbarrier_sm80(%addr: !llvm.ptr, %count: i32, %state: i64) {
    nvvm.mbarrier.init %addr, %count : !llvm.ptr, i32
    %0 = nvvm.mbarrier.arrive %addr : !llvm.ptr -> i64
    %1 = nvvm.mbarrier.arrive_drop %addr : !llvm.ptr -> i64
    %2 = nvvm.mbarrier.arrive.nocomplete %addr, %count : !llvm.ptr, i32 -> i64
    %3 = nvvm.mbarrier.arrive_drop.nocomplete %addr, %count : !llvm.ptr, i32 -> i64
    %4 = nvvm.mbarrier.test.wait %addr, %state : !llvm.ptr, i64 -> i1
    nvvm.mbarrier.inval %addr : !llvm.ptr
    return
  }
}

// -----

gpu.module @mbarrier_sm90 [#nvvm.target<chip = "sm_90">] {
  func.func @mbarrier_sm90(%addr: !llvm.ptr, %shared: !llvm.ptr<3>, %count: i32,
                           %state: i64, %phase: i32, %ticks: i32) {
    %0 = nvvm.mbarrier.arrive.expect_tx %addr, %count : !llvm.ptr, i32 -> i64
    %1 = nvvm.mbarrier.arrive_drop.expect_tx %addr, %count : !llvm.ptr, i32 -> i64
    nvvm.mbarrier.expect_tx %shared, %count : !llvm.ptr<3>, i32
    nvvm.mbarrier.complete_tx %shared, %count : !llvm.ptr<3>, i32
    %2 = nvvm.mbarrier.try_wait %addr, %state : !llvm.ptr, i64 -> i1
    nvvm.mbarrier.try_wait.parity %addr, %phase, %ticks : !llvm.ptr, i32, i32
    return
  }
}

// -----

// Arch-accelerated targets satisfy a plain NVVMRequiresSM floor: sm_90a is
// >= sm_80 and >= sm_90. This is the chip string real Hopper users write.
gpu.module @mbarrier_sm90a [#nvvm.target<chip = "sm_90a">] {
  func.func @mbarrier_sm90a(%addr: !llvm.ptr, %shared: !llvm.ptr<3>, %count: i32) {
    nvvm.mbarrier.init %addr, %count : !llvm.ptr, i32
    nvvm.mbarrier.expect_tx %shared, %count : !llvm.ptr<3>, i32
    return
  }
}

// -----

// NVVMRequiresSM is a floor, not an equality: every sm_80 op stays valid on
// sm_90. These positive modules pass before and after the traits were added;
// they exist to catch a floor set too high, which the negatives cannot.
gpu.module @mbarrier_sm80_ops_on_sm90 [#nvvm.target<chip = "sm_90">] {
  func.func @mbarrier_sm80_ops_on_sm90(%addr: !llvm.ptr, %count: i32, %state: i64) {
    nvvm.mbarrier.init %addr, %count : !llvm.ptr, i32
    %0 = nvvm.mbarrier.arrive %addr : !llvm.ptr -> i64
    %1 = nvvm.mbarrier.arrive_drop %addr : !llvm.ptr -> i64
    %2 = nvvm.mbarrier.arrive.nocomplete %addr, %count : !llvm.ptr, i32 -> i64
    %3 = nvvm.mbarrier.arrive_drop.nocomplete %addr, %count : !llvm.ptr, i32 -> i64
    %4 = nvvm.mbarrier.test.wait %addr, %state : !llvm.ptr, i64 -> i1
    nvvm.mbarrier.inval %addr : !llvm.ptr
    return
  }
}

// -----

// The check stays opt-out, as for every other op in this file.
gpu.module @mbarrier_verify_target_disabled
    [#nvvm.target<chip = "sm_70", verifyTarget = false>] {
  func.func @mbarrier_verify_target_disabled(%addr: !llvm.ptr, %count: i32) {
    nvvm.mbarrier.init %addr, %count : !llvm.ptr, i32
    return
  }
}

// -----

gpu.module @mbarrier_init_sm70 [#nvvm.target<chip = "sm_70">] {
  func.func @mbarrier_init_sm70(%addr: !llvm.ptr, %count: i32) {
    // expected-error @below {{'nvvm.mbarrier.init' op is not supported on sm_70}}
    nvvm.mbarrier.init %addr, %count : !llvm.ptr, i32
    return
  }
}

// -----

gpu.module @mbarrier_inval_sm70 [#nvvm.target<chip = "sm_70">] {
  func.func @mbarrier_inval_sm70(%addr: !llvm.ptr) {
    // expected-error @below {{'nvvm.mbarrier.inval' op is not supported on sm_70}}
    nvvm.mbarrier.inval %addr : !llvm.ptr
    return
  }
}

// -----

gpu.module @mbarrier_arrive_sm70 [#nvvm.target<chip = "sm_70">] {
  func.func @mbarrier_arrive_sm70(%addr: !llvm.ptr) {
    // expected-error @below {{'nvvm.mbarrier.arrive' op is not supported on sm_70}}
    %0 = nvvm.mbarrier.arrive %addr : !llvm.ptr -> i64
    return
  }
}

// -----

gpu.module @mbarrier_arrive_drop_sm70 [#nvvm.target<chip = "sm_70">] {
  func.func @mbarrier_arrive_drop_sm70(%addr: !llvm.ptr) {
    // expected-error @below {{'nvvm.mbarrier.arrive_drop' op is not supported on sm_70}}
    %0 = nvvm.mbarrier.arrive_drop %addr : !llvm.ptr -> i64
    return
  }
}

// -----

gpu.module @mbarrier_arrive_nocomplete_sm70 [#nvvm.target<chip = "sm_70">] {
  func.func @mbarrier_arrive_nocomplete_sm70(%addr: !llvm.ptr, %count: i32) {
    // expected-error @below {{'nvvm.mbarrier.arrive.nocomplete' op is not supported on sm_70}}
    %0 = nvvm.mbarrier.arrive.nocomplete %addr, %count : !llvm.ptr, i32 -> i64
    return
  }
}

// -----

gpu.module @mbarrier_arrive_drop_nocomplete_sm70 [#nvvm.target<chip = "sm_70">] {
  func.func @mbarrier_arrive_drop_nocomplete_sm70(%addr: !llvm.ptr, %count: i32) {
    // expected-error @below {{'nvvm.mbarrier.arrive_drop.nocomplete' op is not supported on sm_70}}
    %0 = nvvm.mbarrier.arrive_drop.nocomplete %addr, %count : !llvm.ptr, i32 -> i64
    return
  }
}

// -----

gpu.module @mbarrier_test_wait_sm70 [#nvvm.target<chip = "sm_70">] {
  func.func @mbarrier_test_wait_sm70(%addr: !llvm.ptr, %state: i64) {
    // expected-error @below {{'nvvm.mbarrier.test.wait' op is not supported on sm_70}}
    %0 = nvvm.mbarrier.test.wait %addr, %state : !llvm.ptr, i64 -> i1
    return
  }
}

// -----

gpu.module @mbarrier_arrive_expect_tx_sm80 [#nvvm.target<chip = "sm_80">] {
  func.func @mbarrier_arrive_expect_tx_sm80(%addr: !llvm.ptr, %count: i32) {
    // expected-error @below {{'nvvm.mbarrier.arrive.expect_tx' op is not supported on sm_80}}
    %0 = nvvm.mbarrier.arrive.expect_tx %addr, %count : !llvm.ptr, i32 -> i64
    return
  }
}

// -----

gpu.module @mbarrier_arrive_drop_expect_tx_sm80 [#nvvm.target<chip = "sm_80">] {
  func.func @mbarrier_arrive_drop_expect_tx_sm80(%addr: !llvm.ptr, %count: i32) {
    // expected-error @below {{'nvvm.mbarrier.arrive_drop.expect_tx' op is not supported on sm_80}}
    %0 = nvvm.mbarrier.arrive_drop.expect_tx %addr, %count : !llvm.ptr, i32 -> i64
    return
  }
}

// -----

gpu.module @mbarrier_expect_tx_sm80 [#nvvm.target<chip = "sm_80">] {
  func.func @mbarrier_expect_tx_sm80(%addr: !llvm.ptr<3>, %count: i32) {
    // expected-error @below {{'nvvm.mbarrier.expect_tx' op is not supported on sm_80}}
    nvvm.mbarrier.expect_tx %addr, %count : !llvm.ptr<3>, i32
    return
  }
}

// -----

gpu.module @mbarrier_complete_tx_sm80 [#nvvm.target<chip = "sm_80">] {
  func.func @mbarrier_complete_tx_sm80(%addr: !llvm.ptr<3>, %count: i32) {
    // expected-error @below {{'nvvm.mbarrier.complete_tx' op is not supported on sm_80}}
    nvvm.mbarrier.complete_tx %addr, %count : !llvm.ptr<3>, i32
    return
  }
}

// -----

gpu.module @mbarrier_try_wait_sm80 [#nvvm.target<chip = "sm_80">] {
  func.func @mbarrier_try_wait_sm80(%addr: !llvm.ptr, %state: i64) {
    // expected-error @below {{'nvvm.mbarrier.try_wait' op is not supported on sm_80}}
    %0 = nvvm.mbarrier.try_wait %addr, %state : !llvm.ptr, i64 -> i1
    return
  }
}

// -----

gpu.module @mbarrier_try_wait_parity_sm80 [#nvvm.target<chip = "sm_80">] {
  func.func @mbarrier_try_wait_parity_sm80(%addr: !llvm.ptr, %phase: i32, %ticks: i32) {
    // expected-error @below {{'nvvm.mbarrier.try_wait.parity' op is not supported on sm_80}}
    nvvm.mbarrier.try_wait.parity %addr, %phase, %ticks : !llvm.ptr, i32, i32
    return
  }
}
