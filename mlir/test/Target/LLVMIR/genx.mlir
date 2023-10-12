// RUN: mlir-translate -mlir-to-llvmir %s | FileCheck %s

llvm.func @genx_special_regs() -> i64 {
  // CHECK-LABEL: genx_special_regs
  // CHECK: call i64 @_Z12get_local_idj(i32 0)
  %1 = genx.workitem.id.x : i64
  // CHECK: call i64 @_Z12get_local_idj(i32 1)
  %2 = genx.workitem.id.y : i64
  // CHECK: call i64 @_Z12get_local_idj(i32 2)
  %3 = genx.workitem.id.z : i64
  // CHECK: call i64 @_Z12get_group_idj(i32 0)
  %4 = genx.workgroup.id.x : i64
  // CHECK: call i64 @_Z12get_group_idj(i32 1)
  %5 = genx.workgroup.id.y : i64
  // CHECK: call i64 @_Z12get_group_idj(i32 2)
  %6 = genx.workgroup.id.z : i64
  // CHECK: call i64 @_Z12get_local_sizej(i32 0)
  %7 = genx.workgroup.dim.x : i64
  // CHECK: call i64 @_Z12get_local_sizej(i32 1)
  %8 = genx.workgroup.dim.y : i64
  // CHECK: call i64 @_Z12get_local_sizej(i32 2)
  %9 = genx.workgroup.dim.z : i64
  // CHECK: call i64 @_Z12get_global_sizej(i32 0)
  %10 = genx.grid.dim.x : i64
  // CHECK: call i64 @_Z12get_global_sizej(i32 1)
  %11 = genx.grid.dim.y : i64
  // CHECK: call i64 @_Z12get_global_sizej(i32 2)
  %12 = genx.grid.dim.z : i64

  llvm.return %1 : i64
}

llvm.func @genx.barrier() {
  // CHECK-LABEL: genx.barrier
  // CHECK: call void @_Z7barrierj(i32 3)
  genx.barrier
  llvm.return
}

llvm.func @genx.atomic_work_item_fence() {
  // CHECK-LABEL: genx.atomic_work_item_fence
  // CHECK: call void @_Z22atomic_work_item_fencej12memory_order12memory_scope(i32 1, i32 0, i32 0)
  genx.atomic_work_item_fence {flags=#genx.memory_fence_flag<LOCAL_MEM_FENCE>, order=#genx.memory_order<Relaxed>, scope=#genx.memory_scope<work_item>}
  // CHECK: call void @_Z22atomic_work_item_fencej12memory_order12memory_scope(i32 2, i32 2, i32 1)
  genx.atomic_work_item_fence {flags=#genx.memory_fence_flag<GLOBAL_MEM_FENCE>, order=#genx.memory_order<Acquire>, scope=#genx.memory_scope<work_group>}
  // CHECK: call void @_Z22atomic_work_item_fencej12memory_order12memory_scope(i32 4, i32 3, i32 2)
  genx.atomic_work_item_fence {flags=#genx.memory_fence_flag<IMAGE_MEM_FENCE>, order=#genx.memory_order<Release>, scope=#genx.memory_scope<device>}
  // CHECK: call void @_Z22atomic_work_item_fencej12memory_order12memory_scope(i32 1, i32 4, i32 3)
  genx.atomic_work_item_fence {flags=#genx.memory_fence_flag<LOCAL_MEM_FENCE>, order=#genx.memory_order<AcquireRelease>, scope=#genx.memory_scope<all_svm_devices>}
  // CHECK: call void @_Z22atomic_work_item_fencej12memory_order12memory_scope(i32 2, i32 5, i32 4)
  genx.atomic_work_item_fence {flags=#genx.memory_fence_flag<GLOBAL_MEM_FENCE>, order=#genx.memory_order<SequentiallyConsistent>, scope=#genx.memory_scope<sub_group>}
  // CHECK: call void @_Z22atomic_work_item_fencej12memory_order12memory_scope(i32 5, i32 2, i32 4)
  genx.atomic_work_item_fence {flags=#genx.memory_fence_flag<LOCAL_MEM_FENCE, IMAGE_MEM_FENCE>, order=#genx.memory_order<Acquire>, scope=#genx.memory_scope<sub_group>}
  llvm.return
}

llvm.func @genx.sub_group_shuffle() {
  // CHECK-LABEL: genx.sub_group_shuffle
  %0 = llvm.mlir.constant(0 : i32) : i32
  // CHECK: %1 = call i32 @_Z21sub_group_shuffle_xorij(i32 0, i32 0)
  %1 = genx.sub_group_shuffle XOR %0, %0 : i32 -> i32
  // CHECK: %2 = call i32 @_Z20sub_group_shuffle_upij(i32 0, i32 0)
  %2 = genx.sub_group_shuffle UP %0, %0 : i32 -> i32
  // CHECK: %3 = call i32 @_Z22sub_group_shuffle_downij(i32 0, i32 0)
  %3 = genx.sub_group_shuffle DOWN %0, %0 : i32 -> i32
  // CHECK: %4 = call i32 @_Z17sub_group_shuffleij(i32 0, i32 0)
  %4 = genx.sub_group_shuffle IDX %0, %0 : i32 -> i32
  %5 = llvm.mlir.constant(0 : i8) : i8
  // CHECK: %5 = call i8 @_Z21sub_group_shuffle_xorcj(i8 0, i32 0)
  %6 = genx.sub_group_shuffle XOR %5, %0 : i8 -> i8
  %7 = llvm.mlir.constant(0 : i16) : i16
  // CHECK: %6 = call i16 @_Z21sub_group_shuffle_xorsj(i16 0, i32 0)
  %8 = genx.sub_group_shuffle XOR %7, %0 : i16 -> i16
  %9 = llvm.mlir.constant(0 : i64) : i64
  // CHECK: %7 = call i64 @_Z21sub_group_shuffle_xorlj(i64 0, i32 0)
  %10 = genx.sub_group_shuffle XOR %9, %0 : i64 -> i64
  %11 = llvm.mlir.constant(0.0 : f16) : f16
  // CHECK: %8 = call half @_Z21sub_group_shuffle_xorDhj(half 0xH0000, i32 0)
  %12 = genx.sub_group_shuffle XOR %11, %0 : f16 -> f16
  %13 = llvm.mlir.constant(0.0 : f32) : f32
  // CHECK: %9 = call float @_Z21sub_group_shuffle_xorfj(float 0.000000e+00, i32 0)
  %14 = genx.sub_group_shuffle XOR %13, %0 : f32 -> f32
  %15 = llvm.mlir.constant(0.0 : f64) : f64
  // CHECK: %10 = call double @_Z21sub_group_shuffle_xordj(double 0.000000e+00, i32 0)
  %16 = genx.sub_group_shuffle XOR %15, %0 : f64 -> f64
  llvm.return
}

llvm.func @genx.atomic.cmpxchg.global.i32(%ptr : !llvm.ptr<i32, 1>, %cmp : i32, %val : i32)  {
  // CHECK-LABEL: genx.atomic.cmpxchg.global.i32
  // CHECK: call i32 @_Z12atom_cmpxchgPU8CLglobalViii(ptr addrspace(1) %0, i32 %1, i32 %2)
  %0 = genx.atomic.cmpxchg %ptr, %cmp, %val : (!llvm.ptr<i32, 1>, i32, i32) -> i32
  llvm.return
}

llvm.func @genx.atomic.cmpxchg.shared.u64(%ptr : !llvm.ptr<i64, 3>, %cmp : i64, %val : i64)  {
  // CHECK-LABEL: genx.atomic.cmpxchg.shared.u64
  // CHECK: call i64 @_Z12atom_cmpxchgPU7CLlocalVlll(ptr addrspace(3) %0, i64 %1, i64 %2)
  %0 = genx.atomic.cmpxchg %ptr, %cmp, %val : (!llvm.ptr<i64, 3>, i64, i64) -> i64
  llvm.return
}

llvm.func @genx.atomic.rmw(%ptr : !llvm.ptr<i32, 1>, %sptr : !llvm.ptr<i64, 3>, %val1 : i32, %val2 : i64) {
  // CHECK-LABEL: genx.atomic.rmw
  // CHECK: call i32 @_Z8atom_andPU8CLglobalVii(ptr addrspace(1) %0, i32 %2)
  %0 = genx.atomic.rmw AND %ptr, %val1 : (!llvm.ptr<i32, 1>, i32) -> i32
  // CHECK: call i32 @_Z7atom_orPU8CLglobalVii(ptr addrspace(1) %0, i32 %2)
  %1 = genx.atomic.rmw OR %ptr, %val1 : (!llvm.ptr<i32, 1>, i32) -> i32
  // CHECK: call i32 @_Z8atom_xorPU8CLglobalVii(ptr addrspace(1) %0, i32 %2)
  %2 = genx.atomic.rmw XOR %ptr, %val1 : (!llvm.ptr<i32, 1>, i32) -> i32
  // CHECK: call i32 @_Z8atom_addPU8CLglobalVii(ptr addrspace(1) %0, i32 %2)
  %3 = genx.atomic.rmw ADD %ptr, %val1 : (!llvm.ptr<i32, 1>, i32) -> i32
  // CHECK: call i32 @_Z8atom_minPU8CLglobalVii(ptr addrspace(1) %0, i32 %2)
  %4 = genx.atomic.rmw MIN %ptr, %val1 : (!llvm.ptr<i32, 1>, i32) -> i32
  // CHECK: call i32 @_Z8atom_maxPU8CLglobalVii(ptr addrspace(1) %0, i32 %2)
  %5 = genx.atomic.rmw MAX %ptr, %val1 : (!llvm.ptr<i32, 1>, i32) -> i32
  // CHECK: call i32 @_Z8atom_xchgPU8CLglobalVii(ptr addrspace(1) %0, i32 %2)
  %6 = genx.atomic.rmw XCHG %ptr, %val1 : (!llvm.ptr<i32, 1>, i32) -> i32

  // CHECK: call i64 @_Z8atom_andPU7CLlocalVll(ptr addrspace(3) %1, i64 %3)
  %7 = genx.atomic.rmw AND %sptr, %val2 : (!llvm.ptr<i64, 3>, i64) -> i64
  // CHECK: call i64 @_Z7atom_orPU7CLlocalVll(ptr addrspace(3) %1, i64 %3)
  %8 = genx.atomic.rmw OR %sptr, %val2 : (!llvm.ptr<i64, 3>, i64) -> i64
  // CHECK: call i64 @_Z8atom_xorPU7CLlocalVll(ptr addrspace(3) %1, i64 %3)
  %9 = genx.atomic.rmw XOR %sptr, %val2 : (!llvm.ptr<i64, 3>, i64) -> i64
  // CHECK: call i64 @_Z8atom_addPU7CLlocalVll(ptr addrspace(3) %1, i64 %3)
  %10 = genx.atomic.rmw ADD %sptr, %val2 : (!llvm.ptr<i64, 3>, i64) -> i64
  // CHECK: call i64 @_Z8atom_minPU7CLlocalVll(ptr addrspace(3) %1, i64 %3)
  %11 = genx.atomic.rmw MIN %sptr, %val2 : (!llvm.ptr<i64, 3>, i64) -> i64
  // CHECK: call i64 @_Z8atom_maxPU7CLlocalVll(ptr addrspace(3) %1, i64 %3)
  %12 = genx.atomic.rmw MAX %sptr, %val2 : (!llvm.ptr<i64, 3>, i64) -> i64
  // CHECK: call i64 @_Z8atom_xchgPU7CLlocalVll(ptr addrspace(3) %1, i64 %3)
  %13 = genx.atomic.rmw XCHG %sptr, %val2 : (!llvm.ptr<i64, 3>, i64) -> i64

  llvm.return
}

llvm.func @genx.dpas(%c : vector<8xi32>, %a : vector<16xi8>, %b : vector<32xi8>) {
  // CHECK: %4 = call <8 x i32> @llvm.genx.GenISA.sub.group.dpas.v8i32.v8i32.v16i8.v32i8(<8 x i32> %0, <16 x i8> %1, <32 x i8> %2, i32 4, i32 4, i32 8, i32 8, i1 false)
  %0 = genx.matrix.dpas %c, %a, %b {pa=#genx.precision_type<S8>, pb=#genx.precision_type<S8>, rc=8:i32} : (vector<8xi32>, vector<16xi8>, vector<32xi8>) -> vector<8xi32>
  llvm.return
}

llvm.func @genx.2Dblockload1x4.32.1.0.0(%ptr : !llvm.ptr<i32>, %base_width : i32, %base_height : i32, %base_pitch : i32, %x : i32, %y : i32) {
  // CHECK: [[PTR:%.*]] = ptrtoint ptr %0 to i64
  // CHECK-NEXT: call <4 x i32> @llvm.genx.GenISA.LSC2DBlockRead.v4i32(i64 [[PTR]], i32 %1, i32 %2, i32 %3, i32 %4, i32 %5, i32 32, i32 4, i32 1, i32 1, i1 false, i1 false)
  %0 = genx.matrix.2Dblockload %ptr, %base_width, %base_height, %base_pitch, %x, %y {elem_size_in_bits=32:i32, tile_width=4:i32, tile_height=1:i32, v_blocks=1:i32, transpose=false, vnni_transform=false} : (!llvm.ptr<i32>, i32, i32, i32, i32, i32) -> vector<4xi32>
  llvm.return
}

llvm.func @genx.2Dblockstore(%ptr : !llvm.ptr<i32>, %base_width : i32, %base_height : i32, %base_pitch : i32, %x : i32, %y : i32, %stored_val : vector<4xi32>) {
  // CHECK: [[PTR:%.*]] = ptrtoint ptr %0 to i64
  // CHECK-NEXT: call void @llvm.genx.GenISA.LSC2DBlockWrite.v4i32(i64 [[PTR]], i32 %1, i32 %2, i32 %3, i32 %4, i32 %5, i32 32, i32 4, i32 1, i32 1, i1 false, i1 false, <4 x i32> %6)
  genx.matrix.2Dblockstore %ptr, %base_width, %base_height, %base_pitch, %x, %y, %stored_val {elem_size_in_bits=32:i32, tile_width=4:i32, tile_height=1:i32, v_blocks=1:i32, transpose=false, vnni_transform=false} : (!llvm.ptr<i32>, i32, i32, i32, i32, i32, vector<4xi32>)
  llvm.return
}
