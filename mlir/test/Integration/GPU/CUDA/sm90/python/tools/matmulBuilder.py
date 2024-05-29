import numpy as np
from mlir import ir
from mlir.dialects import arith
from mlir.dialects import func
from mlir.dialects import gpu
from mlir.dialects import memref
from mlir.dialects import nvgpu
from mlir.dialects import nvvm
from mlir.dialects import llvm
from mlir.dialects import builtin
from mlir.dialects import scf
from mlir.dialects import vector
from mlir.extras import types as T

TMA_LAST_DIM_F16 = 64  # 128B flaot16
WARP_SIZE = 32
WARP_GROUP_SIZE = WARP_SIZE * 4

PRODUCER_REGISTER_SIZE = 40
CONSUMER_REGISTER_SIZE = 232

PRODUCER_PRIMARY_THREAD = 128
CONSUMER_PRIMARY_THREAD = 0

# C++ uses this value to understand whether it's dynamic or not.
MLIR_DYNAMIC = -9223372036854775808

DEBUG = False


class TmaDescriptorBuilder:
    """A class that builds a TMA descriptor."""

    def __init__(self, swizzle, l2promo, oob, interleave, tma_box_shape, memref_ty):
        self.swizzle = swizzle  # mlir.nvgpu.TensorMapSwizzleKind
        self.l2promo = l2promo  # mlir.nvgpu.TensorMapL2PromoKind
        self.oob = oob  # mlir.nvgpu.TensorMapOOBKind
        self.interleave = interleave  # mlir.nvgpu.TensorMapInterleaveKind
        self.tma_box_shape = tma_box_shape
        self.memref_ty = memref_ty  # MemRefType

    @property
    def tensormap_descriptor_ty(self):
        """Returns a tensormap descriptor type."""
        tensorMemrefType = ir.MemRefType.get(
            self.tma_box_shape,
            self.memref_ty.element_type,
            memory_space=ir.Attribute.parse("3"),
        )
        return nvgpu.TensorMapDescriptorType.get(
            tensorMemrefType,
            self.swizzle,
            self.l2promo,
            self.oob,
            self.interleave,
        )

    def tma_descriptor_op(self, device_ptr):
        """Returns a tensormap descriptor op."""
        tma_descriptor_ty = self.tensormap_descriptor_ty
        device_unranked_memref = memref.CastOp(
            ir.UnrankedMemRefType.get(
                self.memref_ty.element_type, self.memref_ty.memory_space
            ),
            device_ptr,
        )
        tma_descriptor_op = nvgpu.TmaCreateDescriptorOp(
            tma_descriptor_ty, device_unranked_memref, map(c, self.tma_box_shape)
        )
        return tma_descriptor_op.result


def debug_print(fmt, *args, predicate=None, threadNumber=-1, forcePrint=False):
    if not DEBUG and not forcePrint:
        return
    type_formats = []
    for arg in args:
        ty_format = None
        if ir.IndexType.isinstance(arg.type):
            ty_format = "%llu"
        if ir.IntegerType.isinstance(arg.type):
            width = ir.IntegerType(arg.type).width
            if width == 64:
                ty_format = "%llu"
            elif width == 32:
                ty_format = "%d"
            elif width == 1:
                ty_format = "%i"
        if ir.F32Type.isinstance(arg.type):
            ty_format = "%f"
        if ty_format is None:
            raise NotImplementedError(arg.type)
        type_formats.append(ty_format)
    if threadNumber != -1:
        tidx = gpu.thread_id(gpu.Dimension.x)
        predicate = arith.cmpi(arith.CmpIPredicate.eq, tidx, c(threadNumber))
        scf.yield_([])
    if_op = scf.IfOp(predicate)
    with ir.InsertionPoint(if_op.then_block):
        gpu.printf(fmt.format(*type_formats) + "\n", args)
        scf.yield_([])


def get_type_size(ty):
    if ir.FloatType.isinstance(ty):
        return ir.FloatType(ty).width // 8
    if ir.IntegerType.isinstance(ty):
        return ir.IntegerType(ty).width // 8
    raise NotImplementedError(ty)


def get_mlir_ty(dtype):
    if dtype == np.float16:
        return T.f16()
    if dtype == np.float32:
        return T.f32()
    if dtype == np.float64:
        return T.f64()
    if dtype == np.int32:
        return T.i32()
    if dtype == np.int64:
        return T.i64()
    raise NotImplementedError(dtype)


def c(value, ty=None):
    ty = T.index() if ty is None else ty
    return arith.constant(ty, value)


def make_kernel_name(
    input_type=np.float16,
    output_type=np.float32,
    M=4096,
    N=4096,
    K=4096,
    BLOCK_M=128,
    BLOCK_N=128,
    BLOCK_K=128,
    num_stages=3,
    use_warp_specialization=False,
):
    kernelName = "warpspecialized" if use_warp_specialization else "multistage"
    return (
        kernelName
        + "_"
        + str(M)
        + "x"
        + str(N)
        + "x"
        + str(K)
        + "_"
        + str(BLOCK_M)
        + "x"
        + str(BLOCK_N)
        + "x"
        + str(BLOCK_K)
        + "_"
        + str(num_stages)
    )


def generate_matmul_ws(
    input_type=np.float16,
    output_type=np.float32,
    M=4096,
    N=4096,
    K=4096,
    BLOCK_M=128,
    BLOCK_N=128,
    BLOCK_K=128,
    num_stages=3,
):
    # Limitaitons for now
    assert input_type == np.float16
    assert output_type == np.float32
    assert BLOCK_M == 128
    assert BLOCK_N == 128
    assert BLOCK_K == 64
    assert M % BLOCK_M == 0
    assert N % BLOCK_N == 0
    assert K % BLOCK_K == 0

    module = ir.Module.create()
    token_ty = ir.Type.parse("!gpu.async.token")
    a_elem_ty = get_mlir_ty(input_type)
    b_elem_ty = get_mlir_ty(input_type)
    c_elem_ty = get_mlir_ty(output_type)
    a_ty = ir.MemRefType.get([M, K], a_elem_ty)
    b_ty = ir.MemRefType.get((K, N), b_elem_ty)
    c_ty = ir.MemRefType.get((M, N), c_elem_ty)
    a_tile_shape = a_tma_shape = (BLOCK_M, TMA_LAST_DIM_F16)
    b_tma_shape = (BLOCK_K, TMA_LAST_DIM_F16)
    b_tile_shape = (BLOCK_K, BLOCK_N)
    txcount = (b_tile_shape[0] * b_tile_shape[1] * get_type_size(a_elem_ty)) + (
        a_tile_shape[0] * a_tile_shape[1] * get_type_size(b_elem_ty)
    )
    smem_space_str = "#gpu.address_space<workgroup>"
    smem_space = ir.Attribute.parse(smem_space_str)
    mbar_ty = ir.Type.parse(
        "!nvgpu.mbarrier.group<memorySpace = "
        + str(smem_space)
        + ", num_barriers = "
        + str(num_stages)
        + ">"
    )
    acc_ty = ir.Type.parse(
        "!nvgpu.warpgroup.accumulator<fragmented=vector<"
        + str(BLOCK_M)
        + "x"
        + str(BLOCK_N)
        + "x"
        + str(c_elem_ty)
        + ">>"
    )
    a_wgmma_ty = ir.Type.parse(
        "!nvgpu.warpgroup.descriptor<tensor=memref<"
        + str(BLOCK_M)
        + "x"
        + str(BLOCK_K)
        + "x"
        + str(a_elem_ty)
        + ", "
        + smem_space_str
        + ">>"
    )
    b_wgmma_ty = ir.Type.parse(
        "!nvgpu.warpgroup.descriptor<tensor=memref<"
        + str(BLOCK_K)
        + "x"
        + str(BLOCK_N)
        + "x"
        + str(a_elem_ty)
        + ", "
        + smem_space_str
        + ">>"
    )
    kernelName = make_kernel_name(
        input_type, output_type, M, N, K, BLOCK_M, BLOCK_N, BLOCK_K, num_stages, True
    )
    with ir.InsertionPoint(module.body):
        fop = func.FuncOp(kernelName, ([a_ty, b_ty, c_ty], []))
        with ir.InsertionPoint(fop.add_entry_block()):
            a_host = fop.arguments[0]
            b_host = fop.arguments[1]
            c_host = fop.arguments[2]
            lhs_tile_bytes = BLOCK_M * BLOCK_K * get_type_size(a_elem_ty)
            rhs_tile_bytes = BLOCK_N * BLOCK_K * get_type_size(b_elem_ty)
            smem_size_input = (lhs_tile_bytes + rhs_tile_bytes) * num_stages
            smem_size_output = BLOCK_M * BLOCK_N * get_type_size(c_elem_ty)
            smem_size = max(smem_size_input, smem_size_output)

            # Step 1. Allocate device memory and memcpy
            t1 = gpu.wait(token_ty, [])
            a_device, t2 = gpu.alloc(a_ty, token_ty, [t1], [], [])
            b_device, t3 = gpu.alloc(b_ty, token_ty, [t2], [], [])
            c_device, t4 = gpu.alloc(c_ty, token_ty, [t3], [], [])
            t5 = gpu.memcpy(token_ty, [t4], a_device, a_host)
            t6 = gpu.memcpy(token_ty, [t5], b_device, b_host)
            t7 = gpu.wait(token_ty, [t6])

            # Step 2. Create TMA Descriptors
            a_tma_desc = TmaDescriptorBuilder(
                nvgpu.TensorMapSwizzleKind.SWIZZLE_128B,
                nvgpu.TensorMapL2PromoKind.L2PROMO_NONE,
                nvgpu.TensorMapOOBKind.OOB_ZERO,
                nvgpu.TensorMapInterleaveKind.INTERLEAVE_NONE,
                a_tma_shape,
                a_ty,
            )

            b_tma_desc = TmaDescriptorBuilder(
                nvgpu.TensorMapSwizzleKind.SWIZZLE_128B,
                nvgpu.TensorMapL2PromoKind.L2PROMO_NONE,
                nvgpu.TensorMapOOBKind.OOB_ZERO,
                nvgpu.TensorMapInterleaveKind.INTERLEAVE_NONE,
                b_tma_shape,
                b_ty,
            )

            a_tma_desc_op = a_tma_desc.tma_descriptor_op(a_device)
            b_tma_desc_op = b_tma_desc.tma_descriptor_op(b_device)

            # Step 3. Launch Kernel with 2 Warpgroups : 1 Producer, 1 Consumer
            cta_m = M // BLOCK_M
            cta_n = N // BLOCK_N
            assert M % BLOCK_M == 0 and N % BLOCK_N == 0
            grid = (cta_m, cta_n, 1)
            block = (WARP_GROUP_SIZE * 2, 1, 1)
            launch_op = gpu.LaunchOp(
                token_ty,
                [t7],
                *map(c, grid),
                *map(c, block),
                dynamicSharedMemorySize=c(smem_size, ty=T.i32()),
            )
            launch_op.body.blocks.append(*([T.index()] * 12))
            with ir.InsertionPoint(launch_op.body.blocks[0]):
                # GPU Step 0. This is need for vectorized ld/st
                memref.assume_alignment(c_device, 16)
                dynamic_smem = gpu.dynamic_shared_memory(
                    ir.MemRefType.get((MLIR_DYNAMIC,), T.i8(), memory_space=smem_space)
                )
                ticks = c(10000000)

                # GPU Step 1. Bootstrapping: find the primary thread, warps, warp groups and etc.
                tidx = gpu.thread_id(gpu.Dimension.x)
                wgPrimaryThread = arith.cmpi(
                    arith.CmpIPredicate.eq, arith.remui(tidx, c(WARP_GROUP_SIZE)), c(0)
                )
                warp_id = arith.divui(tidx, c(32))
                warpgroup_id = arith.divui(warp_id, c(4))
                is_producer = arith.cmpi(
                    arith.CmpIPredicate.eq,
                    warpgroup_id,
                    c(1 if PRODUCER_PRIMARY_THREAD == 128 else 0),
                )
                is_consumer = arith.cmpi(
                    arith.CmpIPredicate.eq,
                    warpgroup_id,
                    c(0 if CONSUMER_PRIMARY_THREAD == 0 else 1),
                )
                producerPrimaryThread = arith.cmpi(
                    arith.CmpIPredicate.eq, tidx, c(PRODUCER_PRIMARY_THREAD)
                )
                consumerPrimaryThread = arith.cmpi(
                    arith.CmpIPredicate.eq, tidx, c(CONSUMER_PRIMARY_THREAD)
                )
                bidx = gpu.block_id(gpu.Dimension.x)
                bidy = gpu.block_id(gpu.Dimension.y)
                dimX = arith.muli(bidx, c(BLOCK_M))
                dimY = arith.muli(bidy, c(BLOCK_N))

                # GPU Step 2. Initialize mbarrier groups
                mbarTMA = nvgpu.mbarrier_create(mbar_ty)
                mbarDONE = nvgpu.mbarrier_create(mbar_ty)
                for i in range(num_stages):
                    nvgpu.mbarrier_init(mbarTMA, c(1), c(i), predicate=wgPrimaryThread)
                    nvgpu.mbarrier_init(mbarDONE, c(1), c(i), predicate=wgPrimaryThread)
                gpu.barrier()

                # GPU Step 3. Prefetch TMA descriptors
                nvgpu.tma_prefetch_descriptor(a_tma_desc_op, predicate=wgPrimaryThread)
                nvgpu.tma_prefetch_descriptor(b_tma_desc_op, predicate=wgPrimaryThread)

                ns = num_stages if num_stages == 1 else num_stages - 1
                # GPU Step 5. Producer Warpgroup (TMA Warpgroup)
                with ir.InsertionPoint(scf.IfOp(is_producer).then_block):
                    # Step 5.1. Reduce register size
                    nvvm.setmaxregister(
                        PRODUCER_REGISTER_SIZE, nvvm.SetMaxRegisterAction.decrease
                    )

                    # Step 5.2. TMA Main Loop
                    for_op = scf.ForOp(
                        c(0), c(K // BLOCK_K), c(1), [arith.constant(T.bool(), 1)]
                    )
                    with ir.InsertionPoint(for_op.body):
                        phaseParity = for_op.inner_iter_args[0]
                        iv = for_op.induction_variable
                        stage = arith.remui(iv, c(num_stages))

                        # Step 5.2.1. Wait mbarDONE
                        debug_print(
                            "[prod] iv={}  | mbarDONE[{}] try_wait  phase={}",
                            iv,
                            stage,
                            phaseParity,
                            predicate=producerPrimaryThread,
                        )
                        nvgpu.MBarrierTryWaitParityOp(
                            mbarDONE, phaseParity, ticks, mbarId=stage
                        )
                        debug_print(
                            "[prod] iv={}  | mbarDONE[{}] try_wait  phase={} [done]",
                            iv,
                            stage,
                            phaseParity,
                            predicate=producerPrimaryThread,
                        )
                        p = arith.cmpi(arith.CmpIPredicate.eq, stage, c(num_stages - 1))
                        phaseParity = arith.select(
                            p,
                            arith.xori(phaseParity, arith.constant(T.bool(), 1)),
                            phaseParity,
                        )

                        # Step 5.2.2. Load TMA
                        a_offset = arith.muli(stage, c(lhs_tile_bytes))
                        a_tma_slice = memref.view(
                            ir.MemRefType.get(
                                a_tma_shape, a_elem_ty, memory_space=smem_space
                            ),
                            dynamic_smem,
                            a_offset,
                            [],
                        )
                        b_offset = arith.addi(
                            arith.muli(stage, c(rhs_tile_bytes)),
                            c(lhs_tile_bytes * num_stages),
                        )
                        b_tma_slice_1 = memref.view(
                            ir.MemRefType.get(
                                b_tma_shape, b_elem_ty, memory_space=smem_space
                            ),
                            dynamic_smem,
                            b_offset,
                            [],
                        )
                        b_offset2 = arith.addi(
                            b_offset,
                            c(BLOCK_K * TMA_LAST_DIM_F16 * get_type_size(b_elem_ty)),
                        )
                        b_tma_slice_2 = memref.view(
                            ir.MemRefType.get(
                                b_tma_shape, b_elem_ty, memory_space=smem_space
                            ),
                            dynamic_smem,
                            b_offset2,
                            [],
                        )
                        debug_print(
                            "[prod] a_offset={} b_offset={} b_offset2={}",
                            a_offset,
                            b_offset,
                            b_offset2,
                            predicate=producerPrimaryThread,
                        )
                        coord = arith.muli(c(64), iv)
                        nvgpu.TmaAsyncLoadOp(
                            a_tma_slice,
                            mbarTMA,
                            a_tma_desc_op,
                            coordinates=[coord, dimX],
                            mbarId=stage,
                            predicate=producerPrimaryThread,
                        )
                        nvgpu.TmaAsyncLoadOp(
                            b_tma_slice_1,
                            mbarTMA,
                            b_tma_desc_op,
                            coordinates=[dimY, coord],
                            mbarId=stage,
                            predicate=producerPrimaryThread,
                        )
                        dimY2 = arith.addi(dimY, c(64))
                        nvgpu.TmaAsyncLoadOp(
                            b_tma_slice_2,
                            mbarTMA,
                            b_tma_desc_op,
                            coordinates=[dimY2, coord],
                            mbarId=stage,
                            predicate=producerPrimaryThread,
                        )

                        # Step 5.2.3. Arrive mbarTMA
                        debug_print(
                            "[prod] iv={}  | mbarTMA[{}] arrive",
                            iv,
                            stage,
                            predicate=producerPrimaryThread,
                        )
                        nvgpu.mbarrier_arrive_expect_tx(
                            mbarTMA, c(txcount), stage, predicate=producerPrimaryThread
                        )
                        debug_print(
                            "[prod] iv={}  | mbarTMA[{}] arrive [done]",
                            iv,
                            stage,
                            predicate=producerPrimaryThread,
                        )
                        scf.yield_([phaseParity])
                    scf.yield_([])

                # GPU Step 6. Consumer Warpgroup (MMA Warpgroup)
                if_op = scf.IfOp(is_consumer)
                with ir.InsertionPoint(if_op.then_block):
                    # Step 6.1. Increase register size
                    nvvm.setmaxregister(
                        CONSUMER_REGISTER_SIZE, nvvm.SetMaxRegisterAction.increase
                    )

                    # GPU Step 6.2. Initialize MMA registers
                    acc = nvgpu.warpgroup_mma_init_accumulator(acc_ty)

                    # Step 6.3. MMA Main Loop
                    for_op = scf.ForOp(
                        c(0), c(K // BLOCK_K), c(1), [acc, arith.constant(T.bool(), 0)]
                    )
                    with ir.InsertionPoint(for_op.body):
                        # Step 6.3.1. Wait mbar1
                        phaseParity = for_op.inner_iter_args[1]
                        iv = for_op.induction_variable
                        stage = arith.remui(iv, c(num_stages))
                        debug_print(
                            "[cons] iv={}  | mbarTMA[{}] try_wait   phase={}",
                            iv,
                            stage,
                            phaseParity,
                            predicate=consumerPrimaryThread,
                        )
                        nvgpu.MBarrierTryWaitParityOp(
                            mbarTMA, phaseParity, ticks, mbarId=stage
                        )
                        debug_print(
                            "[cons] iv={}  | mbarTMA[{}] try_wait   phase={} [done]",
                            iv,
                            stage,
                            phaseParity,
                            predicate=consumerPrimaryThread,
                        )

                        # Step 6.3.2. Create WGMMA Descriptors
                        a_offset = arith.muli(stage, c(lhs_tile_bytes))
                        a_tile_slice = memref.view(
                            ir.MemRefType.get(
                                a_tile_shape, a_elem_ty, memory_space=smem_space
                            ),
                            dynamic_smem,
                            a_offset,
                            [],
                        )
                        b_offset = arith.addi(
                            arith.muli(stage, c(rhs_tile_bytes)),
                            c(lhs_tile_bytes * num_stages),
                        )
                        b_tile_slice = memref.view(
                            ir.MemRefType.get(
                                b_tile_shape, b_elem_ty, memory_space=smem_space
                            ),
                            dynamic_smem,
                            b_offset,
                            [],
                        )
                        debug_print(
                            "[cons] a_offset={} b_offset={}",
                            a_offset,
                            b_offset,
                            predicate=consumerPrimaryThread,
                        )
                        da = nvgpu.WarpgroupGenerateDescriptorOp(
                            a_wgmma_ty, a_tile_slice, a_tma_desc_op
                        )
                        db = nvgpu.WarpgroupGenerateDescriptorOp(
                            b_wgmma_ty, b_tile_slice, b_tma_desc_op
                        )

                        # Step 6.3.3. MMA
                        carry_acc = for_op.inner_iter_args[0]
                        new_acc = nvgpu.WarpgroupMmaOp(
                            acc.type, da, db, carry_acc, transposeB=True
                        )

                        # Step 6.3.4. Arrive mbarDONE
                        if num_stages == 1:
                            p_arrive = consumerPrimaryThread
                        else:
                            p1 = arith.cmpi(arith.CmpIPredicate.sgt, iv, c(0))
                            p_arrive = arith.andi(consumerPrimaryThread, p1)
                        with ir.InsertionPoint(scf.IfOp(p_arrive).then_block):
                            p = arith.cmpi(arith.CmpIPredicate.eq, stage, c(0))
                            barId = arith.select(
                                p, c(num_stages - 1), arith.subi(stage, c(1))
                            )
                            debug_print(
                                "[cons] iv={}  | mbarDONE[{}] arrive ",
                                iv,
                                barId,
                                predicate=consumerPrimaryThread,
                            )
                            nvgpu.mbarrier_arrive(
                                ir.Type.parse("!nvgpu.mbarrier.token"), mbarDONE, barId
                            )
                            debug_print(
                                "[cons] iv={}  | mbarDONE[{}] arrive [done]",
                                iv,
                                barId,
                                predicate=consumerPrimaryThread,
                            )
                            scf.yield_([])

                        p = arith.cmpi(arith.CmpIPredicate.eq, stage, c(num_stages - 1))
                        phaseParity = arith.select(
                            p,
                            arith.xori(phaseParity, arith.constant(T.bool(), 1)),
                            phaseParity,
                        )

                        # Step 6.3.5. Yield
                        scf.yield_([new_acc, phaseParity])

                    # Step 6.3. Wait All WGMMA
                    nvvm.WgmmaWaitGroupSyncOp(0)

                    with ir.InsertionPoint(scf.IfOp(consumerPrimaryThread).then_block):
                        barId = c((K // BLOCK_K) % num_stages)
                        nvgpu.mbarrier_arrive(
                            ir.Type.parse("!nvgpu.mbarrier.token"), mbarDONE, barId
                        )
                        scf.yield_([])

                    # Step 6.4. Epilogue (registers --> shared memory)
                    acc_smem_ty = ir.MemRefType.get(
                        (BLOCK_M, BLOCK_N), c_elem_ty, memory_space=smem_space
                    )
                    acc_smem = memref.view(acc_smem_ty, dynamic_smem, c(0), [])
                    debug_print("[cons]  | Storing", predicate=consumerPrimaryThread)
                    nvgpu.WarpgroupMmaStoreOp(for_op.results[0], acc_smem)
                    scf.yield_([])
                gpu.barrier()

                # GPU Step 9. Epilogue (shared memory --> global memory)
                fd = ir.MemRefType.get(
                    [BLOCK_M * BLOCK_N], c_elem_ty, memory_space=smem_space
                )
                collapsed_smem = memref.view(fd, dynamic_smem, c(0), [])
                rty = ir.MemRefType.get(
                    (BLOCK_M, BLOCK_N),
                    c_elem_ty,
                    ir.Attribute.parse("strided<[" + str(N) + ", 1], offset: ?>"),
                )
                c_device_per_block = memref.SubViewOp(
                    rty,
                    c_device,
                    [dimX, dimY],
                    [],
                    [],
                    [MLIR_DYNAMIC, MLIR_DYNAMIC],
                    [BLOCK_M, BLOCK_N],
                    [1, 1],
                )
                vlen = 1
                for_op = scf.ForOp(
                    tidx, c(BLOCK_M * BLOCK_N), c(vlen * WARP_GROUP_SIZE * 2)
                )
                with ir.InsertionPoint(for_op.body):
                    x = arith.divui(for_op.induction_variable, c(BLOCK_M))
                    y = arith.remui(for_op.induction_variable, c(BLOCK_N))
                    vdata = vector.load(
                        ir.VectorType.get((vlen,), c_elem_ty),
                        collapsed_smem,
                        [for_op.induction_variable],
                    )
                    vector.store(vdata, c_device_per_block, [x, y])
                    scf.yield_([])

                gpu.terminator()

            # Step 4. Copy back to host
            t8 = gpu.wait(token_ty, [launch_op])
            t9 = gpu.memcpy(token_ty, [t8], c_host, c_device)
            gpu.dealloc(token_ty, [t8], a_device)
            gpu.dealloc(token_ty, [t8], b_device)
            gpu.wait(token_ty, [t9])
            gpu.dealloc(token_ty, [t8], c_device)
            func.ReturnOp([])

    fop.attributes["llvm.emit_c_interface"] = ir.UnitAttr.get()
    module.operation.verify()
    return module


def generate_matmul_multistage(
    input_type=np.float16,
    output_type=np.float32,
    M=4096,
    N=4096,
    K=4096,
    BLOCK_M=128,
    BLOCK_N=128,
    BLOCK_K=64,
    num_stages=3,
):
    # Limitaitons for now
    assert input_type == np.float16
    assert output_type == np.float32
    assert BLOCK_M == 128
    assert BLOCK_N == 128
    assert BLOCK_K == 64
    assert M % BLOCK_M == 0
    assert N % BLOCK_N == 0
    assert K % BLOCK_K == 0

    module = ir.Module.create()
    token_ty = ir.Type.parse("!gpu.async.token")
    a_elem_ty = get_mlir_ty(input_type)
    b_elem_ty = get_mlir_ty(input_type)
    c_elem_ty = get_mlir_ty(output_type)
    a_ty = ir.MemRefType.get([M, K], a_elem_ty)
    b_ty = ir.MemRefType.get((K, N), b_elem_ty)
    c_ty = ir.MemRefType.get((M, N), c_elem_ty)
    a_tile_shape = a_tma_shape = (BLOCK_M, TMA_LAST_DIM_F16)
    b_tma_shape = (BLOCK_K, TMA_LAST_DIM_F16)
    b_tile_shape = (BLOCK_K, BLOCK_N)
    txcount = (b_tile_shape[0] * b_tile_shape[1] * get_type_size(a_elem_ty)) + (
        a_tile_shape[0] * a_tile_shape[1] * get_type_size(b_elem_ty)
    )
    smem_space_str = "#gpu.address_space<workgroup>"
    smem_space = ir.Attribute.parse(smem_space_str)
    mbar_ty = ir.Type.parse(
        "!nvgpu.mbarrier.group<memorySpace = "
        + str(smem_space)
        + ", num_barriers = "
        + str(num_stages)
        + ">"
    )
    acc_ty = ir.Type.parse(
        "!nvgpu.warpgroup.accumulator<fragmented=vector<"
        + str(BLOCK_M)
        + "x"
        + str(BLOCK_N)
        + "x"
        + str(c_elem_ty)
        + ">>"
    )
    a_wgmma_ty = ir.Type.parse(
        "!nvgpu.warpgroup.descriptor<tensor=memref<"
        + str(BLOCK_M)
        + "x"
        + str(BLOCK_K)
        + "x"
        + str(a_elem_ty)
        + ", "
        + smem_space_str
        + ">>"
    )
    b_wgmma_ty = ir.Type.parse(
        "!nvgpu.warpgroup.descriptor<tensor=memref<"
        + str(BLOCK_K)
        + "x"
        + str(BLOCK_N)
        + "x"
        + str(a_elem_ty)
        + ", "
        + smem_space_str
        + ">>"
    )

    with ir.InsertionPoint(module.body):
        kernelName = make_kernel_name(
            input_type,
            output_type,
            M,
            N,
            K,
            BLOCK_M,
            BLOCK_N,
            BLOCK_K,
            num_stages,
            False,
        )
        fop = func.FuncOp(kernelName, ([a_ty, b_ty, c_ty], []))
        with ir.InsertionPoint(fop.add_entry_block()):
            a_host = fop.arguments[0]
            b_host = fop.arguments[1]
            c_host = fop.arguments[2]
            lhs_tile_bytes = BLOCK_M * BLOCK_K * get_type_size(a_elem_ty)
            rhs_tile_bytes = BLOCK_N * BLOCK_K * get_type_size(b_elem_ty)
            smem_size_input = (lhs_tile_bytes + rhs_tile_bytes) * num_stages
            smem_size_output = BLOCK_M * BLOCK_N * get_type_size(c_elem_ty)
            smem_size = max(smem_size_input, smem_size_output)

            # Step 1. Allocate device memory and memcpy
            t1 = gpu.wait(token_ty, [])
            a_device, t2 = gpu.alloc(a_ty, token_ty, [t1], [], [])
            b_device, t3 = gpu.alloc(b_ty, token_ty, [t2], [], [])
            c_device, t4 = gpu.alloc(c_ty, token_ty, [t3], [], [])
            t5 = gpu.memcpy(token_ty, [t4], a_device, a_host)
            t6 = gpu.memcpy(token_ty, [t5], b_device, b_host)
            t7 = gpu.wait(token_ty, [t6])

            # Step 2. Create TMA Descriptors
            a_tma_desc = TmaDescriptorBuilder(
                nvgpu.TensorMapSwizzleKind.SWIZZLE_128B,
                nvgpu.TensorMapL2PromoKind.L2PROMO_NONE,
                nvgpu.TensorMapOOBKind.OOB_ZERO,
                nvgpu.TensorMapInterleaveKind.INTERLEAVE_NONE,
                a_tma_shape,
                a_ty,
            )

            b_tma_desc = TmaDescriptorBuilder(
                nvgpu.TensorMapSwizzleKind.SWIZZLE_128B,
                nvgpu.TensorMapL2PromoKind.L2PROMO_NONE,
                nvgpu.TensorMapOOBKind.OOB_ZERO,
                nvgpu.TensorMapInterleaveKind.INTERLEAVE_NONE,
                b_tma_shape,
                b_ty,
            )

            a_tma_desc_op = a_tma_desc.tma_descriptor_op(a_device)
            b_tma_desc_op = b_tma_desc.tma_descriptor_op(b_device)

            # Step 3. Launch Kernel with 1 Warpgroup
            cta_m = M // BLOCK_M
            cta_n = N // BLOCK_N
            assert M % BLOCK_M == 0 and N % BLOCK_N == 0
            grid = (cta_m, cta_n, 1)
            block = (WARP_GROUP_SIZE, 1, 1)
            launch_op = gpu.LaunchOp(
                token_ty,
                [t7],
                *map(c, grid),
                *map(c, block),
                dynamicSharedMemorySize=c(smem_size, ty=T.i32()),
            )
            launch_op.body.blocks.append(*([T.index()] * 12))
            with ir.InsertionPoint(launch_op.body.blocks[0]):
                # GPU Step 0. Bootstrapping
                memref.assume_alignment(c_device, 16)
                dynamic_smem = gpu.dynamic_shared_memory(
                    ir.MemRefType.get((MLIR_DYNAMIC,), T.i8(), memory_space=smem_space)
                )
                ticks = c(10000000)
                tidx = gpu.thread_id(gpu.Dimension.x)
                primaryThread = arith.cmpi(arith.CmpIPredicate.eq, tidx, c(0))
                warpId = arith.divui(tidx, c(32))
                bidx = gpu.block_id(gpu.Dimension.x)
                bidy = gpu.block_id(gpu.Dimension.y)
                dimX = arith.muli(bidx, c(BLOCK_M))
                dimY = arith.muli(bidy, c(BLOCK_N))

                # GPU Step 1. Initialize mbarrier groups
                mbarTMA = nvgpu.mbarrier_create(mbar_ty)
                for i in range(num_stages):
                    nvgpu.mbarrier_init(mbarTMA, c(1), c(i), predicate=primaryThread)
                gpu.barrier()

                # GPU Step 2. Prefetch TMA descriptors
                nvgpu.tma_prefetch_descriptor(a_tma_desc_op, predicate=primaryThread)
                nvgpu.tma_prefetch_descriptor(b_tma_desc_op, predicate=primaryThread)

                # GPU Step 3. Prologue (global memory --> shared memory)
                ns = num_stages if num_stages == 1 else num_stages - 1
                for_op = scf.ForOp(c(0), c(ns), c(1))
                with ir.InsertionPoint(for_op.body):
                    iv = for_op.induction_variable

                    # Step 3.1. Calculate offsets
                    a_offset = arith.muli(iv, c(lhs_tile_bytes))
                    a_tma_slice = memref.view(
                        ir.MemRefType.get(
                            a_tma_shape, a_elem_ty, memory_space=smem_space
                        ),
                        dynamic_smem,
                        a_offset,
                        [],
                    )
                    b_offset = arith.addi(
                        arith.muli(iv, c(rhs_tile_bytes)),
                        c(lhs_tile_bytes * num_stages),
                    )
                    b_tma_slice_1 = memref.view(
                        ir.MemRefType.get(
                            b_tma_shape, b_elem_ty, memory_space=smem_space
                        ),
                        dynamic_smem,
                        b_offset,
                        [],
                    )
                    b_offset2 = arith.addi(
                        b_offset,
                        c(BLOCK_K * TMA_LAST_DIM_F16 * get_type_size(b_elem_ty)),
                    )
                    b_tma_slice_2 = memref.view(
                        ir.MemRefType.get(
                            b_tma_shape, b_elem_ty, memory_space=smem_space
                        ),
                        dynamic_smem,
                        b_offset2,
                        [],
                    )

                    # Step 3.2. TMA Load
                    coord = arith.muli(c(64), iv)
                    dimY2 = arith.addi(dimY, c(64))
                    debug_print(
                        "[Prologue] TMA Load a_offset={} b_offset={} b_offset2={} @ a=({},{}) b=({},{})",
                        a_offset,
                        b_offset,
                        b_offset2,
                        coord,
                        dimX,
                        dimY,
                        coord,
                        predicate=primaryThread,
                    )
                    nvgpu.TmaAsyncLoadOp(
                        a_tma_slice,
                        mbarTMA,
                        a_tma_desc_op,
                        coordinates=[coord, dimX],
                        mbarId=iv,
                        predicate=primaryThread,
                    )
                    nvgpu.TmaAsyncLoadOp(
                        b_tma_slice_1,
                        mbarTMA,
                        b_tma_desc_op,
                        coordinates=[dimY, coord],
                        mbarId=iv,
                        predicate=primaryThread,
                    )
                    nvgpu.TmaAsyncLoadOp(
                        b_tma_slice_2,
                        mbarTMA,
                        b_tma_desc_op,
                        coordinates=[dimY2, coord],
                        mbarId=iv,
                        predicate=primaryThread,
                    )

                    # Step 3.2. mbarTMA arrive
                    debug_print(
                        "[Prologue] mbarTMA[{}] arrive", iv, predicate=primaryThread
                    )
                    nvgpu.mbarrier_arrive_expect_tx(
                        mbarTMA, c(txcount), iv, predicate=primaryThread
                    )
                    debug_print(
                        "[Prologue] mbarTMA[{}] arrive [done]",
                        iv,
                        predicate=primaryThread,
                    )
                    scf.yield_([])

                # GPU Step 4. Main Loop
                acc = nvgpu.warpgroup_mma_init_accumulator(acc_ty)
                for_op = scf.ForOp(
                    c(0), c(K // BLOCK_K), c(1), [acc, arith.constant(T.bool(), 0)]
                )
                with ir.InsertionPoint(for_op.body):
                    # Step 4.1. Wait mbarTMA
                    phaseParity = for_op.inner_iter_args[1]
                    iv = for_op.induction_variable
                    stage = arith.remui(iv, c(num_stages))
                    debug_print(
                        "[MainLoop] mbarTMA[{}] try_wait   phase={}",
                        stage,
                        phaseParity,
                        predicate=primaryThread,
                    )
                    nvgpu.MBarrierTryWaitParityOp(
                        mbarTMA, phaseParity, ticks, mbarId=stage
                    )
                    debug_print(
                        "[MainLoop] mbarTMA[{}] try_wait   phase={} [done]",
                        stage,
                        phaseParity,
                        predicate=primaryThread,
                    )

                    # Step 4.2. Create WGMMA Descriptors
                    a_offset = arith.muli(stage, c(lhs_tile_bytes))
                    a_tile_slice = memref.view(
                        ir.MemRefType.get(
                            a_tile_shape, a_elem_ty, memory_space=smem_space
                        ),
                        dynamic_smem,
                        a_offset,
                        [],
                    )
                    b_offset = arith.addi(
                        arith.muli(stage, c(rhs_tile_bytes)),
                        c(lhs_tile_bytes * num_stages),
                    )
                    b_tile_slice = memref.view(
                        ir.MemRefType.get(
                            b_tile_shape, b_elem_ty, memory_space=smem_space
                        ),
                        dynamic_smem,
                        b_offset,
                        [],
                    )
                    debug_print(
                        "[MainLoop] iv={} MMA a_offset={} b_offset={}",
                        iv,
                        a_offset,
                        b_offset,
                        predicate=primaryThread,
                    )
                    da = nvgpu.WarpgroupGenerateDescriptorOp(
                        a_wgmma_ty, a_tile_slice, a_tma_desc_op
                    )
                    db = nvgpu.WarpgroupGenerateDescriptorOp(
                        b_wgmma_ty, b_tile_slice, b_tma_desc_op
                    )

                    # Step 4.3. MMA
                    carry_acc = for_op.inner_iter_args[0]
                    new_acc = nvgpu.WarpgroupMmaOp(
                        acc.type, da, db, carry_acc, transposeB=True
                    )
                    if num_stages == 1:
                        nvvm.WgmmaWaitGroupSyncOp(0)

                    # Step 4.4. Load TMA for next stage
                    p1 = arith.cmpi(
                        arith.CmpIPredicate.ult,
                        arith.addi(iv, c(ns)),
                        c(K // BLOCK_K),
                    )
                    p = arith.andi(primaryThread, p1)
                    nextStage = arith.addi(iv, c(ns))
                    nextSlot = arith.remui(nextStage, c(num_stages))
                    a_offset = arith.muli(nextSlot, c(lhs_tile_bytes))

                    debug_print(
                        "[MainLoop] mbarTMA[{}] arrive",
                        nextSlot,
                        predicate=p,
                    )
                    nvgpu.mbarrier_arrive_expect_tx(
                        mbarTMA, c(txcount), nextSlot, predicate=p
                    )
                    debug_print(
                        "[MainLoop] mbarTMA[{}] arrive [done]",
                        nextSlot,
                        predicate=p,
                    )

                    a_tma_slice = memref.view(
                        ir.MemRefType.get(
                            a_tma_shape, a_elem_ty, memory_space=smem_space
                        ),
                        dynamic_smem,
                        a_offset,
                        [],
                    )
                    b_offset = arith.addi(
                        arith.muli(nextSlot, c(rhs_tile_bytes)),
                        c(lhs_tile_bytes * num_stages),
                    )
                    b_tma_slice_1 = memref.view(
                        ir.MemRefType.get(
                            b_tma_shape, b_elem_ty, memory_space=smem_space
                        ),
                        dynamic_smem,
                        b_offset,
                        [],
                    )
                    b_offset2 = arith.addi(
                        b_offset,
                        c(BLOCK_K * TMA_LAST_DIM_F16 * get_type_size(b_elem_ty)),
                    )
                    b_tma_slice_2 = memref.view(
                        ir.MemRefType.get(
                            b_tma_shape, b_elem_ty, memory_space=smem_space
                        ),
                        dynamic_smem,
                        b_offset2,
                        [],
                    )

                    coord = arith.muli(c(64), nextStage)
                    debug_print(
                        "[MainLoop] iv={} TMA Load a_offset={} b_offset={} b_offset2={} @ a=({},{}) b=({},{})",
                        iv,
                        a_offset,
                        b_offset,
                        b_offset2,
                        coord,
                        dimX,
                        dimY,
                        coord,
                        predicate=p,
                    )
                    nvgpu.TmaAsyncLoadOp(
                        a_tma_slice,
                        mbarTMA,
                        a_tma_desc_op,
                        coordinates=[coord, dimX],
                        mbarId=nextSlot,
                        predicate=p,
                    )
                    nvgpu.TmaAsyncLoadOp(
                        b_tma_slice_1,
                        mbarTMA,
                        b_tma_desc_op,
                        coordinates=[dimY, coord],
                        mbarId=nextSlot,
                        predicate=p,
                    )
                    dimY2 = arith.addi(dimY, c(64))
                    nvgpu.TmaAsyncLoadOp(
                        b_tma_slice_2,
                        mbarTMA,
                        b_tma_desc_op,
                        coordinates=[dimY2, coord],
                        mbarId=nextSlot,
                        predicate=p,
                    )
                    # Step 4.5. Change the phaseParity
                    p = arith.cmpi(arith.CmpIPredicate.eq, stage, c(num_stages - 1))
                    phaseParity = arith.select(
                        p,
                        arith.xori(phaseParity, arith.constant(T.bool(), 1)),
                        phaseParity,
                    )

                    # Step 4.5. Yield
                    scf.yield_([new_acc, phaseParity])

                # Step 5. Wait All WGMMA groups
                nvvm.WgmmaWaitGroupSyncOp(0)

                # Step 6. Epilogue (registers --> shared memory)
                acc_smem_ty = ir.MemRefType.get(
                    (BLOCK_M, BLOCK_N), c_elem_ty, memory_space=smem_space
                )
                acc_smem = memref.view(acc_smem_ty, dynamic_smem, c(0), [])
                debug_print("Storing", predicate=primaryThread)
                nvgpu.WarpgroupMmaStoreOp(for_op.results[0], acc_smem)
                gpu.barrier()

                # GPU Step 7. Epilogue (shared memory --> global memory)
                fd = ir.MemRefType.get(
                    [BLOCK_M * BLOCK_N], c_elem_ty, memory_space=smem_space
                )
                collapsed_smem = memref.view(fd, dynamic_smem, c(0), [])
                rty = ir.MemRefType.get(
                    (BLOCK_M, BLOCK_N),
                    c_elem_ty,
                    ir.Attribute.parse("strided<[" + str(N) + ", 1], offset: ?>"),
                )
                c_device_per_block = memref.SubViewOp(
                    rty,
                    c_device,
                    [dimX, dimY],
                    [],
                    [],
                    [MLIR_DYNAMIC, MLIR_DYNAMIC],
                    [BLOCK_M, BLOCK_N],
                    [1, 1],
                )
                vlen = 1
                for_op = scf.ForOp(
                    tidx, c(BLOCK_M * BLOCK_N), c(vlen * WARP_GROUP_SIZE)
                )
                with ir.InsertionPoint(for_op.body):
                    x = arith.divui(for_op.induction_variable, c(BLOCK_M))
                    y = arith.remui(for_op.induction_variable, c(BLOCK_N))
                    vdata = vector.load(
                        ir.VectorType.get((vlen,), c_elem_ty),
                        collapsed_smem,
                        [for_op.induction_variable],
                    )
                    vector.store(vdata, c_device_per_block, [x, y])
                    scf.yield_([])

                gpu.terminator()

            # Step 4. Copy back to host
            t8 = gpu.wait(token_ty, [launch_op])
            t9 = gpu.memcpy(token_ty, [t8], c_host, c_device)
            gpu.dealloc(token_ty, [t8], a_device)
            gpu.dealloc(token_ty, [t8], b_device)
            gpu.wait(token_ty, [t9])
            gpu.dealloc(token_ty, [t8], c_device)
            func.ReturnOp([])

    fop.attributes["llvm.emit_c_interface"] = ir.UnitAttr.get()
    module.operation.verify()
    return module
