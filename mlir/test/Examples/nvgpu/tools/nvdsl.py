from enum import Enum
import functools, sys, ctypes, os, errno
import numpy as np
from functools import partialmethod
from mlir import ir
from mlir.dialects import arith, func, gpu, memref, nvgpu
from mlir.extras import types as T
from mlir import runtime as rt
from tools import nvgpucompiler

DEBUG = True
MLIR_DYNAMIC = -9223372036854775808


def const(value: int, ty=None):
    ty = T.index() if ty is None else ty
    if isinstance(value, ir.Value) and (
        value.type.isinstance(value.type) or T.bool().isinstance(value.type)
    ):
        return value
    return arith.constant(ty, value)


def get_type_size(ty):
    if ir.MemRefType.isinstance(ty):
        size = get_type_size(ty.element_type)
        for sz in ty.shape:
            size *= sz
        return size
    if ir.FloatType.isinstance(ty):
        return ir.FloatType(ty).width // 8
    if ir.IntegerType.isinstance(ty):
        return ir.IntegerType(ty).width // 8
    raise NotImplementedError(ty)


def get_mlir_func_obj_ty(inputArgs):
    args = []
    c_int_p = ctypes.c_int * 1
    c_float_p = ctypes.c_float * 1
    c_bool_p = ctypes.c_bool * 1
    for arg in inputArgs:
        if isinstance(arg, bool):
            args.append(c_bool_p(arg))
        elif isinstance(arg, int):
            args.append(c_int_p(arg))
        elif isinstance(arg, float):
            args.append(c_float_p(arg))
        elif isinstance(arg, np.ndarray):
            args.append(
                ctypes.pointer(ctypes.pointer(rt.get_ranked_memref_descriptor(arg)))
            )
        else:
            raise NotImplementedError(arg)
    return args


class Mbarriers:
    def __init__(self, number_of_barriers=1):
        self.mbar_ty = ir.Type.parse(
            "!nvgpu.mbarrier.group<memorySpace=#gpu.address_space<workgroup>, num_barriers = "
            + str(number_of_barriers)
            + ">"
        )
        self.mbar_group_op = nvgpu.mbarrier_create(self.mbar_ty)
        self.number_of_barriers = number_of_barriers

    def __getitem__(self, key):
        self.id_op = const(key)
        return self

    def init(self, count: int, predicate=None):
        count_op = const(count)
        if predicate is None:
            nvgpu.mbarrier_init(self.mbar_group_op, count_op, self.id_op)
        else:
            nvgpu.mbarrier_init(
                self.mbar_group_op, count_op, self.id_op, predicate=predicate
            )

    def arrive(self, txcount: int = 0, predicate=None):
        if txcount != 0:
            txcount_op = const(txcount)
            nvgpu.mbarrier_arrive_expect_tx(
                self.mbar_group_op, txcount_op, self.id_op, predicate=predicate
            )
        else:
            nvgpu.mbarrier_arrive(self.mbar_group_op, self.id_op, predicate=predicate)

    def try_wait(self, phase: bool = False, ticks: int = 10000000):
        ticks_op = const(ticks)
        phase_op = const(phase, T.bool())
        nvgpu.MBarrierTryWaitParityOp(
            self.mbar_group_op,
            phase_op,
            ticks_op,
            mbarId=self.id_op,
        )


class TMA:
    """A class that builds a TMA descriptor."""

    def __init__(
        self,
        shape,
        memref_ty,
        swizzle=nvgpu.TensorMapSwizzleKind.SWIZZLE_NONE,
        l2promo=nvgpu.TensorMapL2PromoKind.L2PROMO_NONE,
        oob=nvgpu.TensorMapOOBKind.OOB_ZERO,
        interleave=nvgpu.TensorMapInterleaveKind.INTERLEAVE_NONE,
    ):
        self.swizzle = swizzle  # mlir.nvgpu.TensorMapSwizzleKind
        self.l2promo = l2promo  # mlir.nvgpu.TensorMapL2PromoKind
        self.oob = oob  # mlir.nvgpu.TensorMapOOBKind
        self.interleave = interleave  # mlir.nvgpu.TensorMapInterleaveKind
        self.shape = shape
        self.memref_ty = memref_ty  # MemRefType
        self.lastDim = 64
        self.requiredLoad = 1
        self.tma_shape = shape
        self.tma_memref = ir.MemRefType.get(shape, memref_ty.element_type)

    @property
    def tensormap_descriptor_ty(self):
        """Returns a tensormap descriptor type."""
        memref_str = f"memref<{self.tma_shape[0]}x{self.tma_shape[1]}x{self.memref_ty.element_type}, 3>"
        parse_str = f"!nvgpu.tensormap.descriptor<tensor = {memref_str},\
                                              swizzle = {self.swizzle},\
                                              l2promo = {self.l2promo},\
                                              oob = {self.oob},\
                                              interleave = {self.interleave}>"

        return ir.Type.parse(parse_str)

    def create_descriptor(self, device_ptr):
        tma_descriptor_ty = self.tensormap_descriptor_ty
        device_unranked_memref = memref.CastOp(
            ir.UnrankedMemRefType.get(
                self.memref_ty.element_type, self.memref_ty.memory_space
            ),
            device_ptr,
        )
        self.tma_descriptor = nvgpu.TmaCreateDescriptorOp(
            tma_descriptor_ty, device_unranked_memref, map(const, self.tma_shape)
        )
        return self.tma_descriptor.result

    def prefetch(self, predicate=None):
        nvgpu.tma_prefetch_descriptor(self.tma_descriptor, predicate=predicate)

    def load(self, dest, mbarrier: Mbarriers, coords=[0, 0], predicate=None):
        coord_ops = [const(c) for c in coords]
        nvgpu.TmaAsyncLoadOp(
            dest,
            mbarrier.mbar_group_op,
            self.tma_descriptor,
            coordinates=coord_ops,
            mbarId=mbarrier.id_op,
            predicate=predicate,
        )


class WarpgroupAccumulatorMatrix:
    def __init__(self, M, N, ty):
        self.M = M
        self.N = N
        self.ty = ty

    @property
    def acc_ty(self):
        parse_str = f"!nvgpu.warpgroup.accumulator<fragmented=vector<{self.M}x{self.N}x{self.ty}>>"
        return ir.Type.parse(parse_str)

    def op(self):
        return nvgpu.warpgroup_mma_init_accumulator(self.acc_ty)


class WarpgroupMatrix:
    def __init__(self, smem, tma_descriptor: TMA, M, N):
        self.tma_descriptor = tma_descriptor
        self.smem = smem
        self.M = M
        self.N = N

    @property
    def wgmma_ty(self):
        parse_str = f"!nvgpu.warpgroup.descriptor<tensor=memref<{self.M}x{self.N}x{self.tma_descriptor.memref_ty.element_type}, #gpu.address_space<workgroup>>>"
        return ir.Type.parse(parse_str)

    def matmul(lhs, rhs, acc):
        wgmma_desc_lhs = nvgpu.warpgroup_generate_descriptor(
            lhs.wgmma_ty, lhs.smem, lhs.tma_descriptor.tma_descriptor
        )
        wgmma_desc_rhs = nvgpu.warpgroup_generate_descriptor(
            rhs.wgmma_ty, rhs.smem, rhs.tma_descriptor.tma_descriptor
        )
        return nvgpu.WarpgroupMmaOp(
            acc.type, wgmma_desc_lhs, wgmma_desc_rhs, acc, transposeB=True
        )


def get_dynamic_shared_memory(shape=None, ty=None, offset: int = 0):
    smem_space_str = "#gpu.address_space<workgroup>"
    smem_space = ir.Attribute.parse(smem_space_str)
    dynamic_smem = gpu.dynamic_shared_memory(
        ir.MemRefType.get((MLIR_DYNAMIC,), T.i8(), memory_space=smem_space)
    )
    if shape is None:
        return dynamic_smem
    memref_ty = ir.MemRefType.get(shape, ty, memory_space=smem_space)
    return memref.view(
        ir.MemRefType.get(
            memref_ty.shape, memref_ty.element_type, memory_space=smem_space
        ),
        dynamic_smem,
        const(offset),
        [],
    )


@staticmethod
def get_mlir_ty(arg):
    def get_mlir_ty_from_np(dtype):
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

    if isinstance(arg, bool):
        return T.bool()
    elif isinstance(arg, int):
        return T.index()
    elif isinstance(arg, float):
        return T.f32()
    elif isinstance(arg, np.ndarray):
        descriptor = rt.get_ranked_memref_descriptor(arg)
        dtype = get_mlir_ty_from_np(arg.dtype)
        shape = descriptor.shape
        return memref.MemRefType.get(shape, dtype)
    raise NotImplementedError(arg)


class NVDSL:
    @staticmethod
    def mlir_gpu_launch(grid=(1, 1, 1), block=(1, 1, 1), smem=0):
        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                launch_op = gpu.LaunchOp(
                    None,
                    [],
                    *map(const, grid),
                    *map(const, block),
                    dynamicSharedMemorySize=arith.constant(T.i32(), smem),
                )
                launch_op.body.blocks.append(*([T.index()] * 12))
                with ir.InsertionPoint(launch_op.body.blocks[0]):
                    result = func(*args, **kwargs)
                    gpu.terminator()
                    return result

            return wrapper

        return decorator

    @staticmethod
    def mlir_func(funcBody):
        @functools.wraps(funcBody)
        def wrapper(*args, **kwargs):
            function_name = funcBody.__name__

            def saveIR(module):
                """Save generated IR"""
                if True:  # self.saveIR:
                    # print(mlir_nvgpu_module)
                    original_stdout = sys.stdout
                    with open("nvdsl.mlir", "w") as f:
                        sys.stdout = f
                        print(module)
                        sys.stdout = original_stdout

            def _binary_op(lhs, rhs, op: str, predAtt="") -> "ArithValue":
                """Generate MLIR's Arith dialects binary operations."""
                rhs = const(rhs)
                if arith._is_float_type(lhs.type) and arith._is_float_type(rhs.type):
                    op += "F"
                    if op.startswith("Cmp"):
                        predicateAttr = getattr(arith, f"CmpFPredicate").__dict__[
                            predAtt
                        ]
                elif arith._is_integer_like_type(
                    lhs.type
                ) and arith._is_integer_like_type(lhs.type):
                    if op == "Div" or op == "Rem":
                        op += "U"
                    op += "I"
                    if op.startswith("Cmp"):
                        predicateAttr = getattr(arith, f"CmpIPredicate").__dict__[
                            predAtt
                        ]
                else:
                    raise NotImplementedError(
                        f"Unsupported '{op}' operands: {lhs}, {rhs}"
                    )

                if op.startswith("Cmp"):
                    op = getattr(arith, f"{op}Op")

                    return op(predicateAttr, lhs, rhs).result
                else:
                    op = getattr(arith, f"{op}Op")
                    return op(lhs, rhs).result

            @ir.register_value_caster(ir.IndexType.static_typeid)
            @ir.register_value_caster(ir.F32Type.static_typeid)
            @ir.register_value_caster(ir.F16Type.static_typeid)
            @ir.register_value_caster(ir.F64Type.static_typeid)
            @ir.register_value_caster(ir.IntegerType.static_typeid)
            class ArithValue(ir.Value):
                """Overloads operators for MLIR's Arith dialects binary operations."""

                def __init__(self, v):
                    super().__init__(v)

                __add__ = partialmethod(_binary_op, op="Add")
                __sub__ = partialmethod(_binary_op, op="Sub")
                __mul__ = partialmethod(_binary_op, op="Mul")
                __truediv__ = partialmethod(_binary_op, op="Div")
                __mod__ = partialmethod(_binary_op, op="Rem")
                __xor__ = partialmethod(_binary_op, op="XOr")
                __lt__ = partialmethod(_binary_op, op="Cmp", predAtt="ult")
                __le__ = partialmethod(_binary_op, op="Cmp", predAtt="ule")
                __eq__ = partialmethod(_binary_op, op="Cmp", predAtt="eq")
                __ne__ = partialmethod(_binary_op, op="Cmp", predAtt="ne")
                __gt__ = partialmethod(_binary_op, op="Cmp", predAtt="ugt")
                __ge__ = partialmethod(_binary_op, op="Cmp", predAtt="uge")
                __and__ = partialmethod(_binary_op, op="And")
                __or__ = partialmethod(_binary_op, op="Or")

                def __str__(self):
                    return (
                        super()
                        .__str__()
                        .replace(ir.Value.__name__, ArithValue.__name__)
                    )

            # Generate MLIR Context and start generating IR
            with ir.Context(), ir.Location.unknown():
                types = []
                for arg in args:
                    types.append(get_mlir_ty(arg))

                # Build IR
                module = ir.Module.create()
                with ir.InsertionPoint(module.body):
                    fop = func.FuncOp(function_name, (types, []))
                    fop.attributes["llvm.emit_c_interface"] = ir.UnitAttr.get()
                    with ir.InsertionPoint(fop.add_entry_block()):
                        fargs = []
                        for i, a in enumerate(types):
                            fargs.append(fop.arguments[i])

                        # Call user function body
                        result = funcBody(*fargs, **kwargs)
                        func.ReturnOp([])

                # Verify the module
                module.operation.verify()

                # Save IR in a file
                # saveIR(module)

                # Compile and JIT MLIR module
                options = f"cubin-chip=sm_90a cubin-features=+ptx80 opt-level=3"
                support_lib = os.getenv("SUPPORT_LIB")
                if not os.path.exists(support_lib):
                    raise FileNotFoundError(
                        errno.ENOENT, os.strerror(errno.ENOENT), support_lib
                    )
                compiler = nvgpucompiler.NvgpuCompiler(
                    options, opt_level=3, shared_libs=[support_lib]
                )
                engine = compiler.compile_and_jit(module)

            # Convert input arguments to MLIR arguments
            newArgs = get_mlir_func_obj_ty(args)

            # Run the compiled program
            engine.invoke(function_name, *newArgs)

            return result

        return wrapper
