//===- IntrinsicsAGX.td - Defines Apple AGX intrinsics -----*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines all of the Apple AGX GPU specific intrinsics. See
// AGXUsage.rst for the semantics of each intrinsic.
//
//===----------------------------------------------------------------------===//

let TargetPrefix = "agx" in {

//===----------------------------------------------------------------------===//
// Special registers and dispatch geometry.
//===----------------------------------------------------------------------===//

// Values that are fixed for the lifetime of a thread. They are read with
// get_sr (or from implicit uniforms) and can be freely speculated.
class AGXReadSRIntrinsic
    : DefaultAttrsIntrinsic<[llvm_i32_ty], [],
                            [NoUndef<RetIndex>, IntrNoMem, IntrSpeculatable]>;

multiclass AGXReadSRIntrinsic_xyz {
  def _x : AGXReadSRIntrinsic;
  def _y : AGXReadSRIntrinsic;
  def _z : AGXReadSRIntrinsic;
}

defm int_agx_thread_position_in_grid : AGXReadSRIntrinsic_xyz;
defm int_agx_threadgroup_position_in_grid : AGXReadSRIntrinsic_xyz;
defm int_agx_threads_per_threadgroup : AGXReadSRIntrinsic_xyz;
defm int_agx_thread_position_in_threadgroup : AGXReadSRIntrinsic_xyz;

// The number of threadgroups in the grid. There is no special register for
// this; the value is an implicit kernel argument (see AGXUsage.rst).
defm int_agx_threadgroups_per_grid : AGXReadSRIntrinsic_xyz;

// A 32-bit value the launch computes from 32-bit word (operand 1) of kernel
// argument (operand 0), and puts in the uniforms (.agx.kernels extension 7):
// for an unsigned divisor d, (2) 2^32 * (2^l - d) / d + 1, (3) min(l, 1) and
// (4) max(l, 1) - 1, l = ceil(log2(d)); (5, 6, 7) the same of |d| for a
// signed divisor, (8, 9, 10) of max(d, 1). Operand 2 says which.
def int_agx_launch_value
    : DefaultAttrsIntrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
                            [NoUndef<RetIndex>, IntrNoMem, IntrSpeculatable,
                             ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>,
                             ImmArg<ArgIndex<2>>]>;

def int_agx_thread_index_in_threadgroup : AGXReadSRIntrinsic;
def int_agx_thread_index_in_simdgroup : AGXReadSRIntrinsic;
def int_agx_simdgroup_index_in_threadgroup : AGXReadSRIntrinsic;
def int_agx_core_id : AGXReadSRIntrinsic;

// Values that depend on which threads of the SIMD-group are active.
class AGXReadActiveSRIntrinsic
    : DefaultAttrsIntrinsic<[llvm_i32_ty], [],
                            [NoUndef<RetIndex>, IntrNoMem, IntrConvergent]>;

def int_agx_active_thread_index_in_simdgroup : AGXReadActiveSRIntrinsic;
def int_agx_total_active_threads_in_simdgroup : AGXReadActiveSRIntrinsic;
def int_agx_active_thread_index_in_quadgroup : AGXReadActiveSRIntrinsic;
def int_agx_total_active_threads_in_quadgroup : AGXReadActiveSRIntrinsic;

// Read an arbitrary special register by number. Prefer the named intrinsics.
def int_agx_read_sr
    : DefaultAttrsIntrinsic<[llvm_i32_ty], [llvm_i32_ty],
                            [IntrNoMem, IntrConvergent, ImmArg<ArgIndex<0>>]>;

//===----------------------------------------------------------------------===//
// Synchronization.
//===----------------------------------------------------------------------===//

// Execution barrier across the threadgroup (threadgroup_barrier).
def int_agx_threadgroup_barrier
    : Intrinsic<[], [],
                [IntrNoMem, IntrHasSideEffects, IntrConvergent,
                 IntrWillReturn, IntrNoCallback, IntrNoFree]>;

// Execution barrier across the caller's aligned block of 4 SIMD-groups
// (block_barrier), ordering their memory accesses. Every SIMD-group of the
// block must reach one. A threadgroup_barrier on chips without it. It is a
// memory operation, so no fence is needed around it: the hardware orders the
// block's accesses (B16) and the compiler moves none across it.
def int_agx_block_barrier
    : Intrinsic<[], [],
                [IntrHasSideEffects, IntrConvergent, IntrWillReturn,
                 IntrNoCallback, IntrNoFree]>;

// Raw memory_barrier with its three hardware fields (a, b, c). The memory
// model lowering of `fence` picks the encodings documented in AGXUsage.rst.
def int_agx_memory_barrier
    : Intrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
                [IntrHasSideEffects, IntrWillReturn, IntrNoCallback,
                 IntrNoFree, ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>,
                 ImmArg<ArgIndex<2>>]>;

// Wait for all outstanding asynchronous operations on a scoreboard slot.
def int_agx_wait
    : Intrinsic<[], [llvm_i32_ty],
                [IntrHasSideEffects, IntrWillReturn, IntrNoCallback,
                 IntrNoFree, ImmArg<ArgIndex<0>>]>;

//===----------------------------------------------------------------------===//
// Floating point environment.
//===----------------------------------------------------------------------===//

// There is no floating point mode register: these model one, and the AGX
// backend turns it into SSA values before instruction selection. The mode is
//   bits 0-1  float rounding (0 toward zero, 1 nearest even, 2 up, 3 down,
//             like llvm.get.rounding),
//   bits 2-3  half and double rounding,
//   bit  4    float denormals flushed to zero,
//   bit  5    half and double denormals flushed to zero.
// Constrained floating point intrinsics with dynamic rounding use it. At
// function entry, rounding is to nearest even and the denormal bits follow
// the function's denormal_fpenv.
def int_agx_set_fpmode
    : Intrinsic<[], [llvm_i32_ty],
                [IntrInaccessibleMemOnly, IntrWillReturn, IntrNoCallback,
                 IntrNoFree]>;
def int_agx_get_fpmode
    : Intrinsic<[llvm_i32_ty], [],
                [IntrInaccessibleMemOnly, IntrReadMem, IntrWillReturn,
                 IntrNoCallback, IntrNoFree]>;

//===----------------------------------------------------------------------===//
// SIMD-group and quad-group operations.
//===----------------------------------------------------------------------===//

class AGXSIMDShuffleIntrinsic
    : DefaultAttrsIntrinsic<[llvm_any_ty], [LLVMMatchType<0>, llvm_i32_ty],
                            [IntrNoMem, IntrConvergent]>;

def int_agx_simd_shuffle : AGXSIMDShuffleIntrinsic;
def int_agx_simd_shuffle_xor : AGXSIMDShuffleIntrinsic;
def int_agx_simd_shuffle_up : AGXSIMDShuffleIntrinsic;
def int_agx_simd_shuffle_down : AGXSIMDShuffleIntrinsic;
def int_agx_quad_shuffle : AGXSIMDShuffleIntrinsic;
def int_agx_quad_shuffle_xor : AGXSIMDShuffleIntrinsic;
def int_agx_quad_shuffle_up : AGXSIMDShuffleIntrinsic;
def int_agx_quad_shuffle_down : AGXSIMDShuffleIntrinsic;

class AGXBallotIntrinsic
    : DefaultAttrsIntrinsic<[llvm_i32_ty], [llvm_i1_ty],
                            [IntrNoMem, IntrConvergent]>;

def int_agx_simd_ballot : AGXBallotIntrinsic;
def int_agx_quad_ballot : AGXBallotIntrinsic;

// Reductions and exclusive scans. The second operand is the hardware
// operation number (AGX::SIMDOp): and = 0, fadd = 1, or = 2, fmul = 3,
// xor = 4, fmin = 5, fmax = 7, iadd = 16, smin = 20, smax = 22, umin = 28,
// umax = 30.
class AGXSIMDReduceIntrinsic
    : DefaultAttrsIntrinsic<[llvm_any_ty], [LLVMMatchType<0>, llvm_i32_ty],
                            [IntrNoMem, IntrConvergent, ImmArg<ArgIndex<1>>]>;

def int_agx_simd_reduce : AGXSIMDReduceIntrinsic;
def int_agx_simd_prefix : AGXSIMDReduceIntrinsic;
def int_agx_quad_reduce : AGXSIMDReduceIntrinsic;
def int_agx_quad_prefix : AGXSIMDReduceIntrinsic;

// simdgroup 8x8 matrix multiply-accumulate. Each thread holds two elements
// of each matrix: D = A * B + C.
def int_agx_simdgroup_matrix_fma
    : DefaultAttrsIntrinsic<[llvm_anyvector_ty],
                            [LLVMMatchType<0>, LLVMMatchType<0>,
                             LLVMMatchType<0>],
                            [IntrNoMem, IntrConvergent]>;

// The matrix unit's simd_matrix_fmadd32 / simd_matrix_fmadd16, which
// AGXCodeGenPrepare uses for llvm.agx.simdgroup.matrix.fma where it gives the
// same results: the <2 x float> (i64) or <2 x half> (i32) of each matrix.
def int_agx_simd_matrix_fmadd32
    : DefaultAttrsIntrinsic<[llvm_i64_ty],
                            [llvm_i64_ty, llvm_i64_ty, llvm_i64_ty],
                            [IntrNoMem, IntrConvergent]>;
def int_agx_simd_matrix_fmadd16
    : DefaultAttrsIntrinsic<[llvm_i32_ty],
                            [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
                            [IntrNoMem, IntrConvergent]>;

//===----------------------------------------------------------------------===//
// Textures.
//===----------------------------------------------------------------------===//

// texture_sample in its bindless form: the 24-byte texture descriptor at byte
// offset T (operand 0) from the texture heap address, which every kernel of a
// module that samples receives in its first uniform registers (AGXUsage.rst,
// "Textures"), sampled with sampler S (operand 1), an index into the command's
// sampler heap, at coordinates (x, y) (y unused for 1D), at level 0. The
// result is the four channels as 32-bit values (floats for float, unorm and
// snorm formats, integers otherwise). Operand 4 is the dimension: 0 for 1D,
// 2 for 2D.
def int_agx_texture_sample
    : DefaultAttrsIntrinsic<[llvm_v4i32_ty],
                            [llvm_i32_ty, llvm_i16_ty, llvm_float_ty,
                             llvm_float_ty, llvm_i32_ty],
                            [IntrReadMem, ImmArg<ArgIndex<4>>]>;

//===----------------------------------------------------------------------===//
// Arithmetic.
//===----------------------------------------------------------------------===//

class AGXFloatUnaryIntrinsic
    : DefaultAttrsIntrinsic<[llvm_anyfloat_ty], [LLVMMatchType<0>],
                            [IntrNoMem, IntrSpeculatable]>;

// Hardware approximations, without any special case handling beyond what
// the instruction does.
def int_agx_rcp : AGXFloatUnaryIntrinsic;
def int_agx_rsqrt : AGXFloatUnaryIntrinsic;
// rsqrt with special cases chosen so that rsqrt_special(x) * x == sqrt(x).
def int_agx_rsqrt_special : AGXFloatUnaryIntrinsic;
// Sine building blocks, the argument is in quadrants: see AGXUsage.rst.
def int_agx_sin_pt_1 : AGXFloatUnaryIntrinsic;
def int_agx_sin_pt_2 : AGXFloatUnaryIntrinsic;
// sin, cos, exp2 and log2 as the hardware approximates them (the code of
// llvm.sin and the others), which LLVM does not evaluate on the host for a
// constant argument: constant and variable arguments give the same bits.
def int_agx_sin : AGXFloatUnaryIntrinsic;
def int_agx_cos : AGXFloatUnaryIntrinsic;
def int_agx_exp2 : AGXFloatUnaryIntrinsic;
def int_agx_log2 : AGXFloatUnaryIntrinsic;

// Screen space derivatives, only meaningful in fragment shaders.
class AGXDerivativeIntrinsic
    : DefaultAttrsIntrinsic<[llvm_anyfloat_ty], [LLVMMatchType<0>],
                            [IntrNoMem, IntrConvergent]>;
def int_agx_dfdx : AGXDerivativeIntrinsic;
def int_agx_dfdy : AGXDerivativeIntrinsic;

// Interleave the low 16 bits of the operands (Morton encode).
def int_agx_interleave
    : DefaultAttrsIntrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty],
                            [IntrNoMem, IntrSpeculatable]>;

//===----------------------------------------------------------------------===//
// Structured control flow. These are inserted by AGXAnnotateControlFlow and
// are not meant to be used by frontends.
//===----------------------------------------------------------------------===//

def int_agx_if : Intrinsic<[llvm_i1_ty], [llvm_i1_ty],
                           [IntrWillReturn, IntrNoCallback, IntrNoFree]>;
def int_agx_else : Intrinsic<[llvm_i1_ty], [],
                             [IntrWillReturn, IntrNoCallback, IntrNoFree]>;
def int_agx_end_cf : Intrinsic<[], [],
                               [IntrWillReturn, IntrNoCallback, IntrNoFree]>;
def int_agx_loop_begin : Intrinsic<[], [],
                                   [IntrWillReturn, IntrNoCallback,
                                    IntrNoFree]>;
def int_agx_loop : Intrinsic<[llvm_i1_ty], [llvm_i1_ty],
                             [IntrWillReturn, IntrNoCallback, IntrNoFree]>;

} // TargetPrefix = "agx"
