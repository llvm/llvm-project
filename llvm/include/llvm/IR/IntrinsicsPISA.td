//===- IntrinsicsPISA.td - Defines PISA intrinsics ---------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines all of the PISA-specific intrinsics.
//
//===----------------------------------------------------------------------===//

// Pointer type definitions.
def pisa_global_ptr_ty   : LLVMQualPointerType<1>;  // (global)*
def pisa_constant_ptr_ty : LLVMQualPointerType<2>;  // (constant)*
def pisa_shared_ptr_ty   : LLVMQualPointerType<3>;  // (shared)*
def pisa_generic_ptr_ty  : LLVMQualPointerType<0>;  // (generic)*

// Base classes.
class PISAIntrinsic<list<LLVMType> ret_types, list<LLVMType> param_types = [],
                    list<IntrinsicProperty> intr_properties = [], string name = "">
    : Intrinsic<ret_types, param_types, intr_properties, name> {
  let TargetPrefix = "pisa";
}

class PISADefaultAttrsIntrinsic<list<LLVMType> ret_types, list<LLVMType> param_types = [],
                                list<IntrinsicProperty> intr_properties = [], string name = "">
    : DefaultAttrsIntrinsic<ret_types, param_types, intr_properties, name> {
  let TargetPrefix = "pisa";
}

class PISAIntrinsicClang<list<LLVMType> RetTypes, list<LLVMType> ParamTypes = [],
                         list<IntrinsicProperty> IntrProperties = [], string ClangName, string Name = "">
    : PISAIntrinsic<RetTypes, ParamTypes, IntrProperties, Name>,
      ClangBuiltin<!strconcat("__builtin_pisa_", ClangName)>;

class PISADefaultAttrsIntrinsicClang<list<LLVMType> RetTypes, list<LLVMType> ParamTypes = [],
                                     list<IntrinsicProperty> IntrProperties = [],
                                     string ClangName, string Name = "">
    : PISADefaultAttrsIntrinsic<RetTypes, ParamTypes, IntrProperties, Name>,
      ClangBuiltin<!strconcat("__builtin_pisa_", ClangName)>;

// Multiclass for XYZ dimension variants.
multiclass PISAIntrinsicXYZ<list<LLVMType> RetTypes, list<LLVMType> ParamTypes = [],
                            list<IntrinsicProperty> IntrProperties = [], string ClangName> {
  def _x : PISADefaultAttrsIntrinsicClang<RetTypes, ParamTypes, IntrProperties, !strconcat(ClangName, "_x")>;
  def _y : PISADefaultAttrsIntrinsicClang<RetTypes, ParamTypes, IntrProperties, !strconcat(ClangName, "_y")>;
  def _z : PISADefaultAttrsIntrinsicClang<RetTypes, ParamTypes, IntrProperties, !strconcat(ClangName, "_z")>;
}

// Constants.
defvar MAX_WORK_DIM = 3;
defvar MAX_LOCAL_SIZE = 65536;
defvar MAX_SUB_GROUP_SIZE = 32;

// Helper classes for common ArgInfo patterns.
class RoundingModeArgInfo<ArgIndex idx> : ArgInfo<idx, [ArgName<"round">, ImmArgPrinter<"printRoundingMode">]>;
class SaturationArgInfo<ArgIndex idx>   : ArgInfo<idx, [ArgName<"saturation">]>;
class NanPropagateArgInfo<ArgIndex idx> : ArgInfo<idx, [ArgName<"nanp">]>;

// Helper class for barrier intrinsics.
class PISAIntrinsicBarrier<string ClangName, list<LLVMType> param_types = []>
    : PISAIntrinsicClang<[], param_types, [IntrNoMem, IntrHasSideEffects, IntrConvergent], ClangName>;

// Helper class for unary float operations with rounding mode.
class PISAUnaryRndIntrinsic
    : PISADefaultAttrsIntrinsic<[llvm_anyfloat_ty], [LLVMMatchType<0>, llvm_i8_ty], [IntrNoMem, ImmArg<ArgIndex<1>>]>;

// Helper class for binary float operations with rounding mode.
class PISABinaryRndIntrinsic
    : PISADefaultAttrsIntrinsic<[llvm_anyfloat_ty], [LLVMMatchType<0>, LLVMMatchType<0>, llvm_i8_ty],
                                [IntrNoMem, ImmArg<ArgIndex<2>>]>;

// Helper class for float operations with rounding mode + saturation flag.
class PISAFloatIntrinsic<LLVMType ret_ty, list<LLVMType> operand_types>
    : PISADefaultAttrsIntrinsic<[ret_ty], [], []> {
  defvar num_params = !size(operand_types);
  let ParamTypes = !listconcat(
    operand_types,
    [llvm_i8_ty, // rounding mode
     llvm_i1_ty] // saturation flag
  );
  let IntrProperties = [
    IntrNoMem,
    ImmArg<ArgIndex<num_params>>,
    ImmArg<ArgIndex<!add(num_params, 1)>>,
    RoundingModeArgInfo<ArgIndex<num_params>>,
    SaturationArgInfo<ArgIndex<!add(num_params, 1)>>
  ];
}


//===----------------------------------------------------------------------===//
// 1. Special Registers
//===----------------------------------------------------------------------===//

defm int_pisa_local_id             : PISAIntrinsicXYZ<[llvm_i32_ty], [], [IntrNoMem, Range<RetIndex, 0, MAX_LOCAL_SIZE>], "local_id">;
defm int_pisa_local_size           : PISAIntrinsicXYZ<[llvm_i32_ty], [], [IntrNoMem, Range<RetIndex, 1, !add(MAX_LOCAL_SIZE, 1)>], "local_size">;
defm int_pisa_enqueued_local_size  : PISAIntrinsicXYZ<[llvm_i32_ty], [], [IntrNoMem, Range<RetIndex, 1, !add(MAX_LOCAL_SIZE, 1)>],
                                                       "enqueued_local_size">;
defm int_pisa_group_id             : PISAIntrinsicXYZ<[llvm_i32_ty], [], [IntrNoMem], "group_id">;
defm int_pisa_group_count          : PISAIntrinsicXYZ<[llvm_i32_ty], [], [IntrNoMem], "group_count">;
defm int_pisa_global_size          : PISAIntrinsicXYZ<[llvm_i64_ty], [], [IntrNoMem], "global_size">;
defm int_pisa_global_offset        : PISAIntrinsicXYZ<[llvm_i64_ty], [], [IntrNoMem], "global_offset">;

def int_pisa_lane_id         : PISADefaultAttrsIntrinsicClang<[llvm_i32_ty], [],
                                   [IntrNoMem, Range<RetIndex, 0, MAX_SUB_GROUP_SIZE>], "lane_id">;
def int_pisa_subgroup_size   : PISADefaultAttrsIntrinsicClang<[llvm_i32_ty], [],
                                   [IntrNoMem, Range<RetIndex, MAX_SUB_GROUP_SIZE, !add(MAX_SUB_GROUP_SIZE, 1)>],
                                   "subgroup_size">;
def int_pisa_work_dim        : PISADefaultAttrsIntrinsicClang<[llvm_i32_ty], [],
                                   [IntrNoMem, Range<RetIndex, 1, !add(MAX_WORK_DIM, 1)>], "work_dim">;

def int_pisa_activemask          : PISAIntrinsicClang<[llvm_i32_ty], [], [IntrNoMem, IntrConvergent], "activemask">;

//===----------------------------------------------------------------------===//
// 2. Integer Arithmetic
//===----------------------------------------------------------------------===//

def int_pisa_dp4a_uu : PISADefaultAttrsIntrinsic<[llvm_i32_ty],
                           [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i1_ty],
                           [IntrNoMem, IntrWillReturn, ImmArg<ArgIndex<3>>,
                            SaturationArgInfo<ArgIndex<3>>]>;
def int_pisa_dp4a_us : PISADefaultAttrsIntrinsic<[llvm_i32_ty],
                           [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i1_ty],
                           [IntrNoMem, IntrWillReturn, ImmArg<ArgIndex<3>>,
                            SaturationArgInfo<ArgIndex<3>>]>;
def int_pisa_dp4a_su : PISADefaultAttrsIntrinsic<[llvm_i32_ty],
                           [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i1_ty],
                           [IntrNoMem, IntrWillReturn, ImmArg<ArgIndex<3>>,
                            SaturationArgInfo<ArgIndex<3>>]>;
def int_pisa_dp4a_ss : PISADefaultAttrsIntrinsic<[llvm_i32_ty],
                           [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i1_ty],
                           [IntrNoMem, IntrWillReturn, ImmArg<ArgIndex<3>>,
                            SaturationArgInfo<ArgIndex<3>>]>;

def int_pisa_smad : PISADefaultAttrsIntrinsic<[llvm_anyint_ty],
                        [LLVMMatchType<0>, LLVMMatchType<0>, LLVMMatchType<0>], [IntrNoMem]>;

//===----------------------------------------------------------------------===//
// 3. Floating Point
//===----------------------------------------------------------------------===//

def int_pisa_fabs   : PISADefaultAttrsIntrinsic<[llvm_anyfloat_ty], [LLVMMatchType<0>], [IntrNoMem]>;
def int_pisa_frcp   : PISADefaultAttrsIntrinsic<[llvm_anyfloat_ty], [LLVMMatchType<0>], [IntrNoMem]>;
def int_pisa_frsqrt : PISADefaultAttrsIntrinsic<[llvm_anyfloat_ty], [LLVMMatchType<0>], [IntrNoMem]>;
def int_pisa_frc    : PISADefaultAttrsIntrinsicClang<[llvm_float_ty], [llvm_float_ty], [IntrNoMem], "frcf">;

def int_pisa_fmin_sat : PISADefaultAttrsIntrinsic<[llvm_anyfloat_ty],
                              [LLVMMatchType<0>, LLVMMatchType<0>, llvm_i1_ty],
                              [IntrNoMem, Commutative, ImmArg<ArgIndex<2>>,
                               NanPropagateArgInfo<ArgIndex<2>>]>;
def int_pisa_fmax_sat : PISADefaultAttrsIntrinsic<[llvm_anyfloat_ty],
                              [LLVMMatchType<0>, LLVMMatchType<0>, llvm_i1_ty],
                              [IntrNoMem, Commutative, ImmArg<ArgIndex<2>>,
                               NanPropagateArgInfo<ArgIndex<2>>]>;

// Float operations with rounding mode + saturation flag.
def int_pisa_fadd    : PISAFloatIntrinsic<llvm_anyfloat_ty, [LLVMMatchType<0>, LLVMMatchType<0>]>;
def int_pisa_fsub    : PISAFloatIntrinsic<llvm_anyfloat_ty, [LLVMMatchType<0>, LLVMMatchType<0>]>;
def int_pisa_fmul    : PISAFloatIntrinsic<llvm_anyfloat_ty, [LLVMMatchType<0>, LLVMMatchType<0>]>;
def int_pisa_fma     : PISAFloatIntrinsic<llvm_anyfloat_ty, [LLVMMatchType<0>, LLVMMatchType<0>, LLVMMatchType<0>]>;

// Binary float operations with explicit rounding mode.
def int_pisa_fdiv_rnd : PISABinaryRndIntrinsic;
def int_pisa_pow_rnd  : PISABinaryRndIntrinsic;

// Unary float operations with explicit rounding mode.
def int_pisa_fsqrt_rnd  : PISAUnaryRndIntrinsic;
def int_pisa_frnd_rnd   : PISAUnaryRndIntrinsic;
def int_pisa_frcp_rnd   : PISAUnaryRndIntrinsic;
def int_pisa_sin_rnd    : PISAUnaryRndIntrinsic;
def int_pisa_cos_rnd    : PISAUnaryRndIntrinsic;
def int_pisa_tanh_rnd   : PISAUnaryRndIntrinsic;
def int_pisa_exp_rnd    : PISAUnaryRndIntrinsic;
def int_pisa_exp2_rnd   : PISAUnaryRndIntrinsic;
def int_pisa_log_rnd    : PISAUnaryRndIntrinsic;
def int_pisa_log2_rnd   : PISAUnaryRndIntrinsic;
def int_pisa_log10_rnd  : PISAUnaryRndIntrinsic;

//===----------------------------------------------------------------------===//
// 4. Logic and Shift
//===----------------------------------------------------------------------===//

def int_pisa_bfi  : PISADefaultAttrsIntrinsicClang<[llvm_i32_ty],
                        [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [IntrNoMem], "bfi">;
def int_pisa_ubfe : PISADefaultAttrsIntrinsicClang<[llvm_i32_ty],
                        [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [IntrNoMem], "ubfe">;
def int_pisa_sbfe : PISADefaultAttrsIntrinsicClang<[llvm_i32_ty],
                        [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [IntrNoMem], "sbfe">;
def int_pisa_bfn  : PISADefaultAttrsIntrinsic<[llvm_anyint_ty],
                        [llvm_i8_ty, LLVMMatchType<0>, LLVMMatchType<0>, LLVMMatchType<0>],
                        [IntrNoMem, ImmArg<ArgIndex<0>>]>;

//===----------------------------------------------------------------------===//
// 5. Data Movement and Conversion
//===----------------------------------------------------------------------===//

// Conversions with explicit rounding mode (metadata).
def int_pisa_fptosi_md : PISADefaultAttrsIntrinsic<[llvm_anyint_ty], [llvm_anyfloat_ty, llvm_metadata_ty], [IntrNoMem]>;
def int_pisa_fptoui_md : PISADefaultAttrsIntrinsic<[llvm_anyint_ty], [llvm_anyfloat_ty, llvm_metadata_ty], [IntrNoMem]>;

// Conversions with explicit rounding mode (i8 immediate).
def int_pisa_fptosi_rnd : PISADefaultAttrsIntrinsic<[llvm_anyint_ty], [llvm_anyfloat_ty, llvm_i8_ty],
                              [IntrNoMem, ImmArg<ArgIndex<1>>]>;
def int_pisa_fptoui_rnd : PISADefaultAttrsIntrinsic<[llvm_anyint_ty], [llvm_anyfloat_ty, llvm_i8_ty],
                              [IntrNoMem, ImmArg<ArgIndex<1>>]>;

// Conversions with rounding mode + saturation flag.
def int_pisa_sitofp : PISAFloatIntrinsic<llvm_anyfloat_ty, [llvm_anyint_ty]>;
def int_pisa_uitofp : PISAFloatIntrinsic<llvm_anyfloat_ty, [llvm_anyint_ty]>;
def int_pisa_ftrunc : PISAFloatIntrinsic<llvm_anyfloat_ty, [llvm_anyfloat_ty]>;

// Address space checks.
def int_pisa_isaddr_private : PISADefaultAttrsIntrinsicClang<[llvm_i32_ty], [pisa_generic_ptr_ty], [IntrNoMem], "isaddr_private">;
def int_pisa_isaddr_shared  : PISADefaultAttrsIntrinsicClang<[llvm_i32_ty], [pisa_generic_ptr_ty], [IntrNoMem], "isaddr_shared">;
def int_pisa_isaddr_global  : PISADefaultAttrsIntrinsicClang<[llvm_i32_ty], [pisa_generic_ptr_ty], [IntrNoMem], "isaddr_global">;

//===----------------------------------------------------------------------===//
// 6. Subgroup Communication
//===----------------------------------------------------------------------===//

def int_pisa_shfl : PISADefaultAttrsIntrinsic<[llvm_i32_ty],
                        [llvm_i8_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i1_ty],
                        [IntrNoMem, IntrConvergent, ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<5>>,
                         ArgInfo<ArgIndex<0>, [ArgName<"mode">, ImmArgPrinter<"printSHFLMode">]>,
                         ArgInfo<ArgIndex<5>, [ArgName<"sg">]>]>;
def int_pisa_ired : PISADefaultAttrsIntrinsic<[llvm_anyint_ty],
                        [llvm_i8_ty, LLVMMatchType<0>, llvm_i32_ty, LLVMMatchType<0>],
                        [IntrNoMem, IntrConvergent, ImmArg<ArgIndex<0>>,
                         ArgInfo<ArgIndex<0>, [ArgName<"op">, ImmArgPrinter<"printIRedOp">]>]>;
def int_pisa_fred : PISADefaultAttrsIntrinsic<[llvm_anyfloat_ty],
                        [llvm_i8_ty, LLVMMatchType<0>, llvm_i32_ty, LLVMMatchType<0>, llvm_i1_ty],
                        [IntrNoMem, IntrConvergent, ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<4>>,
                         ArgInfo<ArgIndex<0>, [ArgName<"op">, ImmArgPrinter<"printFRedOp">]>,
                         NanPropagateArgInfo<ArgIndex<4>>]>;
def int_pisa_redfirstidx : PISADefaultAttrsIntrinsicClang<[llvm_i32_ty], [llvm_i32_ty],
                               [IntrNoMem, IntrConvergent], "redfirstidx">;

//===----------------------------------------------------------------------===//
// 7. Memory and Atomics
//===----------------------------------------------------------------------===//

def int_pisa_cas_fatom : PISADefaultAttrsIntrinsic<[llvm_anyfloat_ty],
                             [llvm_anyptr_ty, LLVMMatchType<0>, LLVMMatchType<0>, llvm_i8_ty],
                             [IntrHasSideEffects, ImmArg<ArgIndex<3>>]>;

//===----------------------------------------------------------------------===//
// 8. Synchronization
//===----------------------------------------------------------------------===//

def int_pisa_workgroup_barrier : PISAIntrinsicBarrier<"workgroup_barrier">;
