//===- IntrinsicsARM.td - Defines ARM intrinsics -----------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines all of the ARM-specific intrinsics.
//
//===----------------------------------------------------------------------===//


//===----------------------------------------------------------------------===//
// TLS

let TargetPrefix = "arm" in {  // All intrinsics start with "llvm.arm.".

// A space-consuming intrinsic primarily for testing ARMConstantIslands. The
// first argument is the number of bytes this "instruction" takes up, the second
// and return value are essentially chains, used to force ordering during ISel.
def int_arm_space : Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>]>;

// 16-bit multiplications
def int_arm_smulbb : GCCBuiltin<"__builtin_arm_smulbb">,
    Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty], [IntrNoMem]>;
def int_arm_smulbt : GCCBuiltin<"__builtin_arm_smulbt">,
    Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty], [IntrNoMem]>;
def int_arm_smultb : GCCBuiltin<"__builtin_arm_smultb">,
    Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty], [IntrNoMem]>;
def int_arm_smultt : GCCBuiltin<"__builtin_arm_smultt">,
    Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty], [IntrNoMem]>;
def int_arm_smulwb : GCCBuiltin<"__builtin_arm_smulwb">,
    Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty], [IntrNoMem]>;
def int_arm_smulwt : GCCBuiltin<"__builtin_arm_smulwt">,
    Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty], [IntrNoMem]>;

//===----------------------------------------------------------------------===//
// Saturating Arithmetic

def int_arm_qadd : GCCBuiltin<"__builtin_arm_qadd">,
    Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty],
    [Commutative, IntrNoMem]>;
def int_arm_qsub : GCCBuiltin<"__builtin_arm_qsub">,
    Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty], [IntrNoMem]>;
def int_arm_ssat : GCCBuiltin<"__builtin_arm_ssat">,
    Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty], [IntrNoMem]>;
def int_arm_usat : GCCBuiltin<"__builtin_arm_usat">,
    Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty], [IntrNoMem]>;

// Accumulating multiplications
def int_arm_smlabb : GCCBuiltin<"__builtin_arm_smlabb">,
    Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
    [IntrNoMem]>;
def int_arm_smlabt : GCCBuiltin<"__builtin_arm_smlabt">,
    Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
    [IntrNoMem]>;
def int_arm_smlatb : GCCBuiltin<"__builtin_arm_smlatb">,
    Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
    [IntrNoMem]>;
def int_arm_smlatt : GCCBuiltin<"__builtin_arm_smlatt">,
    Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
    [IntrNoMem]>;
def int_arm_smlawb : GCCBuiltin<"__builtin_arm_smlawb">,
    Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
    [IntrNoMem]>;
def int_arm_smlawt : GCCBuiltin<"__builtin_arm_smlawt">,
    Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
    [IntrNoMem]>;

// Parallel 16-bit saturation
def int_arm_ssat16 : GCCBuiltin<"__builtin_arm_ssat16">,
    Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty], [IntrNoMem]>;
def int_arm_usat16 : GCCBuiltin<"__builtin_arm_usat16">,
    Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty], [IntrNoMem]>;

// Packing and unpacking
def int_arm_sxtab16 : GCCBuiltin<"__builtin_arm_sxtab16">,
    Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty], [IntrNoMem]>;
def int_arm_sxtb16 : GCCBuiltin<"__builtin_arm_sxtb16">,
    Intrinsic<[llvm_i32_ty], [llvm_i32_ty], [IntrNoMem]>;
def int_arm_uxtab16 : GCCBuiltin<"__builtin_arm_uxtab16">,
    Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty], [IntrNoMem]>;
def int_arm_uxtb16 : GCCBuiltin<"__builtin_arm_uxtb16">,
    Intrinsic<[llvm_i32_ty], [llvm_i32_ty], [IntrNoMem]>;

// Parallel selection, reads the GE flags.
def int_arm_sel : GCCBuiltin<"__builtin_arm_sel">,
    Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty], [IntrReadMem]>;

// Parallel 8-bit addition and subtraction
def int_arm_qadd8  : GCCBuiltin<"__builtin_arm_qadd8">,
    Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty], [IntrNoMem]>;
def int_arm_qsub8  : GCCBuiltin<"__builtin_arm_qsub8">,
    Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty], [IntrNoMem]>;
// Writes to the GE bits.
def int_arm_sadd8  : GCCBuiltin<"__builtin_arm_sadd8">,
    Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty], []>;
def int_arm_shadd8  : GCCBuiltin<"__builtin_arm_shadd8">,
    Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty], [IntrNoMem]>;
def int_arm_shsub8  : GCCBuiltin<"__builtin_arm_shsub8">,
    Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty], [IntrNoMem]>;
// Writes to the GE bits.
def int_arm_ssub8  : GCCBuiltin<"__builtin_arm_ssub8">,
    Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty], []>;
// Writes to the GE bits.
def int_arm_uadd8  : GCCBuiltin<"__builtin_arm_uadd8">,
    Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty], []>;
def int_arm_uhadd8  : GCCBuiltin<"__builtin_arm_uhadd8">,
    Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty], [IntrNoMem]>;
def int_arm_uhsub8  : GCCBuiltin<"__builtin_arm_uhsub8">,
    Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty], [IntrNoMem]>;
def int_arm_uqadd8  : GCCBuiltin<"__builtin_arm_uqadd8">,
    Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty], [IntrNoMem]>;
def int_arm_uqsub8  : GCCBuiltin<"__builtin_arm_uqsub8">,
    Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty], [IntrNoMem]>;
// Writes to the GE bits.
def int_arm_usub8  : GCCBuiltin<"__builtin_arm_usub8">,
    Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty], []>;

// Sum of 8-bit absolute differences
def int_arm_usad8  : GCCBuiltin<"__builtin_arm_usad8">,
    Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty], [IntrNoMem]>;
def int_arm_usada8  : GCCBuiltin<"__builtin_arm_usada8">,
    Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
              [IntrNoMem]>;

// Parallel 16-bit addition and subtraction
def int_arm_qadd16  : GCCBuiltin<"__builtin_arm_qadd16">,
    Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty], [IntrNoMem]>;
def int_arm_qasx  : GCCBuiltin<"__builtin_arm_qasx">,
    Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty], [IntrNoMem]>;
def int_arm_qsax  : GCCBuiltin<"__builtin_arm_qsax">,
    Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty], [IntrNoMem]>;
def int_arm_qsub16  : GCCBuiltin<"__builtin_arm_qsub16">,
    Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty], [IntrNoMem]>;
// Writes to the GE bits.
def int_arm_sadd16  : GCCBuiltin<"__builtin_arm_sadd16">,
    Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty], []>;
// Writes to the GE bits.
def int_arm_sasx  : GCCBuiltin<"__builtin_arm_sasx">,
    Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty], []>;
def int_arm_shadd16  : GCCBuiltin<"__builtin_arm_shadd16">,
    Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty], [IntrNoMem]>;
def int_arm_shasx  : GCCBuiltin<"__builtin_arm_shasx">,
    Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty], [IntrNoMem]>;
def int_arm_shsax  : GCCBuiltin<"__builtin_arm_shsax">,
    Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty], [IntrNoMem]>;
def int_arm_shsub16  : GCCBuiltin<"__builtin_arm_shsub16">,
    Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty], [IntrNoMem]>;
// Writes to the GE bits.
def int_arm_ssax  : GCCBuiltin<"__builtin_arm_ssax">,
    Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty], []>;
// Writes to the GE bits.
def int_arm_ssub16  : GCCBuiltin<"__builtin_arm_ssub16">,
    Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty], []>;
// Writes to the GE bits.
def int_arm_uadd16  : GCCBuiltin<"__builtin_arm_uadd16">,
    Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty], []>;
// Writes to the GE bits.
def int_arm_uasx  : GCCBuiltin<"__builtin_arm_uasx">,
    Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty], []>;
def int_arm_uhadd16  : GCCBuiltin<"__builtin_arm_uhadd16">,
    Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty], [IntrNoMem]>;
def int_arm_uhasx  : GCCBuiltin<"__builtin_arm_uhasx">,
    Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty], [IntrNoMem]>;
def int_arm_uhsax  : GCCBuiltin<"__builtin_arm_uhsax">,
    Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty], [IntrNoMem]>;
def int_arm_uhsub16  : GCCBuiltin<"__builtin_arm_uhsub16">,
    Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty], [IntrNoMem]>;
def int_arm_uqadd16  : GCCBuiltin<"__builtin_arm_uqadd16">,
    Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty], [IntrNoMem]>;
def int_arm_uqasx  : GCCBuiltin<"__builtin_arm_uqasx">,
    Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty], [IntrNoMem]>;
def int_arm_uqsax  : GCCBuiltin<"__builtin_arm_uqsax">,
    Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty], [IntrNoMem]>;
def int_arm_uqsub16  : GCCBuiltin<"__builtin_arm_uqsub16">,
    Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty], [IntrNoMem]>;
// Writes to the GE bits.
def int_arm_usax  : GCCBuiltin<"__builtin_arm_usax">,
    Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty], []>;
// Writes to the GE bits.
def int_arm_usub16  : GCCBuiltin<"__builtin_arm_usub16">,
    Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty], []>;

// Parallel 16-bit multiplication
def int_arm_smlad : GCCBuiltin<"__builtin_arm_smlad">,
    Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
              [IntrNoMem]>;
def int_arm_smladx : GCCBuiltin<"__builtin_arm_smladx">,
    Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
              [IntrNoMem]>;
def int_arm_smlald : GCCBuiltin<"__builtin_arm_smlald">,
    Intrinsic<[llvm_i64_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i64_ty],
              [IntrNoMem]>;
def int_arm_smlaldx : GCCBuiltin<"__builtin_arm_smlaldx">,
    Intrinsic<[llvm_i64_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i64_ty],
              [IntrNoMem]>;
def int_arm_smlsd : GCCBuiltin<"__builtin_arm_smlsd">,
    Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
              [IntrNoMem]>;
def int_arm_smlsdx : GCCBuiltin<"__builtin_arm_smlsdx">,
    Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
              [IntrNoMem]>;
def int_arm_smlsld : GCCBuiltin<"__builtin_arm_smlsld">,
    Intrinsic<[llvm_i64_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i64_ty],
              [IntrNoMem]>;
def int_arm_smlsldx : GCCBuiltin<"__builtin_arm_smlsldx">,
    Intrinsic<[llvm_i64_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i64_ty],
              [IntrNoMem]>;
def int_arm_smuad : GCCBuiltin<"__builtin_arm_smuad">,
    Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty], [IntrNoMem]>;
def int_arm_smuadx : GCCBuiltin<"__builtin_arm_smuadx">,
    Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty], [IntrNoMem]>;
def int_arm_smusd : GCCBuiltin<"__builtin_arm_smusd">,
    Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty], [IntrNoMem]>;
def int_arm_smusdx : GCCBuiltin<"__builtin_arm_smusdx">,
    Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty], [IntrNoMem]>;


//===----------------------------------------------------------------------===//
// Load, Store and Clear exclusive

def int_arm_ldrex : Intrinsic<[llvm_i32_ty], [llvm_anyptr_ty]>;
def int_arm_strex : Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_anyptr_ty]>;

def int_arm_ldaex : Intrinsic<[llvm_i32_ty], [llvm_anyptr_ty]>;
def int_arm_stlex : Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_anyptr_ty]>;

def int_arm_clrex : Intrinsic<[]>;

def int_arm_strexd : Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty,
    llvm_ptr_ty]>;
def int_arm_ldrexd : Intrinsic<[llvm_i32_ty, llvm_i32_ty], [llvm_ptr_ty]>;

def int_arm_stlexd : Intrinsic<[llvm_i32_ty],
                               [llvm_i32_ty, llvm_i32_ty, llvm_ptr_ty]>;
def int_arm_ldaexd : Intrinsic<[llvm_i32_ty, llvm_i32_ty], [llvm_ptr_ty]>;

//===----------------------------------------------------------------------===//
// Data barrier instructions
def int_arm_dmb : GCCBuiltin<"__builtin_arm_dmb">, MSBuiltin<"__dmb">,
                  Intrinsic<[], [llvm_i32_ty]>;
def int_arm_dsb : GCCBuiltin<"__builtin_arm_dsb">, MSBuiltin<"__dsb">,
                  Intrinsic<[], [llvm_i32_ty]>;
def int_arm_isb : GCCBuiltin<"__builtin_arm_isb">, MSBuiltin<"__isb">,
                  Intrinsic<[], [llvm_i32_ty]>;

//===----------------------------------------------------------------------===//
// VFP

def int_arm_get_fpscr : GCCBuiltin<"__builtin_arm_get_fpscr">,
                       Intrinsic<[llvm_i32_ty], [], []>;
def int_arm_set_fpscr : GCCBuiltin<"__builtin_arm_set_fpscr">,
                       Intrinsic<[], [llvm_i32_ty], []>;
def int_arm_vcvtr     : Intrinsic<[llvm_float_ty], [llvm_anyfloat_ty],
                                  [IntrNoMem]>;
def int_arm_vcvtru    : Intrinsic<[llvm_float_ty], [llvm_anyfloat_ty],
                                  [IntrNoMem]>;

//===----------------------------------------------------------------------===//
// Coprocessor

def int_arm_ldc : GCCBuiltin<"__builtin_arm_ldc">,
   Intrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_ptr_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>]>;
def int_arm_ldcl : GCCBuiltin<"__builtin_arm_ldcl">,
   Intrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_ptr_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>]>;
def int_arm_ldc2 : GCCBuiltin<"__builtin_arm_ldc2">,
   Intrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_ptr_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>]>;
def int_arm_ldc2l : GCCBuiltin<"__builtin_arm_ldc2l">,
   Intrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_ptr_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>]>;

def int_arm_stc : GCCBuiltin<"__builtin_arm_stc">,
   Intrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_ptr_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>]>;
def int_arm_stcl : GCCBuiltin<"__builtin_arm_stcl">,
   Intrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_ptr_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>]>;
def int_arm_stc2 : GCCBuiltin<"__builtin_arm_stc2">,
   Intrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_ptr_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>]>;
def int_arm_stc2l : GCCBuiltin<"__builtin_arm_stc2l">,
   Intrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_ptr_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>]>;

// Move to coprocessor
def int_arm_mcr : GCCBuiltin<"__builtin_arm_mcr">,
   Intrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty,
                  llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<3>>, ImmArg<ArgIndex<4>>, ImmArg<ArgIndex<5>>]>;
def int_arm_mcr2 : GCCBuiltin<"__builtin_arm_mcr2">,
   Intrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty,
                  llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<3>>, ImmArg<ArgIndex<4>>, ImmArg<ArgIndex<5>>]>;

// Move from coprocessor
def int_arm_mrc : GCCBuiltin<"__builtin_arm_mrc">,
                  MSBuiltin<"_MoveFromCoprocessor">,
   Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty,
                             llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>, ImmArg<ArgIndex<3>>, ImmArg<ArgIndex<4>>]>;
def int_arm_mrc2 : GCCBuiltin<"__builtin_arm_mrc2">,
                   MSBuiltin<"_MoveFromCoprocessor2">,
   Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty,
                             llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>, ImmArg<ArgIndex<3>>, ImmArg<ArgIndex<4>>]>;

// Coprocessor data processing
def int_arm_cdp : GCCBuiltin<"__builtin_arm_cdp">,
   Intrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty,
                  llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>, ImmArg<ArgIndex<3>>, ImmArg<ArgIndex<4>>, ImmArg<ArgIndex<5>>]>;
def int_arm_cdp2 : GCCBuiltin<"__builtin_arm_cdp2">,
   Intrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty,
                  llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>, ImmArg<ArgIndex<3>>, ImmArg<ArgIndex<4>>, ImmArg<ArgIndex<5>>]>;

// Move from two registers to coprocessor
def int_arm_mcrr : Intrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty,
                                  llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<4>>]>;
def int_arm_mcrr2 : Intrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty,
                                   llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<4>>]>;

def int_arm_mrrc : Intrinsic<[llvm_i32_ty, llvm_i32_ty], [llvm_i32_ty,
                              llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>]>;
def int_arm_mrrc2 : Intrinsic<[llvm_i32_ty, llvm_i32_ty], [llvm_i32_ty,
                               llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>]>;

//===----------------------------------------------------------------------===//
// CRC32

def int_arm_crc32b  : Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty],
    [IntrNoMem]>;
def int_arm_crc32cb : Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty],
    [IntrNoMem]>;
def int_arm_crc32h  : Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty],
    [IntrNoMem]>;
def int_arm_crc32ch : Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty],
    [IntrNoMem]>;
def int_arm_crc32w  : Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty],
    [IntrNoMem]>;
def int_arm_crc32cw : Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty],
    [IntrNoMem]>;

//===----------------------------------------------------------------------===//
// CMSE

def int_arm_cmse_tt : GCCBuiltin<"__builtin_arm_cmse_TT">,
    Intrinsic<[llvm_i32_ty], [llvm_ptr_ty], [IntrNoMem]>;
def int_arm_cmse_ttt : GCCBuiltin<"__builtin_arm_cmse_TTT">,
    Intrinsic<[llvm_i32_ty], [llvm_ptr_ty], [IntrNoMem]>;
def int_arm_cmse_tta : GCCBuiltin<"__builtin_arm_cmse_TTA">,
    Intrinsic<[llvm_i32_ty], [llvm_ptr_ty], [IntrNoMem]>;
def int_arm_cmse_ttat : GCCBuiltin<"__builtin_arm_cmse_TTAT">,
    Intrinsic<[llvm_i32_ty], [llvm_ptr_ty], [IntrNoMem]>;

//===----------------------------------------------------------------------===//
// HINT

def int_arm_hint : Intrinsic<[], [llvm_i32_ty]>;
def int_arm_dbg : Intrinsic<[], [llvm_i32_ty]>;

//===----------------------------------------------------------------------===//
// UND (reserved undefined sequence)

def int_arm_undefined : Intrinsic<[], [llvm_i32_ty]>;

//===----------------------------------------------------------------------===//
// Advanced SIMD (NEON)

// The following classes do not correspond directly to GCC builtins.
class Neon_1Arg_Intrinsic
  : Intrinsic<[llvm_anyvector_ty], [LLVMMatchType<0>], [IntrNoMem]>;
class Neon_1Arg_Narrow_Intrinsic
  : Intrinsic<[llvm_anyvector_ty], [LLVMExtendedType<0>], [IntrNoMem]>;
class Neon_2Arg_Intrinsic
  : Intrinsic<[llvm_anyvector_ty], [LLVMMatchType<0>, LLVMMatchType<0>],
              [IntrNoMem]>;
class Neon_2Arg_Narrow_Intrinsic
  : Intrinsic<[llvm_anyvector_ty], [LLVMExtendedType<0>, LLVMExtendedType<0>],
              [IntrNoMem]>;
class Neon_2Arg_Long_Intrinsic
  : Intrinsic<[llvm_anyvector_ty], [LLVMTruncatedType<0>, LLVMTruncatedType<0>],
              [IntrNoMem]>;
class Neon_3Arg_Intrinsic
  : Intrinsic<[llvm_anyvector_ty],
              [LLVMMatchType<0>, LLVMMatchType<0>, LLVMMatchType<0>],
              [IntrNoMem]>;
class Neon_3Arg_Long_Intrinsic
  : Intrinsic<[llvm_anyvector_ty],
              [LLVMMatchType<0>, LLVMTruncatedType<0>, LLVMTruncatedType<0>],
              [IntrNoMem]>;

class Neon_1FloatArg_Intrinsic
  : Intrinsic<[llvm_anyfloat_ty], [LLVMMatchType<0>], [IntrNoMem]>;

class Neon_CvtFxToFP_Intrinsic
  : Intrinsic<[llvm_anyfloat_ty], [llvm_anyint_ty, llvm_i32_ty], [IntrNoMem]>;
class Neon_CvtFPToFx_Intrinsic
  : Intrinsic<[llvm_anyint_ty], [llvm_anyfloat_ty, llvm_i32_ty], [IntrNoMem]>;
class Neon_CvtFPtoInt_1Arg_Intrinsic
  : Intrinsic<[llvm_anyvector_ty], [llvm_anyvector_ty], [IntrNoMem]>;

class Neon_Compare_Intrinsic
  : Intrinsic<[llvm_anyvector_ty], [llvm_anyvector_ty, LLVMMatchType<1>],
              [IntrNoMem]>;

// The table operands for VTBL and VTBX consist of 1 to 4 v8i8 vectors.
// Besides the table, VTBL has one other v8i8 argument and VTBX has two.
// Overall, the classes range from 2 to 6 v8i8 arguments.
class Neon_Tbl2Arg_Intrinsic
  : Intrinsic<[llvm_v8i8_ty],
              [llvm_v8i8_ty, llvm_v8i8_ty], [IntrNoMem]>;
class Neon_Tbl3Arg_Intrinsic
  : Intrinsic<[llvm_v8i8_ty],
              [llvm_v8i8_ty, llvm_v8i8_ty, llvm_v8i8_ty], [IntrNoMem]>;
class Neon_Tbl4Arg_Intrinsic
  : Intrinsic<[llvm_v8i8_ty],
              [llvm_v8i8_ty, llvm_v8i8_ty, llvm_v8i8_ty, llvm_v8i8_ty],
              [IntrNoMem]>;
class Neon_Tbl5Arg_Intrinsic
  : Intrinsic<[llvm_v8i8_ty],
              [llvm_v8i8_ty, llvm_v8i8_ty, llvm_v8i8_ty, llvm_v8i8_ty,
               llvm_v8i8_ty], [IntrNoMem]>;
class Neon_Tbl6Arg_Intrinsic
  : Intrinsic<[llvm_v8i8_ty],
              [llvm_v8i8_ty, llvm_v8i8_ty, llvm_v8i8_ty, llvm_v8i8_ty,
               llvm_v8i8_ty, llvm_v8i8_ty], [IntrNoMem]>;

// Arithmetic ops

let IntrProperties = [IntrNoMem, Commutative] in {

  // Vector Add.
  def int_arm_neon_vhadds : Neon_2Arg_Intrinsic;
  def int_arm_neon_vhaddu : Neon_2Arg_Intrinsic;
  def int_arm_neon_vrhadds : Neon_2Arg_Intrinsic;
  def int_arm_neon_vrhaddu : Neon_2Arg_Intrinsic;
  def int_arm_neon_vraddhn : Neon_2Arg_Narrow_Intrinsic;

  // Vector Multiply.
  def int_arm_neon_vmulp : Neon_2Arg_Intrinsic;
  def int_arm_neon_vqdmulh : Neon_2Arg_Intrinsic;
  def int_arm_neon_vqrdmulh : Neon_2Arg_Intrinsic;
  def int_arm_neon_vmulls : Neon_2Arg_Long_Intrinsic;
  def int_arm_neon_vmullu : Neon_2Arg_Long_Intrinsic;
  def int_arm_neon_vmullp : Neon_2Arg_Long_Intrinsic;
  def int_arm_neon_vqdmull : Neon_2Arg_Long_Intrinsic;

  // Vector Maximum.
  def int_arm_neon_vmaxs : Neon_2Arg_Intrinsic;
  def int_arm_neon_vmaxu : Neon_2Arg_Intrinsic;
  def int_arm_neon_vmaxnm : Neon_2Arg_Intrinsic;

  // Vector Minimum.
  def int_arm_neon_vmins : Neon_2Arg_Intrinsic;
  def int_arm_neon_vminu : Neon_2Arg_Intrinsic;
  def int_arm_neon_vminnm : Neon_2Arg_Intrinsic;

  // Vector Reciprocal Step.
  def int_arm_neon_vrecps : Neon_2Arg_Intrinsic;

  // Vector Reciprocal Square Root Step.
  def int_arm_neon_vrsqrts : Neon_2Arg_Intrinsic;
}

// Vector Subtract.
def int_arm_neon_vhsubs : Neon_2Arg_Intrinsic;
def int_arm_neon_vhsubu : Neon_2Arg_Intrinsic;
def int_arm_neon_vrsubhn : Neon_2Arg_Narrow_Intrinsic;

// Vector Absolute Compare.
def int_arm_neon_vacge : Neon_Compare_Intrinsic;
def int_arm_neon_vacgt : Neon_Compare_Intrinsic;

// Vector Absolute Differences.
def int_arm_neon_vabds : Neon_2Arg_Intrinsic;
def int_arm_neon_vabdu : Neon_2Arg_Intrinsic;

// Vector Pairwise Add.
def int_arm_neon_vpadd : Neon_2Arg_Intrinsic;

// Vector Pairwise Add Long.
// Note: This is different than the other "long" NEON intrinsics because
// the result vector has half as many elements as the source vector.
// The source and destination vector types must be specified separately.
def int_arm_neon_vpaddls : Intrinsic<[llvm_anyvector_ty], [llvm_anyvector_ty],
                                     [IntrNoMem]>;
def int_arm_neon_vpaddlu : Intrinsic<[llvm_anyvector_ty], [llvm_anyvector_ty],
                                     [IntrNoMem]>;

// Vector Pairwise Add and Accumulate Long.
// Note: This is similar to vpaddl but the destination vector also appears
// as the first argument.
def int_arm_neon_vpadals : Intrinsic<[llvm_anyvector_ty],
                                     [LLVMMatchType<0>, llvm_anyvector_ty],
                                     [IntrNoMem]>;
def int_arm_neon_vpadalu : Intrinsic<[llvm_anyvector_ty],
                                     [LLVMMatchType<0>, llvm_anyvector_ty],
                                     [IntrNoMem]>;

// Vector Pairwise Maximum and Minimum.
def int_arm_neon_vpmaxs : Neon_2Arg_Intrinsic;
def int_arm_neon_vpmaxu : Neon_2Arg_Intrinsic;
def int_arm_neon_vpmins : Neon_2Arg_Intrinsic;
def int_arm_neon_vpminu : Neon_2Arg_Intrinsic;

// Vector Shifts:
//
// The various saturating and rounding vector shift operations need to be
// represented by intrinsics in LLVM, and even the basic VSHL variable shift
// operation cannot be safely translated to LLVM's shift operators.  VSHL can
// be used for both left and right shifts, or even combinations of the two,
// depending on the signs of the shift amounts.  It also has well-defined
// behavior for shift amounts that LLVM leaves undefined.  Only basic shifts
// by constants can be represented with LLVM's shift operators.
//
// The shift counts for these intrinsics are always vectors, even for constant
// shifts, where the constant is replicated.  For consistency with VSHL (and
// other variable shift instructions), left shifts have positive shift counts
// and right shifts have negative shift counts.  This convention is also used
// for constant right shift intrinsics, and to help preserve sanity, the
// intrinsic names use "shift" instead of either "shl" or "shr".  Where
// applicable, signed and unsigned versions of the intrinsics are
// distinguished with "s" and "u" suffixes.  A few NEON shift instructions,
// such as VQSHLU, take signed operands but produce unsigned results; these
// use a "su" suffix.

// Vector Shift.
def int_arm_neon_vshifts : Neon_2Arg_Intrinsic;
def int_arm_neon_vshiftu : Neon_2Arg_Intrinsic;

// Vector Rounding Shift.
def int_arm_neon_vrshifts : Neon_2Arg_Intrinsic;
def int_arm_neon_vrshiftu : Neon_2Arg_Intrinsic;
def int_arm_neon_vrshiftn : Neon_2Arg_Narrow_Intrinsic;

// Vector Saturating Shift.
def int_arm_neon_vqshifts : Neon_2Arg_Intrinsic;
def int_arm_neon_vqshiftu : Neon_2Arg_Intrinsic;
def int_arm_neon_vqshiftsu : Neon_2Arg_Intrinsic;
def int_arm_neon_vqshiftns : Neon_2Arg_Narrow_Intrinsic;
def int_arm_neon_vqshiftnu : Neon_2Arg_Narrow_Intrinsic;
def int_arm_neon_vqshiftnsu : Neon_2Arg_Narrow_Intrinsic;

// Vector Saturating Rounding Shift.
def int_arm_neon_vqrshifts : Neon_2Arg_Intrinsic;
def int_arm_neon_vqrshiftu : Neon_2Arg_Intrinsic;
def int_arm_neon_vqrshiftns : Neon_2Arg_Narrow_Intrinsic;
def int_arm_neon_vqrshiftnu : Neon_2Arg_Narrow_Intrinsic;
def int_arm_neon_vqrshiftnsu : Neon_2Arg_Narrow_Intrinsic;

// Vector Shift and Insert.
def int_arm_neon_vshiftins : Neon_3Arg_Intrinsic;

// Vector Absolute Value and Saturating Absolute Value.
def int_arm_neon_vabs : Neon_1Arg_Intrinsic;
def int_arm_neon_vqabs : Neon_1Arg_Intrinsic;

// Vector Saturating Negate.
def int_arm_neon_vqneg : Neon_1Arg_Intrinsic;

// Vector Count Leading Sign/Zero Bits.
def int_arm_neon_vcls : Neon_1Arg_Intrinsic;

// Vector Reciprocal Estimate.
def int_arm_neon_vrecpe : Neon_1Arg_Intrinsic;

// Vector Reciprocal Square Root Estimate.
def int_arm_neon_vrsqrte : Neon_1Arg_Intrinsic;

// Vector Conversions Between Floating-point and Integer
def int_arm_neon_vcvtau : Neon_CvtFPtoInt_1Arg_Intrinsic;
def int_arm_neon_vcvtas : Neon_CvtFPtoInt_1Arg_Intrinsic;
def int_arm_neon_vcvtnu : Neon_CvtFPtoInt_1Arg_Intrinsic;
def int_arm_neon_vcvtns : Neon_CvtFPtoInt_1Arg_Intrinsic;
def int_arm_neon_vcvtpu : Neon_CvtFPtoInt_1Arg_Intrinsic;
def int_arm_neon_vcvtps : Neon_CvtFPtoInt_1Arg_Intrinsic;
def int_arm_neon_vcvtmu : Neon_CvtFPtoInt_1Arg_Intrinsic;
def int_arm_neon_vcvtms : Neon_CvtFPtoInt_1Arg_Intrinsic;

// Vector Conversions Between Floating-point and Fixed-point.
def int_arm_neon_vcvtfp2fxs : Neon_CvtFPToFx_Intrinsic;
def int_arm_neon_vcvtfp2fxu : Neon_CvtFPToFx_Intrinsic;
def int_arm_neon_vcvtfxs2fp : Neon_CvtFxToFP_Intrinsic;
def int_arm_neon_vcvtfxu2fp : Neon_CvtFxToFP_Intrinsic;

// Vector Conversions Between Half-Precision and Single-Precision.
def int_arm_neon_vcvtfp2hf
    : Intrinsic<[llvm_v4i16_ty], [llvm_v4f32_ty], [IntrNoMem]>;
def int_arm_neon_vcvthf2fp
    : Intrinsic<[llvm_v4f32_ty], [llvm_v4i16_ty], [IntrNoMem]>;

// Narrowing Saturating Vector Moves.
def int_arm_neon_vqmovns : Neon_1Arg_Narrow_Intrinsic;
def int_arm_neon_vqmovnu : Neon_1Arg_Narrow_Intrinsic;
def int_arm_neon_vqmovnsu : Neon_1Arg_Narrow_Intrinsic;

// Vector Table Lookup.
// The first 1-4 arguments are the table.
def int_arm_neon_vtbl1 : Neon_Tbl2Arg_Intrinsic;
def int_arm_neon_vtbl2 : Neon_Tbl3Arg_Intrinsic;
def int_arm_neon_vtbl3 : Neon_Tbl4Arg_Intrinsic;
def int_arm_neon_vtbl4 : Neon_Tbl5Arg_Intrinsic;

// Vector Table Extension.
// Some elements of the destination vector may not be updated, so the original
// value of that vector is passed as the first argument.  The next 1-4
// arguments after that are the table.
def int_arm_neon_vtbx1 : Neon_Tbl3Arg_Intrinsic;
def int_arm_neon_vtbx2 : Neon_Tbl4Arg_Intrinsic;
def int_arm_neon_vtbx3 : Neon_Tbl5Arg_Intrinsic;
def int_arm_neon_vtbx4 : Neon_Tbl6Arg_Intrinsic;

// Vector and Scalar Rounding.
def int_arm_neon_vrintn : Neon_1FloatArg_Intrinsic;
def int_arm_neon_vrintx : Neon_1Arg_Intrinsic;
def int_arm_neon_vrinta : Neon_1Arg_Intrinsic;
def int_arm_neon_vrintz : Neon_1Arg_Intrinsic;
def int_arm_neon_vrintm : Neon_1Arg_Intrinsic;
def int_arm_neon_vrintp : Neon_1Arg_Intrinsic;

// De-interleaving vector loads from N-element structures.
// Source operands are the address and alignment.
def int_arm_neon_vld1 : Intrinsic<[llvm_anyvector_ty],
                                  [llvm_anyptr_ty, llvm_i32_ty],
                                  [IntrReadMem, IntrArgMemOnly]>;
def int_arm_neon_vld2 : Intrinsic<[llvm_anyvector_ty, LLVMMatchType<0>],
                                  [llvm_anyptr_ty, llvm_i32_ty],
                                  [IntrReadMem, IntrArgMemOnly]>;
def int_arm_neon_vld3 : Intrinsic<[llvm_anyvector_ty, LLVMMatchType<0>,
                                   LLVMMatchType<0>],
                                  [llvm_anyptr_ty, llvm_i32_ty],
                                  [IntrReadMem, IntrArgMemOnly]>;
def int_arm_neon_vld4 : Intrinsic<[llvm_anyvector_ty, LLVMMatchType<0>,
                                   LLVMMatchType<0>, LLVMMatchType<0>],
                                  [llvm_anyptr_ty, llvm_i32_ty],
                                  [IntrReadMem, IntrArgMemOnly]>;

def int_arm_neon_vld1x2 : Intrinsic<[llvm_anyvector_ty, LLVMMatchType<0>],
                                    [LLVMAnyPointerType<LLVMMatchType<0>>],
                                    [IntrReadMem, IntrArgMemOnly]>;
def int_arm_neon_vld1x3 : Intrinsic<[llvm_anyvector_ty, LLVMMatchType<0>,
                                     LLVMMatchType<0>],
                                    [LLVMAnyPointerType<LLVMMatchType<0>>],
                                    [IntrReadMem, IntrArgMemOnly]>;
def int_arm_neon_vld1x4 : Intrinsic<[llvm_anyvector_ty, LLVMMatchType<0>,
                                     LLVMMatchType<0>, LLVMMatchType<0>],
                                    [LLVMAnyPointerType<LLVMMatchType<0>>],
                                    [IntrReadMem, IntrArgMemOnly]>;

// Vector load N-element structure to one lane.
// Source operands are: the address, the N input vectors (since only one
// lane is assigned), the lane number, and the alignment.
def int_arm_neon_vld2lane : Intrinsic<[llvm_anyvector_ty, LLVMMatchType<0>],
                                      [llvm_anyptr_ty, LLVMMatchType<0>,
                                       LLVMMatchType<0>, llvm_i32_ty,
                                       llvm_i32_ty], [IntrReadMem, IntrArgMemOnly]>;
def int_arm_neon_vld3lane : Intrinsic<[llvm_anyvector_ty, LLVMMatchType<0>,
                                       LLVMMatchType<0>],
                                      [llvm_anyptr_ty, LLVMMatchType<0>,
                                       LLVMMatchType<0>, LLVMMatchType<0>,
                                       llvm_i32_ty, llvm_i32_ty],
                                      [IntrReadMem, IntrArgMemOnly]>;
def int_arm_neon_vld4lane : Intrinsic<[llvm_anyvector_ty, LLVMMatchType<0>,
                                       LLVMMatchType<0>, LLVMMatchType<0>],
                                      [llvm_anyptr_ty, LLVMMatchType<0>,
                                       LLVMMatchType<0>, LLVMMatchType<0>,
                                       LLVMMatchType<0>, llvm_i32_ty,
                                       llvm_i32_ty], [IntrReadMem, IntrArgMemOnly]>;

// Vector load N-element structure to all lanes.
// Source operands are the address and alignment.
def int_arm_neon_vld2dup : Intrinsic<[llvm_anyvector_ty, LLVMMatchType<0>],
                                     [llvm_anyptr_ty, llvm_i32_ty],
                                     [IntrReadMem, IntrArgMemOnly]>;
def int_arm_neon_vld3dup : Intrinsic<[llvm_anyvector_ty, LLVMMatchType<0>,
                                      LLVMMatchType<0>],
                                     [llvm_anyptr_ty, llvm_i32_ty],
                                     [IntrReadMem, IntrArgMemOnly]>;
def int_arm_neon_vld4dup : Intrinsic<[llvm_anyvector_ty, LLVMMatchType<0>,
                                      LLVMMatchType<0>, LLVMMatchType<0>],
                                     [llvm_anyptr_ty, llvm_i32_ty],
                                     [IntrReadMem, IntrArgMemOnly]>;

// Interleaving vector stores from N-element structures.
// Source operands are: the address, the N vectors, and the alignment.
def int_arm_neon_vst1 : Intrinsic<[],
                                  [llvm_anyptr_ty, llvm_anyvector_ty,
                                   llvm_i32_ty], [IntrArgMemOnly]>;
def int_arm_neon_vst2 : Intrinsic<[],
                                  [llvm_anyptr_ty, llvm_anyvector_ty,
                                   LLVMMatchType<1>, llvm_i32_ty],
                                  [IntrArgMemOnly]>;
def int_arm_neon_vst3 : Intrinsic<[],
                                  [llvm_anyptr_ty, llvm_anyvector_ty,
                                   LLVMMatchType<1>, LLVMMatchType<1>,
                                   llvm_i32_ty], [IntrArgMemOnly]>;
def int_arm_neon_vst4 : Intrinsic<[],
                                  [llvm_anyptr_ty, llvm_anyvector_ty,
                                   LLVMMatchType<1>, LLVMMatchType<1>,
                                   LLVMMatchType<1>, llvm_i32_ty],
                                  [IntrArgMemOnly]>;

def int_arm_neon_vst1x2 : Intrinsic<[],
                                    [llvm_anyptr_ty, llvm_anyvector_ty,
                                     LLVMMatchType<1>],
                                    [IntrArgMemOnly, NoCapture<ArgIndex<0>>]>;
def int_arm_neon_vst1x3 : Intrinsic<[],
                                    [llvm_anyptr_ty, llvm_anyvector_ty,
                                     LLVMMatchType<1>, LLVMMatchType<1>],
                                    [IntrArgMemOnly, NoCapture<ArgIndex<0>>]>;
def int_arm_neon_vst1x4 : Intrinsic<[],
                                    [llvm_anyptr_ty, llvm_anyvector_ty,
                                     LLVMMatchType<1>, LLVMMatchType<1>,
                                     LLVMMatchType<1>],
                                    [IntrArgMemOnly, NoCapture<ArgIndex<0>>]>;

// Vector store N-element structure from one lane.
// Source operands are: the address, the N vectors, the lane number, and
// the alignment.
def int_arm_neon_vst2lane : Intrinsic<[],
                                      [llvm_anyptr_ty, llvm_anyvector_ty,
                                       LLVMMatchType<1>, llvm_i32_ty,
                                       llvm_i32_ty], [IntrArgMemOnly]>;
def int_arm_neon_vst3lane : Intrinsic<[],
                                      [llvm_anyptr_ty, llvm_anyvector_ty,
                                       LLVMMatchType<1>, LLVMMatchType<1>,
                                       llvm_i32_ty, llvm_i32_ty],
                                      [IntrArgMemOnly]>;
def int_arm_neon_vst4lane : Intrinsic<[],
                                      [llvm_anyptr_ty, llvm_anyvector_ty,
                                       LLVMMatchType<1>, LLVMMatchType<1>,
                                       LLVMMatchType<1>, llvm_i32_ty,
                                       llvm_i32_ty], [IntrArgMemOnly]>;

// Vector bitwise select.
def int_arm_neon_vbsl : Intrinsic<[llvm_anyvector_ty],
                        [LLVMMatchType<0>, LLVMMatchType<0>, LLVMMatchType<0>],
                        [IntrNoMem]>;


// Crypto instructions
class AES_1Arg_Intrinsic : Intrinsic<[llvm_v16i8_ty],
                                     [llvm_v16i8_ty], [IntrNoMem]>;
class AES_2Arg_Intrinsic : Intrinsic<[llvm_v16i8_ty],
                                     [llvm_v16i8_ty, llvm_v16i8_ty],
                                     [IntrNoMem]>;

class SHA_1Arg_Intrinsic : Intrinsic<[llvm_i32_ty], [llvm_i32_ty],
                                     [IntrNoMem]>;
class SHA_2Arg_Intrinsic : Intrinsic<[llvm_v4i32_ty],
                                     [llvm_v4i32_ty, llvm_v4i32_ty],
                                     [IntrNoMem]>;
class SHA_3Arg_i32_Intrinsic : Intrinsic<[llvm_v4i32_ty],
                                   [llvm_v4i32_ty, llvm_i32_ty, llvm_v4i32_ty],
                                   [IntrNoMem]>;
class SHA_3Arg_v4i32_Intrinsic : Intrinsic<[llvm_v4i32_ty],
                                   [llvm_v4i32_ty, llvm_v4i32_ty,llvm_v4i32_ty],
                                   [IntrNoMem]>;

def int_arm_neon_aesd : AES_2Arg_Intrinsic;
def int_arm_neon_aese : AES_2Arg_Intrinsic;
def int_arm_neon_aesimc : AES_1Arg_Intrinsic;
def int_arm_neon_aesmc : AES_1Arg_Intrinsic;
def int_arm_neon_sha1h : SHA_1Arg_Intrinsic;
def int_arm_neon_sha1su1 : SHA_2Arg_Intrinsic;
def int_arm_neon_sha256su0 : SHA_2Arg_Intrinsic;
def int_arm_neon_sha1c : SHA_3Arg_i32_Intrinsic;
def int_arm_neon_sha1m : SHA_3Arg_i32_Intrinsic;
def int_arm_neon_sha1p : SHA_3Arg_i32_Intrinsic;
def int_arm_neon_sha1su0: SHA_3Arg_v4i32_Intrinsic;
def int_arm_neon_sha256h: SHA_3Arg_v4i32_Intrinsic;
def int_arm_neon_sha256h2: SHA_3Arg_v4i32_Intrinsic;
def int_arm_neon_sha256su1: SHA_3Arg_v4i32_Intrinsic;

// Armv8.2-A dot product instructions
class Neon_Dot_Intrinsic
  : Intrinsic<[llvm_anyvector_ty],
              [LLVMMatchType<0>, llvm_anyvector_ty,
               LLVMMatchType<1>],
              [IntrNoMem]>;
def int_arm_neon_udot : Neon_Dot_Intrinsic;
def int_arm_neon_sdot : Neon_Dot_Intrinsic;

// v8.6-A Matrix Multiply Intrinsics
class Neon_MatMul_Intrinsic
  : Intrinsic<[llvm_anyvector_ty],
              [LLVMMatchType<0>, llvm_anyvector_ty,
               LLVMMatchType<1>],
              [IntrNoMem]>;
def int_arm_neon_ummla  : Neon_MatMul_Intrinsic;
def int_arm_neon_smmla  : Neon_MatMul_Intrinsic;
def int_arm_neon_usmmla : Neon_MatMul_Intrinsic;
def int_arm_neon_usdot  : Neon_Dot_Intrinsic;

// v8.6-A Bfloat Intrinsics
def int_arm_neon_vcvtfp2bf
    : Intrinsic<[llvm_anyvector_ty], [llvm_v4f32_ty], [IntrNoMem]>;
def int_arm_neon_vcvtbfp2bf
    : Intrinsic<[llvm_bfloat_ty], [llvm_float_ty], [IntrNoMem]>;

def int_arm_neon_bfdot : Neon_Dot_Intrinsic;
def int_arm_neon_bfmmla
    : Intrinsic<[llvm_v4f32_ty],
                [llvm_v4f32_ty, llvm_v8bf16_ty, llvm_v8bf16_ty],
                [IntrNoMem]>;

class Neon_BF16FML_Intrinsic
    : Intrinsic<[llvm_v4f32_ty],
                [llvm_v4f32_ty, llvm_v8bf16_ty, llvm_v8bf16_ty],
                [IntrNoMem]>;
def int_arm_neon_bfmlalb : Neon_BF16FML_Intrinsic;
def int_arm_neon_bfmlalt : Neon_BF16FML_Intrinsic;

def int_arm_cls: Intrinsic<[llvm_i32_ty], [llvm_i32_ty], [IntrNoMem]>;
def int_arm_cls64: Intrinsic<[llvm_i32_ty], [llvm_i64_ty], [IntrNoMem]>;

def int_arm_mve_vctp8  : Intrinsic<[llvm_v16i1_ty], [llvm_i32_ty], [IntrNoMem]>;
def int_arm_mve_vctp16 : Intrinsic<[llvm_v8i1_ty], [llvm_i32_ty], [IntrNoMem]>;
def int_arm_mve_vctp32 : Intrinsic<[llvm_v4i1_ty], [llvm_i32_ty], [IntrNoMem]>;
def int_arm_mve_vctp64 : Intrinsic<[llvm_v2i1_ty], [llvm_i32_ty], [IntrNoMem]>;

// v8.3-A Floating-point complex add
def int_arm_neon_vcadd_rot90  : Neon_2Arg_Intrinsic;
def int_arm_neon_vcadd_rot270 : Neon_2Arg_Intrinsic;

// GNU eabi mcount
def int_arm_gnu_eabi_mcount : Intrinsic<[], [], []>;

def int_arm_mve_pred_i2v : Intrinsic<
  [llvm_anyvector_ty], [llvm_i32_ty], [IntrNoMem]>;
def int_arm_mve_pred_v2i : Intrinsic<
  [llvm_i32_ty], [llvm_anyvector_ty], [IntrNoMem]>;
def int_arm_mve_vreinterpretq : Intrinsic<
  [llvm_anyvector_ty], [llvm_anyvector_ty], [IntrNoMem]>;

def int_arm_mve_min_predicated: Intrinsic<[llvm_anyvector_ty],
   [LLVMMatchType<0>, LLVMMatchType<0>, llvm_i32_ty /* unsigned */,
    llvm_anyvector_ty, LLVMMatchType<0>],
   [IntrNoMem]>;
def int_arm_mve_max_predicated: Intrinsic<[llvm_anyvector_ty],
   [LLVMMatchType<0>, LLVMMatchType<0>, llvm_i32_ty /* unsigned */,
    llvm_anyvector_ty, LLVMMatchType<0>],
   [IntrNoMem]>;
def int_arm_mve_abd_predicated: Intrinsic<[llvm_anyvector_ty],
   [LLVMMatchType<0>, LLVMMatchType<0>, llvm_i32_ty /* unsigned */,
    llvm_anyvector_ty, LLVMMatchType<0>], [IntrNoMem]>;
def int_arm_mve_add_predicated: Intrinsic<[llvm_anyvector_ty],
   [LLVMMatchType<0>, LLVMMatchType<0>, llvm_anyvector_ty, LLVMMatchType<0>],
   [IntrNoMem]>;
def int_arm_mve_and_predicated: Intrinsic<[llvm_anyvector_ty],
   [LLVMMatchType<0>, LLVMMatchType<0>, llvm_anyvector_ty, LLVMMatchType<0>],
   [IntrNoMem]>;
def int_arm_mve_bic_predicated: Intrinsic<[llvm_anyvector_ty],
   [LLVMMatchType<0>, LLVMMatchType<0>, llvm_anyvector_ty, LLVMMatchType<0>],
   [IntrNoMem]>;
def int_arm_mve_eor_predicated: Intrinsic<[llvm_anyvector_ty],
   [LLVMMatchType<0>, LLVMMatchType<0>, llvm_anyvector_ty, LLVMMatchType<0>],
   [IntrNoMem]>;
def int_arm_mve_orn_predicated: Intrinsic<[llvm_anyvector_ty],
   [LLVMMatchType<0>, LLVMMatchType<0>, llvm_anyvector_ty, LLVMMatchType<0>],
   [IntrNoMem]>;
def int_arm_mve_orr_predicated: Intrinsic<[llvm_anyvector_ty],
   [LLVMMatchType<0>, LLVMMatchType<0>, llvm_anyvector_ty, LLVMMatchType<0>],
   [IntrNoMem]>;
def int_arm_mve_sub_predicated: Intrinsic<[llvm_anyvector_ty],
   [LLVMMatchType<0>, LLVMMatchType<0>, llvm_anyvector_ty, LLVMMatchType<0>],
   [IntrNoMem]>;
def int_arm_mve_mul_predicated: Intrinsic<[llvm_anyvector_ty],
   [LLVMMatchType<0>, LLVMMatchType<0>, llvm_anyvector_ty, LLVMMatchType<0>],
   [IntrNoMem]>;
def int_arm_mve_mulh_predicated: Intrinsic<[llvm_anyvector_ty],
   [LLVMMatchType<0>, LLVMMatchType<0>, llvm_i32_ty /* unsigned */,
    llvm_anyvector_ty, LLVMMatchType<0>],
   [IntrNoMem]>;
def int_arm_mve_qdmulh_predicated: Intrinsic<[llvm_anyvector_ty],
   [LLVMMatchType<0>, LLVMMatchType<0>, llvm_anyvector_ty, LLVMMatchType<0>],
   [IntrNoMem]>;
def int_arm_mve_rmulh_predicated: Intrinsic<[llvm_anyvector_ty],
   [LLVMMatchType<0>, LLVMMatchType<0>, llvm_i32_ty /* unsigned */,
    llvm_anyvector_ty, LLVMMatchType<0>],
   [IntrNoMem]>;
def int_arm_mve_qrdmulh_predicated: Intrinsic<[llvm_anyvector_ty],
   [LLVMMatchType<0>, LLVMMatchType<0>, llvm_anyvector_ty, LLVMMatchType<0>],
   [IntrNoMem]>;
def int_arm_mve_mull_int_predicated: Intrinsic<[llvm_anyvector_ty],
   [llvm_anyvector_ty, LLVMMatchType<1>, llvm_i32_ty /* unsigned */,
    llvm_i32_ty /* top */, llvm_anyvector_ty, LLVMMatchType<0>],
   [IntrNoMem]>;
def int_arm_mve_mull_poly_predicated: Intrinsic<[llvm_anyvector_ty],
   [llvm_anyvector_ty, LLVMMatchType<1>, llvm_i32_ty, llvm_anyvector_ty,
    LLVMMatchType<0>],
   [IntrNoMem]>;
def int_arm_mve_qadd_predicated: Intrinsic<[llvm_anyvector_ty],
   [LLVMMatchType<0>, LLVMMatchType<0>, llvm_i32_ty /* unsigned */,
    llvm_anyvector_ty, LLVMMatchType<0>], [IntrNoMem]>;
def int_arm_mve_hadd_predicated: Intrinsic<[llvm_anyvector_ty],
   [LLVMMatchType<0>, LLVMMatchType<0>, llvm_i32_ty /* unsigned */,
    llvm_anyvector_ty, LLVMMatchType<0>], [IntrNoMem]>;
def int_arm_mve_rhadd_predicated: Intrinsic<[llvm_anyvector_ty],
   [LLVMMatchType<0>, LLVMMatchType<0>, llvm_i32_ty /* unsigned */,
    llvm_anyvector_ty, LLVMMatchType<0>], [IntrNoMem]>;
def int_arm_mve_qsub_predicated: Intrinsic<[llvm_anyvector_ty],
   [LLVMMatchType<0>, LLVMMatchType<0>, llvm_i32_ty /* unsigned */,
    llvm_anyvector_ty, LLVMMatchType<0>], [IntrNoMem]>;
def int_arm_mve_hsub_predicated: Intrinsic<[llvm_anyvector_ty],
   [LLVMMatchType<0>, LLVMMatchType<0>, llvm_i32_ty /* unsigned */,
    llvm_anyvector_ty, LLVMMatchType<0>], [IntrNoMem]>;
def int_arm_mve_vmina_predicated: Intrinsic<[llvm_anyvector_ty],
   [LLVMMatchType<0>, LLVMMatchType<0>, llvm_anyvector_ty],
    [IntrNoMem]>;
def int_arm_mve_vmaxa_predicated: Intrinsic<[llvm_anyvector_ty],
   [LLVMMatchType<0>, LLVMMatchType<0>, llvm_anyvector_ty],
    [IntrNoMem]>;
def int_arm_mve_vminnma_predicated: Intrinsic<[llvm_anyvector_ty],
   [LLVMMatchType<0>, LLVMMatchType<0>, llvm_anyvector_ty],
    [IntrNoMem]>;
def int_arm_mve_vmaxnma_predicated: Intrinsic<[llvm_anyvector_ty],
   [LLVMMatchType<0>, LLVMMatchType<0>, llvm_anyvector_ty],
    [IntrNoMem]>;

multiclass MVEPredicated<list<LLVMType> rets, list<LLVMType> params,
                         LLVMType pred = llvm_anyvector_ty,
                         list<IntrinsicProperty> props = [IntrNoMem],
                         list<SDNodeProperty> sdprops = []> {
  def "": Intrinsic<rets, params, props, "", sdprops>;
  def _predicated: Intrinsic<rets, params # [pred], props, "", sdprops>;
}
multiclass MVEPredicatedM<list<LLVMType> rets, list<LLVMType> params,
                          LLVMType pred = llvm_anyvector_ty,
                          list<IntrinsicProperty> props = [IntrNoMem]> {
  def "": Intrinsic<rets, params, props>;
  def _predicated: Intrinsic<rets, params # [pred,
      !if(!eq(rets[0], llvm_anyvector_ty),
          LLVMMatchType<0>, rets[0])], props>;
}

multiclass MVE_minmaxv {
  defm v: MVEPredicated<[llvm_i32_ty],
     [llvm_i32_ty, llvm_anyvector_ty, llvm_i32_ty /* unsigned */]>;
  defm av: MVEPredicated<[llvm_i32_ty],
     [llvm_i32_ty, llvm_anyvector_ty]>;
  defm nmv: MVEPredicated<[llvm_anyfloat_ty],
     [LLVMMatchType<0>, llvm_anyvector_ty]>;
  defm nmav: MVEPredicated<[llvm_anyfloat_ty],
     [LLVMMatchType<0>, llvm_anyvector_ty]>;
}
defm int_arm_mve_min: MVE_minmaxv;
defm int_arm_mve_max: MVE_minmaxv;

defm int_arm_mve_addv: MVEPredicated<[llvm_i32_ty],
   [llvm_anyvector_ty, llvm_i32_ty /* unsigned */]>;
defm int_arm_mve_addlv: MVEPredicated<[llvm_i64_ty],
   [llvm_anyvector_ty, llvm_i32_ty /* unsigned */]>;

// Intrinsic with a predicated and a non-predicated case. The predicated case
// has two additional parameters: inactive (the value for inactive lanes, can
// be undef) and predicate.
multiclass MVEMXPredicated<list<LLVMType> rets, list<LLVMType> flags,
                           list<LLVMType> params, LLVMType inactive,
                           LLVMType predicate,
                           list<IntrinsicProperty> props = [IntrNoMem]> {
  def "":          Intrinsic<rets, flags # params, props>;
  def _predicated: Intrinsic<rets, flags # [inactive] # params # [predicate],
                             props>;
}

defm int_arm_mve_vcvt_narrow: MVEPredicated<[llvm_v8f16_ty],
   [llvm_v8f16_ty, llvm_v4f32_ty, llvm_i32_ty], llvm_v4i1_ty>;
defm int_arm_mve_vcvt_widen: MVEMXPredicated<[llvm_v4f32_ty], [],
   [llvm_v8f16_ty, llvm_i32_ty], llvm_v4f32_ty, llvm_v4i1_ty>;

defm int_arm_mve_vldr_gather_base: MVEPredicated<
   [llvm_anyvector_ty], [llvm_anyvector_ty, llvm_i32_ty],
   llvm_anyvector_ty, [IntrReadMem], [SDNPMemOperand]>;
defm int_arm_mve_vldr_gather_base_wb: MVEPredicated<
   [llvm_anyvector_ty, llvm_anyvector_ty],
   [LLVMMatchType<1>, llvm_i32_ty], llvm_anyvector_ty, [IntrReadMem],
   [SDNPMemOperand]>;
defm int_arm_mve_vstr_scatter_base: MVEPredicated<
   [], [llvm_anyvector_ty, llvm_i32_ty, llvm_anyvector_ty],
   llvm_anyvector_ty, [IntrWriteMem], [SDNPMemOperand]>;
defm int_arm_mve_vstr_scatter_base_wb: MVEPredicated<
   [llvm_anyvector_ty], [LLVMMatchType<0>, llvm_i32_ty, llvm_anyvector_ty],
   llvm_anyvector_ty, [IntrWriteMem], [SDNPMemOperand]>;

// gather_offset takes three i32 parameters. The first is the size of
// memory element loaded, in bits. The second is a left bit shift to
// apply to each offset in the vector parameter (must be either 0, or
// correspond to the element size of the destination vector type). The
// last is 1 to indicate zero extension (if the load is widening), or
// 0 for sign extension.
//
// scatter_offset has the first two of those parameters, but since it
// narrows rather than widening, it doesn't have the last one.
defm int_arm_mve_vldr_gather_offset: MVEPredicated<
   [llvm_anyvector_ty], [llvm_anyptr_ty, llvm_anyvector_ty,
   llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], llvm_anyvector_ty, [IntrReadMem],
   [SDNPMemOperand]>;
defm int_arm_mve_vstr_scatter_offset: MVEPredicated<
   [], [llvm_anyptr_ty, llvm_anyvector_ty, llvm_anyvector_ty,
   llvm_i32_ty, llvm_i32_ty], llvm_anyvector_ty, [IntrWriteMem],
   [SDNPMemOperand]>;

def int_arm_mve_shl_imm_predicated: Intrinsic<[llvm_anyvector_ty],
   [LLVMMatchType<0>, llvm_i32_ty, llvm_anyvector_ty, LLVMMatchType<0>],
   [IntrNoMem]>;
def int_arm_mve_shr_imm_predicated: Intrinsic<[llvm_anyvector_ty],
   [LLVMMatchType<0>, llvm_i32_ty, llvm_i32_ty, // extra i32 is unsigned flag
    llvm_anyvector_ty, LLVMMatchType<0>],
   [IntrNoMem]>;

defm int_arm_mve_vqshl_imm: MVEPredicatedM<[llvm_anyvector_ty],
   [LLVMMatchType<0>, llvm_i32_ty /*shiftcount*/, llvm_i32_ty /*unsigned*/]>;
defm int_arm_mve_vrshr_imm: MVEPredicatedM<[llvm_anyvector_ty],
   [LLVMMatchType<0>, llvm_i32_ty /*shiftcount*/, llvm_i32_ty /*unsigned*/]>;
defm int_arm_mve_vqshlu_imm: MVEPredicatedM<[llvm_anyvector_ty],
   [LLVMMatchType<0>, llvm_i32_ty /*shiftcount*/]>;
defm int_arm_mve_vshll_imm: MVEPredicatedM<[llvm_anyvector_ty],
   [llvm_anyvector_ty, llvm_i32_ty /*shiftcount*/, llvm_i32_ty /*unsigned*/,
                       llvm_i32_ty /*top-half*/]>;

defm int_arm_mve_vsli: MVEPredicated<
   [llvm_anyvector_ty], [LLVMMatchType<0>, LLVMMatchType<0>, llvm_i32_ty]>;
defm int_arm_mve_vsri: MVEPredicated<
   [llvm_anyvector_ty], [LLVMMatchType<0>, LLVMMatchType<0>, llvm_i32_ty]>;

defm int_arm_mve_vshrn: MVEPredicated<
   [llvm_anyvector_ty], [LLVMMatchType<0>, llvm_anyvector_ty,
    llvm_i32_ty /*shiftcount*/, llvm_i32_ty /*saturate*/, llvm_i32_ty /*round*/,
    llvm_i32_ty /*unsigned-out*/, llvm_i32_ty /*unsigned-in*/,
    llvm_i32_ty /*top-half*/]>;

defm int_arm_mve_vshl_scalar: MVEPredicated<
   [llvm_anyvector_ty], [LLVMMatchType<0>, llvm_i32_ty /*shiftcount*/,
    llvm_i32_ty /*saturate*/, llvm_i32_ty /*round*/, llvm_i32_ty /*unsigned*/]>;
defm int_arm_mve_vshl_vector: MVEPredicatedM<
   [llvm_anyvector_ty], [LLVMMatchType<0>, llvm_anyvector_ty /*shiftcounts*/,
    llvm_i32_ty /*saturate*/, llvm_i32_ty /*round*/, llvm_i32_ty /*unsigned*/]>;

// MVE scalar shifts.
class ARM_MVE_qrshift_single<list<LLVMType> value,
                             list<LLVMType> saturate = []> :
  Intrinsic<value, value # [llvm_i32_ty] # saturate, [IntrNoMem]>;
multiclass ARM_MVE_qrshift<list<LLVMType> saturate = []> {
  // Most of these shifts come in 32- and 64-bit versions. But only
  // the 64-bit ones have the extra saturation argument (if any).
  def "": ARM_MVE_qrshift_single<[llvm_i32_ty]>;
  def l:  ARM_MVE_qrshift_single<[llvm_i32_ty, llvm_i32_ty], saturate>;
}
defm int_arm_mve_urshr: ARM_MVE_qrshift;
defm int_arm_mve_uqshl: ARM_MVE_qrshift;
defm int_arm_mve_srshr: ARM_MVE_qrshift;
defm int_arm_mve_sqshl: ARM_MVE_qrshift;
defm int_arm_mve_uqrshl: ARM_MVE_qrshift<[llvm_i32_ty]>;
defm int_arm_mve_sqrshr: ARM_MVE_qrshift<[llvm_i32_ty]>;
// LSLL and ASRL only have 64-bit versions, not 32.
def int_arm_mve_lsll: ARM_MVE_qrshift_single<[llvm_i32_ty, llvm_i32_ty]>;
def int_arm_mve_asrl: ARM_MVE_qrshift_single<[llvm_i32_ty, llvm_i32_ty]>;

def int_arm_mve_vabd: Intrinsic<
   [llvm_anyvector_ty],
   [LLVMMatchType<0>, LLVMMatchType<0>, llvm_i32_ty /* unsigned */],
   [IntrNoMem]>;
def int_arm_mve_vadc: Intrinsic<
   [llvm_anyvector_ty, llvm_i32_ty],
   [LLVMMatchType<0>, LLVMMatchType<0>, llvm_i32_ty], [IntrNoMem]>;
def int_arm_mve_vsbc: Intrinsic<
   [llvm_anyvector_ty, llvm_i32_ty],
   [LLVMMatchType<0>, LLVMMatchType<0>, llvm_i32_ty], [IntrNoMem]>;
def int_arm_mve_vadc_predicated: Intrinsic<
   [llvm_anyvector_ty, llvm_i32_ty],
   [LLVMMatchType<0>, LLVMMatchType<0>, LLVMMatchType<0>,
    llvm_i32_ty, llvm_anyvector_ty], [IntrNoMem]>;
def int_arm_mve_vsbc_predicated: Intrinsic<
   [llvm_anyvector_ty, llvm_i32_ty],
   [LLVMMatchType<0>, LLVMMatchType<0>, LLVMMatchType<0>,
    llvm_i32_ty, llvm_anyvector_ty], [IntrNoMem]>;
def int_arm_mve_vshlc: Intrinsic<
   [llvm_i32_ty /* bits shifted out */, llvm_anyvector_ty],
   [LLVMMatchType<0>, llvm_i32_ty /* bits shifted in */,
    llvm_i32_ty /* shift count */], [IntrNoMem]>;
def int_arm_mve_vshlc_predicated: Intrinsic<
   [llvm_i32_ty /* bits shifted out */, llvm_anyvector_ty],
   [LLVMMatchType<0>, llvm_i32_ty /* bits shifted in */,
    llvm_i32_ty /* shift count */, llvm_anyvector_ty], [IntrNoMem]>;
def int_arm_mve_vmulh: Intrinsic<
   [llvm_anyvector_ty],
   [LLVMMatchType<0>, LLVMMatchType<0>, llvm_i32_ty /* unsigned */],
   [IntrNoMem]>;
def int_arm_mve_vqdmulh: Intrinsic<
   [llvm_anyvector_ty],
   [LLVMMatchType<0>, LLVMMatchType<0>], [IntrNoMem]>;
def int_arm_mve_vhadd: Intrinsic<
   [llvm_anyvector_ty],
   [LLVMMatchType<0>, LLVMMatchType<0>, llvm_i32_ty /* unsigned */],
   [IntrNoMem]>;
def int_arm_mve_vrhadd: Intrinsic<
   [llvm_anyvector_ty],
   [LLVMMatchType<0>, LLVMMatchType<0>, llvm_i32_ty /* unsigned */],
   [IntrNoMem]>;
def int_arm_mve_vhsub: Intrinsic<
   [llvm_anyvector_ty],
   [LLVMMatchType<0>, LLVMMatchType<0>, llvm_i32_ty /* unsigned */],
   [IntrNoMem]>;
def int_arm_mve_vrmulh: Intrinsic<
   [llvm_anyvector_ty],
   [LLVMMatchType<0>, LLVMMatchType<0>, llvm_i32_ty /* unsigned */],
   [IntrNoMem]>;
def int_arm_mve_vqrdmulh: Intrinsic<
   [llvm_anyvector_ty],
   [LLVMMatchType<0>, LLVMMatchType<0>], [IntrNoMem]>;
def int_arm_mve_vmull: Intrinsic<
   [llvm_anyvector_ty],
   [llvm_anyvector_ty, LLVMMatchType<1>, llvm_i32_ty /* unsigned */,
    llvm_i32_ty /* top */], [IntrNoMem]>;
def int_arm_mve_vmull_poly: Intrinsic<
   [llvm_anyvector_ty],
   [llvm_anyvector_ty, LLVMMatchType<1>, llvm_i32_ty], [IntrNoMem]>;

// The first two parameters are compile-time constants:
// * Halving: 0 means  halving (vhcaddq), 1 means non-halving (vcaddq) 
//            instruction. Note: the flag is inverted to match the corresponding
//            bit in the instruction encoding
// * Rotation angle: 0 mean 90 deg, 1 means 180 deg
defm int_arm_mve_vcaddq : MVEMXPredicated<
  [llvm_anyvector_ty],
  [llvm_i32_ty, llvm_i32_ty], [LLVMMatchType<0>, LLVMMatchType<0>],
   LLVMMatchType<0>, llvm_anyvector_ty>;

// The first operand of the following two intrinsics is the rotation angle
// (must be a compile-time constant):
// 0 - 0 deg
// 1 - 90 deg
// 2 - 180 deg
// 3 - 270 deg
defm int_arm_mve_vcmulq : MVEMXPredicated<
  [llvm_anyvector_ty],
  [llvm_i32_ty], [LLVMMatchType<0>, LLVMMatchType<0>],
   LLVMMatchType<0>, llvm_anyvector_ty>;

defm int_arm_mve_vcmlaq : MVEPredicated<
  [llvm_anyvector_ty],
  [llvm_i32_ty, LLVMMatchType<0>, LLVMMatchType<0>, LLVMMatchType<0>],
   llvm_anyvector_ty>;

def int_arm_mve_vld2q: Intrinsic<[llvm_anyvector_ty, LLVMMatchType<0>], [llvm_anyptr_ty],
                                 [IntrReadMem, IntrArgMemOnly]>;
def int_arm_mve_vld4q: Intrinsic<[llvm_anyvector_ty, LLVMMatchType<0>, LLVMMatchType<0>, LLVMMatchType<0>], [llvm_anyptr_ty],
                                 [IntrReadMem, IntrArgMemOnly]>;

def int_arm_mve_vst2q: Intrinsic<[], [llvm_anyptr_ty, llvm_anyvector_ty, LLVMMatchType<1>, llvm_i32_ty],
                                 [IntrWriteMem, IntrArgMemOnly], "", [SDNPMemOperand]>;
def int_arm_mve_vst4q: Intrinsic<[], [llvm_anyptr_ty, llvm_anyvector_ty, LLVMMatchType<1>, LLVMMatchType<1>, LLVMMatchType<1>, llvm_i32_ty],
                                 [IntrWriteMem, IntrArgMemOnly], "", [SDNPMemOperand]>;

// MVE vector absolute difference and accumulate across vector
// The first operand is an 'unsigned' flag. The remaining operands are:
// * accumulator
// * first vector operand
// * second vector operand
// * mask (only in predicated versions)
defm int_arm_mve_vabav: MVEPredicated<
  [llvm_i32_ty],
  [llvm_i32_ty, llvm_i32_ty, llvm_anyvector_ty, LLVMMatchType<0>], llvm_anyvector_ty>;

// The following 3 instrinsics are MVE vector reductions with two vector
// operands.
// The first 3 operands are boolean flags (must be compile-time constants):
// * unsigned - the instruction operates on vectors of unsigned values and
//              unsigned scalars
// * subtract - the instruction performs subtraction after multiplication of
//              lane pairs (e.g., vmlsdav vs vmladav)
// * exchange - the instruction exchanges successive even and odd lanes of
//              the first operands before multiplication of lane pairs
//              (e.g., vmladavx vs vmladav)
// The remaining operands are:
// * accumulator
// * first vector operand
// * second vector operand
// * mask (only in predicated versions)

// Version with 32-bit result, vml{a,s}dav[a][x]
defm int_arm_mve_vmldava: MVEPredicated<
  [llvm_i32_ty],
  [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty,
   llvm_i32_ty, llvm_anyvector_ty, LLVMMatchType<0>],
  llvm_anyvector_ty>;

// Version with 64-bit result, vml{a,s}ldav[a][x]
defm int_arm_mve_vmlldava: MVEPredicated<
  [llvm_i32_ty, llvm_i32_ty],
  [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty,
   llvm_i32_ty, llvm_i32_ty, llvm_anyvector_ty, LLVMMatchType<0>],
  llvm_anyvector_ty>;

// Version with 72-bit rounded result, vrml{a,s}ldavh[a][x]
defm int_arm_mve_vrmlldavha: MVEPredicated<
  [llvm_i32_ty, llvm_i32_ty],
  [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty,
   llvm_i32_ty, llvm_i32_ty, llvm_anyvector_ty, LLVMMatchType<0>],
  llvm_anyvector_ty>;

defm int_arm_mve_vidup: MVEMXPredicated<
   [llvm_anyvector_ty /* output */, llvm_i32_ty /* written-back base */], [],
   [llvm_i32_ty /* base */, llvm_i32_ty /* step */],
   LLVMMatchType<0>, llvm_anyvector_ty>;
defm int_arm_mve_vddup: MVEMXPredicated<
   [llvm_anyvector_ty /* output */, llvm_i32_ty /* written-back base */], [],
   [llvm_i32_ty /* base */, llvm_i32_ty /* step */],
   LLVMMatchType<0>, llvm_anyvector_ty>;
defm int_arm_mve_viwdup: MVEMXPredicated<
   [llvm_anyvector_ty /* output */, llvm_i32_ty /* written-back base */], [],
   [llvm_i32_ty /* base */, llvm_i32_ty /* limit */, llvm_i32_ty /* step */],
   LLVMMatchType<0>, llvm_anyvector_ty>;
defm int_arm_mve_vdwdup: MVEMXPredicated<
   [llvm_anyvector_ty /* output */, llvm_i32_ty /* written-back base */], [],
   [llvm_i32_ty /* base */, llvm_i32_ty /* limit */, llvm_i32_ty /* step */],
   LLVMMatchType<0>, llvm_anyvector_ty>;

// Flags:
// * unsigned
defm int_arm_mve_vcvt_fix: MVEMXPredicated<
  [llvm_anyvector_ty /* output */], [llvm_i32_ty],
  [llvm_anyvector_ty /* input vector */, llvm_i32_ty /* scale */],
  LLVMMatchType<0>, llvm_anyvector_ty>;

def int_arm_mve_vcvt_fp_int_predicated: Intrinsic<
  [llvm_anyvector_ty], [llvm_anyvector_ty, llvm_i32_ty /* unsigned */,
   llvm_anyvector_ty /* predicate */, LLVMMatchType<0> /* inactive */],
  [IntrNoMem]>;

foreach suffix = ["a","n","p","m"] in {
  defm "int_arm_mve_vcvt"#suffix: MVEMXPredicated<
    [llvm_anyvector_ty /* output */], [llvm_i32_ty /* unsigned */],
    [llvm_anyvector_ty /* input */], LLVMMatchType<0>, llvm_anyvector_ty>;
}

def int_arm_mve_vrintn: Intrinsic<
  [llvm_anyvector_ty], [LLVMMatchType<0>], [IntrNoMem]>;
def int_arm_mve_vcls: Intrinsic<
  [llvm_anyvector_ty], [LLVMMatchType<0>], [IntrNoMem]>;

defm int_arm_mve_vbrsr: MVEMXPredicated<
  [llvm_anyvector_ty], [],
  [LLVMMatchType<0>, llvm_i32_ty], LLVMMatchType<0>, llvm_anyvector_ty>;

def int_arm_mve_vqdmull: Intrinsic<
  [llvm_anyvector_ty],
  [llvm_anyvector_ty, LLVMMatchType<1>, llvm_i32_ty],
  [IntrNoMem]>;
def int_arm_mve_vqdmull_predicated: Intrinsic<
  [llvm_anyvector_ty],
  [llvm_anyvector_ty, LLVMMatchType<1>, llvm_i32_ty, llvm_anyvector_ty,
   LLVMMatchType<0>],
  [IntrNoMem]>;

class MVESimpleUnaryPredicated: Intrinsic<[llvm_anyvector_ty],
   [LLVMMatchType<0>, llvm_anyvector_ty, LLVMMatchType<0>], [IntrNoMem]>;

def int_arm_mve_mvn_predicated: MVESimpleUnaryPredicated;
def int_arm_mve_abs_predicated: MVESimpleUnaryPredicated;
def int_arm_mve_neg_predicated: MVESimpleUnaryPredicated;
def int_arm_mve_qabs_predicated: MVESimpleUnaryPredicated;
def int_arm_mve_qneg_predicated: MVESimpleUnaryPredicated;
def int_arm_mve_clz_predicated: MVESimpleUnaryPredicated;
def int_arm_mve_cls_predicated: MVESimpleUnaryPredicated;
def int_arm_mve_vrintz_predicated: MVESimpleUnaryPredicated;
def int_arm_mve_vrintm_predicated: MVESimpleUnaryPredicated;
def int_arm_mve_vrintp_predicated: MVESimpleUnaryPredicated;
def int_arm_mve_vrinta_predicated: MVESimpleUnaryPredicated;
def int_arm_mve_vrintx_predicated: MVESimpleUnaryPredicated;
def int_arm_mve_vrintn_predicated: MVESimpleUnaryPredicated;

def int_arm_mve_vrev_predicated: Intrinsic<[llvm_anyvector_ty],
   [LLVMMatchType<0>, llvm_i32_ty /* size to reverse */,
    llvm_anyvector_ty, LLVMMatchType<0>], [IntrNoMem]>;

def int_arm_mve_vmovl_predicated: Intrinsic<[llvm_anyvector_ty],
   [llvm_anyvector_ty, llvm_i32_ty /* unsigned */, llvm_i32_ty /* top half */,
    llvm_anyvector_ty /* predicate */, LLVMMatchType<0>], [IntrNoMem]>;
def int_arm_mve_vmovn_predicated: Intrinsic<[llvm_anyvector_ty],
   [LLVMMatchType<0>, llvm_anyvector_ty, llvm_i32_ty /* top half */,
    llvm_anyvector_ty /* predicate */], [IntrNoMem]>;

def int_arm_mve_vqmovn: Intrinsic<[llvm_anyvector_ty],
   [LLVMMatchType<0>, llvm_anyvector_ty,
    llvm_i32_ty /* unsigned output */, llvm_i32_ty /* unsigned input */,
    llvm_i32_ty /* top half */], [IntrNoMem]>;
def int_arm_mve_vqmovn_predicated: Intrinsic<[llvm_anyvector_ty],
   [LLVMMatchType<0>, llvm_anyvector_ty,
    llvm_i32_ty /* unsigned output */, llvm_i32_ty /* unsigned input */,
    llvm_i32_ty /* top half */, llvm_anyvector_ty /* pred */], [IntrNoMem]>;

def int_arm_mve_fma_predicated: Intrinsic<[llvm_anyvector_ty],
   [LLVMMatchType<0> /* mult op #1 */, LLVMMatchType<0> /* mult op #2 */,
    LLVMMatchType<0> /* addend */, llvm_anyvector_ty /* pred */], [IntrNoMem]>;
def int_arm_mve_vmla_n_predicated: Intrinsic<[llvm_anyvector_ty],
   [LLVMMatchType<0> /* mult op #1 */, LLVMMatchType<0> /* addend */,
    llvm_i32_ty /* mult op #2 (scalar) */, llvm_anyvector_ty /* pred */],
   [IntrNoMem]>;
def int_arm_mve_vmlas_n_predicated: Intrinsic<[llvm_anyvector_ty],
   [LLVMMatchType<0> /* mult op #1 */, LLVMMatchType<0> /* mult op #2 */,
    llvm_i32_ty /* addend (scalar) */, llvm_anyvector_ty /* pred */],
   [IntrNoMem]>;

defm int_arm_mve_vqdmlah: MVEPredicated<[llvm_anyvector_ty],
  [LLVMMatchType<0> /* mult op #1 */, LLVMMatchType<0> /* addend */,
   llvm_i32_ty /* mult op #2 (scalar) */]>;
defm int_arm_mve_vqrdmlah: MVEPredicated<[llvm_anyvector_ty],
  [LLVMMatchType<0> /* mult op #1 */, LLVMMatchType<0> /* addend */,
   llvm_i32_ty /* mult op #2 (scalar) */]>;
defm int_arm_mve_vqdmlash: MVEPredicated<[llvm_anyvector_ty],
  [LLVMMatchType<0> /* mult op #1 */, LLVMMatchType<0> /* mult op #2 */,
   llvm_i32_ty /* addend (scalar) */]>;
defm int_arm_mve_vqrdmlash: MVEPredicated<[llvm_anyvector_ty],
  [LLVMMatchType<0> /* mult op #1 */, LLVMMatchType<0> /* mult op #2 */,
   llvm_i32_ty /* addend (scalar) */]>;

defm int_arm_mve_vqdmlad: MVEPredicated<[llvm_anyvector_ty],
  [LLVMMatchType<0>, LLVMMatchType<0>, LLVMMatchType<0>,
   llvm_i32_ty /* exchange */, llvm_i32_ty /* round */,
   llvm_i32_ty /* subtract */]>;

// CDE (Custom Datapath Extension)

multiclass CDEGPRIntrinsics<list<LLVMType> args> {
  def "" : Intrinsic<
    [llvm_i32_ty],
    !listconcat([llvm_i32_ty /* coproc */], args, [llvm_i32_ty /* imm */]),
    [IntrNoMem, ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<!add(!size(args), 1)>>]>;
  def a : Intrinsic<
    [llvm_i32_ty],
    !listconcat([llvm_i32_ty /* coproc */, llvm_i32_ty /* acc */], args,
                [llvm_i32_ty /* imm */]),
    [IntrNoMem, ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<!add(!size(args), 2)>>]>;

  def d: Intrinsic<
    [llvm_i32_ty /* lo */, llvm_i32_ty /* hi */],
    !listconcat([llvm_i32_ty /* coproc */], args, [llvm_i32_ty /* imm */]),
    [IntrNoMem, ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<!add(!size(args), 1)>>]>;
  def da: Intrinsic<
    [llvm_i32_ty /* lo */, llvm_i32_ty /* hi */],
    !listconcat([llvm_i32_ty /* coproc */, llvm_i32_ty /* acc_lo */,
                 llvm_i32_ty /* acc_hi */], args, [llvm_i32_ty /* imm */]),
    [IntrNoMem, ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<!add(!size(args), 3)>>]>;
}

defm int_arm_cde_cx1: CDEGPRIntrinsics<[]>;
defm int_arm_cde_cx2: CDEGPRIntrinsics<[llvm_i32_ty]>;
defm int_arm_cde_cx3: CDEGPRIntrinsics<[llvm_i32_ty, llvm_i32_ty]>;

multiclass CDEVCXIntrinsics<list<LLVMType> args> {
  def "" : Intrinsic<
    [llvm_anyfloat_ty],
    !listconcat([llvm_i32_ty /* coproc */], args, [llvm_i32_ty /* imm */]),
    [IntrNoMem, ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<!add(!size(args), 1)>>]>;
  def a : Intrinsic<
    [llvm_anyfloat_ty],
    !listconcat([llvm_i32_ty /* coproc */,  LLVMMatchType<0> /* acc */],
                args, [llvm_i32_ty /* imm */]),
    [IntrNoMem, ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<!add(!size(args), 2)>>]>;
}

defm int_arm_cde_vcx1 : CDEVCXIntrinsics<[]>;
defm int_arm_cde_vcx2 : CDEVCXIntrinsics<[LLVMMatchType<0>]>;
defm int_arm_cde_vcx3 : CDEVCXIntrinsics<[LLVMMatchType<0>, LLVMMatchType<0>]>;

multiclass CDEVCXVecIntrinsics<list<LLVMType> args> {
  def "" : Intrinsic<
    [llvm_v16i8_ty],
    !listconcat([llvm_i32_ty /* coproc */], args, [llvm_i32_ty /* imm */]),
    [IntrNoMem, ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<!add(!size(args), 1)>>]>;
  def a : Intrinsic<
    [llvm_v16i8_ty],
    !listconcat([llvm_i32_ty /* coproc */, llvm_v16i8_ty /* acc */],
                args, [llvm_i32_ty /* imm */]),
    [IntrNoMem, ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<!add(!size(args), 2)>>]>;

  def _predicated : Intrinsic<
    [llvm_anyvector_ty],
    !listconcat([llvm_i32_ty /* coproc */, LLVMMatchType<0> /* inactive */],
                args, [llvm_i32_ty /* imm */, llvm_anyvector_ty /* mask */]),
    [IntrNoMem, ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<!add(!size(args), 2)>>]>;
  def a_predicated : Intrinsic<
    [llvm_anyvector_ty],
    !listconcat([llvm_i32_ty /* coproc */, LLVMMatchType<0> /* acc */],
                args, [llvm_i32_ty /* imm */, llvm_anyvector_ty /* mask */]),
    [IntrNoMem, ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<!add(!size(args), 2)>>]>;
}

defm int_arm_cde_vcx1q : CDEVCXVecIntrinsics<[]>;
defm int_arm_cde_vcx2q : CDEVCXVecIntrinsics<[llvm_v16i8_ty]>;
defm int_arm_cde_vcx3q : CDEVCXVecIntrinsics<[llvm_v16i8_ty, llvm_v16i8_ty]>;

} // end TargetPrefix
