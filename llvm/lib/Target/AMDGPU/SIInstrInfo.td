//===-- SIInstrInfo.td -----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

def isWave32 : Predicate<"Subtarget->getWavefrontSize() == 32">,
  AssemblerPredicate <(all_of FeatureWavefrontSize32)>;
def isWave64 : Predicate<"Subtarget->getWavefrontSize() == 64">,
  AssemblerPredicate <(all_of FeatureWavefrontSize64)>;

class GCNPredicateControl : PredicateControl {
  Predicate SIAssemblerPredicate = isGFX6GFX7;
  Predicate VIAssemblerPredicate = isGFX8GFX9;
}

// Except for the NONE field, this must be kept in sync with the
// SIEncodingFamily enum in SIInstrInfo.cpp and the columns of the
// getMCOpcodeGen table.
def SIEncodingFamily {
  int NONE = -1;
  int SI = 0;
  int VI = 1;
  int SDWA = 2;
  int SDWA9 = 3;
  int GFX80 = 4;
  int GFX9 = 5;
  int GFX10 = 6;
  int SDWA10 = 7;
  int GFX90A = 8;
  int GFX940 = 9;
  int GFX11 = 10;
}

//===----------------------------------------------------------------------===//
// SI DAG Nodes
//===----------------------------------------------------------------------===//

def AMDGPUclamp : SDNode<"AMDGPUISD::CLAMP", SDTFPUnaryOp>;

def SIsbuffer_load : SDNode<"AMDGPUISD::SBUFFER_LOAD",
  SDTypeProfile<1, 3, [SDTCisVT<1, v4i32>, SDTCisVT<2, i32>, SDTCisVT<3, i32>]>,
  [SDNPMayLoad, SDNPMemOperand]
>;

def SIds_ordered_count : SDNode<"AMDGPUISD::DS_ORDERED_COUNT",
  SDTypeProfile<1, 2, [SDTCisVT<0, i32>, SDTCisVT<1, i32>, SDTCisVT<2, i16>]>,
  [SDNPMayLoad, SDNPMayStore, SDNPMemOperand, SDNPHasChain, SDNPInGlue]
>;

def SIatomic_inc : SDNode<"AMDGPUISD::ATOMIC_INC", SDTAtomic2,
  [SDNPMayLoad, SDNPMayStore, SDNPMemOperand, SDNPHasChain]
>;

def SIatomic_dec : SDNode<"AMDGPUISD::ATOMIC_DEC", SDTAtomic2,
  [SDNPMayLoad, SDNPMayStore, SDNPMemOperand, SDNPHasChain]
>;

def SDTAtomic2_f32 : SDTypeProfile<1, 2, [
  SDTCisSameAs<0,2>, SDTCisFP<0>, SDTCisPtrTy<1>
]>;

def SIatomic_fmin : SDNode<"AMDGPUISD::ATOMIC_LOAD_FMIN", SDTAtomic2_f32,
  [SDNPMayLoad, SDNPMayStore, SDNPMemOperand, SDNPHasChain]
>;

def SIatomic_fmax : SDNode<"AMDGPUISD::ATOMIC_LOAD_FMAX", SDTAtomic2_f32,
  [SDNPMayLoad, SDNPMayStore, SDNPMemOperand, SDNPHasChain]
>;

// load_d16_{lo|hi} ptr, tied_input
def SIload_d16 : SDTypeProfile<1, 2, [
  SDTCisPtrTy<1>,
  SDTCisSameAs<0, 2>
]>;


def SDTtbuffer_load : SDTypeProfile<1, 8,
  [                     // vdata
   SDTCisVT<1, v4i32>,  // rsrc
   SDTCisVT<2, i32>,    // vindex(VGPR)
   SDTCisVT<3, i32>,    // voffset(VGPR)
   SDTCisVT<4, i32>,    // soffset(SGPR)
   SDTCisVT<5, i32>,    // offset(imm)
   SDTCisVT<6, i32>,    // format(imm)
   SDTCisVT<7, i32>,    // cachepolicy, swizzled buffer(imm)
   SDTCisVT<8, i1>      // idxen(imm)
  ]>;

def SItbuffer_load :   SDNode<"AMDGPUISD::TBUFFER_LOAD_FORMAT", SDTtbuffer_load,
                              [SDNPMayLoad, SDNPMemOperand, SDNPHasChain]>;
def SItbuffer_load_d16 : SDNode<"AMDGPUISD::TBUFFER_LOAD_FORMAT_D16",
                                SDTtbuffer_load,
                                [SDNPMayLoad, SDNPMemOperand, SDNPHasChain]>;

def SDTtbuffer_store : SDTypeProfile<0, 9,
    [                     // vdata
     SDTCisVT<1, v4i32>,  // rsrc
     SDTCisVT<2, i32>,    // vindex(VGPR)
     SDTCisVT<3, i32>,    // voffset(VGPR)
     SDTCisVT<4, i32>,    // soffset(SGPR)
     SDTCisVT<5, i32>,    // offset(imm)
     SDTCisVT<6, i32>,    // format(imm)
     SDTCisVT<7, i32>,    // cachepolicy, swizzled buffer(imm)
     SDTCisVT<8, i1>      // idxen(imm)
    ]>;

def SItbuffer_store : SDNode<"AMDGPUISD::TBUFFER_STORE_FORMAT", SDTtbuffer_store,
                             [SDNPMayStore, SDNPMemOperand, SDNPHasChain]>;
def SItbuffer_store_d16 : SDNode<"AMDGPUISD::TBUFFER_STORE_FORMAT_D16",
                                SDTtbuffer_store,
                                [SDNPMayStore, SDNPMemOperand, SDNPHasChain]>;

def SDTBufferLoad : SDTypeProfile<1, 7,
    [                    // vdata
     SDTCisVT<1, v4i32>, // rsrc
     SDTCisVT<2, i32>,   // vindex(VGPR)
     SDTCisVT<3, i32>,   // voffset(VGPR)
     SDTCisVT<4, i32>,   // soffset(SGPR)
     SDTCisVT<5, i32>,   // offset(imm)
     SDTCisVT<6, i32>,   // cachepolicy, swizzled buffer(imm)
     SDTCisVT<7, i1>]>;  // idxen(imm)

def SIbuffer_load : SDNode <"AMDGPUISD::BUFFER_LOAD", SDTBufferLoad,
                            [SDNPMemOperand, SDNPHasChain, SDNPMayLoad]>;
def SIbuffer_load_ubyte : SDNode <"AMDGPUISD::BUFFER_LOAD_UBYTE", SDTBufferLoad,
                            [SDNPMemOperand, SDNPHasChain, SDNPMayLoad]>;
def SIbuffer_load_ushort : SDNode <"AMDGPUISD::BUFFER_LOAD_USHORT", SDTBufferLoad,
                            [SDNPMemOperand, SDNPHasChain, SDNPMayLoad]>;
def SIbuffer_load_byte : SDNode <"AMDGPUISD::BUFFER_LOAD_BYTE", SDTBufferLoad,
                            [SDNPMemOperand, SDNPHasChain, SDNPMayLoad]>;
def SIbuffer_load_short: SDNode <"AMDGPUISD::BUFFER_LOAD_SHORT", SDTBufferLoad,
                            [SDNPMemOperand, SDNPHasChain, SDNPMayLoad]>;
def SIbuffer_load_format : SDNode <"AMDGPUISD::BUFFER_LOAD_FORMAT", SDTBufferLoad,
                            [SDNPMemOperand, SDNPHasChain, SDNPMayLoad]>;
def SIbuffer_load_format_d16 : SDNode <"AMDGPUISD::BUFFER_LOAD_FORMAT_D16",
                                SDTBufferLoad,
                                [SDNPMemOperand, SDNPHasChain, SDNPMayLoad]>;

def SDTBufferStore : SDTypeProfile<0, 8,
    [                    // vdata
     SDTCisVT<1, v4i32>, // rsrc
     SDTCisVT<2, i32>,   // vindex(VGPR)
     SDTCisVT<3, i32>,   // voffset(VGPR)
     SDTCisVT<4, i32>,   // soffset(SGPR)
     SDTCisVT<5, i32>,   // offset(imm)
     SDTCisVT<6, i32>,   // cachepolicy, swizzled buffer(imm)
     SDTCisVT<7, i1>]>;  // idxen(imm)

def SIbuffer_store : SDNode <"AMDGPUISD::BUFFER_STORE", SDTBufferStore,
                             [SDNPMayStore, SDNPMemOperand, SDNPHasChain]>;
def SIbuffer_store_byte: SDNode <"AMDGPUISD::BUFFER_STORE_BYTE",
                         SDTBufferStore,
                         [SDNPMayStore, SDNPMemOperand, SDNPHasChain]>;
def SIbuffer_store_short : SDNode <"AMDGPUISD::BUFFER_STORE_SHORT",
                           SDTBufferStore,
                           [SDNPMayStore, SDNPMemOperand, SDNPHasChain]>;
def SIbuffer_store_format : SDNode <"AMDGPUISD::BUFFER_STORE_FORMAT",
                            SDTBufferStore,
                            [SDNPMayStore, SDNPMemOperand, SDNPHasChain]>;
def SIbuffer_store_format_d16 : SDNode <"AMDGPUISD::BUFFER_STORE_FORMAT_D16",
                            SDTBufferStore,
                            [SDNPMayStore, SDNPMemOperand, SDNPHasChain]>;

class SDBufferAtomic<string opcode> : SDNode <opcode,
  SDTypeProfile<1, 8,
       [SDTCisVT<2, v4i32>, // rsrc
       SDTCisVT<3, i32>,   // vindex(VGPR)
       SDTCisVT<4, i32>,   // voffset(VGPR)
       SDTCisVT<5, i32>,   // soffset(SGPR)
       SDTCisVT<6, i32>,   // offset(imm)
       SDTCisVT<7, i32>,   // cachepolicy(imm)
       SDTCisVT<8, i1>]>,  // idxen(imm)
  [SDNPMemOperand, SDNPHasChain, SDNPMayLoad, SDNPMayStore]
>;

def SIbuffer_atomic_swap : SDBufferAtomic <"AMDGPUISD::BUFFER_ATOMIC_SWAP">;
def SIbuffer_atomic_add : SDBufferAtomic <"AMDGPUISD::BUFFER_ATOMIC_ADD">;
def SIbuffer_atomic_sub : SDBufferAtomic <"AMDGPUISD::BUFFER_ATOMIC_SUB">;
def SIbuffer_atomic_smin : SDBufferAtomic <"AMDGPUISD::BUFFER_ATOMIC_SMIN">;
def SIbuffer_atomic_umin : SDBufferAtomic <"AMDGPUISD::BUFFER_ATOMIC_UMIN">;
def SIbuffer_atomic_smax : SDBufferAtomic <"AMDGPUISD::BUFFER_ATOMIC_SMAX">;
def SIbuffer_atomic_umax : SDBufferAtomic <"AMDGPUISD::BUFFER_ATOMIC_UMAX">;
def SIbuffer_atomic_and : SDBufferAtomic <"AMDGPUISD::BUFFER_ATOMIC_AND">;
def SIbuffer_atomic_or : SDBufferAtomic <"AMDGPUISD::BUFFER_ATOMIC_OR">;
def SIbuffer_atomic_xor : SDBufferAtomic <"AMDGPUISD::BUFFER_ATOMIC_XOR">;
def SIbuffer_atomic_inc : SDBufferAtomic <"AMDGPUISD::BUFFER_ATOMIC_INC">;
def SIbuffer_atomic_dec : SDBufferAtomic <"AMDGPUISD::BUFFER_ATOMIC_DEC">;
def SIbuffer_atomic_csub : SDBufferAtomic <"AMDGPUISD::BUFFER_ATOMIC_CSUB">;
def SIbuffer_atomic_fadd : SDBufferAtomic <"AMDGPUISD::BUFFER_ATOMIC_FADD">;
def SIbuffer_atomic_fmin : SDBufferAtomic <"AMDGPUISD::BUFFER_ATOMIC_FMIN">;
def SIbuffer_atomic_fmax : SDBufferAtomic <"AMDGPUISD::BUFFER_ATOMIC_FMAX">;

multiclass SDBufferAtomicNoRet {
  def "_noret" : PatFrag<
    (ops node:$vdata_in, node:$rsrc, node:$vindex, node:$voffset, node:$soffset,
      node:$offset, node:$cachepolicy, node:$idxen),
    (!cast<SDNode>(NAME) node:$vdata_in, node:$rsrc, node:$vindex,
      node:$voffset, node:$soffset, node:$offset, node:$cachepolicy,
      node:$idxen)> {
    let HasNoUse = true;
  }
}

defm SIbuffer_atomic_swap : SDBufferAtomicNoRet;
defm SIbuffer_atomic_add : SDBufferAtomicNoRet;
defm SIbuffer_atomic_sub : SDBufferAtomicNoRet;
defm SIbuffer_atomic_smin : SDBufferAtomicNoRet;
defm SIbuffer_atomic_umin : SDBufferAtomicNoRet;
defm SIbuffer_atomic_smax : SDBufferAtomicNoRet;
defm SIbuffer_atomic_umax : SDBufferAtomicNoRet;
defm SIbuffer_atomic_and : SDBufferAtomicNoRet;
defm SIbuffer_atomic_or : SDBufferAtomicNoRet;
defm SIbuffer_atomic_xor : SDBufferAtomicNoRet;
defm SIbuffer_atomic_inc : SDBufferAtomicNoRet;
defm SIbuffer_atomic_dec : SDBufferAtomicNoRet;
defm SIbuffer_atomic_fadd : SDBufferAtomicNoRet;
defm SIbuffer_atomic_fmin : SDBufferAtomicNoRet;
defm SIbuffer_atomic_fmax : SDBufferAtomicNoRet;

def SIbuffer_atomic_cmpswap : SDNode <"AMDGPUISD::BUFFER_ATOMIC_CMPSWAP",
  SDTypeProfile<1, 9,
    [SDTCisVT<0, i32>,   // dst
     SDTCisVT<1, i32>,   // src
     SDTCisVT<2, i32>,   // cmp
     SDTCisVT<3, v4i32>, // rsrc
     SDTCisVT<4, i32>,   // vindex(VGPR)
     SDTCisVT<5, i32>,   // voffset(VGPR)
     SDTCisVT<6, i32>,   // soffset(SGPR)
     SDTCisVT<7, i32>,   // offset(imm)
     SDTCisVT<8, i32>,   // cachepolicy(imm)
     SDTCisVT<9, i1>]>,  // idxen(imm)
  [SDNPMemOperand, SDNPHasChain, SDNPMayLoad, SDNPMayStore]
>;

def SIbuffer_atomic_cmpswap_noret : PatFrag<
  (ops node:$src, node:$cmp, node:$rsrc, node:$vindex, node:$voffset,
    node:$soffset, node:$offset, node:$cachepolicy, node:$idxen),
  (SIbuffer_atomic_cmpswap node:$src, node:$cmp, node:$rsrc, node:$vindex,
    node:$voffset, node:$soffset, node:$offset, node:$cachepolicy,
    node:$idxen)> {
  let HasNoUse = true;
}

class SDGlobalAtomicNoRtn<string opcode, ValueType ty> : SDNode <opcode,
  SDTypeProfile<0, 2,
      [SDTCisPtrTy<0>,     // vaddr
       SDTCisVT<1, ty>]>,  // vdata
  [SDNPMemOperand, SDNPHasChain, SDNPMayLoad, SDNPMayStore]
>;

def SIpc_add_rel_offset : SDNode<"AMDGPUISD::PC_ADD_REL_OFFSET",
  SDTypeProfile<1, 2, [SDTCisVT<0, iPTR>, SDTCisSameAs<0,1>, SDTCisSameAs<0,2>]>
>;

def SIlds : SDNode<"AMDGPUISD::LDS",
  SDTypeProfile<1, 1, [SDTCisVT<0, iPTR>, SDTCisSameAs<0,1>]>
>;

def SIload_d16_lo : SDNode<"AMDGPUISD::LOAD_D16_LO",
  SIload_d16,
  [SDNPMayLoad, SDNPMemOperand, SDNPHasChain]
>;

def SIload_d16_lo_u8 : SDNode<"AMDGPUISD::LOAD_D16_LO_U8",
  SIload_d16,
  [SDNPMayLoad, SDNPMemOperand, SDNPHasChain]
>;

def SIload_d16_lo_i8 : SDNode<"AMDGPUISD::LOAD_D16_LO_I8",
  SIload_d16,
  [SDNPMayLoad, SDNPMemOperand, SDNPHasChain]
>;

def SIload_d16_hi : SDNode<"AMDGPUISD::LOAD_D16_HI",
  SIload_d16,
  [SDNPMayLoad, SDNPMemOperand, SDNPHasChain]
>;

def SIload_d16_hi_u8 : SDNode<"AMDGPUISD::LOAD_D16_HI_U8",
  SIload_d16,
  [SDNPMayLoad, SDNPMemOperand, SDNPHasChain]
>;

def SIload_d16_hi_i8 : SDNode<"AMDGPUISD::LOAD_D16_HI_I8",
  SIload_d16,
  [SDNPMayLoad, SDNPMemOperand, SDNPHasChain]
>;

def SIdenorm_mode : SDNode<"AMDGPUISD::DENORM_MODE",
  SDTypeProfile<0 ,1, [SDTCisInt<0>]>,
  [SDNPHasChain, SDNPOptInGlue, SDNPOutGlue]
>;

def SIfptrunc_round_upward : SDNode<"AMDGPUISD::FPTRUNC_ROUND_UPWARD",
  SDTFPRoundOp
>;

def SIfptrunc_round_downward : SDNode<"AMDGPUISD::FPTRUNC_ROUND_DOWNWARD",
  SDTFPRoundOp
>;

//===----------------------------------------------------------------------===//
// ValueType helpers
//===----------------------------------------------------------------------===//

// Returns 1 if the source arguments have modifiers, 0 if they do not.
class isFloatType<ValueType SrcVT> {
  bit ret = !or(!eq(SrcVT.Value, f16.Value),
                !eq(SrcVT.Value, f32.Value),
                !eq(SrcVT.Value, f64.Value),
                !eq(SrcVT.Value, v2f16.Value),
                !eq(SrcVT.Value, v4f16.Value),
                !eq(SrcVT.Value, v8f16.Value),
                !eq(SrcVT.Value, v16f16.Value),
                !eq(SrcVT.Value, v2f32.Value),
                !eq(SrcVT.Value, v4f32.Value),
                !eq(SrcVT.Value, v8f32.Value),
                !eq(SrcVT.Value, v2f64.Value),
                !eq(SrcVT.Value, v4f64.Value));
}

// XXX - do v2i16 instructions?
class isIntType<ValueType SrcVT> {
  bit ret = !or(!eq(SrcVT.Value, i8.Value),
                !eq(SrcVT.Value, i16.Value),
                !eq(SrcVT.Value, i32.Value),
                !eq(SrcVT.Value, i64.Value),
                !eq(SrcVT.Value, v4i16.Value),
                !eq(SrcVT.Value, v8i16.Value),
                !eq(SrcVT.Value, v16i16.Value),
                !eq(SrcVT.Value, v2i32.Value),
                !eq(SrcVT.Value, v4i32.Value),
                !eq(SrcVT.Value, v8i32.Value));
}

class isPackedType<ValueType SrcVT> {
  bit ret = !or(!eq(SrcVT.Value, v2i16.Value),
                !eq(SrcVT.Value, v2f16.Value),
                !eq(SrcVT.Value, v4f16.Value),
                !eq(SrcVT.Value, v2i32.Value),
                !eq(SrcVT.Value, v2f32.Value),
                !eq(SrcVT.Value, v4i32.Value),
                !eq(SrcVT.Value, v4f32.Value),
                !eq(SrcVT.Value, v8i32.Value),
                !eq(SrcVT.Value, v8f32.Value));
}

class is16BitType<ValueType SrcVT> {
  bit ret = !or(!eq(SrcVT.Value, i16.Value),
                !eq(SrcVT.Value, f16.Value));
}

//===----------------------------------------------------------------------===//
// PatFrags for global memory operations
//===----------------------------------------------------------------------===//

defm atomic_inc : binary_atomic_op_all_as<SIatomic_inc>;
defm atomic_dec : binary_atomic_op_all_as<SIatomic_dec>;
defm atomic_load_fmin : binary_atomic_op_all_as<SIatomic_fmin, 0>;
defm atomic_load_fmax : binary_atomic_op_all_as<SIatomic_fmax, 0>;

//===----------------------------------------------------------------------===//
// SDNodes PatFrags for loads/stores with a glue input.
// This is for SDNodes and PatFrag for local loads and stores to
// enable s_mov_b32 m0, -1 to be glued to the memory instructions.
//
// These mirror the regular load/store PatFrags and rely on special
// processing during Select() to add the glued copy.
//
//===----------------------------------------------------------------------===//

def AMDGPUld_glue : SDNode <"ISD::LOAD", SDTLoad,
  [SDNPHasChain, SDNPMayLoad, SDNPMemOperand, SDNPInGlue]
>;

def AMDGPUatomic_ld_glue : SDNode <"ISD::ATOMIC_LOAD", SDTAtomicLoad,
  [SDNPHasChain, SDNPMayLoad, SDNPMemOperand, SDNPInGlue]
>;

def unindexedload_glue : PatFrag <(ops node:$ptr), (AMDGPUld_glue node:$ptr)> {
  let IsLoad = 1;
  let IsUnindexed = 1;
}

def load_glue : PatFrag <(ops node:$ptr), (unindexedload_glue node:$ptr)> {
  let IsLoad = 1;
  let IsNonExtLoad = 1;
}

def atomic_load_8_glue : PatFrag<(ops node:$ptr),
  (AMDGPUatomic_ld_glue node:$ptr)> {
  let IsAtomic = 1;
  let MemoryVT = i8;
}

def atomic_load_16_glue : PatFrag<(ops node:$ptr),
  (AMDGPUatomic_ld_glue node:$ptr)> {
  let IsAtomic = 1;
  let MemoryVT = i16;
}

def atomic_load_32_glue : PatFrag<(ops node:$ptr),
  (AMDGPUatomic_ld_glue node:$ptr)> {
  let IsAtomic = 1;
  let MemoryVT = i32;
}

def atomic_load_64_glue : PatFrag<(ops node:$ptr),
  (AMDGPUatomic_ld_glue node:$ptr)> {
  let IsAtomic = 1;
  let MemoryVT = i64;
}

def extload_glue : PatFrag<(ops node:$ptr), (unindexedload_glue node:$ptr)> {
  let IsLoad = 1;
  let IsAnyExtLoad = 1;
}

def sextload_glue : PatFrag<(ops node:$ptr), (unindexedload_glue node:$ptr)> {
  let IsLoad = 1;
  let IsSignExtLoad = 1;
}

def zextload_glue : PatFrag<(ops node:$ptr), (unindexedload_glue node:$ptr)> {
  let IsLoad = 1;
  let IsZeroExtLoad = 1;
}

def extloadi8_glue : PatFrag<(ops node:$ptr), (extload_glue node:$ptr)> {
  let IsLoad = 1;
  let MemoryVT = i8;
}

def zextloadi8_glue : PatFrag<(ops node:$ptr), (zextload_glue node:$ptr)> {
  let IsLoad = 1;
  let MemoryVT = i8;
}

def extloadi16_glue : PatFrag<(ops node:$ptr), (extload_glue node:$ptr)> {
  let IsLoad = 1;
  let MemoryVT = i16;
}

def zextloadi16_glue : PatFrag<(ops node:$ptr), (zextload_glue node:$ptr)> {
  let IsLoad = 1;
  let MemoryVT = i16;
}

def sextloadi8_glue : PatFrag<(ops node:$ptr), (sextload_glue node:$ptr)> {
  let IsLoad = 1;
  let MemoryVT = i8;
}

def sextloadi16_glue : PatFrag<(ops node:$ptr), (sextload_glue node:$ptr)> {
  let IsLoad = 1;
  let MemoryVT = i16;
}


let IsLoad = 1, AddressSpaces = LoadAddress_local.AddrSpaces in {
def load_local_m0 : PatFrag<(ops node:$ptr), (load_glue node:$ptr)> {
  let IsNonExtLoad = 1;
}

def extloadi8_local_m0 : PatFrag<(ops node:$ptr), (extloadi8_glue node:$ptr)>;
def sextloadi8_local_m0 : PatFrag<(ops node:$ptr), (sextloadi8_glue node:$ptr)>;
def zextloadi8_local_m0 : PatFrag<(ops node:$ptr), (zextloadi8_glue node:$ptr)>;

def extloadi16_local_m0 : PatFrag<(ops node:$ptr), (extloadi16_glue node:$ptr)>;
def sextloadi16_local_m0 : PatFrag<(ops node:$ptr), (sextloadi16_glue node:$ptr)>;
def zextloadi16_local_m0 : PatFrag<(ops node:$ptr), (zextloadi16_glue node:$ptr)>;
} // End IsLoad = 1, , AddressSpaces = LoadAddress_local.AddrSpaces

def load_align8_local_m0 : PatFrag<(ops node:$ptr),
                                   (load_local_m0 node:$ptr)> {
  let IsLoad = 1;
  int MinAlignment = 8;
}

def load_align16_local_m0 : PatFrag<(ops node:$ptr),
                                   (load_local_m0 node:$ptr)> {
  let IsLoad = 1;
  int MinAlignment = 16;
}

let IsAtomic = 1, AddressSpaces = LoadAddress_local.AddrSpaces in {
def atomic_load_8_local_m0 : PatFrag<(ops node:$ptr),
                                      (atomic_load_8_glue node:$ptr)>;
def atomic_load_16_local_m0 : PatFrag<(ops node:$ptr),
                                      (atomic_load_16_glue node:$ptr)>;
def atomic_load_32_local_m0 : PatFrag<(ops node:$ptr),
                                      (atomic_load_32_glue node:$ptr)>;
def atomic_load_64_local_m0 : PatFrag<(ops node:$ptr),
                                       (atomic_load_64_glue node:$ptr)>;
} // End let AddressSpaces = LoadAddress_local.AddrSpaces


def AMDGPUst_glue : SDNode <"ISD::STORE", SDTStore,
  [SDNPHasChain, SDNPMayStore, SDNPMemOperand, SDNPInGlue]
>;

def AMDGPUatomic_st_glue : SDNode <"ISD::ATOMIC_STORE", SDTAtomicStore,
  [SDNPHasChain, SDNPMayStore, SDNPMemOperand, SDNPInGlue]
>;

def unindexedstore_glue : PatFrag<(ops node:$val, node:$ptr),
                                   (AMDGPUst_glue node:$val, node:$ptr)> {
  let IsStore = 1;
  let IsUnindexed = 1;
}

def store_glue : PatFrag<(ops node:$val, node:$ptr),
                         (unindexedstore_glue node:$val, node:$ptr)> {
  let IsStore = 1;
  let IsTruncStore = 0;
}

def truncstore_glue : PatFrag<(ops node:$val, node:$ptr),
  (unindexedstore_glue node:$val, node:$ptr)> {
  let IsStore = 1;
  let IsTruncStore = 1;
}

def truncstorei8_glue : PatFrag<(ops node:$val, node:$ptr),
                           (truncstore_glue node:$val, node:$ptr)> {
  let IsStore = 1;
  let MemoryVT = i8;
  let IsTruncStore = 1;
}

def truncstorei16_glue : PatFrag<(ops node:$val, node:$ptr),
                           (truncstore_glue node:$val, node:$ptr)> {
  let IsStore = 1;
  let MemoryVT = i16;
  let IsTruncStore = 1;
}

let IsStore = 1, AddressSpaces = StoreAddress_local.AddrSpaces in {
def store_local_m0 : PatFrag<(ops node:$val, node:$ptr),
                             (store_glue node:$val, node:$ptr)>;
def truncstorei8_local_m0 : PatFrag<(ops node:$val, node:$ptr),
                                    (truncstorei8_glue node:$val, node:$ptr)>;
def truncstorei16_local_m0 : PatFrag<(ops node:$val, node:$ptr),
                                    (truncstorei16_glue node:$val, node:$ptr)>;
}

def store_align8_local_m0 : PatFrag <(ops node:$value, node:$ptr),
                                     (store_local_m0 node:$value, node:$ptr)>,
                            Aligned<8> {
  let IsStore = 1;
}

def store_align16_local_m0 : PatFrag <(ops node:$value, node:$ptr),
                                     (store_local_m0 node:$value, node:$ptr)>,
                            Aligned<16> {
  let IsStore = 1;
}

let PredicateCode = [{return cast<MemSDNode>(N)->getAlignment() < 4;}],
    GISelPredicateCode = [{return (*MI.memoperands_begin())->getAlign() < 4;}],
    AddressSpaces = [ AddrSpaces.Local ] in {
def load_align_less_than_4_local : PatFrag<(ops node:$ptr),
                                           (load_local node:$ptr)> {
  let IsLoad = 1;
  let IsNonExtLoad = 1;
}

def load_align_less_than_4_local_m0 : PatFrag<(ops node:$ptr),
                                              (load_local_m0 node:$ptr)> {
  let IsLoad = 1;
  let IsNonExtLoad = 1;
}

def store_align_less_than_4_local : PatFrag <(ops node:$value, node:$ptr),
                                             (store_local node:$value, node:$ptr)> {
  let IsStore = 1;
  let IsTruncStore = 0;
}

def store_align_less_than_4_local_m0 : PatFrag <(ops node:$value, node:$ptr),
                                                (store_local_m0 node:$value, node:$ptr)> {
  let IsStore = 1;
  let IsTruncStore = 0;
}
}

def atomic_store_8_glue : PatFrag <
  (ops node:$ptr, node:$value),
  (AMDGPUatomic_st_glue node:$ptr, node:$value)> {
  let IsAtomic = 1;
  let MemoryVT = i8;
}

def atomic_store_16_glue : PatFrag <
  (ops node:$ptr, node:$value),
  (AMDGPUatomic_st_glue node:$ptr, node:$value)> {
  let IsAtomic = 1;
  let MemoryVT = i16;
}

def atomic_store_32_glue : PatFrag <
  (ops node:$ptr, node:$value),
  (AMDGPUatomic_st_glue node:$ptr, node:$value)> {
  let IsAtomic = 1;
  let MemoryVT = i32;
}

def atomic_store_64_glue : PatFrag <
  (ops node:$ptr, node:$value),
  (AMDGPUatomic_st_glue node:$ptr, node:$value)> {
  let IsAtomic = 1;
  let MemoryVT = i64;
}

let IsAtomic = 1, AddressSpaces = StoreAddress_local.AddrSpaces in {
def atomic_store_8_local_m0 : PatFrag<(ops node:$ptr, node:$val),
                                       (atomic_store_8_glue node:$ptr, node:$val)>;
def atomic_store_16_local_m0 : PatFrag<(ops node:$ptr, node:$val),
                                       (atomic_store_16_glue node:$ptr, node:$val)>;
def atomic_store_32_local_m0 : PatFrag<(ops node:$ptr, node:$val),
                                       (atomic_store_32_glue node:$ptr, node:$val)>;
def atomic_store_64_local_m0 : PatFrag<(ops node:$ptr, node:$val),
                                       (atomic_store_64_glue node:$ptr, node:$val)>;
} // End let IsAtomic = 1, AddressSpaces = StoreAddress_local.AddrSpaces


def si_setcc_uniform : PatFrag <
  (ops node:$lhs, node:$rhs, node:$cond),
  (setcc node:$lhs, node:$rhs, node:$cond), [{
  return !N->isDivergent();
}]>;

//===----------------------------------------------------------------------===//
// SDNodes PatFrags for a16 loads and stores with 3 components.
// v3f16/v3i16 is widened to v4f16/v4i16, so we need to match on the memory
// load/store size.
//===----------------------------------------------------------------------===//

class mubuf_intrinsic_load<SDPatternOperator name, ValueType vt> : PatFrag <
  (ops node:$rsrc, node:$vindex, node:$voffset, node:$soffset, node:$offset,
            node:$auxiliary, node:$idxen),
  (name node:$rsrc, node:$vindex, node:$voffset, node:$soffset, node:$offset,
            node:$auxiliary, node:$idxen)> {
  let IsLoad = 1;
  let MemoryVT = vt;
}

class mubuf_intrinsic_store<SDPatternOperator name, ValueType vt> : PatFrag <
  (ops node:$vdata, node:$rsrc, node:$vindex, node:$voffset, node:$soffset, node:$offset,
            node:$auxiliary, node:$idxen),
  (name node:$vdata, node:$rsrc, node:$vindex, node:$voffset, node:$soffset, node:$offset,
            node:$auxiliary, node:$idxen)> {
  let IsStore = 1;
  let MemoryVT = vt;
}

class mtbuf_intrinsic_load<SDPatternOperator name, ValueType vt> : PatFrag <
  (ops node:$rsrc, node:$vindex, node:$voffset, node:$soffset, node:$offset,
            node:$format, node:$auxiliary, node:$idxen),
  (name node:$rsrc, node:$vindex, node:$voffset, node:$soffset, node:$offset,
            node:$format, node:$auxiliary, node:$idxen)> {
  let IsLoad = 1;
  let MemoryVT = vt;
}

class mtbuf_intrinsic_store<SDPatternOperator name, ValueType vt> : PatFrag <
  (ops node:$vdata, node:$rsrc, node:$vindex, node:$voffset, node:$soffset, node:$offset,
            node:$format, node:$auxiliary, node:$idxen),
  (name node:$vdata, node:$rsrc, node:$vindex, node:$voffset, node:$soffset, node:$offset,
            node:$format, node:$auxiliary, node:$idxen)> {
  let IsStore = 1;
  let MemoryVT = vt;
}

//===----------------------------------------------------------------------===//
// SDNodes PatFrags for d16 loads
//===----------------------------------------------------------------------===//

class LoadD16Frag <SDPatternOperator op> : PatFrag<
  (ops node:$ptr, node:$tied_in),
  (op node:$ptr, node:$tied_in)> {
  let IsLoad = 1;
}

foreach as = [ "global", "flat", "constant", "local", "private", "region" ] in {
let AddressSpaces = !cast<AddressSpaceList>("LoadAddress_"#as).AddrSpaces in {

def load_d16_hi_#as : LoadD16Frag <SIload_d16_hi>;

def az_extloadi8_d16_hi_#as : LoadD16Frag <SIload_d16_hi_u8> {
  let MemoryVT = i8;
}

def sextloadi8_d16_hi_#as : LoadD16Frag <SIload_d16_hi_i8> {
  let MemoryVT = i8;
}

def load_d16_lo_#as : LoadD16Frag <SIload_d16_lo>;

def az_extloadi8_d16_lo_#as : LoadD16Frag <SIload_d16_lo_u8> {
  let MemoryVT = i8;
}

def sextloadi8_d16_lo_#as : LoadD16Frag <SIload_d16_lo_i8> {
  let MemoryVT = i8;
}

} // End let AddressSpaces = ...
} // End foreach AddrSpace

def lshr_rev : PatFrag <
  (ops node:$src1, node:$src0),
  (srl $src0, $src1)
>;

def ashr_rev : PatFrag <
  (ops node:$src1, node:$src0),
  (sra $src0, $src1)
>;

def lshl_rev : PatFrag <
  (ops node:$src1, node:$src0),
  (shl $src0, $src1)
>;

def add_ctpop : PatFrag <
  (ops node:$src0, node:$src1),
  (add (ctpop $src0), $src1)
>;

def xnor : PatFrag <
  (ops node:$src0, node:$src1),
  (not (xor $src0, $src1))
>;

foreach I = 1-4 in {
def shl#I#_add : PatFrag <
  (ops node:$src0, node:$src1),
  (add (shl_oneuse $src0, (i32 I)), $src1)> {
  // FIXME: Poor substitute for disabling pattern in SelectionDAG
  let PredicateCode = [{return false;}];
  let GISelPredicateCode = [{return true;}];
}
}

multiclass SIAtomicM0Glue2 <string op_name, bit is_amdgpu = 0,
                            SDTypeProfile tc = SDTAtomic2,
                            bit IsInt = 1> {

  def _glue : SDNode <
    !if(is_amdgpu, "AMDGPUISD", "ISD")#"::ATOMIC_"#op_name, tc,
    [SDNPHasChain, SDNPMayStore, SDNPMayLoad, SDNPMemOperand, SDNPInGlue]
  >;

  let AddressSpaces = StoreAddress_local.AddrSpaces in {
    defm _local_m0 : binary_atomic_op <!cast<SDNode>(NAME#"_glue"), IsInt>;
    defm _local_m0 : noret_binary_atomic_op <!cast<SDNode>(NAME#"_glue"),
                                                 IsInt>;
  }

  let AddressSpaces = StoreAddress_region.AddrSpaces in {
    defm _region_m0 : binary_atomic_op <!cast<SDNode>(NAME#"_glue"), IsInt>;
    defm _region_m0 : noret_binary_atomic_op <!cast<SDNode>(NAME#"_glue"),
                                                  IsInt>;
  }
}

defm atomic_load_add : SIAtomicM0Glue2 <"LOAD_ADD">;
defm atomic_load_sub : SIAtomicM0Glue2 <"LOAD_SUB">;
defm atomic_inc : SIAtomicM0Glue2 <"INC", 1>;
defm atomic_dec : SIAtomicM0Glue2 <"DEC", 1>;
defm atomic_load_and : SIAtomicM0Glue2 <"LOAD_AND">;
defm atomic_load_min : SIAtomicM0Glue2 <"LOAD_MIN">;
defm atomic_load_max : SIAtomicM0Glue2 <"LOAD_MAX">;
defm atomic_load_or : SIAtomicM0Glue2 <"LOAD_OR">;
defm atomic_load_xor : SIAtomicM0Glue2 <"LOAD_XOR">;
defm atomic_load_umin : SIAtomicM0Glue2 <"LOAD_UMIN">;
defm atomic_load_umax : SIAtomicM0Glue2 <"LOAD_UMAX">;
defm atomic_swap : SIAtomicM0Glue2 <"SWAP">;
defm atomic_load_fadd : SIAtomicM0Glue2 <"LOAD_FADD", 0, SDTAtomic2_f32, 0>;
defm atomic_load_fmin : SIAtomicM0Glue2 <"LOAD_FMIN", 1, SDTAtomic2_f32, 0>;
defm atomic_load_fmax : SIAtomicM0Glue2 <"LOAD_FMAX", 1, SDTAtomic2_f32, 0>;

def as_i1timm : SDNodeXForm<timm, [{
  return CurDAG->getTargetConstant(N->getZExtValue(), SDLoc(N), MVT::i1);
}]>;

def as_i8imm : SDNodeXForm<imm, [{
  return CurDAG->getTargetConstant(N->getZExtValue(), SDLoc(N), MVT::i8);
}]>;

def as_i8timm : SDNodeXForm<timm, [{
  return CurDAG->getTargetConstant(N->getSExtValue(), SDLoc(N), MVT::i16);
}]>;

def as_i16imm : SDNodeXForm<imm, [{
  return CurDAG->getTargetConstant(N->getSExtValue(), SDLoc(N), MVT::i16);
}]>;

def as_i16timm : SDNodeXForm<timm, [{
  return CurDAG->getTargetConstant(N->getSExtValue(), SDLoc(N), MVT::i16);
}]>;

def as_i32imm: SDNodeXForm<imm, [{
  return CurDAG->getTargetConstant(N->getSExtValue(), SDLoc(N), MVT::i32);
}]>;

def as_i32timm: SDNodeXForm<timm, [{
  return CurDAG->getTargetConstant(N->getSExtValue(), SDLoc(N), MVT::i32);
}]>;

def as_i64imm: SDNodeXForm<imm, [{
  return CurDAG->getTargetConstant(N->getSExtValue(), SDLoc(N), MVT::i64);
}]>;

def cond_as_i32imm: SDNodeXForm<cond, [{
  return CurDAG->getTargetConstant(N->get(), SDLoc(N), MVT::i32);
}]>;

// Copied from the AArch64 backend:
def bitcast_fpimm_to_i32 : SDNodeXForm<fpimm, [{
return CurDAG->getTargetConstant(
  N->getValueAPF().bitcastToAPInt().getZExtValue(), SDLoc(N), MVT::i32);
}]>;

def frameindex_to_targetframeindex : SDNodeXForm<frameindex, [{
  auto FI = cast<FrameIndexSDNode>(N);
  return CurDAG->getTargetFrameIndex(FI->getIndex(), MVT::i32);
}]>;

// Copied from the AArch64 backend:
def bitcast_fpimm_to_i64 : SDNodeXForm<fpimm, [{
return CurDAG->getTargetConstant(
  N->getValueAPF().bitcastToAPInt().getZExtValue(), SDLoc(N), MVT::i64);
}]>;

class bitextract_imm<int bitnum> : SDNodeXForm<imm, [{
  uint64_t Imm = N->getZExtValue();
  unsigned Bit = (Imm >> }] # bitnum # [{ ) & 1;
  return CurDAG->getTargetConstant(Bit, SDLoc(N), MVT::i1);
}]>;

def SIMM16bit : ImmLeaf <i32,
  [{return isInt<16>(Imm);}]
>;

def UIMM16bit : ImmLeaf <i32,
  [{return isUInt<16>(Imm);}]
>;

def i64imm_32bit : ImmLeaf<i64, [{
  return (Imm & 0xffffffffULL) == static_cast<uint64_t>(Imm);
}]>;

def InlineImm16 : ImmLeaf<i16, [{
  return isInlineImmediate16(Imm);
}]>;

def InlineImm32 : ImmLeaf<i32, [{
  return isInlineImmediate32(Imm);
}]>;

def InlineImm64 : ImmLeaf<i64, [{
  return isInlineImmediate64(Imm);
}]>;

def InlineImmFP32 : FPImmLeaf<f32, [{
  return isInlineImmediate(Imm);
}]>;

def InlineImmFP64 : FPImmLeaf<f64, [{
  return isInlineImmediate(Imm);
}]>;


class VGPRImm <dag frag> : PatLeaf<frag, [{
  return isVGPRImm(N);
}]>;

def NegateImm : SDNodeXForm<imm, [{
  return CurDAG->getConstant(-N->getSExtValue(), SDLoc(N), MVT::i32);
}]>;

// TODO: When FP inline imm values work?
def NegSubInlineConst32 : ImmLeaf<i32, [{
  return Imm < -16 && Imm >= -64;
}], NegateImm>;

def NegSubInlineIntConst16 : ImmLeaf<i16, [{
  return Imm < -16 && Imm >= -64;
}], NegateImm>;

def ShiftAmt32Imm : ImmLeaf <i32, [{
  return Imm < 32;
}]>;

def getNegV2I16Imm : SDNodeXForm<build_vector, [{
  return SDValue(packNegConstantV2I16(N, *CurDAG), 0);
}]>;

def NegSubInlineConstV216 : PatLeaf<(build_vector), [{
  assert(N->getNumOperands() == 2);
  assert(N->getOperand(0).getValueType().getSizeInBits() == 16);
  SDValue Src0 = N->getOperand(0);
  SDValue Src1 = N->getOperand(1);
  if (Src0 == Src1)
    return isNegInlineImmediate(Src0.getNode());

  return (isNullConstantOrUndef(Src0) && isNegInlineImmediate(Src1.getNode())) ||
         (isNullConstantOrUndef(Src1) && isNegInlineImmediate(Src0.getNode()));
}], getNegV2I16Imm>;


def fp16_zeros_high_16bits : PatLeaf<(f16 VGPR_32:$src), [{
  return fp16SrcZerosHighBits(N->getOpcode());
}]>;


//===----------------------------------------------------------------------===//
// MUBUF/SMEM Patterns
//===----------------------------------------------------------------------===//

def extract_cpol : SDNodeXForm<timm, [{
  return CurDAG->getTargetConstant(N->getZExtValue() & AMDGPU::CPol::ALL, SDLoc(N), MVT::i8);
}]>;

def extract_swz : SDNodeXForm<timm, [{
  return CurDAG->getTargetConstant((N->getZExtValue() >> 3) & 1, SDLoc(N), MVT::i8);
}]>;

def set_glc : SDNodeXForm<timm, [{
  return CurDAG->getTargetConstant(N->getZExtValue() | AMDGPU::CPol::GLC, SDLoc(N), MVT::i8);
}]>;

//===----------------------------------------------------------------------===//
// Custom Operands
//===----------------------------------------------------------------------===//

def SoppBrTarget : AsmOperandClass {
  let Name = "SoppBrTarget";
  let ParserMethod = "parseSOppBrTarget";
}

def sopp_brtarget : Operand<OtherVT> {
  let EncoderMethod = "getSOPPBrEncoding";
  let DecoderMethod = "decodeSoppBrTarget";
  let OperandType = "OPERAND_PCREL";
  let ParserMatchClass = SoppBrTarget;
}

def si_ga : Operand<iPTR>;

def InterpSlotMatchClass : AsmOperandClass {
  let Name = "InterpSlot";
  let PredicateMethod = "isInterpSlot";
  let ParserMethod = "parseInterpSlot";
  let RenderMethod = "addImmOperands";
}

def InterpSlot : Operand<i32> {
  let PrintMethod = "printInterpSlot";
  let ParserMatchClass = InterpSlotMatchClass;
  let OperandType = "OPERAND_IMMEDIATE";
}

def AttrMatchClass : AsmOperandClass {
  let Name = "Attr";
  let PredicateMethod = "isInterpAttr";
  let ParserMethod = "parseInterpAttr";
  let RenderMethod = "addImmOperands";
}

// It appears to be necessary to create a separate operand for this to
// be able to parse attr<num> with no space.
def Attr : Operand<i32> {
  let PrintMethod = "printInterpAttr";
  let ParserMatchClass = AttrMatchClass;
  let OperandType = "OPERAND_IMMEDIATE";
}

def AttrChanMatchClass : AsmOperandClass {
  let Name = "AttrChan";
  let PredicateMethod = "isAttrChan";
  let RenderMethod = "addImmOperands";
}

def AttrChan : Operand<i32> {
  let PrintMethod = "printInterpAttrChan";
  let ParserMatchClass = AttrChanMatchClass;
  let OperandType = "OPERAND_IMMEDIATE";
}

def SendMsgMatchClass : AsmOperandClass {
  let Name = "SendMsg";
  let PredicateMethod = "isSendMsg";
  let ParserMethod = "parseSendMsgOp";
  let RenderMethod = "addImmOperands";
}

def SwizzleMatchClass : AsmOperandClass {
  let Name = "Swizzle";
  let PredicateMethod = "isSwizzle";
  let ParserMethod = "parseSwizzleOp";
  let RenderMethod = "addImmOperands";
  let IsOptional = 1;
}

def EndpgmMatchClass : AsmOperandClass {
  let Name = "EndpgmImm";
  let PredicateMethod = "isEndpgm";
  let ParserMethod = "parseEndpgmOp";
  let RenderMethod = "addImmOperands";
  let IsOptional = 1;
}

def ExpTgtMatchClass : AsmOperandClass {
  let Name = "ExpTgt";
  let PredicateMethod = "isExpTgt";
  let ParserMethod = "parseExpTgt";
  let RenderMethod = "printExpTgt";
}

def SWaitMatchClass : AsmOperandClass {
  let Name = "SWaitCnt";
  let RenderMethod = "addImmOperands";
  let ParserMethod = "parseSWaitCntOps";
}

def DepCtrMatchClass : AsmOperandClass {
  let Name = "DepCtr";
  let RenderMethod = "addImmOperands";
  let ParserMethod = "parseDepCtrOps";
}

def SDelayMatchClass : AsmOperandClass {
  let Name = "SDelayAlu";
  let RenderMethod = "addImmOperands";
  let ParserMethod = "parseSDelayAluOps";
}

def VReg32OrOffClass : AsmOperandClass {
  let Name = "VReg32OrOff";
  let ParserMethod = "parseVReg32OrOff";
}

let OperandType = "OPERAND_IMMEDIATE" in {
def SendMsgImm : Operand<i32> {
  let PrintMethod = "printSendMsg";
  let ParserMatchClass = SendMsgMatchClass;
}

def SwizzleImm : Operand<i16> {
  let PrintMethod = "printSwizzle";
  let ParserMatchClass = SwizzleMatchClass;
}

def EndpgmImm : Operand<i16> {
  let PrintMethod = "printEndpgm";
  let ParserMatchClass = EndpgmMatchClass;
}

def WAIT_FLAG : Operand <i32> {
  let ParserMatchClass = SWaitMatchClass;
  let PrintMethod = "printWaitFlag";
}

def DepCtrImm : Operand <i32> {
  let ParserMatchClass = DepCtrMatchClass;
  let PrintMethod = "printDepCtr";
}

def DELAY_FLAG : Operand <i32> {
  let ParserMatchClass = SDelayMatchClass;
  let PrintMethod = "printDelayFlag";
}
} // End OperandType = "OPERAND_IMMEDIATE"

include "SIInstrFormats.td"
include "VIInstrFormats.td"

def BoolReg : AsmOperandClass {
  let Name = "BoolReg";
  let ParserMethod = "parseBoolReg";
  let RenderMethod = "addRegOperands";
}

class BoolRC : RegisterOperand<SReg_1> {
  let ParserMatchClass = BoolReg;
  let DecoderMethod = "decodeBoolReg";
}

def SSrc_i1 : RegisterOperand<SReg_1_XEXEC> {
  let ParserMatchClass = BoolReg;
  let DecoderMethod = "decodeBoolReg";
}

def VOPDstS64orS32 : BoolRC {
  let PrintMethod = "printVOPDst";
}

// SCSrc_i1 is the operand for pseudo instructions only.
// Boolean immediates shall not be exposed to codegen instructions.
def SCSrc_i1 : RegisterOperand<SReg_1_XEXEC> {
  let OperandNamespace = "AMDGPU";
  let OperandType = "OPERAND_REG_IMM_INT32";
  let ParserMatchClass = BoolReg;
  let DecoderMethod = "decodeBoolReg";
}

// ===----------------------------------------------------------------------===//
// ExpSrc* Special cases for exp src operands which are printed as
// "off" depending on en operand.
// ===----------------------------------------------------------------------===//

def ExpSrc0 : RegisterOperand<VGPR_32> {
  let PrintMethod = "printExpSrc0";
  let ParserMatchClass = VReg32OrOffClass;
}

def ExpSrc1 : RegisterOperand<VGPR_32> {
  let PrintMethod = "printExpSrc1";
  let ParserMatchClass = VReg32OrOffClass;
}

def ExpSrc2 : RegisterOperand<VGPR_32> {
  let PrintMethod = "printExpSrc2";
  let ParserMatchClass = VReg32OrOffClass;
}

def ExpSrc3 : RegisterOperand<VGPR_32> {
  let PrintMethod = "printExpSrc3";
  let ParserMatchClass = VReg32OrOffClass;
}

class SDWASrc<ValueType vt> : RegisterOperand<VS_32> {
  let OperandNamespace = "AMDGPU";
  string Type = !if(isFloatType<vt>.ret, "FP", "INT");
  let OperandType = "OPERAND_REG_INLINE_C_"#Type#vt.Size;
  let DecoderMethod = "decodeSDWASrc"#vt.Size;
  let EncoderMethod = "getSDWASrcEncoding";
}

def SDWASrc_i32 : SDWASrc<i32>;
def SDWASrc_i16 : SDWASrc<i16>;
def SDWASrc_f32 : SDWASrc<f32>;
def SDWASrc_f16 : SDWASrc<f16>;

def SDWAVopcDst : BoolRC {
  let OperandNamespace = "AMDGPU";
  let OperandType = "OPERAND_SDWA_VOPC_DST";
  let EncoderMethod = "getSDWAVopcDstEncoding";
  let DecoderMethod = "decodeSDWAVopcDst";
  let PrintMethod = "printVOPDst";
}

class NamedMatchClass<string CName, bit Optional = 1> : AsmOperandClass {
  let Name = "Imm"#CName;
  let PredicateMethod = "is"#CName;
  let ParserMethod = !if(Optional, "parseOptionalOperand", "parse"#CName);
  let RenderMethod = "addImmOperands";
  let IsOptional = Optional;
  let DefaultMethod = !if(Optional, "default"#CName, ?);
}

class NamedOperandBit<string Name, AsmOperandClass MatchClass> : Operand<i1> {
  let PrintMethod = "print"#Name;
  let ParserMatchClass = MatchClass;
}

class NamedOperandBit_0<string Name, AsmOperandClass MatchClass> :
  OperandWithDefaultOps<i1, (ops (i1 0))> {
  let PrintMethod = "print"#Name;
  let ParserMatchClass = MatchClass;
}

class NamedOperandBit_1<string Name, AsmOperandClass MatchClass> :
  OperandWithDefaultOps<i1, (ops (i1 1))> {
  let PrintMethod = "print"#Name;
  let ParserMatchClass = MatchClass;
}

class NamedOperandU8<string Name, AsmOperandClass MatchClass> : Operand<i8> {
  let PrintMethod = "print"#Name;
  let ParserMatchClass = MatchClass;
}

class NamedOperandU16<string Name, AsmOperandClass MatchClass> : Operand<i16> {
  let PrintMethod = "print"#Name;
  let ParserMatchClass = MatchClass;
}

class NamedOperandU32<string Name, AsmOperandClass MatchClass> : Operand<i32> {
  let PrintMethod = "print"#Name;
  let ParserMatchClass = MatchClass;
}

class NamedOperandU32_0<string Name, AsmOperandClass MatchClass> :
  OperandWithDefaultOps<i32, (ops (i32 0))> {
  let PrintMethod = "print"#Name;
  let ParserMatchClass = MatchClass;
}

class NamedOperandU32Default0<string Name, AsmOperandClass MatchClass> :
  OperandWithDefaultOps<i32, (ops (i32 0))> {
  let PrintMethod = "print"#Name;
  let ParserMatchClass = MatchClass;
}

class NamedOperandU32Default1<string Name, AsmOperandClass MatchClass> :
  OperandWithDefaultOps<i32, (ops (i32 1))> {
  let PrintMethod = "print"#Name;
  let ParserMatchClass = MatchClass;
}

let OperandType = "OPERAND_IMMEDIATE" in {

def offen : NamedOperandBit<"Offen", NamedMatchClass<"Offen">>;
def idxen : NamedOperandBit<"Idxen", NamedMatchClass<"Idxen">>;
def addr64 : NamedOperandBit<"Addr64", NamedMatchClass<"Addr64">>;

def flat_offset : NamedOperandU16<"FlatOffset", NamedMatchClass<"FlatOffset">>;
def offset : NamedOperandU16<"Offset", NamedMatchClass<"Offset">>;
def offset0 : NamedOperandU8<"Offset0", NamedMatchClass<"Offset0">>;
def offset1 : NamedOperandU8<"Offset1", NamedMatchClass<"Offset1">>;

def gds : NamedOperandBit<"GDS", NamedMatchClass<"GDS">>;

def omod : NamedOperandU32<"OModSI", NamedMatchClass<"OModSI">>;
def omod0 : NamedOperandU32_0<"OModSI", NamedMatchClass<"OModSI">>;

// We need to make the cases with a default of 0 distinct from no
// default to help deal with some cases where the operand appears
// before a mandatory operand.
def clampmod : NamedOperandBit<"ClampSI", NamedMatchClass<"ClampSI">>;
def clampmod0 : NamedOperandBit_0<"ClampSI", NamedMatchClass<"ClampSI">>;
def highmod : NamedOperandBit<"High", NamedMatchClass<"High">>;

def CPol : NamedOperandU32<"CPol", NamedMatchClass<"CPol">>;
def CPol_0 : NamedOperandU32Default0<"CPol", NamedMatchClass<"CPol">>;
def CPol_GLC1 : NamedOperandU32Default1<"CPol", NamedMatchClass<"CPol">>;

def TFE : NamedOperandBit<"TFE", NamedMatchClass<"TFE">>;
def TFE_0 : NamedOperandBit_0<"TFE", NamedMatchClass<"TFE">>;
def SWZ : NamedOperandBit<"SWZ", NamedMatchClass<"SWZ">>;
def SWZ_0 : NamedOperandBit_0<"SWZ", NamedMatchClass<"SWZ">>;
def UNorm : NamedOperandBit<"UNorm", NamedMatchClass<"UNorm">>;
def DA : NamedOperandBit<"DA", NamedMatchClass<"DA">>;
def R128A16 : NamedOperandBit<"R128A16", NamedMatchClass<"R128A16">>;
def GFX10A16 : NamedOperandBit<"GFX10A16", NamedMatchClass<"GFX10A16">>;
def D16 : NamedOperandBit<"D16", NamedMatchClass<"D16">>;
def LWE : NamedOperandBit<"LWE", NamedMatchClass<"LWE">>;
def exp_compr : NamedOperandBit<"ExpCompr", NamedMatchClass<"ExpCompr">>;
def exp_vm : NamedOperandBit<"ExpVM", NamedMatchClass<"ExpVM">>;

def FORMAT : NamedOperandU8<"FORMAT", NamedMatchClass<"FORMAT", 0>>;

def DMask : NamedOperandU16<"DMask", NamedMatchClass<"DMask">>;
def Dim : NamedOperandU8<"Dim", NamedMatchClass<"Dim", 0>>;

def dst_sel : NamedOperandU32<"SDWADstSel", NamedMatchClass<"SDWADstSel">>;
def src0_sel : NamedOperandU32<"SDWASrc0Sel", NamedMatchClass<"SDWASrc0Sel">>;
def src1_sel : NamedOperandU32<"SDWASrc1Sel", NamedMatchClass<"SDWASrc1Sel">>;
def dst_unused : NamedOperandU32<"SDWADstUnused", NamedMatchClass<"SDWADstUnused">>;

def op_sel0 : NamedOperandU32Default0<"OpSel", NamedMatchClass<"OpSel">>;
def op_sel_hi0 : NamedOperandU32Default0<"OpSelHi", NamedMatchClass<"OpSelHi">>;
def neg_lo0 : NamedOperandU32Default0<"NegLo", NamedMatchClass<"NegLo">>;
def neg_hi0 : NamedOperandU32Default0<"NegHi", NamedMatchClass<"NegHi">>;

def dpp8 : NamedOperandU32<"DPP8", NamedMatchClass<"DPP8", 0>>;
def dpp_ctrl : NamedOperandU32<"DPPCtrl", NamedMatchClass<"DPPCtrl", 0>>;

def row_mask : NamedOperandU32<"RowMask", NamedMatchClass<"RowMask">>;
def bank_mask : NamedOperandU32<"BankMask", NamedMatchClass<"BankMask">>;
def bound_ctrl : NamedOperandBit<"BoundCtrl", NamedMatchClass<"BoundCtrl">>;
def FI : NamedOperandU32<"FI", NamedMatchClass<"FI">>;

def blgp : NamedOperandU32<"BLGP", NamedMatchClass<"BLGP">>;
def cbsz : NamedOperandU32<"CBSZ", NamedMatchClass<"CBSZ">>;
def abid : NamedOperandU32<"ABID", NamedMatchClass<"ABID">>;

def hwreg : NamedOperandU32<"Hwreg", NamedMatchClass<"Hwreg", 0>>;

def exp_tgt : NamedOperandU32<"ExpTgt", NamedMatchClass<"ExpTgt", 0>> {

}

def wait_vdst : NamedOperandU8<"WaitVDST", NamedMatchClass<"WaitVDST">>;
def wait_exp : NamedOperandU8<"WaitEXP", NamedMatchClass<"WaitEXP">>;

} // End OperandType = "OPERAND_IMMEDIATE"

class KImmMatchClass<int size> : AsmOperandClass {
  let Name = "KImmFP"#size;
  let PredicateMethod = "isKImmFP"#size;
  let ParserMethod = "parseImm";
  let RenderMethod = "addKImmFP"#size#"Operands";
}

class kimmOperand<ValueType vt> : Operand<vt> {
  let OperandNamespace = "AMDGPU";
  let OperandType = "OPERAND_KIMM"#vt.Size;
  let PrintMethod = "printU"#vt.Size#"ImmOperand";
  let ParserMatchClass = !cast<AsmOperandClass>("KImmFP"#vt.Size#"MatchClass");
  let DecoderMethod = "decodeOperand_f"#vt.Size#"kimm";
}

// 32-bit VALU immediate operand that uses the constant bus.
def KImmFP32MatchClass : KImmMatchClass<32>;
def f32kimm : kimmOperand<i32>;

// 32-bit VALU immediate operand with a 16-bit value that uses the
// constant bus.
def KImmFP16MatchClass : KImmMatchClass<16>;
def f16kimm : kimmOperand<i16>;

class FPInputModsMatchClass <int opSize> : AsmOperandClass {
  let Name = "RegOrImmWithFP"#opSize#"InputMods";
  let ParserMethod = "parseRegOrImmWithFPInputMods";
  let PredicateMethod = "isRegOrImmWithFP"#opSize#"InputMods";
}

class FPVCSrcInputModsMatchClass <int opSize> : FPInputModsMatchClass <opSize> {
  let Name = "RegOrInlineImmWithFP"#opSize#"InputMods";
  let PredicateMethod = "isRegOrInlineImmWithFP"#opSize#"InputMods";
}

def FP16InputModsMatchClass : FPInputModsMatchClass<16>;
def TFP16InputModsMatchClass : FPInputModsMatchClass<16> {
  let Name = "RegOrImmWithTFP16InputMods";
  let PredicateMethod = "isRegOrImmWithTFP16InputMods";
}
def FP32InputModsMatchClass : FPInputModsMatchClass<32>;
def FP64InputModsMatchClass : FPInputModsMatchClass<64>;

def FP16VCSrcInputModsMatchClass : FPVCSrcInputModsMatchClass<16>;
def FP32VCSrcInputModsMatchClass : FPVCSrcInputModsMatchClass<32>;

class InputMods <AsmOperandClass matchClass> : Operand <i32> {
  let OperandNamespace = "AMDGPU";
  let OperandType = "OPERAND_INPUT_MODS";
  let ParserMatchClass = matchClass;
}

class FPInputMods <FPInputModsMatchClass matchClass> : InputMods <matchClass> {
  let PrintMethod = "printOperandAndFPInputMods";
}

def FP16InputMods : FPInputMods<FP16InputModsMatchClass>;
def TFP16InputMods : FPInputMods<TFP16InputModsMatchClass>;
def FP32InputMods : FPInputMods<FP32InputModsMatchClass>;
def FP64InputMods : FPInputMods<FP64InputModsMatchClass>;

def FP16VCSrcInputMods : FPInputMods<FP16VCSrcInputModsMatchClass>;
def FP32VCSrcInputMods : FPInputMods<FP32VCSrcInputModsMatchClass>;

class IntInputModsMatchClass <int opSize> : AsmOperandClass {
  let Name = "RegOrImmWithInt"#opSize#"InputMods";
  let ParserMethod = "parseRegOrImmWithIntInputMods";
  let PredicateMethod = "isRegOrImmWithInt"#opSize#"InputMods";
}
class IntVCSrcInputModsMatchClass <int opSize> : IntInputModsMatchClass <opSize> {
  let Name = "RegOrInlineImmWithInt"#opSize#"InputMods";
  let PredicateMethod = "isRegOrInlineImmWithInt"#opSize#"InputMods";
}
def Int32InputModsMatchClass : IntInputModsMatchClass<32>;
def Int64InputModsMatchClass : IntInputModsMatchClass<64>;
def Int32VCSrcInputModsMatchClass : IntVCSrcInputModsMatchClass<32>;

class IntInputMods <IntInputModsMatchClass matchClass> : InputMods <matchClass> {
  let PrintMethod = "printOperandAndIntInputMods";
}
def Int32InputMods : IntInputMods<Int32InputModsMatchClass>;
def Int64InputMods : IntInputMods<Int64InputModsMatchClass>;
def Int32VCSrcInputMods : IntInputMods<Int32VCSrcInputModsMatchClass>;

class OpSelModsMatchClass : AsmOperandClass {
  let Name = "OpSelMods";
  let ParserMethod = "parseRegOrImm";
  let PredicateMethod = "isRegOrImm";
}

def IntOpSelModsMatchClass : OpSelModsMatchClass;
def IntOpSelMods : InputMods<IntOpSelModsMatchClass>;

class FPSDWAInputModsMatchClass <int opSize> : AsmOperandClass {
  let Name = "SDWAWithFP"#opSize#"InputMods";
  let ParserMethod = "parseRegOrImmWithFPInputMods";
  let PredicateMethod = "isSDWAFP"#opSize#"Operand";
}

def FP16SDWAInputModsMatchClass : FPSDWAInputModsMatchClass<16>;
def FP32SDWAInputModsMatchClass : FPSDWAInputModsMatchClass<32>;

class FPSDWAInputMods <FPSDWAInputModsMatchClass matchClass> :
  InputMods <matchClass> {
  let PrintMethod = "printOperandAndFPInputMods";
}

def FP16SDWAInputMods : FPSDWAInputMods<FP16SDWAInputModsMatchClass>;
def FP32SDWAInputMods : FPSDWAInputMods<FP32SDWAInputModsMatchClass>;

def FPVRegInputModsMatchClass : AsmOperandClass {
  let Name = "VRegWithFPInputMods";
  let ParserMethod = "parseRegWithFPInputMods";
  let PredicateMethod = "isVRegWithInputMods";
}

def FPVRegInputMods : InputMods <FPVRegInputModsMatchClass> {
  let PrintMethod = "printOperandAndFPInputMods";
}

class IntSDWAInputModsMatchClass <int opSize> : AsmOperandClass {
  let Name = "SDWAWithInt"#opSize#"InputMods";
  let ParserMethod = "parseRegOrImmWithIntInputMods";
  let PredicateMethod = "isSDWAInt"#opSize#"Operand";
}

def Int16SDWAInputModsMatchClass : IntSDWAInputModsMatchClass<16>;
def Int32SDWAInputModsMatchClass : IntSDWAInputModsMatchClass<32>;
def Bin32SDWAInputModsMatchClass : IntSDWAInputModsMatchClass<32> {
  let Name = "SDWAWithBin32InputMods";
  let ParserMethod = "parseRegOrImm";
}

class IntSDWAInputMods <IntSDWAInputModsMatchClass matchClass> :
  InputMods <matchClass> {
  let PrintMethod = "printOperandAndIntInputMods";
}

def Int16SDWAInputMods : IntSDWAInputMods<Int16SDWAInputModsMatchClass>;
def Int32SDWAInputMods : IntSDWAInputMods<Int32SDWAInputModsMatchClass>;
def Bin32SDWAInputMods : IntSDWAInputMods<Bin32SDWAInputModsMatchClass>;

def IntVRegInputModsMatchClass : AsmOperandClass {
  let Name = "VRegWithIntInputMods";
  let ParserMethod = "parseRegWithIntInputMods";
  let PredicateMethod = "isVRegWithInputMods";
}

def IntVRegInputMods : InputMods <IntVRegInputModsMatchClass> {
  let PrintMethod = "printOperandAndIntInputMods";
}

class PackedFPInputModsMatchClass <int opSize> : AsmOperandClass {
  let Name = "PackedFP"#opSize#"InputMods";
  let ParserMethod = "parseRegOrImm";
  let PredicateMethod = "isRegOrImm";
//  let PredicateMethod = "isPackedFP"#opSize#"InputMods";
}

class PackedIntInputModsMatchClass <int opSize> : AsmOperandClass {
  let Name = "PackedInt"#opSize#"InputMods";
  let ParserMethod = "parseRegOrImm";
  let PredicateMethod = "isRegOrImm";
//  let PredicateMethod = "isPackedInt"#opSize#"InputMods";
}

def PackedF16InputModsMatchClass : PackedFPInputModsMatchClass<16>;
def PackedI16InputModsMatchClass : PackedIntInputModsMatchClass<16>;

class PackedFPInputMods <PackedFPInputModsMatchClass matchClass> : InputMods <matchClass> {
//  let PrintMethod = "printPackedFPInputMods";
}

class PackedIntInputMods <PackedIntInputModsMatchClass matchClass> : InputMods <matchClass> {
  //let PrintMethod = "printPackedIntInputMods";
}

def PackedF16InputMods : PackedFPInputMods<PackedF16InputModsMatchClass>;
def PackedI16InputMods : PackedIntInputMods<PackedI16InputModsMatchClass>;

//===----------------------------------------------------------------------===//
// Complex patterns
//===----------------------------------------------------------------------===//

def DS1Addr1Offset : ComplexPattern<iPTR, 2, "SelectDS1Addr1Offset">;
def DS64Bit4ByteAligned : ComplexPattern<iPTR, 3, "SelectDS64Bit4ByteAligned">;
def DS128Bit8ByteAligned : ComplexPattern<iPTR, 3, "SelectDS128Bit8ByteAligned">;

def MOVRELOffset : ComplexPattern<iPTR, 2, "SelectMOVRELOffset">;

def VOP3Mods0 : ComplexPattern<untyped, 4, "SelectVOP3Mods0">;
def VOP3Mods  : ComplexPattern<untyped, 2, "SelectVOP3Mods">;
def VOP3NoMods : ComplexPattern<untyped, 1, "SelectVOP3NoMods">;
// VOP3Mods, but the input source is known to never be NaN.
def VOP3Mods_nnan : ComplexPattern<fAny, 2, "SelectVOP3Mods_NNaN">;

def VOP3OMods : ComplexPattern<untyped, 3, "SelectVOP3OMods">;

def VOP3PMods  : ComplexPattern<untyped, 2, "SelectVOP3PMods">;

def VOP3PModsDOT  : ComplexPattern<untyped, 2, "SelectVOP3PModsDOT">;
def DotIUVOP3PMods  : ComplexPattern<untyped, 1, "SelectDotIUVOP3PMods">;
def WMMAOpSelVOP3PMods  : ComplexPattern<untyped, 1, "SelectWMMAOpSelVOP3PMods">;

def VOP3OpSel  : ComplexPattern<untyped, 2, "SelectVOP3OpSel">;

def VOP3OpSelMods  : ComplexPattern<untyped, 2, "SelectVOP3OpSelMods">;

def VOP3PMadMixMods  : ComplexPattern<untyped, 2, "SelectVOP3PMadMixMods">;

def VINTERPMods  : ComplexPattern<untyped, 2, "SelectVINTERPMods">;
def VINTERPModsHi  : ComplexPattern<untyped, 2, "SelectVINTERPModsHi">;

//===----------------------------------------------------------------------===//
// SI assembler operands
//===----------------------------------------------------------------------===//

def SIOperand {
  int ZERO = 0x80;
  int VCC = 0x6A;
  int FLAT_SCR = 0x68;
}

// This should be kept in sync with SISrcMods enum
def SRCMODS {
  int NONE = 0;
  int NEG = 1;
  int ABS = 2;
  int NEG_ABS = 3;

  int NEG_HI = ABS;
  int OP_SEL_0 = 4;
  int OP_SEL_1 = 8;
  int DST_OP_SEL = 8;
}

def DSTCLAMP {
  int NONE = 0;
  int ENABLE = 1;
}

def DSTOMOD {
  int NONE = 0;
}

def HWREG {
  int MODE = 1;
  int STATUS = 2;
  int TRAPSTS = 3;
  int HW_ID = 4;
  int GPR_ALLOC = 5;
  int LDS_ALLOC = 6;
  int IB_STS = 7;
  int MEM_BASES = 15;
  int TBA_LO = 16;
  int TBA_HI = 17;
  int TMA_LO = 18;
  int TMA_HI = 19;
  int FLAT_SCR_LO = 20;
  int FLAT_SCR_HI = 21;
  int XNACK_MASK = 22;
  int POPS_PACKER = 25;
  int SHADER_CYCLES = 29;
}

class getHwRegImm<int Reg, int Offset = 0, int Size = 32> {
  int ret = !and(!or(Reg,
                     !shl(Offset, 6),
                     !shl(!add(Size, -1), 11)), 65535);
}

//===----------------------------------------------------------------------===//
//
// SI Instruction multiclass helpers.
//
// Instructions with _32 take 32-bit operands.
// Instructions with _64 take 64-bit operands.
//
// VOP_* instructions can use either a 32-bit or 64-bit encoding.  The 32-bit
// encoding is the standard encoding, but instruction that make use of
// any of the instruction modifiers must use the 64-bit encoding.
//
// Instructions with _e32 use the 32-bit encoding.
// Instructions with _e64 use the 64-bit encoding.
//
//===----------------------------------------------------------------------===//

class SIMCInstr <string pseudo, int subtarget> {
  string PseudoInstr = pseudo;
  int Subtarget = subtarget;
}

//===----------------------------------------------------------------------===//
// Vector ALU classes
//===----------------------------------------------------------------------===//

class getNumSrcArgs<ValueType Src0, ValueType Src1, ValueType Src2> {
  int ret =
    !if (!eq(Src0.Value, untyped.Value),      0,
      !if (!eq(Src1.Value, untyped.Value),    1,   // VOP1
         !if (!eq(Src2.Value, untyped.Value), 2,   // VOP2
                                              3))); // VOP3
}

// Returns the register class to use for the destination of VOP[123C]
// instructions for the given VT.
class getVALUDstForVT<ValueType VT> {
  RegisterOperand ret = !if(!eq(VT.Size, 32), VOPDstOperand<VGPR_32>,
                          !if(!eq(VT.Size, 128), VOPDstOperand<VReg_128>,
                            !if(!eq(VT.Size, 64), VOPDstOperand<VReg_64>,
                              !if(!eq(VT.Size, 16), VOPDstOperand<VGPR_32>,
                              VOPDstS64orS32)))); // else VT == i1
}

// Returns the register class to use for the destination of VOP[12C]
// instructions with SDWA extension
class getSDWADstForVT<ValueType VT> {
  RegisterOperand ret = !if(!eq(VT.Size, 1),
                            SDWAVopcDst, // VOPC
                            VOPDstOperand<VGPR_32>); // VOP1/2 32-bit dst
}

// Returns the register class to use for source 0 of VOP[12C]
// instructions for the given VT.
class getVOPSrc0ForVT<ValueType VT> {
  bit isFP = isFloatType<VT>.ret;

  RegisterOperand ret =
    !if(isFP,
      !if(!eq(VT.Size, 64),
         VSrc_f64,
         !if(!eq(VT.Value, f16.Value),
            VSrc_f16,
            !if(!eq(VT.Value, v2f16.Value),
               VSrc_v2f16,
               !if(!eq(VT.Value, v4f16.Value),
                 AVSrc_64,
                 VSrc_f32
               )
            )
         )
       ),
       !if(!eq(VT.Size, 64),
          VSrc_b64,
          !if(!eq(VT.Value, i16.Value),
             VSrc_b16,
             !if(!eq(VT.Value, v2i16.Value),
                VSrc_v2b16,
                VSrc_b32
             )
          )
       )
    );
}

class getSOPSrcForVT<ValueType VT> {
  RegisterOperand ret = !if(!eq(VT.Size, 64), SSrc_b64, SSrc_b32);
}

// Returns the vreg register class to use for source operand given VT
class getVregSrcForVT<ValueType VT> {
  RegisterClass ret = !if(!eq(VT.Size, 128), VReg_128,
                        !if(!eq(VT.Size, 96), VReg_96,
                          !if(!eq(VT.Size, 64), VReg_64,
                            !if(!eq(VT.Size, 48), VReg_64,
                              VGPR_32))));
}

class getSDWASrcForVT <ValueType VT> {
  bit isFP = isFloatType<VT>.ret;
  RegisterOperand retFlt = !if(!eq(VT.Size, 16), SDWASrc_f16, SDWASrc_f32);
  RegisterOperand retInt = !if(!eq(VT.Size, 16), SDWASrc_i16, SDWASrc_i32);
  RegisterOperand ret = !if(isFP, retFlt, retInt);
}

// Returns the register class to use for sources of VOP3 instructions for the
// given VT.
class getVOP3SrcForVT<ValueType VT> {
  bit isFP = isFloatType<VT>.ret;
  RegisterOperand ret =
  !if(!eq(VT.Size, 128),
     VSrc_128,
     !if(!eq(VT.Size, 64),
        !if(isFP,
           !if(!eq(VT.Value, v2f32.Value),
               VSrc_v2f32,
               VSrc_f64),
           !if(!eq(VT.Value, v2i32.Value),
               VSrc_v2b32,
           VSrc_b64)),
        !if(!eq(VT.Value, i1.Value),
           SSrc_i1,
           !if(isFP,
              !if(!eq(VT.Value, f16.Value),
                 VSrc_f16,
                 !if(!eq(VT.Value, v2f16.Value),
                    VSrc_v2f16,
                    !if(!eq(VT.Value, v4f16.Value),
                      AVSrc_64,
                      VSrc_f32
                    )
                 )
              ),
              !if(!eq(VT.Value, i16.Value),
                 VSrc_b16,
                 !if(!eq(VT.Value, v2i16.Value),
                    VSrc_v2b16,
                    VSrc_b32
                 )
              )
           )
        )
     )
  );
}

// Src2 of VOP3 DPP instructions cannot be a literal
class getVOP3DPPSrcForVT<ValueType VT> {
  bit isFP = isFloatType<VT>.ret;
  RegisterOperand ret =
      !if (!eq(VT.Value, i1.Value), SSrc_i1,
           !if (isFP,
                !if (!eq(VT.Value, f16.Value), VCSrc_f16,
                     !if (!eq(VT.Value, v2f16.Value), VCSrc_v2f16, VCSrc_f32)),
                !if (!eq(VT.Value, i16.Value), VCSrc_b16,
                     !if (!eq(VT.Value, v2i16.Value), VCSrc_v2b16,
                          VCSrc_b32))));
}

// Float or packed int
class isModifierType<ValueType SrcVT> {
  bit ret = !or(!eq(SrcVT.Value, f16.Value),
                !eq(SrcVT.Value, f32.Value),
                !eq(SrcVT.Value, f64.Value),
                !eq(SrcVT.Value, v2f16.Value),
                !eq(SrcVT.Value, v2i16.Value),
                !eq(SrcVT.Value, v2f32.Value),
                !eq(SrcVT.Value, v2i32.Value),
                !eq(SrcVT.Value, v4f16.Value),
                !eq(SrcVT.Value, v4i16.Value),
                !eq(SrcVT.Value, v4f32.Value),
                !eq(SrcVT.Value, v4i32.Value),
                !eq(SrcVT.Value, v8f16.Value),
                !eq(SrcVT.Value, v8i16.Value),
                !eq(SrcVT.Value, v8f32.Value),
                !eq(SrcVT.Value, v8i32.Value),
                !eq(SrcVT.Value, v16f16.Value),
                !eq(SrcVT.Value, v16i16.Value));
}

// Return type of input modifiers operand for specified input operand
class getSrcMod <ValueType VT, bit EnableF32SrcMods> {
  bit isFP = isFloatType<VT>.ret;
  bit isPacked = isPackedType<VT>.ret;
  Operand ret =  !if(!eq(VT.Size, 64),
                     !if(isFP, FP64InputMods, Int64InputMods),
                       !if(isFP,
                         !if(!eq(VT.Value, f16.Value),
                            FP16InputMods,
                            FP32InputMods
                          ),
                         !if(EnableF32SrcMods, FP32InputMods, Int32InputMods))
                     );
}

class getOpSelMod <ValueType VT> {
  Operand ret = !if(!eq(VT.Value, f16.Value), FP16InputMods, IntOpSelMods);
}

// Return type of input modifiers operand specified input operand for DPP
class getSrcModDPP <ValueType VT> {
  bit isFP = isFloatType<VT>.ret;
  Operand ret = !if(isFP, FPVRegInputMods, IntVRegInputMods);
}

// Return type of input modifiers operand for specified input operand for DPP
class getSrcModVOP3DPP <ValueType VT, bit EnableF32SrcMods> {
  bit isFP = isFloatType<VT>.ret;
  bit isPacked = isPackedType<VT>.ret;
  Operand ret =
      !if (isFP,
           !if (!eq(VT.Value, f16.Value), FP16VCSrcInputMods,
                FP32VCSrcInputMods),
           !if (EnableF32SrcMods, FP32VCSrcInputMods, Int32VCSrcInputMods));
}

// Return type of input modifiers operand specified input operand for SDWA
class getSrcModSDWA <ValueType VT> {
  Operand ret = !if(!eq(VT.Value, f16.Value), FP16SDWAInputMods,
                !if(!eq(VT.Value, f32.Value), FP32SDWAInputMods,
                !if(!eq(VT.Value, i16.Value), Int16SDWAInputMods,
                Int32SDWAInputMods)));
}

// Returns the input arguments for VOP[12C] instructions for the given SrcVT.
class getIns32 <RegisterOperand Src0RC, RegisterOperand Src1RC, int NumSrcArgs> {
  dag ret = !if(!eq(NumSrcArgs, 1), (ins Src0RC:$src0),               // VOP1
            !if(!eq(NumSrcArgs, 2), (ins Src0RC:$src0, Src1RC:$src1), // VOP2
                                    (ins)));
}

// Returns the input arguments for VOP3 instructions for the given SrcVT.
class getIns64 <RegisterOperand Src0RC, RegisterOperand Src1RC,
                RegisterOperand Src2RC, int NumSrcArgs,
                bit HasClamp, bit HasModifiers, bit HasSrc2Mods, bit HasOMod,
                Operand Src0Mod, Operand Src1Mod, Operand Src2Mod> {

  dag ret =
    !if (!eq(NumSrcArgs, 0),
      // VOP1 without input operands (V_NOP, V_CLREXCP)
      (ins),
      /* else */
    !if (!eq(NumSrcArgs, 1),
      !if (HasModifiers,
        // VOP1 with modifiers
        !if(HasOMod,
          (ins Src0Mod:$src0_modifiers, Src0RC:$src0,
               clampmod0:$clamp, omod0:$omod),
          (ins Src0Mod:$src0_modifiers, Src0RC:$src0,
               clampmod0:$clamp))
      /* else */,
        // VOP1 without modifiers
        !if (HasClamp,
          (ins Src0RC:$src0, clampmod0:$clamp),
          (ins Src0RC:$src0))
      /* endif */ ),
    !if (!eq(NumSrcArgs, 2),
      !if (HasModifiers,
        // VOP 2 with modifiers
        !if(HasOMod,
          (ins Src0Mod:$src0_modifiers, Src0RC:$src0,
               Src1Mod:$src1_modifiers, Src1RC:$src1,
               clampmod0:$clamp, omod0:$omod),
           (ins Src0Mod:$src0_modifiers, Src0RC:$src0,
               Src1Mod:$src1_modifiers, Src1RC:$src1,
               clampmod0:$clamp))
      /* else */,
        // VOP2 without modifiers
        !if (HasClamp,
          (ins Src0RC:$src0, Src1RC:$src1, clampmod0:$clamp),
          (ins Src0RC:$src0, Src1RC:$src1))

      /* endif */ )
    /* NumSrcArgs == 3 */,
      !if (HasModifiers,
        !if (HasSrc2Mods,
          // VOP3 with modifiers
          !if (HasOMod,
            (ins Src0Mod:$src0_modifiers, Src0RC:$src0,
                 Src1Mod:$src1_modifiers, Src1RC:$src1,
                 Src2Mod:$src2_modifiers, Src2RC:$src2,
                 clampmod0:$clamp, omod0:$omod),
            !if (HasClamp,
              (ins Src0Mod:$src0_modifiers, Src0RC:$src0,
                   Src1Mod:$src1_modifiers, Src1RC:$src1,
                   Src2Mod:$src2_modifiers, Src2RC:$src2,
                   clampmod0:$clamp),
              (ins Src0Mod:$src0_modifiers, Src0RC:$src0,
                   Src1Mod:$src1_modifiers, Src1RC:$src1,
                   Src2Mod:$src2_modifiers, Src2RC:$src2))),
          // VOP3 with modifiers except src2
          !if (HasOMod,
            (ins Src0Mod:$src0_modifiers, Src0RC:$src0,
                 Src1Mod:$src1_modifiers, Src1RC:$src1,
                 Src2RC:$src2, clampmod0:$clamp, omod0:$omod),
            !if (HasClamp,
              (ins Src0Mod:$src0_modifiers, Src0RC:$src0,
                   Src1Mod:$src1_modifiers, Src1RC:$src1,
                   Src2RC:$src2, clampmod0:$clamp),
              (ins Src0Mod:$src0_modifiers, Src0RC:$src0,
                   Src1Mod:$src1_modifiers, Src1RC:$src1,
                   Src2RC:$src2))))
      /* else */,
        // VOP3 without modifiers
        !if (HasClamp,
          (ins Src0RC:$src0, Src1RC:$src1, Src2RC:$src2, clampmod0:$clamp),
          (ins Src0RC:$src0, Src1RC:$src1, Src2RC:$src2))
      /* endif */ ))));
}

class getInsVOP3Base<RegisterOperand Src0RC, RegisterOperand Src1RC,
                RegisterOperand Src2RC, int NumSrcArgs,
                bit HasClamp, bit HasModifiers, bit HasSrc2Mods, bit HasOMod,
                Operand Src0Mod, Operand Src1Mod, Operand Src2Mod, bit HasOpSel,
                bit IsVOP3P> {
  // getInst64 handles clamp and omod. implicit mutex between vop3p and omod
  dag base = getIns64 <Src0RC, Src1RC, Src2RC, NumSrcArgs,
                HasClamp, HasModifiers, HasSrc2Mods, HasOMod,
                Src0Mod, Src1Mod, Src2Mod>.ret;
  dag opsel = (ins op_sel0:$op_sel);
  dag vop3pOpsel = (ins op_sel_hi0:$op_sel_hi);
  dag vop3pFields = !con(!if(HasOpSel, vop3pOpsel, (ins)), (ins neg_lo0:$neg_lo, neg_hi0:$neg_hi));

  dag ret = !con(base,
                 !if(HasOpSel, opsel,(ins)),
                 !if(IsVOP3P, vop3pFields,(ins)));
}

class getInsVOP3P <RegisterOperand Src0RC, RegisterOperand Src1RC,
                   RegisterOperand Src2RC, int NumSrcArgs, bit HasClamp, bit HasOpSel,
                   Operand Src0Mod, Operand Src1Mod, Operand Src2Mod> {
  dag ret = getInsVOP3Base<Src0RC, Src1RC, Src2RC, NumSrcArgs,
                    HasClamp, 1/*HasModifiers*/, 1/*HasSrc2Mods*/,
                    0/*HasOMod*/, Src0Mod, Src1Mod, Src2Mod,
                    HasOpSel, 1/*IsVOP3P*/>.ret;
}

class getInsVOP3OpSel <RegisterOperand Src0RC, RegisterOperand Src1RC,
                       RegisterOperand Src2RC, int NumSrcArgs,
                       bit HasClamp, bit HasOMod,
                       Operand Src0Mod, Operand Src1Mod, Operand Src2Mod> {
  dag ret = getInsVOP3Base<Src0RC, Src1RC,
                    Src2RC, NumSrcArgs,
                    HasClamp, 1/*HasModifiers*/, 1/*HasSrc2Mods*/, HasOMod,
                    Src0Mod, Src1Mod, Src2Mod, 1/*HasOpSel*/, 0>.ret;
}

class getInsDPPBase <RegisterOperand OldRC, RegisterClass Src0RC, RegisterClass Src1RC,
                 RegisterClass Src2RC, int NumSrcArgs, bit HasModifiers,
                 Operand Src0Mod, Operand Src1Mod, Operand Src2Mod> {

  dag ret = !if (!eq(NumSrcArgs, 0),
                // VOP1 without input operands (V_NOP)
                (ins ),
            !if (!eq(NumSrcArgs, 1),
              !if (HasModifiers,
                // VOP1_DPP with modifiers
                (ins OldRC:$old, Src0Mod:$src0_modifiers,
                     Src0RC:$src0)
              /* else */,
                // VOP1_DPP without modifiers
                (ins OldRC:$old, Src0RC:$src0)
              /* endif */),
            !if (!eq(NumSrcArgs, 2),
              !if (HasModifiers,
                // VOP2_DPP with modifiers
                (ins OldRC:$old,
                     Src0Mod:$src0_modifiers, Src0RC:$src0,
                     Src1Mod:$src1_modifiers, Src1RC:$src1)
              /* else */,
                // VOP2_DPP without modifiers
                (ins OldRC:$old,
                     Src0RC:$src0, Src1RC:$src1)
                )
              /* NumSrcArgs == 3, VOP3 */,
              !if (HasModifiers,
                // VOP3_DPP with modifiers
                (ins OldRC:$old,
                     Src0Mod:$src0_modifiers, Src0RC:$src0,
                     Src1Mod:$src1_modifiers, Src1RC:$src1,
                     Src2Mod:$src2_modifiers, Src2RC:$src2)
              /* else */,
                // VOP3_DPP without modifiers
                (ins OldRC:$old,
                     Src0RC:$src0, Src1RC:$src1,
                     Src2RC:$src2)
                )
             /* endif */)));
}

class getInsDPP <RegisterOperand OldRC, RegisterClass Src0RC, RegisterClass Src1RC,
                 RegisterClass Src2RC, int NumSrcArgs, bit HasModifiers,
                 Operand Src0Mod, Operand Src1Mod, Operand Src2Mod> {
  dag ret = !con(getInsDPPBase<OldRC, Src0RC, Src1RC, Src2RC, NumSrcArgs,
                           HasModifiers, Src0Mod, Src1Mod, Src2Mod>.ret,
                 (ins dpp_ctrl:$dpp_ctrl, row_mask:$row_mask,
                     bank_mask:$bank_mask, bound_ctrl:$bound_ctrl));
}

class getInsDPP16 <RegisterOperand OldRC, RegisterClass Src0RC, RegisterClass Src1RC,
                 RegisterClass Src2RC, int NumSrcArgs, bit HasModifiers,
                 Operand Src0Mod, Operand Src1Mod, Operand Src2Mod> {
  dag ret = !con(getInsDPP<OldRC, Src0RC, Src1RC, Src2RC, NumSrcArgs,
                           HasModifiers, Src0Mod, Src1Mod, Src2Mod>.ret,
                 (ins FI:$fi));
}

class getInsDPP8 <RegisterOperand OldRC, RegisterClass Src0RC, RegisterClass Src1RC,
                 RegisterClass Src2RC, int NumSrcArgs, bit HasModifiers,
                 Operand Src0Mod, Operand Src1Mod, Operand Src2Mod> {
  dag ret = !con(getInsDPPBase<OldRC, Src0RC, Src1RC, Src2RC, NumSrcArgs,
                           HasModifiers, Src0Mod, Src1Mod, Src2Mod>.ret,
                 (ins dpp8:$dpp8, FI:$fi));
}

class getInsVOP3DPPBase<dag VOP3Base, RegisterOperand OldRC, int NumSrcArgs> {
  dag old = ( ins OldRC:$old );
  dag base = VOP3Base;
  dag ret =  !con(
                !if(!ne(NumSrcArgs, 0), old, (ins)),
                base
              );
}

class getInsVOP3DPP<dag VOP3Base, RegisterOperand OldRC, int NumSrcArgs> {
  dag ret = !con(getInsVOP3DPPBase<VOP3Base,OldRC,NumSrcArgs>.ret,
                 (ins dpp_ctrl:$dpp_ctrl, row_mask:$row_mask,
                     bank_mask:$bank_mask, bound_ctrl:$bound_ctrl));
}

class getInsVOP3DPP16<dag VOP3Base, RegisterOperand OldRC, int NumSrcArgs> {
  dag ret = !con(getInsVOP3DPP<VOP3Base,OldRC,NumSrcArgs>.ret,
                 (ins FI:$fi));
}

class getInsVOP3DPP8<dag VOP3Base, RegisterOperand OldRC, int NumSrcArgs> {
  dag ret = !con(getInsVOP3DPPBase<VOP3Base,OldRC,NumSrcArgs>.ret,
                 (ins dpp8:$dpp8, FI:$fi));
}

// Ins for SDWA
class getInsSDWA <RegisterOperand Src0RC, RegisterOperand Src1RC, int NumSrcArgs,
                  bit HasSDWAOMod, Operand Src0Mod, Operand Src1Mod,
                  ValueType DstVT> {

  dag ret = !if(!eq(NumSrcArgs, 0),
               // VOP1 without input operands (V_NOP)
               (ins),
            !if(!eq(NumSrcArgs, 1),
               // VOP1
               !if(!not(HasSDWAOMod),
                  // VOP1_SDWA without omod
                  (ins Src0Mod:$src0_modifiers, Src0RC:$src0,
                       clampmod:$clamp,
                       dst_sel:$dst_sel, dst_unused:$dst_unused,
                       src0_sel:$src0_sel),
                  // VOP1_SDWA with omod
                  (ins Src0Mod:$src0_modifiers, Src0RC:$src0,
                       clampmod:$clamp, omod:$omod,
                       dst_sel:$dst_sel, dst_unused:$dst_unused,
                       src0_sel:$src0_sel)),
            !if(!eq(NumSrcArgs, 2),
               !if(!eq(DstVT.Size, 1),
                  // VOPC_SDWA
                  (ins Src0Mod:$src0_modifiers, Src0RC:$src0,
                       Src1Mod:$src1_modifiers, Src1RC:$src1,
                       clampmod:$clamp, src0_sel:$src0_sel, src1_sel:$src1_sel),
                  // VOP2_SDWA
                  !if(!not(HasSDWAOMod),
                     // VOP2_SDWA without omod
                     (ins Src0Mod:$src0_modifiers, Src0RC:$src0,
                          Src1Mod:$src1_modifiers, Src1RC:$src1,
                          clampmod:$clamp,
                          dst_sel:$dst_sel, dst_unused:$dst_unused,
                          src0_sel:$src0_sel, src1_sel:$src1_sel),
                     // VOP2_SDWA with omod
                     (ins Src0Mod:$src0_modifiers, Src0RC:$src0,
                          Src1Mod:$src1_modifiers, Src1RC:$src1,
                          clampmod:$clamp, omod:$omod,
                          dst_sel:$dst_sel, dst_unused:$dst_unused,
                          src0_sel:$src0_sel, src1_sel:$src1_sel))),
            (ins)/* endif */)));
}

// Outs for DPP
class getOutsDPP <bit HasDst, ValueType DstVT, RegisterOperand DstRCDPP> {
  dag ret = !if(HasDst,
                !if(!eq(DstVT.Size, 1),
                    (outs), // no dst for VOPC, we use "vcc"-token as dst in SDWA VOPC instructions
                    (outs DstRCDPP:$vdst)),
                (outs)); // V_NOP
}

// Outs for SDWA
class getOutsSDWA <bit HasDst, ValueType DstVT, RegisterOperand DstRCSDWA> {
  dag ret = !if(HasDst,
                !if(!eq(DstVT.Size, 1),
                    (outs DstRCSDWA:$sdst),
                    (outs DstRCSDWA:$vdst)),
                (outs)); // V_NOP
}

// Returns the assembly string for the inputs and outputs of a VOP[12C]
// instruction.  This does not add the _e32 suffix, so it can be reused
// by getAsm64.
class getAsm32 <bit HasDst, int NumSrcArgs, ValueType DstVT = i32> {
  string dst = !if(!eq(DstVT.Size, 1), "$sdst", "$vdst"); // use $sdst for VOPC
  string src0 = ", $src0";
  string src1 = ", $src1";
  string src2 = ", $src2";
  string ret = !if(HasDst, dst, "") #
               !if(!eq(NumSrcArgs, 1), src0, "") #
               !if(!eq(NumSrcArgs, 2), src0#src1, "") #
               !if(!eq(NumSrcArgs, 3), src0#src1#src2, "");
}

class getAsmVOPDPart <int NumSrcArgs, string XorY> {
  string dst = "$vdst" # XorY;
  string src0 = ", $src0" # XorY;
  string src1 = ", $vsrc1" # XorY;
  string ret = dst #
               !if(!ge(NumSrcArgs, 1), src0, "") #
               !if(!ge(NumSrcArgs, 2), src1, "");
}

// Returns the assembly string for the inputs and outputs of a VOP3
// instruction.
class getAsm64 <bit HasDst, int NumSrcArgs, bit HasIntClamp, bit HasModifiers,
                bit HasOMod, ValueType DstVT = i32> {
  string dst = !if(!eq(DstVT.Size, 1), "$sdst", "$vdst"); // use $sdst for VOPC
  string src0 = !if(!eq(NumSrcArgs, 1), "$src0_modifiers", "$src0_modifiers,");
  string src1 = !if(!eq(NumSrcArgs, 1), "",
                   !if(!eq(NumSrcArgs, 2), " $src1_modifiers",
                                           " $src1_modifiers,"));
  string src2 = !if(!eq(NumSrcArgs, 3), " $src2_modifiers", "");
  string iclamp = !if(HasIntClamp, "$clamp", "");
  string ret =
  !if(!not(HasModifiers),
      getAsm32<HasDst, NumSrcArgs, DstVT>.ret # iclamp,
      dst#", "#src0#src1#src2#"$clamp"#!if(HasOMod, "$omod", ""));
}

// Returns the assembly string for the inputs and outputs of a VOP3P
// instruction.
class getAsmVOP3P <int NumSrcArgs, bit HasModifiers,
                   bit HasClamp, bit HasOpSel> {
  string dst = "$vdst";
  string src0 = !if(!eq(NumSrcArgs, 1), "$src0", "$src0,");
  string src1 = !if(!eq(NumSrcArgs, 1), "",
                   !if(!eq(NumSrcArgs, 2), " $src1",
                                           " $src1,"));
  string src2 = !if(!eq(NumSrcArgs, 3), " $src2", "");

  string mods = !if(HasModifiers, "$neg_lo$neg_hi", "");
  string clamp = !if(HasClamp, "$clamp", "");
  string opsel = !if(HasOpSel, "$op_sel$op_sel_hi", "");

  // Each modifier is printed as an array of bits for each operand, so
  // all operands are printed as part of src0_modifiers.
  string ret = dst#", "#src0#src1#src2#opsel#mods#clamp;
}

class getAsmVOP3OpSel <int NumSrcArgs,
                       bit HasClamp,
                       bit Src0HasMods,
                       bit Src1HasMods,
                       bit Src2HasMods> {
  string dst = "$vdst";

  string isrc0 = !if(!eq(NumSrcArgs, 1), "$src0", "$src0,");
  string isrc1 = !if(!eq(NumSrcArgs, 1), "",
                     !if(!eq(NumSrcArgs, 2), " $src1",
                                             " $src1,"));
  string isrc2 = !if(!eq(NumSrcArgs, 3), " $src2", "");

  string fsrc0 = !if(!eq(NumSrcArgs, 1), "$src0_modifiers", "$src0_modifiers,");
  string fsrc1 = !if(!eq(NumSrcArgs, 1), "",
                     !if(!eq(NumSrcArgs, 2), " $src1_modifiers",
                                             " $src1_modifiers,"));
  string fsrc2 = !if(!eq(NumSrcArgs, 3), " $src2_modifiers", "");

  string src0 = !if(Src0HasMods, fsrc0, isrc0);
  string src1 = !if(Src1HasMods, fsrc1, isrc1);
  string src2 = !if(Src2HasMods, fsrc2, isrc2);

  string clamp = !if(HasClamp, "$clamp", "");
  // TODO-GFX11 enable here when HasOMod with opsel is enabled
  string omod = "";
  /*string omod =  !if(HasOMod, "$omod", "");*/
  string ret = dst#", "#src0#src1#src2#"$op_sel"#clamp#omod;
}

class getAsmDPP <bit HasDst, int NumSrcArgs, bit HasModifiers, ValueType DstVT = i32> {
  string dst = !if(HasDst,
                   !if(!eq(DstVT.Size, 1),
                       "$sdst",
                       "$vdst"),
                    ""); // use $sdst for VOPC
  string src0 = !if(!eq(NumSrcArgs, 1), "$src0_modifiers", "$src0_modifiers,");
  string src1 = !if(!eq(NumSrcArgs, 1), "",
                   !if(!eq(NumSrcArgs, 2), " $src1_modifiers",
                                           " $src1_modifiers,"));
  string args = !if(!not(HasModifiers),
                     getAsm32<0, NumSrcArgs, DstVT>.ret,
                     ", "#src0#src1);
  string ret = dst#args#" $dpp_ctrl$row_mask$bank_mask$bound_ctrl";
}

class getAsmDPP16 <bit HasDst, int NumSrcArgs, bit HasModifiers, ValueType DstVT = i32> {
  string ret = getAsmDPP<HasDst, NumSrcArgs, HasModifiers, DstVT>.ret#"$fi";
}

class getAsmDPP8 <bit HasDst, int NumSrcArgs, bit HasModifiers, ValueType DstVT = i32>
  : getAsmDPP<HasDst, NumSrcArgs, HasModifiers, DstVT>{
  let ret = dst#args#" $dpp8$fi";
}

class getAsmVOP3DPPBase <int NumSrcArgs, bit HasDst, bit HasClamp,
                       bit HasOpSel, bit HasOMod, bit IsVOP3P,
                       bit HasModifiers, bit Src0HasMods,
                       bit Src1HasMods, bit Src2HasMods, ValueType DstVT = i32> {
  string dst = !if(HasDst,
                   !if(!eq(DstVT.Size, 1),
                       "$sdst",
                       "$vdst"),
                    ""); // use $sdst for VOPC
  string src0nomods = !if(!eq(NumSrcArgs, 1), "$src0", "$src0,");
  string src1nomods = !if(!eq(NumSrcArgs, 1), "",
                    !if(!eq(NumSrcArgs, 2), " $src1",
                                            " $src1,"));
  string src2nomods = !if(!eq(NumSrcArgs, 3), " $src2", "");

  string src0mods = !if(!eq(NumSrcArgs, 1), "$src0_modifiers", "$src0_modifiers,");
  string src1mods = !if(!eq(NumSrcArgs, 1), "",
                    !if(!eq(NumSrcArgs, 2), " $src1_modifiers",
                                            " $src1_modifiers,"));
  string src2mods = !if(!eq(NumSrcArgs, 3), " $src2_modifiers", "");

  string src0 = !if(Src0HasMods, src0mods, src0nomods);
  string src1 = !if(Src1HasMods, src1mods, src1nomods);
  string src2 = !if(Src2HasMods, src2mods, src2nomods);
  string opsel = !if(HasOpSel, "$op_sel", "");
  string 3PMods = !if(IsVOP3P,
                      !if(HasOpSel, "$op_sel_hi", "")
                        #!if(HasModifiers, "$neg_lo$neg_hi", ""),
                      "");
  string clamp = !if(HasClamp, "$clamp", "");
  string omod = !if(HasOMod, "$omod", "");

  string ret = dst#", "#src0#src1#src2#opsel#3PMods#clamp#omod;

}

class getAsmVOP3DPP<string base> {
  string ret = base # " $dpp_ctrl$row_mask$bank_mask$bound_ctrl";
}

class getAsmVOP3DPP16<string base> {
  string ret = getAsmVOP3DPP<base>.ret # "$fi";
}

class getAsmVOP3DPP8<string base> {
  string ret = base # " $dpp8$fi";
}


class getAsmSDWA <bit HasDst, int NumSrcArgs, ValueType DstVT = i32> {
  string dst = !if(HasDst,
                   !if(!eq(DstVT.Size, 1),
                       " vcc", // use vcc token as dst for VOPC instructions
                       "$vdst"),
                    "");
  string src0 = "$src0_modifiers";
  string src1 = "$src1_modifiers";
  string args = !if(!eq(NumSrcArgs, 0),
                    "",
                    !if(!eq(NumSrcArgs, 1),
                        ", "#src0#"$clamp",
                        ", "#src0#", "#src1#"$clamp"
                     )
                );
  string sdwa = !if(!eq(NumSrcArgs, 0),
                    "",
                    !if(!eq(NumSrcArgs, 1),
                        " $dst_sel $dst_unused $src0_sel",
                        !if(!eq(DstVT.Size, 1),
                            " $src0_sel $src1_sel", // No dst_sel and dst_unused for VOPC
                            " $dst_sel $dst_unused $src0_sel $src1_sel"
                        )
                    )
                );
  string ret = dst#args#sdwa;
}

class getAsmSDWA9 <bit HasDst, bit HasOMod, int NumSrcArgs,
                   ValueType DstVT = i32> {
  string dst = !if(HasDst,
                   !if(!eq(DstVT.Size, 1),
                       "$sdst", // VOPC
                       "$vdst"), // VOP1/2
                    "");
  string src0 = "$src0_modifiers";
  string src1 = "$src1_modifiers";
  string out_mods = !if(!not(HasOMod), "$clamp", "$clamp$omod");
  string args = !if(!eq(NumSrcArgs, 0), "",
                    !if(!eq(NumSrcArgs, 1),
                        ", "#src0,
                        ", "#src0#", "#src1
                     )
                );
  string sdwa = !if(!eq(NumSrcArgs, 0), "",
                    !if(!eq(NumSrcArgs, 1),
                        out_mods#" $dst_sel $dst_unused $src0_sel",
                        !if(!eq(DstVT.Size, 1),
                            " $src0_sel $src1_sel", // No dst_sel, dst_unused and output modifiers for VOPC
                            out_mods#" $dst_sel $dst_unused $src0_sel $src1_sel"
                        )
                    )
                );
  string ret = dst#args#sdwa;
}

class getHas64BitOps <int NumSrcArgs, ValueType DstVT, ValueType Src0VT,
                      ValueType Src1VT> {
  bit ret = !if(!eq(NumSrcArgs, 3),
                0,
                !if(!eq(DstVT.Size, 64),
                    1,
                    !if(!eq(Src0VT.Size, 64),
                        1,
                        !if(!eq(Src1VT.Size, 64),
                            1,
                            0
                        )
                    )
                )
            );
}

class getHasSDWA <int NumSrcArgs, ValueType DstVT = i32, ValueType Src0VT = i32,
                  ValueType Src1VT = i32> {
  bit ret = !if(!eq(NumSrcArgs, 3),
                0, // NumSrcArgs == 3 - No SDWA for VOP3 before GFX11
                !if(!eq(DstVT.Size, 64),
                    0, // 64-bit dst - No SDWA for 64-bit operands
                    !if(!eq(Src0VT.Size, 64),
                        0, // 64-bit src0
                        !if(!eq(Src1VT.Size, 64),
                            0, // 64-bit src2
                            1
                        )
                    )
                )
            );
}

// Used when HasModifiers is fixed
class getHas16BitOperand <ValueType DstVT = i32, ValueType Src0VT = i32,
                 ValueType Src1VT = i32, ValueType Src2VT = i32> {
  bit ret = !if(!or(is16BitType<Src0VT>.ret,
                    is16BitType<Src1VT>.ret,
                    is16BitType<Src2VT>.ret,
                    is16BitType<DstVT>.ret),
                1, 0);
}

class getHasDPP <int NumSrcArgs> {
  bit ret = !if(!eq(NumSrcArgs, 3),
                0, // NumSrcArgs == 3 - No DPP for VOP3
                1);
}

class getHasExt32BitDPP <int NumSrcArgs, ValueType DstVT = i32, ValueType Src0VT = i32,
                 ValueType Src1VT = i32> {
  bit ret = !and(getHasDPP<NumSrcArgs>.ret,
                 !not(getHas64BitOps<NumSrcArgs, DstVT, Src0VT, Src1VT>.ret));
}

class getHasExt64BitDPP <int NumSrcArgs, ValueType DstVT = i32, ValueType Src0VT = i32,
                 ValueType Src1VT = i32> {
  bit ret = !and(getHasDPP<NumSrcArgs>.ret,
                 getHas64BitOps<NumSrcArgs, DstVT, Src0VT, Src1VT>.ret);
}

// Function that checks if instruction supports DPP and SDWA
class getHasExt <int NumSrcArgs, ValueType DstVT = i32, ValueType Src0VT = i32,
                 ValueType Src1VT = i32> {
  bit ret = !or(getHasDPP<NumSrcArgs>.ret,
                getHasSDWA<NumSrcArgs, DstVT, Src0VT, Src1VT>.ret);
}

// Return an AGPR+VGPR operand class for the given VGPR register class.
class getLdStRegisterOperand<RegisterClass RC> {
  RegisterOperand ret =
    !if(!eq(RC.Size, 32), AVLdSt_32,
      !if(!eq(RC.Size, 64), AVLdSt_64,
        !if(!eq(RC.Size, 96), AVLdSt_96,
          !if(!eq(RC.Size, 128), AVLdSt_128,
            !if(!eq(RC.Size, 160), AVLdSt_160,
              RegisterOperand<VReg_1> // invalid register
    )))));
}

class BitOr<bit a, bit b> {
  bit ret = !if(a, 1, !if(b, 1, 0));
}

class BitAnd<bit a, bit b> {
  bit ret = !if(a, !if(b, 1, 0), 0);
}

class getHasVOP3DPP <ValueType DstVT = i32, ValueType Src0VT = i32,
                 ValueType Src1VT = i32, ValueType Src2VT = i32> {
  bit ret =    !if(!eq(DstVT.Size, 64),
                    0, // 64-bit dst No DPP for 64-bit operands
                    !if(!eq(Src0VT.Size, 64),
                        0, // 64-bit src0
                        !if(!eq(Src1VT.Size, 64),
                            0, // 64-bit src1
                            !if(!eq(Src2VT.Size, 64),
                                0, // 64-bit src2
                                1
                            )
                        )
                    )
                );
}


def PatGenMode {
  int NoPattern = 0;
  int Pattern   = 1;
}

class VOPProfile <list<ValueType> _ArgVT, bit _EnableF32SrcMods = 0,
                  bit _EnableClamp = 0> {

  field list<ValueType> ArgVT = _ArgVT;
  field bit EnableF32SrcMods = _EnableF32SrcMods;
  field bit EnableClamp = _EnableClamp;

  field ValueType DstVT = ArgVT[0];
  field ValueType Src0VT = ArgVT[1];
  field ValueType Src1VT = ArgVT[2];
  field ValueType Src2VT = ArgVT[3];
  field RegisterOperand DstRC = getVALUDstForVT<DstVT>.ret;
  field RegisterOperand DstRC64 = DstRC;
  field RegisterOperand DstRCDPP = getVALUDstForVT<DstVT>.ret;
  field RegisterOperand DstRCSDWA = getSDWADstForVT<DstVT>.ret;
  field RegisterOperand Src0RC32 = getVOPSrc0ForVT<Src0VT>.ret;
  field RegisterOperand Src1RC32 = RegisterOperand<getVregSrcForVT<Src1VT>.ret>;
  field RegisterOperand Src0RC64 = getVOP3SrcForVT<Src0VT>.ret;
  field RegisterOperand Src1RC64 = getVOP3SrcForVT<Src1VT>.ret;
  field RegisterOperand Src2RC64 = getVOP3SrcForVT<Src2VT>.ret;
  field RegisterClass Src0DPP = getVregSrcForVT<Src0VT>.ret;
  field RegisterClass Src1DPP = getVregSrcForVT<Src1VT>.ret;
  field RegisterClass Src2DPP = getVregSrcForVT<Src2VT>.ret;
  // TODO-GFX11 Choose Src{0,1}VOP3DPP operand class when real 16bit regs exist
  field RegisterOperand Src0VOP3DPP = VGPRSrc_32;
  field RegisterOperand Src1VOP3DPP = VGPRSrc_32;
  field RegisterOperand Src2VOP3DPP = getVOP3DPPSrcForVT<Src2VT>.ret;
  field RegisterOperand Src0SDWA = getSDWASrcForVT<Src0VT>.ret;
  field RegisterOperand Src1SDWA = getSDWASrcForVT<Src0VT>.ret;
  field Operand Src0Mod = getSrcMod<Src0VT, EnableF32SrcMods>.ret;
  field Operand Src1Mod = getSrcMod<Src1VT, EnableF32SrcMods>.ret;
  field Operand Src2Mod = getSrcMod<Src2VT, EnableF32SrcMods>.ret;
  field Operand Src0ModDPP = getSrcModDPP<Src0VT>.ret;
  field Operand Src1ModDPP = getSrcModDPP<Src1VT>.ret;
  field Operand Src2ModDPP = getSrcModDPP<Src2VT>.ret;
  field Operand Src2ModVOP3DPP = getSrcModVOP3DPP<Src2VT, EnableF32SrcMods>.ret;
  field Operand Src0ModSDWA = getSrcModSDWA<Src0VT>.ret;
  field Operand Src1ModSDWA = getSrcModSDWA<Src1VT>.ret;


  field bit HasDst = !ne(DstVT.Value, untyped.Value);
  field bit HasDst32 = HasDst;
  field bit EmitDst = HasDst; // force dst encoding, see v_movreld_b32 special case
  field bit EmitDstSel = EmitDst;
  field int NumSrcArgs = getNumSrcArgs<Src0VT, Src1VT, Src2VT>.ret;
  field bit HasSrc0 = !ne(Src0VT.Value, untyped.Value);
  field bit HasSrc1 = !ne(Src1VT.Value, untyped.Value);
  field bit HasSrc2 = !ne(Src2VT.Value, untyped.Value);

  // HasSrc*FloatMods affects the SDWA encoding. We ignore EnableF32SrcMods.
  field bit HasSrc0FloatMods = isFloatType<Src0VT>.ret;
  field bit HasSrc1FloatMods = isFloatType<Src1VT>.ret;
  field bit HasSrc2FloatMods = isFloatType<Src2VT>.ret;

  // HasSrc*IntMods affects the SDWA encoding. We ignore EnableF32SrcMods.
  field bit HasSrc0IntMods = isIntType<Src0VT>.ret;
  field bit HasSrc1IntMods = isIntType<Src1VT>.ret;
  field bit HasSrc2IntMods = isIntType<Src2VT>.ret;

  field bit HasClamp = !or(isModifierType<Src0VT>.ret, EnableClamp);
  field bit HasSDWAClamp = EmitDst;
  field bit HasFPClamp = !and(isFloatType<DstVT>.ret, HasClamp);
  field bit HasIntClamp = !if(isFloatType<DstVT>.ret, 0, HasClamp);
  field bit HasClampLo = HasClamp;
  field bit HasClampHi = !and(isPackedType<DstVT>.ret, HasClamp);
  field bit HasHigh = 0;

  field bit IsPacked = isPackedType<Src0VT>.ret;
  // TODO-GFX11 likely should wait till enabling 16 bit registers
  /*field bit HasOpSel = !if(!or(getHas16BitOperand<DstVT, Src0VT, Src1VT, */
                       /*Src2VT>.ret, IsPacked), 1, 0);*/
  field bit HasOpSel = IsPacked;
  // TODO-GFX11 opsel + omod should be allowed
  field bit HasOMod = !if(HasOpSel, 0, isFloatType<DstVT>.ret);
  field bit HasSDWAOMod = isFloatType<DstVT>.ret;

  field bit HasModifiers = !or(isModifierType<Src0VT>.ret,
                               isModifierType<Src1VT>.ret,
                               isModifierType<Src2VT>.ret,
                               HasOMod,
                               EnableF32SrcMods);

  field bit HasSrc0Mods = HasModifiers;
  field bit HasSrc1Mods = !if(HasModifiers, !or(HasSrc1FloatMods, HasSrc1IntMods), 0);
  field bit HasSrc2Mods = !if(HasModifiers, !or(HasSrc2FloatMods, HasSrc2IntMods), 0);

  field bit HasExt = getHasExt<NumSrcArgs, DstVT, Src0VT, Src1VT>.ret;
  field bit HasExtVOP3DPP = getHasVOP3DPP<DstVT, Src0VT, Src1VT, Src2VT>.ret;
  field bit HasExtDPP = !if(!or(getHasDPP<NumSrcArgs>.ret,
                HasExtVOP3DPP), 1, 0);
  field bit HasExt32BitDPP = getHasExt32BitDPP<NumSrcArgs, DstVT, Src0VT, Src1VT>.ret;
  field bit HasExt64BitDPP = getHasExt64BitDPP<NumSrcArgs, DstVT, Src0VT, Src1VT>.ret;
  field bit HasExtSDWA = getHasSDWA<NumSrcArgs, DstVT, Src0VT, Src1VT>.ret;
  field bit HasExtSDWA9 = HasExtSDWA;
  field int NeedPatGen = PatGenMode.NoPattern;

  field bit IsMAI = 0;
  field bit IsVOP3P = 0;
  field bit IsDOT = 0;
  field bit IsSingle = 0;
  field bit IsWMMA = 0;

  field Operand Src0PackedMod = !if(HasSrc0FloatMods, PackedF16InputMods, PackedI16InputMods);
  field Operand Src1PackedMod = !if(HasSrc1FloatMods, PackedF16InputMods, PackedI16InputMods);
  field Operand Src2PackedMod = !if(HasSrc2FloatMods, PackedF16InputMods, PackedI16InputMods);

  field dag Outs = !if(HasDst,(outs DstRC:$vdst),(outs));

  // VOP3b instructions are a special case with a second explicit
  // output. This is manually overridden for them.
  field dag Outs32 = Outs;
  field dag Outs64 = !if(HasDst,(outs DstRC64:$vdst),(outs));
  field dag OutsDPP = getOutsDPP<HasDst, DstVT, DstRCDPP>.ret;
  field dag OutsDPP8 = getOutsDPP<HasDst, DstVT, DstRCDPP>.ret;
  field dag OutsVOP3DPP = OutsDPP;
  field dag OutsVOP3DPP8 = OutsDPP8;
  field dag OutsSDWA = getOutsSDWA<HasDst, DstVT, DstRCSDWA>.ret;

  field dag Ins32 = getIns32<Src0RC32, Src1RC32, NumSrcArgs>.ret;
  field dag Ins64 = getIns64<Src0RC64, Src1RC64, Src2RC64, NumSrcArgs,
                             HasIntClamp, HasModifiers, HasSrc2Mods,
                             HasOMod, Src0Mod, Src1Mod, Src2Mod>.ret;
  field dag InsVOP3P = getInsVOP3P<Src0RC64, Src1RC64, Src2RC64,
                                   NumSrcArgs, HasClamp, HasOpSel,
                                   Src0PackedMod, Src1PackedMod, Src2PackedMod>.ret;
  field dag InsVOP3OpSel = getInsVOP3OpSel<Src0RC64, Src1RC64, Src2RC64,
                                NumSrcArgs, HasClamp, HasOMod,
                                getOpSelMod<Src0VT>.ret,
                                getOpSelMod<Src1VT>.ret,
                                getOpSelMod<Src2VT>.ret>.ret;
  field dag InsDPP = !if(HasExtDPP,
                         getInsDPP<DstRCDPP, Src0DPP, Src1DPP, Src2DPP, NumSrcArgs,
                                   HasModifiers, Src0ModDPP, Src1ModDPP, Src2ModDPP>.ret,
                         (ins));
  field dag InsDPP16 = getInsDPP16<DstRCDPP, Src0DPP, Src1DPP, Src2DPP, NumSrcArgs,
                                   HasModifiers, Src0ModDPP, Src1ModDPP, Src2ModDPP>.ret;
  field dag InsDPP8 = getInsDPP8<DstRCDPP, Src0DPP, Src1DPP, Src2DPP,
                                 NumSrcArgs, HasModifiers,
                                 Src0ModDPP, Src1ModDPP, Src2ModDPP>.ret;
  field dag InsVOP3Base = getInsVOP3Base<Src0VOP3DPP, Src1VOP3DPP,
                  Src2VOP3DPP, NumSrcArgs, HasClamp, HasModifiers, HasSrc2Mods, HasOMod,
                  Src0ModDPP, Src1ModDPP, Src2ModVOP3DPP, HasOpSel, IsVOP3P>.ret;
  field dag InsVOP3DPP = getInsVOP3DPP<InsVOP3Base, DstRCDPP, NumSrcArgs>.ret;
  field dag InsVOP3DPP16 = getInsVOP3DPP16<InsVOP3Base, DstRCDPP, NumSrcArgs>.ret;
  field dag InsVOP3DPP8 = getInsVOP3DPP8<InsVOP3Base, DstRCDPP, NumSrcArgs>.ret;
  field dag InsSDWA = getInsSDWA<Src0SDWA, Src1SDWA, NumSrcArgs,
                                 HasSDWAOMod, Src0ModSDWA, Src1ModSDWA,
                                 DstVT>.ret;
  field dag InsVOPDX = (ins Src0RC32:$src0X, Src1RC32:$vsrc1X);
  // It is a slight misnomer to use the deferred f32 operand type for non-float
  // operands, but this operand type will only be used if the other dual
  // component is FMAAK or FMAMK
  field dag InsVOPDXDeferred = (ins !if(!eq(Src0VT.Size, 32), VSrc_f32_Deferred, VSrc_f16_Deferred):$src0X, VGPR_32:$vsrc1X);
  field dag InsVOPDY = (ins Src0RC32:$src0Y, Src1RC32:$vsrc1Y);
  field dag InsVOPDYDeferred = (ins !if(!eq(Src1VT.Size, 32), VSrc_f32_Deferred, VSrc_f16_Deferred):$src0Y, VGPR_32:$vsrc1Y);

  field string Asm32 = getAsm32<HasDst, NumSrcArgs, DstVT>.ret;
  field string Asm64 = getAsm64<HasDst, NumSrcArgs, HasIntClamp, HasModifiers, HasOMod, DstVT>.ret;
  field string AsmVOP3P = getAsmVOP3P<NumSrcArgs, HasModifiers, HasClamp, HasOpSel>.ret;
  field string AsmVOP3OpSel = getAsmVOP3OpSel<NumSrcArgs,
                                              HasClamp,
                                              HasSrc0FloatMods,
                                              HasSrc1FloatMods,
                                              HasSrc2FloatMods>.ret;
  field string AsmDPP = !if(HasExtDPP,
                            getAsmDPP<HasDst, NumSrcArgs, HasModifiers, DstVT>.ret, "");
  field string AsmDPP16 = getAsmDPP16<HasDst, NumSrcArgs, HasModifiers, DstVT>.ret;
  // DPP8 encoding has no fields for modifiers, and it is enforced by setting
  // the asm operand name via this HasModifiers flag
  field string AsmDPP8 = getAsmDPP8<HasDst, NumSrcArgs, 0 /*HasModifiers*/, DstVT>.ret;
  field string AsmVOP3DPPBase = getAsmVOP3DPPBase<NumSrcArgs, HasDst, HasClamp,
   HasOpSel, HasOMod, IsVOP3P, HasModifiers, HasModifiers, HasModifiers,
   HasModifiers, DstVT>.ret;
  field string AsmVOP3DPP = getAsmVOP3DPP<AsmVOP3DPPBase>.ret;
  field string AsmVOP3DPP16 = getAsmVOP3DPP16<AsmVOP3DPPBase>.ret;
  field string AsmVOP3DPP8 = getAsmVOP3DPP8<AsmVOP3DPPBase>.ret;
  field string AsmSDWA = getAsmSDWA<HasDst, NumSrcArgs, DstVT>.ret;
  field string AsmSDWA9 = getAsmSDWA9<HasDst, HasSDWAOMod, NumSrcArgs, DstVT>.ret;
  field string AsmVOPDX = getAsmVOPDPart<NumSrcArgs, "X">.ret;
  field string AsmVOPDY = getAsmVOPDPart<NumSrcArgs, "Y">.ret;
  field string TieRegDPP = "$old";
}

  class VOP_NO_EXT <VOPProfile p> : VOPProfile <p.ArgVT> {
  let HasExt = 0;
  let HasExtDPP = 0;
  let HasExtVOP3DPP = 0;
  let HasExt32BitDPP = 0;
  let HasExt64BitDPP = 0;
  let HasExtSDWA = 0;
  let HasExtSDWA9 = 0;
}

class VOP_PAT_GEN <VOPProfile p, int mode=PatGenMode.NoPattern> : VOPProfile <p.ArgVT> {
  let NeedPatGen = mode;
}
def VOP_F16_F16 : VOPProfile <[f16, f16, untyped, untyped]>;
def VOP_F16_I16 : VOPProfile <[f16, i16, untyped, untyped]>;
def VOP_I16_F16 : VOPProfile <[i16, f16, untyped, untyped]>;
def VOP_I16_I16 : VOPProfile <[i16, i16, untyped, untyped]>;

def VOP_F16_F16_F16 : VOPProfile <[f16, f16, f16, untyped]>;
def VOP_F16_F16_I16 : VOPProfile <[f16, f16, i16, untyped]>;
def VOP_F16_F16_I32 : VOPProfile <[f16, f16, i32, untyped]>;
def VOP_I16_I16_I16 : VOPProfile <[i16, i16, i16, untyped]>;
def VOP_I16_I16_I16_ARITH : VOPProfile <[i16, i16, i16, untyped], 0, /*EnableClamp=*/1>;

def VOP_I16_I16_I16_I16 : VOPProfile <[i16, i16, i16, i16, untyped]>;
def VOP_F16_F16_F16_F16 : VOPProfile <[f16, f16, f16, f16, untyped]>;

def VOP_I32_I16_I16_I32 : VOPProfile <[i32, i16, i16, i32, untyped]>;
def VOP_I32_I16 : VOPProfile <[i32, i16, untyped, untyped]>;

def VOP_V2F16_V2F16_V2F16 : VOPProfile <[v2f16, v2f16, v2f16, untyped]>;
def VOP_V2I16_V2I16_V2I16 : VOPProfile <[v2i16, v2i16, v2i16, untyped]>;
def VOP_B32_F16_F16 : VOPProfile <[i32, f16, f16, untyped]>;

def VOP_V2F16_V2F16_V2F16_V2F16 : VOPProfile <[v2f16, v2f16, v2f16, v2f16]>;
def VOP_V2I16_V2I16_V2I16_V2I16 : VOPProfile <[v2i16, v2i16, v2i16, v2i16]>;
def VOP_V2I16_F32_F32 : VOPProfile <[v2i16, f32, f32, untyped]>;
def VOP_V2I16_I32_I32 : VOPProfile <[v2i16, i32, i32, untyped]>;

def VOP_F16_V2F16_V2F16_F16 : VOPProfile <[f16, v2f16, v2f16, f16]>;
def VOP_I16_V2I16_V2I16_I16 : VOPProfile <[i16, v2i16, v2i16, i16]>;
def VOP_F32_V2I16_V2I16_F32 : VOPProfile <[f32, v2i16, v2i16, f32]>;

def VOP_F32_V2F16_V2F16_V2F16 : VOPProfile <[f32, v2f16, v2f16, v2f16]>;

def VOP_NONE : VOPProfile <[untyped, untyped, untyped, untyped]>;

def VOP_F32_F32 : VOPProfile <[f32, f32, untyped, untyped]>;
def VOP_F32_F64 : VOPProfile <[f32, f64, untyped, untyped]>;
def VOP_F32_I32 : VOPProfile <[f32, i32, untyped, untyped]>;
def VOP_F64_F32 : VOPProfile <[f64, f32, untyped, untyped]>;
def VOP_F64_F64 : VOPProfile <[f64, f64, untyped, untyped]>;
def VOP_F64_I32 : VOPProfile <[f64, i32, untyped, untyped]>;
def VOP_I32_F32 : VOPProfile <[i32, f32, untyped, untyped]>;
def VOP_I32_F64 : VOPProfile <[i32, f64, untyped, untyped]>;
def VOP_I32_I32 : VOPProfile <[i32, i32, untyped, untyped]>;
def VOP_F16_F32 : VOPProfile <[f16, f32, untyped, untyped]>;
def VOP_F32_F16 : VOPProfile <[f32, f16, untyped, untyped]>;
def VOP_I64_I64 : VOPProfile <[i64, i64, untyped, untyped]>;

def VOP_F32_F32_F16 : VOPProfile <[f32, f32, f16, untyped]>;
def VOP_F32_F32_F32 : VOPProfile <[f32, f32, f32, untyped]>;
def VOP_F32_F32_I32 : VOPProfile <[f32, f32, i32, untyped]>;
def VOP_F64_F64_F64 : VOPProfile <[f64, f64, f64, untyped]>;
def VOP_F64_F64_I32 : VOPProfile <[f64, f64, i32, untyped]>;
def VOP_I32_F32_F32 : VOPProfile <[i32, f32, f32, untyped]>;
def VOP_I32_F32_I32 : VOPProfile <[i32, f32, i32, untyped]>;
def VOP_I32_I32_I32 : VOPProfile <[i32, i32, i32, untyped]>;
def VOP_I32_I32_I32_ARITH : VOPProfile <[i32, i32, i32, untyped], 0, /*EnableClamp=*/1>;
def VOP_V2F16_F32_F32 : VOPProfile <[v2f16, f32, f32, untyped]>;
def VOP_F32_F16_F16_F16 : VOPProfile <[f32, f16, f16, f16]>;

def VOP_I64_I64_I32 : VOPProfile <[i64, i64, i32, untyped]>;
def VOP_I64_I32_I64 : VOPProfile <[i64, i32, i64, untyped]>;
def VOP_I64_I64_I64 : VOPProfile <[i64, i64, i64, untyped]>;

def VOP_F16_F32_F16_F32 : VOPProfile <[f16, f32, f16, f32]>;
def VOP_F32_F32_F16_F16 : VOPProfile <[f32, f32, f16, f16]>;
def VOP_F32_F32_F32_F32 : VOPProfile <[f32, f32, f32, f32]>;
def VOP_F64_F64_F64_F64 : VOPProfile <[f64, f64, f64, f64]>;
def VOP_I32_I32_I32_I32 : VOPProfile <[i32, i32, i32, i32]>;
def VOP_I64_I32_I32_I64 : VOPProfile <[i64, i32, i32, i64]>;
def VOP_I32_F32_I32_I32 : VOPProfile <[i32, f32, i32, i32]>;
def VOP_I64_I64_I32_I64 : VOPProfile <[i64, i64, i32, i64]>;
def VOP_V4I32_I64_I32_V4I32 : VOPProfile <[v4i32, i64, i32, v4i32]>;

def VOP_F32_V2F16_V2F16_F32 : VOPProfile <[f32, v2f16, v2f16, f32]>;
def VOP_I32_V2I16_V2I16_I32 : VOPProfile <[i32, v2i16, v2i16, i32]>;

def VOP_V4F32_F32_F32_V4F32       : VOPProfile <[v4f32,  f32,   f32,   v4f32]>;
def VOP_V16F32_F32_F32_V16F32     : VOPProfile <[v16f32, f32,   f32,   v16f32]>;
def VOP_V32F32_F32_F32_V32F32     : VOPProfile <[v32f32, f32,   f32,   v32f32]>;
def VOP_V4F32_V4F16_V4F16_V4F32   : VOPProfile <[v4f32,  v4f16, v4f16, v4f32]>;
def VOP_V16F32_V4F16_V4F16_V16F32 : VOPProfile <[v16f32, v4f16, v4f16, v16f32]>;
def VOP_V32F32_V4F16_V4F16_V32F32 : VOPProfile <[v32f32, v4f16, v4f16, v32f32]>;
def VOP_V4F32_V2I16_V2I16_V4F32   : VOPProfile <[v4f32,  v2i16, v2i16, v4f32]>;
def VOP_V16F32_V2I16_V2I16_V16F32 : VOPProfile <[v16f32, v2i16, v2i16, v16f32]>;
def VOP_V32F32_V2I16_V2I16_V32F32 : VOPProfile <[v32f32, v2i16, v2i16, v32f32]>;
def VOP_V4I32_I32_I32_V4I32       : VOPProfile <[v4i32,  i32,   i32,   v4i32]>;
def VOP_V16I32_I32_I32_V16I32     : VOPProfile <[v16i32, i32,   i32,   v16i32]>;
def VOP_V32I32_I32_I32_V32I32     : VOPProfile <[v32i32, i32,   i32,   v32i32]>;

def VOP_V4F64_F64_F64_V4F64 : VOPProfile <[v4f64, f64, f64, v4f64]>;
def VOP_V1F64_F64_F64_V1F64 : VOPProfile <[v1f64, f64, f64, v1f64]>;

def VOP_V2F32_V2F32_V2F32_V2F32   : VOPProfile <[v2f32,  v2f32, v2f32, v2f32]>;
def VOP_V2F32_V2F32_V2F32         : VOPProfile <[v2f32,  v2f32, v2f32, untyped]>;
def VOP_V2I32_V2I32_V2I32         : VOPProfile <[v2i32,  v2i32, v2i32, untyped]>;
def VOP_V4F32_V4I16_V4I16_V4F32   : VOPProfile <[v4f32,  v4i16, v4i16, v4f32]>;
def VOP_V16F32_V4I16_V4I16_V16F32 : VOPProfile <[v16f32, v4i16, v4i16, v16f32]>;
def VOP_V32F32_V4I16_V4I16_V32F32 : VOPProfile <[v32f32, v4i16, v4i16, v32f32]>;

def VOP_V4I32_I64_I64_V4I32       : VOPProfile <[v4i32,  i64,   i64,   v4i32]>;
def VOP_V16I32_I64_I64_V16I32     : VOPProfile <[v16i32, i64,   i64,   v16i32]>;
def VOP_V4F32_V2F32_V2F32_V4F32   : VOPProfile <[v4f32,  v2f32, v2f32, v4f32]>;
def VOP_V16F32_V2F32_V2F32_V16F32 : VOPProfile <[v16f32, v2f32, v2f32, v16f32]>;
def VOP_V4F32_I64_I64_V4F32       : VOPProfile <[v4f32,  i64,   i64,   v4f32]>;
def VOP_V16F32_I64_I64_V16F32     : VOPProfile <[v16f32, i64,   i64,   v16f32]>;

def VOP_V4F32_V4F16_V8F16_I32     : VOPProfile <[v4f32,  v4f16, v8f16, i32]>;
def VOP_V16F32_V4F16_V8F16_I32    : VOPProfile <[v16f32, v4f16, v8f16, i32]>;
def VOP_V4F32_V4I16_V8I16_I32     : VOPProfile <[v4f32,  v4i16, v8i16, i32]>;
def VOP_V16F32_V4I16_V8I16_I32    : VOPProfile <[v16f32, v4i16, v8i16, i32]>;
def VOP_V4I32_V2I32_V4I32_I32     : VOPProfile <[v4i32,  v2i32, v4i32, i32]>;
def VOP_V16I32_V2I32_V4I32_I32    : VOPProfile <[v16i32, v2i32, v4i32, i32]>;
def VOP_V4F32_V2I32_V4I32_I32     : VOPProfile <[v4f32,  v2i32, v4i32, i32]>;
def VOP_V16F32_V2I32_V4I32_I32    : VOPProfile <[v16f32, v2i32, v4i32, i32]>;

class Commutable_REV <string revOp, bit isOrig> {
  string RevOp = revOp;
  bit IsOrig = isOrig;
}

class AtomicNoRet <string noRetOp, bit isRet> {
  string NoRetOp = noRetOp;
  bit IsRet = isRet;
}

//===----------------------------------------------------------------------===//
// Interpolation opcodes
//===----------------------------------------------------------------------===//

class VINTRPDstOperand <RegisterClass rc> : RegisterOperand <rc, "printVINTRPDst">;

class VINTRP_Pseudo <string opName, dag outs, dag ins, list<dag> pattern> :
  VINTRPCommon <outs, ins, "", pattern>,
  SIMCInstr<opName, SIEncodingFamily.NONE> {
  let isPseudo = 1;
  let isCodeGenOnly = 1;
}

// FIXME-GFX10: WIP.
class VINTRP_Real_si <bits <2> op, string opName, dag outs, dag ins,
                      string asm, int encodingFamily> :
  VINTRPCommon <outs, ins, asm, []>,
  VINTRPe <op>,
  SIMCInstr<opName, encodingFamily> {
}

class VINTRP_Real_vi <bits <2> op, string opName, dag outs, dag ins,
                      string asm> :
  VINTRPCommon <outs, ins, asm, []>,
  VINTRPe_vi <op>,
  SIMCInstr<opName, SIEncodingFamily.VI> {
  let AssemblerPredicate = VIAssemblerPredicate;
  let DecoderNamespace = "GFX8";
}

// FIXME-GFX10: WIP.
multiclass VINTRP_m <bits <2> op, dag outs, dag ins, string asm,
                     list<dag> pattern = []> {
  def "" : VINTRP_Pseudo <NAME, outs, ins, pattern>;

  let AssemblerPredicate = isGFX6GFX7, DecoderNamespace = "GFX6GFX7" in {
    def _si : VINTRP_Real_si <op, NAME, outs, ins, asm, SIEncodingFamily.SI>;
  } // End AssemblerPredicate = isGFX6GFX7, DecoderNamespace = "GFX6GFX7"

  def _vi : VINTRP_Real_vi <op, NAME, outs, ins, asm>;

  let AssemblerPredicate = isGFX10Only, DecoderNamespace = "GFX10" in {
    def _gfx10 : VINTRP_Real_si<op, NAME, outs, ins, asm, SIEncodingFamily.GFX10>;
  } // End AssemblerPredicate = isGFX10Only, DecoderNamespace = "GFX10"
}

//===----------------------------------------------------------------------===//
// Vector instruction mappings
//===----------------------------------------------------------------------===//

// Maps an opcode in e32 form to its e64 equivalent
def getVOPe64 : InstrMapping {
  let FilterClass = "VOP";
  let RowFields = ["OpName"];
  let ColFields = ["Size", "VOP3"];
  let KeyCol = ["4", "0"];
  let ValueCols = [["8", "1"]];
}

// Maps an opcode in e64 form to its e32 equivalent
def getVOPe32 : InstrMapping {
  let FilterClass = "VOP";
  let RowFields = ["OpName"];
  let ColFields = ["Size", "VOP3"];
  let KeyCol = ["8", "1"];
  let ValueCols = [["4", "0"]];
}

// Maps ordinary instructions to their SDWA counterparts
def getSDWAOp : InstrMapping {
  let FilterClass = "VOP";
  let RowFields = ["OpName"];
  let ColFields = ["AsmVariantName"];
  let KeyCol = ["Default"];
  let ValueCols = [["SDWA"]];
}

// Maps SDWA instructions to their ordinary counterparts
def getBasicFromSDWAOp : InstrMapping {
  let FilterClass = "VOP";
  let RowFields = ["OpName"];
  let ColFields = ["AsmVariantName"];
  let KeyCol = ["SDWA"];
  let ValueCols = [["Default"]];
}

// Maps ordinary instructions to their DPP counterparts
def getDPPOp32 : InstrMapping {
  let FilterClass = "VOP";
  let RowFields = ["OpName"];
  let ColFields = ["AsmVariantName"];
  let KeyCol = ["Default"];
  let ValueCols = [["DPP"]];
}

def getDPPOp64 : InstrMapping {
  let FilterClass = "VOP";
  let RowFields = ["OpName"];
  let ColFields = ["AsmVariantName"];
  let KeyCol = ["VOP3"];
  let ValueCols = [["VOP3_DPP"]];
}

// Maps an commuted opcode to its original version
def getCommuteOrig : InstrMapping {
  let FilterClass = "Commutable_REV";
  let RowFields = ["RevOp"];
  let ColFields = ["IsOrig"];
  let KeyCol = ["0"];
  let ValueCols = [["1"]];
}

// Maps an original opcode to its commuted version
def getCommuteRev : InstrMapping {
  let FilterClass = "Commutable_REV";
  let RowFields = ["RevOp"];
  let ColFields = ["IsOrig"];
  let KeyCol = ["1"];
  let ValueCols = [["0"]];
}

def getMCOpcodeGen : InstrMapping {
  let FilterClass = "SIMCInstr";
  let RowFields = ["PseudoInstr"];
  let ColFields = ["Subtarget"];
  let KeyCol = [!cast<string>(SIEncodingFamily.NONE)];
  // These columns must be kept in sync with the SIEncodingFamily enumeration.
  let ValueCols = [[!cast<string>(SIEncodingFamily.SI)],
                   [!cast<string>(SIEncodingFamily.VI)],
                   [!cast<string>(SIEncodingFamily.SDWA)],
                   [!cast<string>(SIEncodingFamily.SDWA9)],
                   // GFX80 encoding is added to work around a multiple matching
                   // issue for buffer instructions with unpacked d16 data. This
                   // does not actually change the encoding, and thus may be
                   // removed later.
                   [!cast<string>(SIEncodingFamily.GFX80)],
                   [!cast<string>(SIEncodingFamily.GFX9)],
                   [!cast<string>(SIEncodingFamily.GFX10)],
                   [!cast<string>(SIEncodingFamily.SDWA10)],
                   [!cast<string>(SIEncodingFamily.GFX90A)],
                   [!cast<string>(SIEncodingFamily.GFX940)],
                   [!cast<string>(SIEncodingFamily.GFX11)]];
}

// Get equivalent SOPK instruction.
def getSOPKOp : InstrMapping {
  let FilterClass = "SOPKInstTable";
  let RowFields = ["BaseCmpOp"];
  let ColFields = ["IsSOPK"];
  let KeyCol = ["0"];
  let ValueCols = [["1"]];
}

def getAddr64Inst : InstrMapping {
  let FilterClass = "MUBUFAddr64Table";
  let RowFields = ["OpName"];
  let ColFields = ["IsAddr64"];
  let KeyCol = ["0"];
  let ValueCols = [["1"]];
}

def getIfAddr64Inst : InstrMapping {
  let FilterClass = "MUBUFAddr64Table";
  let RowFields = ["OpName"];
  let ColFields = ["IsAddr64"];
  let KeyCol = ["1"];
  let ValueCols = [["1"]];
}

// Maps an atomic opcode to its returnless version.
def getAtomicNoRetOp : InstrMapping {
  let FilterClass = "AtomicNoRet";
  let RowFields = ["NoRetOp"];
  let ColFields = ["IsRet"];
  let KeyCol = ["1"];
  let ValueCols = [["0"]];
}

// Maps a GLOBAL to its SADDR form.
def getGlobalSaddrOp : InstrMapping {
  let FilterClass = "GlobalSaddrTable";
  let RowFields = ["SaddrOp"];
  let ColFields = ["IsSaddr"];
  let KeyCol = ["0"];
  let ValueCols = [["1"]];
}

// Maps a GLOBAL SADDR to its VADDR form.
def getGlobalVaddrOp : InstrMapping {
  let FilterClass = "GlobalSaddrTable";
  let RowFields = ["SaddrOp"];
  let ColFields = ["IsSaddr"];
  let KeyCol = ["1"];
  let ValueCols = [["0"]];
}

// Maps a v_cmpx opcode with sdst to opcode without sdst.
def getVCMPXNoSDstOp : InstrMapping {
  let FilterClass = "VCMPXNoSDstTable";
  let RowFields = ["NoSDstOp"];
  let ColFields = ["HasSDst"];
  let KeyCol = ["1"];
  let ValueCols = [["0"]];
}

// Maps a SOPP to a SOPP with S_NOP
def getSOPPWithRelaxation : InstrMapping {
  let FilterClass = "SOPPRelaxTable";
  let RowFields = ["KeyName"];
  let ColFields = ["IsRelaxed"];
  let KeyCol = ["0"];
  let ValueCols = [["1"]];
}

// Maps flat scratch opcodes by addressing modes
def getFlatScratchInstSTfromSS : InstrMapping {
  let FilterClass = "FlatScratchInst";
  let RowFields = ["SVOp"];
  let ColFields = ["Mode"];
  let KeyCol = ["SS"];
  let ValueCols = [["ST"]];
}

def getFlatScratchInstSSfromSV : InstrMapping {
  let FilterClass = "FlatScratchInst";
  let RowFields = ["SVOp"];
  let ColFields = ["Mode"];
  let KeyCol = ["SV"];
  let ValueCols = [["SS"]];
}

def getFlatScratchInstSVfromSVS : InstrMapping {
  let FilterClass = "FlatScratchInst";
  let RowFields = ["SVOp"];
  let ColFields = ["Mode"];
  let KeyCol = ["SVS"];
  let ValueCols = [["SV"]];
}

def getFlatScratchInstSVfromSS : InstrMapping {
  let FilterClass = "FlatScratchInst";
  let RowFields = ["SVOp"];
  let ColFields = ["Mode"];
  let KeyCol = ["SS"];
  let ValueCols = [["SV"]];
}

def getMFMAEarlyClobberOp : InstrMapping {
  let FilterClass = "MFMATable";
  let RowFields = ["FMAOp"];
  let ColFields = ["IsMac"];
  let KeyCol = ["1"];
  let ValueCols = [["0"]];
}

// Maps an v_cmp instruction to its v_cmpx equivalent.
def getVCMPXOpFromVCMP : InstrMapping {
  let FilterClass = "VCMPVCMPXTable";
  let RowFields = ["VCMPOp"];
  let ColFields = ["IsVCMPX"];
  let KeyCol = ["0"];
  let ValueCols = [["1"]];
}

def VOPDComponentTable : GenericTable {
  let FilterClass = "VOPD_Component";
  let CppTypeName = "VOPDComponentInfo";
  let Fields = ["BaseVOP", "VOPDOp", "CanBeVOPDX"];
  let PrimaryKey = ["BaseVOP"];
  let PrimaryKeyName = "getVOPDComponentHelper";
}

def VOPDPairs : GenericTable {
  let FilterClass = "VOPD_Base";
  let CppTypeName = "VOPDInfo";
  let Fields = ["Opcode", "OpX", "OpY"];
  let PrimaryKey = ["Opcode"];
  let PrimaryKeyName = "getVOPDOpcodeHelper";
}

def getVOPDInfoFromComponentOpcodes : SearchIndex {
  let Table = VOPDPairs;
  let Key = ["OpX", "OpY"];
}

include "SIInstructions.td"

include "DSInstructions.td"
include "MIMGInstructions.td"
