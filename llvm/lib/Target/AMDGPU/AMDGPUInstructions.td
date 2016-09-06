//===-- AMDGPUInstructions.td - Common instruction defs ---*- tablegen -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains instruction defs that are common to all hw codegen
// targets.
//
//===----------------------------------------------------------------------===//

class AMDGPUInst <dag outs, dag ins, string asm = "",
  list<dag> pattern = []> : Instruction {
  field bit isRegisterLoad = 0;
  field bit isRegisterStore = 0;

  let Namespace = "AMDGPU";
  let OutOperandList = outs;
  let InOperandList = ins;
  let AsmString = asm;
  let Pattern = pattern;
  let Itinerary = NullALU;

  // SoftFail is a field the disassembler can use to provide a way for
  // instructions to not match without killing the whole decode process. It is
  // mainly used for ARM, but Tablegen expects this field to exist or it fails
  // to build the decode table.
  field bits<64> SoftFail = 0;

  let DecoderNamespace = Namespace;

  let TSFlags{63} = isRegisterLoad;
  let TSFlags{62} = isRegisterStore;
}

class AMDGPUShaderInst <dag outs, dag ins, string asm = "",
  list<dag> pattern = []> : AMDGPUInst<outs, ins, asm, pattern> {

  field bits<32> Inst = 0xffffffff;
}

def FP32Denormals : Predicate<"Subtarget.hasFP32Denormals()">;
def FP64Denormals : Predicate<"Subtarget.hasFP64Denormals()">;
def UnsafeFPMath : Predicate<"TM.Options.UnsafeFPMath">;

def InstFlag : OperandWithDefaultOps <i32, (ops (i32 0))>;
def ADDRIndirect : ComplexPattern<iPTR, 2, "SelectADDRIndirect", [], []>;

// 32-bit VALU immediate operand that uses the constant bus.
def u32kimm : Operand<i32> {
  let OperandNamespace = "AMDGPU";
  let OperandType = "OPERAND_KIMM32";
  let PrintMethod = "printU32ImmOperand";
}

let OperandType = "OPERAND_IMMEDIATE" in {

def u32imm : Operand<i32> {
  let PrintMethod = "printU32ImmOperand";
}

def u16imm : Operand<i16> {
  let PrintMethod = "printU16ImmOperand";
}

def u8imm : Operand<i8> {
  let PrintMethod = "printU8ImmOperand";
}

} // End OperandType = "OPERAND_IMMEDIATE"

//===--------------------------------------------------------------------===//
// Custom Operands
//===--------------------------------------------------------------------===//
def brtarget   : Operand<OtherVT>;

//===----------------------------------------------------------------------===//
// PatLeafs for floating-point comparisons
//===----------------------------------------------------------------------===//

def COND_OEQ : PatLeaf <
  (cond),
  [{return N->get() == ISD::SETOEQ || N->get() == ISD::SETEQ;}]
>;

def COND_ONE : PatLeaf <
  (cond),
  [{return N->get() == ISD::SETONE || N->get() == ISD::SETNE;}]
>;

def COND_OGT : PatLeaf <
  (cond),
  [{return N->get() == ISD::SETOGT || N->get() == ISD::SETGT;}]
>;

def COND_OGE : PatLeaf <
  (cond),
  [{return N->get() == ISD::SETOGE || N->get() == ISD::SETGE;}]
>;

def COND_OLT : PatLeaf <
  (cond),
  [{return N->get() == ISD::SETOLT || N->get() == ISD::SETLT;}]
>;

def COND_OLE : PatLeaf <
  (cond),
  [{return N->get() == ISD::SETOLE || N->get() == ISD::SETLE;}]
>;


def COND_O : PatLeaf <(cond), [{return N->get() == ISD::SETO;}]>;
def COND_UO : PatLeaf <(cond), [{return N->get() == ISD::SETUO;}]>;

//===----------------------------------------------------------------------===//
// PatLeafs for unsigned / unordered comparisons
//===----------------------------------------------------------------------===//

def COND_UEQ : PatLeaf <(cond), [{return N->get() == ISD::SETUEQ;}]>;
def COND_UNE : PatLeaf <(cond), [{return N->get() == ISD::SETUNE;}]>;
def COND_UGT : PatLeaf <(cond), [{return N->get() == ISD::SETUGT;}]>;
def COND_UGE : PatLeaf <(cond), [{return N->get() == ISD::SETUGE;}]>;
def COND_ULT : PatLeaf <(cond), [{return N->get() == ISD::SETULT;}]>;
def COND_ULE : PatLeaf <(cond), [{return N->get() == ISD::SETULE;}]>;

// XXX - For some reason R600 version is preferring to use unordered
// for setne?
def COND_UNE_NE : PatLeaf <
  (cond),
  [{return N->get() == ISD::SETUNE || N->get() == ISD::SETNE;}]
>;

//===----------------------------------------------------------------------===//
// PatLeafs for signed comparisons
//===----------------------------------------------------------------------===//

def COND_SGT : PatLeaf <(cond), [{return N->get() == ISD::SETGT;}]>;
def COND_SGE : PatLeaf <(cond), [{return N->get() == ISD::SETGE;}]>;
def COND_SLT : PatLeaf <(cond), [{return N->get() == ISD::SETLT;}]>;
def COND_SLE : PatLeaf <(cond), [{return N->get() == ISD::SETLE;}]>;

//===----------------------------------------------------------------------===//
// PatLeafs for integer equality
//===----------------------------------------------------------------------===//

def COND_EQ : PatLeaf <
  (cond),
  [{return N->get() == ISD::SETEQ || N->get() == ISD::SETUEQ;}]
>;

def COND_NE : PatLeaf <
  (cond),
  [{return N->get() == ISD::SETNE || N->get() == ISD::SETUNE;}]
>;

def COND_NULL : PatLeaf <
  (cond),
  [{(void)N; return false;}]
>;


//===----------------------------------------------------------------------===//
// Misc. PatFrags
//===----------------------------------------------------------------------===//

class HasOneUseBinOp<SDPatternOperator op> : PatFrag<
  (ops node:$src0, node:$src1),
  (op $src0, $src1),
  [{ return N->hasOneUse(); }]
>;

//===----------------------------------------------------------------------===//
// Load/Store Pattern Fragments
//===----------------------------------------------------------------------===//

class PrivateMemOp <dag ops, dag frag> : PatFrag <ops, frag, [{
  return cast<MemSDNode>(N)->getAddressSpace() == AMDGPUAS::PRIVATE_ADDRESS;
}]>;

class PrivateLoad <SDPatternOperator op> : PrivateMemOp <
  (ops node:$ptr), (op node:$ptr)
>;

class PrivateStore <SDPatternOperator op> : PrivateMemOp <
  (ops node:$value, node:$ptr), (op node:$value, node:$ptr)
>;

def load_private : PrivateLoad <load>;

def truncstorei8_private : PrivateStore <truncstorei8>;
def truncstorei16_private : PrivateStore <truncstorei16>;
def store_private : PrivateStore <store>;

class GlobalMemOp <dag ops, dag frag> : PatFrag <ops, frag, [{
  return cast<MemSDNode>(N)->getAddressSpace() == AMDGPUAS::GLOBAL_ADDRESS;
}]>;

// Global address space loads
class GlobalLoad <SDPatternOperator op> : GlobalMemOp <
  (ops node:$ptr), (op node:$ptr)
>;

def global_load : GlobalLoad <load>;

// Global address space stores
class GlobalStore <SDPatternOperator op> : GlobalMemOp <
  (ops node:$value, node:$ptr), (op node:$value, node:$ptr)
>;

def global_store : GlobalStore <store>;
def global_store_atomic : GlobalStore<atomic_store>;


class ConstantMemOp <dag ops, dag frag> : PatFrag <ops, frag, [{
  return cast<MemSDNode>(N)->getAddressSpace() == AMDGPUAS::CONSTANT_ADDRESS;
}]>;

// Constant address space loads
class ConstantLoad <SDPatternOperator op> : ConstantMemOp <
  (ops node:$ptr), (op node:$ptr)
>;

def constant_load : ConstantLoad<load>;

class LocalMemOp <dag ops, dag frag> : PatFrag <ops, frag, [{
  return cast<MemSDNode>(N)->getAddressSpace() == AMDGPUAS::LOCAL_ADDRESS;
}]>;

// Local address space loads
class LocalLoad <SDPatternOperator op> : LocalMemOp <
  (ops node:$ptr), (op node:$ptr)
>;

class LocalStore <SDPatternOperator op> : LocalMemOp <
  (ops node:$value, node:$ptr), (op node:$value, node:$ptr)
>;

class FlatMemOp <dag ops, dag frag> : PatFrag <ops, frag, [{
  return cast<MemSDNode>(N)->getAddressSPace() == AMDGPUAS::FLAT_ADDRESS;
}]>;

class FlatLoad <SDPatternOperator op> : FlatMemOp <
  (ops node:$ptr), (op node:$ptr)
>;

class AZExtLoadBase <SDPatternOperator ld_node>: PatFrag<(ops node:$ptr),
                                              (ld_node node:$ptr), [{
  LoadSDNode *L = cast<LoadSDNode>(N);
  return L->getExtensionType() == ISD::ZEXTLOAD ||
         L->getExtensionType() == ISD::EXTLOAD;
}]>;

def az_extload : AZExtLoadBase <unindexedload>;

def az_extloadi8 : PatFrag<(ops node:$ptr), (az_extload node:$ptr), [{
  return cast<LoadSDNode>(N)->getMemoryVT() == MVT::i8;
}]>;

def az_extloadi8_global : GlobalLoad <az_extloadi8>;
def sextloadi8_global : GlobalLoad <sextloadi8>;

def az_extloadi8_constant : ConstantLoad <az_extloadi8>;
def sextloadi8_constant : ConstantLoad <sextloadi8>;

def az_extloadi8_local : LocalLoad <az_extloadi8>;
def sextloadi8_local : LocalLoad <sextloadi8>;

def extloadi8_private : PrivateLoad <az_extloadi8>;
def sextloadi8_private : PrivateLoad <sextloadi8>;

def az_extloadi16 : PatFrag<(ops node:$ptr), (az_extload node:$ptr), [{
  return cast<LoadSDNode>(N)->getMemoryVT() == MVT::i16;
}]>;

def az_extloadi16_global : GlobalLoad <az_extloadi16>;
def sextloadi16_global : GlobalLoad <sextloadi16>;

def az_extloadi16_constant : ConstantLoad <az_extloadi16>;
def sextloadi16_constant : ConstantLoad <sextloadi16>;

def az_extloadi16_local : LocalLoad <az_extloadi16>;
def sextloadi16_local : LocalLoad <sextloadi16>;

def extloadi16_private : PrivateLoad <az_extloadi16>;
def sextloadi16_private : PrivateLoad <sextloadi16>;

def az_extloadi32 : PatFrag<(ops node:$ptr), (az_extload node:$ptr), [{
  return cast<LoadSDNode>(N)->getMemoryVT() == MVT::i32;
}]>;

def az_extloadi32_global : GlobalLoad <az_extloadi32>;

def az_extloadi32_flat : FlatLoad <az_extloadi32>;

def az_extloadi32_constant : ConstantLoad <az_extloadi32>;

def truncstorei8_global : GlobalStore <truncstorei8>;
def truncstorei16_global : GlobalStore <truncstorei16>;

def local_store : LocalStore <store>;
def truncstorei8_local : LocalStore <truncstorei8>;
def truncstorei16_local : LocalStore <truncstorei16>;

def local_load : LocalLoad <load>;

class Aligned8Bytes <dag ops, dag frag> : PatFrag <ops, frag, [{
    return cast<MemSDNode>(N)->getAlignment() % 8 == 0;
}]>;

def local_load_aligned8bytes : Aligned8Bytes <
  (ops node:$ptr), (local_load node:$ptr)
>;

def local_store_aligned8bytes : Aligned8Bytes <
  (ops node:$val, node:$ptr), (local_store node:$val, node:$ptr)
>;

class local_binary_atomic_op<SDNode atomic_op> :
  PatFrag<(ops node:$ptr, node:$value),
    (atomic_op node:$ptr, node:$value), [{
  return cast<MemSDNode>(N)->getAddressSpace() == AMDGPUAS::LOCAL_ADDRESS;
}]>;


def atomic_swap_local : local_binary_atomic_op<atomic_swap>;
def atomic_load_add_local : local_binary_atomic_op<atomic_load_add>;
def atomic_load_sub_local : local_binary_atomic_op<atomic_load_sub>;
def atomic_load_and_local : local_binary_atomic_op<atomic_load_and>;
def atomic_load_or_local : local_binary_atomic_op<atomic_load_or>;
def atomic_load_xor_local : local_binary_atomic_op<atomic_load_xor>;
def atomic_load_nand_local : local_binary_atomic_op<atomic_load_nand>;
def atomic_load_min_local : local_binary_atomic_op<atomic_load_min>;
def atomic_load_max_local : local_binary_atomic_op<atomic_load_max>;
def atomic_load_umin_local : local_binary_atomic_op<atomic_load_umin>;
def atomic_load_umax_local : local_binary_atomic_op<atomic_load_umax>;

def mskor_global : PatFrag<(ops node:$val, node:$ptr),
                            (AMDGPUstore_mskor node:$val, node:$ptr), [{
  return cast<MemSDNode>(N)->getAddressSpace() == AMDGPUAS::GLOBAL_ADDRESS;
}]>;

multiclass AtomicCmpSwapLocal <SDNode cmp_swap_node> {

  def _32_local : PatFrag <
    (ops node:$ptr, node:$cmp, node:$swap),
    (cmp_swap_node node:$ptr, node:$cmp, node:$swap), [{
      AtomicSDNode *AN = cast<AtomicSDNode>(N);
      return AN->getMemoryVT() == MVT::i32 &&
             AN->getAddressSpace() == AMDGPUAS::LOCAL_ADDRESS;
  }]>;

  def _64_local : PatFrag<
    (ops node:$ptr, node:$cmp, node:$swap),
    (cmp_swap_node node:$ptr, node:$cmp, node:$swap), [{
      AtomicSDNode *AN = cast<AtomicSDNode>(N);
      return AN->getMemoryVT() == MVT::i64 &&
             AN->getAddressSpace() == AMDGPUAS::LOCAL_ADDRESS;
  }]>;
}

defm atomic_cmp_swap : AtomicCmpSwapLocal <atomic_cmp_swap>;

def mskor_flat : PatFrag<(ops node:$val, node:$ptr),
                            (AMDGPUstore_mskor node:$val, node:$ptr), [{
  return cast<MemSDNode>(N)->getAddressSpace() == AMDGPUAS::FLAT_ADDRESS;
}]>;

class global_binary_atomic_op<SDNode atomic_op> : PatFrag<
  (ops node:$ptr, node:$value),
  (atomic_op node:$ptr, node:$value),
  [{return cast<MemSDNode>(N)->getAddressSpace() == AMDGPUAS::GLOBAL_ADDRESS;}]
>;

class flat_binary_atomic_op<SDNode atomic_op> : PatFrag<
  (ops node:$ptr, node:$value),
  (atomic_op node:$ptr, node:$value),
  [{return cast<MemSDNode>(N)->getAddressSpace() == AMDGPUAS::FLAT_ADDRESS;}]
>;

def atomic_swap_global : global_binary_atomic_op<atomic_swap>;
def atomic_add_global : global_binary_atomic_op<atomic_load_add>;
def atomic_and_global : global_binary_atomic_op<atomic_load_and>;
def atomic_max_global : global_binary_atomic_op<atomic_load_max>;
def atomic_min_global : global_binary_atomic_op<atomic_load_min>;
def atomic_or_global : global_binary_atomic_op<atomic_load_or>;
def atomic_sub_global : global_binary_atomic_op<atomic_load_sub>;
def atomic_umax_global : global_binary_atomic_op<atomic_load_umax>;
def atomic_umin_global : global_binary_atomic_op<atomic_load_umin>;
def atomic_xor_global : global_binary_atomic_op<atomic_load_xor>;

def atomic_cmp_swap_global : global_binary_atomic_op<AMDGPUatomic_cmp_swap>;
def atomic_cmp_swap_global_nortn : PatFrag<
  (ops node:$ptr, node:$value),
  (atomic_cmp_swap_global node:$ptr, node:$value),
  [{ return SDValue(N, 0).use_empty(); }]
>;

def atomic_swap_flat : flat_binary_atomic_op<atomic_swap>;
def atomic_add_flat : flat_binary_atomic_op<atomic_load_add>;
def atomic_and_flat : flat_binary_atomic_op<atomic_load_and>;
def atomic_max_flat : flat_binary_atomic_op<atomic_load_max>;
def atomic_min_flat : flat_binary_atomic_op<atomic_load_min>;
def atomic_or_flat : flat_binary_atomic_op<atomic_load_or>;
def atomic_sub_flat : flat_binary_atomic_op<atomic_load_sub>;
def atomic_umax_flat : flat_binary_atomic_op<atomic_load_umax>;
def atomic_umin_flat : flat_binary_atomic_op<atomic_load_umin>;
def atomic_xor_flat : flat_binary_atomic_op<atomic_load_xor>;

def atomic_cmp_swap_flat : flat_binary_atomic_op<AMDGPUatomic_cmp_swap>;

//===----------------------------------------------------------------------===//
// Misc Pattern Fragments
//===----------------------------------------------------------------------===//

class Constants {
int TWO_PI = 0x40c90fdb;
int PI = 0x40490fdb;
int TWO_PI_INV = 0x3e22f983;
int FP_UINT_MAX_PLUS_1 = 0x4f800000;    // 1 << 32 in floating point encoding
int FP32_ONE = 0x3f800000;
int FP32_NEG_ONE = 0xbf800000;
int FP64_ONE = 0x3ff0000000000000;
int FP64_NEG_ONE = 0xbff0000000000000;
}
def CONST : Constants;

def FP_ZERO : PatLeaf <
  (fpimm),
  [{return N->getValueAPF().isZero();}]
>;

def FP_ONE : PatLeaf <
  (fpimm),
  [{return N->isExactlyValue(1.0);}]
>;

def FP_HALF : PatLeaf <
  (fpimm),
  [{return N->isExactlyValue(0.5);}]
>;

let isCodeGenOnly = 1, isPseudo = 1 in {

let usesCustomInserter = 1  in {

class CLAMP <RegisterClass rc> : AMDGPUShaderInst <
  (outs rc:$dst),
  (ins rc:$src0),
  "CLAMP $dst, $src0",
  [(set f32:$dst, (AMDGPUclamp f32:$src0, (f32 FP_ZERO), (f32 FP_ONE)))]
>;

class FABS <RegisterClass rc> : AMDGPUShaderInst <
  (outs rc:$dst),
  (ins rc:$src0),
  "FABS $dst, $src0",
  [(set f32:$dst, (fabs f32:$src0))]
>;

class FNEG <RegisterClass rc> : AMDGPUShaderInst <
  (outs rc:$dst),
  (ins rc:$src0),
  "FNEG $dst, $src0",
  [(set f32:$dst, (fneg f32:$src0))]
>;

} // usesCustomInserter = 1

multiclass RegisterLoadStore <RegisterClass dstClass, Operand addrClass,
                    ComplexPattern addrPat> {
let UseNamedOperandTable = 1 in {

  def RegisterLoad : AMDGPUShaderInst <
    (outs dstClass:$dst),
    (ins addrClass:$addr, i32imm:$chan),
    "RegisterLoad $dst, $addr",
    [(set i32:$dst, (AMDGPUregister_load addrPat:$addr, (i32 timm:$chan)))]
  > {
    let isRegisterLoad = 1;
  }

  def RegisterStore : AMDGPUShaderInst <
    (outs),
    (ins dstClass:$val, addrClass:$addr, i32imm:$chan),
    "RegisterStore $val, $addr",
    [(AMDGPUregister_store i32:$val, addrPat:$addr, (i32 timm:$chan))]
  > {
    let isRegisterStore = 1;
  }
}
}

} // End isCodeGenOnly = 1, isPseudo = 1

/* Generic helper patterns for intrinsics */
/* -------------------------------------- */

class POW_Common <AMDGPUInst log_ieee, AMDGPUInst exp_ieee, AMDGPUInst mul>
  : Pat <
  (fpow f32:$src0, f32:$src1),
  (exp_ieee (mul f32:$src1, (log_ieee f32:$src0)))
>;

/* Other helper patterns */
/* --------------------- */

/* Extract element pattern */
class Extract_Element <ValueType sub_type, ValueType vec_type, int sub_idx,
                       SubRegIndex sub_reg>
  : Pat<
  (sub_type (extractelt vec_type:$src, sub_idx)),
  (EXTRACT_SUBREG $src, sub_reg)
>;

/* Insert element pattern */
class Insert_Element <ValueType elem_type, ValueType vec_type,
                      int sub_idx, SubRegIndex sub_reg>
  : Pat <
  (insertelt vec_type:$vec, elem_type:$elem, sub_idx),
  (INSERT_SUBREG $vec, $elem, sub_reg)
>;

// XXX: Convert to new syntax and use COPY_TO_REG, once the DFAPacketizer
// can handle COPY instructions.
// bitconvert pattern
class BitConvert <ValueType dt, ValueType st, RegisterClass rc> : Pat <
  (dt (bitconvert (st rc:$src0))),
  (dt rc:$src0)
>;

// XXX: Convert to new syntax and use COPY_TO_REG, once the DFAPacketizer
// can handle COPY instructions.
class DwordAddrPat<ValueType vt, RegisterClass rc> : Pat <
  (vt (AMDGPUdwordaddr (vt rc:$addr))),
  (vt rc:$addr)
>;

// BFI_INT patterns

multiclass BFIPatterns <Instruction BFI_INT,
                        Instruction LoadImm32,
                        RegisterClass RC64> {
  // Definition from ISA doc:
  // (y & x) | (z & ~x)
  def : Pat <
    (or (and i32:$y, i32:$x), (and i32:$z, (not i32:$x))),
    (BFI_INT $x, $y, $z)
  >;

  // SHA-256 Ch function
  // z ^ (x & (y ^ z))
  def : Pat <
    (xor i32:$z, (and i32:$x, (xor i32:$y, i32:$z))),
    (BFI_INT $x, $y, $z)
  >;

  def : Pat <
    (fcopysign f32:$src0, f32:$src1),
    (BFI_INT (LoadImm32 0x7fffffff), $src0, $src1)
  >;

  def : Pat <
    (f64 (fcopysign f64:$src0, f64:$src1)),
    (REG_SEQUENCE RC64,
      (i32 (EXTRACT_SUBREG $src0, sub0)), sub0,
      (BFI_INT (LoadImm32 0x7fffffff),
               (i32 (EXTRACT_SUBREG $src0, sub1)),
               (i32 (EXTRACT_SUBREG $src1, sub1))), sub1)
  >;
}

// SHA-256 Ma patterns

// ((x & z) | (y & (x | z))) -> BFI_INT (XOR x, y), z, y
class SHA256MaPattern <Instruction BFI_INT, Instruction XOR> : Pat <
  (or (and i32:$x, i32:$z), (and i32:$y, (or i32:$x, i32:$z))),
  (BFI_INT (XOR i32:$x, i32:$y), i32:$z, i32:$y)
>;

// Bitfield extract patterns

def IMMZeroBasedBitfieldMask : PatLeaf <(imm), [{
  return isMask_32(N->getZExtValue());
}]>;

def IMMPopCount : SDNodeXForm<imm, [{
  return CurDAG->getTargetConstant(countPopulation(N->getZExtValue()), SDLoc(N),
                                   MVT::i32);
}]>;

class BFEPattern <Instruction BFE, Instruction MOV> : Pat <
  (i32 (and (i32 (srl i32:$src, i32:$rshift)), IMMZeroBasedBitfieldMask:$mask)),
  (BFE $src, $rshift, (MOV (i32 (IMMPopCount $mask))))
>;

// rotr pattern
class ROTRPattern <Instruction BIT_ALIGN> : Pat <
  (rotr i32:$src0, i32:$src1),
  (BIT_ALIGN $src0, $src0, $src1)
>;

// This matches 16 permutations of
// max(min(x, y), min(max(x, y), z))
class IntMed3Pat<Instruction med3Inst,
                 SDPatternOperator max,
                 SDPatternOperator max_oneuse,
                 SDPatternOperator min_oneuse> : Pat<
  (max (min_oneuse i32:$src0, i32:$src1),
       (min_oneuse (max_oneuse i32:$src0, i32:$src1), i32:$src2)),
  (med3Inst $src0, $src1, $src2)
>;

let Properties = [SDNPCommutative, SDNPAssociative] in {
def smax_oneuse : HasOneUseBinOp<smax>;
def smin_oneuse : HasOneUseBinOp<smin>;
def umax_oneuse : HasOneUseBinOp<umax>;
def umin_oneuse : HasOneUseBinOp<umin>;
} // Properties = [SDNPCommutative, SDNPAssociative]


// 24-bit arithmetic patterns
def umul24 : PatFrag <(ops node:$x, node:$y), (mul node:$x, node:$y)>;

// Special conversion patterns

def cvt_rpi_i32_f32 : PatFrag <
  (ops node:$src),
  (fp_to_sint (ffloor (fadd $src, FP_HALF))),
  [{ (void) N; return TM.Options.NoNaNsFPMath; }]
>;

def cvt_flr_i32_f32 : PatFrag <
  (ops node:$src),
  (fp_to_sint (ffloor $src)),
  [{ (void)N; return TM.Options.NoNaNsFPMath; }]
>;

class IMad24Pat<Instruction Inst> : Pat <
  (add (AMDGPUmul_i24 i32:$src0, i32:$src1), i32:$src2),
  (Inst $src0, $src1, $src2)
>;

class UMad24Pat<Instruction Inst> : Pat <
  (add (AMDGPUmul_u24 i32:$src0, i32:$src1), i32:$src2),
  (Inst $src0, $src1, $src2)
>;

class RcpPat<Instruction RcpInst, ValueType vt> : Pat <
  (fdiv FP_ONE, vt:$src),
  (RcpInst $src)
>;

class RsqPat<Instruction RsqInst, ValueType vt> : Pat <
  (AMDGPUrcp (fsqrt vt:$src)),
  (RsqInst $src)
>;

include "R600Instructions.td"
include "R700Instructions.td"
include "EvergreenInstructions.td"
include "CaymanInstructions.td"

include "SIInstrInfo.td"

