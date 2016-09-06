//===-- FLATInstructions.td - FLAT Instruction Defintions -----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

def FLATAtomic : ComplexPattern<i64, 3, "SelectFlat">;

//===----------------------------------------------------------------------===//
// FLAT classes
//===----------------------------------------------------------------------===//

class FLAT_Pseudo<string opName, dag outs, dag ins,
                  string asmOps, list<dag> pattern=[]> :
  InstSI<outs, ins, "", pattern>,
  SIMCInstr<opName, SIEncodingFamily.NONE> {

  let isPseudo = 1;
  let isCodeGenOnly = 1;

  let SubtargetPredicate = isCIVI;

  let FLAT = 1;
  // Internally, FLAT instruction are executed as both an LDS and a
  // Buffer instruction; so, they increment both VM_CNT and LGKM_CNT
  // and are not considered done until both have been decremented.
  let VM_CNT = 1;
  let LGKM_CNT = 1;

  let Uses = [EXEC, FLAT_SCR]; // M0

  let UseNamedOperandTable = 1;
  let hasSideEffects = 0;
  let SchedRW = [WriteVMEM];

  string Mnemonic = opName;
  string AsmOperands = asmOps;

  bits<1> has_vdst = 1;
  bits<1> has_data = 1;
  bits<1> has_glc  = 1;
  bits<1> glcValue = 0;
}

class FLAT_Real <bits<7> op, FLAT_Pseudo ps> :
  InstSI <ps.OutOperandList, ps.InOperandList, ps.Mnemonic # ps.AsmOperands, []>,
  Enc64 {

  let isPseudo = 0;
  let isCodeGenOnly = 0;

  // copy relevant pseudo op flags
  let SubtargetPredicate = ps.SubtargetPredicate;
  let AsmMatchConverter  = ps.AsmMatchConverter;

  // encoding fields
  bits<8> addr;
  bits<8> data;
  bits<8> vdst;
  bits<1> slc;
  bits<1> glc;
  bits<1> tfe;

  // 15-0 is reserved.
  let Inst{16}    = !if(ps.has_glc, glc, ps.glcValue);
  let Inst{17}    = slc;
  let Inst{24-18} = op;
  let Inst{31-26} = 0x37; // Encoding.
  let Inst{39-32} = addr;
  let Inst{47-40} = !if(ps.has_data, data, ?);
  // 54-48 is reserved.
  let Inst{55}    = tfe;
  let Inst{63-56} = !if(ps.has_vdst, vdst, ?);
}

class FLAT_Load_Pseudo <string opName, RegisterClass regClass> : FLAT_Pseudo<
  opName,
  (outs regClass:$vdst),
  (ins VReg_64:$addr, glc:$glc, slc:$slc, tfe:$tfe),
  " $vdst, $addr$glc$slc$tfe"> {
  let has_data = 0;
  let mayLoad = 1;
}

class FLAT_Store_Pseudo <string opName, RegisterClass vdataClass> : FLAT_Pseudo<
  opName,
  (outs),
  (ins VReg_64:$addr, vdataClass:$data, glc:$glc, slc:$slc, tfe:$tfe),
  " $addr, $data$glc$slc$tfe"> {
  let mayLoad  = 0;
  let mayStore = 1;
  let has_vdst = 0;
}

multiclass FLAT_Atomic_Pseudo<
  string opName,
  RegisterClass vdst_rc,
  ValueType vt,
  SDPatternOperator atomic = null_frag,
  ValueType data_vt = vt,
  RegisterClass data_rc = vdst_rc> {

  def "" : FLAT_Pseudo <opName,
    (outs),
    (ins VReg_64:$addr, data_rc:$data, slc:$slc, tfe:$tfe),
    " $addr, $data$slc$tfe",
    []>,
    AtomicNoRet <NAME, 0> {
    let mayLoad = 1;
    let mayStore = 1;
    let has_glc  = 0;
    let glcValue = 0;
    let has_vdst = 0;
    let PseudoInstr = NAME;
  }

  def _RTN : FLAT_Pseudo <opName,
    (outs vdst_rc:$vdst),
    (ins VReg_64:$addr, data_rc:$data, slc:$slc, tfe:$tfe),
    " $vdst, $addr, $data glc$slc$tfe",
    [(set vt:$vdst,
      (atomic (FLATAtomic i64:$addr, i1:$slc, i1:$tfe), data_vt:$data))]>,
    AtomicNoRet <NAME, 1> {
    let mayLoad  = 1;
    let mayStore = 1;
    let hasPostISelHook = 1;
    let has_glc  = 0;
    let glcValue = 1;
    let PseudoInstr = NAME # "_RTN";
  }
}

class flat_binary_atomic_op<SDNode atomic_op> : PatFrag<
  (ops node:$ptr, node:$value),
  (atomic_op node:$ptr, node:$value),
  [{return cast<MemSDNode>(N)->getAddressSpace() == AMDGPUAS::FLAT_ADDRESS;}]
>;

def atomic_cmp_swap_flat : flat_binary_atomic_op<AMDGPUatomic_cmp_swap>;
def atomic_swap_flat     : flat_binary_atomic_op<atomic_swap>;
def atomic_add_flat      : flat_binary_atomic_op<atomic_load_add>;
def atomic_and_flat      : flat_binary_atomic_op<atomic_load_and>;
def atomic_max_flat      : flat_binary_atomic_op<atomic_load_max>;
def atomic_min_flat      : flat_binary_atomic_op<atomic_load_min>;
def atomic_or_flat       : flat_binary_atomic_op<atomic_load_or>;
def atomic_sub_flat      : flat_binary_atomic_op<atomic_load_sub>;
def atomic_umax_flat     : flat_binary_atomic_op<atomic_load_umax>;
def atomic_umin_flat     : flat_binary_atomic_op<atomic_load_umin>;
def atomic_xor_flat      : flat_binary_atomic_op<atomic_load_xor>;
def atomic_inc_flat      : flat_binary_atomic_op<SIatomic_inc>;
def atomic_dec_flat      : flat_binary_atomic_op<SIatomic_dec>;



//===----------------------------------------------------------------------===//
// Flat Instructions
//===----------------------------------------------------------------------===//

def FLAT_LOAD_UBYTE    : FLAT_Load_Pseudo <"flat_load_ubyte", VGPR_32>;
def FLAT_LOAD_SBYTE    : FLAT_Load_Pseudo <"flat_load_sbyte", VGPR_32>;
def FLAT_LOAD_USHORT   : FLAT_Load_Pseudo <"flat_load_ushort", VGPR_32>;
def FLAT_LOAD_SSHORT   : FLAT_Load_Pseudo <"flat_load_sshort", VGPR_32>;
def FLAT_LOAD_DWORD    : FLAT_Load_Pseudo <"flat_load_dword", VGPR_32>;
def FLAT_LOAD_DWORDX2  : FLAT_Load_Pseudo <"flat_load_dwordx2", VReg_64>;
def FLAT_LOAD_DWORDX4  : FLAT_Load_Pseudo <"flat_load_dwordx4", VReg_128>;
def FLAT_LOAD_DWORDX3  : FLAT_Load_Pseudo <"flat_load_dwordx3", VReg_96>;

def FLAT_STORE_BYTE    : FLAT_Store_Pseudo <"flat_store_byte", VGPR_32>;
def FLAT_STORE_SHORT   : FLAT_Store_Pseudo <"flat_store_short", VGPR_32>;
def FLAT_STORE_DWORD   : FLAT_Store_Pseudo <"flat_store_dword", VGPR_32>;
def FLAT_STORE_DWORDX2 : FLAT_Store_Pseudo <"flat_store_dwordx2", VReg_64>;
def FLAT_STORE_DWORDX4 : FLAT_Store_Pseudo <"flat_store_dwordx4", VReg_128>;
def FLAT_STORE_DWORDX3 : FLAT_Store_Pseudo <"flat_store_dwordx3", VReg_96>;

defm FLAT_ATOMIC_CMPSWAP    : FLAT_Atomic_Pseudo <"flat_atomic_cmpswap",
                                VGPR_32, i32, atomic_cmp_swap_flat,
                                v2i32, VReg_64>;

defm FLAT_ATOMIC_CMPSWAP_X2 : FLAT_Atomic_Pseudo <"flat_atomic_cmpswap_x2",
                                VReg_64, i64, atomic_cmp_swap_flat,
                                v2i64, VReg_128>;

defm FLAT_ATOMIC_SWAP       : FLAT_Atomic_Pseudo <"flat_atomic_swap",
                                VGPR_32, i32, atomic_swap_flat>;

defm FLAT_ATOMIC_SWAP_X2    : FLAT_Atomic_Pseudo <"flat_atomic_swap_x2",
                                VReg_64, i64, atomic_swap_flat>;

defm FLAT_ATOMIC_ADD        : FLAT_Atomic_Pseudo <"flat_atomic_add",
                                VGPR_32, i32, atomic_add_flat>;

defm FLAT_ATOMIC_SUB        : FLAT_Atomic_Pseudo <"flat_atomic_sub",
                                VGPR_32, i32, atomic_sub_flat>;

defm FLAT_ATOMIC_SMIN       : FLAT_Atomic_Pseudo <"flat_atomic_smin",
                                VGPR_32, i32, atomic_min_flat>;

defm FLAT_ATOMIC_UMIN       : FLAT_Atomic_Pseudo <"flat_atomic_umin",
                                VGPR_32, i32, atomic_umin_flat>;

defm FLAT_ATOMIC_SMAX       : FLAT_Atomic_Pseudo <"flat_atomic_smax",
                                VGPR_32, i32, atomic_max_flat>;

defm FLAT_ATOMIC_UMAX       : FLAT_Atomic_Pseudo <"flat_atomic_umax",
                                VGPR_32, i32, atomic_umax_flat>;

defm FLAT_ATOMIC_AND        : FLAT_Atomic_Pseudo <"flat_atomic_and",
                                VGPR_32, i32, atomic_and_flat>;

defm FLAT_ATOMIC_OR         : FLAT_Atomic_Pseudo <"flat_atomic_or",
                                VGPR_32, i32, atomic_or_flat>;

defm FLAT_ATOMIC_XOR        : FLAT_Atomic_Pseudo <"flat_atomic_xor",
                                VGPR_32, i32, atomic_xor_flat>;

defm FLAT_ATOMIC_INC        : FLAT_Atomic_Pseudo <"flat_atomic_inc",
                                VGPR_32, i32, atomic_inc_flat>;

defm FLAT_ATOMIC_DEC        : FLAT_Atomic_Pseudo <"flat_atomic_dec",
                                VGPR_32, i32, atomic_dec_flat>;

defm FLAT_ATOMIC_ADD_X2     : FLAT_Atomic_Pseudo <"flat_atomic_add_x2",
                                VReg_64, i64, atomic_add_flat>;

defm FLAT_ATOMIC_SUB_X2     : FLAT_Atomic_Pseudo <"flat_atomic_sub_x2",
                                VReg_64, i64, atomic_sub_flat>;

defm FLAT_ATOMIC_SMIN_X2    : FLAT_Atomic_Pseudo <"flat_atomic_smin_x2",
                                VReg_64, i64, atomic_min_flat>;

defm FLAT_ATOMIC_UMIN_X2    : FLAT_Atomic_Pseudo <"flat_atomic_umin_x2",
                                VReg_64, i64, atomic_umin_flat>;

defm FLAT_ATOMIC_SMAX_X2    : FLAT_Atomic_Pseudo <"flat_atomic_smax_x2",
                                VReg_64, i64, atomic_max_flat>;

defm FLAT_ATOMIC_UMAX_X2    : FLAT_Atomic_Pseudo <"flat_atomic_umax_x2",
                                VReg_64, i64, atomic_umax_flat>;

defm FLAT_ATOMIC_AND_X2     : FLAT_Atomic_Pseudo <"flat_atomic_and_x2",
                                VReg_64, i64, atomic_and_flat>;

defm FLAT_ATOMIC_OR_X2      : FLAT_Atomic_Pseudo <"flat_atomic_or_x2",
                                VReg_64, i64, atomic_or_flat>;

defm FLAT_ATOMIC_XOR_X2     : FLAT_Atomic_Pseudo <"flat_atomic_xor_x2",
                                VReg_64, i64, atomic_xor_flat>;

defm FLAT_ATOMIC_INC_X2     : FLAT_Atomic_Pseudo <"flat_atomic_inc_x2",
                                VReg_64, i64, atomic_inc_flat>;

defm FLAT_ATOMIC_DEC_X2     : FLAT_Atomic_Pseudo <"flat_atomic_dec_x2",
                                VReg_64, i64, atomic_dec_flat>;

let SubtargetPredicate = isCI in { // CI Only flat instructions : FIXME Only?

defm FLAT_ATOMIC_FCMPSWAP    : FLAT_Atomic_Pseudo <"flat_atomic_fcmpswap",
                                VGPR_32, f32, null_frag, v2f32, VReg_64>;

defm FLAT_ATOMIC_FCMPSWAP_X2 : FLAT_Atomic_Pseudo <"flat_atomic_fcmpswap_x2",
                                VReg_64, f64, null_frag, v2f64, VReg_128>;

defm FLAT_ATOMIC_FMIN        : FLAT_Atomic_Pseudo <"flat_atomic_fmin",
                                VGPR_32, f32>;

defm FLAT_ATOMIC_FMAX        : FLAT_Atomic_Pseudo <"flat_atomic_fmax",
                                VGPR_32, f32>;

defm FLAT_ATOMIC_FMIN_X2     : FLAT_Atomic_Pseudo <"flat_atomic_fmin_x2",
                                VReg_64, f64>;

defm FLAT_ATOMIC_FMAX_X2     : FLAT_Atomic_Pseudo <"flat_atomic_fmax_x2",
                                VReg_64, f64>;

} // End SubtargetPredicate = isCI

//===----------------------------------------------------------------------===//
// Flat Patterns
//===----------------------------------------------------------------------===//

class flat_ld <SDPatternOperator ld> : PatFrag<(ops node:$ptr),
                                               (ld node:$ptr), [{
  auto const AS = cast<MemSDNode>(N)->getAddressSpace();
  return AS == AMDGPUAS::FLAT_ADDRESS ||
         AS == AMDGPUAS::GLOBAL_ADDRESS ||
         AS == AMDGPUAS::CONSTANT_ADDRESS;
}]>;

class flat_st <SDPatternOperator st> : PatFrag<(ops node:$val, node:$ptr),
                                               (st node:$val, node:$ptr), [{
  auto const AS = cast<MemSDNode>(N)->getAddressSpace();
  return AS == AMDGPUAS::FLAT_ADDRESS ||
         AS == AMDGPUAS::GLOBAL_ADDRESS;
}]>;

def atomic_flat_load   : flat_ld <atomic_load>;
def flat_load          : flat_ld <load>;
def flat_az_extloadi8  : flat_ld <az_extloadi8>;
def flat_sextloadi8    : flat_ld <sextloadi8>;
def flat_az_extloadi16 : flat_ld <az_extloadi16>;
def flat_sextloadi16   : flat_ld <sextloadi16>;

def atomic_flat_store  : flat_st <atomic_store>;
def flat_store         : flat_st <store>;
def flat_truncstorei8  : flat_st <truncstorei8>;
def flat_truncstorei16 : flat_st <truncstorei16>;

// Patterns for global loads with no offset.
class FlatLoadPat <FLAT_Pseudo inst, SDPatternOperator node, ValueType vt> : Pat <
  (vt (node i64:$addr)),
  (inst $addr, 0, 0, 0)
>;

class FlatLoadAtomicPat <FLAT_Pseudo inst, SDPatternOperator node, ValueType vt> : Pat <
  (vt (node i64:$addr)),
  (inst $addr, 1, 0, 0)
>;

class FlatStorePat <FLAT_Pseudo inst, SDPatternOperator node, ValueType vt> : Pat <
  (node vt:$data, i64:$addr),
  (inst $addr, $data, 0, 0, 0)
>;

class FlatStoreAtomicPat <FLAT_Pseudo inst, SDPatternOperator node, ValueType vt> : Pat <
  // atomic store follows atomic binop convention so the address comes
  // first.
  (node i64:$addr, vt:$data),
  (inst $addr, $data, 1, 0, 0)
>;

class FlatAtomicPat <FLAT_Pseudo inst, SDPatternOperator node, ValueType vt,
                     ValueType data_vt = vt> : Pat <
  (vt (node i64:$addr, data_vt:$data)),
  (inst $addr, $data, 0, 0)
>;

let Predicates = [isCIVI] in {

def : FlatLoadPat <FLAT_LOAD_UBYTE, flat_az_extloadi8, i32>;
def : FlatLoadPat <FLAT_LOAD_SBYTE, flat_sextloadi8, i32>;
def : FlatLoadPat <FLAT_LOAD_USHORT, flat_az_extloadi16, i32>;
def : FlatLoadPat <FLAT_LOAD_SSHORT, flat_sextloadi16, i32>;
def : FlatLoadPat <FLAT_LOAD_DWORD, flat_load, i32>;
def : FlatLoadPat <FLAT_LOAD_DWORDX2, flat_load, v2i32>;
def : FlatLoadPat <FLAT_LOAD_DWORDX4, flat_load, v4i32>;

def : FlatLoadAtomicPat <FLAT_LOAD_DWORD, atomic_flat_load, i32>;
def : FlatLoadAtomicPat <FLAT_LOAD_DWORDX2, atomic_flat_load, i64>;

def : FlatStorePat <FLAT_STORE_BYTE, flat_truncstorei8, i32>;
def : FlatStorePat <FLAT_STORE_SHORT, flat_truncstorei16, i32>;
def : FlatStorePat <FLAT_STORE_DWORD, flat_store, i32>;
def : FlatStorePat <FLAT_STORE_DWORDX2, flat_store, v2i32>;
def : FlatStorePat <FLAT_STORE_DWORDX4, flat_store, v4i32>;

def : FlatStoreAtomicPat <FLAT_STORE_DWORD, atomic_flat_store, i32>;
def : FlatStoreAtomicPat <FLAT_STORE_DWORDX2, atomic_flat_store, i64>;

def : FlatAtomicPat <FLAT_ATOMIC_ADD_RTN, atomic_add_global, i32>;
def : FlatAtomicPat <FLAT_ATOMIC_SUB_RTN, atomic_sub_global, i32>;
def : FlatAtomicPat <FLAT_ATOMIC_INC_RTN, atomic_inc_global, i32>;
def : FlatAtomicPat <FLAT_ATOMIC_DEC_RTN, atomic_dec_global, i32>;
def : FlatAtomicPat <FLAT_ATOMIC_AND_RTN, atomic_and_global, i32>;
def : FlatAtomicPat <FLAT_ATOMIC_SMAX_RTN, atomic_max_global, i32>;
def : FlatAtomicPat <FLAT_ATOMIC_UMAX_RTN, atomic_umax_global, i32>;
def : FlatAtomicPat <FLAT_ATOMIC_SMIN_RTN, atomic_min_global, i32>;
def : FlatAtomicPat <FLAT_ATOMIC_UMIN_RTN, atomic_umin_global, i32>;
def : FlatAtomicPat <FLAT_ATOMIC_OR_RTN, atomic_or_global, i32>;
def : FlatAtomicPat <FLAT_ATOMIC_SWAP_RTN, atomic_swap_global, i32>;
def : FlatAtomicPat <FLAT_ATOMIC_CMPSWAP_RTN, atomic_cmp_swap_global, i32, v2i32>;
def : FlatAtomicPat <FLAT_ATOMIC_XOR_RTN, atomic_xor_global, i32>;

def : FlatAtomicPat <FLAT_ATOMIC_ADD_X2_RTN, atomic_add_global, i64>;
def : FlatAtomicPat <FLAT_ATOMIC_SUB_X2_RTN, atomic_sub_global, i64>;
def : FlatAtomicPat <FLAT_ATOMIC_INC_X2_RTN, atomic_inc_global, i64>;
def : FlatAtomicPat <FLAT_ATOMIC_DEC_X2_RTN, atomic_dec_global, i64>;
def : FlatAtomicPat <FLAT_ATOMIC_AND_X2_RTN, atomic_and_global, i64>;
def : FlatAtomicPat <FLAT_ATOMIC_SMAX_X2_RTN, atomic_max_global, i64>;
def : FlatAtomicPat <FLAT_ATOMIC_UMAX_X2_RTN, atomic_umax_global, i64>;
def : FlatAtomicPat <FLAT_ATOMIC_SMIN_X2_RTN, atomic_min_global, i64>;
def : FlatAtomicPat <FLAT_ATOMIC_UMIN_X2_RTN, atomic_umin_global, i64>;
def : FlatAtomicPat <FLAT_ATOMIC_OR_X2_RTN, atomic_or_global, i64>;
def : FlatAtomicPat <FLAT_ATOMIC_SWAP_X2_RTN, atomic_swap_global, i64>;
def : FlatAtomicPat <FLAT_ATOMIC_CMPSWAP_X2_RTN, atomic_cmp_swap_global, i64, v2i64>;
def : FlatAtomicPat <FLAT_ATOMIC_XOR_X2_RTN, atomic_xor_global, i64>;

} // End Predicates = [isCIVI]



//===----------------------------------------------------------------------===//
// Target
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// CI
//===----------------------------------------------------------------------===//

class FLAT_Real_ci <bits<7> op, FLAT_Pseudo ps> :
  FLAT_Real <op, ps>,
  SIMCInstr <ps.PseudoInstr, SIEncodingFamily.SI> {
  let AssemblerPredicate = isCIOnly;
  let DecoderNamespace="CI";
}

def FLAT_LOAD_UBYTE_ci         : FLAT_Real_ci <0x8,  FLAT_LOAD_UBYTE>;
def FLAT_LOAD_SBYTE_ci         : FLAT_Real_ci <0x9,  FLAT_LOAD_SBYTE>;
def FLAT_LOAD_USHORT_ci        : FLAT_Real_ci <0xa,  FLAT_LOAD_USHORT>;
def FLAT_LOAD_SSHORT_ci        : FLAT_Real_ci <0xb,  FLAT_LOAD_SSHORT>;
def FLAT_LOAD_DWORD_ci         : FLAT_Real_ci <0xc,  FLAT_LOAD_DWORD>;
def FLAT_LOAD_DWORDX2_ci       : FLAT_Real_ci <0xd,  FLAT_LOAD_DWORDX2>;
def FLAT_LOAD_DWORDX4_ci       : FLAT_Real_ci <0xe,  FLAT_LOAD_DWORDX4>;
def FLAT_LOAD_DWORDX3_ci       : FLAT_Real_ci <0xf,  FLAT_LOAD_DWORDX3>;

def FLAT_STORE_BYTE_ci         : FLAT_Real_ci <0x18, FLAT_STORE_BYTE>;
def FLAT_STORE_SHORT_ci        : FLAT_Real_ci <0x1a, FLAT_STORE_SHORT>;
def FLAT_STORE_DWORD_ci        : FLAT_Real_ci <0x1c, FLAT_STORE_DWORD>;
def FLAT_STORE_DWORDX2_ci      : FLAT_Real_ci <0x1d, FLAT_STORE_DWORDX2>;
def FLAT_STORE_DWORDX4_ci      : FLAT_Real_ci <0x1e, FLAT_STORE_DWORDX4>;
def FLAT_STORE_DWORDX3_ci      : FLAT_Real_ci <0x1f, FLAT_STORE_DWORDX3>;

multiclass FLAT_Real_Atomics_ci <bits<7> op, FLAT_Pseudo ps> {
  def _ci     : FLAT_Real_ci<op, !cast<FLAT_Pseudo>(ps.PseudoInstr)>;
  def _RTN_ci : FLAT_Real_ci<op, !cast<FLAT_Pseudo>(ps.PseudoInstr # "_RTN")>;
}

defm FLAT_ATOMIC_SWAP          : FLAT_Real_Atomics_ci <0x30, FLAT_ATOMIC_SWAP>;
defm FLAT_ATOMIC_CMPSWAP       : FLAT_Real_Atomics_ci <0x31, FLAT_ATOMIC_CMPSWAP>;
defm FLAT_ATOMIC_ADD           : FLAT_Real_Atomics_ci <0x32, FLAT_ATOMIC_ADD>;
defm FLAT_ATOMIC_SUB           : FLAT_Real_Atomics_ci <0x33, FLAT_ATOMIC_SUB>;
defm FLAT_ATOMIC_SMIN          : FLAT_Real_Atomics_ci <0x35, FLAT_ATOMIC_SMIN>;
defm FLAT_ATOMIC_UMIN          : FLAT_Real_Atomics_ci <0x36, FLAT_ATOMIC_UMIN>;
defm FLAT_ATOMIC_SMAX          : FLAT_Real_Atomics_ci <0x37, FLAT_ATOMIC_SMAX>;
defm FLAT_ATOMIC_UMAX          : FLAT_Real_Atomics_ci <0x38, FLAT_ATOMIC_UMAX>;
defm FLAT_ATOMIC_AND           : FLAT_Real_Atomics_ci <0x39, FLAT_ATOMIC_AND>;
defm FLAT_ATOMIC_OR            : FLAT_Real_Atomics_ci <0x3a, FLAT_ATOMIC_OR>;
defm FLAT_ATOMIC_XOR           : FLAT_Real_Atomics_ci <0x3b, FLAT_ATOMIC_XOR>;
defm FLAT_ATOMIC_INC           : FLAT_Real_Atomics_ci <0x3c, FLAT_ATOMIC_INC>;
defm FLAT_ATOMIC_DEC           : FLAT_Real_Atomics_ci <0x3d, FLAT_ATOMIC_DEC>;
defm FLAT_ATOMIC_SWAP_X2       : FLAT_Real_Atomics_ci <0x50, FLAT_ATOMIC_SWAP_X2>;
defm FLAT_ATOMIC_CMPSWAP_X2    : FLAT_Real_Atomics_ci <0x51, FLAT_ATOMIC_CMPSWAP_X2>;
defm FLAT_ATOMIC_ADD_X2        : FLAT_Real_Atomics_ci <0x52, FLAT_ATOMIC_ADD_X2>;
defm FLAT_ATOMIC_SUB_X2        : FLAT_Real_Atomics_ci <0x53, FLAT_ATOMIC_SUB_X2>;
defm FLAT_ATOMIC_SMIN_X2       : FLAT_Real_Atomics_ci <0x55, FLAT_ATOMIC_SMIN_X2>;
defm FLAT_ATOMIC_UMIN_X2       : FLAT_Real_Atomics_ci <0x56, FLAT_ATOMIC_UMIN_X2>;
defm FLAT_ATOMIC_SMAX_X2       : FLAT_Real_Atomics_ci <0x57, FLAT_ATOMIC_SMAX_X2>;
defm FLAT_ATOMIC_UMAX_X2       : FLAT_Real_Atomics_ci <0x58, FLAT_ATOMIC_UMAX_X2>;
defm FLAT_ATOMIC_AND_X2        : FLAT_Real_Atomics_ci <0x59, FLAT_ATOMIC_AND_X2>;
defm FLAT_ATOMIC_OR_X2         : FLAT_Real_Atomics_ci <0x5a, FLAT_ATOMIC_OR_X2>;
defm FLAT_ATOMIC_XOR_X2        : FLAT_Real_Atomics_ci <0x5b, FLAT_ATOMIC_XOR_X2>;
defm FLAT_ATOMIC_INC_X2        : FLAT_Real_Atomics_ci <0x5c, FLAT_ATOMIC_INC_X2>;
defm FLAT_ATOMIC_DEC_X2        : FLAT_Real_Atomics_ci <0x5d, FLAT_ATOMIC_DEC_X2>;

// CI Only flat instructions
defm FLAT_ATOMIC_FCMPSWAP      : FLAT_Real_Atomics_ci <0x3e, FLAT_ATOMIC_FCMPSWAP>;
defm FLAT_ATOMIC_FMIN          : FLAT_Real_Atomics_ci <0x3f, FLAT_ATOMIC_FMIN>;
defm FLAT_ATOMIC_FMAX          : FLAT_Real_Atomics_ci <0x40, FLAT_ATOMIC_FMAX>;
defm FLAT_ATOMIC_FCMPSWAP_X2   : FLAT_Real_Atomics_ci <0x5e, FLAT_ATOMIC_FCMPSWAP_X2>;
defm FLAT_ATOMIC_FMIN_X2       : FLAT_Real_Atomics_ci <0x5f, FLAT_ATOMIC_FMIN_X2>;
defm FLAT_ATOMIC_FMAX_X2       : FLAT_Real_Atomics_ci <0x60, FLAT_ATOMIC_FMAX_X2>;


//===----------------------------------------------------------------------===//
// VI
//===----------------------------------------------------------------------===//

class FLAT_Real_vi <bits<7> op, FLAT_Pseudo ps> :
  FLAT_Real <op, ps>,
  SIMCInstr <ps.PseudoInstr, SIEncodingFamily.VI> {
  let AssemblerPredicate = isVI;
  let DecoderNamespace="VI";
}

def FLAT_LOAD_UBYTE_vi         : FLAT_Real_vi <0x10, FLAT_LOAD_UBYTE>;
def FLAT_LOAD_SBYTE_vi         : FLAT_Real_vi <0x11, FLAT_LOAD_SBYTE>;
def FLAT_LOAD_USHORT_vi        : FLAT_Real_vi <0x12, FLAT_LOAD_USHORT>;
def FLAT_LOAD_SSHORT_vi        : FLAT_Real_vi <0x13, FLAT_LOAD_SSHORT>;
def FLAT_LOAD_DWORD_vi         : FLAT_Real_vi <0x14, FLAT_LOAD_DWORD>;
def FLAT_LOAD_DWORDX2_vi       : FLAT_Real_vi <0x15, FLAT_LOAD_DWORDX2>;
def FLAT_LOAD_DWORDX4_vi       : FLAT_Real_vi <0x17, FLAT_LOAD_DWORDX4>;
def FLAT_LOAD_DWORDX3_vi       : FLAT_Real_vi <0x16, FLAT_LOAD_DWORDX3>;

def FLAT_STORE_BYTE_vi         : FLAT_Real_vi <0x18, FLAT_STORE_BYTE>;
def FLAT_STORE_SHORT_vi        : FLAT_Real_vi <0x1a, FLAT_STORE_SHORT>;
def FLAT_STORE_DWORD_vi        : FLAT_Real_vi <0x1c, FLAT_STORE_DWORD>;
def FLAT_STORE_DWORDX2_vi      : FLAT_Real_vi <0x1d, FLAT_STORE_DWORDX2>;
def FLAT_STORE_DWORDX4_vi      : FLAT_Real_vi <0x1f, FLAT_STORE_DWORDX4>;
def FLAT_STORE_DWORDX3_vi      : FLAT_Real_vi <0x1e, FLAT_STORE_DWORDX3>;

multiclass FLAT_Real_Atomics_vi <bits<7> op, FLAT_Pseudo ps> {
  def _vi     : FLAT_Real_vi<op, !cast<FLAT_Pseudo>(ps.PseudoInstr)>;
  def _RTN_vi : FLAT_Real_vi<op, !cast<FLAT_Pseudo>(ps.PseudoInstr # "_RTN")>;
}

defm FLAT_ATOMIC_SWAP       : FLAT_Real_Atomics_vi <0x40, FLAT_ATOMIC_SWAP>;
defm FLAT_ATOMIC_CMPSWAP    : FLAT_Real_Atomics_vi <0x41, FLAT_ATOMIC_CMPSWAP>;
defm FLAT_ATOMIC_ADD        : FLAT_Real_Atomics_vi <0x42, FLAT_ATOMIC_ADD>;
defm FLAT_ATOMIC_SUB        : FLAT_Real_Atomics_vi <0x43, FLAT_ATOMIC_SUB>;
defm FLAT_ATOMIC_SMIN       : FLAT_Real_Atomics_vi <0x44, FLAT_ATOMIC_SMIN>;
defm FLAT_ATOMIC_UMIN       : FLAT_Real_Atomics_vi <0x45, FLAT_ATOMIC_UMIN>;
defm FLAT_ATOMIC_SMAX       : FLAT_Real_Atomics_vi <0x46, FLAT_ATOMIC_SMAX>;
defm FLAT_ATOMIC_UMAX       : FLAT_Real_Atomics_vi <0x47, FLAT_ATOMIC_UMAX>;
defm FLAT_ATOMIC_AND        : FLAT_Real_Atomics_vi <0x48, FLAT_ATOMIC_AND>;
defm FLAT_ATOMIC_OR         : FLAT_Real_Atomics_vi <0x49, FLAT_ATOMIC_OR>;
defm FLAT_ATOMIC_XOR        : FLAT_Real_Atomics_vi <0x4a, FLAT_ATOMIC_XOR>;
defm FLAT_ATOMIC_INC        : FLAT_Real_Atomics_vi <0x4b, FLAT_ATOMIC_INC>;
defm FLAT_ATOMIC_DEC        : FLAT_Real_Atomics_vi <0x4c, FLAT_ATOMIC_DEC>;
defm FLAT_ATOMIC_SWAP_X2    : FLAT_Real_Atomics_vi <0x60, FLAT_ATOMIC_SWAP_X2>;
defm FLAT_ATOMIC_CMPSWAP_X2 : FLAT_Real_Atomics_vi <0x61, FLAT_ATOMIC_CMPSWAP_X2>;
defm FLAT_ATOMIC_ADD_X2     : FLAT_Real_Atomics_vi <0x62, FLAT_ATOMIC_ADD_X2>;
defm FLAT_ATOMIC_SUB_X2     : FLAT_Real_Atomics_vi <0x63, FLAT_ATOMIC_SUB_X2>;
defm FLAT_ATOMIC_SMIN_X2    : FLAT_Real_Atomics_vi <0x64, FLAT_ATOMIC_SMIN_X2>;
defm FLAT_ATOMIC_UMIN_X2    : FLAT_Real_Atomics_vi <0x65, FLAT_ATOMIC_UMIN_X2>;
defm FLAT_ATOMIC_SMAX_X2    : FLAT_Real_Atomics_vi <0x66, FLAT_ATOMIC_SMAX_X2>;
defm FLAT_ATOMIC_UMAX_X2    : FLAT_Real_Atomics_vi <0x67, FLAT_ATOMIC_UMAX_X2>;
defm FLAT_ATOMIC_AND_X2     : FLAT_Real_Atomics_vi <0x68, FLAT_ATOMIC_AND_X2>;
defm FLAT_ATOMIC_OR_X2      : FLAT_Real_Atomics_vi <0x69, FLAT_ATOMIC_OR_X2>;
defm FLAT_ATOMIC_XOR_X2     : FLAT_Real_Atomics_vi <0x6a, FLAT_ATOMIC_XOR_X2>;
defm FLAT_ATOMIC_INC_X2     : FLAT_Real_Atomics_vi <0x6b, FLAT_ATOMIC_INC_X2>;
defm FLAT_ATOMIC_DEC_X2     : FLAT_Real_Atomics_vi <0x6c, FLAT_ATOMIC_DEC_X2>;

