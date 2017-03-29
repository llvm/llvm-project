//===---- SMInstructions.td - Scalar Memory Instruction Defintions --------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

def smrd_offset_8 : NamedOperandU32<"SMRDOffset8",
                                  NamedMatchClass<"SMRDOffset8">> {
  let OperandType = "OPERAND_IMMEDIATE";
}

def smrd_offset_20 : NamedOperandU32<"SMRDOffset20",
                                  NamedMatchClass<"SMRDOffset20">> {
  let OperandType = "OPERAND_IMMEDIATE";
}

//===----------------------------------------------------------------------===//
// Scalar Memory classes
//===----------------------------------------------------------------------===//

class SM_Pseudo <string opName, dag outs, dag ins, string asmOps, list<dag> pattern=[]> :
  InstSI <outs, ins, "", pattern>,
  SIMCInstr<opName, SIEncodingFamily.NONE> {
  let isPseudo = 1;
  let isCodeGenOnly = 1;

  let LGKM_CNT = 1;
  let SMRD = 1;
  let mayStore = 0;
  let mayLoad = 1;
  let hasSideEffects = 0;
  let UseNamedOperandTable = 1;
  let SchedRW = [WriteSMEM];
  let SubtargetPredicate = isGCN;

  string Mnemonic = opName;
  string AsmOperands = asmOps;

  bits<1> has_sbase = 1;
  bits<1> has_sdst = 1;
  bit has_glc = 0;
  bits<1> has_offset = 1;
  bits<1> offset_is_imm = 0;
}

class SM_Real <SM_Pseudo ps>
  : InstSI<ps.OutOperandList, ps.InOperandList, ps.Mnemonic # ps.AsmOperands, []> {

  let isPseudo = 0;
  let isCodeGenOnly = 0;

  // copy relevant pseudo op flags
  let SubtargetPredicate = ps.SubtargetPredicate;
  let AsmMatchConverter  = ps.AsmMatchConverter;

  // encoding
  bits<7>  sbase;
  bits<7>  sdst;
  bits<32> offset;
  bits<1> imm = !if(ps.has_offset, ps.offset_is_imm, 0);
}

class SM_Load_Pseudo <string opName, dag outs, dag ins, string asmOps, list<dag> pattern=[]>
  : SM_Pseudo<opName, outs, ins, asmOps, pattern> {
  RegisterClass BaseClass;
  let mayLoad = 1;
  let mayStore = 0;
  let has_glc = 1;
}

class SM_Store_Pseudo <string opName, dag ins, string asmOps, list<dag> pattern = []>
  : SM_Pseudo<opName, (outs), ins, asmOps, pattern> {
  RegisterClass BaseClass;
  RegisterClass SrcClass;
  let mayLoad = 0;
  let mayStore = 1;
  let has_glc = 1;
  let ScalarStore = 1;
}

multiclass SM_Pseudo_Loads<string opName,
                           RegisterClass baseClass,
                           RegisterClass dstClass> {
  def _IMM  : SM_Load_Pseudo <opName,
                              (outs dstClass:$sdst),
                              (ins baseClass:$sbase, i32imm:$offset, i1imm:$glc),
                              " $sdst, $sbase, $offset$glc", []> {
    let offset_is_imm = 1;
    let BaseClass = baseClass;
    let PseudoInstr = opName # "_IMM";
    let has_glc = 1;
  }

  def _SGPR  : SM_Load_Pseudo <opName,
                              (outs dstClass:$sdst),
                              (ins baseClass:$sbase, SReg_32:$soff, i1imm:$glc),
                              " $sdst, $sbase, $offset$glc", []> {
    let BaseClass = baseClass;
    let PseudoInstr = opName # "_SGPR";
    let has_glc = 1;
  }
}

multiclass SM_Pseudo_Stores<string opName,
                           RegisterClass baseClass,
                           RegisterClass srcClass> {
  def _IMM  : SM_Store_Pseudo <opName,
    (ins srcClass:$sdata, baseClass:$sbase, i32imm:$offset, i1imm:$glc),
    " $sdata, $sbase, $offset$glc", []> {
    let offset_is_imm = 1;
    let BaseClass = baseClass;
    let SrcClass = srcClass;
    let PseudoInstr = opName # "_IMM";
  }

  def _SGPR  : SM_Store_Pseudo <opName,
    (ins srcClass:$sdata, baseClass:$sbase, SReg_32:$soff, i1imm:$glc),
    " $sdata, $sbase, $offset$glc", []> {
    let BaseClass = baseClass;
    let SrcClass = srcClass;
    let PseudoInstr = opName # "_SGPR";
  }
}

class SM_Time_Pseudo<string opName, SDPatternOperator node> : SM_Pseudo<
  opName, (outs SReg_64_XEXEC:$sdst), (ins),
  " $sdst", [(set i64:$sdst, (node))]> {
  let hasSideEffects = 1;
  // FIXME: mayStore = ? is a workaround for tablegen bug for different
  // inferred mayStore flags for the instruction pattern vs. standalone
  // Pat. Each considers the other contradictory.
  let mayStore = ?;
  let mayLoad = ?;
  let has_sbase = 0;
  let has_offset = 0;
}

class SM_Inval_Pseudo <string opName, SDPatternOperator node> : SM_Pseudo<
  opName, (outs), (ins), "", [(node)]> {
  let hasSideEffects = 1;
  let mayStore = 1;
  let has_sdst = 0;
  let has_sbase = 0;
  let has_offset = 0;
}


//===----------------------------------------------------------------------===//
// Scalar Memory Instructions
//===----------------------------------------------------------------------===//

// We are using the SReg_32_XM0 and not the SReg_32 register class for 32-bit
// SMRD instructions, because the SReg_32_XM0 register class does not include M0
// and writing to M0 from an SMRD instruction will hang the GPU.

// XXX - SMEM instructions do not allow exec for data operand, but
// does sdst for SMRD on SI/CI?
defm S_LOAD_DWORD    : SM_Pseudo_Loads <"s_load_dword", SReg_64, SReg_32_XM0_XEXEC>;
defm S_LOAD_DWORDX2  : SM_Pseudo_Loads <"s_load_dwordx2", SReg_64, SReg_64_XEXEC>;
defm S_LOAD_DWORDX4  : SM_Pseudo_Loads <"s_load_dwordx4", SReg_64, SReg_128>;
defm S_LOAD_DWORDX8  : SM_Pseudo_Loads <"s_load_dwordx8", SReg_64, SReg_256>;
defm S_LOAD_DWORDX16 : SM_Pseudo_Loads <"s_load_dwordx16", SReg_64, SReg_512>;

defm S_BUFFER_LOAD_DWORD : SM_Pseudo_Loads <
  "s_buffer_load_dword", SReg_128, SReg_32_XM0_XEXEC
>;

// FIXME: exec_lo/exec_hi appear to be allowed for SMRD loads on
// SI/CI, bit disallowed for SMEM on VI.
defm S_BUFFER_LOAD_DWORDX2 : SM_Pseudo_Loads <
  "s_buffer_load_dwordx2", SReg_128, SReg_64_XEXEC
>;

defm S_BUFFER_LOAD_DWORDX4 : SM_Pseudo_Loads <
  "s_buffer_load_dwordx4", SReg_128, SReg_128
>;

defm S_BUFFER_LOAD_DWORDX8 : SM_Pseudo_Loads <
  "s_buffer_load_dwordx8", SReg_128, SReg_256
>;

defm S_BUFFER_LOAD_DWORDX16 : SM_Pseudo_Loads <
  "s_buffer_load_dwordx16", SReg_128, SReg_512
>;

defm S_STORE_DWORD : SM_Pseudo_Stores <"s_store_dword", SReg_64, SReg_32_XM0_XEXEC>;
defm S_STORE_DWORDX2 : SM_Pseudo_Stores <"s_store_dwordx2", SReg_64, SReg_64_XEXEC>;
defm S_STORE_DWORDX4 : SM_Pseudo_Stores <"s_store_dwordx4", SReg_64, SReg_128>;

defm S_BUFFER_STORE_DWORD : SM_Pseudo_Stores <
  "s_buffer_store_dword", SReg_128, SReg_32_XM0_XEXEC
>;

defm S_BUFFER_STORE_DWORDX2 : SM_Pseudo_Stores <
  "s_buffer_store_dwordx2", SReg_128, SReg_64_XEXEC
>;

defm S_BUFFER_STORE_DWORDX4 : SM_Pseudo_Stores <
  "s_buffer_store_dwordx4", SReg_128, SReg_128
>;


def S_MEMTIME : SM_Time_Pseudo <"s_memtime", int_amdgcn_s_memtime>;
def S_DCACHE_INV : SM_Inval_Pseudo <"s_dcache_inv", int_amdgcn_s_dcache_inv>;

let SubtargetPredicate = isCIVI in {
def S_DCACHE_INV_VOL : SM_Inval_Pseudo <"s_dcache_inv_vol", int_amdgcn_s_dcache_inv_vol>;
} // let SubtargetPredicate = isCIVI

let SubtargetPredicate = isVI in {
def S_DCACHE_WB     : SM_Inval_Pseudo <"s_dcache_wb", int_amdgcn_s_dcache_wb>;
def S_DCACHE_WB_VOL : SM_Inval_Pseudo <"s_dcache_wb_vol", int_amdgcn_s_dcache_wb_vol>;
def S_MEMREALTIME   : SM_Time_Pseudo <"s_memrealtime", int_amdgcn_s_memrealtime>;
} // SubtargetPredicate = isVI



//===----------------------------------------------------------------------===//
// Scalar Memory Patterns
//===----------------------------------------------------------------------===//


def smrd_load : PatFrag <(ops node:$ptr), (load node:$ptr), [{
  auto Ld = cast<LoadSDNode>(N);
  return Ld->getAlignment() >= 4  &&
    ((Ld->getAddressSpace() == AMDGPUASI.CONSTANT_ADDRESS &&
    static_cast<const SITargetLowering *>(getTargetLowering())->isMemOpUniform(N)) ||
    (Subtarget->getScalarizeGlobalBehavior() && Ld->getAddressSpace() == AMDGPUASI.GLOBAL_ADDRESS &&
    static_cast<const SITargetLowering *>(getTargetLowering())->isMemOpUniform(N) &&
    static_cast<const SITargetLowering *>(getTargetLowering())->isMemOpHasNoClobberedMemOperand(N)));
}]>;

def SMRDImm         : ComplexPattern<i64, 2, "SelectSMRDImm">;
def SMRDImm32       : ComplexPattern<i64, 2, "SelectSMRDImm32">;
def SMRDSgpr        : ComplexPattern<i64, 2, "SelectSMRDSgpr">;
def SMRDBufferImm   : ComplexPattern<i32, 1, "SelectSMRDBufferImm">;
def SMRDBufferImm32 : ComplexPattern<i32, 1, "SelectSMRDBufferImm32">;
def SMRDBufferSgpr  : ComplexPattern<i32, 1, "SelectSMRDBufferSgpr">;

let Predicates = [isGCN] in {

multiclass SMRD_Pattern <string Instr, ValueType vt> {

  // 1. IMM offset
  def : Pat <
    (smrd_load (SMRDImm i64:$sbase, i32:$offset)),
    (vt (!cast<SM_Pseudo>(Instr#"_IMM") $sbase, $offset, 0))
  >;

  // 2. SGPR offset
  def : Pat <
    (smrd_load (SMRDSgpr i64:$sbase, i32:$offset)),
    (vt (!cast<SM_Pseudo>(Instr#"_SGPR") $sbase, $offset, 0))
  >;
}

let Predicates = [isSICI] in {
def : Pat <
  (i64 (readcyclecounter)),
  (S_MEMTIME)
>;
}

// Global and constant loads can be selected to either MUBUF or SMRD
// instructions, but SMRD instructions are faster so we want the instruction
// selector to prefer those.
let AddedComplexity = 100 in {

defm : SMRD_Pattern <"S_LOAD_DWORD",    i32>;
defm : SMRD_Pattern <"S_LOAD_DWORDX2",  v2i32>;
defm : SMRD_Pattern <"S_LOAD_DWORDX4",  v4i32>;
defm : SMRD_Pattern <"S_LOAD_DWORDX8",  v8i32>;
defm : SMRD_Pattern <"S_LOAD_DWORDX16", v16i32>;

// 1. Offset as an immediate
def SM_LOAD_PATTERN : Pat <  // name this pattern to reuse AddedComplexity on CI
  (SIload_constant v4i32:$sbase, (SMRDBufferImm i32:$offset)),
  (S_BUFFER_LOAD_DWORD_IMM $sbase, $offset, 0)
>;

// 2. Offset loaded in an 32bit SGPR
def : Pat <
  (SIload_constant v4i32:$sbase, (SMRDBufferSgpr i32:$offset)),
  (S_BUFFER_LOAD_DWORD_SGPR $sbase, $offset, 0)
>;

} // End let AddedComplexity = 100

} // let Predicates = [isGCN]

let Predicates = [isVI] in {

def : Pat <
  (i64 (readcyclecounter)),
  (S_MEMREALTIME)
>;

} // let Predicates = [isVI]


//===----------------------------------------------------------------------===//
// Targets
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// SI
//===----------------------------------------------------------------------===//

class SMRD_Real_si <bits<5> op, SM_Pseudo ps>
  : SM_Real<ps>
  , SIMCInstr<ps.PseudoInstr, SIEncodingFamily.SI>
  , Enc32 {

  let AssemblerPredicates = [isSICI];
  let DecoderNamespace = "SICI";

  let Inst{7-0}   = !if(ps.has_offset, offset{7-0}, ?);
  let Inst{8}     = imm;
  let Inst{14-9}  = !if(ps.has_sbase, sbase{6-1}, ?);
  let Inst{21-15} = !if(ps.has_sdst, sdst{6-0}, ?);
  let Inst{26-22} = op;
  let Inst{31-27} = 0x18; //encoding
}

// FIXME: Assembler should reject trying to use glc on SMRD
// instructions on SI.
multiclass SM_Real_Loads_si<bits<5> op, string ps,
                            SM_Load_Pseudo immPs = !cast<SM_Load_Pseudo>(ps#_IMM),
                            SM_Load_Pseudo sgprPs = !cast<SM_Load_Pseudo>(ps#_SGPR)> {

  def _IMM_si : SMRD_Real_si <op, immPs> {
    let InOperandList = (ins immPs.BaseClass:$sbase, smrd_offset_8:$offset, GLC:$glc);
  }

  // FIXME: The operand name $offset is inconsistent with $soff used
  // in the pseudo
  def _SGPR_si : SMRD_Real_si <op, sgprPs> {
    let InOperandList = (ins sgprPs.BaseClass:$sbase, SReg_32:$offset, GLC:$glc);
  }

}

defm S_LOAD_DWORD           : SM_Real_Loads_si <0x00, "S_LOAD_DWORD">;
defm S_LOAD_DWORDX2         : SM_Real_Loads_si <0x01, "S_LOAD_DWORDX2">;
defm S_LOAD_DWORDX4         : SM_Real_Loads_si <0x02, "S_LOAD_DWORDX4">;
defm S_LOAD_DWORDX8         : SM_Real_Loads_si <0x03, "S_LOAD_DWORDX8">;
defm S_LOAD_DWORDX16        : SM_Real_Loads_si <0x04, "S_LOAD_DWORDX16">;
defm S_BUFFER_LOAD_DWORD    : SM_Real_Loads_si <0x08, "S_BUFFER_LOAD_DWORD">;
defm S_BUFFER_LOAD_DWORDX2  : SM_Real_Loads_si <0x09, "S_BUFFER_LOAD_DWORDX2">;
defm S_BUFFER_LOAD_DWORDX4  : SM_Real_Loads_si <0x0a, "S_BUFFER_LOAD_DWORDX4">;
defm S_BUFFER_LOAD_DWORDX8  : SM_Real_Loads_si <0x0b, "S_BUFFER_LOAD_DWORDX8">;
defm S_BUFFER_LOAD_DWORDX16 : SM_Real_Loads_si <0x0c, "S_BUFFER_LOAD_DWORDX16">;

def S_MEMTIME_si    : SMRD_Real_si <0x1e, S_MEMTIME>;
def S_DCACHE_INV_si : SMRD_Real_si <0x1f, S_DCACHE_INV>;


//===----------------------------------------------------------------------===//
// VI
//===----------------------------------------------------------------------===//

class SMEM_Real_vi <bits<8> op, SM_Pseudo ps>
  : SM_Real<ps>
  , SIMCInstr<ps.PseudoInstr, SIEncodingFamily.VI>
  , Enc64 {
  bit glc;

  let AssemblerPredicates = [isVI];
  let DecoderNamespace = "VI";

  let Inst{5-0}   = !if(ps.has_sbase, sbase{6-1}, ?);
  let Inst{12-6}  = !if(ps.has_sdst, sdst{6-0}, ?);

  let Inst{16} = !if(ps.has_glc, glc, ?);
  let Inst{17} = imm;
  let Inst{25-18} = op;
  let Inst{31-26} = 0x30; //encoding
  let Inst{51-32} = !if(ps.has_offset, offset{19-0}, ?);
}

multiclass SM_Real_Loads_vi<bits<8> op, string ps,
                            SM_Load_Pseudo immPs = !cast<SM_Load_Pseudo>(ps#_IMM),
                            SM_Load_Pseudo sgprPs = !cast<SM_Load_Pseudo>(ps#_SGPR)> {
  def _IMM_vi : SMEM_Real_vi <op, immPs> {
    let InOperandList = (ins immPs.BaseClass:$sbase, smrd_offset_20:$offset, GLC:$glc);
  }
  def _SGPR_vi : SMEM_Real_vi <op, sgprPs> {
    let InOperandList = (ins sgprPs.BaseClass:$sbase, SReg_32:$offset, GLC:$glc);
  }
}

class SMEM_Real_Store_vi <bits<8> op, SM_Pseudo ps> : SMEM_Real_vi <op, ps> {
  // encoding
  bits<7> sdata;

  let sdst = ?;
  let Inst{12-6}  = !if(ps.has_sdst, sdata{6-0}, ?);
}

multiclass SM_Real_Stores_vi<bits<8> op, string ps,
                            SM_Store_Pseudo immPs = !cast<SM_Store_Pseudo>(ps#_IMM),
                            SM_Store_Pseudo sgprPs = !cast<SM_Store_Pseudo>(ps#_SGPR)> {
  // FIXME: The operand name $offset is inconsistent with $soff used
  // in the pseudo
  def _IMM_vi : SMEM_Real_Store_vi <op, immPs> {
    let InOperandList = (ins immPs.SrcClass:$sdata, immPs.BaseClass:$sbase, smrd_offset_20:$offset, GLC:$glc);
  }

  def _SGPR_vi : SMEM_Real_Store_vi <op, sgprPs> {
    let InOperandList = (ins sgprPs.SrcClass:$sdata, sgprPs.BaseClass:$sbase, SReg_32:$offset, GLC:$glc);
  }
}

defm S_LOAD_DWORD           : SM_Real_Loads_vi <0x00, "S_LOAD_DWORD">;
defm S_LOAD_DWORDX2         : SM_Real_Loads_vi <0x01, "S_LOAD_DWORDX2">;
defm S_LOAD_DWORDX4         : SM_Real_Loads_vi <0x02, "S_LOAD_DWORDX4">;
defm S_LOAD_DWORDX8         : SM_Real_Loads_vi <0x03, "S_LOAD_DWORDX8">;
defm S_LOAD_DWORDX16        : SM_Real_Loads_vi <0x04, "S_LOAD_DWORDX16">;
defm S_BUFFER_LOAD_DWORD    : SM_Real_Loads_vi <0x08, "S_BUFFER_LOAD_DWORD">;
defm S_BUFFER_LOAD_DWORDX2  : SM_Real_Loads_vi <0x09, "S_BUFFER_LOAD_DWORDX2">;
defm S_BUFFER_LOAD_DWORDX4  : SM_Real_Loads_vi <0x0a, "S_BUFFER_LOAD_DWORDX4">;
defm S_BUFFER_LOAD_DWORDX8  : SM_Real_Loads_vi <0x0b, "S_BUFFER_LOAD_DWORDX8">;
defm S_BUFFER_LOAD_DWORDX16 : SM_Real_Loads_vi <0x0c, "S_BUFFER_LOAD_DWORDX16">;

defm S_STORE_DWORD : SM_Real_Stores_vi <0x10, "S_STORE_DWORD">;
defm S_STORE_DWORDX2 : SM_Real_Stores_vi <0x11, "S_STORE_DWORDX2">;
defm S_STORE_DWORDX4 : SM_Real_Stores_vi <0x12, "S_STORE_DWORDX4">;

defm S_BUFFER_STORE_DWORD    : SM_Real_Stores_vi <0x18, "S_BUFFER_STORE_DWORD">;
defm S_BUFFER_STORE_DWORDX2  : SM_Real_Stores_vi <0x19, "S_BUFFER_STORE_DWORDX2">;
defm S_BUFFER_STORE_DWORDX4  : SM_Real_Stores_vi <0x1a, "S_BUFFER_STORE_DWORDX4">;

// These instructions use same encoding
def S_DCACHE_INV_vi         : SMEM_Real_vi <0x20, S_DCACHE_INV>;
def S_DCACHE_WB_vi          : SMEM_Real_vi <0x21, S_DCACHE_WB>;
def S_DCACHE_INV_VOL_vi     : SMEM_Real_vi <0x22, S_DCACHE_INV_VOL>;
def S_DCACHE_WB_VOL_vi      : SMEM_Real_vi <0x23, S_DCACHE_WB_VOL>;
def S_MEMTIME_vi            : SMEM_Real_vi <0x24, S_MEMTIME>;
def S_MEMREALTIME_vi        : SMEM_Real_vi <0x25, S_MEMREALTIME>;


//===----------------------------------------------------------------------===//
// CI
//===----------------------------------------------------------------------===//

def smrd_literal_offset : NamedOperandU32<"SMRDLiteralOffset",
                                          NamedMatchClass<"SMRDLiteralOffset">> {
  let OperandType = "OPERAND_IMMEDIATE";
}

class SMRD_Real_Load_IMM_ci <bits<5> op, SM_Load_Pseudo ps> :
  SM_Real<ps>,
  Enc64 {

  let AssemblerPredicates = [isCIOnly];
  let DecoderNamespace = "CI";
  let InOperandList = (ins ps.BaseClass:$sbase, smrd_literal_offset:$offset, GLC:$glc);

  let LGKM_CNT = ps.LGKM_CNT;
  let SMRD = ps.SMRD;
  let mayLoad = ps.mayLoad;
  let mayStore = ps.mayStore;
  let hasSideEffects = ps.hasSideEffects;
  let SchedRW = ps.SchedRW;
  let UseNamedOperandTable = ps.UseNamedOperandTable;

  let Inst{7-0}   = 0xff;
  let Inst{8}     = 0;
  let Inst{14-9}  = sbase{6-1};
  let Inst{21-15} = sdst{6-0};
  let Inst{26-22} = op;
  let Inst{31-27} = 0x18; //encoding
  let Inst{63-32} = offset{31-0};
}

def S_LOAD_DWORD_IMM_ci           : SMRD_Real_Load_IMM_ci <0x00, S_LOAD_DWORD_IMM>;
def S_LOAD_DWORDX2_IMM_ci         : SMRD_Real_Load_IMM_ci <0x01, S_LOAD_DWORDX2_IMM>;
def S_LOAD_DWORDX4_IMM_ci         : SMRD_Real_Load_IMM_ci <0x02, S_LOAD_DWORDX4_IMM>;
def S_LOAD_DWORDX8_IMM_ci         : SMRD_Real_Load_IMM_ci <0x03, S_LOAD_DWORDX8_IMM>;
def S_LOAD_DWORDX16_IMM_ci        : SMRD_Real_Load_IMM_ci <0x04, S_LOAD_DWORDX16_IMM>;
def S_BUFFER_LOAD_DWORD_IMM_ci    : SMRD_Real_Load_IMM_ci <0x08, S_BUFFER_LOAD_DWORD_IMM>;
def S_BUFFER_LOAD_DWORDX2_IMM_ci  : SMRD_Real_Load_IMM_ci <0x09, S_BUFFER_LOAD_DWORDX2_IMM>;
def S_BUFFER_LOAD_DWORDX4_IMM_ci  : SMRD_Real_Load_IMM_ci <0x0a, S_BUFFER_LOAD_DWORDX4_IMM>;
def S_BUFFER_LOAD_DWORDX8_IMM_ci  : SMRD_Real_Load_IMM_ci <0x0b, S_BUFFER_LOAD_DWORDX8_IMM>;
def S_BUFFER_LOAD_DWORDX16_IMM_ci : SMRD_Real_Load_IMM_ci <0x0c, S_BUFFER_LOAD_DWORDX16_IMM>;

class SMRD_Real_ci <bits<5> op, SM_Pseudo ps>
  : SM_Real<ps>
  , SIMCInstr<ps.PseudoInstr, SIEncodingFamily.SI>
  , Enc32 {

  let AssemblerPredicates = [isCIOnly];
  let DecoderNamespace = "CI";

  let Inst{7-0}   = !if(ps.has_offset, offset{7-0}, ?);
  let Inst{8}     = imm;
  let Inst{14-9}  = !if(ps.has_sbase, sbase{6-1}, ?);
  let Inst{21-15} = !if(ps.has_sdst, sdst{6-0}, ?);
  let Inst{26-22} = op;
  let Inst{31-27} = 0x18; //encoding
}

def S_DCACHE_INV_VOL_ci : SMRD_Real_ci <0x1d, S_DCACHE_INV_VOL>;

let AddedComplexity = SM_LOAD_PATTERN.AddedComplexity in {

class SMRD_Pattern_ci <string Instr, ValueType vt> : Pat <
  (smrd_load (SMRDImm32 i64:$sbase, i32:$offset)),
  (vt (!cast<SM_Pseudo>(Instr#"_IMM_ci") $sbase, $offset, 0))> {
  let Predicates = [isCIOnly];
}

def : SMRD_Pattern_ci <"S_LOAD_DWORD",    i32>;
def : SMRD_Pattern_ci <"S_LOAD_DWORDX2",  v2i32>;
def : SMRD_Pattern_ci <"S_LOAD_DWORDX4",  v4i32>;
def : SMRD_Pattern_ci <"S_LOAD_DWORDX8",  v8i32>;
def : SMRD_Pattern_ci <"S_LOAD_DWORDX16", v16i32>;

def : Pat <
  (SIload_constant v4i32:$sbase, (SMRDBufferImm32 i32:$offset)),
  (S_BUFFER_LOAD_DWORD_IMM_ci $sbase, $offset, 0)> {
  let Predicates = [isCI]; // should this be isCIOnly?
}

} // End let AddedComplexity = SM_LOAD_PATTERN.AddedComplexity

