//===- MipsEVAInstrInfo.td - EVA ASE instructions -*- tablegen ------------*-=//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file describes Mips EVA ASE instructions.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
//
// Instruction encodings
//
//===----------------------------------------------------------------------===//

// Memory Load/Store EVA encodings
class LBE_ENC     : SPECIAL3_EVA_LOAD_STORE_FM<OPCODE6_LBE>;
class LBuE_ENC    : SPECIAL3_EVA_LOAD_STORE_FM<OPCODE6_LBuE>;
class LHE_ENC     : SPECIAL3_EVA_LOAD_STORE_FM<OPCODE6_LHE>;
class LHuE_ENC    : SPECIAL3_EVA_LOAD_STORE_FM<OPCODE6_LHuE>;
class LWE_ENC     : SPECIAL3_EVA_LOAD_STORE_FM<OPCODE6_LWE>;

class SBE_ENC     : SPECIAL3_EVA_LOAD_STORE_FM<OPCODE6_SBE>;
class SHE_ENC     : SPECIAL3_EVA_LOAD_STORE_FM<OPCODE6_SHE>;
class SWE_ENC     : SPECIAL3_EVA_LOAD_STORE_FM<OPCODE6_SWE>;

// load/store left/right EVA encodings
class LWLE_ENC    : SPECIAL3_EVA_LOAD_STORE_FM<OPCODE6_LWLE>;
class LWRE_ENC    : SPECIAL3_EVA_LOAD_STORE_FM<OPCODE6_LWRE>;
class SWLE_ENC    : SPECIAL3_EVA_LOAD_STORE_FM<OPCODE6_SWLE>;
class SWRE_ENC    : SPECIAL3_EVA_LOAD_STORE_FM<OPCODE6_SWRE>;

// Load-linked EVA, Store-conditional EVA encodings
class LLE_ENC     : SPECIAL3_EVA_LOAD_STORE_FM<OPCODE6_LLE>;
class SCE_ENC     : SPECIAL3_EVA_LOAD_STORE_FM<OPCODE6_SCE>;

class TLBINV_ENC  : TLB_FM<OPCODE6_TLBINV>;
class TLBINVF_ENC : TLB_FM<OPCODE6_TLBINVF>;

class CACHEE_ENC  : SPECIAL3_EVA_LOAD_STORE_FM<OPCODE6_CACHEE>;
class PREFE_ENC   : SPECIAL3_EVA_LOAD_STORE_FM<OPCODE6_PREFE>;

//===----------------------------------------------------------------------===//
//
// Instruction descriptions
//
//===----------------------------------------------------------------------===//

// Memory Load/Store EVA descriptions
class LOAD_EVA_DESC_BASE<string instr_asm, RegisterOperand GPROpnd,
                         InstrItinClass itin = NoItinerary> {
  dag OutOperandList = (outs GPROpnd:$rt);
  dag InOperandList = (ins mem_simm9:$addr);
  string AsmString = !strconcat(instr_asm, "\t$rt, $addr");
  list<dag> Pattern = [];
  string DecoderMethod = "DecodeMemEVA";
  bit canFoldAsLoad = 1;
  bit mayLoad = 1;
  InstrItinClass Itinerary = itin;
}

class LBE_DESC  : LOAD_EVA_DESC_BASE<"lbe",  GPR32Opnd, II_LBE>;
class LBuE_DESC : LOAD_EVA_DESC_BASE<"lbue", GPR32Opnd, II_LBUE>;
class LHE_DESC  : LOAD_EVA_DESC_BASE<"lhe",  GPR32Opnd, II_LHE>;
class LHuE_DESC : LOAD_EVA_DESC_BASE<"lhue", GPR32Opnd, II_LHUE>;
class LWE_DESC  : LOAD_EVA_DESC_BASE<"lwe",  GPR32Opnd, II_LWE>;

class STORE_EVA_DESC_BASE<string instr_asm, RegisterOperand GPROpnd,
                          SDPatternOperator OpNode = null_frag,
                          InstrItinClass itin = NoItinerary> {
  dag OutOperandList = (outs);
  dag InOperandList = (ins GPROpnd:$rt, mem_simm9:$addr);
  string AsmString = !strconcat(instr_asm, "\t$rt, $addr");
  list<dag> Pattern = [];
  string DecoderMethod = "DecodeMemEVA";
  bit mayStore = 1;
  InstrItinClass Itinerary = itin;
}

class SBE_DESC  : STORE_EVA_DESC_BASE<"sbe",  GPR32Opnd, null_frag, II_SBE>;
class SHE_DESC  : STORE_EVA_DESC_BASE<"she",  GPR32Opnd, null_frag, II_SHE>;
class SWE_DESC  : STORE_EVA_DESC_BASE<"swe",  GPR32Opnd, null_frag, II_SWE>;

// Load/Store Left/Right EVA descriptions
class LOAD_LEFT_RIGHT_EVA_DESC_BASE<string instr_asm, RegisterOperand GPROpnd,
                                    InstrItinClass itin = NoItinerary> {
  dag OutOperandList = (outs GPROpnd:$rt);
  dag InOperandList = (ins mem_simm9:$addr, GPROpnd:$src);
  string AsmString = !strconcat(instr_asm, "\t$rt, $addr");
  list<dag> Pattern = [];
  string DecoderMethod = "DecodeMemEVA";
  string Constraints = "$src = $rt";
  bit canFoldAsLoad = 1;
  InstrItinClass Itinerary = itin;
}

class LWLE_DESC  : LOAD_LEFT_RIGHT_EVA_DESC_BASE<"lwle",  GPR32Opnd, II_LWLE>;
class LWRE_DESC  : LOAD_LEFT_RIGHT_EVA_DESC_BASE<"lwre",  GPR32Opnd, II_LWRE>;

class STORE_LEFT_RIGHT_EVA_DESC_BASE<string instr_asm, RegisterOperand GPROpnd,
                                     InstrItinClass itin = NoItinerary> {
  dag OutOperandList = (outs);
  dag InOperandList = (ins GPROpnd:$rt, mem_simm9:$addr);
  string AsmString = !strconcat(instr_asm, "\t$rt, $addr");
  list<dag> Pattern = [];
  string DecoderMethod = "DecodeMemEVA";
  InstrItinClass Itinerary = itin;
}

class SWLE_DESC  : LOAD_LEFT_RIGHT_EVA_DESC_BASE<"swle",  GPR32Opnd, II_SWLE>;
class SWRE_DESC  : LOAD_LEFT_RIGHT_EVA_DESC_BASE<"swre",  GPR32Opnd, II_SWRE>;

// Load-linked EVA, Store-conditional EVA descriptions
class LLE_DESC_BASE<string instr_asm, RegisterOperand GPROpnd,
                    InstrItinClass itin = NoItinerary> {
  dag OutOperandList = (outs GPROpnd:$rt);
  dag InOperandList = (ins mem_simm9:$addr);
  string AsmString = !strconcat(instr_asm, "\t$rt, $addr");
  list<dag> Pattern = [];
  bit mayLoad = 1;
  string DecoderMethod = "DecodeMemEVA";
  InstrItinClass Itinerary = itin;
}

class LLE_DESC : LLE_DESC_BASE<"lle", GPR32Opnd, II_LLE>;

class SCE_DESC_BASE<string instr_asm, RegisterOperand GPROpnd,
                    InstrItinClass itin = NoItinerary> {
  dag OutOperandList = (outs GPROpnd:$dst);
  dag InOperandList = (ins GPROpnd:$rt, mem_simm9:$addr);
  string AsmString = !strconcat(instr_asm, "\t$rt, $addr");
  list<dag> Pattern = [];
  bit mayStore = 1;
  string Constraints = "$rt = $dst";
  string DecoderMethod = "DecodeMemEVA";
  InstrItinClass Itinerary = itin;
}

class SCE_DESC : SCE_DESC_BASE<"sce", GPR32Opnd, II_SCE>;

class TLB_DESC_BASE<string instr_asm, InstrItinClass itin = NoItinerary> {
  dag OutOperandList = (outs);
  dag InOperandList = (ins);
  string AsmString = instr_asm;
  list<dag> Pattern = [];
  InstrItinClass Itinerary = itin;
}

class TLBINV_DESC  : TLB_DESC_BASE<"tlbinv", II_TLBINV>;
class TLBINVF_DESC : TLB_DESC_BASE<"tlbinvf", II_TLBINVF>;

class CACHEE_DESC_BASE<string instr_asm, Operand MemOpnd,
                       InstrItinClass itin = NoItinerary> {
  dag OutOperandList = (outs);
  dag InOperandList = (ins  MemOpnd:$addr, uimm5:$hint);
  string AsmString = !strconcat(instr_asm, "\t$hint, $addr");
  list<dag> Pattern = [];
  string DecoderMethod = "DecodeCacheeOp_CacheOpR6";
  InstrItinClass Itinerary = itin;
}

class CACHEE_DESC  : CACHEE_DESC_BASE<"cachee", mem_simm9, II_CACHEE>;
class PREFE_DESC   : CACHEE_DESC_BASE<"prefe", mem_simm9, II_PREFE>;

//===----------------------------------------------------------------------===//
//
// Instruction definitions
//
//===----------------------------------------------------------------------===//

/// Load and Store EVA Instructions
def LBE     : LBE_ENC, LBE_DESC, INSN_EVA;
def LBuE    : LBuE_ENC, LBuE_DESC, INSN_EVA;
def LHE     : LHE_ENC, LHE_DESC, INSN_EVA;
def LHuE    : LHuE_ENC, LHuE_DESC, INSN_EVA;
let AdditionalPredicates = [NotInMicroMips] in {
def LWE     : LWE_ENC, LWE_DESC, INSN_EVA;
}
def SBE     : SBE_ENC, SBE_DESC, INSN_EVA;
def SHE     : SHE_ENC, SHE_DESC, INSN_EVA;
let AdditionalPredicates = [NotInMicroMips] in {
def SWE     : SWE_ENC, SWE_DESC, INSN_EVA;
}

/// load/store left/right EVA
let AdditionalPredicates = [NotInMicroMips] in {
def LWLE    : LWLE_ENC, LWLE_DESC, INSN_EVA_NOT_32R6_64R6;
def LWRE    : LWRE_ENC, LWRE_DESC, INSN_EVA_NOT_32R6_64R6;
def SWLE    : SWLE_ENC, SWLE_DESC, INSN_EVA_NOT_32R6_64R6;
def SWRE    : SWRE_ENC, SWRE_DESC, INSN_EVA_NOT_32R6_64R6;
}

/// Load-linked EVA, Store-conditional EVA
let AdditionalPredicates = [NotInMicroMips] in {
def LLE     : LLE_ENC, LLE_DESC, INSN_EVA;
def SCE     : SCE_ENC, SCE_DESC, INSN_EVA;
}

let AdditionalPredicates = [NotInMicroMips] in {
  def TLBINV  : TLBINV_ENC, TLBINV_DESC, INSN_EVA;
  def TLBINVF : TLBINVF_ENC, TLBINVF_DESC, INSN_EVA;
}

def CACHEE  : CACHEE_ENC, CACHEE_DESC, INSN_EVA;
def PREFE   : PREFE_ENC, PREFE_DESC, INSN_EVA;
