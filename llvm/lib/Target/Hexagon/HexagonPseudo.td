//===--- HexagonPseudo.td -------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

let PrintMethod = "printGlobalOperand" in {
  def globaladdress : Operand<i32>;
  def globaladdressExt : Operand<i32>;
}

let isPseudo = 1 in {
let isCodeGenOnly = 0 in
def A2_iconst : Pseudo<(outs IntRegs:$Rd32),
    (ins s27_2Imm:$Ii), "${Rd32}=iconst(#${Ii})">;

def DUPLEX_Pseudo : InstHexagon<(outs),
    (ins s32_0Imm:$offset), "DUPLEX", [], "", DUPLEX, TypePSEUDO>;
}

let isExtendable = 1, opExtendable = 1, opExtentBits = 6,
    isAsmParserOnly = 1 in
def TFRI64_V2_ext : ALU64_rr<(outs DoubleRegs:$dst),
                             (ins s32_0Imm:$src1, s8_0Imm:$src2),
                             "$dst=combine(#$src1,#$src2)">;

// HI/LO Instructions
let isReMaterializable = 1, isMoveImm = 1, hasSideEffects = 0,
    hasNewValue = 1, opNewValue = 0 in
class REG_IMMED<string RegHalf, bit Rs, bits<3> MajOp, bit MinOp>
  : InstHexagon<(outs IntRegs:$dst),
              (ins u16_0Imm:$imm_value),
              "$dst"#RegHalf#"=#$imm_value", [], "", ALU32_2op_tc_1_SLOT0123, TypeALU32_2op>, OpcodeHexagon {
    bits<5> dst;
    bits<32> imm_value;

    let Inst{27} = Rs;
    let Inst{26-24} = MajOp;
    let Inst{21} = MinOp;
    let Inst{20-16} = dst;
    let Inst{23-22} = imm_value{15-14};
    let Inst{13-0} = imm_value{13-0};
}

let isAsmParserOnly = 1 in {
  def LO : REG_IMMED<".l", 0b0, 0b001, 0b1>;
  def HI : REG_IMMED<".h", 0b0, 0b010, 0b1>;
}

let isReMaterializable = 1, isMoveImm = 1, isAsmParserOnly = 1 in {
  def CONST32 : CONSTLDInst<(outs IntRegs:$Rd), (ins i32imm:$v),
                "$Rd = CONST32(#$v)", []>;
  def CONST64 : CONSTLDInst<(outs DoubleRegs:$Rd), (ins i64imm:$v),
                "$Rd = CONST64(#$v)", []>;
}

let hasSideEffects = 0, isReMaterializable = 1, isPseudo = 1,
    isCodeGenOnly = 1 in
def PS_true : SInst<(outs PredRegs:$dst), (ins), "", []>;

let hasSideEffects = 0, isReMaterializable = 1, isPseudo = 1,
    isCodeGenOnly = 1 in
def PS_false : SInst<(outs PredRegs:$dst), (ins), "", []>;

let Defs = [R29, R30], Uses = [R31, R30, R29], isPseudo = 1 in
def ADJCALLSTACKDOWN : Pseudo<(outs), (ins i32imm:$amt),
                              ".error \"should not emit\" ", []>;

let Defs = [R29, R30, R31], Uses = [R29], isPseudo = 1 in
def ADJCALLSTACKUP : Pseudo<(outs), (ins i32imm:$amt1, i32imm:$amt2),
                             ".error \"should not emit\" ", []>;


let isBranch = 1, isTerminator = 1, hasSideEffects = 0,
    Defs = [PC, LC0], Uses = [SA0, LC0] in {
def ENDLOOP0 : Endloop<(outs), (ins b30_2Imm:$offset),
                       ":endloop0",
                       []>;
}

let isBranch = 1, isTerminator = 1, hasSideEffects = 0,
    Defs = [PC, LC1], Uses = [SA1, LC1] in {
def ENDLOOP1 : Endloop<(outs), (ins b30_2Imm:$offset),
                       ":endloop1",
                       []>;
}

let isExtendable = 1, isExtentSigned = 1, opExtentBits = 9, opExtentAlign = 2,
    opExtendable = 0, hasSideEffects = 0 in
class LOOP_iBase<string mnemonic, Operand brOp, bit mustExtend = 0>
         : CRInst<(outs), (ins brOp:$offset, u10_0Imm:$src2),
           #mnemonic#"($offset,#$src2)",
           [], "" , CR_tc_3x_SLOT3> {
    bits<9> offset;
    bits<10> src2;

    let IClass = 0b0110;

    let Inst{27-22} = 0b100100;
    let Inst{21} = !if (!eq(mnemonic, "loop0"), 0b0, 0b1);
    let Inst{20-16} = src2{9-5};
    let Inst{12-8} = offset{8-4};
    let Inst{7-5} = src2{4-2};
    let Inst{4-3} = offset{3-2};
    let Inst{1-0} = src2{1-0};
}

let isExtendable = 1, isExtentSigned = 1, opExtentBits = 9, opExtentAlign = 2,
    opExtendable = 0, hasSideEffects = 0 in
class LOOP_rBase<string mnemonic, Operand brOp, bit mustExtend = 0>
         : CRInst<(outs), (ins brOp:$offset, IntRegs:$src2),
           #mnemonic#"($offset,$src2)",
           [], "" ,CR_tc_3x_SLOT3> {
    bits<9> offset;
    bits<5> src2;

    let IClass = 0b0110;

    let Inst{27-22} = 0b000000;
    let Inst{21} = !if (!eq(mnemonic, "loop0"), 0b0, 0b1);
    let Inst{20-16} = src2;
    let Inst{12-8} = offset{8-4};
    let Inst{4-3} = offset{3-2};
  }

multiclass LOOP_ri<string mnemonic> {
  let isCodeGenOnly = 1, isExtended = 1, opExtendable = 0 in {
    def iext: LOOP_iBase<mnemonic, b30_2Imm, 1>;
    def rext: LOOP_rBase<mnemonic, b30_2Imm, 1>;
  }
}


let Defs = [SA0, LC0, USR] in
defm J2_loop0 : LOOP_ri<"loop0">;

// Interestingly only loop0's appear to set usr.lpcfg
let Defs = [SA1, LC1] in
defm J2_loop1 : LOOP_ri<"loop1">;

let isCall = 1, hasSideEffects = 1, isPredicable = 0,
    isExtended = 0, isExtendable = 1, opExtendable = 0,
    isExtentSigned = 1, opExtentBits = 24, opExtentAlign = 2 in
class T_Call<string ExtStr>
  : JInst<(outs), (ins a30_2Imm:$dst),
      "call " # ExtStr # "$dst", [], "", J_tc_2early_SLOT23> {
  let BaseOpcode = "call";
  bits<24> dst;

  let IClass = 0b0101;
  let Inst{27-25} = 0b101;
  let Inst{24-16,13-1} = dst{23-2};
  let Inst{0} = 0b0;
}

let isCodeGenOnly = 1, isCall = 1, hasSideEffects = 1, Defs = [R16],
    isPredicable = 0 in
def CALLProfile :  T_Call<"">;

let isCodeGenOnly = 1, isCall = 1, hasSideEffects = 1,
    Defs = [PC, R31, R6, R7, P0] in
def PS_call_stk : T_Call<"">;

let isCall = 1, hasSideEffects = 1, cofMax1 = 1 in
class JUMPR_MISC_CALLR<bit isPred, bit isPredNot,
               dag InputDag = (ins IntRegs:$Rs)>
  : JInst<(outs), InputDag,
      !if(isPred, !if(isPredNot, "if (!$Pu) callr $Rs",
                                 "if ($Pu) callr $Rs"),
                                 "callr $Rs"),
      [], "", J_tc_2early_SLOT2> {
    bits<5> Rs;
    bits<2> Pu;
    let isPredicated = isPred;
    let isPredicatedFalse = isPredNot;

    let IClass = 0b0101;
    let Inst{27-25} = 0b000;
    let Inst{24-23} = !if (isPred, 0b10, 0b01);
    let Inst{22} = 0;
    let Inst{21} = isPredNot;
    let Inst{9-8} = !if (isPred, Pu, 0b00);
    let Inst{20-16} = Rs;

  }

let isCodeGenOnly = 1 in {
  def PS_callr_nr : JUMPR_MISC_CALLR<0, 1>; // Call, no return.
}

let isCall = 1, hasSideEffects = 1,
    isExtended = 0, isExtendable = 1, opExtendable = 0, isCodeGenOnly = 1,
    BaseOpcode = "PS_call_nr", isExtentSigned = 1, opExtentAlign = 2,
    Itinerary = J_tc_2early_SLOT23 in
class Call_nr<bits<5> nbits, bit isPred, bit isFalse, dag iops>
  : Pseudo<(outs), iops, "">, PredRel {
    bits<2> Pu;
    bits<17> dst;
    let opExtentBits = nbits;
    let isPredicable = 0;  // !if(isPred, 0, 1);
    let isPredicated = 0;  // isPred;
    let isPredicatedFalse = isFalse;
}

def PS_call_nr : Call_nr<24, 0, 0, (ins s32_0Imm:$Ii)>;
//def PS_call_nrt: Call_nr<17, 1, 0, (ins PredRegs:$Pu, s32_0Imm:$dst)>;
//def PS_call_nrf: Call_nr<17, 1, 1, (ins PredRegs:$Pu, s32_0Imm:$dst)>;

let isBranch = 1, isIndirectBranch = 1, isBarrier = 1, Defs = [PC],
    isPredicable = 1, hasSideEffects = 0, InputType = "reg",
    cofMax1 = 1 in
class T_JMPr
  :  InstHexagon<(outs), (ins IntRegs:$dst), "jumpr $dst", [],
                 "", J_tc_2early_SLOT2, TypeJ>, OpcodeHexagon {
    bits<5> dst;

    let IClass = 0b0101;
    let Inst{27-21} = 0b0010100;
    let Inst{20-16} = dst;
}

// A return through builtin_eh_return.
let isReturn = 1, isTerminator = 1, isBarrier = 1, hasSideEffects = 0,
    isCodeGenOnly = 1, Defs = [PC], Uses = [R28], isPredicable = 0 in
def EH_RETURN_JMPR : T_JMPr;

// Indirect tail-call.
let isPseudo = 1, isCall = 1, isReturn = 1, isBarrier = 1, isPredicable = 0,
    isTerminator = 1, isCodeGenOnly = 1 in
def PS_tailcall_r : T_JMPr;

//
// Direct tail-calls.
let isPseudo = 1, isCall = 1, isReturn = 1, isBarrier = 1, isPredicable = 0,
    isTerminator = 1, isCodeGenOnly = 1 in
def PS_tailcall_i : Pseudo<(outs), (ins a30_2Imm:$dst), "", []>;

let isCodeGenOnly = 1, isPseudo = 1, Uses = [R30], hasSideEffects = 0 in
def PS_aligna : Pseudo<(outs IntRegs:$Rd), (ins u32_0Imm:$A), "", []>;

// Generate frameindex addresses. The main reason for the offset operand is
// that every instruction that is allowed to have frame index as an operand
// will then have that operand followed by an immediate operand (the offset).
// This simplifies the frame-index elimination code.
//
let isMoveImm = 1, isAsCheapAsAMove = 1, isReMaterializable = 1,
    isPseudo = 1, isCodeGenOnly = 1, hasSideEffects = 0 in {
  def PS_fi  : Pseudo<(outs IntRegs:$Rd),
                         (ins IntRegs:$fi, s32_0Imm:$off), "">;
  def PS_fia : Pseudo<(outs IntRegs:$Rd),
                         (ins IntRegs:$Rs, IntRegs:$fi, s32_0Imm:$off), "">;
}

class CondStr<string CReg, bit True, bit New> {
  string S = "if (" # !if(True,"","!") # CReg # !if(New,".new","") # ") ";
}
class JumpOpcStr<string Mnemonic, bit New, bit Taken> {
  string S = Mnemonic # !if(Taken, ":t", ":nt");
}
let isBranch = 1, isIndirectBranch = 1, Defs = [PC], isPredicated = 1,
    hasSideEffects = 0, InputType = "reg", cofMax1 = 1 in
class T_JMPr_c <bit PredNot, bit isPredNew, bit isTak>
  :  InstHexagon<(outs), (ins PredRegs:$src, IntRegs:$dst),
                 CondStr<"$src", !if(PredNot,0,1), isPredNew>.S #
                 JumpOpcStr<"jumpr", isPredNew, isTak>.S # " $dst",
                 [], "", J_tc_2early_SLOT2, TypeJ>, OpcodeHexagon {

    let isTaken = isTak;
    let isPredicatedFalse = PredNot;
    let isPredicatedNew = isPredNew;
    bits<2> src;
    bits<5> dst;

    let IClass = 0b0101;

    let Inst{27-22} = 0b001101;
    let Inst{21} = PredNot;
    let Inst{20-16} = dst;
    let Inst{12} = isTak;
    let Inst{11} = isPredNew;
    let Inst{9-8} = src;
}
multiclass JMPR_Pred<bit PredNot> {
  def NAME        : T_JMPr_c<PredNot, 0, 0>; // not taken
  // Predicate new
  def NAME#newpt  : T_JMPr_c<PredNot, 1, 1>; // taken
  def NAME#new    : T_JMPr_c<PredNot, 1, 0>; // not taken
}
multiclass JMPR_base<string BaseOp> {
  let BaseOpcode = BaseOp in {
    def NAME : T_JMPr;
    defm t : JMPR_Pred<0>;
    defm f : JMPR_Pred<1>;
  }
}
let isTerminator = 1, hasSideEffects = 0, isReturn = 1, isCodeGenOnly = 1, isBarrier = 1 in
defm PS_jmpret : JMPR_base<"JMPret">, PredNewRel;

//defm V6_vtran2x2_map : HexagonMapping<(outs VectorRegs:$Vy32, VectorRegs:$Vx32), (ins VectorRegs:$Vx32in, IntRegs:$Rt32), "vtrans2x2(${Vy32},${Vx32},${Rt32})", (V6_vshuff VectorRegs:$Vy32, VectorRegs:$Vx32, VectorRegs:$Vx32in, IntRegs:$Rt32)>;

// The reason for the custom inserter is to record all ALLOCA instructions
// in MachineFunctionInfo.
let Defs = [R29], isCodeGenOnly = 1, isPseudo = 1, hasSideEffects = 1 in
def PS_alloca: InstHexagon<(outs IntRegs:$Rd),
      (ins IntRegs:$Rs, u32_0Imm:$A), "",
      [], "", ALU32_2op_tc_1_SLOT0123, TypeALU32_2op>;

// Load predicate.
let isExtendable = 1, opExtendable = 2, isExtentSigned = 1, opExtentBits = 13,
    isCodeGenOnly = 1, isPseudo = 1, hasSideEffects = 0 in
def LDriw_pred : LDInst<(outs PredRegs:$dst),
                        (ins IntRegs:$addr, s32_0Imm:$off),
                        ".error \"should not emit\"", []>;

// Load modifier.
let isExtendable = 1, opExtendable = 2, isExtentSigned = 1, opExtentBits = 13,
    isCodeGenOnly = 1, isPseudo = 1, hasSideEffects = 0 in
def LDriw_mod : LDInst<(outs ModRegs:$dst),
                        (ins IntRegs:$addr, s32_0Imm:$off),
                        ".error \"should not emit\"", []>;

// Vector load
let Predicates = [HasV60T, UseHVX] in
let mayLoad = 1, hasSideEffects = 0 in
  class V6_LDInst<dag outs, dag ins, string asmstr, list<dag> pattern = [],
                  string cstr = "", InstrItinClass itin = CVI_VM_LD,
                  IType type = TypeCVI_VM_LD>
  : InstHexagon<outs, ins, asmstr, pattern, cstr, itin, type>;

// Vector store
let Predicates = [HasV60T, UseHVX] in
let mayStore = 1, hasSideEffects = 0 in
class V6_STInst<dag outs, dag ins, string asmstr, list<dag> pattern = [],
                string cstr = "", InstrItinClass itin = CVI_VM_ST,
                IType type = TypeCVI_VM_ST>
: InstHexagon<outs, ins, asmstr, pattern, cstr, itin, type>;

let isCodeGenOnly = 1, isPseudo = 1 in
def PS_pselect : ALU64_rr<(outs DoubleRegs:$Rd),
      (ins PredRegs:$Pu, DoubleRegs:$Rs, DoubleRegs:$Rt),
      ".error \"should not emit\" ", []>;

let isBranch = 1, isBarrier = 1, Defs = [PC], hasSideEffects = 0,
    isPredicable = 1,
    isExtendable = 1, opExtendable = 0, isExtentSigned = 1,
    opExtentBits = 24, opExtentAlign = 2, InputType = "imm" in
class T_JMP<string ExtStr>
  : JInst_CJUMP_UCJUMP<(outs), (ins b30_2Imm:$dst),
      "jump " # ExtStr # "$dst",
      [], "", J_tc_2early_CJUMP_UCJUMP_ARCHDEPSLOT> {
    bits<24> dst;
    let IClass = 0b0101;

    let Inst{27-25} = 0b100;
    let Inst{24-16} = dst{23-15};
    let Inst{13-1} = dst{14-2};
}

// Restore registers and dealloc return function call.
let isCall = 1, isBarrier = 1, isReturn = 1, isTerminator = 1,
    Defs = [R29, R30, R31, PC], isPredicable = 0, isAsmParserOnly = 1 in {
  def RESTORE_DEALLOC_RET_JMP_V4 : T_JMP<"">;

  let isExtended = 1, opExtendable = 0 in
  def RESTORE_DEALLOC_RET_JMP_V4_EXT : T_JMP<"">;

  let Defs = [R14, R15, R28, R29, R30, R31, PC] in {
    def RESTORE_DEALLOC_RET_JMP_V4_PIC : T_JMP<"">;

    let isExtended = 1, opExtendable = 0 in
    def RESTORE_DEALLOC_RET_JMP_V4_EXT_PIC : T_JMP<"">;
  }
}

// Restore registers and dealloc frame before a tail call.
let isCall = 1, Defs = [R29, R30, R31, PC], isAsmParserOnly = 1 in {
  def RESTORE_DEALLOC_BEFORE_TAILCALL_V4 : T_Call<"">, PredRel;

  let isExtended = 1, opExtendable = 0 in
  def RESTORE_DEALLOC_BEFORE_TAILCALL_V4_EXT : T_Call<"">, PredRel;

  let Defs = [R14, R15, R28, R29, R30, R31, PC] in {
    def RESTORE_DEALLOC_BEFORE_TAILCALL_V4_PIC : T_Call<"">, PredRel;

    let isExtended = 1, opExtendable = 0 in
    def RESTORE_DEALLOC_BEFORE_TAILCALL_V4_EXT_PIC : T_Call<"">, PredRel;
  }
}

// Save registers function call.
let isCall = 1, Uses = [R29, R31], isAsmParserOnly = 1 in {
  def SAVE_REGISTERS_CALL_V4 : T_Call<"">, PredRel;

  let isExtended = 1, opExtendable = 0 in
  def SAVE_REGISTERS_CALL_V4_EXT : T_Call<"">, PredRel;

  let Defs = [P0] in
  def SAVE_REGISTERS_CALL_V4STK : T_Call<"">, PredRel;

  let Defs = [P0], isExtended = 1, opExtendable = 0 in
  def SAVE_REGISTERS_CALL_V4STK_EXT : T_Call<"">, PredRel;

  let Defs = [R14, R15, R28] in
  def SAVE_REGISTERS_CALL_V4_PIC : T_Call<"">, PredRel;

  let Defs = [R14, R15, R28], isExtended = 1, opExtendable = 0 in
  def SAVE_REGISTERS_CALL_V4_EXT_PIC : T_Call<"">, PredRel;

  let Defs = [R14, R15, R28, P0] in
  def SAVE_REGISTERS_CALL_V4STK_PIC : T_Call<"">, PredRel;

  let Defs = [R14, R15, R28, P0], isExtended = 1, opExtendable = 0 in
  def SAVE_REGISTERS_CALL_V4STK_EXT_PIC : T_Call<"">, PredRel;
}

// Vector load/store pseudos

let isPseudo = 1, isCodeGenOnly = 1 in
class STrivv_template<RegisterClass RC>
  : V6_STInst<(outs), (ins IntRegs:$addr, s32_0Imm:$off, RC:$src), "", []>;

def PS_vstorerw_ai: STrivv_template<VecDblRegs>,
      Requires<[HasV60T,UseHVXSgl]>;
def PS_vstorerwu_ai: STrivv_template<VecDblRegs>,
      Requires<[HasV60T,UseHVXSgl]>;
def PS_vstorerw_ai_128B: STrivv_template<VecDblRegs128B>,
      Requires<[HasV60T,UseHVXDbl]>;
def PS_vstorerwu_ai_128B: STrivv_template<VecDblRegs128B>,
      Requires<[HasV60T,UseHVXDbl]>;


let isPseudo = 1, isCodeGenOnly = 1 in
class LDrivv_template<RegisterClass RC>
  : V6_LDInst<(outs RC:$dst), (ins IntRegs:$addr, s32_0Imm:$off), "", []>;

def PS_vloadrw_ai: LDrivv_template<VecDblRegs>,
      Requires<[HasV60T,UseHVXSgl]>;
def PS_vloadrwu_ai: LDrivv_template<VecDblRegs>,
      Requires<[HasV60T,UseHVXSgl]>;
def PS_vloadrw_ai_128B: LDrivv_template<VecDblRegs128B>,
      Requires<[HasV60T,UseHVXDbl]>;
def PS_vloadrwu_ai_128B: LDrivv_template<VecDblRegs128B>,
      Requires<[HasV60T,UseHVXDbl]>;

// Store vector predicate pseudo.
let isExtendable = 1, opExtendable = 1, isExtentSigned = 1, opExtentBits = 13,
    isCodeGenOnly = 1, isPseudo = 1, mayStore = 1, hasSideEffects = 0 in {
  def PS_vstorerq_ai : STInst<(outs),
              (ins IntRegs:$base, s32_0Imm:$offset, VecPredRegs:$src1),
              ".error \"should not emit\" ", []>,
              Requires<[HasV60T,UseHVXSgl]>;

  def PS_vstorerq_ai_128B : STInst<(outs),
              (ins IntRegs:$base, s32_0Imm:$offset, VectorRegs:$src1),
              ".error \"should not emit\" ", []>,
            Requires<[HasV60T,UseHVXSgl]>;

  def PS_vloadrq_ai : STInst<(outs),
              (ins IntRegs:$base, s32_0Imm:$offset, VecPredRegs128B:$src1),
              ".error \"should not emit\" ", []>,
            Requires<[HasV60T,UseHVXDbl]>;

  def PS_vloadrq_ai_128B : STInst<(outs),
              (ins IntRegs:$base, s32_0Imm:$offset, VecPredRegs128B:$src1),
              ".error \"should not emit\" ", []>,
            Requires<[HasV60T,UseHVXDbl]>;
}

class VSELInst<dag outs, dag ins, string asmstr, list<dag> pattern = [],
              string cstr = "", InstrItinClass itin = CVI_VA_DV,
              IType type = TypeCVI_VA_DV>
  : InstHexagon<outs, ins, asmstr, pattern, cstr, itin, type>;

let isCodeGenOnly = 1, isPseudo = 1, hasSideEffects = 0 in {
  def PS_vselect: VSELInst<(outs VectorRegs:$dst),
        (ins PredRegs:$src1, VectorRegs:$src2, VectorRegs:$src3), "", []>,
        Requires<[HasV60T,UseHVXSgl]>;
  def PS_vselect_128B: VSELInst<(outs VectorRegs128B:$dst),
        (ins PredRegs:$src1, VectorRegs128B:$src2, VectorRegs128B:$src3),
        "", []>, Requires<[HasV60T,UseHVXDbl]>;
  def PS_wselect: VSELInst<(outs VecDblRegs:$dst),
        (ins PredRegs:$src1, VecDblRegs:$src2, VecDblRegs:$src3), "", []>,
        Requires<[HasV60T,UseHVXSgl]>;
  def PS_wselect_128B: VSELInst<(outs VecDblRegs128B:$dst),
        (ins PredRegs:$src1, VecDblRegs128B:$src2, VecDblRegs128B:$src3),
        "", []>, Requires<[HasV60T,UseHVXDbl]>;
}

// Store predicate.
let isExtendable = 1, opExtendable = 1, isExtentSigned = 1, opExtentBits = 13,
    isCodeGenOnly = 1, isPseudo = 1, hasSideEffects = 0 in
def STriw_pred : STInst<(outs),
      (ins IntRegs:$addr, s32_0Imm:$off, PredRegs:$src1),
      ".error \"should not emit\"", []>;
// Store modifier.
let isExtendable = 1, opExtendable = 1, isExtentSigned = 1, opExtentBits = 13,
    isCodeGenOnly = 1, isPseudo = 1, hasSideEffects = 0 in
def STriw_mod : STInst<(outs),
      (ins IntRegs:$addr, s32_0Imm:$off, ModRegs:$src1),
      ".error \"should not emit\"", []>;

let isExtendable = 1, opExtendable = 1, opExtentBits = 6,
    isAsmParserOnly = 1 in
def TFRI64_V4 : ALU64_rr<(outs DoubleRegs:$dst), (ins u64_0Imm:$src1),
                         "$dst = #$src1">;

// Hexagon doesn't have a vector multiply with C semantics.
// Instead, generate a pseudo instruction that gets expaneded into two
// scalar MPYI instructions.
// This is expanded by ExpandPostRAPseudos.
let isPseudo = 1 in
def PS_vmulw : PseudoM<(outs DoubleRegs:$Rd),
      (ins DoubleRegs:$Rs, DoubleRegs:$Rt), "", []>;

let isPseudo = 1 in
def PS_vmulw_acc : PseudoM<(outs DoubleRegs:$Rd),
      (ins DoubleRegs:$Rx, DoubleRegs:$Rs, DoubleRegs:$Rt), "", [],
      "$Rd = $Rx">;

def DuplexIClass0:  InstDuplex < 0 >;
def DuplexIClass1:  InstDuplex < 1 >;
def DuplexIClass2:  InstDuplex < 2 >;
let isExtendable = 1 in {
  def DuplexIClass3:  InstDuplex < 3 >;
  def DuplexIClass4:  InstDuplex < 4 >;
  def DuplexIClass5:  InstDuplex < 5 >;
  def DuplexIClass6:  InstDuplex < 6 >;
  def DuplexIClass7:  InstDuplex < 7 >;
}
def DuplexIClass8:  InstDuplex < 8 >;
def DuplexIClass9:  InstDuplex < 9 >;
def DuplexIClassA:  InstDuplex < 0xA >;
def DuplexIClassB:  InstDuplex < 0xB >;
def DuplexIClassC:  InstDuplex < 0xC >;
def DuplexIClassD:  InstDuplex < 0xD >;
def DuplexIClassE:  InstDuplex < 0xE >;
def DuplexIClassF:  InstDuplex < 0xF >;
