//===----------------------------------------------------------------------===//
// InArch profiles and nodes
//===----------------------------------------------------------------------===//

def SDT_InArchBrCC : SDTypeProfile<0, 4, [ SDTCisSameAs<0, 1>,
                                         SDTCisVT<2, OtherVT>,
                                         SDTCisVT<3, OtherVT>]>;


def inarch_ret      : SDNode<"InArchISD::RET", SDTNone,
                            [SDNPHasChain, SDNPOptInGlue, SDNPVariadic]>;


def inarch_brcc     : SDNode<"InArchISD::BR_CC", SDT_InArchBrCC,
                            [SDNPHasChain]>;

//===----------------------------------------------------------------------===//
// Instruction format superclass
//===----------------------------------------------------------------------===//



include "InArchInstrFormats.td"

//===----------------------------------------------------------------------===//
// InArch Operand, Complex Patterns and Transformations Definitions.
//===----------------------------------------------------------------------===//

class SImmAsmOperand<int width> : AsmOperandClass {
  let Name = "SImm" # width;
  let RenderMethod = "addImmOperands";
  let DiagnosticType = !strconcat("Invalid", Name);
}

def simm16 : Operand<i32>, ImmLeaf<i32, [{return isInt<16>(Imm);}]> {
  let ParserMatchClass = SImmAsmOperand<16>;
  let EncoderMethod = "getSImm16OpValue";
  let DecoderMethod = "decodeSImmOperand<16>";
  let MCOperandPredicate = [{
    int64_t Imm;
    if (MCOp.evaluateAsConstantImm(Imm))
      return isInt<16>(Imm);
    return MCOp.isBareSymbolRef();
  }];
  let OperandType = "OPERAND_SIMM16";
  let OperandNamespace = "InArchOp";
}

def btarget16 : Operand<OtherVT> {
  let ParserMatchClass = SImmAsmOperand<16>;
  let PrintMethod = "printBranchOperand";
  let EncoderMethod = "getBranchTarget16OpValue";
  let DecoderMethod = "decodeSImmOperand<16>";
  let MCOperandPredicate = [{
    int64_t Imm;
    if (MCOp.evaluateAsConstantImm(Imm))
      return isInt<16>(Imm);
    return MCOp.isBareSymbolRef();
  }];
  let OperandType = "OPERAND_PCREL";
}

def BaseAddr : ComplexPattern<iPTR, 1, "SelectBaseAddr">;

let hasSideEffects = 0, mayLoad = 0, mayStore = 0 in
class ArithmRR<bits<8> op, string opcodestr>
    : InArchInst<op, (outs GPR:$r1), (ins GPR:$r2, GPR:$r3_imm),
              !strconcat(opcodestr, " $r1 $r2 $r3_imm"), []>;

let hasSideEffects = 0, mayLoad = 0, mayStore = 0 in
class ArithmRI<bits<8> op, string opcodestr>
    : InArchInst<op, (outs GPR:$r1), (ins GPR:$r2, simm16:$r3_imm),
              !strconcat(opcodestr, " $r1 $r2 $r3_imm"), []>;

class PatGpr<SDPatternOperator OpNode, InArchInst Inst>
    : Pat<(OpNode GPR:$r1), (Inst GPR:$r1)>;
class PatGprGpr<SDPatternOperator OpNode, InArchInst Inst>
    : Pat<(OpNode GPR:$r1, GPR:$r2), (Inst GPR:$r1, GPR:$r2)>;
class PatGprImm<SDPatternOperator OpNode, InArchInst Inst, ImmLeaf ImmType>
    : Pat<(i32 (OpNode (i32 GPR:$r1), ImmType:$r3_imm)),
          (Inst GPR:$r1, ImmType:$r3_imm)>;

class PatBcc<CondCode Cond, InArchInst Inst>
    : Pat<(inarch_brcc GPR:$r1, GPR:$r2, Cond, bb:$r3_imm),
          (Inst GPR:$r1, GPR:$r2, btarget16:$r3_imm)>;

//===----------------------------------------------------------------------===//
// InArch Instructions
//===----------------------------------------------------------------------===//

let hasSideEffects = 0, mayLoad = 0, mayStore = 0, isBranch = 1, isTerminator = 1 in
class BranchCC<bits<8> op, string opcodestr>
    : InArchInst<op, (outs), (ins GPR:$r1, GPR:$r2, btarget16:$r3_imm),
              !strconcat(opcodestr, " $r1 $r2 $r3_imm"), []>;

let hasSideEffects = 0, mayLoad = 0, mayStore = 0, isCall = 1, isBranch = 1, isTerminator = 1, r2 = 0, r3_imm = 0 in
def BR : InArchInst<0xBB, (outs), (ins GPR:$r1), "BR $r1",
                  [(brind GPR:$r1)]>;

let isBarrier = 1, isReturn = 1, isTerminator = 1 in
def PseudoRET : InArchPseudoInst<(outs), (ins), "# PseudoRET",
                [(inarch_ret)]>,
                PseudoInstExpansion<(BR R0)>;

def BEQ  : BranchCC<0x70, "B.EQ">;
def BNE  : BranchCC<0x71, "B.NE">;
def BLE  : BranchCC<0x72, "B.LE">;
def BGT  : BranchCC<0x73, "B.GT">;
def BLEU  : BranchCC<0x74, "B.LEU">;
def BGEU  : BranchCC<0x74, "B.GEU">;
def BGTU  : BranchCC<0x75, "B.GTU">;
def BLTU  : BranchCC<0x76, "B.LTU">;

def : PatBcc<SETEQ, BEQ>;
def : PatBcc<SETNE, BNE>;
def : PatBcc<SETLE, BLE>;
def : PatBcc<SETGT, BGT>;
def : PatBcc<SETULE, BLEU>;
def : PatBcc<SETUGT, BGTU>;
def : PatBcc<SETUGE, BGEU>;
def : PatBcc<SETULT, BLTU>;

// movli
let isAsCheapAsAMove = 1, isCodeGenOnly = 1, isReMaterializable = 1, r2 = 0 in
def MOVLI : InArchInst<0xAA, (outs GPR:$r1), (ins simm16:$r3_imm),
                     "MOVl_i $r1 $r3_imm",
                     [(set GPR:$r1, simm16:$r3_imm)]>;

let hasSideEffects = 0, mayLoad = 0, mayStore = 0, isBarrier = 1, isBranch = 1, isTerminator = 1, r1 = 0, r2 = 0 in
def B : InArchInst<0xB0, (outs), (ins btarget16:$r3_imm), "B $r3_imm",
                 [(br bb:$r3_imm)]>;

let hasSideEffects = 0, mayLoad = 0, mayStore = 1 in
def STI_: InArchInst<0xCA, (outs), (ins GPR:$r1, GPR:$r2, simm16:$r3_imm),
              "STi $r1 ${r2} ${r3_imm}",
              []>;
def : Pat<(store (i32 GPR:$r1), BaseAddr:$r2),
        (STI_ GPR:$r1, BaseAddr:$r2, 0)>;

// ldi
let hasSideEffects = 0, mayLoad = 1, mayStore = 0 in
def LDI: InArchInst<0xCB, (outs GPR:$r1), (ins GPR:$r2, simm16:$r3_imm),
              "LDi $r1 ${r2} ${r3_imm}",
              []>;
def : Pat<(i32 (load BaseAddr:$r2)), (LDI BaseAddr:$r2, 0)>;

def ADD : ArithmRR<0x30, "ADD">;  def : PatGprGpr<add,  ADD>;
def SUB : ArithmRR<0x31, "SUB">;  def : PatGprGpr<sub,  SUB>;
def MUL : ArithmRR<0x32, "MUL">;  def : PatGprGpr<mul,  MUL>;
def DIV : ArithmRR<0x33, "DIV">;  def : PatGprGpr<sdiv, DIV>;
def OR  : ArithmRR<0x34, "OR">;   def : PatGprGpr<or,   OR>;
def AND : ArithmRR<0x35, "AND">;  def : PatGprGpr<and,  AND>;
def XOR : ArithmRR<0x36, "XOR">;  def : PatGprGpr<xor,  XOR>;

def ADDI : ArithmRI<0x40, "ADDi">; def : PatGprImm<add, ADDI, simm16>;
def MULI : ArithmRI<0x42, "MULi">; def : PatGprImm<mul, MULI, simm16>;
def ORI  : ArithmRI<0x44, "ORi">;  def : PatGprImm<mul, ORI,  simm16>;

// flush
let r1 = 0, r2 = 0, r3_imm = 0 in
def FLUSH : InArchInst<0xFF, (outs), (ins), "FLUSH", [(int_inarch_flush)]>;

// putpixel
def PUTPIXEL : InArchInst<0xEE, (outs), (ins GPR:$r1, GPR:$r2, GPR:$r3_imm),
                      "PUTPIXEL $r1 $r2 $r3_imm", [(int_inarch_putpixel GPR:$r1, GPR:$r2, GPR:$r3_imm)]>;

// rand
let r2 = 0, r3_imm = 0 in
def RAND : InArchInst<0xBB, (outs GPR:$r1), (ins),
                      "RAND $r1", [(set GPR:$r1, (int_inarch_rand))]>;
