include "llvm/Target/Target.td"

let Namespace = "SC32" in {
  def GP0 : Register<"GP0">;
  def GP1 : Register<"GP1">;
  def GP2 : Register<"GP2">;
  def GP3 : Register<"GP3">;
  def GP4 : Register<"GP4">;
  def GP5 : Register<"GP5">;
  def GP6 : Register<"GP6">;
  def GP7 : Register<"GP7">;
  def GP8 : Register<"GP8">;
  def GP9 : Register<"GP9">;
  def GP10 : Register<"GP10">;
  def GP11 : Register<"GP11">;
  def GP12 : Register<"GP12">;
  def GP13 : Register<"GP13">;
  def GP14 : Register<"GP14">;
  def GP15 : Register<"GP15">;
  def GP16 : Register<"GP16">;
  def GP17 : Register<"GP17">;
  def GP18 : Register<"GP18">;
  def GP19 : Register<"GP19">;
  def GP20 : Register<"GP20">;
  def GP21 : Register<"GP21">;
  def GP22 : Register<"GP22">;
  def GP23 : Register<"GP23">;
  def GP24 : Register<"GP24">;
  def GP25 : Register<"GP25">;
  def GP26 : Register<"GP26">;
  def GP27 : Register<"GP27">;
  def GP28 : Register<"GP28">;
  def GP29 : Register<"GP29">;
  def GP30 : Register<"GP30">;
  def GP31 : Register<"GP31">;

  def FLAGS : Register<"FLAGS">;
}

def GP : RegisterClass<"SC32", [i32], 32, (sequence "GP%u", 0, 31)>;

def CC_SC32 : CallingConv<[CCAssignToReg<[GP23, GP24, GP25, GP26]>]>;
def RetCC_SC32 : CallingConv<[CCAssignToReg<[GP28]>]>;

def CSR : CalleeSavedRegs<(add)>;

defm : RemapAllTargetPseudoPointerOperands<GP>;

def SDTSC32cmp : SDTypeProfile<0, 2, [SDTCisInt<0>, SDTCisSameAs<0, 1>]>;
def SDTSC32jmp : SDTypeProfile<0, 2, [SDTCisVT<0, OtherVT>, SDTCisVT<1, OtherVT>]>;

def SC32cmp : SDNode<"SC32ISD::CMP", SDTSC32cmp, [SDNPOutGlue]>;
def SC32jmp : SDNode<"SC32ISD::JMP", SDTSC32jmp, [SDNPHasChain, SDNPInGlue]>;

def SDT_SC32Call : SDTypeProfile<0, -1, [SDTCisVT<0, iPTR>]>;
def SDT_SC32Ret : SDTypeProfile<0, 0, []>;

def call : SDNode<"SC32ISD::CALL", SDT_SC32Call, [SDNPHasChain, SDNPOptInGlue, SDNPOutGlue, SDNPVariadic]>;
def retglue : SDNode<"SC32ISD::RET_GLUE", SDT_SC32Ret, [SDNPHasChain, SDNPOptInGlue, SDNPVariadic]>;

def LO16 : SDNodeXForm<imm, [{
  return CurDAG->getTargetConstant(N->getZExtValue() & 0xFFFF, SDLoc(N), MVT::i32);
}]>;

def HI16 : SDNodeXForm<imm, [{
  return CurDAG->getTargetConstant(N->getZExtValue() >> 16, SDLoc(N), MVT::i32);
}]>;

def LLIimm : PatLeaf<(imm), [{
  return isUInt<16>(N->getZExtValue());
}]>;

def LUIimm : PatLeaf<(imm), [{
  return isShiftedUInt<16, 16>(N->getZExtValue());
}]>;

def uimm20 : PatLeaf<(imm), [{
  return isUInt<20>(N->getZExtValue());
}]>;

def brtarget : Operand<OtherVT>;
def calltarget : Operand<i32>;

multiclass Operator<string OpStr, SDNode OpNode> {
  def "" : Instruction {
    let OutOperandList = (outs GP:$out);
    let InOperandList = (ins GP:$lhs, GP:$rhs);
    let Constraints = "$out = $lhs";
    let Pattern = [(set i32:$out, (OpNode i32:$lhs, i32:$rhs))];
    let AsmString = !strconcat(OpStr, " $lhs, $rhs");
  }

  def I : Instruction {
    let OutOperandList = (outs GP:$out);
    let InOperandList = (ins GP:$lhs, i32imm:$rhs);
    let Constraints = "$out = $lhs";
    let Pattern = [(set i32:$out, (OpNode i32:$lhs, uimm20:$rhs))];
    let AsmString = !strconcat(OpStr, "I $lhs, $rhs");
  }
}

class Jump<string OpStr, CondCode OpCC> : Instruction {
  let OutOperandList = (outs);
  let InOperandList = (ins brtarget:$tgt);
  let Pattern = [(SC32jmp OpCC, bb:$tgt)];
  let AsmString = !strconcat(OpStr, " $tgt");
  let Uses = [FLAGS];
  let isTerminator = 1;
}

let Namespace = "SC32" in {
  defm ADD : Operator<"ADD", add>;
  defm SUB : Operator<"SUB", sub>;
  defm MUL : Operator<"MUL", mul>;
  defm DIV : Operator<"DIV", udiv>;
  defm AND : Operator<"AND", and>;
  defm OR : Operator<"OR", or>;
  defm XOR : Operator<"XOR", xor>;
  defm SHL : Operator<"SHL", shl>;
  defm SHR : Operator<"SHR", srl>;
  defm SRA : Operator<"SRA", sra>;

  def JEQ : Jump<"JEQ", SETEQ>;
  def JNE : Jump<"JNE", SETNE>;
  def JLE : Jump<"JLE", SETLE>;
  def JLT : Jump<"JLT", SETLT>;
  def JGT : Jump<"JGT", SETGT>;
  def JGE : Jump<"JGE", SETGE>;
  def JLEU : Jump<"JLEU", SETULE>;
  def JLTU : Jump<"JLTU", SETULT>;
  def JGTU : Jump<"JGTU", SETUGT>;
  def JGEU : Jump<"JGEU", SETUGE>;

  def ST : Instruction {
    let OutOperandList = (outs);
    let InOperandList = (ins GP:$val, GP:$base, i32imm:$offset);
    let AsmString = "ST $val, $base, $offset";
  }

  def STB : Instruction {
    let OutOperandList = (outs);
    let InOperandList = (ins GP:$val, GP:$base, i32imm:$offset);
    let AsmString = "STB $val, $base, $offset";
  }

  def LD : Instruction {
    let OutOperandList = (outs GP:$val);
    let InOperandList = (ins GP:$base, i32imm:$offset);
    let AsmString = "LD $val, $base, $offset";
  }

  def LDB : Instruction {
    let OutOperandList = (outs GP:$val);
    let InOperandList = (ins GP:$base, i32imm:$offset);
    let AsmString = "LDB $val, $base, $offset";
  }

  def LUI : Instruction {
    let OutOperandList = (outs GP:$dst);
    let InOperandList = (ins i32imm:$val);
    let AsmString = "LUI $dst, $val";
  }

  def LLI : Instruction {
    let OutOperandList = (outs GP:$dst);
    let InOperandList = (ins i32imm:$val);
    let AsmString = "LLI $dst, $val";
  }

  def MOV : Instruction {
    let OutOperandList = (outs GP:$dst);
    let InOperandList = (ins GP:$src);
    let AsmString = "MOV $dst, $src";
  }

  def CALLI : Instruction {
    let OutOperandList = (outs);
    let InOperandList = (ins calltarget:$tgt);
    let Pattern = [(call tglobaladdr:$tgt)];
    let AsmString = "CALLI $tgt";
    let isCall = 1;
  }

  def RET : Instruction {
    let OutOperandList = (outs);
    let InOperandList = (ins);
    let Pattern = [(retglue)];
    let AsmString = "RET";
    let isTerminator = 1;
    let isReturn = 1;
  }

  def JMP : Instruction {
    let OutOperandList = (outs);
    let InOperandList = (ins brtarget:$tgt);
    let Pattern = [(br bb:$tgt)];
    let AsmString = "JMP $tgt";
    let isTerminator = 1;
  }

  def CMP : Instruction {
    let OutOperandList = (outs);
    let InOperandList = (ins GP:$lhs, GP:$rhs);
    let Pattern = [(SC32cmp i32:$lhs, i32:$rhs)];
    let AsmString = "CMP $lhs, $rhs";
    let Defs = [FLAGS];
  }

  def PUSH : Instruction {
    let OutOperandList = (outs);
    let InOperandList = (ins GP:$src);
    let AsmString = "PUSH $src";
  }

  def POP : Instruction {
    let InOperandList = (ins);
    let OutOperandList = (outs GP:$dst);
    let AsmString = "POP $dst";
  }
}

def : Pat<(i32 LLIimm:$val), (LLI imm:$val)>;

def : Pat<(i32 LUIimm:$val), (LUI (HI16 imm:$val))>;

def : Pat<(i32 imm:$val), (ORI (LUI (HI16 imm:$val)), (LO16 imm:$val))>;

def : Pat<(store i32:$val, i32:$addr), (ST i32:$val, i32:$addr, (i32 0))>;
def : Pat<(truncstorei8 i32:$val, i32:$addr), (STB i32:$val, i32:$addr, (i32 0))>;

def : Pat<(load i32:$addr), (LD i32:$addr, (i32 0))>;
def : Pat<(zextloadi8 i32:$addr), (LDB i32:$addr, (i32 0))>;
def : Pat<(extloadi8 i32:$addr), (LDB i32:$addr, (i32 0))>;

def SC32InstrInfo : InstrInfo;

def SC32 : Target {
  let InstructionSet = SC32InstrInfo;
}

def : Processor<"generic", NoItineraries, []>;
