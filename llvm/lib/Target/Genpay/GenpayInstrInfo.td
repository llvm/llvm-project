def SDTGenpayRet : SDTypeProfile<0, 1, [SDTCisInt<0>]>;

def GenpayRet : SDNode<"GenpayISD::RET", SDTGenpayRet,
                       [SDNPHasChain, SDNPOptInGlue]>;

// Signed 16-bit immediate. (Not provided by Target.td/TargetSelectionDAG.td;
// each target defines its own immediate PatLeaves, MIPS-style.)
def immSExt16 : PatLeaf<(imm), [{
  return isInt<16>(N->getSExtValue());
}]>;

class GenpayInst<dag outs, dag ins, string asmstr, list<dag> pattern>
    : Instruction {
  let Namespace = "Genpay";
  let OutOperandList = outs;
  let InOperandList = ins;
  let AsmString = asmstr;
  let Pattern = pattern;
}

def ADDrr : GenpayInst<(outs GPR:$dst), (ins GPR:$src1, GPR:$src2),
                       "add\t$dst, $src1, $src2",
                       [(set GPR:$dst, (add GPR:$src1, GPR:$src2))]>;
def SUBrr : GenpayInst<(outs GPR:$dst), (ins GPR:$src1, GPR:$src2),
                       "sub\t$dst, $src1, $src2",
                       [(set GPR:$dst, (sub GPR:$src1, GPR:$src2))]>;
def ANDrr : GenpayInst<(outs GPR:$dst), (ins GPR:$src1, GPR:$src2),
                       "and\t$dst, $src1, $src2",
                       [(set GPR:$dst, (and GPR:$src1, GPR:$src2))]>;
def ORrr  : GenpayInst<(outs GPR:$dst), (ins GPR:$src1, GPR:$src2),
                       "or\t$dst, $src1, $src2",
                       [(set GPR:$dst, (or GPR:$src1, GPR:$src2))]>;
def XORrr : GenpayInst<(outs GPR:$dst), (ins GPR:$src1, GPR:$src2),
                       "xor\t$dst, $src1, $src2",
                       [(set GPR:$dst, (xor GPR:$src1, GPR:$src2))]>;

// Register move; also used to lower target COPYs at MC-lowering time
def MOVrr : GenpayInst<(outs GPR:$dst), (ins GPR:$src),
                       "mov\t$dst, $src",
                       [(set GPR:$dst, GPR:$src)]>;

// 16-bit signed immediates only; wider constants need LUI/OR pairs (future work)
def LI : GenpayInst<(outs GPR:$dst), (ins i64imm:$imm),
                    "li\t$dst, $imm",
                    [(set GPR:$dst, (i64 immSExt16:$imm))]>;

// Memory ops exist but are not pattern-matched yet
def LD : GenpayInst<(outs GPR:$dst), (ins GPR:$addr),
                    "ld\t$dst, [$addr]", []>;
def ST : GenpayInst<(outs), (ins GPR:$src, GPR:$addr),
                    "st\t$src, [$addr]", []>;

def RET : GenpayInst<(outs), (ins GPR:$src),
                     "ret\t$src",
                     [(GenpayRet GPR:$src)]> {
  let isReturn = 1;
  let isTerminator = 1;
  let hasCtrlDep = 1;
  let isBarrier = 1;
}
