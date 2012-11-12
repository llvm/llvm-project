//===-- MBlazeInstrFPU.td - MBlaze FPU Instruction defs ----*- tablegen -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// MBlaze profiles and nodes
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// MBlaze Operand, Complex Patterns and Transformations Definitions.
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Memory Access Instructions
//===----------------------------------------------------------------------===//
class LoadFM<bits<6> op, string instr_asm, PatFrag OpNode> :
             TA<op, 0x000, (outs GPR:$dst), (ins memrr:$addr),
                !strconcat(instr_asm, "   $dst, $addr"),
                [(set (f32 GPR:$dst), (OpNode xaddr:$addr))], IIC_MEMl>;

class LoadFMI<bits<6> op, string instr_asm, PatFrag OpNode> :
              TB<op, (outs GPR:$dst), (ins memri:$addr),
                 !strconcat(instr_asm, "   $dst, $addr"),
                 [(set (f32 GPR:$dst), (OpNode iaddr:$addr))], IIC_MEMl>;

class StoreFM<bits<6> op, string instr_asm, PatFrag OpNode> :
              TA<op, 0x000, (outs), (ins GPR:$dst, memrr:$addr),
                 !strconcat(instr_asm, "   $dst, $addr"),
                 [(OpNode (f32 GPR:$dst), xaddr:$addr)], IIC_MEMs>;

class StoreFMI<bits<6> op, string instr_asm, PatFrag OpNode> :
               TB<op, (outs), (ins GPR:$dst, memrr:$addr),
                  !strconcat(instr_asm, "   $dst, $addr"),
                  [(OpNode (f32 GPR:$dst), iaddr:$addr)], IIC_MEMs>;

class ArithF<bits<6> op, bits<11> flags, string instr_asm, SDNode OpNode,
             InstrItinClass itin> :
             TA<op, flags, (outs GPR:$dst), (ins GPR:$b, GPR:$c),
                !strconcat(instr_asm, "   $dst, $b, $c"),
                [(set GPR:$dst, (OpNode GPR:$b, GPR:$c))], itin>;

class CmpFN<bits<6> op, bits<11> flags, string instr_asm,
            InstrItinClass itin> :
            TA<op, flags, (outs GPR:$dst), (ins GPR:$b, GPR:$c),
               !strconcat(instr_asm, "   $dst, $b, $c"),
               [], itin>;

class ArithFR<bits<6> op, bits<11> flags, string instr_asm, SDNode OpNode,
             InstrItinClass itin> :
             TAR<op, flags, (outs GPR:$dst), (ins GPR:$b, GPR:$c),
                 !strconcat(instr_asm, "   $dst, $c, $b"),
                 [(set GPR:$dst, (OpNode GPR:$b, GPR:$c))], itin>;

class LogicFI<bits<6> op, string instr_asm> :
             TB<op, (outs GPR:$dst), (ins GPR:$b, fimm:$c),
                !strconcat(instr_asm, "   $dst, $b, $c"),
                [], IIC_ALU>;

let rb=0 in {
  class ArithF2<bits<6> op, bits<11> flags, string instr_asm,
                InstrItinClass itin> :
                TA<op, flags, (outs GPR:$dst), (ins GPR:$b),
                   !strconcat(instr_asm, "   $dst, $b"),
                   [], itin>;

  class ArithIF<bits<6> op, bits<11> flags, string instr_asm,
                InstrItinClass itin> :
                TA<op, flags, (outs GPR:$dst), (ins GPR:$b),
                   !strconcat(instr_asm, "   $dst, $b"),
                   [], itin>;

  class ArithFI<bits<6> op, bits<11> flags, string instr_asm,
                InstrItinClass itin> :
                TA<op, flags, (outs GPR:$dst), (ins GPR:$b),
                   !strconcat(instr_asm, "   $dst, $b"),
                   [], itin>;
}

//===----------------------------------------------------------------------===//
// Pseudo instructions
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// FPU Arithmetic Instructions
//===----------------------------------------------------------------------===//
let Predicates=[HasFPU] in {
  def FORI   : LogicFI<0x28, "ori    ">;
  def FADD   :  ArithF<0x16, 0x000, "fadd   ", fadd, IIC_FPU>;
  def FRSUB  : ArithFR<0x16, 0x080, "frsub  ", fsub, IIC_FPU>;
  def FMUL   :  ArithF<0x16, 0x100, "fmul   ", fmul, IIC_FPU>;
  def FDIV   :  ArithF<0x16, 0x180, "fdiv   ", fdiv, IIC_FPUd>;
}

let Predicates=[HasFPU], isCodeGenOnly=1 in {
  def LWF    :   LoadFM<0x32, "lw      ", load>;
  def LWFI   :  LoadFMI<0x3A, "lwi     ", load>;

  def SWF    :  StoreFM<0x36, "sw      ", store>;
  def SWFI   : StoreFMI<0x3E, "swi     ", store>;
}

let Predicates=[HasFPU,HasSqrt] in {
  def FLT    : ArithIF<0x16, 0x280, "flt    ", IIC_FPUf>;
  def FINT   : ArithFI<0x16, 0x300, "fint   ", IIC_FPUi>;
  def FSQRT  : ArithF2<0x16, 0x380, "fsqrt  ", IIC_FPUs>;
}

let isAsCheapAsAMove = 1 in {
  def FCMP_UN : CmpFN<0x16, 0x200, "fcmp.un", IIC_FPUc>;
  def FCMP_LT : CmpFN<0x16, 0x210, "fcmp.lt", IIC_FPUc>;
  def FCMP_EQ : CmpFN<0x16, 0x220, "fcmp.eq", IIC_FPUc>;
  def FCMP_LE : CmpFN<0x16, 0x230, "fcmp.le", IIC_FPUc>;
  def FCMP_GT : CmpFN<0x16, 0x240, "fcmp.gt", IIC_FPUc>;
  def FCMP_NE : CmpFN<0x16, 0x250, "fcmp.ne", IIC_FPUc>;
  def FCMP_GE : CmpFN<0x16, 0x260, "fcmp.ge", IIC_FPUc>;
}


let usesCustomInserter = 1 in {
  def Select_FCC : MBlazePseudo<(outs GPR:$dst),
    (ins GPR:$T, GPR:$F, GPR:$CMP, i32imm:$CC),
    "; SELECT_FCC PSEUDO!",
    []>;
}

// Floating point conversions
let Predicates=[HasFPU] in {
  def : Pat<(sint_to_fp GPR:$V), (FLT GPR:$V)>;
  def : Pat<(fp_to_sint GPR:$V), (FINT GPR:$V)>;
  def : Pat<(fsqrt GPR:$V), (FSQRT GPR:$V)>;
}

// SET_CC operations
let Predicates=[HasFPU] in {
  def : Pat<(setcc (f32 GPR:$L), (f32 GPR:$R), SETEQ),
            (Select_CC (ADDIK (i32 R0), 1), (ADDIK (i32 R0), 0),
                       (FCMP_EQ GPR:$L, GPR:$R), 2)>;
  def : Pat<(setcc (f32 GPR:$L), (f32 GPR:$R), SETNE),
            (Select_CC (ADDIK (i32 R0), 1), (ADDIK (i32 R0), 0),
                       (FCMP_EQ GPR:$L, GPR:$R), 1)>;
  def : Pat<(setcc (f32 GPR:$L), (f32 GPR:$R), SETOEQ),
            (Select_CC (ADDIK (i32 R0), 1), (ADDIK (i32 R0), 0),
                       (FCMP_EQ GPR:$L, GPR:$R), 2)>;
 def : Pat<(setcc (f32 GPR:$L), (f32 GPR:$R), SETONE),
            (Select_CC (ADDIK (i32 R0), 1), (ADDIK (i32 R0), 0),
                       (XOR (FCMP_UN GPR:$L, GPR:$R),
                            (FCMP_EQ GPR:$L, GPR:$R)), 2)>;
  def : Pat<(setcc (f32 GPR:$L), (f32 GPR:$R), SETONE),
            (Select_CC (ADDIK (i32 R0), 1), (ADDIK (i32 R0), 0),
                       (OR (FCMP_UN GPR:$L, GPR:$R),
                           (FCMP_EQ GPR:$L, GPR:$R)), 2)>;
  def : Pat<(setcc (f32 GPR:$L), (f32 GPR:$R), SETGT),
            (Select_CC (ADDIK (i32 R0), 1), (ADDIK (i32 R0), 0),
                       (FCMP_GT GPR:$L, GPR:$R), 2)>;
  def : Pat<(setcc (f32 GPR:$L), (f32 GPR:$R), SETLT),
            (Select_CC (ADDIK (i32 R0), 1), (ADDIK (i32 R0), 0),
                       (FCMP_LT GPR:$L, GPR:$R), 2)>;
  def : Pat<(setcc (f32 GPR:$L), (f32 GPR:$R), SETGE),
            (Select_CC (ADDIK (i32 R0), 1), (ADDIK (i32 R0), 0),
                       (FCMP_GE GPR:$L, GPR:$R), 2)>;
  def : Pat<(setcc (f32 GPR:$L), (f32 GPR:$R), SETLE),
            (Select_CC (ADDIK (i32 R0), 1), (ADDIK (i32 R0), 0),
                       (FCMP_LE GPR:$L, GPR:$R), 2)>;
  def : Pat<(setcc (f32 GPR:$L), (f32 GPR:$R), SETOGT),
            (Select_CC (ADDIK (i32 R0), 1), (ADDIK (i32 R0), 0),
                       (FCMP_GT GPR:$L, GPR:$R), 2)>;
  def : Pat<(setcc (f32 GPR:$L), (f32 GPR:$R), SETOLT),
            (Select_CC (ADDIK (i32 R0), 1), (ADDIK (i32 R0), 0),
                       (FCMP_LT GPR:$L, GPR:$R), 2)>;
  def : Pat<(setcc (f32 GPR:$L), (f32 GPR:$R), SETOGE),
            (Select_CC (ADDIK (i32 R0), 1), (ADDIK (i32 R0), 0),
                       (FCMP_GE GPR:$L, GPR:$R), 2)>;
  def : Pat<(setcc (f32 GPR:$L), (f32 GPR:$R), SETOLE),
            (Select_CC (ADDIK (i32 R0), 1), (ADDIK (i32 R0), 0),
                       (FCMP_LE GPR:$L, GPR:$R), 2)>;
  def : Pat<(setcc (f32 GPR:$L), (f32 GPR:$R), SETUEQ),
            (Select_CC (ADDIK (i32 R0), 1), (ADDIK (i32 R0), 0),
                       (OR (FCMP_UN GPR:$L, GPR:$R),
                           (FCMP_EQ GPR:$L, GPR:$R)), 2)>;
  def : Pat<(setcc (f32 GPR:$L), (f32 GPR:$R), SETUNE),
            (Select_CC (ADDIK (i32 R0), 1), (ADDIK (i32 R0), 0),
                       (FCMP_NE GPR:$L, GPR:$R), 2)>;
  def : Pat<(setcc (f32 GPR:$L), (f32 GPR:$R), SETUGT),
            (Select_CC (ADDIK (i32 R0), 1), (ADDIK (i32 R0), 0),
                       (OR (FCMP_UN GPR:$L, GPR:$R),
                           (FCMP_GT GPR:$L, GPR:$R)), 2)>;
  def : Pat<(setcc (f32 GPR:$L), (f32 GPR:$R), SETULT),
            (Select_CC (ADDIK (i32 R0), 1), (ADDIK (i32 R0), 0),
                       (OR (FCMP_UN GPR:$L, GPR:$R),
                           (FCMP_LT GPR:$L, GPR:$R)), 2)>;
  def : Pat<(setcc (f32 GPR:$L), (f32 GPR:$R), SETUGE),
            (Select_CC (ADDIK (i32 R0), 1), (ADDIK (i32 R0), 0),
                       (OR (FCMP_UN GPR:$L, GPR:$R),
                           (FCMP_GE GPR:$L, GPR:$R)), 2)>;
  def : Pat<(setcc (f32 GPR:$L), (f32 GPR:$R), SETULE),
            (Select_CC (ADDIK (i32 R0), 1), (ADDIK (i32 R0), 0),
                       (OR (FCMP_UN GPR:$L, GPR:$R),
                           (FCMP_LE GPR:$L, GPR:$R)), 2)>;
  def : Pat<(setcc (f32 GPR:$L), (f32 GPR:$R), SETO),
            (Select_CC (ADDIK (i32 R0), 1), (ADDIK (i32 R0), 0),
                       (FCMP_UN GPR:$L, GPR:$R), 1)>;
  def : Pat<(setcc (f32 GPR:$L), (f32 GPR:$R), SETUO),
            (Select_CC (ADDIK (i32 R0), 1), (ADDIK (i32 R0), 0),
                       (FCMP_UN GPR:$L, GPR:$R), 2)>;
}

// SELECT operations
def : Pat<(select (i32 GPR:$C), (f32 GPR:$T), (f32 GPR:$F)),
          (Select_FCC GPR:$T, GPR:$F, GPR:$C, 2)>;

//===----------------------------------------------------------------------===//
// Patterns for Floating Point Instructions
//===----------------------------------------------------------------------===//
def : Pat<(f32 fpimm:$imm), (FORI (i32 R0), fpimm:$imm)>;
