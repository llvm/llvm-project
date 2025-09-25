//=== SparcInstrUAOSA.td - UltraSPARC/Oracle SPARC Architecture extensions ===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains instruction formats, definitions and patterns needed for
// UA 2005, UA 2007, and OSA 2011 instructions on SPARC.
//===----------------------------------------------------------------------===//

class UA2005RegWin<string asmstr, bits<5> fcn>
    : F3_1<2, 0b110001, (outs), (ins), asmstr, []> {
  let rd = fcn;
  let rs1 = 0;
  let rs2 = 0;
}

// Convenience template for 4-operand instructions
class FourOp<string OpcStr, bits<6> op3val, bits<4> op5val,
        RegisterClass RC>
      : F3_4<op3val, op5val, (outs RC:$rd), (ins RC:$rs1, RC:$rs2, RC:$rs3),
        !strconcat(OpcStr, " $rs1, $rs2, $rs3, $rd")>;

/// F2_56 multiclass - Define a F2_5/F2_6 pattern in one shot.
multiclass F2_56<string OpcStr, bits<1> cc> {
  def rr  : F2_5<cc, (outs),
                 (ins cbtarget:$imm10, CCOp:$cond, IntRegs:$rs1, IntRegs:$rs2),
                 !strconcat(OpcStr, "$cond $rs1, $rs2, $imm10")>;
  def ri  : F2_6<cc, (outs),
                 (ins cbtarget:$imm10, CCOp:$cond, IntRegs:$rs1, simm5Op:$simm5),
                 !strconcat(OpcStr, "$cond $rs1, $simm5, $imm10")>;
}

// UltraSPARC Architecture 2005 Instructions
let Predicates = [HasUA2005] in {
let hasSideEffects = 1 in {
def ALLCLEAN : UA2005RegWin<"allclean", 0b00010>;
def INVALW   : UA2005RegWin<"invalw", 0b00101>;
def NORMALW  : UA2005RegWin<"normalw", 0b00100>;
def OTHERW   : UA2005RegWin<"otherw", 0b00011>;
}
} // Predicates = [HasUA2005]

// UltraSPARC Architecture 2007 Instructions
let Predicates = [HasUA2007] in {
def FMADDS : FourOp<"fmadds", 0b110111, 0b0001, FPRegs>;
def FMADDD : FourOp<"fmaddd", 0b110111, 0b0010, DFPRegs>;
def FMSUBS : FourOp<"fmsubs", 0b110111, 0b0101, FPRegs>;
def FMSUBD : FourOp<"fmsubd", 0b110111, 0b0110, DFPRegs>;

def FNMADDS : FourOp<"fnmadds", 0b110111, 0b1101, FPRegs>;
def FNMADDD : FourOp<"fnmaddd", 0b110111, 0b1110, DFPRegs>;
def FNMSUBS : FourOp<"fnmsubs", 0b110111, 0b1001, FPRegs>;
def FNMSUBD : FourOp<"fnmsubd", 0b110111, 0b1010, DFPRegs>;
} // Predicates = [HasUA2007]

// Oracle SPARC Architecture 2011 Instructions
let Predicates = [HasOSA2011] in {
let isBranch = 1, isTerminator = 1, hasDelaySlot = 0 in {
defm CWBCOND : F2_56<"cwb", 0>;
defm CXBCOND : F2_56<"cxb", 1>;
}

def FPMADDX   : FourOp<"fpmaddx", 0b110111, 0b0000, DFPRegs>;
def FPMADDXHI : FourOp<"fpmaddxhi", 0b110111, 0b0100, DFPRegs>;
} // Predicates = [HasOSA2011]

// UA2007 instruction patterns.
let Predicates = [HasUA2007] in {
def : Pat<(f32 (any_fma f32:$rs1, f32:$rs2, f32:$add)), (FMADDS $rs1, $rs2, $add)>;
def : Pat<(f64 (any_fma f64:$rs1, f64:$rs2, f64:$add)), (FMADDD $rs1, $rs2, $add)>;
def : Pat<(f32 (any_fma f32:$rs1, f32:$rs2, (fneg f32:$sub))), (FMSUBS $rs1, $rs2, $sub)>;
def : Pat<(f64 (any_fma f64:$rs1, f64:$rs2, (fneg f64:$sub))), (FMSUBD $rs1, $rs2, $sub)>;
def : Pat<(f32 (fneg (any_fma f32:$rs1, f32:$rs2, f32:$add))), (FNMADDS $rs1, $rs2, $add)>;
def : Pat<(f64 (fneg (any_fma f64:$rs1, f64:$rs2, f64:$add))), (FNMADDD $rs1, $rs2, $add)>;
def : Pat<(f32 (fneg (any_fma f32:$rs1, f32:$rs2, (fneg f32:$sub)))), (FNMSUBS $rs1, $rs2, $sub)>;
def : Pat<(f64 (fneg (any_fma f64:$rs1, f64:$rs2, (fneg f64:$sub)))), (FNMSUBD $rs1, $rs2, $sub)>;
} // Predicates = [HasUA2007]
