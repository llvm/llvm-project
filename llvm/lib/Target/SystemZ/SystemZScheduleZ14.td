//-- SystemZScheduleZ14.td - SystemZ Scheduling Definitions ----*- tblgen -*-=//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the machine model for Z14 to support instruction
// scheduling and other instruction cost heuristics.
//
//===----------------------------------------------------------------------===//

def Z14Model : SchedMachineModel {

    let UnsupportedFeatures = Arch12UnsupportedFeatures.List;

    let IssueWidth = 8;
    let MicroOpBufferSize = 60;     // Issue queues
    let LoadLatency = 1;            // Optimistic load latency.

    let PostRAScheduler = 1;

    // Extra cycles for a mispredicted branch.
    let MispredictPenalty = 20;
}

let SchedModel = Z14Model in  {

// These definitions could be put in a subtarget common include file,
// but it seems the include system in Tablegen currently rejects
// multiple includes of same file.
def : WriteRes<GroupAlone, []> {
  let NumMicroOps = 0;
  let BeginGroup  = 1;
  let EndGroup    = 1;
}
def : WriteRes<BeginGroup, []> {
  let NumMicroOps = 0;
  let BeginGroup  = 1;
}
def : WriteRes<EndGroup, []> {
  let NumMicroOps = 0;
  let EndGroup    = 1;
}
def : WriteRes<Lat2, []> { let Latency = 2; let NumMicroOps = 0;}
def : WriteRes<Lat3, []> { let Latency = 3; let NumMicroOps = 0;}
def : WriteRes<Lat4, []> { let Latency = 4; let NumMicroOps = 0;}
def : WriteRes<Lat5, []> { let Latency = 5; let NumMicroOps = 0;}
def : WriteRes<Lat6, []> { let Latency = 6; let NumMicroOps = 0;}
def : WriteRes<Lat7, []> { let Latency = 7; let NumMicroOps = 0;}
def : WriteRes<Lat8, []> { let Latency = 8; let NumMicroOps = 0;}
def : WriteRes<Lat9, []> { let Latency = 9; let NumMicroOps = 0;}
def : WriteRes<Lat10, []> { let Latency = 10; let NumMicroOps = 0;}
def : WriteRes<Lat11, []> { let Latency = 11; let NumMicroOps = 0;}
def : WriteRes<Lat12, []> { let Latency = 12; let NumMicroOps = 0;}
def : WriteRes<Lat15, []> { let Latency = 15; let NumMicroOps = 0;}
def : WriteRes<Lat20, []> { let Latency = 20; let NumMicroOps = 0;}
def : WriteRes<Lat30, []> { let Latency = 30; let NumMicroOps = 0;}

// Execution units.
def Z14_FXaUnit     : ProcResource<2>;
def Z14_FXbUnit     : ProcResource<2>;
def Z14_LSUnit      : ProcResource<2>;
def Z14_VecUnit     : ProcResource<2>;
def Z14_VecFPdUnit  : ProcResource<2> { let BufferSize = 1; /* blocking */ }
def Z14_VBUnit      : ProcResource<2>;

// Subtarget specific definitions of scheduling resources.
def : WriteRes<FXa,     [Z14_FXaUnit]> { let Latency = 1; }
def : WriteRes<FXa2,    [Z14_FXaUnit, Z14_FXaUnit]> { let Latency = 2; }
def : WriteRes<FXb,     [Z14_FXbUnit]> { let Latency = 1; }
def : WriteRes<LSU,     [Z14_LSUnit]>  { let Latency = 4; }
def : WriteRes<VecBF,   [Z14_VecUnit]> { let Latency = 8; }
def : WriteRes<VecBF2,  [Z14_VecUnit, Z14_VecUnit]> { let Latency = 9; }
def : WriteRes<VecDF,   [Z14_VecUnit]> { let Latency = 8; }
def : WriteRes<VecDF2,  [Z14_VecUnit, Z14_VecUnit]> { let Latency = 9; }
def : WriteRes<VecDFX,  [Z14_VecUnit]> { let Latency = 1; }
def : WriteRes<VecDFX2, [Z14_VecUnit, Z14_VecUnit]> { let Latency = 2; }
def : WriteRes<VecFPd,  [Z14_VecFPdUnit, Z14_VecFPdUnit, Z14_VecFPdUnit,
                         Z14_VecFPdUnit, Z14_VecFPdUnit, Z14_VecFPdUnit,
                         Z14_VecFPdUnit, Z14_VecFPdUnit, Z14_VecFPdUnit,
                         Z14_VecFPdUnit, Z14_VecFPdUnit, Z14_VecFPdUnit,
                         Z14_VecFPdUnit, Z14_VecFPdUnit, Z14_VecFPdUnit,
                         Z14_VecFPdUnit, Z14_VecFPdUnit, Z14_VecFPdUnit,
                         Z14_VecFPdUnit, Z14_VecFPdUnit, Z14_VecFPdUnit,
                         Z14_VecFPdUnit, Z14_VecFPdUnit, Z14_VecFPdUnit,
                         Z14_VecFPdUnit, Z14_VecFPdUnit, Z14_VecFPdUnit,
                         Z14_VecFPdUnit, Z14_VecFPdUnit, Z14_VecFPdUnit]>
                         { let Latency = 30; }
def : WriteRes<VecMul,  [Z14_VecUnit]> { let Latency = 5; }
def : WriteRes<VecStr,  [Z14_VecUnit]> { let Latency = 4; }
def : WriteRes<VecXsPm, [Z14_VecUnit]> { let Latency = 3; }
def : WriteRes<VBU,     [Z14_VBUnit]>; // Virtual Branching Unit

// -------------------------- INSTRUCTIONS ---------------------------------- //

// InstRW constructs have been used in order to preserve the
// readability of the InstrInfo files.

// For each instruction, as matched by a regexp, provide a list of
// resources that it needs. These will be combined into a SchedClass.

//===----------------------------------------------------------------------===//
// Stack allocation
//===----------------------------------------------------------------------===//

def : InstRW<[FXa], (instregex "ADJDYNALLOC$")>; // Pseudo -> LA / LAY

//===----------------------------------------------------------------------===//
// Branch instructions
//===----------------------------------------------------------------------===//

// Branch
def : InstRW<[VBU], (instregex "(Call)?BRC(L)?(Asm.*)?$")>;
def : InstRW<[VBU], (instregex "(Call)?J(G)?(Asm.*)?$")>;
def : InstRW<[FXb], (instregex "(Call)?BC(R)?(Asm.*)?$")>;
def : InstRW<[FXb], (instregex "(Call)?B(R)?(Asm.*)?$")>;
def : InstRW<[FXb, LSU, Lat5], (instregex "BI(C)?(Asm.*)?$")>;
def : InstRW<[FXa, EndGroup], (instregex "BRCT(G)?$")>;
def : InstRW<[FXb, FXa, Lat2, GroupAlone], (instregex "BRCTH$")>;
def : InstRW<[FXb, FXa, Lat2, GroupAlone], (instregex "BCT(G)?(R)?$")>;
def : InstRW<[FXa, FXa, FXb, FXb, Lat4, GroupAlone],
             (instregex "B(R)?X(H|L).*$")>;

// Compare and branch
def : InstRW<[FXb], (instregex "C(L)?(G)?(I|R)J(Asm.*)?$")>;
def : InstRW<[FXb, FXb, Lat2, GroupAlone],
             (instregex "C(L)?(G)?(I|R)B(Call|Return|Asm.*)?$")>;

//===----------------------------------------------------------------------===//
// Trap instructions
//===----------------------------------------------------------------------===//

// Trap
def : InstRW<[VBU], (instregex "(Cond)?Trap$")>;

// Compare and trap
def : InstRW<[FXb], (instregex "C(G)?(I|R)T(Asm.*)?$")>;
def : InstRW<[FXb], (instregex "CL(G)?RT(Asm.*)?$")>;
def : InstRW<[FXb], (instregex "CL(F|G)IT(Asm.*)?$")>;
def : InstRW<[FXb, LSU, Lat5], (instregex "CL(G)?T(Asm.*)?$")>;

//===----------------------------------------------------------------------===//
// Call and return instructions
//===----------------------------------------------------------------------===//

// Call
def : InstRW<[VBU, FXa, FXa, Lat3, GroupAlone], (instregex "(Call)?BRAS$")>;
def : InstRW<[FXa, FXa, FXb, Lat3, GroupAlone], (instregex "(Call)?BRASL$")>;
def : InstRW<[FXa, FXa, FXb, Lat3, GroupAlone], (instregex "(Call)?BAS(R)?$")>;
def : InstRW<[FXa, FXa, FXb, Lat3, GroupAlone], (instregex "TLS_(G|L)DCALL$")>;

// Return
def : InstRW<[FXb, EndGroup], (instregex "Return$")>;
def : InstRW<[FXb], (instregex "CondReturn$")>;

//===----------------------------------------------------------------------===//
// Select instructions
//===----------------------------------------------------------------------===//

// Select pseudo
def : InstRW<[FXa], (instregex "Select(32|64|32Mux)$")>;

// CondStore pseudos
def : InstRW<[FXa], (instregex "CondStore16(Inv)?$")>;
def : InstRW<[FXa], (instregex "CondStore16Mux(Inv)?$")>;
def : InstRW<[FXa], (instregex "CondStore32(Inv)?$")>;
def : InstRW<[FXa], (instregex "CondStore32Mux(Inv)?$")>;
def : InstRW<[FXa], (instregex "CondStore64(Inv)?$")>;
def : InstRW<[FXa], (instregex "CondStore8(Inv)?$")>;
def : InstRW<[FXa], (instregex "CondStore8Mux(Inv)?$")>;

//===----------------------------------------------------------------------===//
// Move instructions
//===----------------------------------------------------------------------===//

// Moves
def : InstRW<[FXb, LSU, Lat5], (instregex "MV(G|H)?HI$")>;
def : InstRW<[FXb, LSU, Lat5], (instregex "MVI(Y)?$")>;

// Move character
def : InstRW<[FXb, LSU, LSU, LSU, Lat8, GroupAlone], (instregex "MVC$")>;
def : InstRW<[LSU, Lat30, GroupAlone], (instregex "MVCL(E|U)?$")>;

// Pseudo -> reg move
def : InstRW<[FXa], (instregex "COPY(_TO_REGCLASS)?$")>;
def : InstRW<[FXa], (instregex "EXTRACT_SUBREG$")>;
def : InstRW<[FXa], (instregex "INSERT_SUBREG$")>;
def : InstRW<[FXa], (instregex "REG_SEQUENCE$")>;
def : InstRW<[FXa], (instregex "SUBREG_TO_REG$")>;

// Loads
def : InstRW<[LSU], (instregex "L(Y|FH|RL|Mux|CBB)?$")>;
def : InstRW<[LSU], (instregex "LG(RL)?$")>;
def : InstRW<[LSU], (instregex "L128$")>;

def : InstRW<[FXa], (instregex "LLIH(F|H|L)$")>;
def : InstRW<[FXa], (instregex "LLIL(F|H|L)$")>;

def : InstRW<[FXa], (instregex "LG(F|H)I$")>;
def : InstRW<[FXa], (instregex "LHI(Mux)?$")>;
def : InstRW<[FXa], (instregex "LR(Mux)?$")>;

// Load and zero rightmost byte
def : InstRW<[LSU], (instregex "LZR(F|G)$")>;

// Load and trap
def : InstRW<[FXb, LSU, Lat5], (instregex "L(FH|G)?AT$")>;

// Load and test
def : InstRW<[FXa, LSU, Lat5], (instregex "LT(G)?$")>;
def : InstRW<[FXa], (instregex "LT(G)?R$")>;

// Stores
def : InstRW<[FXb, LSU, Lat5], (instregex "STG(RL)?$")>;
def : InstRW<[FXb, LSU, Lat5], (instregex "ST128$")>;
def : InstRW<[FXb, LSU, Lat5], (instregex "ST(Y|FH|RL|Mux)?$")>;

// String moves.
def : InstRW<[LSU, Lat30, GroupAlone], (instregex "MVST$")>;

//===----------------------------------------------------------------------===//
// Conditional move instructions
//===----------------------------------------------------------------------===//

def : InstRW<[FXa, Lat2], (instregex "LOCRMux$")>;
def : InstRW<[FXa, Lat2], (instregex "LOC(G|FH)?R(Asm.*)?$")>;
def : InstRW<[FXa, Lat2], (instregex "LOC(G|H)?HI(Mux|(Asm.*))?$")>;
def : InstRW<[FXa, LSU, Lat6], (instregex "LOC(G|FH|Mux)?(Asm.*)?$")>;
def : InstRW<[FXb, LSU, Lat5], (instregex "STOC(G|FH|Mux)?(Asm.*)?$")>;

//===----------------------------------------------------------------------===//
// Sign extensions
//===----------------------------------------------------------------------===//

def : InstRW<[FXa], (instregex "L(B|H|G)R$")>;
def : InstRW<[FXa], (instregex "LG(B|H|F)R$")>;

def : InstRW<[FXa, LSU, Lat5], (instregex "LTGF$")>;
def : InstRW<[FXa], (instregex "LTGFR$")>;

def : InstRW<[FXa, LSU, Lat5], (instregex "LB(H|Mux)?$")>;
def : InstRW<[FXa, LSU, Lat5], (instregex "LH(Y)?$")>;
def : InstRW<[FXa, LSU, Lat5], (instregex "LH(H|Mux|RL)$")>;
def : InstRW<[FXa, LSU, Lat5], (instregex "LG(B|H|F)$")>;
def : InstRW<[FXa, LSU, Lat5], (instregex "LG(H|F)RL$")>;

//===----------------------------------------------------------------------===//
// Zero extensions
//===----------------------------------------------------------------------===//

def : InstRW<[FXa], (instregex "LLCR(Mux)?$")>;
def : InstRW<[FXa], (instregex "LLHR(Mux)?$")>;
def : InstRW<[FXa], (instregex "LLG(C|H|F|T)R$")>;
def : InstRW<[LSU], (instregex "LLC(Mux)?$")>;
def : InstRW<[LSU], (instregex "LLH(Mux)?$")>;
def : InstRW<[FXa, LSU, Lat5], (instregex "LL(C|H)H$")>;
def : InstRW<[LSU], (instregex "LLHRL$")>;
def : InstRW<[LSU], (instregex "LLG(C|H|F|T|HRL|FRL)$")>;

// Load and zero rightmost byte
def : InstRW<[LSU], (instregex "LLZRGF$")>;

// Load and trap
def : InstRW<[FXb, LSU, Lat5], (instregex "LLG(F|T)?AT$")>;

//===----------------------------------------------------------------------===//
// Truncations
//===----------------------------------------------------------------------===//

def : InstRW<[FXb, LSU, Lat5], (instregex "STC(H|Y|Mux)?$")>;
def : InstRW<[FXb, LSU, Lat5], (instregex "STH(H|Y|RL|Mux)?$")>;
def : InstRW<[FXb, LSU, Lat5], (instregex "STCM(H|Y)?$")>;

//===----------------------------------------------------------------------===//
// Multi-register moves
//===----------------------------------------------------------------------===//

// Load multiple (estimated average of 5 ops)
def : InstRW<[LSU, LSU, LSU, LSU, LSU, Lat10, GroupAlone],
             (instregex "LM(H|Y|G)?$")>;

// Load multiple disjoint
def : InstRW<[LSU, Lat30, GroupAlone], (instregex "LMD$")>;

// Store multiple (estimated average of ceil(5/2) FXb ops)
def : InstRW<[LSU, LSU, FXb, FXb, FXb, Lat10,
              GroupAlone], (instregex "STM(G|H|Y)?$")>;

//===----------------------------------------------------------------------===//
// Byte swaps
//===----------------------------------------------------------------------===//

def : InstRW<[FXa], (instregex "LRV(G)?R$")>;
def : InstRW<[FXa, LSU, Lat5], (instregex "LRV(G|H)?$")>;
def : InstRW<[FXb, LSU, Lat5], (instregex "STRV(G|H)?$")>;
def : InstRW<[LSU, Lat30, GroupAlone], (instregex "MVCIN$")>;

//===----------------------------------------------------------------------===//
// Load address instructions
//===----------------------------------------------------------------------===//

def : InstRW<[FXa], (instregex "LA(Y|RL)?$")>;

// Load the Global Offset Table address ( -> larl )
def : InstRW<[FXa], (instregex "GOT$")>;

//===----------------------------------------------------------------------===//
// Absolute and Negation
//===----------------------------------------------------------------------===//

def : InstRW<[FXa], (instregex "LP(G)?R$")>;
def : InstRW<[FXa, FXa, Lat2, BeginGroup], (instregex "L(N|P)GFR$")>;
def : InstRW<[FXa], (instregex "LN(R|GR)$")>;
def : InstRW<[FXa], (instregex "LC(R|GR)$")>;
def : InstRW<[FXa, FXa, Lat2, BeginGroup], (instregex "LCGFR$")>;

//===----------------------------------------------------------------------===//
// Insertion
//===----------------------------------------------------------------------===//

def : InstRW<[FXa, LSU, Lat5], (instregex "IC(Y)?$")>;
def : InstRW<[FXa, LSU, Lat5], (instregex "IC32(Y)?$")>;
def : InstRW<[FXa, LSU, Lat5], (instregex "ICM(H|Y)?$")>;
def : InstRW<[FXa], (instregex "II(F|H|L)Mux$")>;
def : InstRW<[FXa], (instregex "IIHF(64)?$")>;
def : InstRW<[FXa], (instregex "IIHH(64)?$")>;
def : InstRW<[FXa], (instregex "IIHL(64)?$")>;
def : InstRW<[FXa], (instregex "IILF(64)?$")>;
def : InstRW<[FXa], (instregex "IILH(64)?$")>;
def : InstRW<[FXa], (instregex "IILL(64)?$")>;

//===----------------------------------------------------------------------===//
// Addition
//===----------------------------------------------------------------------===//

def : InstRW<[FXa, LSU, Lat5], (instregex "A(Y)?$")>;
def : InstRW<[FXa, LSU, Lat6], (instregex "AH(Y)?$")>;
def : InstRW<[FXa], (instregex "AIH$")>;
def : InstRW<[FXa], (instregex "AFI(Mux)?$")>;
def : InstRW<[FXa, LSU, Lat5], (instregex "AG$")>;
def : InstRW<[FXa], (instregex "AGFI$")>;
def : InstRW<[FXa], (instregex "AGHI(K)?$")>;
def : InstRW<[FXa], (instregex "AGR(K)?$")>;
def : InstRW<[FXa], (instregex "AHI(K)?$")>;
def : InstRW<[FXa], (instregex "AHIMux(K)?$")>;
def : InstRW<[FXa, LSU, Lat5], (instregex "AL(Y)?$")>;
def : InstRW<[FXa], (instregex "AL(FI|HSIK)$")>;
def : InstRW<[FXa, LSU, Lat5], (instregex "ALG(F)?$")>;
def : InstRW<[FXa], (instregex "ALGHSIK$")>;
def : InstRW<[FXa], (instregex "ALGF(I|R)$")>;
def : InstRW<[FXa], (instregex "ALGR(K)?$")>;
def : InstRW<[FXa], (instregex "ALR(K)?$")>;
def : InstRW<[FXa], (instregex "AR(K)?$")>;
def : InstRW<[FXa], (instregex "A(L)?HHHR$")>;
def : InstRW<[FXa, Lat2], (instregex "A(L)?HHLR$")>;
def : InstRW<[FXa], (instregex "ALSIH(N)?$")>;
def : InstRW<[FXb, LSU, Lat5], (instregex "A(L)?(G)?SI$")>;

// Logical addition with carry
def : InstRW<[FXa, LSU, Lat6, GroupAlone], (instregex "ALC(G)?$")>;
def : InstRW<[FXa, Lat2, GroupAlone], (instregex "ALC(G)?R$")>;

// Add with sign extension (16/32 -> 64)
def : InstRW<[FXa, LSU, Lat6], (instregex "AG(F|H)$")>;
def : InstRW<[FXa, Lat2], (instregex "AGFR$")>;

//===----------------------------------------------------------------------===//
// Subtraction
//===----------------------------------------------------------------------===//

def : InstRW<[FXa, LSU, Lat5], (instregex "S(G|Y)?$")>;
def : InstRW<[FXa, LSU, Lat6], (instregex "SH(Y)?$")>;
def : InstRW<[FXa], (instregex "SGR(K)?$")>;
def : InstRW<[FXa], (instregex "SLFI$")>;
def : InstRW<[FXa, LSU, Lat5], (instregex "SL(G|GF|Y)?$")>;
def : InstRW<[FXa], (instregex "SLGF(I|R)$")>;
def : InstRW<[FXa], (instregex "SLGR(K)?$")>;
def : InstRW<[FXa], (instregex "SLR(K)?$")>;
def : InstRW<[FXa], (instregex "SR(K)?$")>;
def : InstRW<[FXa], (instregex "S(L)?HHHR$")>;
def : InstRW<[FXa, Lat2], (instregex "S(L)?HHLR$")>;

// Subtraction with borrow
def : InstRW<[FXa, LSU, Lat6, GroupAlone], (instregex "SLB(G)?$")>;
def : InstRW<[FXa, Lat2, GroupAlone], (instregex "SLB(G)?R$")>;

// Subtraction with sign extension (16/32 -> 64)
def : InstRW<[FXa, LSU, Lat6], (instregex "SG(F|H)$")>;
def : InstRW<[FXa, Lat2], (instregex "SGFR$")>;

//===----------------------------------------------------------------------===//
// AND
//===----------------------------------------------------------------------===//

def : InstRW<[FXa, LSU, Lat5], (instregex "N(G|Y)?$")>;
def : InstRW<[FXa], (instregex "NGR(K)?$")>;
def : InstRW<[FXa], (instregex "NI(FMux|HMux|LMux)$")>;
def : InstRW<[FXb, LSU, Lat5], (instregex "NI(Y)?$")>;
def : InstRW<[FXa], (instregex "NIHF(64)?$")>;
def : InstRW<[FXa], (instregex "NIHH(64)?$")>;
def : InstRW<[FXa], (instregex "NIHL(64)?$")>;
def : InstRW<[FXa], (instregex "NILF(64)?$")>;
def : InstRW<[FXa], (instregex "NILH(64)?$")>;
def : InstRW<[FXa], (instregex "NILL(64)?$")>;
def : InstRW<[FXa], (instregex "NR(K)?$")>;
def : InstRW<[LSU, LSU, FXb, Lat9, BeginGroup], (instregex "NC$")>;

//===----------------------------------------------------------------------===//
// OR
//===----------------------------------------------------------------------===//

def : InstRW<[FXa, LSU, Lat5], (instregex "O(G|Y)?$")>;
def : InstRW<[FXa], (instregex "OGR(K)?$")>;
def : InstRW<[FXb, LSU, Lat5], (instregex "OI(Y)?$")>;
def : InstRW<[FXa], (instregex "OI(FMux|HMux|LMux)$")>;
def : InstRW<[FXa], (instregex "OIHF(64)?$")>;
def : InstRW<[FXa], (instregex "OIHH(64)?$")>;
def : InstRW<[FXa], (instregex "OIHL(64)?$")>;
def : InstRW<[FXa], (instregex "OILF(64)?$")>;
def : InstRW<[FXa], (instregex "OILH(64)?$")>;
def : InstRW<[FXa], (instregex "OILL(64)?$")>;
def : InstRW<[FXa], (instregex "OR(K)?$")>;
def : InstRW<[LSU, LSU, FXb, Lat9, BeginGroup], (instregex "OC$")>;

//===----------------------------------------------------------------------===//
// XOR
//===----------------------------------------------------------------------===//

def : InstRW<[FXa, LSU, Lat5], (instregex "X(G|Y)?$")>;
def : InstRW<[FXb, LSU, Lat5], (instregex "XI(Y)?$")>;
def : InstRW<[FXa], (instregex "XIFMux$")>;
def : InstRW<[FXa], (instregex "XGR(K)?$")>;
def : InstRW<[FXa], (instregex "XIHF(64)?$")>;
def : InstRW<[FXa], (instregex "XILF(64)?$")>;
def : InstRW<[FXa], (instregex "XR(K)?$")>;
def : InstRW<[LSU, LSU, FXb, Lat9, BeginGroup], (instregex "XC$")>;

//===----------------------------------------------------------------------===//
// Multiplication
//===----------------------------------------------------------------------===//

def : InstRW<[FXa, LSU, Lat9], (instregex "MS(GF|Y)?$")>;
def : InstRW<[FXa, Lat5], (instregex "MS(R|FI)$")>;
def : InstRW<[FXa, LSU, Lat11], (instregex "MSG$")>;
def : InstRW<[FXa, Lat7], (instregex "MSGR$")>;
def : InstRW<[FXa, Lat5], (instregex "MSGF(I|R)$")>;
def : InstRW<[FXa2, LSU, Lat12, GroupAlone], (instregex "MLG$")>;
def : InstRW<[FXa2, Lat8, GroupAlone], (instregex "MLGR$")>;
def : InstRW<[FXa, Lat4], (instregex "MGHI$")>;
def : InstRW<[FXa, Lat4], (instregex "MHI$")>;
def : InstRW<[FXa, LSU, Lat8], (instregex "MH(Y)?$")>;
def : InstRW<[FXa2, Lat6, GroupAlone], (instregex "M(L)?R$")>;
def : InstRW<[FXa2, LSU, Lat10, GroupAlone], (instregex "M(FY|L)?$")>;
def : InstRW<[FXa, LSU, Lat8], (instregex "MGH$")>;
def : InstRW<[FXa, FXa, LSU, Lat12, GroupAlone], (instregex "MG$")>;
def : InstRW<[FXa, FXa, Lat8, GroupAlone], (instregex "MGRK$")>;
def : InstRW<[FXa, LSU, Lat9], (instregex "MSC$")>;
def : InstRW<[FXa, LSU, Lat11], (instregex "MSGC$")>;
def : InstRW<[FXa, Lat5], (instregex "MSRKC$")>;
def : InstRW<[FXa, Lat7], (instregex "MSGRKC$")>;

//===----------------------------------------------------------------------===//
// Division and remainder
//===----------------------------------------------------------------------===//

def : InstRW<[FXa2, FXa2, Lat20, GroupAlone], (instregex "DR$")>;
def : InstRW<[FXa2, FXa2, LSU, Lat30, GroupAlone], (instregex "D$")>;
def : InstRW<[FXa2, Lat30, GroupAlone], (instregex "DSG(F)?R$")>;
def : InstRW<[LSU, FXa2, Lat30, GroupAlone], (instregex "DSG(F)?$")>;
def : InstRW<[FXa2, FXa2, Lat20, GroupAlone], (instregex "DLR$")>;
def : InstRW<[FXa2, FXa2, Lat30, GroupAlone], (instregex "DLGR$")>;
def : InstRW<[FXa2, FXa2, LSU, Lat30, GroupAlone], (instregex "DL(G)?$")>;

//===----------------------------------------------------------------------===//
// Shifts
//===----------------------------------------------------------------------===//

def : InstRW<[FXa], (instregex "SLL(G|K)?$")>;
def : InstRW<[FXa], (instregex "SRL(G|K)?$")>;
def : InstRW<[FXa], (instregex "SRA(G|K)?$")>;
def : InstRW<[FXa], (instregex "SLA(G|K)?$")>;
def : InstRW<[FXa, FXa, FXa, FXa, LSU, Lat8, GroupAlone],
             (instregex "S(L|R)D(A|L)$")>;

// Rotate
def : InstRW<[FXa, LSU, Lat6], (instregex "RLL(G)?$")>;

// Rotate and insert
def : InstRW<[FXa], (instregex "RISBG(N|32)?$")>;
def : InstRW<[FXa], (instregex "RISBH(G|H|L)$")>;
def : InstRW<[FXa], (instregex "RISBL(G|H|L)$")>;
def : InstRW<[FXa], (instregex "RISBMux$")>;

// Rotate and Select
def : InstRW<[FXa, FXa, Lat2, BeginGroup], (instregex "R(N|O|X)SBG$")>;

//===----------------------------------------------------------------------===//
// Comparison
//===----------------------------------------------------------------------===//

def : InstRW<[FXb, LSU, Lat5], (instregex "C(G|Y|Mux|RL)?$")>;
def : InstRW<[FXb], (instregex "C(F|H)I(Mux)?$")>;
def : InstRW<[FXb], (instregex "CG(F|H)I$")>;
def : InstRW<[FXb, LSU, Lat5], (instregex "CG(HSI|RL)$")>;
def : InstRW<[FXb], (instregex "C(G)?R$")>;
def : InstRW<[FXb], (instregex "CIH$")>;
def : InstRW<[FXb, LSU, Lat5], (instregex "CH(F|SI)$")>;
def : InstRW<[FXb, LSU, Lat5], (instregex "CL(Y|Mux|FHSI)?$")>;
def : InstRW<[FXb], (instregex "CLFI(Mux)?$")>;
def : InstRW<[FXb, LSU, Lat5], (instregex "CLG(HRL|HSI)?$")>;
def : InstRW<[FXb, LSU, Lat5], (instregex "CLGF(RL)?$")>;
def : InstRW<[FXb], (instregex "CLGF(I|R)$")>;
def : InstRW<[FXb], (instregex "CLGR$")>;
def : InstRW<[FXb, LSU, Lat5], (instregex "CLGRL$")>;
def : InstRW<[FXb, LSU, Lat5], (instregex "CLH(F|RL|HSI)$")>;
def : InstRW<[FXb], (instregex "CLIH$")>;
def : InstRW<[FXb, LSU, Lat5], (instregex "CLI(Y)?$")>;
def : InstRW<[FXb], (instregex "CLR$")>;
def : InstRW<[FXb, LSU, Lat5], (instregex "CLRL$")>;
def : InstRW<[FXb], (instregex "C(L)?HHR$")>;
def : InstRW<[FXb, Lat2], (instregex "C(L)?HLR$")>;

// Compare halfword
def : InstRW<[FXb, LSU, Lat6], (instregex "CH(Y|RL)?$")>;
def : InstRW<[FXb, LSU, Lat6], (instregex "CGH(RL)?$")>;
def : InstRW<[FXa, FXb, LSU, Lat6, BeginGroup], (instregex "CHHSI$")>;

// Compare with sign extension (32 -> 64)
def : InstRW<[FXb, LSU, Lat6], (instregex "CGF(RL)?$")>;
def : InstRW<[FXb, Lat2], (instregex "CGFR$")>;

// Compare logical character
def : InstRW<[FXb, LSU, LSU, Lat9, BeginGroup], (instregex "CLC$")>;
def : InstRW<[LSU, Lat30, GroupAlone], (instregex "CLCL(E|U)?$")>;
def : InstRW<[LSU, Lat30, GroupAlone], (instregex "CLST$")>;

// Test under mask
def : InstRW<[FXb, LSU, Lat5], (instregex "TM(Y)?$")>;
def : InstRW<[FXb], (instregex "TM(H|L)Mux$")>;
def : InstRW<[FXb], (instregex "TMHH(64)?$")>;
def : InstRW<[FXb], (instregex "TMHL(64)?$")>;
def : InstRW<[FXb], (instregex "TMLH(64)?$")>;
def : InstRW<[FXb], (instregex "TMLL(64)?$")>;

// Compare logical characters under mask
def : InstRW<[FXb, LSU, Lat6], (instregex "CLM(H|Y)?$")>;

//===----------------------------------------------------------------------===//
// Prefetch and execution hint
//===----------------------------------------------------------------------===//

def : InstRW<[LSU], (instregex "PFD(RL)?$")>;
def : InstRW<[FXb, Lat2], (instregex "BPP$")>;
def : InstRW<[FXb, EndGroup], (instregex "BPRP$")>;
def : InstRW<[FXb], (instregex "NIAI$")>;

//===----------------------------------------------------------------------===//
// Atomic operations
//===----------------------------------------------------------------------===//

def : InstRW<[FXb, EndGroup], (instregex "Serialize$")>;

def : InstRW<[FXb, LSU, Lat5], (instregex "LAA(G)?$")>;
def : InstRW<[FXb, LSU, Lat5], (instregex "LAAL(G)?$")>;
def : InstRW<[FXb, LSU, Lat5], (instregex "LAN(G)?$")>;
def : InstRW<[FXb, LSU, Lat5], (instregex "LAO(G)?$")>;
def : InstRW<[FXb, LSU, Lat5], (instregex "LAX(G)?$")>;

// Test and set
def : InstRW<[FXb, LSU, Lat5, EndGroup], (instregex "TS$")>;

// Compare and swap
def : InstRW<[FXa, FXb, LSU, Lat6, GroupAlone], (instregex "CS(G|Y)?$")>;

// Compare double and swap
def : InstRW<[FXa, FXa, FXb, FXb, FXa, LSU, Lat10, GroupAlone],
             (instregex "CDS(Y)?$")>;
def : InstRW<[FXa, FXa, FXb, FXb, LSU, FXb, FXb, LSU, LSU, Lat20, GroupAlone],
             (instregex "CDSG$")>;

// Compare and swap and store
def : InstRW<[FXa, LSU, Lat30], (instregex "CSST$")>;

// Perform locked operation
def : InstRW<[LSU, Lat30, GroupAlone], (instregex "PLO$")>;

// Load/store pair from/to quadword
def : InstRW<[LSU, LSU, Lat5, GroupAlone], (instregex "LPQ$")>;
def : InstRW<[FXb, FXb, LSU, Lat6, GroupAlone], (instregex "STPQ$")>;

// Load pair disjoint
def : InstRW<[LSU, LSU, Lat5, GroupAlone], (instregex "LPD(G)?$")>;

//===----------------------------------------------------------------------===//
// Translate and convert
//===----------------------------------------------------------------------===//

def : InstRW<[LSU, Lat30, GroupAlone], (instregex "TR$")>;
def : InstRW<[FXa, FXa, FXa, LSU, LSU, Lat30, GroupAlone], (instregex "TRT$")>;
def : InstRW<[FXa, LSU, Lat30], (instregex "TRTR$")>;
def : InstRW<[FXa, Lat30], (instregex "TR(TR)?(T)?(E|EOpt)?$")>;
def : InstRW<[LSU, Lat30], (instregex "TR(T|O)(T|O)(Opt)?$")>;
def : InstRW<[FXa, Lat30], (instregex "CU(12|14|21|24|41|42)(Opt)?$")>;
def : InstRW<[FXa, Lat30], (instregex "(CUUTF|CUTFU)(Opt)?$")>;

//===----------------------------------------------------------------------===//
// Message-security assist
//===----------------------------------------------------------------------===//

def : InstRW<[FXa, Lat30], (instregex "KM(C|F|O|CTR|A)?$")>;
def : InstRW<[FXa, Lat30], (instregex "(KIMD|KLMD|KMAC)$")>;
def : InstRW<[FXa, Lat30], (instregex "(PCC|PPNO|PRNO)$")>;

//===----------------------------------------------------------------------===//
// Guarded storage
//===----------------------------------------------------------------------===//

def : InstRW<[LSU], (instregex "LGG$")>;
def : InstRW<[LSU, Lat5], (instregex "LLGFSG$")>;
def : InstRW<[LSU, Lat30], (instregex "(L|ST)GSC$")>;

//===----------------------------------------------------------------------===//
// Decimal arithmetic
//===----------------------------------------------------------------------===//

def : InstRW<[FXb, VecDF, VecDF, LSU, LSU, Lat30, GroupAlone],
             (instregex "CVBG$")>;
def : InstRW<[FXb, VecDF, LSU, Lat30, GroupAlone], (instregex "CVB(Y)?$")>;
def : InstRW<[FXb, FXb, FXb, VecDF2, VecDF2, LSU, Lat30, GroupAlone],
             (instregex "CVDG$")>;
def : InstRW<[FXb, VecDF, FXb, LSU, Lat30, GroupAlone], (instregex "CVD(Y)?$")>;
def : InstRW<[LSU, Lat10, GroupAlone], (instregex "MVO$")>;
def : InstRW<[LSU, Lat30, GroupAlone], (instregex "MV(N|Z)$")>;
def : InstRW<[LSU, Lat30, GroupAlone], (instregex "(PACK|PKA|PKU)$")>;
def : InstRW<[LSU, Lat12, GroupAlone], (instregex "UNPK(A|U)$")>;
def : InstRW<[FXb, LSU, LSU, Lat9, BeginGroup], (instregex "UNPK$")>;

def : InstRW<[FXb, VecDFX, LSU, LSU, LSU, Lat9, GroupAlone],
             (instregex "(A|S|ZA)P$")>;
def : InstRW<[FXb, VecDFX2, VecDFX2, LSU, LSU, LSU, Lat30, GroupAlone],
             (instregex "(M|D)P$")>;
def : InstRW<[FXb, VecDFX, VecDFX, LSU, LSU, Lat15, GroupAlone],
             (instregex "SRP$")>;
def : InstRW<[VecDFX, LSU, LSU, Lat5, GroupAlone], (instregex "CP$")>;
def : InstRW<[VecDFX, LSU, Lat4, BeginGroup], (instregex "TP$")>;
def : InstRW<[LSU, Lat30, GroupAlone], (instregex "ED(MK)?$")>;

//===----------------------------------------------------------------------===//
// Access registers
//===----------------------------------------------------------------------===//

// Extract/set/copy access register
def : InstRW<[LSU], (instregex "(EAR|SAR|CPYA)$")>;

// Load address extended
def : InstRW<[LSU, FXa, Lat5, BeginGroup], (instregex "LAE(Y)?$")>;

// Load/store access multiple (not modeled precisely)
def : InstRW<[LSU, Lat30, GroupAlone], (instregex "(L|ST)AM(Y)?$")>;

//===----------------------------------------------------------------------===//
// Program mask and addressing mode
//===----------------------------------------------------------------------===//

// Insert Program Mask
def : InstRW<[FXa, Lat3, EndGroup], (instregex "IPM$")>;

// Set Program Mask
def : InstRW<[LSU, EndGroup], (instregex "SPM$")>;

// Branch and link
def : InstRW<[FXa, FXa, FXb, Lat5, GroupAlone], (instregex "BAL(R)?$")>;

// Test addressing mode
def : InstRW<[FXb], (instregex "TAM$")>;

// Set addressing mode
def : InstRW<[FXb, Lat2, EndGroup], (instregex "SAM(24|31|64)$")>;

// Branch (and save) and set mode.
def : InstRW<[FXa, FXb, Lat2, GroupAlone], (instregex "BSM$")>;
def : InstRW<[FXa, FXa, FXb, Lat3, GroupAlone], (instregex "BASSM$")>;

//===----------------------------------------------------------------------===//
// Transactional execution
//===----------------------------------------------------------------------===//

// Transaction begin
def : InstRW<[LSU, LSU, FXb, FXb, FXb, FXb, FXb, Lat15, GroupAlone],
              (instregex "TBEGIN(C|_nofloat)?$")>;

// Transaction end
def : InstRW<[FXb, GroupAlone], (instregex "TEND$")>;

// Transaction abort
def : InstRW<[LSU, GroupAlone], (instregex "TABORT$")>;

// Extract Transaction Nesting Depth
def : InstRW<[FXa], (instregex "ETND$")>;

// Nontransactional store
def : InstRW<[FXb, LSU, Lat5], (instregex "NTSTG$")>;

//===----------------------------------------------------------------------===//
// Processor assist
//===----------------------------------------------------------------------===//

def : InstRW<[FXb, GroupAlone], (instregex "PPA$")>;

//===----------------------------------------------------------------------===//
// Miscellaneous Instructions.
//===----------------------------------------------------------------------===//

// Find leftmost one
def : InstRW<[FXa, FXa, Lat4, GroupAlone], (instregex "FLOGR$")>;

// Population count
def : InstRW<[FXa, Lat3], (instregex "POPCNT$")>;

// Extend
def : InstRW<[FXa], (instregex "AEXT128$")>;
def : InstRW<[FXa], (instregex "ZEXT128$")>;

// String instructions
def : InstRW<[FXa, LSU, Lat30], (instregex "SRST$")>;
def : InstRW<[FXa, Lat30], (instregex "SRSTU$")>;
def : InstRW<[LSU, Lat30, GroupAlone], (instregex "CUSE$")>;

// Various complex instructions
def : InstRW<[LSU, Lat30], (instregex "CFC$")>;
def : InstRW<[FXb, LSU, Lat30], (instregex "UPT$")>;
def : InstRW<[LSU, Lat30], (instregex "CKSM$")>;
def : InstRW<[FXa, Lat30], (instregex "CMPSC$")>;

// Execute
def : InstRW<[FXb, GroupAlone], (instregex "EX(RL)?$")>;

//===----------------------------------------------------------------------===//
// .insn directive instructions
//===----------------------------------------------------------------------===//

// An "empty" sched-class will be assigned instead of the "invalid sched-class".
// getNumDecoderSlots() will then return 1 instead of 0.
def : InstRW<[], (instregex "Insn.*")>;


// ----------------------------- Floating point ----------------------------- //

//===----------------------------------------------------------------------===//
// FP: Select instructions
//===----------------------------------------------------------------------===//

def : InstRW<[FXa], (instregex "Select(F32|F64|F128|VR128)$")>;
def : InstRW<[FXa], (instregex "CondStoreF32(Inv)?$")>;
def : InstRW<[FXa], (instregex "CondStoreF64(Inv)?$")>;

//===----------------------------------------------------------------------===//
// FP: Move instructions
//===----------------------------------------------------------------------===//

// Load zero
def : InstRW<[FXb], (instregex "LZ(DR|ER)$")>;
def : InstRW<[FXb, FXb, Lat2, BeginGroup], (instregex "LZXR$")>;

// Load
def : InstRW<[VecXsPm], (instregex "LER$")>;
def : InstRW<[FXb], (instregex "LD(R|R32|GR)$")>;
def : InstRW<[FXb, Lat3], (instregex "LGDR$")>;
def : InstRW<[FXb, FXb, Lat2, GroupAlone], (instregex "LXR$")>;

// Load and Test
def : InstRW<[VecXsPm, Lat4], (instregex "LT(D|E)BR$")>;
def : InstRW<[VecXsPm, Lat4], (instregex "LTEBRCompare(_VecPseudo)?$")>;
def : InstRW<[VecXsPm, Lat4], (instregex "LTDBRCompare(_VecPseudo)?$")>;
def : InstRW<[VecDF2, VecDF2, Lat11, GroupAlone], (instregex "LTXBR$")>;
def : InstRW<[VecDF2, VecDF2, Lat11, GroupAlone],
             (instregex "LTXBRCompare(_VecPseudo)?$")>;

// Copy sign
def : InstRW<[VecXsPm], (instregex "CPSDRd(d|s)$")>;
def : InstRW<[VecXsPm], (instregex "CPSDRs(d|s)$")>;

//===----------------------------------------------------------------------===//
// FP: Load instructions
//===----------------------------------------------------------------------===//

def : InstRW<[VecXsPm, LSU, Lat7], (instregex "LE(Y)?$")>;
def : InstRW<[LSU], (instregex "LD(Y|E32)?$")>;
def : InstRW<[LSU], (instregex "LX$")>;

//===----------------------------------------------------------------------===//
// FP: Store instructions
//===----------------------------------------------------------------------===//

def : InstRW<[FXb, LSU, Lat7], (instregex "STD(Y)?$")>;
def : InstRW<[FXb, LSU, Lat7], (instregex "STE(Y)?$")>;
def : InstRW<[FXb, LSU, Lat5], (instregex "STX$")>;

//===----------------------------------------------------------------------===//
// FP: Conversion instructions
//===----------------------------------------------------------------------===//

// Load rounded
def : InstRW<[VecBF], (instregex "LEDBR(A)?$")>;
def : InstRW<[VecDF, VecDF, Lat20], (instregex "LEXBR(A)?$")>;
def : InstRW<[VecDF, VecDF, Lat20], (instregex "LDXBR(A)?$")>;

// Load lengthened
def : InstRW<[VecBF, LSU, Lat12], (instregex "LDEB$")>;
def : InstRW<[VecBF], (instregex "LDEBR$")>;
def : InstRW<[VecBF2, VecBF2, LSU, Lat12 , GroupAlone], (instregex "LX(D|E)B$")>;
def : InstRW<[VecBF2, VecBF2, GroupAlone], (instregex "LX(D|E)BR$")>;

// Convert from fixed / logical
def : InstRW<[FXb, VecBF, Lat9, BeginGroup], (instregex "CE(F|G)BR(A)?$")>;
def : InstRW<[FXb, VecBF, Lat9, BeginGroup], (instregex "CD(F|G)BR(A)?$")>;
def : InstRW<[FXb, VecDF2, VecDF2, Lat12, GroupAlone], (instregex "CX(F|G)BR(A)?$")>;
def : InstRW<[FXb, VecBF, Lat9, BeginGroup], (instregex "CEL(F|G)BR$")>;
def : InstRW<[FXb, VecBF, Lat9, BeginGroup], (instregex "CDL(F|G)BR$")>;
def : InstRW<[FXb, VecDF2, VecDF2, Lat12, GroupAlone], (instregex "CXL(F|G)BR$")>;

// Convert to fixed / logical
def : InstRW<[FXb, VecBF, Lat11, BeginGroup], (instregex "CF(E|D)BR(A)?$")>;
def : InstRW<[FXb, VecBF, Lat11, BeginGroup], (instregex "CG(E|D)BR(A)?$")>;
def : InstRW<[FXb, VecDF, VecDF, Lat20, BeginGroup], (instregex "C(F|G)XBR(A)?$")>;
def : InstRW<[FXb, VecBF, Lat11, GroupAlone], (instregex "CLFEBR$")>;
def : InstRW<[FXb, VecBF, Lat11, BeginGroup], (instregex "CLFDBR$")>;
def : InstRW<[FXb, VecBF, Lat11, BeginGroup], (instregex "CLG(E|D)BR$")>;
def : InstRW<[FXb, VecDF, VecDF, Lat20, BeginGroup], (instregex "CL(F|G)XBR$")>;

//===----------------------------------------------------------------------===//
// FP: Unary arithmetic
//===----------------------------------------------------------------------===//

// Load Complement / Negative / Positive
def : InstRW<[VecXsPm, Lat4], (instregex "L(C|N|P)DBR$")>;
def : InstRW<[VecXsPm, Lat4], (instregex "L(C|N|P)EBR$")>;
def : InstRW<[FXb], (instregex "LCDFR(_32)?$")>;
def : InstRW<[FXb], (instregex "LNDFR(_32)?$")>;
def : InstRW<[FXb], (instregex "LPDFR(_32)?$")>;
def : InstRW<[VecDF2, VecDF2, Lat11, GroupAlone], (instregex "L(C|N|P)XBR$")>;

// Square root
def : InstRW<[VecFPd, LSU], (instregex "SQ(E|D)B$")>;
def : InstRW<[VecFPd], (instregex "SQ(E|D)BR$")>;
def : InstRW<[VecFPd, VecFPd, GroupAlone], (instregex "SQXBR$")>;

// Load FP integer
def : InstRW<[VecBF], (instregex "FIEBR(A)?$")>;
def : InstRW<[VecBF], (instregex "FIDBR(A)?$")>;
def : InstRW<[VecDF2, VecDF2, Lat11, GroupAlone], (instregex "FIXBR(A)?$")>;

//===----------------------------------------------------------------------===//
// FP: Binary arithmetic
//===----------------------------------------------------------------------===//

// Addition
def : InstRW<[VecBF, LSU, Lat12], (instregex "A(E|D)B$")>;
def : InstRW<[VecBF], (instregex "A(E|D)BR$")>;
def : InstRW<[VecDF2, VecDF2, Lat10, GroupAlone], (instregex "AXBR$")>;

// Subtraction
def : InstRW<[VecBF, LSU, Lat12], (instregex "S(E|D)B$")>;
def : InstRW<[VecBF], (instregex "S(E|D)BR$")>;
def : InstRW<[VecDF2, VecDF2, Lat11, GroupAlone], (instregex "SXBR$")>;

// Multiply
def : InstRW<[VecBF, LSU, Lat12], (instregex "M(D|DE|EE)B$")>;
def : InstRW<[VecBF], (instregex "M(D|DE|EE)BR$")>;
def : InstRW<[VecBF2, VecBF2, LSU, Lat12, GroupAlone], (instregex "MXDB$")>;
def : InstRW<[VecBF2, VecBF2, GroupAlone], (instregex "MXDBR$")>;
def : InstRW<[VecDF2, VecDF2, Lat20, GroupAlone], (instregex "MXBR$")>;

// Multiply and add / subtract
def : InstRW<[VecBF2, LSU, Lat12, GroupAlone], (instregex "M(A|S)EB$")>;
def : InstRW<[VecBF, GroupAlone], (instregex "M(A|S)EBR$")>;
def : InstRW<[VecBF2, LSU, Lat12, GroupAlone], (instregex "M(A|S)DB$")>;
def : InstRW<[VecBF], (instregex "M(A|S)DBR$")>;

// Division
def : InstRW<[VecFPd, LSU], (instregex "D(E|D)B$")>;
def : InstRW<[VecFPd], (instregex "D(E|D)BR$")>;
def : InstRW<[VecFPd, VecFPd, GroupAlone], (instregex "DXBR$")>;

// Divide to integer
def : InstRW<[VecFPd, Lat30], (instregex "DI(E|D)BR$")>;

//===----------------------------------------------------------------------===//
// FP: Comparisons
//===----------------------------------------------------------------------===//

// Compare
def : InstRW<[VecXsPm, LSU, Lat8], (instregex "(K|C)(E|D)B$")>;
def : InstRW<[VecXsPm, Lat4], (instregex "(K|C)(E|D)BR?$")>;
def : InstRW<[VecDF, VecDF, Lat20, GroupAlone], (instregex "(K|C)XBR$")>;

// Test Data Class
def : InstRW<[LSU, VecXsPm, Lat9], (instregex "TC(E|D)B$")>;
def : InstRW<[LSU, VecDF2, VecDF2, Lat15, GroupAlone], (instregex "TCXB$")>;

//===----------------------------------------------------------------------===//
// FP: Floating-point control register instructions
//===----------------------------------------------------------------------===//

def : InstRW<[FXa, LSU, Lat4, GroupAlone], (instregex "EFPC$")>;
def : InstRW<[FXb, LSU, Lat5, GroupAlone], (instregex "STFPC$")>;
def : InstRW<[LSU, Lat3, GroupAlone], (instregex "SFPC$")>;
def : InstRW<[LSU, LSU, Lat6, GroupAlone], (instregex "LFPC$")>;
def : InstRW<[FXa, Lat30], (instregex "SFASR$")>;
def : InstRW<[FXa, LSU, Lat30], (instregex "LFAS$")>;
def : InstRW<[FXb, Lat3, GroupAlone], (instregex "SRNM(B|T)?$")>;


// --------------------- Hexadecimal floating point ------------------------- //

//===----------------------------------------------------------------------===//
// HFP: Move instructions
//===----------------------------------------------------------------------===//

// Load and Test
def : InstRW<[VecXsPm, Lat4], (instregex "LT(D|E)R$")>;
def : InstRW<[VecDF2, VecDF2, Lat11, GroupAlone], (instregex "LTXR$")>;

//===----------------------------------------------------------------------===//
// HFP: Conversion instructions
//===----------------------------------------------------------------------===//

// Load rounded
def : InstRW<[VecBF], (instregex "(LEDR|LRER)$")>;
def : InstRW<[VecBF], (instregex "LEXR$")>;
def : InstRW<[VecDF2], (instregex "(LDXR|LRDR)$")>;

// Load lengthened
def : InstRW<[LSU], (instregex "LDE$")>;
def : InstRW<[FXb], (instregex "LDER$")>;
def : InstRW<[VecBF2, VecBF2, LSU, Lat12, GroupAlone], (instregex "LX(D|E)$")>;
def : InstRW<[VecBF2, VecBF2, GroupAlone], (instregex "LX(D|E)R$")>;

// Convert from fixed
def : InstRW<[FXb, VecBF, Lat9, BeginGroup], (instregex "CE(F|G)R$")>;
def : InstRW<[FXb, VecBF, Lat9, BeginGroup], (instregex "CD(F|G)R$")>;
def : InstRW<[FXb, VecDF2, VecDF2, Lat12, GroupAlone], (instregex "CX(F|G)R$")>;

// Convert to fixed
def : InstRW<[FXb, VecBF, Lat11, BeginGroup], (instregex "CF(E|D)R$")>;
def : InstRW<[FXb, VecBF, Lat11, BeginGroup], (instregex "CG(E|D)R$")>;
def : InstRW<[FXb, VecDF, VecDF, Lat20, BeginGroup], (instregex "C(F|G)XR$")>;

// Convert BFP to HFP / HFP to BFP.
def : InstRW<[VecBF], (instregex "THD(E)?R$")>;
def : InstRW<[VecBF], (instregex "TB(E)?DR$")>;

//===----------------------------------------------------------------------===//
// HFP: Unary arithmetic
//===----------------------------------------------------------------------===//

// Load Complement / Negative / Positive
def : InstRW<[VecXsPm, Lat4], (instregex "L(C|N|P)DR$")>;
def : InstRW<[VecXsPm, Lat4], (instregex "L(C|N|P)ER$")>;
def : InstRW<[VecDF2, VecDF2, Lat11, GroupAlone], (instregex "L(C|N|P)XR$")>;

// Halve
def : InstRW<[VecBF], (instregex "H(E|D)R$")>;

// Square root
def : InstRW<[VecFPd, LSU], (instregex "SQ(E|D)$")>;
def : InstRW<[VecFPd], (instregex "SQ(E|D)R$")>;
def : InstRW<[VecFPd, VecFPd, GroupAlone], (instregex "SQXR$")>;

// Load FP integer
def : InstRW<[VecBF], (instregex "FIER$")>;
def : InstRW<[VecBF], (instregex "FIDR$")>;
def : InstRW<[VecDF2, VecDF2, Lat11, GroupAlone], (instregex "FIXR$")>;

//===----------------------------------------------------------------------===//
// HFP: Binary arithmetic
//===----------------------------------------------------------------------===//

// Addition
def : InstRW<[VecBF, LSU, Lat12], (instregex "A(E|D|U|W)$")>;
def : InstRW<[VecBF], (instregex "A(E|D|U|W)R$")>;
def : InstRW<[VecDF2, VecDF2, Lat10, GroupAlone], (instregex "AXR$")>;

// Subtraction
def : InstRW<[VecBF, LSU, Lat12], (instregex "S(E|D|U|W)$")>;
def : InstRW<[VecBF], (instregex "S(E|D|U|W)R$")>;
def : InstRW<[VecDF2, VecDF2, Lat11, GroupAlone], (instregex "SXR$")>;

// Multiply
def : InstRW<[VecBF, LSU, Lat12], (instregex "M(D|DE|E|EE)$")>;
def : InstRW<[VecBF], (instregex "M(D|DE|E|EE)R$")>;
def : InstRW<[VecBF2, VecBF2, LSU, Lat12, GroupAlone], (instregex "MXD$")>;
def : InstRW<[VecBF2, VecBF2, GroupAlone], (instregex "MXDR$")>;
def : InstRW<[VecDF2, VecDF2, Lat20, GroupAlone], (instregex "MXR$")>;
def : InstRW<[VecBF2, VecBF2, LSU, Lat12, GroupAlone], (instregex "MY$")>;
def : InstRW<[VecBF2, LSU, Lat12, GroupAlone], (instregex "MY(H|L)$")>;
def : InstRW<[VecBF2, VecBF2, GroupAlone], (instregex "MYR$")>;
def : InstRW<[VecBF, GroupAlone], (instregex "MY(H|L)R$")>;

// Multiply and add / subtract
def : InstRW<[VecBF2, LSU, Lat12, GroupAlone], (instregex "M(A|S)E$")>;
def : InstRW<[VecBF, GroupAlone], (instregex "M(A|S)ER$")>;
def : InstRW<[VecBF2, LSU, Lat12, GroupAlone], (instregex "M(A|S)D$")>;
def : InstRW<[VecBF, GroupAlone], (instregex "M(A|S)DR$")>;
def : InstRW<[VecBF2, LSU, Lat12, GroupAlone], (instregex "MAY(H|L)$")>;
def : InstRW<[VecBF2, VecBF2, LSU, Lat12, GroupAlone], (instregex "MAY$")>;
def : InstRW<[VecBF, GroupAlone], (instregex "MAY(H|L)R$")>;
def : InstRW<[VecBF2, VecBF2, GroupAlone], (instregex "MAYR$")>;

// Division
def : InstRW<[VecFPd, LSU], (instregex "D(E|D)$")>;
def : InstRW<[VecFPd], (instregex "D(E|D)R$")>;
def : InstRW<[VecFPd, VecFPd, GroupAlone], (instregex "DXR$")>;

//===----------------------------------------------------------------------===//
// HFP: Comparisons
//===----------------------------------------------------------------------===//

// Compare
def : InstRW<[VecBF, LSU, Lat12], (instregex "C(E|D)$")>;
def : InstRW<[VecBF], (instregex "C(E|D)R$")>;
def : InstRW<[VecDF, VecDF, Lat20, GroupAlone], (instregex "CXR$")>;


// ------------------------ Decimal floating point -------------------------- //

//===----------------------------------------------------------------------===//
// DFP: Move instructions
//===----------------------------------------------------------------------===//

// Load and Test
def : InstRW<[VecDF], (instregex "LTDTR$")>;
def : InstRW<[VecDF2, VecDF2, Lat11, GroupAlone], (instregex "LTXTR$")>;

//===----------------------------------------------------------------------===//
// DFP: Conversion instructions
//===----------------------------------------------------------------------===//

// Load rounded
def : InstRW<[VecDF, Lat15], (instregex "LEDTR$")>;
def : InstRW<[VecDF, VecDF, Lat20], (instregex "LDXTR$")>;

// Load lengthened
def : InstRW<[VecDF], (instregex "LDETR$")>;
def : InstRW<[VecDF2, VecDF2, Lat11, GroupAlone], (instregex "LXDTR$")>;

// Convert from fixed / logical
def : InstRW<[FXb, VecDF, Lat30, BeginGroup], (instregex "CD(F|G)TR(A)?$")>;
def : InstRW<[FXb, VecDF2, VecDF2, Lat30, GroupAlone], (instregex "CX(F|G)TR(A)?$")>;
def : InstRW<[FXb, VecDF, Lat30, BeginGroup], (instregex "CDL(F|G)TR$")>;
def : InstRW<[FXb, VecDF2, VecDF2, Lat30, GroupAlone], (instregex "CXL(F|G)TR$")>;

// Convert to fixed / logical
def : InstRW<[FXb, VecDF, Lat30, BeginGroup], (instregex "C(F|G)DTR(A)?$")>;
def : InstRW<[FXb, VecDF, VecDF, Lat30, BeginGroup], (instregex "C(F|G)XTR(A)?$")>;
def : InstRW<[FXb, VecDF, Lat30, BeginGroup], (instregex "CL(F|G)DTR$")>;
def : InstRW<[FXb, VecDF, VecDF, Lat30, BeginGroup], (instregex "CL(F|G)XTR$")>;

// Convert from / to signed / unsigned packed
def : InstRW<[FXb, VecDF, Lat9, BeginGroup], (instregex "CD(S|U)TR$")>;
def : InstRW<[FXb, FXb, VecDF2, VecDF2, Lat15, GroupAlone], (instregex "CX(S|U)TR$")>;
def : InstRW<[FXb, VecDF, Lat12, BeginGroup], (instregex "C(S|U)DTR$")>;
def : InstRW<[FXb, FXb, VecDF2, VecDF2, Lat15, GroupAlone], (instregex "C(S|U)XTR$")>;

// Convert from / to zoned
def : InstRW<[LSU, VecDF, Lat11, BeginGroup], (instregex "CDZT$")>;
def : InstRW<[LSU, LSU, VecDF2, VecDF2, Lat15, GroupAlone], (instregex "CXZT$")>;
def : InstRW<[FXb, LSU, VecDF, Lat11, BeginGroup], (instregex "CZDT$")>;
def : InstRW<[FXb, LSU, VecDF, VecDF, Lat15, GroupAlone], (instregex "CZXT$")>;

// Convert from / to packed
def : InstRW<[LSU, VecDF, Lat11, BeginGroup], (instregex "CDPT$")>;
def : InstRW<[LSU, LSU, VecDF2, VecDF2, Lat15, GroupAlone], (instregex "CXPT$")>;
def : InstRW<[FXb, LSU, VecDF, Lat11, BeginGroup], (instregex "CPDT$")>;
def : InstRW<[FXb, LSU, VecDF, VecDF, Lat15, GroupAlone], (instregex "CPXT$")>;

// Perform floating-point operation
def : InstRW<[FXb, Lat30], (instregex "PFPO$")>;

//===----------------------------------------------------------------------===//
// DFP: Unary arithmetic
//===----------------------------------------------------------------------===//

// Load FP integer
def : InstRW<[VecDF], (instregex "FIDTR$")>;
def : InstRW<[VecDF2, VecDF2, Lat11, GroupAlone], (instregex "FIXTR$")>;

// Extract biased exponent
def : InstRW<[FXb, VecDF, Lat12, BeginGroup], (instregex "EEDTR$")>;
def : InstRW<[FXb, VecDF, Lat12, BeginGroup], (instregex "EEXTR$")>;

// Extract significance
def : InstRW<[FXb, VecDF, Lat12, BeginGroup], (instregex "ESDTR$")>;
def : InstRW<[FXb, VecDF, VecDF, Lat15, BeginGroup], (instregex "ESXTR$")>;

//===----------------------------------------------------------------------===//
// DFP: Binary arithmetic
//===----------------------------------------------------------------------===//

// Addition
def : InstRW<[VecDF], (instregex "ADTR(A)?$")>;
def : InstRW<[VecDF2, VecDF2, Lat10, GroupAlone], (instregex "AXTR(A)?$")>;

// Subtraction
def : InstRW<[VecDF], (instregex "SDTR(A)?$")>;
def : InstRW<[VecDF2, VecDF2, Lat11, GroupAlone], (instregex "SXTR(A)?$")>;

// Multiply
def : InstRW<[VecDF, Lat30], (instregex "MDTR(A)?$")>;
def : InstRW<[VecDF2, VecDF2, Lat30, GroupAlone], (instregex "MXTR(A)?$")>;

// Division
def : InstRW<[VecDF, Lat30], (instregex "DDTR(A)?$")>;
def : InstRW<[VecDF2, VecDF2, Lat30, GroupAlone], (instregex "DXTR(A)?$")>;

// Quantize
def : InstRW<[VecDF], (instregex "QADTR$")>;
def : InstRW<[VecDF2, VecDF2, Lat11, GroupAlone], (instregex "QAXTR$")>;

// Reround
def : InstRW<[FXb, VecDF, Lat11, BeginGroup], (instregex "RRDTR$")>;
def : InstRW<[FXb, VecDF2, VecDF2, Lat15, GroupAlone], (instregex "RRXTR$")>;

// Shift significand left/right
def : InstRW<[LSU, VecDF, Lat11, GroupAlone], (instregex "S(L|R)DT$")>;
def : InstRW<[LSU, VecDF2, VecDF2, Lat15, GroupAlone], (instregex "S(L|R)XT$")>;

// Insert biased exponent
def : InstRW<[FXb, VecDF, Lat11, BeginGroup], (instregex "IEDTR$")>;
def : InstRW<[FXb, VecDF2, VecDF2, Lat15, GroupAlone], (instregex "IEXTR$")>;

//===----------------------------------------------------------------------===//
// DFP: Comparisons
//===----------------------------------------------------------------------===//

// Compare
def : InstRW<[VecDF], (instregex "(K|C)DTR$")>;
def : InstRW<[VecDF, VecDF, Lat11, GroupAlone], (instregex "(K|C)XTR$")>;

// Compare biased exponent
def : InstRW<[VecDF], (instregex "CEDTR$")>;
def : InstRW<[VecDF], (instregex "CEXTR$")>;

// Test Data Class/Group
def : InstRW<[LSU, VecDF, Lat11], (instregex "TD(C|G)(E|D)T$")>;
def : InstRW<[LSU, VecDF, VecDF, Lat15, GroupAlone], (instregex "TD(C|G)XT$")>;


// --------------------------------- Vector --------------------------------- //

//===----------------------------------------------------------------------===//
// Vector: Move instructions
//===----------------------------------------------------------------------===//

def : InstRW<[FXb], (instregex "VLR(32|64)?$")>;
def : InstRW<[FXb, Lat4], (instregex "VLGV(B|F|G|H)?$")>;
def : InstRW<[FXb], (instregex "VLVG(B|F|G|H)?$")>;
def : InstRW<[FXb, Lat2], (instregex "VLVGP(32)?$")>;

//===----------------------------------------------------------------------===//
// Vector: Immediate instructions
//===----------------------------------------------------------------------===//

def : InstRW<[VecXsPm], (instregex "VZERO$")>;
def : InstRW<[VecXsPm], (instregex "VONE$")>;
def : InstRW<[VecXsPm], (instregex "VGBM$")>;
def : InstRW<[VecXsPm], (instregex "VGM(B|F|G|H)?$")>;
def : InstRW<[VecXsPm], (instregex "VREPI(B|F|G|H)?$")>;
def : InstRW<[VecXsPm], (instregex "VLEI(B|F|G|H)$")>;

//===----------------------------------------------------------------------===//
// Vector: Loads
//===----------------------------------------------------------------------===//

def : InstRW<[LSU], (instregex "VL(L|BB)?$")>;
def : InstRW<[LSU], (instregex "VL(32|64)$")>;
def : InstRW<[LSU], (instregex "VLLEZ(B|F|G|H|LF)?$")>;
def : InstRW<[LSU], (instregex "VLREP(B|F|G|H)?$")>;
def : InstRW<[VecXsPm, LSU, Lat7], (instregex "VLE(B|F|G|H)$")>;
def : InstRW<[FXb, LSU, VecXsPm, Lat11, BeginGroup], (instregex "VGE(F|G)$")>;
def : InstRW<[LSU, LSU, LSU, LSU, LSU, Lat10, GroupAlone],
              (instregex "VLM$")>;
def : InstRW<[LSU, Lat5], (instregex "VLRL(R)?$")>;

//===----------------------------------------------------------------------===//
// Vector: Stores
//===----------------------------------------------------------------------===//

def : InstRW<[FXb, LSU, Lat8], (instregex "VST(L|32|64)?$")>;
def : InstRW<[FXb, LSU, Lat8], (instregex "VSTE(F|G)$")>;
def : InstRW<[FXb, LSU, VecXsPm, Lat11, BeginGroup], (instregex "VSTE(B|H)$")>;
def : InstRW<[LSU, LSU, FXb, FXb, FXb, FXb, FXb, Lat20, GroupAlone],
              (instregex "VSTM$")>;
def : InstRW<[FXb, FXb, LSU, Lat12, BeginGroup], (instregex "VSCE(F|G)$")>;
def : InstRW<[FXb, LSU, Lat8], (instregex "VSTRL(R)?$")>;

//===----------------------------------------------------------------------===//
// Vector: Selects and permutes
//===----------------------------------------------------------------------===//

def : InstRW<[VecXsPm], (instregex "VMRH(B|F|G|H)?$")>;
def : InstRW<[VecXsPm], (instregex "VMRL(B|F|G|H)?$")>;
def : InstRW<[VecXsPm], (instregex "VPERM$")>;
def : InstRW<[VecXsPm], (instregex "VPDI$")>;
def : InstRW<[VecXsPm], (instregex "VBPERM$")>;
def : InstRW<[VecXsPm], (instregex "VREP(B|F|G|H)?$")>;
def : InstRW<[VecXsPm], (instregex "VSEL$")>;

//===----------------------------------------------------------------------===//
// Vector: Widening and narrowing
//===----------------------------------------------------------------------===//

def : InstRW<[VecXsPm], (instregex "VPK(F|G|H)?$")>;
def : InstRW<[VecXsPm], (instregex "VPKS(F|G|H)?$")>;
def : InstRW<[VecXsPm, Lat4], (instregex "VPKS(F|G|H)S$")>;
def : InstRW<[VecXsPm], (instregex "VPKLS(F|G|H)?$")>;
def : InstRW<[VecXsPm, Lat4], (instregex "VPKLS(F|G|H)S$")>;
def : InstRW<[VecXsPm], (instregex "VSEG(B|F|H)?$")>;
def : InstRW<[VecXsPm], (instregex "VUPH(B|F|H)?$")>;
def : InstRW<[VecXsPm], (instregex "VUPL(B|F)?$")>;
def : InstRW<[VecXsPm], (instregex "VUPLH(B|F|H|W)?$")>;
def : InstRW<[VecXsPm], (instregex "VUPLL(B|F|H)?$")>;

//===----------------------------------------------------------------------===//
// Vector: Integer arithmetic
//===----------------------------------------------------------------------===//

def : InstRW<[VecXsPm], (instregex "VA(B|F|G|H|Q|C|CQ)?$")>;
def : InstRW<[VecXsPm], (instregex "VACC(B|F|G|H|Q|C|CQ)?$")>;
def : InstRW<[VecXsPm], (instregex "VAVG(B|F|G|H)?$")>;
def : InstRW<[VecXsPm], (instregex "VAVGL(B|F|G|H)?$")>;
def : InstRW<[VecXsPm], (instregex "VN(C|O|N|X)?$")>;
def : InstRW<[VecXsPm], (instregex "VO(C)?$")>;
def : InstRW<[VecMul], (instregex "VCKSM$")>;
def : InstRW<[VecXsPm], (instregex "VCLZ(B|F|G|H)?$")>;
def : InstRW<[VecXsPm], (instregex "VCTZ(B|F|G|H)?$")>;
def : InstRW<[VecXsPm], (instregex "VX$")>;
def : InstRW<[VecMul], (instregex "VGFM?$")>;
def : InstRW<[VecMul], (instregex "VGFMA(B|F|G|H)?$")>;
def : InstRW<[VecMul], (instregex "VGFM(B|F|G|H)$")>;
def : InstRW<[VecXsPm], (instregex "VLC(B|F|G|H)?$")>;
def : InstRW<[VecXsPm], (instregex "VLP(B|F|G|H)?$")>;
def : InstRW<[VecXsPm], (instregex "VMX(B|F|G|H)?$")>;
def : InstRW<[VecXsPm], (instregex "VMXL(B|F|G|H)?$")>;
def : InstRW<[VecXsPm], (instregex "VMN(B|F|G|H)?$")>;
def : InstRW<[VecXsPm], (instregex "VMNL(B|F|G|H)?$")>;
def : InstRW<[VecMul], (instregex "VMAL(B|F)?$")>;
def : InstRW<[VecMul], (instregex "VMALE(B|F|H)?$")>;
def : InstRW<[VecMul], (instregex "VMALH(B|F|H|W)?$")>;
def : InstRW<[VecMul], (instregex "VMALO(B|F|H)?$")>;
def : InstRW<[VecMul], (instregex "VMAO(B|F|H)?$")>;
def : InstRW<[VecMul], (instregex "VMAE(B|F|H)?$")>;
def : InstRW<[VecMul], (instregex "VMAH(B|F|H)?$")>;
def : InstRW<[VecMul], (instregex "VME(B|F|H)?$")>;
def : InstRW<[VecMul], (instregex "VMH(B|F|H)?$")>;
def : InstRW<[VecMul], (instregex "VML(B|F)?$")>;
def : InstRW<[VecMul], (instregex "VMLE(B|F|H)?$")>;
def : InstRW<[VecMul], (instregex "VMLH(B|F|H|W)?$")>;
def : InstRW<[VecMul], (instregex "VMLO(B|F|H)?$")>;
def : InstRW<[VecMul], (instregex "VMO(B|F|H)?$")>;
def : InstRW<[VecBF2], (instregex "VMSL(G)?$")>;

def : InstRW<[VecXsPm], (instregex "VPOPCT(B|F|G|H)?$")>;

def : InstRW<[VecXsPm], (instregex "VERLL(B|F|G|H)?$")>;
def : InstRW<[VecXsPm], (instregex "VERLLV(B|F|G|H)?$")>;
def : InstRW<[VecXsPm], (instregex "VERIM(B|F|G|H)?$")>;
def : InstRW<[VecXsPm], (instregex "VESL(B|F|G|H)?$")>;
def : InstRW<[VecXsPm], (instregex "VESLV(B|F|G|H)?$")>;
def : InstRW<[VecXsPm], (instregex "VESRA(B|F|G|H)?$")>;
def : InstRW<[VecXsPm], (instregex "VESRAV(B|F|G|H)?$")>;
def : InstRW<[VecXsPm], (instregex "VESRL(B|F|G|H)?$")>;
def : InstRW<[VecXsPm], (instregex "VESRLV(B|F|G|H)?$")>;

def : InstRW<[VecXsPm], (instregex "VSL(DB)?$")>;
def : InstRW<[VecXsPm], (instregex "VSLB$")>;
def : InstRW<[VecXsPm], (instregex "VSR(A|L)$")>;
def : InstRW<[VecXsPm], (instregex "VSR(A|L)B$")>;

def : InstRW<[VecXsPm], (instregex "VSB(I|IQ|CBI|CBIQ)?$")>;
def : InstRW<[VecXsPm], (instregex "VSCBI(B|F|G|H|Q)?$")>;
def : InstRW<[VecXsPm], (instregex "VS(F|G|H|Q)?$")>;

def : InstRW<[VecMul], (instregex "VSUM(B|H)?$")>;
def : InstRW<[VecMul], (instregex "VSUMG(F|H)?$")>;
def : InstRW<[VecMul], (instregex "VSUMQ(F|G)?$")>;

//===----------------------------------------------------------------------===//
// Vector: Integer comparison
//===----------------------------------------------------------------------===//

def : InstRW<[VecXsPm, Lat4], (instregex "VEC(B|F|G|H)?$")>;
def : InstRW<[VecXsPm, Lat4], (instregex "VECL(B|F|G|H)?$")>;
def : InstRW<[VecXsPm], (instregex "VCEQ(B|F|G|H)?$")>;
def : InstRW<[VecXsPm, Lat4], (instregex "VCEQ(B|F|G|H)S$")>;
def : InstRW<[VecXsPm], (instregex "VCH(B|F|G|H)?$")>;
def : InstRW<[VecXsPm, Lat4], (instregex "VCH(B|F|G|H)S$")>;
def : InstRW<[VecXsPm], (instregex "VCHL(B|F|G|H)?$")>;
def : InstRW<[VecXsPm, Lat4], (instregex "VCHL(B|F|G|H)S$")>;
def : InstRW<[VecStr, Lat5], (instregex "VTM$")>;

//===----------------------------------------------------------------------===//
// Vector: Floating-point arithmetic
//===----------------------------------------------------------------------===//

// Conversion and rounding
def : InstRW<[VecBF], (instregex "VCD(L)?G$")>;
def : InstRW<[VecBF], (instregex "VCD(L)?GB$")>;
def : InstRW<[VecBF], (instregex "WCD(L)?GB$")>;
def : InstRW<[VecBF], (instregex "VC(L)?GD$")>;
def : InstRW<[VecBF], (instregex "VC(L)?GDB$")>;
def : InstRW<[VecBF], (instregex "WC(L)?GDB$")>;
def : InstRW<[VecBF], (instregex "VL(DE|ED)$")>;
def : InstRW<[VecBF], (instregex "VL(DE|ED)B$")>;
def : InstRW<[VecBF], (instregex "WL(DE|ED)B$")>;
def : InstRW<[VecBF], (instregex "VFL(L|R)$")>;
def : InstRW<[VecBF], (instregex "VFL(LS|RD)$")>;
def : InstRW<[VecBF], (instregex "WFL(LS|RD)$")>;
def : InstRW<[VecBF2], (instregex "WFLLD$")>;
def : InstRW<[VecDF2, Lat10], (instregex "WFLRX$")>;
def : InstRW<[VecBF2], (instregex "VFI$")>;
def : InstRW<[VecBF], (instregex "VFIDB$")>;
def : InstRW<[VecBF], (instregex "WFIDB$")>;
def : InstRW<[VecBF2], (instregex "VFISB$")>;
def : InstRW<[VecBF], (instregex "WFISB$")>;
def : InstRW<[VecDF2, Lat10], (instregex "WFIXB$")>;

// Sign operations
def : InstRW<[VecXsPm], (instregex "VFPSO$")>;
def : InstRW<[VecXsPm], (instregex "(V|W)FPSODB$")>;
def : InstRW<[VecXsPm], (instregex "(V|W)FPSOSB$")>;
def : InstRW<[VecXsPm], (instregex "WFPSOXB$")>;
def : InstRW<[VecXsPm], (instregex "(V|W)FL(C|N|P)DB$")>;
def : InstRW<[VecXsPm], (instregex "(V|W)FL(C|N|P)SB$")>;
def : InstRW<[VecXsPm], (instregex "WFL(C|N|P)XB$")>;

// Minimum / maximum
def : InstRW<[VecXsPm], (instregex "VF(MAX|MIN)$")>;
def : InstRW<[VecXsPm], (instregex "VF(MAX|MIN)DB$")>;
def : InstRW<[VecXsPm], (instregex "WF(MAX|MIN)DB$")>;
def : InstRW<[VecXsPm], (instregex "VF(MAX|MIN)SB$")>;
def : InstRW<[VecXsPm], (instregex "WF(MAX|MIN)SB$")>;
def : InstRW<[VecDFX], (instregex "WF(MAX|MIN)XB$")>;

// Test data class
def : InstRW<[VecXsPm, Lat4], (instregex "VFTCI$")>;
def : InstRW<[VecXsPm, Lat4], (instregex "(V|W)FTCIDB$")>;
def : InstRW<[VecXsPm, Lat4], (instregex "(V|W)FTCISB$")>;
def : InstRW<[VecDFX, Lat4], (instregex "WFTCIXB$")>;

// Add / subtract
def : InstRW<[VecBF2], (instregex "VF(A|S)$")>;
def : InstRW<[VecBF], (instregex "VF(A|S)DB$")>;
def : InstRW<[VecBF], (instregex "WF(A|S)DB$")>;
def : InstRW<[VecBF2], (instregex "VF(A|S)SB$")>;
def : InstRW<[VecBF], (instregex "WF(A|S)SB$")>;
def : InstRW<[VecDF2, Lat10], (instregex "WF(A|S)XB$")>;

// Multiply / multiply-and-add/subtract
def : InstRW<[VecBF2], (instregex "VFM$")>;
def : InstRW<[VecBF], (instregex "VFMDB$")>;
def : InstRW<[VecBF], (instregex "WFMDB$")>;
def : InstRW<[VecBF2], (instregex "VFMSB$")>;
def : InstRW<[VecBF], (instregex "WFMSB$")>;
def : InstRW<[VecDF2, Lat20], (instregex "WFMXB$")>;
def : InstRW<[VecBF2], (instregex "VF(N)?M(A|S)$")>;
def : InstRW<[VecBF], (instregex "VF(N)?M(A|S)DB$")>;
def : InstRW<[VecBF], (instregex "WF(N)?M(A|S)DB$")>;
def : InstRW<[VecBF2], (instregex "VF(N)?M(A|S)SB$")>;
def : InstRW<[VecBF], (instregex "WF(N)?M(A|S)SB$")>;
def : InstRW<[VecDF2, Lat20], (instregex "WF(N)?M(A|S)XB$")>;

// Divide / square root
def : InstRW<[VecFPd], (instregex "VFD$")>;
def : InstRW<[VecFPd], (instregex "(V|W)FDDB$")>;
def : InstRW<[VecFPd], (instregex "(V|W)FDSB$")>;
def : InstRW<[VecFPd], (instregex "WFDXB$")>;
def : InstRW<[VecFPd], (instregex "VFSQ$")>;
def : InstRW<[VecFPd], (instregex "(V|W)FSQDB$")>;
def : InstRW<[VecFPd], (instregex "(V|W)FSQSB$")>;
def : InstRW<[VecFPd], (instregex "WFSQXB$")>;

//===----------------------------------------------------------------------===//
// Vector: Floating-point comparison
//===----------------------------------------------------------------------===//

def : InstRW<[VecXsPm], (instregex "VF(C|K)(E|H|HE)$")>;
def : InstRW<[VecXsPm], (instregex "VF(C|K)(E|H|HE)DB$")>;
def : InstRW<[VecXsPm], (instregex "WF(C|K)(E|H|HE)DB$")>;
def : InstRW<[VecXsPm], (instregex "VF(C|K)(E|H|HE)SB$")>;
def : InstRW<[VecXsPm], (instregex "WF(C|K)(E|H|HE)SB$")>;
def : InstRW<[VecDFX], (instregex "WF(C|K)(E|H|HE)XB$")>;
def : InstRW<[VecXsPm, Lat4], (instregex "VF(C|K)(E|H|HE)DBS$")>;
def : InstRW<[VecXsPm, Lat4], (instregex "WF(C|K)(E|H|HE)DBS$")>;
def : InstRW<[VecXsPm, Lat4], (instregex "VF(C|K)(E|H|HE)SBS$")>;
def : InstRW<[VecXsPm, Lat4], (instregex "WF(C|K)(E|H|HE)SBS$")>;
def : InstRW<[VecDFX, Lat4], (instregex "WF(C|K)(E|H|HE)XBS$")>;
def : InstRW<[VecXsPm, Lat4], (instregex "WF(C|K)$")>;
def : InstRW<[VecXsPm, Lat4], (instregex "WF(C|K)DB$")>;
def : InstRW<[VecXsPm, Lat4], (instregex "WF(C|K)SB$")>;
def : InstRW<[VecDFX, Lat4], (instregex "WF(C|K)XB$")>;

//===----------------------------------------------------------------------===//
// Vector: Floating-point insertion and extraction
//===----------------------------------------------------------------------===//

def : InstRW<[FXb], (instregex "LEFR$")>;
def : InstRW<[FXb, Lat4], (instregex "LFER$")>;

//===----------------------------------------------------------------------===//
// Vector: String instructions
//===----------------------------------------------------------------------===//

def : InstRW<[VecStr], (instregex "VFAE(B)?$")>;
def : InstRW<[VecStr, Lat5], (instregex "VFAEBS$")>;
def : InstRW<[VecStr], (instregex "VFAE(F|H)$")>;
def : InstRW<[VecStr, Lat5], (instregex "VFAE(F|H)S$")>;
def : InstRW<[VecStr], (instregex "VFAEZ(B|F|H)$")>;
def : InstRW<[VecStr, Lat5], (instregex "VFAEZ(B|F|H)S$")>;
def : InstRW<[VecStr], (instregex "VFEE(B|F|H|ZB|ZF|ZH)?$")>;
def : InstRW<[VecStr, Lat5], (instregex "VFEE(B|F|H|ZB|ZF|ZH)S$")>;
def : InstRW<[VecStr], (instregex "VFENE(B|F|H|ZB|ZF|ZH)?$")>;
def : InstRW<[VecStr, Lat5], (instregex "VFENE(B|F|H|ZB|ZF|ZH)S$")>;
def : InstRW<[VecStr], (instregex "VISTR(B|F|H)?$")>;
def : InstRW<[VecStr, Lat5], (instregex "VISTR(B|F|H)S$")>;
def : InstRW<[VecStr], (instregex "VSTRC(B|F|H)?$")>;
def : InstRW<[VecStr, Lat5], (instregex "VSTRC(B|F|H)S$")>;
def : InstRW<[VecStr], (instregex "VSTRCZ(B|F|H)$")>;
def : InstRW<[VecStr, Lat5], (instregex "VSTRCZ(B|F|H)S$")>;

//===----------------------------------------------------------------------===//
// Vector: Packed-decimal instructions
//===----------------------------------------------------------------------===//

def : InstRW<[VecDF, VecDF, Lat10], (instregex "VLIP$")>;
def : InstRW<[VecDFX, LSU, GroupAlone], (instregex "VPKZ$")>;
def : InstRW<[VecDFX, FXb, LSU, Lat12, BeginGroup], (instregex "VUPKZ$")>;
def : InstRW<[VecDF, VecDF, FXb, Lat20, GroupAlone], (instregex "VCVB(G)?$")>;
def : InstRW<[VecDF, VecDF, FXb, Lat20, GroupAlone], (instregex "VCVD(G)?$")>;
def : InstRW<[VecDFX], (instregex "V(A|S)P$")>;
def : InstRW<[VecDF, VecDF, Lat30, GroupAlone], (instregex "VM(S)?P$")>;
def : InstRW<[VecDF, VecDF, Lat30, GroupAlone], (instregex "V(D|R)P$")>;
def : InstRW<[VecDFX, Lat30, GroupAlone], (instregex "VSDP$")>;
def : InstRW<[VecDF, VecDF, Lat11], (instregex "VSRP$")>;
def : InstRW<[VecDFX], (instregex "VPSOP$")>;
def : InstRW<[VecDFX], (instregex "V(T|C)P$")>;


// -------------------------------- System ---------------------------------- //

//===----------------------------------------------------------------------===//
// System: Program-Status Word Instructions
//===----------------------------------------------------------------------===//

def : InstRW<[FXb, Lat30], (instregex "EPSW$")>;
def : InstRW<[FXb, LSU, Lat30], (instregex "LPSW(E)?$")>;
def : InstRW<[FXa, Lat3, GroupAlone], (instregex "IPK$")>;
def : InstRW<[LSU, EndGroup], (instregex "SPKA$")>;
def : InstRW<[LSU, EndGroup], (instregex "SSM$")>;
def : InstRW<[FXb, LSU, GroupAlone], (instregex "ST(N|O)SM$")>;
def : InstRW<[FXa, Lat3], (instregex "IAC$")>;
def : InstRW<[LSU, EndGroup], (instregex "SAC(F)?$")>;

//===----------------------------------------------------------------------===//
// System: Control Register Instructions
//===----------------------------------------------------------------------===//

def : InstRW<[FXb, LSU, Lat30], (instregex "LCTL(G)?$")>;
def : InstRW<[LSU, Lat30], (instregex "STCT(L|G)$")>;
def : InstRW<[LSU], (instregex "E(P|S)A(I)?R$")>;
def : InstRW<[FXb, Lat30], (instregex "SSA(I)?R$")>;
def : InstRW<[FXb, Lat30], (instregex "ESEA$")>;

//===----------------------------------------------------------------------===//
// System: Prefix-Register Instructions
//===----------------------------------------------------------------------===//

def : InstRW<[FXb, LSU, Lat30], (instregex "SPX$")>;
def : InstRW<[FXb, LSU, Lat30], (instregex "STPX$")>;

//===----------------------------------------------------------------------===//
// System: Storage-Key and Real Memory Instructions
//===----------------------------------------------------------------------===//

def : InstRW<[FXb, Lat30], (instregex "ISKE$")>;
def : InstRW<[FXb, Lat30], (instregex "IVSK$")>;
def : InstRW<[FXb, Lat30], (instregex "SSKE(Opt)?$")>;
def : InstRW<[FXb, Lat30], (instregex "RRB(E|M)$")>;
def : InstRW<[FXb, Lat30], (instregex "IRBM$")>;
def : InstRW<[FXb, Lat30], (instregex "PFMF$")>;
def : InstRW<[FXb, Lat30], (instregex "TB$")>;
def : InstRW<[FXb, LSU, Lat30], (instregex "PGIN$")>;
def : InstRW<[FXb, LSU, Lat30], (instregex "PGOUT$")>;

//===----------------------------------------------------------------------===//
// System: Dynamic-Address-Translation Instructions
//===----------------------------------------------------------------------===//

def : InstRW<[FXb, LSU, Lat30], (instregex "IPTE(Opt)?(Opt)?$")>;
def : InstRW<[FXb, Lat30], (instregex "IDTE(Opt)?$")>;
def : InstRW<[FXb, Lat30], (instregex "CRDTE(Opt)?$")>;
def : InstRW<[FXb, Lat30], (instregex "PTLB$")>;
def : InstRW<[FXb, LSU, Lat30], (instregex "CSP(G)?$")>;
def : InstRW<[FXb, LSU, Lat30], (instregex "LPTEA$")>;
def : InstRW<[FXb, LSU, Lat30], (instregex "LRA(Y|G)?$")>;
def : InstRW<[FXb, LSU, Lat30], (instregex "STRAG$")>;
def : InstRW<[FXb, LSU, Lat30], (instregex "LURA(G)?$")>;
def : InstRW<[FXb, LSU, Lat30], (instregex "STUR(A|G)$")>;
def : InstRW<[FXb, LSU, Lat30], (instregex "TPROT$")>;

//===----------------------------------------------------------------------===//
// System: Memory-move Instructions
//===----------------------------------------------------------------------===//

def : InstRW<[FXa, FXa, FXb, LSU, Lat8, GroupAlone], (instregex "MVC(K|P|S)$")>;
def : InstRW<[FXa, LSU, Lat6, GroupAlone], (instregex "MVC(S|D)K$")>;
def : InstRW<[FXb, LSU, Lat30], (instregex "MVCOS$")>;
def : InstRW<[LSU, Lat30, GroupAlone], (instregex "MVPG$")>;

//===----------------------------------------------------------------------===//
// System: Address-Space Instructions
//===----------------------------------------------------------------------===//

def : InstRW<[FXb, LSU, Lat30], (instregex "LASP$")>;
def : InstRW<[LSU, GroupAlone], (instregex "PALB$")>;
def : InstRW<[FXb, LSU, Lat30], (instregex "PC$")>;
def : InstRW<[FXb, Lat30], (instregex "PR$")>;
def : InstRW<[FXb, Lat30], (instregex "PT(I)?$")>;
def : InstRW<[FXb, LSU, Lat30], (instregex "RP$")>;
def : InstRW<[FXb, Lat30], (instregex "BS(G|A)$")>;
def : InstRW<[FXb, Lat20], (instregex "TAR$")>;

//===----------------------------------------------------------------------===//
// System: Linkage-Stack Instructions
//===----------------------------------------------------------------------===//

def : InstRW<[FXb, Lat30, EndGroup], (instregex "BAKR$")>;
def : InstRW<[FXb, Lat30], (instregex "EREG(G)?$")>;
def : InstRW<[FXb, Lat30], (instregex "(E|M)STA$")>;

//===----------------------------------------------------------------------===//
// System: Time-Related Instructions
//===----------------------------------------------------------------------===//

def : InstRW<[FXb, Lat30], (instregex "PTFF$")>;
def : InstRW<[FXb, LSU, Lat20], (instregex "SCK$")>;
def : InstRW<[FXb, Lat30], (instregex "SCKPF$")>;
def : InstRW<[FXb, LSU, Lat20], (instregex "SCKC$")>;
def : InstRW<[LSU, LSU, GroupAlone], (instregex "SPT$")>;
def : InstRW<[LSU, LSU, LSU, FXa, FXa, FXb, Lat9, GroupAlone],
             (instregex "STCK(F)?$")>;
def : InstRW<[LSU, LSU, LSU, LSU, FXa, FXa, FXb, FXb, Lat11, GroupAlone],
             (instregex "STCKE$")>;
def : InstRW<[FXb, LSU, Lat9], (instregex "STCKC$")>;
def : InstRW<[LSU, LSU, FXb, Lat5, BeginGroup], (instregex "STPT$")>;

//===----------------------------------------------------------------------===//
// System: CPU-Related Instructions
//===----------------------------------------------------------------------===//

def : InstRW<[FXb, LSU, Lat30], (instregex "STAP$")>;
def : InstRW<[FXb, LSU, Lat30], (instregex "STIDP$")>;
def : InstRW<[FXb, LSU, Lat30], (instregex "STSI$")>;
def : InstRW<[FXb, LSU, Lat30], (instregex "STFL(E)?$")>;
def : InstRW<[FXb, LSU, Lat30], (instregex "ECAG$")>;
def : InstRW<[FXa, LSU, Lat30], (instregex "ECTG$")>;
def : InstRW<[FXb, Lat30], (instregex "PTF$")>;
def : InstRW<[FXb, Lat30], (instregex "PCKMO$")>;

//===----------------------------------------------------------------------===//
// System: Miscellaneous Instructions
//===----------------------------------------------------------------------===//

def : InstRW<[FXb, Lat30], (instregex "SVC$")>;
def : InstRW<[FXb, GroupAlone], (instregex "MC$")>;
def : InstRW<[FXb, Lat30], (instregex "DIAG$")>;
def : InstRW<[FXb], (instregex "TRAC(E|G)$")>;
def : InstRW<[FXb, Lat30], (instregex "TRAP(2|4)$")>;
def : InstRW<[FXb, Lat30], (instregex "SIGP$")>;
def : InstRW<[FXb, LSU, Lat30], (instregex "SIGA$")>;
def : InstRW<[FXb, LSU, Lat30], (instregex "SIE$")>;

//===----------------------------------------------------------------------===//
// System: CPU-Measurement Facility Instructions
//===----------------------------------------------------------------------===//

def : InstRW<[FXb], (instregex "LPP$")>;
def : InstRW<[FXb, Lat30], (instregex "ECPGA$")>;
def : InstRW<[FXb, Lat30], (instregex "E(C|P)CTR$")>;
def : InstRW<[FXb, Lat30], (instregex "LCCTL$")>;
def : InstRW<[FXb, LSU, Lat30], (instregex "L(P|S)CTL$")>;
def : InstRW<[FXb, LSU, Lat30], (instregex "Q(S|CTR)I$")>;
def : InstRW<[FXb, Lat30], (instregex "S(C|P)CTR$")>;

//===----------------------------------------------------------------------===//
// System: I/O Instructions
//===----------------------------------------------------------------------===//

def : InstRW<[FXb, Lat30], (instregex "(C|H|R|X)SCH$")>;
def : InstRW<[FXb, LSU, Lat30], (instregex "(M|S|ST|T)SCH$")>;
def : InstRW<[FXb, Lat30], (instregex "RCHP$")>;
def : InstRW<[FXb, Lat30], (instregex "SCHM$")>;
def : InstRW<[FXb, LSU, Lat30], (instregex "STC(PS|RW)$")>;
def : InstRW<[FXb, LSU, Lat30], (instregex "TPI$")>;
def : InstRW<[FXb, Lat30], (instregex "SAL$")>;

}

