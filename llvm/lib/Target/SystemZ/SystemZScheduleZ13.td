//-- SystemZScheduleZ13.td - SystemZ Scheduling Definitions ----*- tblgen -*-=//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the machine model for Z13 to support instruction
// scheduling and other instruction cost heuristics.
//
//===----------------------------------------------------------------------===//

def Z13Model : SchedMachineModel {
    
    let IssueWidth = 6;             // 2 * 3 instructions decoded per cycle.
    let MicroOpBufferSize = 60;     // Issue queues
    let LoadLatency = 1;            // Optimistic load latency.

    let PostRAScheduler = 1;

    // Extra cycles for a mispredicted branch.
    let MispredictPenalty = 8;
}

let SchedModel = Z13Model in  {

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
def Z13_FXaUnit     : ProcResource<2>;
def Z13_FXbUnit     : ProcResource<2>;
def Z13_LSUnit      : ProcResource<2>;
def Z13_VecBFUnit   : ProcResource<2>;
def Z13_VecDFUnit   : ProcResource<2>;
def Z13_VecFPdUnit  : ProcResource<2> { let BufferSize = 1; /* blocking */ }
def Z13_VecMulUnit  : ProcResource<2>;
def Z13_VecStrUnit  : ProcResource<2>;
def Z13_VecXsPmUnit : ProcResource<2>;
def Z13_VBUnit      : ProcResource<1>;

// Subtarget specific definitions of scheduling resources.
def : WriteRes<FXa,     [Z13_FXaUnit]>   { let Latency = 1; }
def : WriteRes<FXb,     [Z13_FXbUnit]>   { let Latency = 1; }
def : WriteRes<LSU,     [Z13_LSUnit]>    { let Latency = 4; }
def : WriteRes<VecBF,   [Z13_VecBFUnit]> { let Latency = 8; }
def : WriteRes<VecDF,   [Z13_VecDFUnit]>;
def : WriteRes<VecFPd,  [Z13_VecFPdUnit, Z13_VecFPdUnit, Z13_VecFPdUnit,
                         Z13_VecFPdUnit, Z13_VecFPdUnit, Z13_VecFPdUnit,
                         Z13_VecFPdUnit, Z13_VecFPdUnit, Z13_VecFPdUnit,
                         Z13_VecFPdUnit, Z13_VecFPdUnit, Z13_VecFPdUnit,
                         Z13_VecFPdUnit, Z13_VecFPdUnit, Z13_VecFPdUnit,
                         Z13_VecFPdUnit, Z13_VecFPdUnit, Z13_VecFPdUnit,
                         Z13_VecFPdUnit, Z13_VecFPdUnit, Z13_VecFPdUnit,
                         Z13_VecFPdUnit, Z13_VecFPdUnit, Z13_VecFPdUnit,
                         Z13_VecFPdUnit, Z13_VecFPdUnit, Z13_VecFPdUnit,
                         Z13_VecFPdUnit, Z13_VecFPdUnit, Z13_VecFPdUnit]>
                         { let Latency = 30; }
def : WriteRes<VecMul,  [Z13_VecMulUnit]>  { let Latency = 5; }
def : WriteRes<VecStr,  [Z13_VecStrUnit]>  { let Latency = 4; }
def : WriteRes<VecXsPm, [Z13_VecXsPmUnit]> { let Latency = 3; }
def : WriteRes<VBU,     [Z13_VBUnit]>; // Virtual Branching Unit

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
// Control flow instructions
//===----------------------------------------------------------------------===//

// Return
def : InstRW<[FXb, EndGroup], (instregex "Return$")>;
def : InstRW<[FXb], (instregex "CondReturn$")>;

// Compare and branch
def : InstRW<[FXb], (instregex "(Asm.*)?C(I|R)J$")>;
def : InstRW<[FXb], (instregex "(Asm.*)?CG(I|R)J$")>;
def : InstRW<[FXb], (instregex "(Asm.*)?CL(I|R)J$")>;
def : InstRW<[FXb], (instregex "(Asm.*)?CLG(I|R)J$")>;
def : InstRW<[FXb], (instregex "(Asm.*)?CG(R|I)J$")>;
def : InstRW<[FXb], (instregex "CLR$")>;
def : InstRW<[FXb, FXb, Lat2, GroupAlone], (instregex "(Asm.*)?CIB(Call|Return)?$")>;
def : InstRW<[FXb, FXb, Lat2, GroupAlone], (instregex "(Asm.*)?CLIB(Call|Return)?$")>;
def : InstRW<[FXb, FXb, Lat2, GroupAlone], (instregex "(Asm.*)?CLGIB(Call|Return)?$")>;
def : InstRW<[FXb, FXb, Lat2, GroupAlone], (instregex "(Asm.*)?CGIB(Call|Return)?$")>;
def : InstRW<[FXb, FXb, Lat2, GroupAlone], (instregex "(Asm.*)?CGRB(Call|Return)?$")>;
def : InstRW<[FXb, FXb, Lat2, GroupAlone], (instregex "(Asm.*)?CLGRB(Call|Return)?$")>;
def : InstRW<[FXb, FXb, Lat2, GroupAlone], (instregex "CLR(Call|Return)?$")>;
def : InstRW<[FXb, FXb, Lat2, GroupAlone], (instregex "(Asm.*)?CLRB(Call|Return)?$")>;
def : InstRW<[FXb, FXb, Lat2, GroupAlone], (instregex "(Asm.*)?CRB(Call|Return)?$")>;

// Branch
def : InstRW<[FXb], (instregex "(Asm.*)?BR$")>;
def : InstRW<[FXb], (instregex "(Asm)?BC(R)?$")>;
def : InstRW<[VBU], (instregex "(Asm)?BRC(L)?$")>;
def : InstRW<[FXa, EndGroup], (instregex "BRCT(G)?$")>;
def : InstRW<[VBU], (instregex "(Asm.*)?JG$")>;
def : InstRW<[VBU], (instregex "J$")>;
// (Need to avoid conflict with "(Asm.*)?CG(I|R)J$")
def : InstRW<[VBU], (instregex "Asm(EAlt|E|HAlt|HE|H|LAlt|LE|LH|L|NEAlt|NE)J$")>;
def : InstRW<[VBU], (instregex "Asm(NHAlt|NHE|NH|NLAlt|NLE|NLH|NL|NO|O)J$")>;
def : InstRW<[FXa, FXa, FXb, FXb, Lat4, GroupAlone], (instregex "BRX(H|LE)$")>;

// Trap
def : InstRW<[VBU], (instregex "(Cond)?Trap$")>;

// Compare and trap
def : InstRW<[FXb], (instregex "(Asm.*)?C(G)?IT$")>;
def : InstRW<[FXb], (instregex "(Asm.*)?C(G)?RT$")>;
def : InstRW<[FXb], (instregex "(Asm.*)?CLG(I|R)T$")>;
def : InstRW<[FXb], (instregex "(Asm.*)?CLFIT$")>;
def : InstRW<[FXb], (instregex "(Asm.*)?CLRT$")>;

//===----------------------------------------------------------------------===//
// Select instructions
//===----------------------------------------------------------------------===//

// Select pseudo 
def : InstRW<[FXa], (instregex "Select(32|64|32Mux)$")>;

// CondStore pseudos
def : InstRW<[FXa], (instregex "CondStore16(Inv)?$")>;
def : InstRW<[FXa], (instregex "CondStore16Mux(Inv)?$")>;
def : InstRW<[FXa], (instregex "CondStore32(Inv)?$")>;
def : InstRW<[FXa], (instregex "CondStore64(Inv)?$")>;
def : InstRW<[FXa], (instregex "CondStore8(Inv)?$")>;
def : InstRW<[FXa], (instregex "CondStore8Mux(Inv)?$")>;

//===----------------------------------------------------------------------===//
// Call instructions
//===----------------------------------------------------------------------===//

def : InstRW<[VBU, FXa, FXa, Lat3, GroupAlone], (instregex "BRAS$")>;
def : InstRW<[FXa, FXa, FXb, Lat3, GroupAlone], (instregex "(Call)?BASR$")>;
def : InstRW<[FXb], (instregex "CallB(C)?R$")>;
def : InstRW<[FXa, FXa, FXb, Lat3, GroupAlone], (instregex "(Call)?BRASL$")>;
def : InstRW<[FXa, FXa, FXb, Lat3, GroupAlone], (instregex "TLS_(G|L)DCALL$")>;
def : InstRW<[VBU], (instregex "CallBRCL$")>;
def : InstRW<[VBU], (instregex "CallJG$")>;

//===----------------------------------------------------------------------===//
// Move instructions
//===----------------------------------------------------------------------===//

// Moves
def : InstRW<[FXb, LSU, Lat5], (instregex "MV(G|H)?HI$")>;
def : InstRW<[FXb, LSU, Lat5], (instregex "MVI(Y)?$")>;

// Move character
def : InstRW<[FXb, LSU, LSU, LSU, Lat8, GroupAlone], (instregex "MVC$")>;

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

// Load and test
def : InstRW<[FXa, LSU, Lat5], (instregex "LT(G)?$")>;
def : InstRW<[FXa], (instregex "LT(G)?R$")>;

// Load on condition
def : InstRW<[FXa, LSU, Lat6], (instregex "(Asm.*)?LOC(G)?$")>;
def : InstRW<[FXa, Lat2], (instregex "(Asm.*)?LOC(G)?R$")>;
def : InstRW<[FXa, Lat2], (instregex "(Asm.*)?LOC(G)?HI$")>;

// Stores
def : InstRW<[FXb, LSU, Lat5], (instregex "STG(RL)?$")>;
def : InstRW<[FXb, LSU, Lat5], (instregex "ST128$")>;
def : InstRW<[FXb, LSU, Lat5], (instregex "ST(Y|FH|RL|Mux)?$")>;

// Store on condition
def : InstRW<[FXb, LSU, Lat5], (instregex "(Asm.*)?STOC(G)?$")>;

// String moves.
def : InstRW<[LSU, Lat30, GroupAlone], (instregex "MVST$")>;

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
def : InstRW<[FXa], (instregex "LLG(C|H|F)R$")>;
def : InstRW<[LSU], (instregex "LLC(Mux)?$")>;
def : InstRW<[LSU], (instregex "LLH(Mux)?$")>;
def : InstRW<[FXa, LSU, Lat5], (instregex "LL(C|H)H$")>;
def : InstRW<[LSU], (instregex "LLHRL$")>;
def : InstRW<[LSU], (instregex "LLG(C|H|F|HRL|FRL)$")>;

//===----------------------------------------------------------------------===//
// Truncations
//===----------------------------------------------------------------------===//

def : InstRW<[FXb, LSU, Lat5], (instregex "STC(H|Y|Mux)?$")>;
def : InstRW<[FXb, LSU, Lat5], (instregex "STH(H|Y|RL|Mux)?$")>;

//===----------------------------------------------------------------------===//
// Multi-register moves
//===----------------------------------------------------------------------===//

// Load multiple (estimated average of 5 ops)
def : InstRW<[LSU, LSU, LSU, LSU, LSU, Lat10, GroupAlone],
             (instregex "LM(H|Y|G)?$")>;

// Store multiple (estimated average of ceil(5/2) FXb ops)
def : InstRW<[LSU, LSU, FXb, FXb, FXb, Lat10,
              GroupAlone], (instregex "STM(G|H|Y)?$")>;

//===----------------------------------------------------------------------===//
// Byte swaps
//===----------------------------------------------------------------------===//

def : InstRW<[FXa], (instregex "LRV(G)?R$")>;
def : InstRW<[FXa, LSU, Lat5], (instregex "LRV(G|H)?$")>;
def : InstRW<[FXb, LSU, Lat5], (instregex "STRV(G|H)?$")>;

//===----------------------------------------------------------------------===//
// Load address instructions
//===----------------------------------------------------------------------===//

def : InstRW<[FXa], (instregex "LA(Y|RL)?$")>;

// Load the Global Offset Table address ( -> larl )
def : InstRW<[FXa], (instregex "GOT$")>;

//===----------------------------------------------------------------------===//
// Absolute and Negation
//===----------------------------------------------------------------------===//

def : InstRW<[FXa, Lat2], (instregex "LP(G)?R$")>;
def : InstRW<[FXa, FXa, Lat3, BeginGroup], (instregex "L(N|P)GFR$")>;
def : InstRW<[FXa, Lat2], (instregex "LN(R|GR)$")>;
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
def : InstRW<[FXb, LSU, Lat5], (instregex "A(G)?SI$")>;

// Logical addition with carry
def : InstRW<[FXa, LSU, Lat6, GroupAlone], (instregex "ALC(G)?$")>;
def : InstRW<[FXa, Lat2, GroupAlone], (instregex "ALC(G)?R$")>;

// Add with sign extension (32 -> 64)
def : InstRW<[FXa, LSU, Lat6], (instregex "AGF$")>;
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

// Subtraction with borrow
def : InstRW<[FXa, LSU, Lat6, GroupAlone], (instregex "SLB(G)?$")>;
def : InstRW<[FXa, Lat2, GroupAlone], (instregex "SLB(G)?R$")>;

// Subtraction with sign extension (32 -> 64)
def : InstRW<[FXa, LSU, Lat6], (instregex "SGF$")>;
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

def : InstRW<[FXa, LSU, Lat10], (instregex "MS(GF|Y)?$")>;
def : InstRW<[FXa, Lat6], (instregex "MS(R|FI)$")>;
def : InstRW<[FXa, LSU, Lat12], (instregex "MSG$")>;
def : InstRW<[FXa, Lat8], (instregex "MSGR$")>;
def : InstRW<[FXa, Lat6], (instregex "MSGF(I|R)$")>;
def : InstRW<[FXa, LSU, Lat15, GroupAlone], (instregex "MLG$")>;
def : InstRW<[FXa, Lat9, GroupAlone], (instregex "MLGR$")>;
def : InstRW<[FXa, Lat5], (instregex "MGHI$")>;
def : InstRW<[FXa, Lat5], (instregex "MHI$")>;
def : InstRW<[FXa, LSU, Lat9], (instregex "MH(Y)?$")>;

//===----------------------------------------------------------------------===//
// Division and remainder
//===----------------------------------------------------------------------===//

def : InstRW<[FXa, Lat30, GroupAlone], (instregex "DSG(F)?R$")>;
def : InstRW<[LSU, FXa, Lat30, GroupAlone], (instregex "DSG(F)?$")>;
def : InstRW<[FXa, FXa, Lat20, GroupAlone], (instregex "DLR$")>;
def : InstRW<[FXa, FXa, Lat30, GroupAlone], (instregex "DLGR$")>;
def : InstRW<[FXa, FXa, LSU, Lat30, GroupAlone], (instregex "DL(G)?$")>;

//===----------------------------------------------------------------------===//
// Shifts
//===----------------------------------------------------------------------===//

def : InstRW<[FXa], (instregex "SLL(G|K)?$")>;
def : InstRW<[FXa], (instregex "SRL(G|K)?$")>;
def : InstRW<[FXa], (instregex "SRA(G|K)?$")>;
def : InstRW<[FXa], (instregex "SLA(K)?$")>;

// Rotate
def : InstRW<[FXa, LSU, Lat6], (instregex "RLL(G)?$")>;

// Rotate and insert
def : InstRW<[FXa], (instregex "RISBG(N|32)?$")>;
def : InstRW<[FXa], (instregex "RISBH(G|H|L)$")>;
def : InstRW<[FXa], (instregex "RISBL(G|H|L)$")>;
def : InstRW<[FXa], (instregex "RISBMux$")>;

// Rotate and Select
def : InstRW<[FXa, FXa, Lat3, BeginGroup], (instregex "R(N|O|X)SBG$")>;

//===----------------------------------------------------------------------===//
// Comparison
//===----------------------------------------------------------------------===//

def : InstRW<[FXb, LSU, Lat5], (instregex "C(G|Y|Mux|RL)?$")>;
def : InstRW<[FXb], (instregex "CFI(Mux)?$")>;
def : InstRW<[FXb], (instregex "CG(F|H)I$")>;
def : InstRW<[FXb, LSU, Lat5], (instregex "CG(HSI|RL)$")>;
def : InstRW<[FXb], (instregex "C(G)?R$")>;
def : InstRW<[FXb], (instregex "C(HI|IH)$")>;
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

// Compare halfword
def : InstRW<[FXb, LSU, Lat6], (instregex "CH(Y|RL)?$")>;
def : InstRW<[FXb, LSU, Lat6], (instregex "CGH(RL)?$")>;
def : InstRW<[FXa, FXb, LSU, Lat6, BeginGroup], (instregex "CHHSI$")>;

// Compare with sign extension (32 -> 64)
def : InstRW<[FXb, LSU, Lat6], (instregex "CGF(RL)?$")>;
def : InstRW<[FXb, Lat2], (instregex "CGFR$")>;

// Compare logical character
def : InstRW<[FXb, LSU, LSU, Lat9, BeginGroup], (instregex "CLC$")>;

def : InstRW<[LSU, Lat30, GroupAlone], (instregex "CLST$")>;

// Test under mask
def : InstRW<[FXb, LSU, Lat5], (instregex "TM(Y)?$")>;
def : InstRW<[FXb], (instregex "TM(H|L)Mux$")>;
def : InstRW<[FXb], (instregex "TMHH(64)?$")>;
def : InstRW<[FXb], (instregex "TMHL(64)?$")>;
def : InstRW<[FXb], (instregex "TMLH(64)?$")>;
def : InstRW<[FXb], (instregex "TMLL(64)?$")>;

//===----------------------------------------------------------------------===//
// Prefetch
//===----------------------------------------------------------------------===//

def : InstRW<[LSU], (instregex "PFD(RL)?$")>;

//===----------------------------------------------------------------------===//
// Atomic operations
//===----------------------------------------------------------------------===//

def : InstRW<[FXb, EndGroup], (instregex "Serialize$")>;

def : InstRW<[FXb, LSU, Lat5], (instregex "LAA(G)?$")>;
def : InstRW<[FXb, LSU, Lat5], (instregex "LAAL(G)?$")>;
def : InstRW<[FXb, LSU, Lat5], (instregex "LAN(G)?$")>;
def : InstRW<[FXb, LSU, Lat5], (instregex "LAO(G)?$")>;
def : InstRW<[FXb, LSU, Lat5], (instregex "LAX(G)?$")>;

// Compare and swap
def : InstRW<[FXa, FXb, LSU, Lat6, GroupAlone], (instregex "CS(G|Y)?$")>;

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

def : InstRW<[FXb], (instregex "PPA$")>;

//===----------------------------------------------------------------------===//
// Miscellaneous Instructions.
//===----------------------------------------------------------------------===//

// Insert Program Mask
def : InstRW<[FXa, Lat3, EndGroup], (instregex "IPM$")>;

// Extract access register
def : InstRW<[LSU], (instregex "EAR$")>;

// Find leftmost one
def : InstRW<[FXa, Lat6, GroupAlone], (instregex "FLOGR$")>;

// Population count
def : InstRW<[FXa, Lat3], (instregex "POPCNT$")>;

// Extend
def : InstRW<[FXa], (instregex "AEXT128_64$")>;
def : InstRW<[FXa], (instregex "ZEXT128_(32|64)$")>;

// String instructions
def : InstRW<[FXa, LSU, Lat30], (instregex "SRST$")>;

// Move with key
def : InstRW<[FXa, FXa, FXb, LSU, Lat8, GroupAlone], (instregex "MVCK$")>;

// Extract CPU Time
def : InstRW<[FXa, Lat5, LSU], (instregex "ECTG$")>;

// Execute
def : InstRW<[FXb, GroupAlone], (instregex "EX(RL)?$")>;

// Program return
def : InstRW<[FXb, Lat30], (instregex "PR$")>;

// Inline assembly
def : InstRW<[LSU, LSU, LSU, FXa, FXa, FXb, Lat9, GroupAlone],
             (instregex "STCK(F)?$")>;
def : InstRW<[LSU, LSU, LSU, LSU, FXa, FXa, FXb, FXb, Lat11, GroupAlone],
             (instregex "STCKE$")>;
def : InstRW<[FXa, LSU, Lat5], (instregex "STFLE$")>;
def : InstRW<[FXb, Lat30], (instregex "SVC$")>;

// Store real address
def : InstRW<[FXb, LSU, Lat5], (instregex "STRAG$")>;

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

def : InstRW<[FXa], (instregex "SelectF(32|64|128)$")>;
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
def : InstRW<[VecDF, VecDF, Lat11, GroupAlone], (instregex "LTXBR$")>;
def : InstRW<[VecDF, VecDF, Lat11, GroupAlone],
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
def : InstRW<[VecBF, VecBF, LSU, Lat12 , GroupAlone], (instregex "LX(D|E)B$")>;
def : InstRW<[VecBF, VecBF, Lat9 , GroupAlone], (instregex "LX(D|E)BR$")>;

// Convert from fixed / logical
def : InstRW<[FXb, VecBF, Lat9, BeginGroup], (instregex "CE(F|G)BR$")>;
def : InstRW<[FXb, VecBF, Lat9, BeginGroup], (instregex "CD(F|G)BR$")>;
def : InstRW<[FXb, VecDF, VecDF, Lat12, GroupAlone], (instregex "CX(F|G)BR$")>;
def : InstRW<[FXb, VecBF, Lat9, BeginGroup], (instregex "CEL(F|G)BR$")>;
def : InstRW<[FXb, VecBF, Lat9, BeginGroup], (instregex "CDL(F|G)BR$")>;
def : InstRW<[FXb, VecDF, VecDF, Lat12, GroupAlone], (instregex "CXL(F|G)BR$")>;

// Convert to fixed / logical
def : InstRW<[FXb, VecBF, Lat11, BeginGroup], (instregex "CF(E|D)BR$")>;
def : InstRW<[FXb, VecBF, Lat11, BeginGroup], (instregex "CG(E|D)BR$")>;
def : InstRW<[FXb, VecDF, VecDF, Lat20, BeginGroup], (instregex "C(F|G)XBR$")>;
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
def : InstRW<[VecDF, VecDF, Lat11, GroupAlone], (instregex "L(C|N|P)XBR$")>;

// Square root
def : InstRW<[VecFPd, LSU], (instregex "SQ(E|D)B$")>;
def : InstRW<[VecFPd], (instregex "SQ(E|D)BR$")>;
def : InstRW<[VecFPd, VecFPd, GroupAlone], (instregex "SQXBR$")>;

// Load FP integer
def : InstRW<[VecBF], (instregex "FIEBR(A)?$")>;
def : InstRW<[VecBF], (instregex "FIDBR(A)?$")>;
def : InstRW<[VecDF, VecDF, Lat11, GroupAlone], (instregex "FIXBR(A)?$")>;

//===----------------------------------------------------------------------===//
// FP: Binary arithmetic
//===----------------------------------------------------------------------===//

// Addition
def : InstRW<[VecBF, LSU, Lat12], (instregex "A(E|D)B$")>;
def : InstRW<[VecBF], (instregex "A(E|D)BR$")>;
def : InstRW<[VecDF, VecDF, Lat11, GroupAlone], (instregex "AXBR$")>;

// Subtraction
def : InstRW<[VecBF, LSU, Lat12], (instregex "S(E|D)B$")>;
def : InstRW<[VecBF], (instregex "S(E|D)BR$")>;
def : InstRW<[VecDF, VecDF, Lat11, GroupAlone], (instregex "SXBR$")>;

// Multiply
def : InstRW<[VecBF, LSU, Lat12], (instregex "M(D|DE|EE)B$")>;
def : InstRW<[VecBF], (instregex "M(D|DE|EE)BR$")>;
def : InstRW<[VecBF, VecBF, LSU, Lat12, GroupAlone], (instregex "MXDB$")>;
def : InstRW<[VecBF, VecBF, Lat9, GroupAlone], (instregex "MXDBR$")>;
def : InstRW<[VecDF, VecDF, Lat20, GroupAlone], (instregex "MXBR$")>;

// Multiply and add / subtract
def : InstRW<[VecBF, LSU, Lat12, GroupAlone], (instregex "M(A|S)EB$")>;
def : InstRW<[VecBF, GroupAlone], (instregex "M(A|S)EBR$")>;
def : InstRW<[VecBF, LSU, Lat12, GroupAlone], (instregex "M(A|S)DB$")>;
def : InstRW<[VecBF], (instregex "M(A|S)DBR$")>;

// Division
def : InstRW<[VecFPd, LSU], (instregex "D(E|D)B$")>;
def : InstRW<[VecFPd], (instregex "D(E|D)BR$")>;
def : InstRW<[VecFPd, VecFPd, GroupAlone], (instregex "DXBR$")>;

//===----------------------------------------------------------------------===//
// FP: Comparisons
//===----------------------------------------------------------------------===//

// Compare
def : InstRW<[VecXsPm, LSU, Lat8], (instregex "C(E|D)B$")>;
def : InstRW<[VecXsPm, Lat4], (instregex "C(E|D)BR?$")>;
def : InstRW<[VecDF, VecDF, Lat20, GroupAlone], (instregex "CXBR$")>;

// Test Data Class
def : InstRW<[LSU, VecXsPm, Lat9], (instregex "TC(E|D)B$")>;
def : InstRW<[LSU, VecDF, VecDF, Lat15, GroupAlone], (instregex "TCXB$")>;


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
def : InstRW<[VecXsPm], (instregex "VLEI(B|F|G|H)$")>;
def : InstRW<[VecXsPm], (instregex "VREPI(B|F|G|H)?$")>;

//===----------------------------------------------------------------------===//
// Vector: Loads
//===----------------------------------------------------------------------===//

def : InstRW<[LSU], (instregex "VL(L|BB)?$")>;
def : InstRW<[LSU], (instregex "VL(32|64)$")>;
def : InstRW<[LSU], (instregex "VLLEZ(B|F|G|H)?$")>;
def : InstRW<[LSU], (instregex "VLREP(B|F|G|H)?$")>;
def : InstRW<[VecXsPm, LSU, Lat7], (instregex "VLE(B|F|G|H)$")>;
def : InstRW<[FXb, LSU, VecXsPm, Lat11, BeginGroup], (instregex "VGE(F|G)$")>;
def : InstRW<[LSU, LSU, LSU, LSU, LSU, Lat10, GroupAlone],
              (instregex "VLM$")>;

//===----------------------------------------------------------------------===//
// Vector: Stores
//===----------------------------------------------------------------------===//

def : InstRW<[FXb, LSU, Lat8], (instregex "VST(L|32|64)?$")>;
def : InstRW<[FXb, LSU, Lat8], (instregex "VSTE(F|G)$")>;
def : InstRW<[FXb, LSU, VecXsPm, Lat11, BeginGroup], (instregex "VSTE(B|H)$")>;
def : InstRW<[LSU, LSU, FXb, FXb, FXb, FXb, FXb, Lat20, GroupAlone],
              (instregex "VSTM$")>;
def : InstRW<[FXb, FXb, LSU, Lat12, BeginGroup], (instregex "VSCE(F|G)$")>;

//===----------------------------------------------------------------------===//
// Vector: Selects and permutes
//===----------------------------------------------------------------------===//

def : InstRW<[VecXsPm], (instregex "VMRH(B|F|G|H)?$")>;
def : InstRW<[VecXsPm], (instregex "VMRL(B|F|G|H)?$")>;
def : InstRW<[VecXsPm], (instregex "VPERM$")>;
def : InstRW<[VecXsPm], (instregex "VPDI$")>;
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
def : InstRW<[VecXsPm], (instregex "VN(C|O)?$")>;
def : InstRW<[VecXsPm], (instregex "VO$")>;
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

def : InstRW<[VecXsPm], (instregex "VPOPCT$")>;

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
def : InstRW<[VecXsPm, VecXsPm, Lat8], (instregex "VSLB$")>;
def : InstRW<[VecXsPm], (instregex "VSR(A|L)$")>;
def : InstRW<[VecXsPm, VecXsPm, Lat8], (instregex "VSR(A|L)B$")>;

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

def : InstRW<[VecBF], (instregex "VCD(G|GB|LG|LGB)$")>;
def : InstRW<[VecBF], (instregex "WCD(GB|LGB)$")>;
def : InstRW<[VecBF], (instregex "VC(L)?GD$")>;
def : InstRW<[VecBF], (instregex "(V|W)FADB$")>;
def : InstRW<[VecBF], (instregex "(V|W)CGDB$")>;
def : InstRW<[VecBF], (instregex "VF(I|M|A|S)$")>;
def : InstRW<[VecBF], (instregex "VF(I|M|S)DB$")>;
def : InstRW<[VecBF], (instregex "WF(I|M|S)DB$")>;
def : InstRW<[VecBF], (instregex "(V|W)CLGDB$")>;
def : InstRW<[VecXsPm], (instregex "VFL(C|N|P)DB$")>;
def : InstRW<[VecXsPm], (instregex "WFL(C|N|P)DB$")>;
def : InstRW<[VecBF], (instregex "VFM(A|S)$")>;
def : InstRW<[VecBF], (instregex "VFM(A|S)DB$")>;
def : InstRW<[VecBF], (instregex "WFM(A|S)DB$")>;
def : InstRW<[VecXsPm], (instregex "VFPSO$")>;
def : InstRW<[VecXsPm], (instregex "(V|W)FPSODB$")>;
def : InstRW<[VecXsPm, Lat4], (instregex "VFTCI(DB)?$")>;
def : InstRW<[VecXsPm, Lat4], (instregex "WFTCIDB$")>;
def : InstRW<[VecBF], (instregex "VL(DE|ED)$")>;
def : InstRW<[VecBF], (instregex "VL(DE|ED)B$")>;
def : InstRW<[VecBF], (instregex "WL(DE|ED)B$")>;

// divide / square root
def : InstRW<[VecFPd], (instregex "VFD$")>;
def : InstRW<[VecFPd], (instregex "(V|W)FDDB$")>;
def : InstRW<[VecFPd], (instregex "VFSQ$")>;
def : InstRW<[VecFPd], (instregex "(V|W)FSQDB$")>;

//===----------------------------------------------------------------------===//
// Vector: Floating-point comparison
//===----------------------------------------------------------------------===//

def : InstRW<[VecXsPm], (instregex "VFC(E|H|HE)$")>;
def : InstRW<[VecXsPm], (instregex "VFC(E|H|HE)DB$")>;
def : InstRW<[VecXsPm, Lat4], (instregex "WF(C|K)$")>;
def : InstRW<[VecXsPm], (instregex "WFC(E|H|HE)DB$")>;
def : InstRW<[VecXsPm, Lat4], (instregex "VFC(E|H|HE)DBS$")>;
def : InstRW<[VecXsPm, Lat4], (instregex "WFC(E|H|HE)DBS$")>;
def : InstRW<[VecXsPm, Lat4], (instregex "WF(C|K)DB$")>;

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

}

